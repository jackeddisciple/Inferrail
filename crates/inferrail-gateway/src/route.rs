use std::sync::Arc;

use inferrail_control::{Lease, Nanos, Registry, Replica};
use inferrail_core::ids::{ModelId, ReplicaId};
use inferrail_policy::{Candidate, RouteContext, RoutingPolicy};

use crate::errors::GatewayError;

/// Picks a replica for `model` and takes a lease on it. Only replicas that
/// serve the model, are routable and have room are ever offered to the
/// policy; untried ones are preferred over ones this request already used.
pub(crate) fn select(
    registry: &Registry,
    policy: &dyn RoutingPolicy,
    model: &ModelId,
    now: Nanos,
    tried: &[ReplicaId],
) -> Result<Lease, GatewayError> {
    let fleet = registry.load();
    let pool = fleet.serving(model);
    let routable = pool.iter().filter(|r| r.is_routable(now));
    let mut eligible: Vec<&Arc<Replica>> = routable.clone().filter(|r| r.has_room()).collect();
    if eligible.iter().any(|r| !tried.contains(r.id())) {
        eligible.retain(|r| !tried.contains(r.id()));
    }
    let mut candidates: Vec<Candidate<'_>> = Vec::with_capacity(eligible.len());
    while !eligible.is_empty() {
        candidates.clear();
        candidates.extend(eligible.iter().map(|r| Candidate {
            id: r.id(),
            in_flight: r.in_flight(),
        }));
        let ctx = RouteContext {
            model,
            candidates: &candidates,
        };
        let Some(i) = policy.choose(&ctx).filter(|&i| i < eligible.len()) else {
            break;
        };
        if let Some(lease) = eligible[i].try_acquire() {
            return Ok(lease);
        }
        eligible.remove(i);
    }
    Err(if routable.count() == 0 {
        GatewayError::unavailable(format!("no healthy replica serves `{model}`"))
    } else {
        GatewayError::unavailable(format!("every replica serving `{model}` is at capacity"))
    })
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;
    use inferrail_control::{HealthPolicy, Limits, Origin, ReplicaSpec};
    use inferrail_core::config::PolicyKind;
    use proptest::prelude::*;

    const MODELS: [&str; 3] = ["a", "b", "c"];

    #[derive(Debug, Clone)]
    struct Member {
        models: Vec<bool>,
        probe_down: bool,
        ejected: bool,
        cap: Option<u32>,
        held: u32,
    }

    fn member() -> impl Strategy<Value = Member> {
        (
            prop::collection::vec(any::<bool>(), 3),
            prop::bool::weighted(0.25),
            prop::bool::weighted(0.25),
            prop::option::of(1u32..4),
            0u32..4,
        )
            .prop_map(|(models, probe_down, ejected, cap, held)| Member {
                models,
                probe_down,
                ejected,
                cap,
                held,
            })
    }

    fn policy() -> HealthPolicy {
        HealthPolicy {
            unhealthy_threshold: 1,
            healthy_threshold: 1,
            ejection_backoff: Duration::from_secs(60),
            max_ejection_backoff: Duration::from_secs(60),
        }
    }

    fn build(members: &[Member]) -> (Registry, Vec<Lease>) {
        let specs = members
            .iter()
            .enumerate()
            .map(|(i, m)| ReplicaSpec {
                id: ReplicaId::new(format!("r{i}")).unwrap(),
                url: format!("http://10.0.0.{i}:8000").parse().unwrap(),
                models: MODELS
                    .iter()
                    .zip(&m.models)
                    .filter(|(_, on)| **on)
                    .map(|(name, _)| ModelId::new(name).unwrap())
                    .collect(),
                origin: Origin::Static,
                limits: Limits {
                    max_concurrency: m.cap,
                    max_pending: None,
                },
                kv_capacity_blocks: None,
            })
            .collect();
        let registry = Registry::new(specs);
        let mut held = Vec::new();
        for (replica, m) in registry.load().replicas().iter().zip(members) {
            if m.probe_down {
                replica.health().record_probe(false, &policy());
            }
            if m.ejected {
                replica.health().record_failure(10, &policy());
            }
            held.extend((0..m.held).filter_map(|_| replica.try_acquire()));
        }
        (registry, held)
    }

    proptest! {
        #[test]
        fn never_routes_to_an_unhealthy_or_wrong_model_replica(
            members in prop::collection::vec(member(), 0..12),
            kind in prop::sample::select(PolicyKind::ALL.to_vec()),
            seed: u64,
            tried in prop::collection::vec(0usize..12, 0..4),
        ) {
            let (registry, _held) = build(&members);
            let policy = inferrail_policy::build(kind, seed);
            let tried: Vec<ReplicaId> =
                tried.iter().map(|i| ReplicaId::new(format!("r{i}")).unwrap()).collect();
            let now = 20;
            for name in MODELS {
                let model = ModelId::new(name).unwrap();
                let fleet = registry.snapshot();
                let open = fleet
                    .serving(&model)
                    .iter()
                    .any(|r| r.is_routable(now) && r.has_room());
                for _ in 0..4 {
                    match select(&registry, policy.as_ref(), &model, now, &tried) {
                        Ok(lease) => {
                            let replica = lease.replica();
                            prop_assert!(replica.serves(&model));
                            prop_assert!(replica.is_routable(now));
                            if let Some(cap) = replica.limits().max_concurrency {
                                prop_assert!(replica.in_flight() <= cap);
                            }
                        }
                        Err(_) => prop_assert!(!open),
                    }
                }
            }
        }
    }

    #[test]
    fn untried_replicas_are_preferred_then_tried_ones_reused() {
        let members = vec![
            Member {
                models: vec![true, false, false],
                probe_down: false,
                ejected: false,
                cap: None,
                held: 0,
            };
            3
        ];
        let (registry, _) = build(&members);
        let policy = inferrail_policy::build(PolicyKind::RoundRobin, 0);
        let model = ModelId::new("a").unwrap();
        let tried: Vec<ReplicaId> = ["r0", "r2"]
            .iter()
            .map(|id| ReplicaId::new(id).unwrap())
            .collect();
        for _ in 0..5 {
            let lease = select(&registry, policy.as_ref(), &model, 1, &tried).unwrap();
            assert_eq!(lease.replica().id().as_str(), "r1");
        }
        let all: Vec<ReplicaId> = (0..3)
            .map(|i| ReplicaId::new(format!("r{i}")).unwrap())
            .collect();
        assert!(select(&registry, policy.as_ref(), &model, 1, &all).is_ok());
    }

    #[test]
    fn errors_tell_capacity_from_health() {
        let mut members = vec![Member {
            models: vec![true, false, false],
            probe_down: false,
            ejected: false,
            cap: Some(1),
            held: 1,
        }];
        let (registry, _held) = build(&members);
        let policy = inferrail_policy::build(PolicyKind::P2c, 0);
        let model = ModelId::new("a").unwrap();
        let err = select(&registry, policy.as_ref(), &model, 1, &[]).unwrap_err();
        assert!(err.message().contains("at capacity"), "{}", err.message());

        members[0].probe_down = true;
        let (registry, _held) = build(&members);
        let err = select(&registry, policy.as_ref(), &model, 1, &[]).unwrap_err();
        assert!(
            err.message().contains("no healthy replica"),
            "{}",
            err.message()
        );
        assert!(err.retry_after().is_some());

        let other = ModelId::new("b").unwrap();
        assert!(select(&registry, policy.as_ref(), &other, 1, &[]).is_err());
    }
}
