use std::sync::Arc;

use inferrail_control::{Lease, Nanos, Registry, Replica};
use inferrail_core::ids::{ModelId, ReplicaId};
use inferrail_index::{KvIndex, Slot};
use inferrail_policy::{Candidate, RouteContext, RoutingPolicy};

use crate::errors::GatewayError;
use crate::prompts::Prompt;

/// The policy's choice, with what the KV index predicted.
#[derive(Debug)]
pub(crate) struct Selected {
    pub(crate) lease: Lease,
    /// How many replicas were offered to the policy.
    pub(crate) offered: usize,
    /// Prompt tokens predicted to be cached on the chosen replica, when
    /// the KV index made a prediction.
    pub(crate) cached_tokens: Option<u32>,
    /// Each offered replica's prediction, when asked for and made.
    pub(crate) predictions: Vec<(ReplicaId, u32)>,
}

/// What a routing decision knows about the prompt.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Request<'a> {
    pub(crate) model: &'a ModelId,
    /// Prompt tokens, counted or estimated, for the lease.
    pub(crate) tokens: u32,
    pub(crate) prompt: Option<&'a Prompt>,
    pub(crate) index: Option<&'a KvIndex>,
    /// Whether to return every offered replica's prediction.
    pub(crate) explain: bool,
}

/// Picks a replica for the request and takes a lease on it. Only replicas
/// that serve the model, are routable and have room are ever offered;
/// untried ones are preferred over ones this request already used. Each
/// is offered with the prompt tokens the KV index predicts it has cached.
pub(crate) fn select(
    registry: &Registry,
    policy: &dyn RoutingPolicy,
    request: Request<'_>,
    now: Nanos,
    tried: &[ReplicaId],
) -> Result<Selected, GatewayError> {
    let model = request.model;
    let fleet = registry.load();
    let pool = fleet.serving(model);
    let routable = pool.iter().filter(|r| r.is_routable(now));
    let mut eligible: Vec<&Arc<Replica>> = routable.clone().filter(|r| r.has_room()).collect();
    if eligible.iter().any(|r| !tried.contains(r.id())) {
        eligible.retain(|r| !tried.contains(r.id()));
    }
    let offered = eligible.len();
    let predicted = predict(&eligible, request);
    let made = predicted.is_some();
    let mut cached = predicted.unwrap_or_else(|| vec![0; eligible.len()]);
    let predictions = if request.explain && made {
        eligible
            .iter()
            .map(|r| r.id().clone())
            .zip(cached.iter().copied())
            .collect()
    } else {
        Vec::new()
    };
    let mut candidates: Vec<Candidate<'_>> = Vec::with_capacity(eligible.len());
    while !eligible.is_empty() {
        candidates.clear();
        candidates.extend(
            eligible
                .iter()
                .zip(&cached)
                .map(|(r, &cached_tokens)| Candidate {
                    id: r.id(),
                    in_flight: r.in_flight(),
                    cached_tokens,
                }),
        );
        let ctx = RouteContext {
            model,
            prompt_tokens: request.prompt.map(|p| p.tokens),
            candidates: &candidates,
        };
        let Some(i) = policy.choose(&ctx).filter(|&i| i < eligible.len()) else {
            break;
        };
        if let Some(lease) = eligible[i].try_acquire(request.tokens) {
            return Ok(Selected {
                lease,
                offered,
                cached_tokens: made.then_some(cached[i]),
                predictions,
            });
        }
        eligible.remove(i);
        cached.remove(i);
    }
    Err(if routable.count() == 0 {
        GatewayError::unavailable(format!("no healthy replica serves `{model}`"))
    } else {
        GatewayError::unavailable(format!("every replica serving `{model}` is at capacity"))
    })
}

/// Each eligible replica's predicted cached tokens, or `None` when the
/// prompt was not tokenized or the index is off, so nothing was predicted.
fn predict(eligible: &[&Arc<Replica>], request: Request<'_>) -> Option<Vec<u32>> {
    let (Some(prompt), Some(index)) = (request.prompt, request.index) else {
        return None;
    };
    if prompt.cacheable == 0 || eligible.is_empty() {
        return Some(vec![0; eligible.len()]);
    }
    let slots: Vec<Slot> = eligible.iter().map(|r| index.slot(r.id())).collect();
    Some(
        index
            .longest_prefix(&prompt.hashes[..prompt.cacheable], &slots)
            .into_iter()
            .map(|blocks| prompt.cached_tokens(blocks))
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;
    use inferrail_control::{HealthPolicy, Limits, Origin, ReplicaSpec};
    use inferrail_core::config::PolicyKind;
    use proptest::prelude::*;

    const MODELS: [&str; 3] = ["a", "b", "c"];

    fn plain(model: &ModelId) -> Request<'_> {
        Request {
            model,
            tokens: 0,
            prompt: None,
            index: None,
            explain: false,
        }
    }

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
                replica.health().eject(10, &policy());
            }
            held.extend((0..m.held).filter_map(|_| replica.try_acquire(0)));
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
                    match select(&registry, policy.as_ref(), plain(&model), now, &tried) {
                        Ok(selected) => {
                            let replica = selected.lease.replica();
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
            let selected = select(&registry, policy.as_ref(), plain(&model), 1, &tried).unwrap();
            assert_eq!(selected.offered, 1);
            assert_eq!(selected.lease.replica().id().as_str(), "r1");
        }
        let all: Vec<ReplicaId> = (0..3)
            .map(|i| ReplicaId::new(format!("r{i}")).unwrap())
            .collect();
        assert!(select(&registry, policy.as_ref(), plain(&model), 1, &all).is_ok());
    }

    #[test]
    fn candidates_carry_what_the_index_predicts() {
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
        let index = KvIndex::new();
        let prompt = Prompt {
            tokens: 64,
            hashes: (1..=4).map(inferrail_core::ids::BlockHash::new).collect(),
            cacheable: 3,
            block_size: 16,
            cache: inferrail_tokenize::CacheStats::default(),
        };
        index.record(
            index.slot(&ReplicaId::new("r1").unwrap()),
            &prompt.hashes,
            100,
        );
        let request = Request {
            model: &model,
            tokens: prompt.tokens,
            prompt: Some(&prompt),
            index: Some(&index),
            explain: true,
        };
        let mut seen = std::collections::HashSet::new();
        for _ in 0..6 {
            let selected = select(&registry, policy.as_ref(), request, 1, &[]).unwrap();
            let winner = selected.lease.replica().id().as_str().to_owned();
            assert_eq!(
                selected.cached_tokens,
                Some(if winner == "r1" { 48 } else { 0 })
            );
            assert_eq!(selected.lease.tokens(), 64);
            let mut predictions = selected.predictions.clone();
            predictions.sort();
            let r = |i: &str| ReplicaId::new(i).unwrap();
            assert_eq!(predictions, [(r("r0"), 0), (r("r1"), 48), (r("r2"), 0)]);
            seen.insert(winner);
        }
        assert_eq!(seen.len(), 3);
        let quiet = Request {
            explain: false,
            ..request
        };
        assert!(
            select(&registry, policy.as_ref(), quiet, 1, &[])
                .unwrap()
                .predictions
                .is_empty()
        );
        for unpredicted in [
            Request {
                index: None,
                ..request
            },
            Request {
                prompt: None,
                ..request
            },
        ] {
            let selected = select(&registry, policy.as_ref(), unpredicted, 1, &[]).unwrap();
            assert_eq!(selected.cached_tokens, None);
            assert!(selected.predictions.is_empty());
        }
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
        let err = select(&registry, policy.as_ref(), plain(&model), 1, &[]).unwrap_err();
        assert!(err.message().contains("at capacity"), "{}", err.message());

        members[0].probe_down = true;
        let (registry, _held) = build(&members);
        let err = select(&registry, policy.as_ref(), plain(&model), 1, &[]).unwrap_err();
        assert!(
            err.message().contains("no healthy replica"),
            "{}",
            err.message()
        );
        assert!(err.retry_after().is_some());

        let other = ModelId::new("b").unwrap();
        assert!(select(&registry, policy.as_ref(), plain(&other), 1, &[]).is_err());
    }
}
