use std::sync::Arc;
use std::time::Duration;

use inferrail_control::{Lease, Nanos, Registry, Replica};
use inferrail_core::ids::{ModelId, ReplicaId};
use inferrail_index::{KvIndex, Slot};
use inferrail_policy::{
    Candidate, Choice, CostModel, Estimate, Explanation, Hits, RouteContext, RoutingPolicy,
};
use tokio::time::Instant;

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
    pub(crate) choice: Choice,
    /// The cost model's estimate on the chosen replica, from what it is
    /// predicted to hold. Its prefill is what the replica's queue grew by.
    pub(crate) estimate: Estimate,
}

/// What a routing decision knows about the request.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Request<'a> {
    pub(crate) model: &'a ModelId,
    /// Prompt tokens, counted or estimated.
    pub(crate) tokens: u32,
    pub(crate) prompt: Option<&'a Prompt>,
    pub(crate) index: Option<&'a KvIndex>,
    /// The conversation the request belongs to, when the client named one.
    pub(crate) session: Option<u64>,
    pub(crate) cost: CostModel,
    /// How old an engine's load report may be and still count.
    pub(crate) max_age: Duration,
    /// Whether to return every offered replica's prediction.
    pub(crate) explain: bool,
}

impl Request<'_> {
    fn blocks(&self) -> &[inferrail_core::ids::BlockHash] {
        self.prompt.map_or(&[], |p| &p.hashes)
    }

    fn context<'a>(&'a self, now: Nanos, candidates: &'a [Candidate<'a>]) -> RouteContext<'a> {
        RouteContext {
            model: self.model,
            prompt_tokens: self.tokens,
            blocks: self.blocks(),
            block_size: self
                .prompt
                .map_or(0, |p| u32::try_from(p.block_size).unwrap_or(u32::MAX)),
            session: self.session,
            now,
            candidates,
        }
    }
}

/// A replica as policies see it: the gateway's own counts, its queued
/// prefill, and the engine's report while that is fresh.
pub(crate) fn candidate(
    replica: &Replica,
    cached_tokens: u32,
    now: Nanos,
    instant: Instant,
    max_age: Duration,
) -> Candidate<'_> {
    let in_flight = replica.in_flight();
    let engine = replica
        .engine_load()
        .fresh(instant, max_age)
        .map(|s| s.load);
    Candidate {
        id: replica.id(),
        in_flight,
        cached_tokens,
        queued_secs: replica.queued(now).as_secs_f64(),
        running: engine.map_or_else(
            || in_flight.saturating_sub(replica.prefilling()),
            |e| e.running,
        ),
        kv_usage: engine.map(|e| e.kv_usage),
    }
}

/// The replicas a request may be offered: they serve the model, are
/// routable and have room; untried ones are preferred over ones this
/// request already used.
fn eligible<'a>(
    pool: &'a [Arc<Replica>],
    now: Nanos,
    tried: &[ReplicaId],
) -> Vec<&'a Arc<Replica>> {
    let mut eligible: Vec<&Arc<Replica>> = pool
        .iter()
        .filter(|r| r.is_routable(now) && r.has_room())
        .collect();
    if eligible.iter().any(|r| !tried.contains(r.id())) {
        eligible.retain(|r| !tried.contains(r.id()));
    }
    eligible
}

/// Picks a replica for the request and takes a lease on it. Each offered
/// replica comes with the prompt tokens the KV index predicts it has
/// cached. The chosen replica's queue grows by the request's estimated
/// prefill, whatever the policy.
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
    let mut eligible = eligible(pool, now, tried);
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
    let instant = Instant::now();
    let mut candidates: Vec<Candidate<'_>> = Vec::with_capacity(eligible.len());
    while !eligible.is_empty() {
        candidates.clear();
        candidates.extend(
            eligible
                .iter()
                .zip(&cached)
                .map(|(r, &tokens)| candidate(r, tokens, now, instant, request.max_age)),
        );
        let ctx = request.context(now, &candidates);
        let Some(choice) = policy.choose(&ctx).filter(|c| c.index < eligible.len()) else {
            break;
        };
        let i = choice.index;
        if let Some(lease) = eligible[i].try_acquire(request.tokens) {
            let estimate =
                request
                    .cost
                    .estimate(request.tokens, &candidates[i], Hits::gpu(cached[i]));
            let work = estimate.prefill_secs + estimate.tier_secs;
            lease.enqueue(
                now,
                Duration::try_from_secs_f64(work).unwrap_or(Duration::ZERO),
            );
            return Ok(Selected {
                lease,
                offered,
                cached_tokens: made.then_some(cached[i]),
                predictions,
                choice,
                estimate,
            });
        }
        eligible.remove(i);
        cached.remove(i);
    }
    Err(if pool.iter().any(|r| r.is_routable(now)) {
        GatewayError::unavailable(format!("every replica serving `{model}` is at capacity"))
    } else {
        GatewayError::unavailable(format!("no healthy replica serves `{model}`"))
    })
}

/// One replica serving the model, as an explanation shows it.
#[derive(Debug)]
pub(crate) struct Considered<'a> {
    pub(crate) replica: &'a Arc<Replica>,
    /// Why it was not offered to the policy.
    pub(crate) unavailable: Option<&'static str>,
    /// Its place among the candidates when it was offered.
    pub(crate) offered: Option<usize>,
}

/// What `select` would offer and what the policy thinks of it, without
/// taking a lease or changing any state.
pub(crate) fn explain<'f>(
    pool: &'f [Arc<Replica>],
    policy: &dyn RoutingPolicy,
    request: Request<'_>,
    now: Nanos,
) -> (
    Vec<Considered<'f>>,
    Vec<Candidate<'f>>,
    Option<Vec<u32>>,
    Explanation,
) {
    let eligible = eligible(pool, now, &[]);
    let predicted = predict(&eligible, request);
    let instant = Instant::now();
    let candidates: Vec<Candidate<'f>> = eligible
        .iter()
        .enumerate()
        .map(|(i, r)| {
            let cached = predicted.as_ref().map_or(0, |p| p[i]);
            candidate(r, cached, now, instant, request.max_age)
        })
        .collect();
    let explanation = policy.explain(&request.context(now, &candidates));
    let considered = pool
        .iter()
        .map(|replica| Considered {
            replica,
            unavailable: if !replica.health().probe_up() {
                Some("unhealthy")
            } else if !replica.is_routable(now) {
                Some("ejected")
            } else if !replica.has_room() {
                Some("full")
            } else {
                None
            },
            offered: eligible.iter().position(|r| Arc::ptr_eq(r, replica)),
        })
        .collect();
    (considered, candidates, predicted, explanation)
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
            session: None,
            cost: CostModel::default(),
            max_age: Duration::from_millis(250),
            explain: false,
        }
    }

    fn build_policy(kind: PolicyKind, seed: u64) -> Box<dyn RoutingPolicy> {
        inferrail_policy::build(kind, &inferrail_policy::Settings::seeded(seed))
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
            let policy = build_policy(kind, seed);
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
        let policy = build_policy(PolicyKind::RoundRobin, 0);
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
        let policy = build_policy(PolicyKind::RoundRobin, 0);
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
            tokens: prompt.tokens,
            prompt: Some(&prompt),
            index: Some(&index),
            explain: true,
            ..plain(&model)
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
        let policy = build_policy(PolicyKind::P2c, 0);
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
