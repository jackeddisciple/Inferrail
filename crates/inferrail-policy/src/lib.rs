//! Routing policy trait, baseline policies and the TTFT cost model.

use std::fmt;

use inferrail_core::config::PolicyKind;
use inferrail_core::ids::{ModelId, ReplicaId};

mod least_request;
mod p2c;
mod random;
mod rng;
mod round_robin;

pub use least_request::LeastRequest;
pub use p2c::P2c;
pub use random::Random;
pub use rng::{SplitMix, mix_seed};
pub use round_robin::RoundRobin;

/// A replica the gateway already found eligible: it serves the model, is
/// healthy and has room under its caps.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Candidate<'a> {
    pub id: &'a ReplicaId,
    pub in_flight: u32,
}

#[derive(Debug, Clone, Copy)]
pub struct RouteContext<'a> {
    pub model: &'a ModelId,
    pub candidates: &'a [Candidate<'a>],
}

pub trait RoutingPolicy: Send + Sync + fmt::Debug {
    fn kind(&self) -> PolicyKind;

    /// Picks an index into `ctx.candidates`; `None` only when it is empty.
    fn choose(&self, ctx: &RouteContext<'_>) -> Option<usize>;
}

/// Whether `kind` has its own implementation yet. The cache aware kinds need
/// the KV index; until it exists they route by their documented degraded
/// path, power of two choices.
pub const fn is_native(kind: PolicyKind) -> bool {
    !kind.is_cache_aware()
}

pub fn build(kind: PolicyKind, seed: u64) -> Box<dyn RoutingPolicy> {
    match kind {
        PolicyKind::RoundRobin => Box::new(RoundRobin::new(seed as usize)),
        PolicyKind::Random => Box::new(Random::new(seed)),
        PolicyKind::LeastRequest => Box::new(LeastRequest::new(seed)),
        PolicyKind::P2c => Box::new(P2c::new(seed)),
        PolicyKind::PrefixHash | PolicyKind::KvAware | PolicyKind::CostModel => {
            Box::new(Degraded {
                kind,
                fallback: P2c::new(seed),
            })
        }
    }
}

#[derive(Debug)]
struct Degraded {
    kind: PolicyKind,
    fallback: P2c,
}

impl RoutingPolicy for Degraded {
    fn kind(&self) -> PolicyKind {
        self.kind
    }

    fn choose(&self, ctx: &RouteContext<'_>) -> Option<usize> {
        self.fallback.choose(ctx)
    }
}

#[cfg(test)]
mod tests {
    use super::testing::Pool;
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn build_returns_the_requested_kind() {
        for kind in PolicyKind::ALL {
            assert_eq!(build(kind, 1).kind(), kind);
        }
        assert!(is_native(PolicyKind::P2c));
        assert!(!is_native(PolicyKind::CostModel));
    }

    #[test]
    fn cache_aware_kinds_fall_back_to_p2c_choices() {
        let pool = Pool::new(6);
        let candidates = pool.candidates(&[3, 1, 4, 1, 5, 9]);
        for kind in [
            PolicyKind::PrefixHash,
            PolicyKind::KvAware,
            PolicyKind::CostModel,
        ] {
            let (degraded, p2c) = (build(kind, 77), P2c::new(77));
            for _ in 0..200 {
                assert_eq!(
                    pool.pick(degraded.as_ref(), &candidates),
                    pool.pick(&p2c, &candidates)
                );
            }
        }
    }

    #[test]
    fn seeds_differ_per_model_but_are_stable() {
        assert_eq!(mix_seed(1, "llama"), mix_seed(1, "llama"));
        assert_ne!(mix_seed(1, "llama"), mix_seed(1, "qwen"));
        assert_ne!(mix_seed(1, "llama"), mix_seed(2, "llama"));
    }

    fn loads() -> impl Strategy<Value = Vec<u32>> {
        prop::collection::vec(prop_oneof![0u32..4, 0u32..10_000], 0..80)
    }

    proptest! {
        #[test]
        fn every_kind_picks_an_index_in_range(
            loads in loads(),
            seed: u64,
            kind in prop::sample::select(PolicyKind::ALL.to_vec()),
        ) {
            let pool = Pool::new(loads.len());
            let candidates = pool.candidates(&loads);
            let policy = build(kind, seed);
            let ctx = RouteContext { model: &pool.model, candidates: &candidates };
            for _ in 0..8 {
                match policy.choose(&ctx) {
                    None => prop_assert!(loads.is_empty()),
                    Some(i) => prop_assert!(i < loads.len()),
                }
            }
        }

        #[test]
        fn least_request_always_picks_a_minimum(loads in loads(), seed: u64) {
            prop_assume!(!loads.is_empty());
            let pool = Pool::new(loads.len());
            let candidates = pool.candidates(&loads);
            let min = *loads.iter().min().unwrap();
            let policy = LeastRequest::new(seed);
            for _ in 0..8 {
                prop_assert_eq!(loads[pool.pick(&policy, &candidates)], min);
            }
        }

        #[test]
        fn p2c_never_picks_a_unique_maximum(loads in loads(), seed: u64) {
            prop_assume!(loads.len() >= 2);
            let max = *loads.iter().max().unwrap();
            prop_assume!(loads.iter().filter(|&&l| l == max).count() == 1);
            let pool = Pool::new(loads.len());
            let candidates = pool.candidates(&loads);
            let policy = P2c::new(seed);
            for _ in 0..16 {
                prop_assert!(loads[pool.pick(&policy, &candidates)] < max);
            }
        }
    }
}

#[cfg(test)]
pub(crate) mod testing {
    use super::*;

    pub(crate) struct Pool {
        pub(crate) model: ModelId,
        pub(crate) ids: Vec<ReplicaId>,
    }

    impl Pool {
        pub(crate) fn new(n: usize) -> Self {
            Self {
                model: ModelId::new("m").unwrap(),
                ids: (0..n)
                    .map(|i| ReplicaId::new(format!("r{i}")).unwrap())
                    .collect(),
            }
        }

        pub(crate) fn candidates(&self, loads: &[u32]) -> Vec<Candidate<'_>> {
            self.ids
                .iter()
                .zip(loads.iter().copied().chain(std::iter::repeat(0)))
                .map(|(id, in_flight)| Candidate { id, in_flight })
                .collect()
        }

        pub(crate) fn pick(
            &self,
            policy: &dyn RoutingPolicy,
            candidates: &[Candidate<'_>],
        ) -> usize {
            policy
                .choose(&RouteContext {
                    model: &self.model,
                    candidates,
                })
                .expect("a candidate")
        }
    }

    pub(crate) fn chi_square(counts: &[u64]) -> f64 {
        let total: u64 = counts.iter().sum();
        let expected = total as f64 / counts.len() as f64;
        counts
            .iter()
            .map(|&c| (c as f64 - expected).powi(2) / expected)
            .sum()
    }
}
