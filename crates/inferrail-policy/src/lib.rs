//! Routing policy trait, baseline policies and the TTFT cost model.

use std::fmt;

use inferrail_core::config::{Config, HotPrefixConfig, KvAwareConfig, ModelConfig, PolicyKind};
use inferrail_core::ids::{BlockHash, ModelId, ReplicaId};
use serde::Serialize;

mod cost;
mod cost_model;
mod guards;
mod hashing;
mod kv_aware;
mod least_request;
mod p2c;
mod prefix_hash;
mod random;
mod rng;
mod round_robin;

pub use cost::{CostModel, Estimate, Hits, Term};
pub use cost_model::CostModelPolicy;
pub use guards::load_cap;
pub use kv_aware::KvAware;
pub use least_request::LeastRequest;
pub use p2c::P2c;
pub use prefix_hash::PrefixHash;
pub use random::Random;
pub use rng::{SplitMix, mix_seed};
pub use round_robin::RoundRobin;

/// A replica the gateway already found eligible: it serves the model, is
/// healthy and has room under its caps.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Candidate<'a> {
    pub id: &'a ReplicaId,
    pub in_flight: u32,
    /// Prompt tokens the KV index predicts this replica serves from cache.
    pub cached_tokens: u32,
    /// Prefill work already queued on the replica, in seconds.
    pub queued_secs: f64,
    /// Requests the replica is running.
    pub running: u32,
    /// Share of its KV cache the engine reports in use, while that report
    /// is fresh.
    pub kv_usage: Option<f64>,
}

impl<'a> Candidate<'a> {
    /// A candidate known only by its requests in flight.
    pub fn new(id: &'a ReplicaId, in_flight: u32) -> Self {
        Self {
            id,
            in_flight,
            cached_tokens: 0,
            queued_secs: 0.0,
            running: 0,
            kv_usage: None,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct RouteContext<'a> {
    pub model: &'a ModelId,
    /// The prompt's length in tokens: counted when it was tokenized at the
    /// edge, estimated from the body otherwise.
    pub prompt_tokens: u32,
    /// Hashes of the prompt's full blocks; empty when it was not tokenized.
    pub blocks: &'a [BlockHash],
    /// Names the conversation a request belongs to, when the client did.
    pub session: Option<u64>,
    /// The gateway's clock in nanoseconds, for rates.
    pub now: u64,
    pub candidates: &'a [Candidate<'a>],
}

impl<'a> RouteContext<'a> {
    /// A context with nothing known about the prompt.
    pub fn new(model: &'a ModelId, candidates: &'a [Candidate<'a>]) -> Self {
        Self {
            model,
            prompt_tokens: 0,
            blocks: &[],
            session: None,
            now: 0,
            candidates,
        }
    }
}

/// What steered a choice away from the policy's plain rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Steer {
    /// The conversation's last replica, kept by session affinity.
    Session,
    /// An owner of a hot root prefix, priced as holding it.
    HotPrefix,
    /// The prompt had no full block to hash, so a fallback rule chose.
    NoPrefix,
}

impl Steer {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Session => "session",
            Self::HotPrefix => "hot_prefix",
            Self::NoPrefix => "no_prefix",
        }
    }
}

/// Why a policy passed over a candidate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Exclusion {
    /// It would hold more than its share under bounded load.
    BoundedLoad,
    /// Its KV cache is too full for a prompt this long.
    KvPressure,
}

impl Exclusion {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::BoundedLoad => "bounded_load",
            Self::KvPressure => "kv_pressure",
        }
    }
}

/// Candidates the load guards set aside in one decision.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct Guarded {
    pub bounded_load: u32,
    pub kv_pressure: u32,
    /// Every candidate was under KV pressure, so that guard stood down.
    pub kv_waived: bool,
}

/// A policy's pick, with what shaped it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Choice {
    pub index: usize,
    pub steer: Option<Steer>,
    pub guarded: Guarded,
    /// The cost model's estimate for the chosen replica.
    pub estimate: Option<Estimate>,
}

impl Choice {
    pub fn new(index: usize) -> Self {
        Self {
            index,
            steer: None,
            guarded: Guarded::default(),
            estimate: None,
        }
    }
}

/// How a policy sees each candidate, without routing anything.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Explanation {
    /// The policy's rule in words.
    pub rule: &'static str,
    /// The candidate `choose` would pick now, unless chance decides it.
    pub winner: Option<usize>,
    /// The term of the estimate that set the winner furthest ahead of the
    /// runner up.
    pub decided_by: Option<Term>,
    pub steer: Option<Steer>,
    pub guarded: Guarded,
    pub hot_prefix: Option<HotPrefix>,
    /// One entry per candidate, in the order they were offered.
    pub candidates: Vec<Scored>,
}

impl Explanation {
    /// A rule that scores nothing, with the candidate it would pick.
    pub fn plain(rule: &'static str, winner: Option<usize>, candidates: usize) -> Self {
        Self {
            rule,
            winner,
            decided_by: None,
            steer: None,
            guarded: Guarded::default(),
            hot_prefix: None,
            candidates: vec![Scored::default(); candidates],
        }
    }
}

/// The demand a root prefix brings, in replicas' worth of prefill, and how
/// many owners it gets for it.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct HotPrefix {
    pub demand: f64,
    pub owners: usize,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize)]
pub struct Scored {
    pub excluded: Option<Exclusion>,
    /// The policy's score, higher is better, for policies that score.
    pub score: Option<f64>,
    pub estimate: Option<Estimate>,
    /// An owner of the prompt's hot root prefix.
    pub owner: bool,
}

pub trait RoutingPolicy: Send + Sync + fmt::Debug {
    fn kind(&self) -> PolicyKind;

    /// Picks a candidate; `None` only when there are none.
    fn choose(&self, ctx: &RouteContext<'_>) -> Option<Choice>;

    /// How the policy sees every candidate now. Changes no state, so the
    /// next `choose` is unaffected.
    fn explain(&self, ctx: &RouteContext<'_>) -> Explanation;
}

/// Everything a policy may need from the configuration, for one model.
#[derive(Debug, Clone, PartialEq)]
pub struct Settings {
    pub seed: u64,
    pub epsilon: f64,
    pub kappa: f64,
    pub short_prompt_tokens: u32,
    pub prefix_blocks: u32,
    pub kv_aware: KvAwareConfig,
    pub cost: CostModel,
    /// `None` when hot prefix replication is off.
    pub hot_prefix: Option<HotPrefixConfig>,
    pub session_affinity: bool,
}

impl Settings {
    /// The settings for `model`, with randomness drawn from `seed`.
    pub fn new(config: &Config, model: &ModelConfig, seed: u64) -> Self {
        let routing = &config.routing;
        Self {
            seed,
            epsilon: routing.epsilon,
            kappa: routing.kappa,
            short_prompt_tokens: routing.short_prompt_tokens,
            prefix_blocks: routing.prefix_hash.blocks,
            kv_aware: routing.kv_aware.clone(),
            cost: CostModel::from(config.cost_model_for(model)),
            hot_prefix: routing
                .hot_prefix
                .enabled
                .then(|| routing.hot_prefix.clone()),
            session_affinity: routing.session_affinity,
        }
    }

    /// Default settings with randomness drawn from `seed`.
    pub fn seeded(seed: u64) -> Self {
        let routing = inferrail_core::config::RoutingConfig::default();
        Self {
            seed,
            epsilon: routing.epsilon,
            kappa: routing.kappa,
            short_prompt_tokens: routing.short_prompt_tokens,
            prefix_blocks: routing.prefix_hash.blocks,
            kv_aware: routing.kv_aware,
            cost: CostModel::from(routing.cost_model),
            hot_prefix: Some(routing.hot_prefix),
            session_affinity: routing.session_affinity,
        }
    }
}

pub fn build(kind: PolicyKind, settings: &Settings) -> Box<dyn RoutingPolicy> {
    let seed = settings.seed;
    match kind {
        PolicyKind::RoundRobin => Box::new(RoundRobin::new(seed as usize)),
        PolicyKind::Random => Box::new(Random::new(seed)),
        PolicyKind::LeastRequest => Box::new(LeastRequest::new(seed)),
        PolicyKind::P2c => Box::new(P2c::new(seed)),
        PolicyKind::PrefixHash => Box::new(PrefixHash::new(settings.prefix_blocks, seed)),
        PolicyKind::KvAware => Box::new(KvAware::new(&settings.kv_aware, seed)),
        PolicyKind::CostModel => Box::new(CostModelPolicy::new(settings)),
    }
}

/// An index below `len` that `keep` accepts, drawn uniformly by `rng`;
/// `None` when it accepts none.
pub(crate) fn pick_among(
    rng: &SplitMix,
    len: usize,
    mut keep: impl FnMut(usize) -> bool,
) -> Option<usize> {
    let mut chosen = None;
    let mut seen = 0;
    for i in 0..len {
        if keep(i) {
            seen += 1;
            if seen == 1 || rng.below(seen) == 0 {
                chosen = Some(i);
            }
        }
    }
    chosen
}

#[cfg(test)]
mod tests {
    use super::testing::Pool;
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn build_returns_the_requested_kind() {
        for kind in PolicyKind::ALL {
            assert_eq!(build(kind, &Settings::seeded(1)).kind(), kind);
        }
    }

    #[test]
    fn settings_follow_the_config_and_the_model() {
        let config = Config::from_yaml_str(
            "models:\n  - name: a\n  - name: b\n    cost_model: {prefill_token_secs: 0.001}\nreplicas:\n  - {id: r, url: 'http://h:1', models: [a, b]}\nrouting:\n  epsilon: 0.5\n  session_affinity: true\n  hot_prefix: {enabled: false}\n",
        )
        .unwrap();
        let a = Settings::new(&config, &config.models[0], 3);
        assert_eq!(a.seed, 3);
        assert_eq!(a.epsilon, 0.5);
        assert!(a.session_affinity);
        assert_eq!(a.hot_prefix, None);
        assert_eq!(a.cost, CostModel::from(config.routing.cost_model));
        let b = Settings::new(&config, &config.models[1], 3);
        assert_eq!(b.cost.prefill_token_secs, 0.001);
        let defaults = Settings::seeded(3);
        assert!(defaults.hot_prefix.is_some());
        assert!(!defaults.session_affinity);
    }

    #[test]
    fn seeds_differ_per_model_but_are_stable() {
        assert_eq!(mix_seed(1, "llama"), mix_seed(1, "llama"));
        assert_ne!(mix_seed(1, "llama"), mix_seed(1, "qwen"));
        assert_ne!(mix_seed(1, "llama"), mix_seed(2, "llama"));
    }

    #[test]
    fn pick_among_is_uniform_over_what_it_keeps() {
        let rng = SplitMix::new(5);
        let mut counts = [0u64; 6];
        for _ in 0..30_000 {
            let i = pick_among(&rng, 6, |i| i % 2 == 0).unwrap();
            counts[i] += 1;
        }
        assert_eq!((counts[1], counts[3], counts[5]), (0, 0, 0));
        let kept = [counts[0], counts[2], counts[4]];
        let stat = testing::chi_square(&kept);
        assert!(stat < 13.82, "chi square {stat} for {kept:?}");
        assert_eq!(pick_among(&rng, 4, |_| false), None);
    }

    fn loads() -> impl Strategy<Value = Vec<u32>> {
        prop::collection::vec(prop_oneof![0u32..4, 0u32..10_000], 0..80)
    }

    proptest! {
        #[test]
        fn every_kind_picks_an_index_in_range_and_explains_every_candidate(
            loads in loads(),
            seed: u64,
            kind in prop::sample::select(PolicyKind::ALL.to_vec()),
        ) {
            let pool = Pool::new(loads.len());
            let candidates = pool.candidates(&loads);
            let policy = build(kind, &Settings::seeded(seed));
            let ctx = RouteContext::new(&pool.model, &candidates);
            for _ in 0..8 {
                let explained = policy.explain(&ctx);
                prop_assert_eq!(explained.candidates.len(), loads.len());
                match policy.choose(&ctx) {
                    None => prop_assert!(loads.is_empty()),
                    Some(choice) => prop_assert!(choice.index < loads.len()),
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
                .map(|(id, in_flight)| Candidate::new(id, in_flight))
                .collect()
        }

        pub(crate) fn pick(
            &self,
            policy: &dyn RoutingPolicy,
            candidates: &[Candidate<'_>],
        ) -> usize {
            policy
                .choose(&RouteContext::new(&self.model, candidates))
                .expect("a candidate")
                .index
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

    pub(crate) fn chain(seed: u64, len: usize) -> Vec<BlockHash> {
        (0..len as u64)
            .map(|i| {
                let x = (seed << 32 | i).wrapping_mul(0x9e37_79b9_7f4a_7c15);
                BlockHash::new(u128::from(x) << 64 | u128::from(x.rotate_left(29) ^ i))
            })
            .collect()
    }
}
