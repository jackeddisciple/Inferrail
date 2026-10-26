use inferrail_core::config::CostModelConfig;
use serde::Serialize;

use crate::Candidate;

/// Load time per token from the CPU tier: about 128 KiB of KV per token
/// for an 8B model over PCIe 4.
const CPU_LOAD_TOKEN_SECS: f64 = 5e-6;
/// Load time per token from local NVMe, about five times the CPU tier.
const DISK_LOAD_TOKEN_SECS: f64 = 25e-6;

/// Estimated time to first token on one replica, `W + P + X`: prefill work
/// queued there, prefill of the uncached part of the prompt, and loading
/// cached blocks from slower tiers.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CostModel {
    /// `a`: prefill time per uncached token.
    pub prefill_token_secs: f64,
    /// `b`: attention time per pair of an uncached token and a token it
    /// attends to.
    pub prefill_pair_secs: f64,
    /// `d`: fixed time per prefill.
    pub prefill_fixed_secs: f64,
    /// Time each running request adds to the queue.
    pub decode_interference_secs: f64,
    pub cpu_load_token_secs: f64,
    pub disk_load_token_secs: f64,
}

impl From<CostModelConfig> for CostModel {
    fn from(config: CostModelConfig) -> Self {
        Self {
            prefill_token_secs: config.prefill_token_secs,
            prefill_pair_secs: config.prefill_pair_secs,
            prefill_fixed_secs: config.prefill_fixed_secs,
            decode_interference_secs: config.decode_interference_secs,
            cpu_load_token_secs: CPU_LOAD_TOKEN_SECS,
            disk_load_token_secs: DISK_LOAD_TOKEN_SECS,
        }
    }
}

impl Default for CostModel {
    fn default() -> Self {
        Self::from(CostModelConfig::default())
    }
}

/// Where a replica holds the leading tokens of a prompt: the first `gpu`
/// tokens on the GPU, the next `cpu` in CPU memory, the next `disk` on disk.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Hits {
    pub gpu: u32,
    pub cpu: u32,
    pub disk: u32,
}

impl Hits {
    pub fn gpu(tokens: u32) -> Self {
        Self {
            gpu: tokens,
            ..Self::default()
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize)]
pub struct Estimate {
    /// `W`: queued prefill and interference from running requests.
    pub queue_secs: f64,
    /// `P`: prefill of the tokens computed.
    pub prefill_secs: f64,
    /// `X`: loading cached tokens from CPU or disk.
    pub tier_secs: f64,
    pub ttft_secs: f64,
}

/// A part of the estimate, named in explanations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Term {
    Queue,
    Prefill,
    TierLoad,
}

impl Term {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Queue => "queue",
            Self::Prefill => "prefill",
            Self::TierLoad => "tier_load",
        }
    }
}

impl Estimate {
    fn new(queue_secs: f64, prefill_secs: f64, tier_secs: f64) -> Self {
        Self {
            queue_secs,
            prefill_secs,
            tier_secs,
            ttft_secs: queue_secs + prefill_secs + tier_secs,
        }
    }

    /// The term in which this estimate beats `other` by the most.
    pub fn ahead_of(&self, other: &Self) -> Term {
        [
            (Term::Queue, other.queue_secs - self.queue_secs),
            (Term::Prefill, other.prefill_secs - self.prefill_secs),
            (Term::TierLoad, other.tier_secs - self.tier_secs),
        ]
        .into_iter()
        .fold((Term::Queue, f64::NEG_INFINITY), |best, (term, lead)| {
            if lead > best.1 { (term, lead) } else { best }
        })
        .0
    }
}

impl CostModel {
    /// `P(u, c) = a*u + b*u*(c + u/2) + d` for `u` tokens computed while
    /// attending to `c` cached ones.
    pub fn prefill_secs(&self, uncached: u32, cached: u32) -> f64 {
        let (u, c) = (f64::from(uncached), f64::from(cached));
        self.prefill_token_secs * u
            + self.prefill_pair_secs * u * (c + u / 2.0)
            + self.prefill_fixed_secs
    }

    /// `W`: what is queued on the replica ahead of a new request.
    pub fn queue_secs(&self, candidate: &Candidate<'_>) -> f64 {
        candidate.queued_secs.max(0.0)
            + self.decode_interference_secs * f64::from(candidate.running)
    }

    /// The estimate for a prompt of `prompt` tokens on `candidate`, whose
    /// cached prefix lies in the tiers `hits` describes. A tier hit only
    /// counts when loading it beats computing it again.
    pub fn estimate(&self, prompt: u32, candidate: &Candidate<'_>, hits: Hits) -> Estimate {
        let queue = self.queue_secs(candidate);
        let gpu = hits.gpu.min(prompt);
        let cpu = hits.cpu.min(prompt - gpu);
        let disk = hits.disk.min(prompt - gpu - cpu);
        let recompute = self.prefill_secs(prompt - gpu, gpu);
        if cpu + disk == 0 {
            return Estimate::new(queue, recompute, 0.0);
        }
        let held = gpu + cpu + disk;
        let load =
            self.cpu_load_token_secs * f64::from(cpu) + self.disk_load_token_secs * f64::from(disk);
        let rest = self.prefill_secs(prompt - held, held);
        if load + rest < recompute {
            Estimate::new(queue, rest, load)
        } else {
            Estimate::new(queue, recompute, 0.0)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CostModelPolicy, RouteContext, RoutingPolicy, Settings};
    use inferrail_core::ids::{ModelId, ReplicaId};
    use proptest::prelude::*;

    fn linear(token_secs: f64) -> CostModel {
        CostModel {
            prefill_token_secs: token_secs,
            prefill_pair_secs: 0.0,
            prefill_fixed_secs: 0.0,
            ..CostModel::default()
        }
    }

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    /// The worked example: prefill at 10,000 tokens a second, a 6,000 token
    /// request sharing a 4,000 token system prompt.
    #[test]
    fn the_worked_example_picks_the_replica_with_cache_and_a_short_queue() {
        let ids: Vec<ReplicaId> = ["r1", "r2", "r3"]
            .iter()
            .map(|i| ReplicaId::new(*i).unwrap())
            .collect();
        let model = ModelId::new("m").unwrap();
        let cost = linear(1e-4);
        let replica = |i: usize, cached: u32, queued: u32| Candidate {
            cached_tokens: cached,
            queued_secs: cost.prefill_token_secs * f64::from(queued),
            ..Candidate::new(&ids[i], 1)
        };
        let fleet = [
            replica(0, 4_000, 6_000),
            replica(1, 0, 0),
            replica(2, 4_000, 2_000),
        ];
        let ttft: Vec<f64> = fleet
            .iter()
            .map(|c| {
                cost.estimate(6_000, c, Hits::gpu(c.cached_tokens))
                    .ttft_secs
            })
            .collect();
        assert!(close(ttft[0], 0.8) && close(ttft[1], 0.6) && close(ttft[2], 0.4));
        let settings = Settings {
            cost,
            hot_prefix: None,
            ..Settings::seeded(1)
        };
        let policy = CostModelPolicy::new(&settings);
        let ctx = RouteContext {
            prompt_tokens: 6_000,
            ..RouteContext::new(&model, &fleet)
        };
        let choice = policy.choose(&ctx).unwrap();
        assert_eq!(choice.index, 2);
        assert!(close(choice.estimate.unwrap().ttft_secs, 0.4));
        let explained = policy.explain(&ctx);
        assert_eq!(explained.winner, Some(2));
        assert_eq!(explained.decided_by, Some(Term::Prefill));

        let mut busier = fleet;
        busier[2].queued_secs = 0.5;
        let ctx = RouteContext {
            prompt_tokens: 6_000,
            ..RouteContext::new(&model, &busier)
        };
        assert_eq!(policy.choose(&ctx).unwrap().index, 1);
        assert_eq!(policy.explain(&ctx).decided_by, Some(Term::Queue));
    }

    #[test]
    fn prefill_follows_the_calibrated_shape() {
        let cost = CostModel::default();
        assert!(close(cost.prefill_secs(0, 0), 5e-3));
        let cold = cost.prefill_secs(6_000, 0);
        assert!(close(cold, 100e-6 * 6e3 + 3.3e-9 * 6e3 * 3e3 + 5e-3));
        let warm = cost.prefill_secs(2_000, 4_000);
        assert!(warm < cold / 2.0);
    }

    #[test]
    fn decodes_interfere_only_when_priced() {
        let id = ReplicaId::new("r").unwrap();
        let busy = Candidate {
            running: 10,
            queued_secs: 0.25,
            ..Candidate::new(&id, 10)
        };
        assert!(close(CostModel::default().queue_secs(&busy), 0.25));
        let priced = CostModel {
            decode_interference_secs: 0.01,
            ..CostModel::default()
        };
        assert!(close(priced.queue_secs(&busy), 0.35));
    }

    #[test]
    fn tier_hits_count_only_when_loading_beats_recompute() {
        let id = ReplicaId::new("r").unwrap();
        let idle = Candidate::new(&id, 0);
        let cost = linear(1e-4);
        let hits = Hits {
            gpu: 1_000,
            cpu: 2_000,
            disk: 0,
        };
        let loaded = cost.estimate(4_000, &idle, hits);
        assert!(close(loaded.tier_secs, 2_000.0 * CPU_LOAD_TOKEN_SECS));
        assert!(close(loaded.prefill_secs, 0.1));
        let slow = CostModel {
            cpu_load_token_secs: 2e-4,
            ..cost
        };
        let recomputed = slow.estimate(4_000, &idle, hits);
        assert_eq!(recomputed.tier_secs, 0.0);
        assert!(close(recomputed.prefill_secs, 0.3));
        let clipped = cost.estimate(
            1_500,
            &idle,
            Hits {
                gpu: 1_000,
                cpu: 2_000,
                disk: 9_000,
            },
        );
        assert!(close(clipped.tier_secs, 500.0 * CPU_LOAD_TOKEN_SECS));
        assert_eq!(clipped.prefill_secs, 0.0);
    }

    #[test]
    fn the_leading_term_is_the_biggest_gap() {
        let a = Estimate::new(0.1, 0.2, 0.0);
        let b = Estimate::new(0.15, 0.6, 0.0);
        assert_eq!(a.ahead_of(&b), Term::Prefill);
        let idle = Estimate::new(0.0, 0.6, 0.0);
        let queued = Estimate::new(0.5, 0.2, 0.0);
        assert_eq!(idle.ahead_of(&queued), Term::Queue);
        let c = Estimate::new(0.1, 0.2, 0.05);
        let d = Estimate::new(0.1, 0.2, 0.3);
        assert_eq!(c.ahead_of(&d), Term::TierLoad);
        assert_eq!(Term::TierLoad.as_str(), "tier_load");
    }

    proptest! {
        #[test]
        fn more_cache_never_costs_more(
            prompt in 1u32..200_000,
            hit in 0u32..200_000,
            extra in 1u32..10_000,
            queued in 0.0f64..10.0,
        ) {
            let id = ReplicaId::new("r").unwrap();
            let candidate = Candidate { queued_secs: queued, ..Candidate::new(&id, 0) };
            let cost = CostModel::default();
            let hit = hit.min(prompt - 1);
            let more = (hit + extra).min(prompt - 1);
            let less = cost.estimate(prompt, &candidate, Hits::gpu(hit));
            let most = cost.estimate(prompt, &candidate, Hits::gpu(more));
            prop_assert!(most.ttft_secs <= less.ttft_secs + 1e-12);
            prop_assert!(less.ttft_secs.is_finite());
            prop_assert!(close(less.ttft_secs, less.queue_secs + less.prefill_secs + less.tier_secs));
        }

        #[test]
        fn a_tier_hit_never_costs_more_than_ignoring_it(
            prompt in 1u32..100_000,
            gpu in 0u32..100_000,
            cpu in 0u32..100_000,
            disk in 0u32..100_000,
        ) {
            let id = ReplicaId::new("r").unwrap();
            let idle = Candidate::new(&id, 0);
            let cost = CostModel::default();
            let with = cost.estimate(prompt, &idle, Hits { gpu, cpu, disk });
            let without = cost.estimate(prompt, &idle, Hits::gpu(gpu));
            prop_assert!(with.ttft_secs <= without.ttft_secs + 1e-12);
        }
    }
}
