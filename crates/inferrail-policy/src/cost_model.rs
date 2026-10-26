use inferrail_core::config::PolicyKind;

use crate::cost::{CostModel, Estimate, Hits};
use crate::guards::{load_cap, under_pressure, within_load};
use crate::{
    Choice, Exclusion, Explanation, Guarded, RouteContext, RoutingPolicy, Scored, Settings,
    SplitMix, pick_among,
};

const RULE: &str = "the lowest estimated time to first token within the load guards";

/// Inferrail's policy: the replica with the lowest estimated time to first
/// token, `W + P + X`, among those the load guards allow. Bounded load
/// skips replicas past their share of requests in flight, and KV pressure
/// keeps long prompts off replicas whose cache is nearly full.
#[derive(Debug)]
pub struct CostModelPolicy {
    cost: CostModel,
    epsilon: f64,
    kappa: f64,
    short_prompt_tokens: u32,
    rng: SplitMix,
}

/// The candidates as the estimates and the guards see them.
struct Plan {
    estimates: Vec<Estimate>,
    excluded: Vec<Option<Exclusion>>,
    guarded: Guarded,
}

impl Plan {
    fn allowed(&self, i: usize) -> bool {
        self.excluded[i].is_none()
    }

    fn ttft(&self, i: usize) -> f64 {
        self.estimates[i].ttft_secs
    }
}

impl CostModelPolicy {
    pub fn new(settings: &Settings) -> Self {
        Self {
            cost: settings.cost,
            epsilon: settings.epsilon,
            kappa: settings.kappa,
            short_prompt_tokens: settings.short_prompt_tokens,
            rng: SplitMix::new(settings.seed),
        }
    }

    /// Estimates every candidate and applies the guards.
    fn plan(&self, ctx: &RouteContext<'_>) -> Plan {
        let candidates = ctx.candidates;
        let n = candidates.len();
        let prompt = ctx.prompt_tokens;
        let estimates: Vec<Estimate> = candidates
            .iter()
            .map(|c| self.cost.estimate(prompt, c, Hits::gpu(c.cached_tokens)))
            .collect();
        let mut excluded = vec![None; n];
        let mut guarded = Guarded::default();
        let cap = load_cap(candidates, self.epsilon);
        for (slot, candidate) in excluded.iter_mut().zip(candidates) {
            if !within_load(candidate, cap) {
                *slot = Some(Exclusion::BoundedLoad);
                guarded.bounded_load += 1;
            }
        }
        let pressured: Vec<usize> = (0..n)
            .filter(|&i| {
                let c = &candidates[i];
                let uncached = prompt - c.cached_tokens.min(prompt);
                excluded[i].is_none()
                    && under_pressure(c, self.kappa, uncached, self.short_prompt_tokens)
            })
            .collect();
        if pressured.len() == excluded.iter().filter(|e| e.is_none()).count() {
            guarded.kv_waived = !pressured.is_empty();
        } else {
            for &i in &pressured {
                excluded[i] = Some(Exclusion::KvPressure);
            }
            guarded.kv_pressure = pressured.len() as u32;
        }
        Plan {
            estimates,
            excluded,
            guarded,
        }
    }

    /// Whether candidate `i` is among the cheapest allowed: lowest estimate,
    /// then fewest requests in flight.
    fn cheapest(ctx: &RouteContext<'_>, plan: &Plan) -> impl Fn(usize) -> bool {
        let n = ctx.candidates.len();
        let low = (0..n)
            .filter(|&i| plan.allowed(i))
            .map(|i| plan.ttft(i))
            .fold(f64::INFINITY, f64::min);
        let fewest = (0..n)
            .filter(|&i| plan.allowed(i) && plan.ttft(i) == low)
            .map(|i| ctx.candidates[i].in_flight)
            .min()
            .unwrap_or(0);
        let best: Vec<bool> = (0..n)
            .map(|i| {
                plan.allowed(i) && plan.ttft(i) == low && ctx.candidates[i].in_flight == fewest
            })
            .collect();
        move |i| best[i]
    }
}

impl RoutingPolicy for CostModelPolicy {
    fn kind(&self) -> PolicyKind {
        PolicyKind::CostModel
    }

    fn choose(&self, ctx: &RouteContext<'_>) -> Option<Choice> {
        if ctx.candidates.is_empty() {
            return None;
        }
        let plan = self.plan(ctx);
        let best = Self::cheapest(ctx, &plan);
        let index = pick_among(&self.rng, ctx.candidates.len(), best)?;
        Some(Choice {
            index,
            steer: None,
            guarded: plan.guarded,
            estimate: Some(plan.estimates[index]),
        })
    }

    fn explain(&self, ctx: &RouteContext<'_>) -> Explanation {
        let n = ctx.candidates.len();
        if n == 0 {
            return Explanation::plain(RULE, None, 0);
        }
        let plan = self.plan(ctx);
        let candidates = (0..n)
            .map(|i| Scored {
                excluded: plan.excluded[i],
                score: None,
                estimate: Some(plan.estimates[i]),
                owner: false,
            })
            .collect();
        let best = Self::cheapest(ctx, &plan);
        let tied: Vec<usize> = (0..n).filter(|&i| best(i)).collect();
        let winner = (tied.len() == 1).then(|| tied[0]);
        let decided_by = winner.and_then(|w| {
            let runner_up = (0..n)
                .filter(|&i| i != w && plan.allowed(i))
                .min_by(|&a, &b| plan.ttft(a).total_cmp(&plan.ttft(b)))?;
            (plan.ttft(runner_up) > plan.ttft(w))
                .then(|| plan.estimates[w].ahead_of(&plan.estimates[runner_up]))
        });
        Explanation {
            rule: RULE,
            winner,
            decided_by,
            steer: None,
            guarded: plan.guarded,
            hot_prefix: None,
            candidates,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::{Pool, chain};
    use crate::{Candidate, Term};
    use inferrail_core::ids::{BlockHash, ReplicaId};
    use proptest::prelude::*;

    fn settings() -> Settings {
        Settings {
            hot_prefix: None,
            ..Settings::seeded(1)
        }
    }

    fn fleet<'a>(
        ids: &'a [ReplicaId],
        rows: &[(u32, u32, f64, Option<f64>)],
    ) -> Vec<Candidate<'a>> {
        ids.iter()
            .zip(rows)
            .map(|(id, &(in_flight, cached, queued, kv))| Candidate {
                cached_tokens: cached,
                queued_secs: queued,
                running: in_flight,
                kv_usage: kv,
                ..Candidate::new(id, in_flight)
            })
            .collect()
    }

    fn context<'a>(
        pool: &'a Pool,
        candidates: &'a [Candidate<'a>],
        prompt: u32,
        blocks: &'a [BlockHash],
    ) -> RouteContext<'a> {
        RouteContext {
            prompt_tokens: prompt,
            blocks,
            ..RouteContext::new(&pool.model, candidates)
        }
    }

    #[test]
    fn the_best_cache_loses_to_bounded_load() {
        let pool = Pool::new(3);
        let policy = CostModelPolicy::new(&settings());
        let candidates = fleet(
            &pool.ids,
            &[(9, 4_000, 0.0, None), (2, 0, 0.0, None), (1, 0, 0.0, None)],
        );
        let choice = policy
            .choose(&context(&pool, &candidates, 4_200, &[]))
            .unwrap();
        assert_eq!(load_cap(&candidates, 1.0), 9);
        assert_ne!(choice.index, 0);
        assert_eq!(choice.guarded.bounded_load, 1);
        let explained = policy.explain(&context(&pool, &candidates, 4_200, &[]));
        assert_eq!(
            explained.candidates[0].excluded,
            Some(Exclusion::BoundedLoad)
        );
        assert_eq!(explained.winner, Some(choice.index));
    }

    #[test]
    fn long_prompts_avoid_a_full_kv_cache_short_ones_do_not() {
        let pool = Pool::new(2);
        let policy = CostModelPolicy::new(&settings());
        let candidates = fleet(
            &pool.ids,
            &[(1, 3_000, 0.0, Some(0.97)), (1, 0, 0.0, Some(0.5))],
        );
        let long = policy
            .choose(&context(&pool, &candidates, 3_500, &[]))
            .unwrap();
        assert_eq!(long.index, 1);
        assert_eq!(long.guarded.kv_pressure, 1);
        let short = policy
            .choose(&context(&pool, &candidates, 3_200, &[]))
            .unwrap();
        assert_eq!(short.index, 0);
        assert_eq!(short.guarded.kv_pressure, 0);

        let full = fleet(
            &pool.ids,
            &[(1, 3_000, 0.0, Some(0.97)), (1, 0, 0.0, Some(0.99))],
        );
        let waived = policy.choose(&context(&pool, &full, 3_500, &[])).unwrap();
        assert_eq!(waived.index, 0);
        assert!(waived.guarded.kv_waived);
    }

    #[test]
    fn idle_identical_replicas_share_the_traffic() {
        let pool = Pool::new(4);
        let policy = CostModelPolicy::new(&settings());
        let candidates = fleet(&pool.ids, &[(0, 0, 0.0, None); 4]);
        let mut counts = [0u64; 4];
        for _ in 0..8_000 {
            let choice = policy
                .choose(&context(&pool, &candidates, 500, &[]))
                .unwrap();
            counts[choice.index] += 1;
        }
        assert!(crate::testing::chi_square(&counts) < 16.27, "{counts:?}");
        let explained = policy.explain(&context(&pool, &candidates, 500, &[]));
        assert_eq!((explained.winner, explained.decided_by), (None, None));
    }

    #[test]
    fn a_winner_is_explained_by_its_biggest_lead() {
        let pool = Pool::new(2);
        let policy = CostModelPolicy::new(&settings());
        let cached = fleet(&pool.ids, &[(1, 2_000, 0.0, None), (1, 0, 0.0, None)]);
        let explained = policy.explain(&context(&pool, &cached, 2_100, &[]));
        assert_eq!(explained.winner, Some(0));
        assert_eq!(explained.decided_by, Some(Term::Prefill));
        assert_eq!(explained.rule, RULE);
        let queued = fleet(&pool.ids, &[(1, 0, 0.9, None), (1, 0, 0.0, None)]);
        let explained = policy.explain(&context(&pool, &queued, 2_100, &[]));
        assert_eq!(
            (explained.winner, explained.decided_by),
            (Some(1), Some(Term::Queue))
        );
    }

    #[derive(Debug, Clone)]
    enum Op {
        Arrive {
            cached: Vec<u32>,
            kv: Vec<Option<f64>>,
            prefix: u64,
            session: Option<u64>,
        },
        Depart(usize),
    }

    fn op(n: usize) -> impl Strategy<Value = Op> {
        prop_oneof![
            3 => (
                prop::collection::vec(prop_oneof![Just(0u32), 0u32..4_000], n),
                prop::collection::vec(prop::option::of(0.0f64..1.0), n),
                0u64..4,
                prop::option::of(0u64..6),
            )
                .prop_map(|(cached, kv, prefix, session)| Op::Arrive {
                    cached,
                    kv,
                    prefix,
                    session
                }),
            2 => any::<usize>().prop_map(Op::Depart),
        ]
    }

    proptest! {
        #[test]
        fn no_replica_ever_takes_more_than_its_bounded_share(
            n in 1usize..9,
            epsilon in 0.01f64..2.0,
            seed: u64,
            ops in prop::collection::vec(op(8), 1..200),
        ) {
            let pool = Pool::new(n);
            let policy = CostModelPolicy::new(&Settings {
                epsilon,
                seed,
                ..Settings::seeded(seed)
            });
            let prefixes: Vec<Vec<BlockHash>> = (0..4).map(|p| chain(p, 8)).collect();
            let mut loads = vec![0u32; n];
            for (step, op) in ops.into_iter().enumerate() {
                match op {
                    Op::Arrive { cached, kv, prefix, session } => {
                        let candidates: Vec<Candidate<'_>> = pool
                            .ids
                            .iter()
                            .enumerate()
                            .map(|(i, id)| Candidate {
                                cached_tokens: cached[i],
                                kv_usage: kv[i],
                                queued_secs: f64::from(loads[i]) * 0.05,
                                ..Candidate::new(id, loads[i])
                            })
                            .collect();
                        let ctx = RouteContext {
                            prompt_tokens: 4_100,
                            blocks: &prefixes[prefix as usize],
                            session,
                            now: step as u64 * 1_000_000,
                            ..RouteContext::new(&pool.model, &candidates)
                        };
                        let total: u64 = loads.iter().map(|&l| u64::from(l)).sum::<u64>() + 1;
                        let bound = ((1.0 + epsilon) * total as f64 / n as f64).ceil() as u64;
                        let choice = policy.choose(&ctx).unwrap();
                        loads[choice.index] += 1;
                        prop_assert!(u64::from(loads[choice.index]) <= bound,
                            "replica {} holds {} over a bound of {bound}", choice.index, loads[choice.index]);
                    }
                    Op::Depart(i) => {
                        let busy: Vec<usize> = (0..n).filter(|&r| loads[r] > 0).collect();
                        if !busy.is_empty() {
                            loads[busy[i % busy.len()]] -= 1;
                        }
                    }
                }
            }
        }

        #[test]
        fn explain_never_moves_the_next_choice(
            loads in prop::collection::vec(0u32..20, 1..8),
            cached in prop::collection::vec(0u32..2_000, 8),
            seed: u64,
        ) {
            let pool = Pool::new(loads.len());
            let candidates: Vec<Candidate<'_>> = pool
                .ids
                .iter()
                .zip(&loads)
                .zip(&cached)
                .map(|((id, &load), &cached)| Candidate {
                    cached_tokens: cached,
                    ..Candidate::new(id, load)
                })
                .collect();
            let prompt = chain(seed, 6);
            let ctx = RouteContext {
                prompt_tokens: 2_000,
                blocks: &prompt,
                session: Some(seed),
                ..RouteContext::new(&pool.model, &candidates)
            };
            let make = || CostModelPolicy::new(&Settings {
                seed,
                ..Settings::seeded(seed)
            });
            let (plain, explained) = (make(), make());
            for _ in 0..3 {
                let _ = explained.explain(&ctx);
                prop_assert_eq!(plain.choose(&ctx), explained.choose(&ctx));
            }
        }
    }
}
