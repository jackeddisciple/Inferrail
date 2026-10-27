use std::cmp::Reverse;

use inferrail_core::config::PolicyKind;

use crate::cost::{CostModel, Estimate, Hits};
use crate::guards::{load_cap, under_pressure, within_load};
use crate::hashing::{prefix_key, tag, weight};
use crate::sessions::Sessions;
use crate::sketch::Sketch;
use crate::{
    Choice, Exclusion, Explanation, Guarded, HotPrefix, RouteContext, RoutingPolicy, Scored,
    Settings, SplitMix, Steer, pick_among,
};

const RULE: &str = "the lowest estimated time to first token within the load guards";

#[derive(Debug)]
struct Hot {
    blocks: u32,
    utilization: f64,
    sketch: Sketch,
}

/// Inferrail's policy: the replica with the lowest estimated time to first
/// token, `W + P + X`, among those the load guards allow. Bounded load
/// skips replicas past their share of requests in flight, KV pressure keeps
/// long prompts off replicas whose cache is nearly full, a hot root prefix
/// is spread over just enough owners, and with session affinity a
/// conversation stays on its last replica while that one is allowed.
#[derive(Debug)]
pub struct CostModelPolicy {
    cost: CostModel,
    epsilon: f64,
    kappa: f64,
    short_prompt_tokens: u32,
    hot: Option<Hot>,
    sessions: Option<Sessions>,
    rng: SplitMix,
}

/// The candidates as the estimates and the guards see them.
struct Plan {
    estimates: Vec<Estimate>,
    excluded: Vec<Option<Exclusion>>,
    guarded: Guarded,
    hot: Option<HotPrefix>,
    owner: Vec<bool>,
    /// Every estimate before hot prefix credit, once any was credited.
    uncredited: Option<Vec<Estimate>>,
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
            hot: settings.hot_prefix.as_ref().map(|hot| Hot {
                blocks: hot.blocks,
                utilization: hot.utilization,
                sketch: Sketch::new(hot.window),
            }),
            sessions: settings.session_affinity.then(Sessions::new),
            rng: SplitMix::new(settings.seed),
        }
    }

    /// Estimates every candidate and applies the guards. The request's own
    /// prefill work counts toward its root prefix's demand; with `observe`
    /// it is recorded for the requests that follow.
    fn plan(&self, ctx: &RouteContext<'_>, observe: bool) -> Plan {
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
        let mut plan = Plan {
            estimates,
            excluded,
            guarded,
            hot: None,
            owner: vec![false; n],
            uncredited: None,
        };
        self.spread_hot_prefix(ctx, &mut plan, observe);
        plan
    }

    /// A root prefix that needs more than `utilization` of one replica's
    /// prefill gets `ceil(demand / utilization)` owners: the candidates
    /// holding most of it, then those its key hashes to. An owner is priced
    /// as if it held the longest prefix the owners hold that is hot in its
    /// own right, its demand tracked like the root's, so the cost model
    /// shares that prefix among them by their queues while a conversation's
    /// history, which only its own turns bring, counts only where it is;
    /// the first request an owner takes warms it.
    fn spread_hot_prefix(&self, ctx: &RouteContext<'_>, plan: &mut Plan, observe: bool) {
        let Some(hot) = &self.hot else { return };
        let Some(root) = prefix_key(ctx.blocks, hot.blocks) else {
            return;
        };
        let candidates = ctx.candidates;
        let n = candidates.len();
        let work = micros(
            plan.estimates
                .iter()
                .map(|e| e.prefill_secs + e.tier_secs)
                .fold(f64::INFINITY, f64::min),
        );
        let demand = hot.sketch.rate(root, work, ctx.now) / 1e6;
        let owners = ((demand / hot.utilization).ceil() as usize).clamp(1, n.max(1));
        plan.hot = Some(HotPrefix { demand, owners });
        let mut order: Vec<usize> = Vec::new();
        if owners > 1 {
            order.extend(0..n);
            order.sort_by_key(|&i| {
                let c = &candidates[i];
                Reverse((c.cached_tokens, weight(root, c.id)))
            });
        }
        let root_tokens = (ctx.blocks.len().min(hot.blocks as usize) as u32)
            .saturating_mul(ctx.block_size)
            .max(1);
        let holdings: Vec<u32> = match order.get(..owners) {
            Some(owned) => owned.iter().map(|&i| candidates[i].cached_tokens).collect(),
            None => candidates
                .iter()
                .map(|c| c.cached_tokens)
                .max()
                .into_iter()
                .collect(),
        };
        let holdings = holdings.into_iter().filter(|&held| held >= root_tokens);
        let held: Vec<(u32, u64)> = [holdings.clone().max(), holdings.min()]
            .into_iter()
            .flatten()
            .filter(|_| ctx.block_size > 0)
            .filter_map(|tokens| {
                prefix_key(ctx.blocks, tokens / ctx.block_size).map(|key| (tokens, key))
            })
            .collect();
        let shared = held
            .iter()
            .find(|&&(_, key)| {
                let rate = if key == root {
                    demand
                } else {
                    hot.sketch.rate(key, work, ctx.now) / 1e6
                };
                rate > hot.utilization
            })
            .map_or(0, |&(tokens, _)| tokens);
        if observe {
            hot.sketch.add(root, work, ctx.now);
            for (i, &(_, key)) in held.iter().enumerate() {
                if key != root && held[..i].iter().all(|&(_, seen)| seen != key) {
                    hot.sketch.add(key, work, ctx.now);
                }
            }
        }
        if owners < 2 {
            return;
        }
        for &i in &order[..owners] {
            plan.owner[i] = true;
            if candidates[i].cached_tokens < shared {
                plan.uncredited
                    .get_or_insert_with(|| plan.estimates.clone());
                plan.estimates[i] =
                    self.cost
                        .estimate(ctx.prompt_tokens, &candidates[i], Hits::gpu(shared));
            }
        }
    }

    /// The conversation's last replica, unless a load guard rules it out.
    fn sticky(&self, ctx: &RouteContext<'_>, plan: &Plan) -> Option<usize> {
        let replica = self.sessions.as_ref()?.get(ctx.session?)?;
        (0..ctx.candidates.len()).find(|&i| tag(ctx.candidates[i].id) == replica && plan.allowed(i))
    }

    /// Whether candidate `i` is among the cheapest allowed by `estimates`:
    /// lowest estimate, then fewest requests in flight.
    fn cheapest(
        ctx: &RouteContext<'_>,
        plan: &Plan,
        estimates: &[Estimate],
    ) -> impl Fn(usize) -> bool + use<> {
        let n = ctx.candidates.len();
        let ttft = |i: usize| estimates[i].ttft_secs;
        let low = (0..n)
            .filter(|&i| plan.allowed(i))
            .map(ttft)
            .fold(f64::INFINITY, f64::min);
        let fewest = (0..n)
            .filter(|&i| plan.allowed(i) && ttft(i) == low)
            .map(|i| ctx.candidates[i].in_flight)
            .min()
            .unwrap_or(0);
        let best: Vec<bool> = (0..n)
            .map(|i| plan.allowed(i) && ttft(i) == low && ctx.candidates[i].in_flight == fewest)
            .collect();
        move |i| best[i]
    }

    /// Whether winner `w` won only by the credit of a hot prefix it does
    /// not hold yet: credited, and not among the cheapest without it.
    fn lifted(ctx: &RouteContext<'_>, plan: &Plan, w: usize) -> bool {
        plan.uncredited.as_ref().is_some_and(|before| {
            before[w] != plan.estimates[w] && !Self::cheapest(ctx, plan, before)(w)
        })
    }
}

fn micros(secs: f64) -> u64 {
    if secs.is_finite() && secs > 0.0 {
        (secs * 1e6).round() as u64
    } else {
        0
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
        let plan = self.plan(ctx, true);
        let (index, steer) = match self.sticky(ctx, &plan) {
            Some(i) => (i, Some(Steer::Session)),
            None => {
                let best = Self::cheapest(ctx, &plan, &plan.estimates);
                let i = pick_among(&self.rng, ctx.candidates.len(), best)?;
                (i, Self::lifted(ctx, &plan, i).then_some(Steer::HotPrefix))
            }
        };
        if let (Some(sessions), Some(session)) = (&self.sessions, ctx.session) {
            sessions.set(session, tag(ctx.candidates[index].id));
        }
        Some(Choice {
            index,
            steer,
            guarded: plan.guarded,
            estimate: Some(plan.estimates[index]),
        })
    }

    fn explain(&self, ctx: &RouteContext<'_>) -> Explanation {
        let n = ctx.candidates.len();
        if n == 0 {
            return Explanation::plain(RULE, None, 0);
        }
        let plan = self.plan(ctx, false);
        let candidates = (0..n)
            .map(|i| Scored {
                excluded: plan.excluded[i],
                score: None,
                estimate: Some(plan.estimates[i]),
                owner: plan.owner[i],
            })
            .collect();
        let (winner, steer, decided_by) = match self.sticky(ctx, &plan) {
            Some(i) => (Some(i), Some(Steer::Session), None),
            None => {
                let best = Self::cheapest(ctx, &plan, &plan.estimates);
                let tied: Vec<usize> = (0..n).filter(|&i| best(i)).collect();
                let winner = (tied.len() == 1).then(|| tied[0]);
                let decided_by = winner.and_then(|w| {
                    let runner_up = (0..n)
                        .filter(|&i| i != w && plan.allowed(i))
                        .min_by(|&a, &b| plan.ttft(a).total_cmp(&plan.ttft(b)))?;
                    (plan.ttft(runner_up) > plan.ttft(w))
                        .then(|| plan.estimates[w].ahead_of(&plan.estimates[runner_up]))
                });
                let steer = winner
                    .filter(|&w| Self::lifted(ctx, &plan, w))
                    .map(|_| Steer::HotPrefix);
                (winner, steer, decided_by)
            }
        };
        Explanation {
            rule: RULE,
            winner,
            decided_by,
            steer,
            guarded: plan.guarded,
            hot_prefix: plan.hot,
            candidates,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;
    use crate::testing::{Pool, chain};
    use crate::{Candidate, Term};
    use inferrail_core::config::HotPrefixConfig;
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
            block_size: 16,
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
    fn sessions_stick_until_their_replica_is_overloaded() {
        let pool = Pool::new(3);
        let policy = CostModelPolicy::new(&Settings {
            session_affinity: true,
            ..settings()
        });
        let session = |candidates: &[Candidate<'_>], key: u64| {
            policy
                .choose(&RouteContext {
                    session: Some(key),
                    prompt_tokens: 300,
                    ..RouteContext::new(&pool.model, candidates)
                })
                .unwrap()
        };
        let calm = fleet(
            &pool.ids,
            &[(1, 0, 0.0, None), (1, 0, 0.3, None), (1, 0, 0.0, None)],
        );
        let first = session(&calm, 77);
        assert_eq!(first.steer, None);
        let mut cheaper = calm.clone();
        cheaper[first.index].queued_secs = 0.2;
        let again = session(&cheaper, 77);
        assert_eq!(
            (again.index, again.steer),
            (first.index, Some(Steer::Session))
        );
        let mut hot = calm.clone();
        hot[first.index].in_flight = 9;
        let moved = session(&hot, 77);
        assert_ne!(moved.index, first.index);
        assert_eq!(moved.steer, None);
        let back = session(&calm, 77);
        assert_eq!(
            (back.index, back.steer),
            (moved.index, Some(Steer::Session))
        );
        let explained = policy.explain(&RouteContext {
            session: Some(77),
            prompt_tokens: 300,
            ..RouteContext::new(&pool.model, &calm)
        });
        assert_eq!(explained.steer, Some(Steer::Session));
        assert_eq!(explained.winner, Some(moved.index));
    }

    fn hot_settings(window: Duration) -> Settings {
        Settings {
            hot_prefix: Some(HotPrefixConfig {
                enabled: true,
                blocks: 4,
                utilization: 0.5,
                window,
            }),
            ..Settings::seeded(1)
        }
    }

    /// Routes `count` requests for one prompt every `interval` ns through a
    /// fleet whose queues drain in real time and whose replicas hold the
    /// prompt's prefix once they served it; returns each choice.
    fn serve_one_prefix(
        policy: &CostModelPolicy,
        pool: &Pool,
        count: u64,
        interval: u64,
        prompt: &[BlockHash],
    ) -> Vec<Choice> {
        let cost = CostModel::default();
        let n = pool.ids.len();
        let (tokens, held) = (prompt.len() as u32 * 16 + 6, prompt.len() as u32 * 16);
        let mut busy_until = vec![0u64; n];
        let mut cached = vec![0u32; n];
        (0..count)
            .map(|i| {
                let now = i * interval;
                let candidates: Vec<Candidate<'_>> = pool
                    .ids
                    .iter()
                    .enumerate()
                    .map(|(r, id)| Candidate {
                        cached_tokens: cached[r],
                        queued_secs: busy_until[r].saturating_sub(now) as f64 / 1e9,
                        ..Candidate::new(id, 0)
                    })
                    .collect();
                let ctx = RouteContext {
                    now,
                    ..context(pool, &candidates, tokens, prompt)
                };
                let choice = policy.choose(&ctx).unwrap();
                let r = choice.index;
                let work = cost.prefill_secs(tokens - cached[r], cached[r]);
                busy_until[r] = busy_until[r].max(now) + (work * 1e9) as u64;
                cached[r] = held;
                choice
            })
            .collect()
    }

    #[test]
    fn a_hot_prefix_spreads_over_just_enough_owners_without_losing_hits() {
        let pool = Pool::new(6);
        let prompt = chain(1, 64);
        let hot = CostModelPolicy::new(&hot_settings(Duration::from_secs(2)));
        let choices = serve_one_prefix(&hot, &pool, 2_000, 10_000_000, &prompt);
        let late = &choices[1_000..];
        let mut share = [0u64; 6];
        for choice in late {
            share[choice.index] += 1;
        }
        let used: Vec<usize> = (0..6).filter(|&r| share[r] > 0).collect();
        let work = CostModel::default().prefill_secs(6, 1_024);
        let owners = (work * 100.0 / 0.5).ceil() as usize;
        assert_eq!(owners, 2);
        assert_eq!(used.len(), owners, "{share:?}");
        assert!(used.iter().all(|&r| share[r] > 300), "{share:?}");
        let first_visits: Vec<usize> = (0..choices.len())
            .filter(|&i| choices[..i].iter().all(|p| p.index != choices[i].index))
            .collect();
        assert_eq!(
            first_visits.len(),
            owners,
            "only a new owner computes the prefix again"
        );
        let steered: Vec<usize> = (0..choices.len())
            .filter(|&i| choices[i].steer == Some(Steer::HotPrefix))
            .collect();
        assert_eq!(
            steered,
            first_visits[1..],
            "only the owner taking the prefix on is steered there"
        );

        let plain = CostModelPolicy::new(&settings());
        let choices = serve_one_prefix(&plain, &pool, 2_000, 10_000_000, &prompt);
        let spot = choices[0].index;
        assert!(
            choices.iter().all(|c| c.index == spot),
            "one replica takes it all"
        );
    }

    /// Routes `count` first turns of new conversations that open with
    /// `system`, over a fleet whose replicas hold and have queued `held`,
    /// one a nanosecond from `start`; returns the last choice and how
    /// explain priced it.
    fn first_turns(
        policy: &CostModelPolicy,
        pool: &Pool,
        system: &[BlockHash],
        held: &[(u32, f64)],
        count: u64,
        start: u64,
    ) -> (Choice, Vec<Estimate>) {
        let rows: Vec<(u32, u32, f64, Option<f64>)> = held
            .iter()
            .map(|&(tokens, queued)| (1, tokens, queued, None))
            .collect();
        let candidates = fleet(&pool.ids, &rows);
        let mut last = None;
        for i in 0..count {
            let prompt: Vec<BlockHash> = system.iter().copied().chain(chain(100 + i, 2)).collect();
            let ctx = RouteContext {
                now: start + i,
                ..context(pool, &candidates, 1_060, &prompt)
            };
            let explained = policy.explain(&ctx);
            let choice = policy.choose(&ctx).unwrap();
            let estimates = explained
                .candidates
                .iter()
                .map(|s| s.estimate.unwrap())
                .collect();
            last = Some((choice, estimates));
        }
        last.unwrap()
    }

    #[test]
    fn a_conversation_keeps_its_history_when_its_system_prompt_turns_hot() {
        let pool = Pool::new(3);
        let cost = CostModel::default();
        let system = chain(8, 64);
        let turn: Vec<BlockHash> = system.iter().copied().chain(chain(9, 134)).collect();
        let conversation = |policy: &CostModelPolicy, held: [u32; 3], now: u64| {
            let candidates = fleet(
                &pool.ids,
                &[
                    (1, held[0], 0.05, None),
                    (1, held[1], 0.0, None),
                    (1, held[2], 0.0, None),
                ],
            );
            let ctx = RouteContext {
                now,
                ..context(&pool, &candidates, 3_172, &turn)
            };
            let explained = policy.explain(&ctx);
            let estimates: Vec<Estimate> = explained
                .candidates
                .iter()
                .map(|s| s.estimate.unwrap())
                .collect();
            let truth: Vec<Estimate> = candidates
                .iter()
                .map(|c| cost.estimate(3_172, c, Hits::gpu(c.cached_tokens)))
                .collect();
            (policy.choose(&ctx).unwrap(), estimates, truth)
        };

        let shared = CostModelPolicy::new(&hot_settings(Duration::from_millis(100)));
        first_turns(
            &shared,
            &pool,
            &system,
            &[(1_024, 0.0), (1_024, 0.0), (0, 0.0)],
            30,
            0,
        );
        let (choice, estimates, truth) = conversation(&shared, [3_072, 1_024, 0], 30);
        assert_eq!(choice.index, 0);
        assert_eq!(estimates[..2], truth[..2]);
        let system_only = fleet(&pool.ids, &[(1, 0, 0.0, None); 3]);
        let warm = cost.estimate(3_172, &system_only[2], Hits::gpu(1_024));
        assert_eq!(
            estimates[2], warm,
            "a cold owner is credited with the system prompt"
        );

        let single = CostModelPolicy::new(&hot_settings(Duration::from_millis(100)));
        let (spread, estimates) = first_turns(
            &single,
            &pool,
            &system,
            &[(1_024, 0.3), (0, 0.0), (0, 0.01)],
            30,
            0,
        );
        assert_eq!(spread.index, 1, "first turns spread to a cold owner");
        assert_eq!(
            estimates[1],
            cost.estimate(1_060, &system_only[1], Hits::gpu(1_024))
        );
        assert_eq!(
            spread.steer, None,
            "it was the cheapest without the credit too"
        );
        let (choice, estimates, truth) = conversation(&single, [3_072, 0, 0], 30);
        assert_eq!(choice.index, 0, "{estimates:?}");
        assert_eq!(estimates, truth, "no owner is credited with the history");
        let (lifted, _) = first_turns(
            &single,
            &pool,
            &system,
            &[(1_024, 0.05), (0, 0.0), (0, 0.01)],
            1,
            31,
        );
        assert_eq!((lifted.index, lifted.steer), (1, Some(Steer::HotPrefix)));
    }

    #[test]
    fn a_cold_prefix_keeps_a_single_owner() {
        let pool = Pool::new(4);
        let prompt = chain(5, 64);
        let policy = CostModelPolicy::new(&hot_settings(Duration::from_secs(2)));
        let choices = serve_one_prefix(&policy, &pool, 300, 100_000_000, &prompt);
        assert!(choices.iter().all(|c| c.index == choices[0].index));
        assert!(choices.iter().all(|c| c.steer.is_none()));
        let candidates = fleet(&pool.ids, &[(0, 1_024, 0.0, None); 4]);
        let explained = policy.explain(&RouteContext {
            now: 300 * 100_000_000,
            ..context(&pool, &candidates, 1_030, &prompt)
        });
        let hot = explained.hot_prefix.unwrap();
        assert_eq!(hot.owners, 1, "{hot:?}");
        assert!(explained.candidates.iter().all(|s| !s.owner));
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
                session_affinity: true,
                ..hot_settings(Duration::from_millis(100))
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
                session_affinity: true,
                ..Settings::seeded(seed)
            });
            let (plain, explained) = (make(), make());
            for _ in 0..3 {
                let _ = explained.explain(&ctx);
                prop_assert_eq!(plain.choose(&ctx), explained.choose(&ctx));
            }
        }

        #[test]
        fn explain_names_the_replica_choose_picks(
            seed: u64,
            steps in prop::collection::vec(
                (
                    prop::collection::vec(
                        (0u32..6, 0u32..1_100, 0.0f64..0.3, prop::option::of(0.8f64..1.0)),
                        1..6,
                    ),
                    0u64..3,
                    prop::option::of(0u64..4),
                    0u64..50_000_000,
                ),
                1..40,
            ),
        ) {
            let pool = Pool::new(6);
            let policy = CostModelPolicy::new(&Settings {
                seed,
                session_affinity: true,
                ..hot_settings(Duration::from_millis(100))
            });
            let prefixes: Vec<Vec<BlockHash>> = (0..3).map(|p| chain(p, 64)).collect();
            let mut now = 0;
            for (rows, prefix, session, gap) in steps {
                now += gap;
                let candidates = fleet(&pool.ids, &rows);
                let ctx = RouteContext {
                    session,
                    now,
                    ..context(&pool, &candidates, 1_100, &prefixes[prefix as usize])
                };
                let explained = policy.explain(&ctx);
                let chosen = policy.choose(&ctx).unwrap();
                if let Some(winner) = explained.winner {
                    prop_assert_eq!(winner, chosen.index, "{:?}", explained);
                    prop_assert_eq!(explained.steer, chosen.steer);
                }
                prop_assert_eq!(explained.guarded, chosen.guarded);
            }
        }
    }
}
