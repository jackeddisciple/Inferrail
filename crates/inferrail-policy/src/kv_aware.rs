use inferrail_core::config::{KvAwareConfig, PolicyKind};

use crate::{
    Candidate, Choice, Explanation, RouteContext, RoutingPolicy, Scored, SplitMix, pick_among,
};

/// The weighted score several open routers use: `cache_weight` times the
/// share of the prompt a replica holds, minus `load_weight` times its
/// requests in flight over the busiest candidate's. The best score wins;
/// ties go to fewer requests in flight, then to chance.
#[derive(Debug)]
pub struct KvAware {
    cache_weight: f64,
    load_weight: f64,
    rng: SplitMix,
}

impl KvAware {
    pub fn new(weights: &KvAwareConfig, seed: u64) -> Self {
        Self {
            cache_weight: weights.cache_weight,
            load_weight: weights.load_weight,
            rng: SplitMix::new(seed),
        }
    }

    fn scores(&self, ctx: &RouteContext<'_>) -> Vec<f64> {
        let busiest = ctx
            .candidates
            .iter()
            .map(|c| c.in_flight)
            .max()
            .unwrap_or(0);
        ctx.candidates
            .iter()
            .map(|c| self.score(c, ctx.prompt_tokens, busiest))
            .collect()
    }

    fn score(&self, candidate: &Candidate<'_>, prompt: u32, busiest: u32) -> f64 {
        let hit = if prompt == 0 {
            0.0
        } else {
            f64::from(candidate.cached_tokens.min(prompt)) / f64::from(prompt)
        };
        let load = if busiest == 0 {
            0.0
        } else {
            f64::from(candidate.in_flight) / f64::from(busiest)
        };
        self.cache_weight * hit - self.load_weight * load
    }

    /// Whether candidate `i` is among the best: top score, then fewest
    /// requests in flight.
    fn best(ctx: &RouteContext<'_>, scores: &[f64]) -> impl Fn(usize) -> bool {
        let top = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let fewest = ctx
            .candidates
            .iter()
            .zip(scores)
            .filter(|(_, s)| **s == top)
            .map(|(c, _)| c.in_flight)
            .min()
            .unwrap_or(0);
        let tops: Vec<bool> = ctx
            .candidates
            .iter()
            .zip(scores)
            .map(|(c, s)| *s == top && c.in_flight == fewest)
            .collect();
        move |i| tops[i]
    }
}

impl RoutingPolicy for KvAware {
    fn kind(&self) -> PolicyKind {
        PolicyKind::KvAware
    }

    fn choose(&self, ctx: &RouteContext<'_>) -> Option<Choice> {
        let scores = self.scores(ctx);
        pick_among(&self.rng, scores.len(), Self::best(ctx, &scores)).map(Choice::new)
    }

    fn explain(&self, ctx: &RouteContext<'_>) -> Explanation {
        let scores = self.scores(ctx);
        let best = Self::best(ctx, &scores);
        let tied: Vec<usize> = (0..scores.len()).filter(|&i| best(i)).collect();
        Explanation {
            candidates: scores
                .iter()
                .map(|&s| Scored {
                    score: Some(s),
                    ..Scored::default()
                })
                .collect(),
            ..Explanation::plain(
                "cache_weight times the share cached minus load_weight times the relative load",
                (tied.len() == 1).then(|| tied[0]),
                scores.len(),
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::{Pool, chi_square};

    fn weights(cache: f64, load: f64) -> KvAwareConfig {
        KvAwareConfig {
            cache_weight: cache,
            load_weight: load,
        }
    }

    fn choose(policy: &KvAware, pool: &Pool, fleet: &[(u32, u32)], prompt: u32) -> usize {
        let candidates: Vec<Candidate<'_>> = pool
            .ids
            .iter()
            .zip(fleet)
            .map(|(id, &(in_flight, cached))| Candidate {
                cached_tokens: cached,
                ..Candidate::new(id, in_flight)
            })
            .collect();
        let ctx = RouteContext {
            prompt_tokens: prompt,
            ..RouteContext::new(&pool.model, &candidates)
        };
        policy.choose(&ctx).unwrap().index
    }

    #[test]
    fn cache_wins_until_the_load_gap_outweighs_it() {
        let pool = Pool::new(3);
        let policy = KvAware::new(&weights(1.0, 1.0), 1);
        assert_eq!(
            choose(&policy, &pool, &[(4, 900), (4, 0), (3, 0)], 1_000),
            0
        );
        assert_eq!(
            choose(&policy, &pool, &[(8, 900), (4, 0), (0, 0)], 1_000),
            2
        );
        assert_eq!(
            choose(&policy, &pool, &[(4, 500), (4, 0), (3, 0)], 1_000),
            0
        );
        let loyal = KvAware::new(&weights(3.0, 1.0), 1);
        assert_eq!(choose(&loyal, &pool, &[(8, 900), (4, 0), (0, 0)], 1_000), 0);
        let balancer = KvAware::new(&weights(0.0, 1.0), 1);
        assert_eq!(
            choose(&balancer, &pool, &[(8, 999), (4, 0), (1, 0)], 1_000),
            2
        );
    }

    #[test]
    fn without_a_prompt_it_balances_like_least_request() {
        let pool = Pool::new(4);
        let policy = KvAware::new(&weights(1.0, 1.0), 2);
        let mut counts = [0u64; 4];
        for _ in 0..20_000 {
            counts[choose(&policy, &pool, &[(3, 0), (1, 0), (1, 0), (5, 0)], 0)] += 1;
        }
        assert_eq!((counts[0], counts[3]), (0, 0));
        let tied = [counts[1], counts[2]];
        assert!(chi_square(&tied) < 10.83, "{tied:?}");
    }

    #[test]
    fn explain_matches_choose_when_the_best_is_unique() {
        let pool = Pool::new(3);
        let policy = KvAware::new(&weights(1.0, 1.0), 1);
        let ids = &pool.ids;
        let candidates = [
            Candidate {
                cached_tokens: 800,
                ..Candidate::new(&ids[0], 2)
            },
            Candidate::new(&ids[1], 2),
            Candidate::new(&ids[2], 1),
        ];
        let ctx = RouteContext {
            prompt_tokens: 1_000,
            ..RouteContext::new(&pool.model, &candidates)
        };
        let explained = policy.explain(&ctx);
        assert_eq!(explained.winner, Some(0));
        assert_eq!(policy.choose(&ctx).unwrap().index, 0);
        let scores: Vec<f64> = explained
            .candidates
            .iter()
            .map(|s| s.score.unwrap())
            .collect();
        assert_eq!(scores, [0.8 - 1.0, -1.0, -0.5]);
        let even = [Candidate::new(&ids[0], 1), Candidate::new(&ids[1], 1)];
        assert_eq!(
            policy
                .explain(&RouteContext::new(&pool.model, &even))
                .winner,
            None
        );
    }
}
