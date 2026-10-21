use inferrail_core::config::PolicyKind;

use crate::{RouteContext, RoutingPolicy, SplitMix};

/// Fewest in flight requests; ties are broken uniformly at random.
#[derive(Debug)]
pub struct LeastRequest {
    rng: SplitMix,
}

impl LeastRequest {
    pub fn new(seed: u64) -> Self {
        Self {
            rng: SplitMix::new(seed),
        }
    }
}

impl RoutingPolicy for LeastRequest {
    fn kind(&self) -> PolicyKind {
        PolicyKind::LeastRequest
    }

    fn choose(&self, ctx: &RouteContext<'_>) -> Option<usize> {
        let (first, rest) = ctx.candidates.split_first()?;
        let (mut best, mut min, mut ties) = (0, first.in_flight, 1);
        for (i, candidate) in rest.iter().enumerate() {
            let load = candidate.in_flight;
            if load < min {
                (best, min, ties) = (i + 1, load, 1);
            } else if load == min {
                ties += 1;
                if self.rng.below(ties) == 0 {
                    best = i + 1;
                }
            }
        }
        Some(best)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::{Pool, chi_square};

    #[test]
    fn picks_the_least_loaded() {
        let pool = Pool::new(5);
        let candidates = pool.candidates(&[4, 9, 1, 7, 3]);
        let policy = LeastRequest::new(0);
        for _ in 0..100 {
            assert_eq!(pool.pick(&policy, &candidates), 2);
        }
    }

    #[test]
    fn spreads_ties_uniformly() {
        let pool = Pool::new(6);
        let candidates = pool.candidates(&[2, 5, 2, 2, 9, 2]);
        let policy = LeastRequest::new(3);
        let mut counts = [0u64; 6];
        for _ in 0..40_000 {
            counts[pool.pick(&policy, &candidates)] += 1;
        }
        assert_eq!((counts[1], counts[4]), (0, 0));
        let tied = [counts[0], counts[2], counts[3], counts[5]];
        let stat = chi_square(&tied);
        assert!(stat < 16.27, "chi square {stat} for {tied:?}");
    }

    #[test]
    fn same_seed_same_tie_breaks() {
        let pool = Pool::new(4);
        let candidates = pool.candidates(&[]);
        let run = |seed| -> Vec<usize> {
            let policy = LeastRequest::new(seed);
            (0..100).map(|_| pool.pick(&policy, &candidates)).collect()
        };
        assert_eq!(run(5), run(5));
    }
}
