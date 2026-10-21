use inferrail_core::config::PolicyKind;

use crate::{RouteContext, RoutingPolicy, SplitMix};

/// Power of two choices: sample two distinct candidates, keep the less loaded.
#[derive(Debug)]
pub struct P2c {
    rng: SplitMix,
}

impl P2c {
    pub fn new(seed: u64) -> Self {
        Self {
            rng: SplitMix::new(seed),
        }
    }
}

impl RoutingPolicy for P2c {
    fn kind(&self) -> PolicyKind {
        PolicyKind::P2c
    }

    fn choose(&self, ctx: &RouteContext<'_>) -> Option<usize> {
        let candidates = ctx.candidates;
        match candidates.len() {
            0 => None,
            1 => Some(0),
            len => {
                let a = self.rng.below(len);
                let mut b = self.rng.below(len - 1);
                if b >= a {
                    b += 1;
                }
                Some(if candidates[b].in_flight < candidates[a].in_flight {
                    b
                } else {
                    a
                })
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::{Pool, chi_square};

    #[test]
    fn idle_replica_wins_whenever_it_is_sampled() {
        let pool = Pool::new(4);
        let candidates = pool.candidates(&[0, 10, 10, 10]);
        let policy = P2c::new(11);
        let draws = 40_000;
        let wins = (0..draws)
            .filter(|_| pool.pick(&policy, &candidates) == 0)
            .count();
        let share = wins as f64 / draws as f64;
        assert!((share - 0.5).abs() < 0.015, "share {share}");
    }

    #[test]
    fn equal_loads_spread_uniformly() {
        let pool = Pool::new(5);
        let candidates = pool.candidates(&[]);
        let policy = P2c::new(2);
        let mut counts = [0u64; 5];
        for _ in 0..50_000 {
            counts[pool.pick(&policy, &candidates)] += 1;
        }
        let stat = chi_square(&counts);
        assert!(stat < 18.47, "chi square {stat} for {counts:?}");
    }

    #[test]
    fn single_candidate_needs_no_draw() {
        let pool = Pool::new(1);
        assert_eq!(pool.pick(&P2c::new(0), &pool.candidates(&[3])), 0);
    }
}
