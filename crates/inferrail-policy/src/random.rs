use inferrail_core::config::PolicyKind;

use crate::{Choice, Explanation, RouteContext, RoutingPolicy, SplitMix};

#[derive(Debug)]
pub struct Random {
    rng: SplitMix,
}

impl Random {
    pub fn new(seed: u64) -> Self {
        Self {
            rng: SplitMix::new(seed),
        }
    }
}

impl RoutingPolicy for Random {
    fn kind(&self) -> PolicyKind {
        PolicyKind::Random
    }

    fn choose(&self, ctx: &RouteContext<'_>) -> Option<Choice> {
        let len = ctx.candidates.len();
        (len > 0).then(|| Choice::new(self.rng.below(len)))
    }

    fn explain(&self, ctx: &RouteContext<'_>) -> Explanation {
        let len = ctx.candidates.len();
        Explanation::plain("uniform at random", (len == 1).then_some(0), len)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::{Pool, chi_square};

    #[test]
    fn uniform_under_a_fixed_seed() {
        let pool = Pool::new(6);
        let candidates = pool.candidates(&[50, 0, 0, 0, 0, 9]);
        let policy = Random::new(42);
        let mut counts = [0u64; 6];
        for _ in 0..60_000 {
            counts[pool.pick(&policy, &candidates)] += 1;
        }
        let stat = chi_square(&counts);
        assert!(stat < 20.52, "chi square {stat} for {counts:?}");
    }

    #[test]
    fn same_seed_same_choices() {
        let pool = Pool::new(5);
        let candidates = pool.candidates(&[]);
        let run = |seed| -> Vec<usize> {
            let policy = Random::new(seed);
            (0..200).map(|_| pool.pick(&policy, &candidates)).collect()
        };
        assert_eq!(run(9), run(9));
        assert_ne!(run(9), run(10));
    }

    #[test]
    fn single_candidate_and_empty_set() {
        let pool = Pool::new(1);
        let policy = Random::new(0);
        assert_eq!(pool.pick(&policy, &pool.candidates(&[])), 0);
        let ctx = RouteContext::new(&pool.model, &[]);
        assert_eq!(policy.choose(&ctx), None);
        assert_eq!(policy.kind(), PolicyKind::Random);
    }
}
