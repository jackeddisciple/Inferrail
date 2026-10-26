use inferrail_core::config::PolicyKind;

use crate::hashing::{prefix_key, weight};
use crate::{Choice, Explanation, P2c, RouteContext, RoutingPolicy, Scored, Steer};

/// Naive cache affinity: the prompt's first `blocks` blocks pick a replica
/// by rendezvous hashing, whatever its load. Prompts without a full block
/// go to the less loaded of two random replicas.
#[derive(Debug)]
pub struct PrefixHash {
    blocks: u32,
    fallback: P2c,
}

impl PrefixHash {
    pub fn new(blocks: u32, seed: u64) -> Self {
        Self {
            blocks,
            fallback: P2c::new(seed),
        }
    }

    fn owner(ctx: &RouteContext<'_>, key: u64) -> Option<usize> {
        (0..ctx.candidates.len()).max_by_key(|&i| weight(key, ctx.candidates[i].id))
    }
}

impl RoutingPolicy for PrefixHash {
    fn kind(&self) -> PolicyKind {
        PolicyKind::PrefixHash
    }

    fn choose(&self, ctx: &RouteContext<'_>) -> Option<Choice> {
        match prefix_key(ctx.blocks, self.blocks) {
            Some(key) => Self::owner(ctx, key).map(Choice::new),
            None => self.fallback.choose(ctx).map(|choice| Choice {
                steer: Some(Steer::NoPrefix),
                ..choice
            }),
        }
    }

    fn explain(&self, ctx: &RouteContext<'_>) -> Explanation {
        let n = ctx.candidates.len();
        let Some(key) = prefix_key(ctx.blocks, self.blocks) else {
            return Explanation {
                steer: Some(Steer::NoPrefix),
                ..Explanation::plain("no full block: the less loaded of two at random", None, n)
            };
        };
        Explanation {
            candidates: ctx
                .candidates
                .iter()
                .map(|c| Scored {
                    score: Some(weight(key, c.id) as f64 / u64::MAX as f64),
                    ..Scored::default()
                })
                .collect(),
            ..Explanation::plain(
                "the replica the leading blocks hash to",
                Self::owner(ctx, key),
                n,
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::{Pool, chain, chi_square};
    use inferrail_core::ids::BlockHash;

    fn pick(policy: &PrefixHash, pool: &Pool, loads: &[u32], blocks: &[BlockHash]) -> Choice {
        let candidates = pool.candidates(loads);
        let ctx = RouteContext {
            blocks,
            ..RouteContext::new(&pool.model, &candidates)
        };
        policy.choose(&ctx).unwrap()
    }

    #[test]
    fn a_prefix_always_lands_on_the_same_replica_whatever_the_load() {
        let pool = Pool::new(5);
        let policy = PrefixHash::new(4, 1);
        let prompt = chain(9, 40);
        let first = pick(&policy, &pool, &[0; 5], &prompt).index;
        let mut loads = [0; 5];
        loads[first] = 1_000;
        for tail in 4..40 {
            assert_eq!(pick(&policy, &pool, &loads, &prompt[..tail]).index, first);
        }
        assert_eq!(pick(&policy, &pool, &loads, &prompt).steer, None);
    }

    #[test]
    fn prefixes_spread_and_only_the_owner_moves_when_it_leaves() {
        let pool = Pool::new(6);
        let policy = PrefixHash::new(2, 1);
        let prompts: Vec<_> = (0..6_000).map(|seed| chain(seed, 3)).collect();
        let owners: Vec<usize> = prompts
            .iter()
            .map(|p| pick(&policy, &pool, &[0; 6], p).index)
            .collect();
        let mut counts = [0u64; 6];
        for &owner in &owners {
            counts[owner] += 1;
        }
        assert!(chi_square(&counts) < 20.52, "{counts:?}");
        let survivors = Pool {
            model: pool.model.clone(),
            ids: pool.ids[1..].to_vec(),
        };
        for (prompt, &owner) in prompts.iter().zip(&owners) {
            let now = pick(&policy, &survivors, &[0; 5], prompt).index + 1;
            if owner != 0 {
                assert_eq!(now, owner);
            }
        }
    }

    #[test]
    fn prompts_without_a_full_block_fall_back_to_p2c() {
        let pool = Pool::new(3);
        let policy = PrefixHash::new(4, 1);
        let fallback = P2c::new(1);
        let candidates = pool.candidates(&[5, 0, 9]);
        let ctx = RouteContext::new(&pool.model, &candidates);
        for _ in 0..50 {
            let choice = policy.choose(&ctx).unwrap();
            assert_eq!(choice.steer, Some(Steer::NoPrefix));
            assert_eq!(choice.index, fallback.choose(&ctx).unwrap().index);
        }
        assert_eq!(policy.explain(&ctx).winner, None);
    }

    #[test]
    fn explain_names_the_owner_without_moving_anything() {
        let pool = Pool::new(4);
        let policy = PrefixHash::new(4, 1);
        let prompt = chain(3, 8);
        let candidates = pool.candidates(&[]);
        let ctx = RouteContext {
            blocks: &prompt,
            ..RouteContext::new(&pool.model, &candidates)
        };
        let explained = policy.explain(&ctx);
        let winner = explained.winner.unwrap();
        assert_eq!(winner, policy.choose(&ctx).unwrap().index);
        let best = explained
            .candidates
            .iter()
            .map(|s| s.score.unwrap())
            .fold(f64::MIN, f64::max);
        assert_eq!(explained.candidates[winner].score, Some(best));
    }
}
