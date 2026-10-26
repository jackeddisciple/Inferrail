//! Which replicas hold each block: a 64 way sharded map from block hash to
//! a replica set. Shards are picked by the hash's top bits and buckets by
//! its low bits, so the two never correlate.

use std::collections::HashMap;
use std::sync::{PoisonError, RwLock};

use crate::IdentityHash;
use crate::set::ReplicaSet;

const SHARDS: usize = 64;

#[derive(Debug)]
pub(crate) struct BlockMap {
    shards: Box<[RwLock<HashMap<u128, ReplicaSet, IdentityHash>>]>,
}

impl Default for BlockMap {
    fn default() -> Self {
        Self {
            shards: (0..SHARDS).map(|_| RwLock::default()).collect(),
        }
    }
}

fn shard_of(hash: u128) -> usize {
    (hash >> 122) as usize
}

impl BlockMap {
    pub(crate) fn insert(&self, hash: u128, slot: u32) {
        let mut shard = self.shards[shard_of(hash)]
            .write()
            .unwrap_or_else(PoisonError::into_inner);
        shard.entry(hash).or_default().insert(slot);
    }

    pub(crate) fn remove(&self, hash: u128, slot: u32) {
        let mut shard = self.shards[shard_of(hash)]
            .write()
            .unwrap_or_else(PoisonError::into_inner);
        if let Some(set) = shard.get_mut(&hash) {
            set.remove(slot);
            if set.is_empty() {
                shard.remove(&hash);
            }
        }
    }

    pub(crate) fn holders(&self, hash: u128) -> ReplicaSet {
        self.shards[shard_of(hash)]
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .get(&hash)
            .cloned()
            .unwrap_or_default()
    }

    /// Walks the chain from its root keeping the members of `alive` that
    /// hold every block so far, and reports each one that drops out with the
    /// depth it reached. Stops as soon as nobody is left.
    pub(crate) fn walk(
        &self,
        chain: &[u128],
        alive: &mut ReplicaSet,
        mut dropped: impl FnMut(u32, usize),
    ) {
        for (depth, &hash) in chain.iter().enumerate() {
            if alive.is_empty() {
                return;
            }
            let shard = self.shards[shard_of(hash)]
                .read()
                .unwrap_or_else(PoisonError::into_inner);
            match shard.get(&hash) {
                Some(holders) => alive.retain_in(holders, |slot| dropped(slot, depth)),
                None => {
                    for slot in alive.iter() {
                        dropped(slot, depth);
                    }
                    alive.clear();
                }
            }
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.shards
            .iter()
            .map(|s| s.read().unwrap_or_else(PoisonError::into_inner).len())
            .sum()
    }

    /// Bytes held by the shards' tables and spilled replica sets.
    pub(crate) fn bytes(&self) -> usize {
        self.shards
            .iter()
            .map(|s| {
                let shard = s.read().unwrap_or_else(PoisonError::into_inner);
                shard.capacity() * (std::mem::size_of::<(u128, ReplicaSet)>() + 1)
                    + shard.values().map(ReplicaSet::heap_bytes).sum::<usize>()
            })
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use std::collections::{HashMap as Std, HashSet};

    #[derive(Debug, Clone)]
    enum Op {
        Stored(u32, u128),
        Removed(u32, u128),
        Cleared(u32),
    }

    fn op() -> impl Strategy<Value = Op> {
        let hash =
            (0u128..24).prop_map(|h| h.wrapping_mul(0x9e37_79b9_7f4a_7c15_f39c_c060_5ced_c834) ^ h);
        prop_oneof![
            6 => (0u32..70, hash.clone()).prop_map(|(r, h)| Op::Stored(r, h)),
            3 => (0u32..70, hash).prop_map(|(r, h)| Op::Removed(r, h)),
            1 => (0u32..70).prop_map(Op::Cleared),
        ]
    }

    proptest! {
        /// Invariant 3: after any sequence of stores, removals and clears the
        /// map equals a plain map from replica to its set of blocks.
        #[test]
        fn equals_a_naive_reference_after_any_sequence(ops in prop::collection::vec(op(), 0..200)) {
            let map = BlockMap::default();
            let mut naive: Std<u32, HashSet<u128>> = Std::new();
            for op in &ops {
                match *op {
                    Op::Stored(r, h) => {
                        map.insert(h, r);
                        naive.entry(r).or_default().insert(h);
                    }
                    Op::Removed(r, h) => {
                        map.remove(h, r);
                        naive.entry(r).or_default().remove(&h);
                    }
                    Op::Cleared(r) => {
                        for h in naive.remove(&r).unwrap_or_default() {
                            map.remove(h, r);
                        }
                    }
                }
            }
            let mut every: HashSet<u128> = naive.values().flatten().copied().collect();
            every.extend(ops.iter().filter_map(|o| match o { Op::Stored(_, h) | Op::Removed(_, h) => Some(*h), Op::Cleared(_) => None }));
            for h in every {
                let want: HashSet<u32> = naive.iter().filter(|(_, hs)| hs.contains(&h)).map(|(r, _)| *r).collect();
                prop_assert_eq!(map.holders(h).iter().collect::<HashSet<_>>(), want);
            }
            prop_assert_eq!(map.len(), naive.values().flatten().collect::<HashSet<_>>().len());
        }
    }
}
