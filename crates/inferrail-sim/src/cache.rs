//! Engine side prefix cache: chained block hashes and an LRU of full blocks.
//!
//! The hash is deliberately not the router's: the router must never depend
//! on engine hash values, so the simulator uses its own 64 bit chain.

use std::collections::HashMap;

use xxhash_rust::xxh3::xxh3_64_with_seed;

pub type BlockHash = u64;

const ROOT: BlockHash = 0x9e37_79b9_7f4a_7c15;
const NIL: usize = usize::MAX;

/// Hashes every full block; a trailing partial block is never cached.
pub fn block_hashes(tokens: &[u32], block_size: usize) -> Vec<BlockHash> {
    assert!(block_size > 0, "block size must be positive");
    let mut bytes = Vec::with_capacity(block_size * 4);
    let mut parent = ROOT;
    tokens
        .chunks_exact(block_size)
        .map(|block| {
            bytes.clear();
            for token in block {
                bytes.extend_from_slice(&token.to_le_bytes());
            }
            parent = xxh3_64_with_seed(&bytes, parent);
            parent
        })
        .collect()
}

#[derive(Debug, Clone, Copy)]
struct Node {
    hash: BlockHash,
    prev: usize,
    next: usize,
}

/// Fixed capacity LRU over block hashes with O(1) touch, insert and evict.
#[derive(Debug)]
pub struct PrefixCache {
    capacity: usize,
    index: HashMap<BlockHash, usize>,
    nodes: Vec<Node>,
    free: Vec<usize>,
    head: usize,
    tail: usize,
    evictions: u64,
}

impl PrefixCache {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            index: HashMap::with_capacity(capacity.min(1 << 20)),
            nodes: Vec::new(),
            free: Vec::new(),
            head: NIL,
            tail: NIL,
            evictions: 0,
        }
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn len(&self) -> usize {
        self.index.len()
    }

    pub fn is_empty(&self) -> bool {
        self.index.is_empty()
    }

    pub fn evictions(&self) -> u64 {
        self.evictions
    }

    pub fn contains(&self, hash: BlockHash) -> bool {
        self.index.contains_key(&hash)
    }

    /// Number of leading blocks present, touching each one that matched.
    pub fn match_prefix(&mut self, hashes: &[BlockHash]) -> usize {
        let matched = hashes
            .iter()
            .take_while(|h| self.index.contains_key(h))
            .count();
        self.touch_chain(&hashes[..matched]);
        matched
    }

    /// Inserts or refreshes a chain. The chain is touched tail first so its
    /// root ends most recent and eviction trims long prefixes from the end,
    /// the order vLLM frees blocks in.
    pub fn insert(&mut self, hashes: &[BlockHash]) {
        if self.capacity == 0 {
            return;
        }
        for &hash in hashes.iter().rev() {
            match self.index.get(&hash) {
                Some(&slot) => self.move_to_front(slot),
                None => {
                    if self.index.len() == self.capacity {
                        self.evict_lru();
                    }
                    let slot = self.alloc(hash);
                    self.index.insert(hash, slot);
                    self.push_front(slot);
                }
            }
        }
    }

    pub fn clear(&mut self) {
        self.index.clear();
        self.nodes.clear();
        self.free.clear();
        self.head = NIL;
        self.tail = NIL;
    }

    fn touch_chain(&mut self, hashes: &[BlockHash]) {
        for hash in hashes.iter().rev() {
            if let Some(&slot) = self.index.get(hash) {
                self.move_to_front(slot);
            }
        }
    }

    fn alloc(&mut self, hash: BlockHash) -> usize {
        let node = Node {
            hash,
            prev: NIL,
            next: NIL,
        };
        match self.free.pop() {
            Some(slot) => {
                self.nodes[slot] = node;
                slot
            }
            None => {
                self.nodes.push(node);
                self.nodes.len() - 1
            }
        }
    }

    fn evict_lru(&mut self) {
        let slot = self.tail;
        if slot == NIL {
            return;
        }
        self.unlink(slot);
        self.index.remove(&self.nodes[slot].hash);
        self.free.push(slot);
        self.evictions += 1;
    }

    fn move_to_front(&mut self, slot: usize) {
        if self.head != slot {
            self.unlink(slot);
            self.push_front(slot);
        }
    }

    fn push_front(&mut self, slot: usize) {
        self.nodes[slot].prev = NIL;
        self.nodes[slot].next = self.head;
        if self.head != NIL {
            self.nodes[self.head].prev = slot;
        }
        self.head = slot;
        if self.tail == NIL {
            self.tail = slot;
        }
    }

    fn unlink(&mut self, slot: usize) {
        let Node { prev, next, .. } = self.nodes[slot];
        if prev != NIL {
            self.nodes[prev].next = next;
        } else {
            self.head = next;
        }
        if next != NIL {
            self.nodes[next].prev = prev;
        } else {
            self.tail = prev;
        }
        self.nodes[slot].prev = NIL;
        self.nodes[slot].next = NIL;
    }

    #[cfg(test)]
    fn recency(&self) -> Vec<BlockHash> {
        let mut out = Vec::with_capacity(self.len());
        let mut at = self.head;
        while at != NIL {
            out.push(self.nodes[at].hash);
            at = self.nodes[at].next;
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn partial_blocks_are_not_hashed() {
        assert_eq!(block_hashes(&[1, 2, 3], 4).len(), 0);
        assert_eq!(block_hashes(&[1, 2, 3, 4, 5], 4).len(), 1);
        assert_eq!(block_hashes(&(0..64).collect::<Vec<_>>(), 16).len(), 4);
    }

    #[test]
    fn identical_blocks_at_different_depths_hash_differently() {
        let hashes = block_hashes(&[7; 32], 16);
        assert_ne!(hashes[0], hashes[1]);
    }

    #[test]
    fn lru_evicts_least_recent_and_counts() {
        let mut cache = PrefixCache::new(2);
        cache.insert(&[1]);
        cache.insert(&[2]);
        assert_eq!(cache.match_prefix(&[1]), 1);
        cache.insert(&[3]);
        assert!(cache.contains(1) && cache.contains(3) && !cache.contains(2));
        assert_eq!(cache.evictions(), 1);
    }

    #[test]
    fn eviction_trims_chains_from_the_tail() {
        let mut cache = PrefixCache::new(4);
        cache.insert(&[10, 11, 12, 13]);
        cache.insert(&[20]);
        assert_eq!(cache.match_prefix(&[10, 11, 12, 13]), 3);
        assert!(!cache.contains(13));
    }

    #[test]
    fn match_stops_at_first_gap() {
        let mut cache = PrefixCache::new(8);
        cache.insert(&[1, 2, 3]);
        assert_eq!(cache.match_prefix(&[1, 2, 9, 3]), 2);
        assert_eq!(cache.match_prefix(&[9, 1]), 0);
        assert_eq!(cache.match_prefix(&[]), 0);
    }

    #[test]
    fn zero_capacity_caches_nothing_and_clear_resets() {
        let mut cache = PrefixCache::new(0);
        cache.insert(&[1, 2]);
        assert!(cache.is_empty());
        let mut cache = PrefixCache::new(3);
        cache.insert(&[1, 2, 3]);
        cache.clear();
        assert!(cache.is_empty());
        cache.insert(&[4]);
        assert_eq!(cache.recency(), [4]);
    }

    #[derive(Debug, Clone)]
    enum Op {
        Insert(Vec<BlockHash>),
        Match(Vec<BlockHash>),
        Clear,
    }

    fn op() -> impl Strategy<Value = Op> {
        let chain = proptest::collection::vec(0u64..40, 0..10);
        prop_oneof![
            8 => chain.clone().prop_map(Op::Insert),
            8 => chain.prop_map(Op::Match),
            1 => Just(Op::Clear),
        ]
    }

    /// Reference model: a plain vector ordered most recent first.
    #[derive(Default)]
    struct Naive {
        order: Vec<BlockHash>,
        capacity: usize,
    }

    impl Naive {
        fn touch(&mut self, hash: BlockHash) {
            self.order.retain(|&h| h != hash);
            self.order.insert(0, hash);
        }

        fn insert(&mut self, chain: &[BlockHash]) {
            if self.capacity == 0 {
                return;
            }
            for &hash in chain.iter().rev() {
                if !self.order.contains(&hash) && self.order.len() == self.capacity {
                    self.order.pop();
                }
                self.touch(hash);
            }
        }

        fn match_prefix(&mut self, chain: &[BlockHash]) -> usize {
            let n = chain.iter().take_while(|h| self.order.contains(h)).count();
            for &hash in chain[..n].iter().rev() {
                self.touch(hash);
            }
            n
        }
    }

    proptest! {
        #[test]
        fn matches_naive_reference(capacity in 0usize..12, ops in proptest::collection::vec(op(), 0..60)) {
            let mut cache = PrefixCache::new(capacity);
            let mut naive = Naive { capacity, ..Naive::default() };
            for op in ops {
                match op {
                    Op::Insert(chain) => {
                        cache.insert(&chain);
                        naive.insert(&chain);
                    }
                    Op::Match(chain) => {
                        prop_assert_eq!(cache.match_prefix(&chain), naive.match_prefix(&chain));
                    }
                    Op::Clear => {
                        cache.clear();
                        naive.order.clear();
                    }
                }
                prop_assert!(cache.len() <= capacity);
                prop_assert_eq!(cache.recency(), naive.order.clone());
            }
        }

        #[test]
        fn changing_a_token_changes_its_block_and_every_later_one(
            tokens in proptest::collection::vec(any::<u32>(), 16..160),
            at in any::<prop::sample::Index>(),
            delta in 1u32..,
        ) {
            let before = block_hashes(&tokens, 16);
            let mut changed = tokens.clone();
            let i = at.index(tokens.len());
            changed[i] = changed[i].wrapping_add(delta);
            let after = block_hashes(&changed, 16);
            let block = i / 16;
            for j in 0..before.len() {
                if j < block {
                    prop_assert_eq!(before[j], after[j]);
                } else {
                    prop_assert_ne!(before[j], after[j]);
                }
            }
        }

        #[test]
        fn prefix_hashes_do_not_depend_on_what_follows(
            tokens in proptest::collection::vec(any::<u32>(), 0..200),
            cut in any::<prop::sample::Index>(),
            block_size in 1usize..33,
        ) {
            let n = cut.index(tokens.len() + 1);
            let full = block_hashes(&tokens, block_size);
            let head = block_hashes(&tokens[..n], block_size);
            prop_assert_eq!(&full[..head.len()], head.as_slice());
        }
    }
}
