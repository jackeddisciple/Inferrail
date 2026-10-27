//! The KV cache as vLLM v0.31.0 keeps it: a fixed pool of blocks that
//! requests hold by reference. A block keeps its hash after its last holder
//! lets go, so it still serves prefix hits, until the pool hands it out
//! again. Free blocks are handed out from the head of one queue: blocks
//! without a hash go to its head, cached ones to its tail, so the least
//! recently freed cached block is the first to be evicted.

use std::collections::HashMap;

use crate::cache::BlockHash;

pub type BlockId = u32;

const NIL: BlockId = BlockId::MAX;

#[derive(Debug, Clone, Copy)]
struct Block {
    hash: Option<BlockHash>,
    refs: u32,
    prev: BlockId,
    next: BlockId,
    queued: bool,
}

/// Every block of one engine's KV cache. More than one block may hold the
/// same hash when two requests computed the same prefix at once; lookups
/// find one of them and each is evicted on its own.
#[derive(Debug)]
pub struct BlockPool {
    blocks: Vec<Block>,
    head: BlockId,
    tail: BlockId,
    free: usize,
    cached: HashMap<BlockHash, BlockId>,
    duplicates: HashMap<BlockHash, Vec<BlockId>>,
    hashed: usize,
    caching: bool,
    evictions: u64,
    unhashed: Vec<BlockId>,
}

impl BlockPool {
    pub fn new(num_blocks: usize, caching: bool) -> Self {
        let n = BlockId::try_from(num_blocks).unwrap_or(NIL - 1);
        let blocks = (0..n)
            .map(|i| Block {
                hash: None,
                refs: 0,
                prev: if i == 0 { NIL } else { i - 1 },
                next: if i + 1 == n { NIL } else { i + 1 },
                queued: true,
            })
            .collect();
        Self {
            blocks,
            head: if n == 0 { NIL } else { 0 },
            tail: if n == 0 { NIL } else { n - 1 },
            free: n as usize,
            cached: HashMap::with_capacity((n as usize).min(1 << 20)),
            duplicates: HashMap::new(),
            hashed: 0,
            caching,
            evictions: 0,
            unhashed: Vec::new(),
        }
    }

    pub fn capacity(&self) -> usize {
        self.blocks.len()
    }

    pub fn free_blocks(&self) -> usize {
        self.free
    }

    /// Blocks some request holds; shared blocks count once.
    pub fn used_blocks(&self) -> usize {
        self.blocks.len() - self.free
    }

    /// Blocks that hold a hash, held or free.
    pub fn cached_blocks(&self) -> usize {
        self.hashed
    }

    pub fn evictions(&self) -> u64 {
        self.evictions
    }

    pub fn is_caching(&self) -> bool {
        self.caching
    }

    /// The share of blocks held, as vLLM's `kv_cache_usage_perc`.
    pub fn usage(&self) -> f64 {
        if self.blocks.is_empty() {
            0.0
        } else {
            self.used_blocks() as f64 / self.blocks.len() as f64
        }
    }

    /// The blocks holding the longest run of `hashes` from the first one.
    pub fn longest_hit(&self, hashes: &[BlockHash], out: &mut Vec<BlockId>) {
        out.clear();
        if !self.caching {
            return;
        }
        for hash in hashes {
            match self.cached.get(hash) {
                Some(&id) => out.push(id),
                None => break,
            }
        }
    }

    /// Whether the pool can take a reference on `hit` and hand out `fresh`
    /// more blocks: hit blocks nobody holds count against the free ones.
    pub fn fits(&self, hit: &[BlockId], fresh: usize) -> bool {
        let idle = hit
            .iter()
            .filter(|&&id| self.blocks[id as usize].queued)
            .count();
        fresh + idle <= self.free
    }

    /// Takes a reference on each block, out of the free queue if it was there.
    pub fn touch(&mut self, ids: &[BlockId]) {
        for &id in ids {
            if self.blocks[id as usize].queued {
                self.unlink(id);
            }
            self.blocks[id as usize].refs += 1;
        }
    }

    /// Takes `n` blocks from the head of the free queue into `out`, evicting
    /// the hash each one held into `evicted`. Takes nothing when fewer than
    /// `n` are free.
    pub fn allocate(
        &mut self,
        n: usize,
        out: &mut Vec<BlockId>,
        evicted: &mut Vec<BlockHash>,
    ) -> bool {
        if n > self.free {
            return false;
        }
        for _ in 0..n {
            let id = self.head;
            self.unlink(id);
            if let Some(hash) = self.forget(id) {
                self.evictions += 1;
                evicted.push(hash);
            }
            self.blocks[id as usize].refs = 1;
            out.push(id);
        }
        true
    }

    /// Records that a full block holds `hash`. Returns false when caching is
    /// off or the block already holds it.
    pub fn cache(&mut self, id: BlockId, hash: BlockHash) -> bool {
        if !self.caching || self.blocks[id as usize].hash == Some(hash) {
            return false;
        }
        self.forget(id);
        self.blocks[id as usize].hash = Some(hash);
        self.hashed += 1;
        match self.cached.get(&hash) {
            None => {
                self.cached.insert(hash, id);
            }
            Some(_) => self.duplicates.entry(hash).or_default().push(id),
        }
        true
    }

    /// Drops one reference on each of a request's blocks, given in its
    /// order. The request's last blocks are freed first, so its tail is
    /// evicted before its head.
    pub fn free(&mut self, ids: &[BlockId]) {
        let mut unhashed = std::mem::take(&mut self.unhashed);
        for &id in ids.iter().rev() {
            let block = &mut self.blocks[id as usize];
            debug_assert!(block.refs > 0, "freeing block {id} twice");
            block.refs -= 1;
            if block.refs == 0 {
                if block.hash.is_some() {
                    self.push_back(id);
                } else {
                    unhashed.push(id);
                }
            }
        }
        for &id in unhashed.iter().rev() {
            self.push_front(id);
        }
        unhashed.clear();
        self.unhashed = unhashed;
    }

    /// Forgets every hash, so nothing hits until blocks are cached again.
    pub fn reset(&mut self) {
        self.cached.clear();
        self.duplicates.clear();
        for block in &mut self.blocks {
            block.hash = None;
        }
        self.hashed = 0;
    }

    fn forget(&mut self, id: BlockId) -> Option<BlockHash> {
        let hash = self.blocks[id as usize].hash.take()?;
        self.hashed -= 1;
        if self.cached.get(&hash) == Some(&id) {
            match self.duplicates.get_mut(&hash).and_then(Vec::pop) {
                Some(other) => {
                    self.cached.insert(hash, other);
                }
                None => {
                    self.cached.remove(&hash);
                }
            }
        } else if let Some(others) = self.duplicates.get_mut(&hash) {
            others.retain(|&b| b != id);
        }
        if self.duplicates.get(&hash).is_some_and(Vec::is_empty) {
            self.duplicates.remove(&hash);
        }
        Some(hash)
    }

    fn unlink(&mut self, id: BlockId) {
        let Block { prev, next, .. } = self.blocks[id as usize];
        if prev == NIL {
            self.head = next;
        } else {
            self.blocks[prev as usize].next = next;
        }
        if next == NIL {
            self.tail = prev;
        } else {
            self.blocks[next as usize].prev = prev;
        }
        let block = &mut self.blocks[id as usize];
        block.prev = NIL;
        block.next = NIL;
        block.queued = false;
        self.free -= 1;
    }

    fn push_front(&mut self, id: BlockId) {
        let head = self.head;
        let block = &mut self.blocks[id as usize];
        block.prev = NIL;
        block.next = head;
        block.queued = true;
        if head == NIL {
            self.tail = id;
        } else {
            self.blocks[head as usize].prev = id;
        }
        self.head = id;
        self.free += 1;
    }

    fn push_back(&mut self, id: BlockId) {
        let tail = self.tail;
        let block = &mut self.blocks[id as usize];
        block.prev = tail;
        block.next = NIL;
        block.queued = true;
        if tail == NIL {
            self.head = id;
        } else {
            self.blocks[tail as usize].next = id;
        }
        self.tail = id;
        self.free += 1;
    }

    #[cfg(test)]
    fn queue(&self) -> Vec<BlockId> {
        let mut out = Vec::new();
        let mut at = self.head;
        while at != NIL {
            out.push(at);
            at = self.blocks[at as usize].next;
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn take(pool: &mut BlockPool, n: usize) -> Vec<BlockId> {
        let mut out = Vec::new();
        assert!(pool.allocate(n, &mut out, &mut Vec::new()));
        out
    }

    #[test]
    fn shared_blocks_count_once_and_survive_their_holders() {
        let mut pool = BlockPool::new(8, true);
        let a = take(&mut pool, 3);
        for (id, hash) in a.iter().zip([10, 11, 12]) {
            assert!(pool.cache(*id, hash));
        }
        let mut hit = Vec::new();
        pool.longest_hit(&[10, 11, 99], &mut hit);
        assert_eq!(hit, a[..2]);
        pool.touch(&hit);
        assert_eq!(pool.used_blocks(), 3);
        pool.free(&a);
        assert_eq!(pool.used_blocks(), 2);
        pool.free(&hit);
        assert_eq!((pool.used_blocks(), pool.cached_blocks()), (0, 3));
        pool.longest_hit(&[10, 11, 12], &mut hit);
        assert_eq!(hit.len(), 3);
    }

    #[test]
    fn uncached_blocks_are_reused_first_and_cached_ones_in_lru_order() {
        let mut pool = BlockPool::new(4, true);
        let blocks = take(&mut pool, 4);
        pool.cache(blocks[0], 1);
        pool.cache(blocks[1], 2);
        pool.free(&blocks);
        let queue = pool.queue();
        assert_eq!(&queue[..2], &[blocks[3], blocks[2]]);
        assert_eq!(&queue[2..], &[blocks[1], blocks[0]]);
        let mut evicted = Vec::new();
        let mut out = Vec::new();
        assert!(pool.allocate(3, &mut out, &mut evicted));
        assert_eq!(evicted, [2]);
        assert_eq!(pool.evictions(), 1);
        let mut hit = Vec::new();
        pool.longest_hit(&[1], &mut hit);
        assert_eq!(hit, [blocks[0]]);
    }

    #[test]
    fn allocation_never_takes_more_than_is_free() {
        let mut pool = BlockPool::new(2, true);
        let mut out = Vec::new();
        assert!(!pool.allocate(3, &mut out, &mut Vec::new()));
        assert!(out.is_empty());
        assert_eq!(pool.free_blocks(), 2);
        assert!(pool.allocate(2, &mut out, &mut Vec::new()));
        assert!(!pool.allocate(1, &mut out, &mut Vec::new()));
        assert_eq!(pool.usage(), 1.0);
    }

    #[test]
    fn duplicates_are_found_until_the_last_one_goes() {
        let mut pool = BlockPool::new(3, true);
        let a = take(&mut pool, 1);
        let b = take(&mut pool, 1);
        assert!(pool.cache(a[0], 7));
        assert!(pool.cache(b[0], 7));
        assert!(!pool.cache(b[0], 7));
        pool.free(&a);
        pool.free(&b);
        let mut evicted = Vec::new();
        let mut out = Vec::new();
        pool.allocate(2, &mut out, &mut evicted);
        assert_eq!(evicted, [7]);
        let mut hit = Vec::new();
        pool.longest_hit(&[7], &mut hit);
        assert_eq!(hit.len(), 1);
        pool.allocate(1, &mut out, &mut evicted);
        assert_eq!(evicted, [7, 7]);
        pool.longest_hit(&[7], &mut hit);
        assert!(hit.is_empty());
    }

    #[test]
    fn caching_off_never_hits_and_reset_forgets() {
        let mut off = BlockPool::new(2, false);
        let blocks = take(&mut off, 1);
        assert!(!off.cache(blocks[0], 5));
        let mut hit = Vec::new();
        off.longest_hit(&[5], &mut hit);
        assert!(hit.is_empty());

        let mut pool = BlockPool::new(2, true);
        let blocks = take(&mut pool, 2);
        pool.cache(blocks[0], 1);
        pool.cache(blocks[1], 2);
        pool.reset();
        assert_eq!(pool.cached_blocks(), 0);
        pool.longest_hit(&[1, 2], &mut hit);
        assert!(hit.is_empty());
        pool.free(&blocks);
        assert_eq!(pool.free_blocks(), 2);
    }

    #[derive(Debug, Clone)]
    enum Op {
        Admit {
            hashes: Vec<BlockHash>,
            extra: usize,
        },
        Release(prop::sample::Index),
        Reset,
    }

    fn op() -> impl Strategy<Value = Op> {
        prop_oneof![
            6 => (prop::collection::vec(0u64..12, 0..6), 0usize..4)
                .prop_map(|(hashes, extra)| Op::Admit { hashes, extra }),
            4 => any::<prop::sample::Index>().prop_map(Op::Release),
            1 => Just(Op::Reset),
        ]
    }

    /// Reference model: the free queue as a plain vector, head first.
    #[derive(Debug, Default)]
    struct Naive {
        hash: Vec<Option<BlockHash>>,
        refs: Vec<u32>,
        queue: Vec<BlockId>,
    }

    impl Naive {
        fn new(n: usize) -> Self {
            Self {
                hash: vec![None; n],
                refs: vec![0; n],
                queue: (0..n as BlockId).collect(),
            }
        }

        fn holder(&self, hash: BlockHash) -> bool {
            self.hash.contains(&Some(hash))
        }

        fn hit(&self, hashes: &[BlockHash]) -> usize {
            hashes.iter().take_while(|h| self.holder(**h)).count()
        }
    }

    proptest! {
        #[test]
        fn matches_a_naive_pool(capacity in 1usize..10, ops in prop::collection::vec(op(), 0..80)) {
            let mut pool = BlockPool::new(capacity, true);
            let mut naive = Naive::new(capacity);
            let mut held: Vec<Vec<BlockId>> = Vec::new();
            for op in ops {
                match op {
                    Op::Admit { hashes, extra } => {
                        let mut hit = Vec::new();
                        pool.longest_hit(&hashes, &mut hit);
                        prop_assert_eq!(hit.len(), naive.hit(&hashes));
                        let missing = hashes.len() - hit.len() + extra;
                        let mut fresh = Vec::new();
                        let mut evicted = Vec::new();
                        let idle = hit.iter().filter(|&&b| naive.refs[b as usize] == 0).count();
                        let ok = missing + idle <= naive.queue.len();
                        prop_assert_eq!(pool.fits(&hit, missing), ok);
                        if !ok {
                            continue;
                        }
                        pool.touch(&hit);
                        for &b in &hit {
                            naive.queue.retain(|&q| q != b);
                            naive.refs[b as usize] += 1;
                        }
                        prop_assert!(pool.allocate(missing, &mut fresh, &mut evicted));
                        let want: Vec<BlockId> = naive.queue.drain(..missing).collect();
                        prop_assert_eq!(&fresh, &want);
                        let mut want_evicted = Vec::new();
                        for &b in &want {
                            if let Some(h) = naive.hash[b as usize].take() {
                                want_evicted.push(h);
                            }
                            naive.refs[b as usize] = 1;
                        }
                        prop_assert_eq!(evicted, want_evicted);
                        let mut blocks = hit.clone();
                        blocks.extend(&fresh);
                        for (i, hash) in hashes.iter().enumerate().skip(hit.len()) {
                            let b = blocks[i];
                            if pool.cache(b, *hash) {
                                naive.hash[b as usize] = Some(*hash);
                            }
                        }
                        held.push(blocks);
                    }
                    Op::Release(which) => {
                        if held.is_empty() {
                            continue;
                        }
                        let blocks = held.swap_remove(which.index(held.len()));
                        pool.free(&blocks);
                        let mut front = Vec::new();
                        for &b in blocks.iter().rev() {
                            naive.refs[b as usize] -= 1;
                            if naive.refs[b as usize] == 0 {
                                if naive.hash[b as usize].is_some() {
                                    naive.queue.push(b);
                                } else {
                                    front.push(b);
                                }
                            }
                        }
                        naive.queue.splice(0..0, front);
                    }
                    Op::Reset => {
                        pool.reset();
                        naive.hash.iter_mut().for_each(|h| *h = None);
                    }
                }
                prop_assert_eq!(pool.queue(), naive.queue.clone());
                prop_assert_eq!(pool.free_blocks(), naive.queue.len());
                prop_assert_eq!(pool.used_blocks(), naive.refs.iter().filter(|&&r| r > 0).count());
                prop_assert_eq!(pool.cached_blocks(), naive.hash.iter().filter(|h| h.is_some()).count());
            }
            for blocks in held {
                pool.free(&blocks);
            }
            prop_assert_eq!(pool.free_blocks(), capacity);
        }
    }
}
