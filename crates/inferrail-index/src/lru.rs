//! A fixed capacity LRU over block hashes with O(1) touch, insert and evict,
//! reporting which blocks entered and left so the shared map can follow.

use std::collections::HashMap;

use crate::IdentityHash;

const NIL: u32 = u32::MAX;

#[derive(Debug, Clone, Copy)]
struct Node {
    hash: u128,
    prev: u32,
    next: u32,
}

#[derive(Debug, Default)]
pub(crate) struct Lru {
    capacity: usize,
    index: HashMap<u128, u32, IdentityHash>,
    nodes: Vec<Node>,
    free: Vec<u32>,
    head: u32,
    tail: u32,
}

impl Lru {
    pub(crate) fn new(capacity: usize) -> Self {
        Self {
            capacity,
            head: NIL,
            tail: NIL,
            ..Self::default()
        }
    }

    pub(crate) fn capacity(&self) -> usize {
        self.capacity
    }

    pub(crate) fn len(&self) -> usize {
        self.index.len()
    }

    pub(crate) fn hashes(&self) -> impl Iterator<Item = u128> + '_ {
        self.index.keys().copied()
    }

    pub(crate) fn contains(&self, hash: u128) -> bool {
        self.index.contains_key(&hash)
    }

    /// Inserts or refreshes a chain tail first, so its root ends most recent
    /// and eviction trims long prefixes from the end, the order engines free
    /// blocks in. Calls `added` and `evicted` for blocks entering and leaving.
    pub(crate) fn insert(
        &mut self,
        chain: &[u128],
        mut added: impl FnMut(u128),
        mut evicted: impl FnMut(u128),
    ) {
        if self.capacity == 0 {
            return;
        }
        for &hash in chain.iter().rev() {
            if let Some(&slot) = self.index.get(&hash) {
                self.move_to_front(slot);
                continue;
            }
            if self.index.len() >= self.capacity
                && let Some(gone) = self.pop_tail()
            {
                evicted(gone);
            }
            let slot = self.alloc(hash);
            self.index.insert(hash, slot);
            self.push_front(slot);
            added(hash);
        }
    }

    /// Shrinks or grows to `capacity`, evicting least recent blocks first.
    pub(crate) fn resize(&mut self, capacity: usize, mut evicted: impl FnMut(u128)) {
        self.capacity = capacity;
        while self.index.len() > capacity {
            match self.pop_tail() {
                Some(gone) => evicted(gone),
                None => break,
            }
        }
    }

    /// Drops one block, wherever it is in the recency order.
    pub(crate) fn remove(&mut self, hash: u128) -> bool {
        let Some(slot) = self.index.remove(&hash) else {
            return false;
        };
        self.unlink(slot);
        self.free.push(slot);
        true
    }

    pub(crate) fn clear(&mut self) {
        let capacity = self.capacity;
        *self = Self::new(capacity);
    }

    fn pop_tail(&mut self) -> Option<u128> {
        let slot = self.tail;
        if slot == NIL {
            return None;
        }
        self.unlink(slot);
        let hash = self.nodes[slot as usize].hash;
        self.index.remove(&hash);
        self.free.push(slot);
        Some(hash)
    }

    fn alloc(&mut self, hash: u128) -> u32 {
        let node = Node {
            hash,
            prev: NIL,
            next: NIL,
        };
        match self.free.pop() {
            Some(slot) => {
                self.nodes[slot as usize] = node;
                slot
            }
            None => {
                self.nodes.push(node);
                (self.nodes.len() - 1) as u32
            }
        }
    }

    fn move_to_front(&mut self, slot: u32) {
        if self.head != slot {
            self.unlink(slot);
            self.push_front(slot);
        }
    }

    fn push_front(&mut self, slot: u32) {
        self.nodes[slot as usize].prev = NIL;
        self.nodes[slot as usize].next = self.head;
        if self.head != NIL {
            self.nodes[self.head as usize].prev = slot;
        }
        self.head = slot;
        if self.tail == NIL {
            self.tail = slot;
        }
    }

    fn unlink(&mut self, slot: u32) {
        let Node { prev, next, .. } = self.nodes[slot as usize];
        if prev != NIL {
            self.nodes[prev as usize].next = next;
        } else {
            self.head = next;
        }
        if next != NIL {
            self.nodes[next as usize].prev = prev;
        } else {
            self.tail = prev;
        }
        self.nodes[slot as usize].prev = NIL;
        self.nodes[slot as usize].next = NIL;
    }

    /// Bytes held, for memory accounting.
    pub(crate) fn bytes(&self) -> usize {
        self.nodes.capacity() * std::mem::size_of::<Node>()
            + self.free.capacity() * 4
            + self.index.capacity() * (std::mem::size_of::<(u128, u32)>() + 1)
    }

    #[cfg(test)]
    pub(crate) fn recency(&self) -> Vec<u128> {
        let mut out = Vec::with_capacity(self.len());
        let mut at = self.head;
        while at != NIL {
            out.push(self.nodes[at as usize].hash);
            at = self.nodes[at as usize].next;
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn insert(lru: &mut Lru, chain: &[u128]) -> (Vec<u128>, Vec<u128>) {
        let (mut added, mut evicted) = (Vec::new(), Vec::new());
        lru.insert(chain, |h| added.push(h), |h| evicted.push(h));
        (added, evicted)
    }

    #[test]
    fn long_chains_are_trimmed_from_the_tail() {
        let mut lru = Lru::new(4);
        insert(&mut lru, &[10, 11, 12, 13]);
        let (added, evicted) = insert(&mut lru, &[20]);
        assert_eq!((added, evicted), (vec![20], vec![13]));
        assert_eq!(lru.recency(), [20, 10, 11, 12]);
        let (added, evicted) = insert(&mut lru, &[1, 2, 3, 4, 5, 6]);
        assert_eq!(added, [6, 5, 4, 3, 2, 1]);
        assert_eq!(evicted.len(), 6);
        assert_eq!(lru.recency(), [1, 2, 3, 4]);
    }

    #[test]
    fn refreshing_touches_without_reporting() {
        let mut lru = Lru::new(3);
        insert(&mut lru, &[1, 2]);
        let (added, evicted) = insert(&mut lru, &[2, 1]);
        assert!(added.is_empty() && evicted.is_empty());
        assert_eq!(lru.recency(), [2, 1]);
    }

    #[test]
    fn resizing_evicts_least_recent_and_zero_capacity_holds_nothing() {
        let mut lru = Lru::new(5);
        insert(&mut lru, &[1, 2, 3, 4, 5]);
        let mut gone = Vec::new();
        lru.resize(2, |h| gone.push(h));
        assert_eq!((gone, lru.recency()), (vec![5, 4, 3], vec![1, 2]));
        let mut empty = Lru::new(0);
        assert_eq!(insert(&mut empty, &[1]), (vec![], vec![]));
        lru.clear();
        assert_eq!((lru.len(), lru.capacity()), (0, 2));
    }

    #[test]
    fn removed_blocks_leave_the_order_and_free_their_node() {
        let mut lru = Lru::new(4);
        insert(&mut lru, &[1, 2, 3, 4]);
        assert!(lru.remove(2) && !lru.remove(2) && !lru.remove(9));
        assert!(lru.remove(1) && lru.remove(4));
        assert_eq!(lru.recency(), [3]);
        assert!(!lru.contains(1) && lru.contains(3));
        insert(&mut lru, &[5, 6, 7]);
        assert_eq!(lru.recency(), [5, 6, 7, 3]);
        assert_eq!(lru.nodes.len(), 4);
    }

    /// The same reference model the simulator's cache is held to.
    #[derive(Default)]
    struct Naive {
        order: Vec<u128>,
        capacity: usize,
    }

    impl Naive {
        fn insert(&mut self, chain: &[u128]) {
            if self.capacity == 0 {
                return;
            }
            for &hash in chain.iter().rev() {
                if !self.order.contains(&hash) && self.order.len() >= self.capacity {
                    self.order.pop();
                }
                self.order.retain(|&h| h != hash);
                self.order.insert(0, hash);
            }
        }
    }

    proptest! {
        #[test]
        fn matches_a_naive_lru(
            capacity in 0usize..10,
            chains in prop::collection::vec(prop::collection::vec(0u128..30, 0..8), 0..50),
            shrink_to in prop::option::of(0usize..10),
        ) {
            let mut lru = Lru::new(capacity);
            let mut naive = Naive { capacity, ..Naive::default() };
            for chain in &chains {
                let mut held: std::collections::HashSet<u128> = lru.recency().into_iter().collect();
                let events = std::cell::RefCell::new(Vec::new());
                lru.insert(chain, |h| events.borrow_mut().push((true, h)), |h| events.borrow_mut().push((false, h)));
                for (added, h) in events.into_inner() {
                    if added {
                        prop_assert!(held.insert(h), "added a block already held");
                    } else {
                        prop_assert!(held.remove(&h), "evicted a block not held");
                    }
                }
                naive.insert(chain);
                prop_assert_eq!(lru.recency(), naive.order.clone());
                prop_assert_eq!(held, lru.recency().into_iter().collect());
                prop_assert!(lru.len() <= capacity);
            }
            if let Some(size) = shrink_to {
                let mut gone = Vec::new();
                lru.resize(size, |h| gone.push(h));
                naive.order.truncate(size);
                prop_assert_eq!(lru.recency(), naive.order.clone());
            }
        }
    }
}
