use std::collections::HashMap;
use std::sync::{Arc, Mutex, PoisonError, RwLock};

use inferrail_core::ids::{BlockHash, ReplicaId};

use crate::lru::Lru;
use crate::map::BlockMap;
use crate::set::ReplicaSet;

/// One replica's place in the index: its slot and the generation the slot
/// had when it was handed out, so a forgotten replica's slot can be reused
/// without the old owner ever writing into it again.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Slot {
    index: u32,
    generation: u64,
}

impl Slot {
    pub fn index(self) -> u32 {
        self.index
    }
}

#[derive(Debug)]
struct Cell {
    generation: u64,
    shadow: Lru,
}

#[derive(Debug, Default)]
struct Registry {
    by_id: HashMap<ReplicaId, Slot>,
    ids: Vec<Option<ReplicaId>>,
    free: Vec<u32>,
}

/// What a replica is predicted to hold.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReplicaStats {
    pub replica: ReplicaId,
    pub blocks: usize,
    pub capacity: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexStats {
    /// Distinct blocks held by at least one replica.
    pub entries: usize,
    /// Bytes held by the shared map and every shadow cache.
    pub bytes: usize,
    pub replicas: Vec<ReplicaStats>,
}

/// The KV index in approximate mode: every routed prompt is recorded as
/// cached on the replica it went to, in a shadow LRU sized to that
/// replica's KV capacity, and a shared map answers which replicas hold the
/// longest prefix of a prompt. Readers never wait on recording.
#[derive(Debug, Default)]
pub struct KvIndex {
    blocks: BlockMap,
    registry: RwLock<Registry>,
    cells: RwLock<Vec<Arc<Mutex<Cell>>>>,
}

fn raw(hashes: &[BlockHash]) -> impl Iterator<Item = u128> + '_ {
    hashes.iter().map(|h| h.get())
}

impl KvIndex {
    pub fn new() -> Self {
        Self::default()
    }

    /// The replica's slot, registering it on first sight.
    pub fn slot(&self, id: &ReplicaId) -> Slot {
        if let Some(slot) = self.find(id) {
            return slot;
        }
        let mut registry = self
            .registry
            .write()
            .unwrap_or_else(PoisonError::into_inner);
        if let Some(&slot) = registry.by_id.get(id) {
            return slot;
        }
        let mut cells = self.cells.write().unwrap_or_else(PoisonError::into_inner);
        let slot = match registry.free.pop() {
            Some(index) => {
                let generation = cells[index as usize]
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .generation;
                registry.ids[index as usize] = Some(id.clone());
                Slot { index, generation }
            }
            None => {
                let index = u32::try_from(cells.len()).unwrap_or(u32::MAX);
                cells.push(Arc::new(Mutex::new(Cell {
                    generation: 0,
                    shadow: Lru::new(0),
                })));
                registry.ids.push(Some(id.clone()));
                Slot {
                    index,
                    generation: 0,
                }
            }
        };
        registry.by_id.insert(id.clone(), slot);
        slot
    }

    pub fn find(&self, id: &ReplicaId) -> Option<Slot> {
        self.registry
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .by_id
            .get(id)
            .copied()
    }

    fn cell(&self, slot: Slot) -> Option<Arc<Mutex<Cell>>> {
        self.cells
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .get(slot.index as usize)
            .cloned()
    }

    /// Records a prompt routed to `slot`: its blocks become the most recent
    /// on that replica, whose shadow cache holds `capacity` blocks. Returns
    /// the blocks the replica did not hold before, for [`KvIndex::revoke`].
    pub fn record(&self, slot: Slot, hashes: &[BlockHash], capacity: usize) -> Vec<BlockHash> {
        let Some(cell) = self.cell(slot) else {
            return Vec::new();
        };
        let mut cell = cell.lock().unwrap_or_else(PoisonError::into_inner);
        if cell.generation != slot.generation {
            return Vec::new();
        }
        let index = slot.index;
        if cell.shadow.capacity() != capacity {
            cell.shadow
                .resize(capacity, |gone| self.blocks.remove(gone, index));
        }
        let chain: Vec<u128> = raw(hashes).collect();
        let fresh: Vec<u128> = chain
            .iter()
            .copied()
            .filter(|&hash| !cell.shadow.contains(hash))
            .collect();
        cell.shadow.insert(
            &chain,
            |added| self.blocks.insert(added, index),
            |gone| self.blocks.remove(gone, index),
        );
        fresh
            .into_iter()
            .filter(|&hash| cell.shadow.contains(hash))
            .map(BlockHash::new)
            .collect()
    }

    /// Takes back blocks a record added, for a prompt the engine never
    /// computed because it refused the request or failed before answering.
    pub fn revoke(&self, slot: Slot, hashes: &[BlockHash]) {
        if hashes.is_empty() {
            return;
        }
        let Some(cell) = self.cell(slot) else { return };
        let mut cell = cell.lock().unwrap_or_else(PoisonError::into_inner);
        if cell.generation != slot.generation {
            return;
        }
        for hash in raw(hashes) {
            if cell.shadow.remove(hash) {
                self.blocks.remove(hash, slot.index);
            }
        }
    }

    /// Drops everything recorded for `slot`, as when its engine restarted.
    pub fn clear(&self, slot: Slot) {
        let Some(cell) = self.cell(slot) else { return };
        let mut cell = cell.lock().unwrap_or_else(PoisonError::into_inner);
        if cell.generation == slot.generation {
            for hash in cell.shadow.hashes() {
                self.blocks.remove(hash, slot.index);
            }
            cell.shadow.clear();
        }
    }

    /// Removes a replica that left the fleet and frees its slot.
    pub fn forget(&self, id: &ReplicaId) {
        let mut registry = self
            .registry
            .write()
            .unwrap_or_else(PoisonError::into_inner);
        let Some(slot) = registry.by_id.remove(id) else {
            return;
        };
        if let Some(cell) = self.cell(slot) {
            let mut cell = cell.lock().unwrap_or_else(PoisonError::into_inner);
            for hash in cell.shadow.hashes() {
                self.blocks.remove(hash, slot.index);
            }
            cell.shadow = Lru::new(0);
            cell.generation += 1;
        }
        registry.ids[slot.index as usize] = None;
        registry.free.push(slot.index);
    }

    /// The replicas currently known to the index.
    pub fn replicas(&self) -> Vec<ReplicaId> {
        self.registry
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .ids
            .iter()
            .flatten()
            .cloned()
            .collect()
    }

    /// For each candidate, how many leading blocks of `hashes` it holds.
    /// A replica missing block `i` never counts as holding a later one.
    pub fn longest_prefix(&self, hashes: &[BlockHash], candidates: &[Slot]) -> Vec<usize> {
        let mut out = vec![0; candidates.len()];
        self.longest_prefix_into(hashes, candidates, &mut out);
        out
    }

    pub fn longest_prefix_into(
        &self,
        hashes: &[BlockHash],
        candidates: &[Slot],
        out: &mut [usize],
    ) {
        let chain: Vec<u128> = raw(hashes).collect();
        let mut alive: ReplicaSet = candidates.iter().map(|s| s.index).collect();
        let top = candidates
            .iter()
            .map(|s| s.index as usize + 1)
            .max()
            .unwrap_or(0);
        let mut depth = vec![chain.len(); top];
        self.blocks
            .walk(&chain, &mut alive, |slot, at| depth[slot as usize] = at);
        for (out, slot) in out.iter_mut().zip(candidates) {
            *out = depth[slot.index as usize];
        }
    }

    pub fn stats(&self) -> IndexStats {
        let registry = self.registry.read().unwrap_or_else(PoisonError::into_inner);
        let cells = self.cells.read().unwrap_or_else(PoisonError::into_inner);
        let mut bytes = self.blocks.bytes();
        let mut replicas = Vec::new();
        for (index, id) in registry.ids.iter().enumerate() {
            let cell = cells[index].lock().unwrap_or_else(PoisonError::into_inner);
            bytes += cell.shadow.bytes();
            if let Some(id) = id {
                replicas.push(ReplicaStats {
                    replica: id.clone(),
                    blocks: cell.shadow.len(),
                    capacity: cell.shadow.capacity(),
                });
            }
        }
        IndexStats {
            entries: self.blocks.len(),
            bytes,
            replicas,
        }
    }

    #[cfg(test)]
    pub(crate) fn holders(&self, hash: u128) -> ReplicaSet {
        self.blocks.holders(hash)
    }

    /// Checks that the shared map holds exactly what the shadow caches
    /// hold. Meant for tests and debugging; it locks every replica in turn,
    /// so writers must be quiet for the answer to mean anything.
    pub fn verify(&self) -> Result<(), String> {
        let cells = self.cells.read().unwrap_or_else(PoisonError::into_inner);
        let mut want: HashMap<u128, ReplicaSet> = HashMap::new();
        for (index, cell) in cells.iter().enumerate() {
            let cell = cell.lock().unwrap_or_else(PoisonError::into_inner);
            for hash in cell.shadow.hashes() {
                want.entry(hash).or_default().insert(index as u32);
            }
            if cell.shadow.len() > cell.shadow.capacity() {
                return Err(format!("slot {index} holds more than its capacity"));
            }
        }
        if want.len() != self.blocks.len() {
            return Err(format!(
                "the map has {} blocks, the shadow caches {}",
                self.blocks.len(),
                want.len()
            ));
        }
        for (hash, holders) in want {
            let found = self.blocks.holders(hash);
            if found != holders {
                return Err(format!(
                    "block {hash:032x}: map {found:?}, shadows {holders:?}"
                ));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use std::collections::{HashMap as Std, HashSet};

    fn id(i: usize) -> ReplicaId {
        ReplicaId::new(format!("r{i}")).unwrap()
    }

    fn chain(seed: u128, len: usize) -> Vec<BlockHash> {
        (0..len as u128)
            .map(|i| BlockHash::new(seed * 1_000 + i + 1))
            .collect()
    }

    #[test]
    fn longest_prefix_follows_recorded_routes() {
        let index = KvIndex::new();
        let (a, b, c) = (index.slot(&id(0)), index.slot(&id(1)), index.slot(&id(2)));
        let shared = chain(1, 10);
        index.record(a, &shared, 100);
        index.record(b, &shared[..4], 100);
        assert_eq!(index.longest_prefix(&shared, &[a, b, c]), [10, 4, 0]);
        assert_eq!(index.longest_prefix(&shared[..2], &[c, a]), [0, 2]);
        assert_eq!(index.longest_prefix(&[], &[a]), [0]);
        assert!(index.longest_prefix(&shared, &[]).is_empty());
        let mut other = shared.clone();
        other[5] = BlockHash::new(42);
        assert_eq!(index.longest_prefix(&other, &[a, b]), [5, 4]);
    }

    #[test]
    fn capacity_bounds_each_replica_and_eviction_clears_bits() {
        let index = KvIndex::new();
        let a = index.slot(&id(0));
        index.record(a, &chain(1, 8), 8);
        index.record(a, &chain(2, 4), 8);
        assert_eq!(index.longest_prefix(&chain(1, 8), &[a]), [4]);
        assert_eq!(index.longest_prefix(&chain(2, 4), &[a]), [4]);
        index.record(a, &chain(2, 4), 2);
        assert_eq!(index.stats().replicas[0].blocks, 2);
        assert_eq!(index.stats().entries, 2);
        index.record(a, &chain(3, 4), 0);
        assert_eq!(index.stats().entries, 0);
    }

    #[test]
    fn forgotten_slots_are_reused_and_stale_handles_do_nothing() {
        let index = KvIndex::new();
        let old = index.slot(&id(0));
        index.record(old, &chain(1, 3), 10);
        index.forget(&id(0));
        assert_eq!(index.stats().entries, 0);
        let fresh = index.slot(&id(7));
        assert_eq!(fresh.index(), old.index());
        index.record(old, &chain(2, 3), 10);
        index.clear(old);
        assert_eq!(index.stats().entries, 0);
        index.record(fresh, &chain(3, 3), 10);
        assert_eq!(index.longest_prefix(&chain(3, 3), &[fresh]), [3]);
        assert_eq!(index.replicas(), [id(7)]);
        index.forget(&id(99));
        index.clear(fresh);
        assert_eq!(index.longest_prefix(&chain(3, 3), &[fresh]), [0]);
    }

    #[test]
    fn revoking_takes_back_only_what_a_record_added() {
        let index = KvIndex::new();
        let a = index.slot(&id(0));
        let shared = chain(1, 6);
        assert_eq!(index.record(a, &shared[..4], 100), &shared[..4]);
        let fresh = index.record(a, &shared, 100);
        assert_eq!(fresh, &shared[4..]);
        index.revoke(a, &fresh);
        assert_eq!(index.longest_prefix(&shared, &[a]), [4]);
        assert_eq!(index.stats().replicas[0].blocks, 4);
        index.verify().unwrap();
        let long = chain(2, 10);
        let kept = index.record(a, &long, 3);
        assert_eq!(
            kept,
            &long[..3],
            "blocks evicted while recording are not returned"
        );
        index.forget(&id(0));
        let b = index.slot(&id(1));
        index.record(b, &long[..3], 10);
        index.revoke(a, &kept);
        assert_eq!(
            index.longest_prefix(&long, &[b]),
            [3],
            "a stale slot revokes nothing"
        );
        index.verify().unwrap();
    }

    #[derive(Debug, Clone)]
    enum Op {
        Record(usize, u128, usize, usize),
        Revoke(usize),
        Clear(usize),
        Forget(usize),
        Lookup(u128, usize),
    }

    fn op() -> impl Strategy<Value = Op> {
        prop_oneof![
            8 => (0usize..5, 0u128..4, 0usize..12, 0usize..20).prop_map(|(r, s, n, c)| Op::Record(r, s, n, c)),
            2 => (0usize..5).prop_map(Op::Revoke),
            1 => (0usize..5).prop_map(Op::Clear),
            1 => (0usize..5).prop_map(Op::Forget),
            4 => (0u128..4, 0usize..14).prop_map(|(s, n)| Op::Lookup(s, n)),
        ]
    }

    /// Per replica, blocks most recent first, as an engine LRU would hold them.
    #[derive(Default)]
    struct Naive(Std<usize, (Vec<u128>, usize)>);

    impl Naive {
        fn record(&mut self, r: usize, chain: &[u128], capacity: usize) {
            let (order, cap) = self.0.entry(r).or_default();
            *cap = capacity;
            order.truncate(capacity);
            if capacity == 0 {
                return;
            }
            for &h in chain.iter().rev() {
                if !order.contains(&h) && order.len() >= capacity {
                    order.pop();
                }
                order.retain(|&x| x != h);
                order.insert(0, h);
            }
        }

        fn holds(&self, r: usize, h: u128) -> bool {
            self.0.get(&r).is_some_and(|(order, _)| order.contains(&h))
        }
    }

    proptest! {
        /// Invariants 3 and 4: the shared map always equals the union of the
        /// shadow caches, and a reported prefix of depth d means the replica
        /// holds blocks 0 to d - 1 and not block d.
        #[test]
        fn matches_a_naive_reference_model(ops in prop::collection::vec(op(), 0..120)) {
            let index = KvIndex::new();
            let mut naive = Naive::default();
            let mut receipts: Std<usize, (Slot, Vec<BlockHash>)> = Std::new();
            for op in &ops {
                match *op {
                    Op::Record(r, seed, len, capacity) => {
                        let hashes = chain(seed, len);
                        let slot = index.slot(&id(r));
                        let before: HashSet<u128> = naive
                            .0
                            .get(&r)
                            .map(|(order, _)| order.iter().take(capacity).copied().collect())
                            .unwrap_or_default();
                        let fresh = index.record(slot, &hashes, capacity);
                        let raw: Vec<u128> = hashes.iter().map(|h| h.get()).collect();
                        naive.record(r, &raw, capacity);
                        let want: Vec<u128> = raw.iter().copied().filter(|h| !before.contains(h) && naive.holds(r, *h)).collect();
                        let mut got: Vec<u128> = fresh.iter().map(|h| h.get()).collect();
                        got.sort_unstable();
                        let mut want = want;
                        want.sort_unstable();
                        want.dedup();
                        prop_assert_eq!(got, want);
                        receipts.insert(r, (slot, fresh));
                    }
                    Op::Revoke(r) => {
                        if let Some((slot, fresh)) = receipts.remove(&r) {
                            index.revoke(slot, &fresh);
                            if index.find(&id(r)) == Some(slot)
                                && let Some((order, _)) = naive.0.get_mut(&r)
                            {
                                order.retain(|h| !fresh.iter().any(|f| f.get() == *h));
                            }
                        }
                    }
                    Op::Clear(r) => {
                        if let Some(slot) = index.find(&id(r)) {
                            index.clear(slot);
                        }
                        if let Some((order, _)) = naive.0.get_mut(&r) {
                            order.clear();
                        }
                    }
                    Op::Forget(r) => {
                        index.forget(&id(r));
                        naive.0.remove(&r);
                    }
                    Op::Lookup(seed, len) => {
                        let hashes = chain(seed, len);
                        let known: Vec<usize> = (0..5).filter(|r| index.find(&id(*r)).is_some()).collect();
                        let slots: Vec<Slot> = known.iter().map(|r| index.find(&id(*r)).unwrap()).collect();
                        let got = index.longest_prefix(&hashes, &slots);
                        for (r, depth) in known.iter().zip(&got) {
                            let want = hashes.iter().take_while(|h| naive.holds(*r, h.get())).count();
                            prop_assert_eq!(*depth, want);
                            prop_assert!(hashes[..*depth].iter().all(|h| naive.holds(*r, h.get())));
                            if *depth < hashes.len() {
                                prop_assert!(!naive.holds(*r, hashes[*depth].get()));
                            }
                        }
                    }
                }
                let universe: HashSet<u128> = (0..4u128).flat_map(|s| (0..14u128).map(move |i| s * 1_000 + i + 1)).collect();
                for h in universe {
                    let want: HashSet<u32> = naive
                        .0
                        .keys()
                        .filter(|r| naive.holds(**r, h))
                        .map(|r| index.find(&id(*r)).unwrap().index())
                        .collect();
                    prop_assert_eq!(index.holders(h).iter().collect::<HashSet<_>>(), want);
                }
                prop_assert_eq!(index.verify(), Ok(()));
                for stats in index.stats().replicas {
                    let r: usize = stats.replica.as_str()[1..].parse().unwrap();
                    let (order, capacity) = naive.0.get(&r).cloned().unwrap_or_default();
                    prop_assert_eq!(stats.blocks, order.len());
                    prop_assert!(stats.blocks <= capacity.max(stats.capacity));
                }
            }
        }
    }
}
