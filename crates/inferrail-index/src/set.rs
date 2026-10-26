/// Replicas by slot. One word inline covers 64 replicas, the common fleet
/// size; larger fleets spill to a boxed slice of words.
#[derive(Clone)]
pub struct ReplicaSet(Words);

#[derive(Clone)]
enum Words {
    Inline(u64),
    Heap(Box<[u64]>),
}

/// Sets are equal when they hold the same slots, however they are stored.
impl PartialEq for ReplicaSet {
    fn eq(&self, other: &Self) -> bool {
        let (a, b) = (self.words(), other.words());
        let (short, long) = if a.len() <= b.len() { (a, b) } else { (b, a) };
        short == &long[..short.len()] && long[short.len()..].iter().all(|w| *w == 0)
    }
}

impl Eq for ReplicaSet {}

impl Default for ReplicaSet {
    fn default() -> Self {
        Self(Words::Inline(0))
    }
}

impl std::fmt::Debug for ReplicaSet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_set().entries(self.iter()).finish()
    }
}

impl ReplicaSet {
    fn words(&self) -> &[u64] {
        match &self.0 {
            Words::Inline(word) => std::slice::from_ref(word),
            Words::Heap(words) => words,
        }
    }

    fn words_mut(&mut self, len: usize) -> &mut [u64] {
        if len > self.words().len() {
            let mut grown = vec![0u64; len.next_power_of_two()];
            grown[..self.words().len()].copy_from_slice(self.words());
            self.0 = Words::Heap(grown.into_boxed_slice());
        }
        match &mut self.0 {
            Words::Inline(word) => std::slice::from_mut(word),
            Words::Heap(words) => words,
        }
    }

    pub fn insert(&mut self, slot: u32) -> bool {
        let (word, bit) = (slot as usize / 64, 1u64 << (slot % 64));
        let words = self.words_mut(word + 1);
        let fresh = words[word] & bit == 0;
        words[word] |= bit;
        fresh
    }

    pub fn remove(&mut self, slot: u32) -> bool {
        let (word, bit) = (slot as usize / 64, 1u64 << (slot % 64));
        let words = match &mut self.0 {
            Words::Inline(word) => std::slice::from_mut(word),
            Words::Heap(words) => words,
        };
        match words.get_mut(word) {
            Some(w) if *w & bit != 0 => {
                *w &= !bit;
                true
            }
            _ => false,
        }
    }

    pub fn contains(&self, slot: u32) -> bool {
        self.words()
            .get(slot as usize / 64)
            .is_some_and(|w| w & (1 << (slot % 64)) != 0)
    }

    pub fn is_empty(&self) -> bool {
        self.words().iter().all(|w| *w == 0)
    }

    pub fn len(&self) -> usize {
        self.words().iter().map(|w| w.count_ones() as usize).sum()
    }

    pub fn clear(&mut self) {
        self.0 = Words::Inline(0);
    }

    /// Keeps only the members also in `other`, calling `dropped` for each
    /// member that leaves.
    pub fn retain_in(&mut self, other: &Self, mut dropped: impl FnMut(u32)) {
        let theirs = other.words();
        let words = match &mut self.0 {
            Words::Inline(word) => std::slice::from_mut(word),
            Words::Heap(words) => words,
        };
        for (i, word) in words.iter_mut().enumerate() {
            let keep = theirs.get(i).copied().unwrap_or(0);
            let mut gone = *word & !keep;
            while gone != 0 {
                dropped(i as u32 * 64 + gone.trailing_zeros());
                gone &= gone - 1;
            }
            *word &= keep;
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = u32> + '_ {
        self.words().iter().enumerate().flat_map(|(i, &word)| {
            let mut rest = word;
            std::iter::from_fn(move || {
                (rest != 0).then(|| {
                    let bit = rest.trailing_zeros();
                    rest &= rest - 1;
                    i as u32 * 64 + bit
                })
            })
        })
    }

    /// Heap bytes beyond the inline word, for memory accounting.
    pub fn heap_bytes(&self) -> usize {
        match &self.0 {
            Words::Inline(_) => 0,
            Words::Heap(words) => words.len() * 8,
        }
    }
}

impl FromIterator<u32> for ReplicaSet {
    fn from_iter<I: IntoIterator<Item = u32>>(iter: I) -> Self {
        let mut set = Self::default();
        for slot in iter {
            set.insert(slot);
        }
        set
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use std::collections::BTreeSet;

    #[test]
    fn small_sets_stay_inline_and_large_ones_spill() {
        let mut set = ReplicaSet::default();
        assert!(set.insert(3) && !set.insert(3) && set.insert(63));
        assert_eq!(set.heap_bytes(), 0);
        assert!(set.insert(200));
        assert!(set.heap_bytes() >= 32);
        assert_eq!(set.iter().collect::<Vec<_>>(), [3, 63, 200]);
        assert!(set.remove(200) && !set.remove(200) && !set.remove(9_999));
        assert_eq!(set.len(), 2);
        assert_eq!(
            set,
            ReplicaSet::from_iter([3, 63]),
            "spilled and inline sets compare by members"
        );
        assert_ne!(set, ReplicaSet::from_iter([3]));
        set.clear();
        assert!(set.is_empty() && set.heap_bytes() == 0);
    }

    proptest! {
        #[test]
        fn behaves_like_a_set_of_slots(
            a in prop::collection::btree_set(0u32..300, 0..40),
            b in prop::collection::btree_set(0u32..300, 0..40),
            gone in prop::collection::vec(0u32..300, 0..20),
        ) {
            let mut set: ReplicaSet = a.iter().copied().collect();
            let mut want = a.clone();
            for slot in &gone {
                prop_assert_eq!(set.remove(*slot), want.remove(slot));
            }
            prop_assert_eq!(set.iter().collect::<BTreeSet<_>>(), want.clone());
            prop_assert_eq!(set.len(), want.len());
            let other: ReplicaSet = b.iter().copied().collect();
            let mut dropped = BTreeSet::new();
            set.retain_in(&other, |s| { dropped.insert(s); });
            let kept: BTreeSet<u32> = want.intersection(&b).copied().collect();
            prop_assert_eq!(set.iter().collect::<BTreeSet<_>>(), kept.clone());
            prop_assert_eq!(dropped, want.difference(&kept).copied().collect::<BTreeSet<_>>());
            for slot in 0..300 {
                prop_assert_eq!(set.contains(slot), kept.contains(&slot));
            }
        }
    }
}
