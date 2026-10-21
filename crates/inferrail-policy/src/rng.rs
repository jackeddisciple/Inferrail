use std::sync::atomic::{AtomicU64, Ordering};

const GAMMA: u64 = 0x9e37_79b9_7f4a_7c15;

/// SplitMix64 on an atomic counter: lock free, and for sequential callers the
/// stream is fully determined by the seed.
#[derive(Debug)]
pub struct SplitMix(AtomicU64);

impl SplitMix {
    pub fn new(seed: u64) -> Self {
        Self(AtomicU64::new(seed))
    }

    pub fn next_u64(&self) -> u64 {
        let mut z = self
            .0
            .fetch_add(GAMMA, Ordering::Relaxed)
            .wrapping_add(GAMMA);
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Uniform in `0..n` by multiply and shift; `n` must be positive.
    pub fn below(&self, n: usize) -> usize {
        ((u128::from(self.next_u64()) * n as u128) >> 64) as usize
    }
}

/// Derives an independent stream seed for one label, such as a model name.
pub fn mix_seed(seed: u64, label: &str) -> u64 {
    let hash = label.bytes().fold(0xcbf2_9ce4_8422_2325u64, |h, b| {
        (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3)
    });
    SplitMix::new(seed ^ hash).next_u64()
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn known_answers_match_the_reference_splitmix64() {
        let rng = SplitMix::new(0);
        assert_eq!(rng.next_u64(), 0xe220_a839_7b1d_cdaf);
        assert_eq!(rng.next_u64(), 0x6e78_9e6a_a1b9_65f4);
        assert_eq!(rng.next_u64(), 0x06c4_5d18_8009_454f);
    }

    #[test]
    fn same_seed_same_stream() {
        let (a, b, c) = (SplitMix::new(7), SplitMix::new(7), SplitMix::new(8));
        let first: Vec<u64> = (0..64).map(|_| a.next_u64()).collect();
        let second: Vec<u64> = (0..64).map(|_| b.next_u64()).collect();
        let other: Vec<u64> = (0..64).map(|_| c.next_u64()).collect();
        assert_eq!(first, second);
        assert_ne!(first, other);
    }

    #[test]
    fn concurrent_draws_never_repeat_a_state() {
        let rng = std::sync::Arc::new(SplitMix::new(1));
        let handles: Vec<_> = (0..8)
            .map(|_| {
                let rng = std::sync::Arc::clone(&rng);
                std::thread::spawn(move || (0..5_000).map(|_| rng.next_u64()).collect::<Vec<_>>())
            })
            .collect();
        let mut all: Vec<u64> = handles
            .into_iter()
            .flat_map(|h| h.join().unwrap())
            .collect();
        all.sort_unstable();
        all.dedup();
        assert_eq!(all.len(), 40_000);
    }

    proptest! {
        #[test]
        fn below_stays_in_range(seed: u64, n in 1usize..10_000) {
            let rng = SplitMix::new(seed);
            for _ in 0..32 {
                prop_assert!(rng.below(n) < n);
            }
        }
    }
}
