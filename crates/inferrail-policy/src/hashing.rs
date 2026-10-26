//! Stable hashes of replicas and prompt prefixes, so the same prefix maps
//! to the same replicas on every gateway and across restarts.

use inferrail_core::ids::{BlockHash, ReplicaId};
use xxhash_rust::xxh3::xxh3_64_with_seed;

/// The key of a prompt's first `blocks` blocks: the hash of the last of
/// them, which names the whole prefix since hashes chain. A prompt with
/// fewer full blocks uses all it has; one with none has no key.
pub(crate) fn prefix_key(hashes: &[BlockHash], blocks: u32) -> Option<u64> {
    let take = hashes.len().min(blocks as usize);
    let last = hashes.get(take.checked_sub(1)?)?.get();
    Some(last as u64 ^ (last >> 64) as u64)
}

/// Rendezvous hashing: the replica with the highest weight for a key owns
/// it, and removing a replica only moves the keys it owned.
pub(crate) fn weight(key: u64, replica: &ReplicaId) -> u64 {
    xxh3_64_with_seed(replica.as_str().as_bytes(), key)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chain(len: u128) -> Vec<BlockHash> {
        (1..=len)
            .map(|i| BlockHash::new((i << 64) | (i * 7)))
            .collect()
    }

    #[test]
    fn the_key_is_the_last_of_the_leading_blocks() {
        let hashes = chain(10);
        assert_eq!(prefix_key(&hashes, 4), prefix_key(&hashes[..4], 4));
        assert_eq!(prefix_key(&hashes, 4), prefix_key(&hashes[..6], 4));
        assert_ne!(prefix_key(&hashes, 4), prefix_key(&hashes, 5));
        assert_eq!(prefix_key(&hashes[..2], 4), prefix_key(&hashes[..2], 2));
        assert_eq!(prefix_key(&[], 4), None);
        assert_eq!(prefix_key(&hashes, 0), None);
    }

    #[test]
    fn weights_spread_keys_evenly() {
        let ids: Vec<ReplicaId> = (0..5)
            .map(|i| ReplicaId::new(format!("gpu-{i}")).unwrap())
            .collect();
        let mut owned = [0u64; 5];
        for key in 0..50_000u64 {
            let owner = (0..5).max_by_key(|&i| weight(key, &ids[i])).unwrap();
            owned[owner] += 1;
        }
        let stat = crate::testing::chi_square(&owned);
        assert!(stat < 18.47, "chi square {stat} for {owned:?}");
    }
}
