//! The engine's own chained block hashes.
//!
//! The hash is deliberately not the router's: the router must never depend
//! on engine hash values, so the simulator uses its own 64 bit chain.

use xxhash_rust::xxh3::xxh3_64_with_seed;

pub type BlockHash = u64;

const ROOT: BlockHash = 0x9e37_79b9_7f4a_7c15;

/// Hashes every full block; a trailing partial block is never cached.
pub fn block_hashes(tokens: &[u32], block_size: usize) -> Vec<BlockHash> {
    let mut hashes = Vec::with_capacity(tokens.len() / block_size.max(1));
    extend_hashes(tokens, block_size, &mut hashes);
    hashes
}

/// Hashes the full blocks of `tokens` past the ones `hashes` already
/// covers, chaining from the last of them.
pub fn extend_hashes(tokens: &[u32], block_size: usize, hashes: &mut Vec<BlockHash>) {
    assert!(block_size > 0, "block size must be positive");
    let done = hashes.len() * block_size;
    if tokens.len() < done + block_size {
        return;
    }
    let mut bytes = Vec::with_capacity(block_size * 4);
    let mut parent = hashes.last().copied().unwrap_or(ROOT);
    for block in tokens[done..].chunks_exact(block_size) {
        bytes.clear();
        for token in block {
            bytes.extend_from_slice(&token.to_le_bytes());
        }
        parent = xxh3_64_with_seed(&bytes, parent);
        hashes.push(parent);
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

    proptest! {
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
        fn extending_matches_hashing_at_once(
            tokens in proptest::collection::vec(any::<u32>(), 0..200),
            cuts in proptest::collection::vec(any::<prop::sample::Index>(), 0..6),
            block_size in 1usize..33,
        ) {
            let mut cuts: Vec<usize> = cuts.iter().map(|c| c.index(tokens.len() + 1)).collect();
            cuts.sort_unstable();
            let mut hashes = Vec::new();
            for cut in cuts {
                extend_hashes(&tokens[..cut], block_size, &mut hashes);
            }
            extend_hashes(&tokens, block_size, &mut hashes);
            prop_assert_eq!(hashes, block_hashes(&tokens, block_size));
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
