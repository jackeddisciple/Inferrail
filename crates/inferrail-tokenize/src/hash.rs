//! Chained XXH3 128 bit hashes over full blocks of token ids.
//!
//! Block 0 hashes the salt followed by its tokens, and every later block
//! hashes the previous block's hash followed by its tokens, so one hash
//! names a whole prefix. A trailing partial block is never hashed, because
//! engines never cache one. Tokens and parent hashes are little endian.

use inferrail_core::ids::BlockHash;
use xxhash_rust::xxh3::xxh3_128;

const DOMAIN: &[u8] = b"inferrail kv block v1";
const PARENT: usize = 16;

/// What keeps identical token ids apart: the model, the LoRA adapter and
/// vLLM's per request `cache_salt`. Fields are length prefixed, so no two
/// distinct salts share bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Salt(u128);

impl Salt {
    pub fn new(model: &str, adapter: Option<&str>, cache_salt: Option<&str>) -> Self {
        let len = DOMAIN.len()
            + 27
            + model.len()
            + adapter.map_or(0, str::len)
            + cache_salt.map_or(0, str::len);
        let mut bytes = Vec::with_capacity(len);
        bytes.extend_from_slice(DOMAIN);
        for field in [Some(model), adapter, cache_salt] {
            match field {
                None => bytes.push(0),
                Some(text) => {
                    bytes.push(1);
                    bytes.extend_from_slice(&(text.len() as u64).to_le_bytes());
                    bytes.extend_from_slice(text.as_bytes());
                }
            }
        }
        Self(xxh3_128(&bytes))
    }

    pub const fn get(self) -> u128 {
        self.0
    }
}

/// Hashes prompts into blocks of a fixed size, reusing one buffer.
#[derive(Debug, Clone)]
pub struct BlockHasher {
    block_size: usize,
    buf: Vec<u8>,
}

impl BlockHasher {
    /// # Panics
    ///
    /// When `block_size` is zero; configuration never allows one.
    pub fn new(block_size: usize) -> Self {
        assert!(block_size > 0, "block size must be positive");
        Self {
            block_size,
            buf: vec![0; PARENT + 4 * block_size],
        }
    }

    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// Appends the hash of every full block of `tokens` to `out`.
    pub fn hash_into(&mut self, tokens: &[u32], salt: Salt, out: &mut Vec<BlockHash>) {
        out.reserve(tokens.len() / self.block_size);
        let mut parent = salt.0;
        for block in tokens.chunks_exact(self.block_size) {
            self.buf[..PARENT].copy_from_slice(&parent.to_le_bytes());
            for (slot, token) in self.buf[PARENT..]
                .as_chunks_mut::<4>()
                .0
                .iter_mut()
                .zip(block)
            {
                slot.copy_from_slice(&token.to_le_bytes());
            }
            parent = xxh3_128(&self.buf);
            out.push(BlockHash::new(parent));
        }
    }

    pub fn hash(&mut self, tokens: &[u32], salt: Salt) -> Vec<BlockHash> {
        let mut out = Vec::new();
        self.hash_into(tokens, salt, &mut out);
        out
    }
}

/// Blocks an engine can serve from cache for a prompt of `len` tokens: it
/// always computes the last token itself, so a prompt that is entirely
/// cached still recomputes its final block.
pub const fn cacheable_blocks(len: usize, block_size: usize) -> usize {
    if block_size == 0 {
        return 0;
    }
    len.saturating_sub(1) / block_size
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn salt() -> Salt {
        Salt::new("Qwen/Qwen3-8B", None, None)
    }

    #[test]
    fn only_full_blocks_are_hashed() {
        let mut hasher = BlockHasher::new(4);
        assert!(hasher.hash(&[1, 2, 3], salt()).is_empty());
        assert_eq!(hasher.hash(&[1, 2, 3, 4, 5], salt()).len(), 1);
        assert_eq!(BlockHasher::new(16).hash(&[0; 64], salt()).len(), 4);
        assert_eq!(hasher.block_size(), 4);
    }

    /// Computed independently with the Python `xxhash` package from the
    /// layout in the module docs, so a change to either side shows up.
    #[test]
    fn known_answers_never_change() {
        let tokens: Vec<u32> = (0..48).collect();
        let hashes = BlockHasher::new(16).hash(&tokens, salt());
        let hex: Vec<String> = hashes.iter().map(ToString::to_string).collect();
        assert_eq!(salt().get(), 0x21ec_9ab5_cf5d_18fe_e5f3_a6a3_b8fa_3f4a);
        assert_eq!(
            hex,
            [
                "1379105aab2b0c0c1b70d8f7301b88ac",
                "16107e4f9ce2e34c26016cfc514c97d5",
                "75a3fa76a6194a7adf08e775630fce03",
            ]
        );
    }

    #[test]
    fn salts_keep_models_adapters_and_tenants_apart() {
        let tokens = [7u32; 16];
        let mut hasher = BlockHasher::new(16);
        let salts = [
            Salt::new("m", None, None),
            Salt::new("n", None, None),
            Salt::new("m", Some("lora"), None),
            Salt::new("m", None, Some("tenant")),
            Salt::new("m", Some(""), None),
            Salt::new("ab", Some("c"), None),
            Salt::new("a", Some("bc"), None),
            Salt::new("m", Some("x"), Some("y")),
            Salt::new("m", Some("y"), Some("x")),
        ];
        let hashes: Vec<BlockHash> = salts.iter().map(|&s| hasher.hash(&tokens, s)[0]).collect();
        for (i, a) in hashes.iter().enumerate() {
            for b in &hashes[i + 1..] {
                assert_ne!(a, b);
            }
        }
    }

    #[test]
    fn identical_blocks_at_different_depths_differ() {
        let hashes = BlockHasher::new(16).hash(&[9; 64], salt());
        let unique: std::collections::HashSet<_> = hashes.iter().collect();
        assert_eq!(unique.len(), 4);
    }

    #[test]
    fn engines_recompute_the_last_token() {
        assert_eq!(cacheable_blocks(0, 16), 0);
        assert_eq!(cacheable_blocks(16, 16), 0);
        assert_eq!(cacheable_blocks(17, 16), 1);
        assert_eq!(cacheable_blocks(33, 16), 2);
        assert_eq!(cacheable_blocks(32, 16), 1);
        assert_eq!(cacheable_blocks(10, 0), 0);
    }

    #[test]
    #[should_panic(expected = "block size must be positive")]
    fn zero_block_size_is_refused() {
        let _ = BlockHasher::new(0);
    }

    proptest! {
        #[test]
        fn changing_a_token_changes_its_block_and_every_later_one(
            tokens in prop::collection::vec(any::<u32>(), 1..300),
            at in any::<prop::sample::Index>(),
            delta in 1u32..,
            block_size in 1usize..33,
        ) {
            let mut hasher = BlockHasher::new(block_size);
            let before = hasher.hash(&tokens, salt());
            let mut changed = tokens.clone();
            let i = at.index(tokens.len());
            changed[i] = changed[i].wrapping_add(delta);
            let after = hasher.hash(&changed, salt());
            prop_assert_eq!(before.len(), after.len());
            for j in 0..before.len() {
                if j < i / block_size {
                    prop_assert_eq!(before[j], after[j]);
                } else {
                    prop_assert_ne!(before[j], after[j]);
                }
            }
        }

        #[test]
        fn a_prefix_hashes_the_same_alone_or_inside_a_longer_prompt(
            tokens in prop::collection::vec(any::<u32>(), 0..400),
            cut in any::<prop::sample::Index>(),
            block_size in 1usize..65,
        ) {
            let n = cut.index(tokens.len() + 1);
            let mut hasher = BlockHasher::new(block_size);
            let full = hasher.hash(&tokens, salt());
            let head = hasher.hash(&tokens[..n], salt());
            prop_assert_eq!(head.len(), n / block_size);
            prop_assert_eq!(&full[..head.len()], head.as_slice());
        }

        #[test]
        fn hashing_into_appends_and_reuses_the_buffer(
            a in prop::collection::vec(any::<u32>(), 0..100),
            b in prop::collection::vec(any::<u32>(), 0..100),
        ) {
            let mut hasher = BlockHasher::new(8);
            let mut out = Vec::new();
            hasher.hash_into(&a, salt(), &mut out);
            hasher.hash_into(&b, salt(), &mut out);
            let mut fresh = BlockHasher::new(8);
            let mut want = fresh.hash(&a, salt());
            want.extend(fresh.hash(&b, salt()));
            prop_assert_eq!(out, want);
        }
    }
}
