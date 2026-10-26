//! Reuse of tokenization across requests.
//!
//! A tokenizer first cuts the text at every added token matched before
//! normalization, then normalizes, pre tokenizes and tokenizes each piece on
//! its own. A piece seen before, such as a shared system prompt or an
//! earlier turn of a conversation, therefore always yields the same ids and
//! is served from memory. Pipelines where a piece's ids could depend on its
//! position (a Metaspace that only prepends to the first piece, added tokens
//! that strip neighboring spaces, special tokens split as text) are never
//! cached.

use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, PoisonError};

use daachorse::{DoubleArrayAhoCorasick, DoubleArrayAhoCorasickBuilder, MatchKind};
use tokenizers::Tokenizer;
use xxhash_rust::xxh3::xxh3_128;

use crate::TokenizeError;

const MIN_CACHED_BYTES: usize = 256;
const SHARDS: usize = 16;

#[derive(Default)]
struct Identity(u64);

impl Hasher for Identity {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, bytes: &[u8]) {
        for chunk in bytes.chunks(8) {
            let mut word = [0u8; 8];
            word[..chunk.len()].copy_from_slice(chunk);
            self.0 ^= u64::from_le_bytes(word);
        }
    }

    fn write_u128(&mut self, value: u128) {
        self.0 = value as u64;
    }
}

type Map = HashMap<u128, Arc<[u32]>, BuildHasherDefault<Identity>>;

#[derive(Default)]
struct Shard {
    current: Map,
    previous: Map,
    tokens: usize,
}

/// Hit and miss counts of the segment cache since startup.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
}

pub(crate) struct Segmenter {
    trie: Option<DoubleArrayAhoCorasick<u32>>,
    shards: Box<[Mutex<Shard>]>,
    budget: usize,
    hits: AtomicU64,
    misses: AtomicU64,
}

impl std::fmt::Debug for Segmenter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Segmenter")
            .field("caching", &self.caching())
            .field("budget", &self.budget)
            .finish_non_exhaustive()
    }
}

fn first_only_metaspace(value: &serde_json::Value) -> bool {
    match value {
        serde_json::Value::Object(map) => {
            (map.get("type").and_then(|t| t.as_str()) == Some("Metaspace")
                && map.get("prepend_scheme").and_then(|s| s.as_str()) == Some("first"))
                || map.values().any(first_only_metaspace)
        }
        serde_json::Value::Array(items) => items.iter().any(first_only_metaspace),
        _ => false,
    }
}

impl Segmenter {
    /// `budget` is how many token ids the cache may hold; zero disables it.
    pub(crate) fn new(tokenizer: &Tokenizer, budget: usize) -> Self {
        let disabled = Self {
            trie: None,
            shards: Box::new([]),
            budget: 0,
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
        };
        if budget == 0 || tokenizer.get_encode_special_tokens() {
            return disabled;
        }
        let pre = tokenizer
            .get_pre_tokenizer()
            .and_then(|p| serde_json::to_value(p).ok())
            .unwrap_or_default();
        if first_only_metaspace(&pre) {
            return disabled;
        }
        let added = tokenizer.get_added_tokens_decoder();
        let raw: Vec<(String, u32)> = added
            .iter()
            .filter(|(_, t)| !t.normalized)
            .map(|(id, t)| (t.content.clone(), *id))
            .collect();
        if added
            .values()
            .any(|t| !t.normalized && (t.lstrip || t.rstrip || t.single_word))
        {
            return disabled;
        }
        let trie = if raw.is_empty() {
            None
        } else {
            match DoubleArrayAhoCorasickBuilder::new()
                .match_kind(MatchKind::LeftmostLongest)
                .build_with_values(raw)
            {
                Ok(trie) => Some(trie),
                Err(_) => return disabled,
            }
        };
        Self {
            trie,
            shards: (0..SHARDS).map(|_| Mutex::new(Shard::default())).collect(),
            budget: budget.div_ceil(SHARDS),
            ..disabled
        }
    }

    pub(crate) fn caching(&self) -> bool {
        !self.shards.is_empty()
    }

    pub(crate) fn stats(&self) -> CacheStats {
        CacheStats {
            hits: self.hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
        }
    }

    /// Tokenizes `text`, counting this call's cache hits and misses in
    /// `stats` as well as in the totals.
    pub(crate) fn encode(
        &self,
        tokenizer: &Tokenizer,
        text: &str,
        add_special_tokens: bool,
        stats: &mut CacheStats,
    ) -> Result<Vec<u32>, TokenizeError> {
        if add_special_tokens || !self.caching() {
            return direct(tokenizer, text, add_special_tokens);
        }
        let mut out = Vec::with_capacity(text.len() / 3 + 8);
        let mut start = 0;
        if let Some(trie) = &self.trie {
            for found in trie.leftmost_find_iter(text) {
                if start < found.start() {
                    self.segment(tokenizer, &text[start..found.start()], &mut out, stats)?;
                }
                out.push(found.value());
                start = found.end();
            }
        }
        if start < text.len() {
            self.segment(tokenizer, &text[start..], &mut out, stats)?;
        }
        Ok(out)
    }

    fn segment(
        &self,
        tokenizer: &Tokenizer,
        piece: &str,
        out: &mut Vec<u32>,
        stats: &mut CacheStats,
    ) -> Result<(), TokenizeError> {
        if piece.len() < MIN_CACHED_BYTES {
            out.extend(direct(tokenizer, piece, false)?);
            return Ok(());
        }
        let key = xxh3_128(piece.as_bytes());
        let shard = &self.shards[(key >> 124) as usize % SHARDS];
        if let Some(ids) = lookup(shard, key) {
            self.hits.fetch_add(1, Ordering::Relaxed);
            stats.hits += 1;
            out.extend_from_slice(&ids);
            return Ok(());
        }
        self.misses.fetch_add(1, Ordering::Relaxed);
        stats.misses += 1;
        let ids: Arc<[u32]> = direct(tokenizer, piece, false)?.into();
        out.extend_from_slice(&ids);
        if ids.len() <= self.budget {
            let mut shard = shard.lock().unwrap_or_else(PoisonError::into_inner);
            if shard.tokens + ids.len() > self.budget {
                shard.previous = std::mem::take(&mut shard.current);
                shard.tokens = 0;
            }
            shard.tokens += ids.len();
            shard.current.insert(key, ids);
        }
        Ok(())
    }
}

fn lookup(shard: &Mutex<Shard>, key: u128) -> Option<Arc<[u32]>> {
    let mut shard = shard.lock().unwrap_or_else(PoisonError::into_inner);
    if let Some(ids) = shard.current.get(&key) {
        return Some(Arc::clone(ids));
    }
    let ids = shard.previous.remove(&key)?;
    shard.tokens += ids.len();
    shard.current.insert(key, Arc::clone(&ids));
    Some(ids)
}

fn direct(
    tokenizer: &Tokenizer,
    text: &str,
    add_special_tokens: bool,
) -> Result<Vec<u32>, TokenizeError> {
    tokenizer
        .encode_fast(text, add_special_tokens)
        .map(|e| e.get_ids().to_vec())
        .map_err(|e| TokenizeError::Tokenizer(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tokenizer(pre_tokenizer: serde_json::Value, lstrip: bool) -> Tokenizer {
        let vocab: serde_json::Map<String, serde_json::Value> =
            ["<s>", "</s>", "<unk>", "▁a", "▁b", "a", "b", "▁"]
                .iter()
                .enumerate()
                .map(|(i, t)| ((*t).to_owned(), serde_json::json!(i)))
                .collect();
        let json = serde_json::json!({
            "version": "1.0", "truncation": null, "padding": null,
            "added_tokens": [
                {"id": 0, "content": "<s>", "single_word": false, "lstrip": lstrip, "rstrip": false, "normalized": false, "special": true},
                {"id": 1, "content": "</s>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}
            ],
            "normalizer": null, "pre_tokenizer": pre_tokenizer, "post_processor": null, "decoder": null,
            "model": {"type": "BPE", "dropout": null, "unk_token": "<unk>", "continuing_subword_prefix": null,
                      "end_of_word_suffix": null, "fuse_unk": false, "byte_fallback": false, "ignore_merges": false,
                      "vocab": vocab, "merges": []}
        });
        Tokenizer::from_bytes(json.to_string().as_bytes()).unwrap()
    }

    fn metaspace(scheme: &str) -> serde_json::Value {
        serde_json::json!({"type": "Metaspace", "replacement": "▁", "prepend_scheme": scheme, "split": true})
    }

    #[test]
    fn position_dependent_pipelines_are_never_cached() {
        assert!(Segmenter::new(&tokenizer(metaspace("always"), false), 1000).caching());
        assert!(!Segmenter::new(&tokenizer(metaspace("first"), false), 1000).caching());
        let nested = serde_json::json!({"type": "Sequence", "pretokenizers": [metaspace("first")]});
        assert!(!Segmenter::new(&tokenizer(nested, false), 1000).caching());
        assert!(!Segmenter::new(&tokenizer(metaspace("always"), true), 1000).caching());
        assert!(!Segmenter::new(&tokenizer(metaspace("always"), false), 0).caching());
        let mut split = tokenizer(metaspace("always"), false);
        split.set_encode_special_tokens(true);
        assert!(!Segmenter::new(&split, 1000).caching());
    }

    #[test]
    fn cached_pieces_tokenize_exactly_like_the_whole_text() {
        let tok = tokenizer(metaspace("always"), false);
        let segmenter = Segmenter::new(&tok, 1 << 20);
        let long = "a b ".repeat(100);
        let text = format!("<s>{long}</s>{long}<s>b a</s>");
        let want = direct(&tok, &text, false).unwrap();
        let (mut first, mut second) = (CacheStats::default(), CacheStats::default());
        assert_eq!(
            segmenter.encode(&tok, &text, false, &mut first).unwrap(),
            want
        );
        assert_eq!(
            segmenter.encode(&tok, &text, false, &mut second).unwrap(),
            want
        );
        assert_eq!(
            (first.misses, first.hits, second.misses, second.hits),
            (1, 1, 0, 2)
        );
        let stats = segmenter.stats();
        assert_eq!((stats.misses, stats.hits), (1, 3));
        assert_eq!(
            segmenter
                .encode(&tok, &text, true, &mut CacheStats::default())
                .unwrap(),
            direct(&tok, &text, true).unwrap()
        );
    }

    #[test]
    fn the_budget_bounds_what_is_kept() {
        let tok = tokenizer(metaspace("always"), false);
        let segmenter = Segmenter::new(&tok, SHARDS * 300);
        for i in 0..200 {
            let text = format!("{} {}", "a ".repeat(150), "b ".repeat(i % 50 + 1));
            assert_eq!(
                segmenter
                    .encode(&tok, &text, false, &mut CacheStats::default())
                    .unwrap(),
                direct(&tok, &text, false).unwrap()
            );
        }
        for shard in segmenter.shards.iter() {
            let shard = shard.lock().unwrap();
            let held: usize = shard
                .current
                .values()
                .chain(shard.previous.values())
                .map(|v| v.len())
                .sum();
            assert!(held <= 2 * segmenter.budget, "{held}");
        }
    }
}
