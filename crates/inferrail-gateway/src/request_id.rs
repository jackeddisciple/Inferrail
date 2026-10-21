//! Request IDs: keep a well formed client `x-request-id`, otherwise mint one.

use std::hash::{BuildHasher, RandomState};
use std::sync::atomic::{AtomicU64, Ordering};

use hyper::header::{HeaderMap, HeaderName, HeaderValue};
use xxhash_rust::xxh3::xxh3_128;

pub const REQUEST_ID: HeaderName = HeaderName::from_static("x-request-id");
pub const MAX_LEN: usize = 256;

#[derive(Debug)]
pub struct RequestIds {
    seed: u64,
    counter: AtomicU64,
}

impl Default for RequestIds {
    fn default() -> Self {
        Self::with_seed(RandomState::new().hash_one("inferrail request ids"))
    }
}

impl RequestIds {
    pub fn with_seed(seed: u64) -> Self {
        Self {
            seed,
            counter: AtomicU64::new(0),
        }
    }

    pub fn mint(&self) -> HeaderValue {
        let n = self.counter.fetch_add(1, Ordering::Relaxed);
        let mut key = [0u8; 16];
        key[..8].copy_from_slice(&self.seed.to_le_bytes());
        key[8..].copy_from_slice(&n.to_le_bytes());
        let hex = format!("{:032x}", xxh3_128(&key));
        HeaderValue::from_str(&hex).unwrap_or_else(|_| HeaderValue::from_static("0"))
    }

    pub fn resolve(&self, headers: &HeaderMap) -> HeaderValue {
        match headers.get(&REQUEST_ID) {
            Some(value) if is_valid(value.as_bytes()) => value.clone(),
            _ => self.mint(),
        }
    }
}

pub fn is_valid(id: &[u8]) -> bool {
    (1..=MAX_LEN).contains(&id.len()) && id.iter().all(|b| (0x21..=0x7e).contains(b))
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;

    #[test]
    fn minted_ids_are_unique_hex() {
        let ids = RequestIds::with_seed(1);
        let minted: HashSet<String> = (0..10_000)
            .map(|_| ids.mint().to_str().unwrap().to_owned())
            .collect();
        assert_eq!(minted.len(), 10_000);
        assert!(
            minted
                .iter()
                .all(|id| id.len() == 32 && id.bytes().all(|b| b.is_ascii_hexdigit()))
        );
    }

    #[test]
    fn seeds_change_the_sequence() {
        assert_ne!(
            RequestIds::with_seed(1).mint(),
            RequestIds::with_seed(2).mint()
        );
        assert_eq!(
            RequestIds::with_seed(3).mint(),
            RequestIds::with_seed(3).mint()
        );
    }

    #[test]
    fn valid_client_ids_are_kept_and_bad_ones_replaced() {
        let ids = RequestIds::with_seed(0);
        let mut headers = HeaderMap::new();
        headers.insert(REQUEST_ID, HeaderValue::from_static("client-42"));
        assert_eq!(ids.resolve(&headers), "client-42");
        for bad in [
            &b""[..],
            b"has space",
            b"tab\there",
            "\u{e9}".as_bytes(),
            &[b'a'; 257],
        ] {
            headers.insert(REQUEST_ID, HeaderValue::from_bytes(bad).unwrap());
            assert_eq!(ids.resolve(&headers).len(), 32, "{bad:?}");
        }
        assert!(is_valid(&[b'z'; 256]));
        assert_eq!(ids.resolve(&HeaderMap::new()).len(), 32);
    }
}
