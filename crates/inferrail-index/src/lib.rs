//! KV block index in approximate and exact modes, with tier tracking.
//!
//! Block hashes are uniformly distributed already, so tables keyed by them
//! use the hash itself instead of hashing it again.

use std::hash::{BuildHasherDefault, Hasher};

mod index;
mod lru;
mod map;
mod set;

pub use index::{IndexStats, KvIndex, ReplicaStats, Slot};
pub use set::ReplicaSet;

#[derive(Default)]
pub(crate) struct Identity(u64);

impl Hasher for Identity {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, bytes: &[u8]) {
        for chunk in bytes.chunks(8) {
            let mut word = [0u8; 8];
            word[..chunk.len()].copy_from_slice(chunk);
            self.0 = self.0.rotate_left(5) ^ u64::from_le_bytes(word);
        }
    }

    fn write_u128(&mut self, value: u128) {
        self.0 = value as u64;
    }
}

pub(crate) type IdentityHash = BuildHasherDefault<Identity>;
