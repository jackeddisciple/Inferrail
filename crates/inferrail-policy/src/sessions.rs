//! The replica each conversation last went to, in a fixed size table. A
//! slot holds a session tag and a replica tag in one word, so reads and
//! writes are single atomic operations; a session whose slot another took
//! over is simply routed afresh. Keys are mixed before they are split into
//! slot and tag, so keys alike in any pattern spread, and a lookup checks
//! 48 bits of the mixed key.

use std::sync::atomic::{AtomicU64, Ordering};

const SLOTS: usize = 1 << 16;

#[derive(Debug)]
pub(crate) struct Sessions {
    slots: Box<[AtomicU64]>,
}

fn place(session: u64) -> (usize, u64) {
    let mut z = session;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^= z >> 31;
    (z as usize & (SLOTS - 1), z & !u64::from(u32::MAX))
}

impl Sessions {
    pub(crate) fn new() -> Self {
        Self {
            slots: (0..SLOTS).map(|_| AtomicU64::new(0)).collect(),
        }
    }

    /// The tag of the replica `session` last went to. Replica tags are
    /// never zero, so an empty slot matches no session.
    pub(crate) fn get(&self, session: u64) -> Option<u32> {
        let (slot, tag) = place(session);
        let word = self.slots[slot].load(Ordering::Relaxed);
        let replica = word as u32;
        (word & !u64::from(u32::MAX) == tag && replica != 0).then_some(replica)
    }

    pub(crate) fn set(&self, session: u64, replica: u32) {
        let (slot, tag) = place(session);
        self.slots[slot].store(tag | u64::from(replica), Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sessions_remember_their_last_replica() {
        let sessions = Sessions::new();
        assert_eq!(sessions.get(1), None);
        sessions.set(1, 7);
        sessions.set(2, 9);
        assert_eq!((sessions.get(1), sessions.get(2)), (Some(7), Some(9)));
        sessions.set(1, 11);
        assert_eq!(sessions.get(1), Some(11));
        assert_eq!(sessions.get(0), None);
    }

    #[test]
    fn keys_that_differ_in_any_bit_are_not_mistaken_for_each_other() {
        for (a, b) in [
            (4u64 << 32 | 3, 5 << 32 | 3),
            (3, 1 << 16 | 3),
            (7, 1 << 31 | 7),
        ] {
            let sessions = Sessions::new();
            sessions.set(a, 1);
            assert_eq!(sessions.get(b), None, "{a:#x} and {b:#x}");
            sessions.set(b, 2);
            assert!(
                matches!(sessions.get(a), None | Some(1)),
                "{a:#x} and {b:#x}"
            );
        }
    }

    #[test]
    fn a_session_that_lost_its_slot_is_forgotten_not_misrouted() {
        let sessions = Sessions::new();
        let a = 3u64;
        let b = (4..).find(|&b| place(b).0 == place(a).0).unwrap();
        sessions.set(a, 1);
        sessions.set(b, 2);
        assert_eq!(sessions.get(a), None);
        assert_eq!(sessions.get(b), Some(2));
    }
}
