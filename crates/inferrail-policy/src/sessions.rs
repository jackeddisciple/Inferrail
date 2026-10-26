//! The replica each conversation last went to, in a fixed size table. A
//! slot holds a session tag and a replica tag in one word, so reads and
//! writes are single atomic operations; a session whose slot another took
//! over is simply routed afresh.

use std::sync::atomic::{AtomicU64, Ordering};

const SLOTS: usize = 1 << 16;

#[derive(Debug)]
pub(crate) struct Sessions {
    slots: Box<[AtomicU64]>,
}

fn place(session: u64) -> (usize, u64) {
    let slot = session as usize & (SLOTS - 1);
    let tag = (session >> 32) | 1;
    (slot, tag << 32)
}

impl Sessions {
    pub(crate) fn new() -> Self {
        Self {
            slots: (0..SLOTS).map(|_| AtomicU64::new(0)).collect(),
        }
    }

    /// The tag of the replica `session` last went to.
    pub(crate) fn get(&self, session: u64) -> Option<u32> {
        let (slot, tag) = place(session);
        let word = self.slots[slot].load(Ordering::Relaxed);
        (word & !u64::from(u32::MAX) == tag).then_some(word as u32)
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
    }

    #[test]
    fn a_session_that_lost_its_slot_is_forgotten_not_misrouted() {
        let sessions = Sessions::new();
        let (a, b) = (5u64 << 32 | 3, 6u64 << 32 | 3);
        sessions.set(a, 1);
        sessions.set(b, 2);
        assert_eq!(sessions.get(a), None);
        assert_eq!(sessions.get(b), Some(2));
    }
}
