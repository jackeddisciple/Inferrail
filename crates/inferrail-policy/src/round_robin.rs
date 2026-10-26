use std::sync::atomic::{AtomicUsize, Ordering};

use inferrail_core::config::PolicyKind;

use crate::{RouteContext, RoutingPolicy};

#[derive(Debug, Default)]
pub struct RoundRobin {
    next: AtomicUsize,
}

impl RoundRobin {
    pub fn new(start: usize) -> Self {
        Self {
            next: AtomicUsize::new(start),
        }
    }
}

impl RoutingPolicy for RoundRobin {
    fn kind(&self) -> PolicyKind {
        PolicyKind::RoundRobin
    }

    fn choose(&self, ctx: &RouteContext<'_>) -> Option<usize> {
        let len = ctx.candidates.len();
        (len > 0).then(|| self.next.fetch_add(1, Ordering::Relaxed) % len)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::testing::Pool;

    #[test]
    fn cycles_through_candidates_in_order() {
        let pool = Pool::new(3);
        let candidates = pool.candidates(&[]);
        let policy = RoundRobin::new(1);
        let picks: Vec<usize> = (0..7).map(|_| pool.pick(&policy, &candidates)).collect();
        assert_eq!(picks, [1, 2, 0, 1, 2, 0, 1]);
    }

    #[test]
    fn spreads_exactly_evenly_and_ignores_load() {
        let pool = Pool::new(4);
        let candidates = pool.candidates(&[100, 0, 7, 3]);
        let policy = RoundRobin::default();
        let mut counts = [0u64; 4];
        for _ in 0..4_000 {
            counts[pool.pick(&policy, &candidates)] += 1;
        }
        assert_eq!(counts, [1_000; 4]);
    }

    #[test]
    fn concurrent_callers_stay_exactly_balanced() {
        let policy = Arc::new(RoundRobin::default());
        let handles: Vec<_> = (0..8)
            .map(|_| {
                let policy = Arc::clone(&policy);
                std::thread::spawn(move || {
                    let pool = Pool::new(4);
                    let candidates = pool.candidates(&[]);
                    let mut counts = [0u64; 4];
                    for _ in 0..10_000 {
                        counts[pool.pick(policy.as_ref(), &candidates)] += 1;
                    }
                    counts
                })
            })
            .collect();
        let mut total = [0u64; 4];
        for handle in handles {
            for (t, c) in total.iter_mut().zip(handle.join().unwrap()) {
                *t += c;
            }
        }
        assert_eq!(total, [20_000; 4]);
    }

    #[test]
    fn empty_set_gives_none() {
        let pool = Pool::new(0);
        let ctx = RouteContext {
            model: &pool.model,
            prompt_tokens: None,
            candidates: &[],
        };
        assert_eq!(RoundRobin::default().choose(&ctx), None);
        assert_eq!(RoundRobin::default().kind(), PolicyKind::RoundRobin);
    }
}
