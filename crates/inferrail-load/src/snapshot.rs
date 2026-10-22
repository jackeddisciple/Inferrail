use std::sync::Arc;
use std::time::Duration;

use arc_swap::ArcSwapOption;
use tokio::time::Instant;

use crate::EngineLoad;

/// An engine's load as scraped at `at`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sample {
    pub load: EngineLoad,
    pub at: Instant,
}

impl Sample {
    pub fn age(&self, now: Instant) -> Duration {
        now.saturating_duration_since(self.at)
    }
}

/// The latest sample for one replica, replaced atomically by the collector
/// and read without locks.
#[derive(Debug, Default)]
pub struct LoadCell {
    latest: ArcSwapOption<Sample>,
}

impl LoadCell {
    pub fn publish(&self, load: EngineLoad, at: Instant) {
        self.latest.store(Some(Arc::new(Sample { load, at })));
    }

    pub fn latest(&self) -> Option<Sample> {
        self.latest.load().as_deref().copied()
    }

    /// The latest sample unless it is older than `max_age`; stale load is
    /// treated as unknown rather than trusted.
    pub fn fresh(&self, now: Instant, max_age: Duration) -> Option<Sample> {
        self.latest().filter(|s| s.age(now) <= max_age)
    }
}

/// A replica's load as the gateway sees it: its own counters, always
/// current, merged with the engine's report when that is fresh.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ReplicaLoad {
    pub in_flight: u32,
    pub pending: u32,
    /// Estimated prompt tokens of requests in flight.
    pub in_flight_tokens: u64,
    /// Estimated prompt tokens of requests that have not sent a byte back yet.
    pub prefill_tokens: u64,
    /// The engine's report, `None` when missing or older than the limit.
    pub engine: Option<EngineLoad>,
    /// Age of the latest report, fresh or not.
    pub age: Option<Duration>,
}

impl ReplicaLoad {
    pub fn is_stale(&self) -> bool {
        self.engine.is_none()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Engine;

    fn load(running: u32) -> EngineLoad {
        EngineLoad {
            engine: Engine::Vllm,
            running,
            waiting: 0,
            kv_usage: 0.5,
            prefix_queries: 0,
            prefix_hits: 0,
            preemptions: 0,
            kv_capacity_tokens: None,
        }
    }

    #[tokio::test(start_paused = true)]
    async fn samples_go_stale_after_max_age() {
        let cell = LoadCell::default();
        let max_age = Duration::from_millis(250);
        assert_eq!(cell.fresh(Instant::now(), max_age), None);
        cell.publish(load(3), Instant::now());
        tokio::time::advance(Duration::from_millis(250)).await;
        let sample = cell.fresh(Instant::now(), max_age).unwrap();
        assert_eq!(sample.load.running, 3);
        assert_eq!(sample.age(Instant::now()), max_age);
        tokio::time::advance(Duration::from_millis(1)).await;
        assert_eq!(cell.fresh(Instant::now(), max_age), None);
        assert_eq!(cell.latest().unwrap().load.running, 3);
        cell.publish(load(4), Instant::now());
        assert_eq!(cell.fresh(Instant::now(), max_age).unwrap().load.running, 4);
    }

    #[tokio::test(start_paused = true)]
    async fn a_sample_from_the_future_is_age_zero() {
        let cell = LoadCell::default();
        let now = Instant::now();
        cell.publish(load(1), now + Duration::from_secs(1));
        assert_eq!(cell.latest().unwrap().age(now), Duration::ZERO);
        assert!(cell.fresh(now, Duration::ZERO).is_some());
    }
}
