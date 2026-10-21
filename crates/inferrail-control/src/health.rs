use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::time::Duration;

use inferrail_core::config::HealthConfig;

/// Nanoseconds since a [`Clock`]'s epoch. Zero means "never".
pub type Nanos = u64;

#[derive(Debug, Clone, Copy)]
pub struct Clock {
    epoch: tokio::time::Instant,
}

impl Default for Clock {
    fn default() -> Self {
        Self {
            epoch: tokio::time::Instant::now(),
        }
    }
}

impl Clock {
    pub fn now(&self) -> Nanos {
        u64::try_from(self.epoch.elapsed().as_nanos())
            .unwrap_or(u64::MAX)
            .max(1)
    }
}

const MAX_MULTIPLIER: u32 = 32;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HealthPolicy {
    pub unhealthy_threshold: u32,
    pub healthy_threshold: u32,
    pub ejection_backoff: Duration,
    pub max_ejection_backoff: Duration,
}

impl From<&HealthConfig> for HealthPolicy {
    fn from(config: &HealthConfig) -> Self {
        Self {
            unhealthy_threshold: config.unhealthy_threshold.max(1),
            healthy_threshold: config.healthy_threshold.max(1),
            ejection_backoff: config.ejection_backoff,
            max_ejection_backoff: config.max_ejection_backoff,
        }
    }
}

impl HealthPolicy {
    /// Length of the `n`th ejection in a row: doubles from the base, capped.
    pub fn backoff(&self, n: u32) -> Duration {
        let factor = 1u32 << n.clamp(1, MAX_MULTIPLIER).saturating_sub(1).min(31);
        self.ejection_backoff
            .saturating_mul(factor)
            .min(self.max_ejection_backoff)
    }
}

/// Passive outlier state and active probe state of one replica. Reads are
/// plain atomic loads; concurrent failures eject the replica exactly once.
#[derive(Debug)]
pub struct Health {
    probe_up: AtomicBool,
    probe_failures: AtomicU32,
    probe_successes: AtomicU32,
    failures: AtomicU32,
    ejected_until: AtomicU64,
    ejections: AtomicU32,
}

impl Default for Health {
    fn default() -> Self {
        Self {
            probe_up: AtomicBool::new(true),
            probe_failures: AtomicU32::new(0),
            probe_successes: AtomicU32::new(0),
            failures: AtomicU32::new(0),
            ejected_until: AtomicU64::new(0),
            ejections: AtomicU32::new(0),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Ejection {
    pub length: Duration,
    pub consecutive: u32,
}

impl Health {
    pub fn is_routable(&self, now: Nanos) -> bool {
        self.probe_up() && !self.is_ejected(now)
    }

    pub fn probe_up(&self) -> bool {
        self.probe_up.load(Ordering::Acquire)
    }

    pub fn is_ejected(&self, now: Nanos) -> bool {
        now < self.ejected_until.load(Ordering::Acquire)
    }

    pub fn ejected_until(&self) -> Nanos {
        self.ejected_until.load(Ordering::Acquire)
    }

    pub fn consecutive_failures(&self) -> u32 {
        self.failures.load(Ordering::Relaxed)
    }

    pub fn record_success(&self) {
        if self.failures.load(Ordering::Relaxed) != 0 {
            self.failures.store(0, Ordering::Relaxed);
        }
    }

    /// Counts a failed request. True once the replica has failed
    /// `unhealthy_threshold` times in a row while not ejected, which makes
    /// it eligible for [`Health::eject`]. Failures while ejected are ignored.
    pub fn record_failure(&self, now: Nanos, policy: &HealthPolicy) -> bool {
        if self.is_ejected(now) {
            return false;
        }
        let failures = self.failures.fetch_add(1, Ordering::AcqRel) + 1;
        failures >= policy.unhealthy_threshold
    }

    /// Ejects the replica for its next backoff window unless it already is;
    /// of any number of concurrent callers exactly one succeeds.
    pub fn eject(&self, now: Nanos, policy: &HealthPolicy) -> Option<Ejection> {
        let until = self.ejected_until.load(Ordering::Acquire);
        if now < until {
            return None;
        }
        let clean = until == 0 || now - until >= duration_nanos(policy.max_ejection_backoff);
        let consecutive = if clean {
            1
        } else {
            (self.ejections.load(Ordering::Relaxed) + 1).min(MAX_MULTIPLIER)
        };
        let length = policy.backoff(consecutive);
        let next = now.saturating_add(duration_nanos(length)).max(now + 1);
        self.ejected_until
            .compare_exchange(until, next, Ordering::AcqRel, Ordering::Acquire)
            .ok()?;
        self.ejections.store(consecutive, Ordering::Relaxed);
        self.failures.store(0, Ordering::Relaxed);
        Some(Ejection {
            length,
            consecutive,
        })
    }

    /// Records an active probe; returns the new state when it flips. Only the
    /// health checker calls this, one probe at a time per replica.
    pub fn record_probe(&self, ok: bool, policy: &HealthPolicy) -> Option<bool> {
        let up = self.probe_up();
        let (streak, other, threshold) = if ok {
            (
                &self.probe_successes,
                &self.probe_failures,
                policy.healthy_threshold,
            )
        } else {
            (
                &self.probe_failures,
                &self.probe_successes,
                policy.unhealthy_threshold,
            )
        };
        other.store(0, Ordering::Relaxed);
        let count = streak.fetch_add(1, Ordering::Relaxed).saturating_add(1);
        (up != ok && count >= threshold).then(|| {
            self.probe_up.store(ok, Ordering::Release);
            streak.store(0, Ordering::Relaxed);
            ok
        })
    }
}

fn duration_nanos(d: Duration) -> u64 {
    u64::try_from(d.as_nanos()).unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use proptest::prelude::*;

    const MS: u64 = 1_000_000;

    fn fail(health: &Health, now: Nanos, p: &HealthPolicy) -> Option<Ejection> {
        if health.record_failure(now, p) {
            health.eject(now, p)
        } else {
            None
        }
    }

    fn policy() -> HealthPolicy {
        HealthPolicy {
            unhealthy_threshold: 3,
            healthy_threshold: 2,
            ejection_backoff: Duration::from_millis(100),
            max_ejection_backoff: Duration::from_millis(1_000),
        }
    }

    #[test]
    fn backoff_doubles_and_caps() {
        let p = policy();
        let lengths: Vec<u64> = (1..=7).map(|n| p.backoff(n).as_millis() as u64).collect();
        assert_eq!(lengths, [100, 200, 400, 800, 1_000, 1_000, 1_000]);
        assert_eq!(p.backoff(u32::MAX), p.max_ejection_backoff);
        assert_eq!(p.backoff(0), p.ejection_backoff);
    }

    #[test]
    fn threshold_failures_eject_and_success_resets_the_streak() {
        let (health, p) = (Health::default(), policy());
        assert!(fail(&health, 10, &p).is_none());
        assert!(fail(&health, 11, &p).is_none());
        health.record_success();
        assert!(fail(&health, 12, &p).is_none());
        assert!(fail(&health, 13, &p).is_none());
        let ejection = fail(&health, 14, &p).unwrap();
        assert_eq!(ejection.length, Duration::from_millis(100));
        assert!(!health.is_routable(15));
        assert!(health.is_routable(14 + 100 * MS));
    }

    #[test]
    fn repeated_ejections_back_off_until_a_clean_stretch() {
        let (health, p) = (Health::default(), policy());
        let mut now = 1;
        let mut lengths = Vec::new();
        for _ in 0..6 {
            let mut ejection = None;
            while ejection.is_none() {
                ejection = fail(&health, now, &p);
            }
            let ejection = ejection.unwrap();
            lengths.push(ejection.length.as_millis());
            now = health.ejected_until();
        }
        assert_eq!(lengths, [100, 200, 400, 800, 1_000, 1_000]);
        now += 1_000 * MS;
        for _ in 0..2 {
            assert!(fail(&health, now, &p).is_none());
        }
        assert_eq!(fail(&health, now, &p).unwrap().length.as_millis(), 100);
    }

    #[test]
    fn failures_while_ejected_are_ignored() {
        let (health, p) = (Health::default(), policy());
        for _ in 0..3 {
            fail(&health, 5, &p);
        }
        for _ in 0..50 {
            assert!(fail(&health, 6, &p).is_none());
        }
        assert_eq!(health.consecutive_failures(), 0);
    }

    #[test]
    fn probes_flip_state_after_their_thresholds() {
        let (health, p) = (Health::default(), policy());
        assert_eq!(health.record_probe(false, &p), None);
        assert_eq!(health.record_probe(true, &p), None);
        assert_eq!(health.record_probe(false, &p), None);
        assert_eq!(health.record_probe(false, &p), None);
        assert_eq!(health.record_probe(false, &p), Some(false));
        assert!(!health.is_routable(1));
        assert_eq!(health.record_probe(false, &p), None);
        assert_eq!(health.record_probe(true, &p), None);
        assert_eq!(health.record_probe(true, &p), Some(true));
        assert!(health.is_routable(1));
    }

    #[test]
    fn concurrent_failures_eject_exactly_once() {
        let health = Arc::new(Health::default());
        let barrier = Arc::new(std::sync::Barrier::new(16));
        let threads: Vec<_> = (0..16)
            .map(|_| {
                let (health, barrier) = (Arc::clone(&health), Arc::clone(&barrier));
                std::thread::spawn(move || {
                    barrier.wait();
                    (0..100)
                        .filter(|_| fail(&health, 7, &policy()).is_some())
                        .count()
                })
            })
            .collect();
        let ejections: usize = threads.into_iter().map(|t| t.join().unwrap()).sum();
        assert_eq!(ejections, 1);
        assert!(health.is_ejected(8));
    }

    #[test]
    fn a_declined_ejection_keeps_the_replica_eligible() {
        let (health, p) = (Health::default(), policy());
        assert!(!health.record_failure(1, &p));
        assert!(!health.record_failure(2, &p));
        assert!(health.record_failure(3, &p));
        assert!(
            health.record_failure(4, &p),
            "still eligible after a decline"
        );
        assert!(health.is_routable(5));
        assert!(health.eject(5, &p).is_some());
        assert!(health.eject(6, &p).is_none(), "already ejected");
        assert!(!health.record_failure(7, &p));
    }

    #[test]
    fn clock_never_reports_zero() {
        assert!(Clock::default().now() >= 1);
    }

    #[derive(Debug, Clone)]
    enum Event {
        Success,
        Failure { allowed: bool },
        Probe(bool),
        Advance(u64),
    }

    #[derive(Default)]
    struct Model {
        up: bool,
        probe_ok: u32,
        probe_bad: u32,
        failures: u32,
        until: u64,
        streak: u32,
    }

    impl Model {
        fn failure(&mut self, now: u64, p: &HealthPolicy, allowed: bool) -> Option<Duration> {
            if now < self.until {
                return None;
            }
            self.failures += 1;
            if self.failures < p.unhealthy_threshold || !allowed {
                return None;
            }
            let max = p.max_ejection_backoff.as_nanos() as u64;
            self.streak = if self.until == 0 || now - self.until >= max {
                1
            } else {
                (self.streak + 1).min(32)
            };
            let mut length = p.ejection_backoff;
            for _ in 1..self.streak {
                length = (length * 2).min(p.max_ejection_backoff);
            }
            self.until = now + length.as_nanos() as u64;
            self.failures = 0;
            Some(length)
        }

        fn probe(&mut self, ok: bool, p: &HealthPolicy) -> Option<bool> {
            if ok {
                self.probe_bad = 0;
                self.probe_ok += 1;
                if !self.up && self.probe_ok >= p.healthy_threshold {
                    self.up = true;
                    self.probe_ok = 0;
                    return Some(true);
                }
            } else {
                self.probe_ok = 0;
                self.probe_bad += 1;
                if self.up && self.probe_bad >= p.unhealthy_threshold {
                    self.up = false;
                    self.probe_bad = 0;
                    return Some(false);
                }
            }
            None
        }
    }

    fn event() -> impl Strategy<Value = Event> {
        prop_oneof![
            2 => Just(Event::Success),
            5 => prop::bool::weighted(0.8).prop_map(|allowed| Event::Failure { allowed }),
            2 => any::<bool>().prop_map(Event::Probe),
            3 => (0u64..1_500).prop_map(|ms| Event::Advance(ms * MS)),
        ]
    }

    proptest! {
        #[test]
        fn matches_a_naive_reference_model(
            events in prop::collection::vec(event(), 1..300),
            unhealthy in 1u32..5,
            healthy in 1u32..4,
            base_ms in 1u64..200,
            cap_factor in 1u64..20,
        ) {
            let p = HealthPolicy {
                unhealthy_threshold: unhealthy,
                healthy_threshold: healthy,
                ejection_backoff: Duration::from_millis(base_ms),
                max_ejection_backoff: Duration::from_millis(base_ms * cap_factor),
            };
            let health = Health::default();
            let mut model = Model { up: true, ..Model::default() };
            let mut now = 1u64;
            for event in events {
                match event {
                    Event::Success => {
                        health.record_success();
                        model.failures = 0;
                    }
                    Event::Failure { allowed } => {
                        let got = if health.record_failure(now, &p) && allowed {
                            health.eject(now, &p).map(|e| e.length)
                        } else {
                            None
                        };
                        prop_assert_eq!(got, model.failure(now, &p, allowed));
                        if let Some(length) = got {
                            prop_assert!(length <= p.max_ejection_backoff);
                            prop_assert!(!health.is_routable(now));
                        }
                    }
                    Event::Probe(ok) => {
                        prop_assert_eq!(health.record_probe(ok, &p), model.probe(ok, &p));
                    }
                    Event::Advance(dt) => now += dt,
                }
                prop_assert_eq!(health.is_routable(now), model.up && now >= model.until);
            }
        }
    }
}
