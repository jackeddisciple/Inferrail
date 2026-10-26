//! A count-min sketch over a sliding window, lock free: amounts added under
//! a key are summed in the current window, and a rate blends it with the
//! previous window by how much of the current one has passed.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

const ROWS: usize = 4;
const WIDTH: usize = 1024;

#[derive(Debug)]
pub(crate) struct Sketch {
    window: u64,
    started: AtomicU64,
    current: Box<[AtomicU64]>,
    previous: Box<[AtomicU64]>,
}

fn cells() -> Box<[AtomicU64]> {
    (0..ROWS * WIDTH).map(|_| AtomicU64::new(0)).collect()
}

fn slots(key: u64) -> [usize; ROWS] {
    std::array::from_fn(|row| {
        let mut z = key.wrapping_add((row as u64 + 1).wrapping_mul(0x9e37_79b9_7f4a_7c15));
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        row * WIDTH + ((z ^ (z >> 31)) as usize & (WIDTH - 1))
    })
}

impl Sketch {
    pub(crate) fn new(window: Duration) -> Self {
        Self {
            window: u64::try_from(window.as_nanos()).unwrap_or(u64::MAX).max(1),
            started: AtomicU64::new(0),
            current: cells(),
            previous: cells(),
        }
    }

    /// Adds `amount` under `key` at `now`, in nanoseconds of any clock that
    /// does not go backwards.
    pub(crate) fn add(&self, key: u64, amount: u64, now: u64) {
        self.rotate(now);
        for slot in slots(key) {
            self.current[slot].fetch_add(amount, Ordering::Relaxed);
        }
    }

    /// The amount added under `key` per second around `now`. Never below
    /// the true rate over the windows it covers, as count-min sketches go.
    pub(crate) fn rate(&self, key: u64, now: u64) -> f64 {
        let started = self.started.load(Ordering::Acquire);
        let age = now.saturating_sub(started);
        let min = |cells: &[AtomicU64]| {
            slots(key)
                .into_iter()
                .map(|slot| cells[slot].load(Ordering::Relaxed))
                .min()
                .unwrap_or(0) as f64
        };
        let window = self.window as f64;
        let summed = if age >= 2 * self.window {
            0.0
        } else if age >= self.window {
            min(&self.current) * (1.0 - (age - self.window) as f64 / window)
        } else {
            min(&self.current) + min(&self.previous) * (1.0 - age as f64 / window)
        };
        summed / (window / 1e9)
    }

    /// Starts a new window once the current one is over. One caller wins
    /// the switch; amounts racing with it may land in either window.
    fn rotate(&self, now: u64) {
        let started = self.started.load(Ordering::Acquire);
        let elapsed = now.saturating_sub(started);
        if elapsed < self.window {
            return;
        }
        let next = started + elapsed / self.window * self.window;
        if self
            .started
            .compare_exchange(started, next, Ordering::AcqRel, Ordering::Relaxed)
            .is_err()
        {
            return;
        }
        let consecutive = elapsed < 2 * self.window;
        for (current, previous) in self.current.iter().zip(self.previous.iter()) {
            let moved = current.swap(0, Ordering::Relaxed);
            previous.store(if consecutive { moved } else { 0 }, Ordering::Relaxed);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    const SECOND: u64 = 1_000_000_000;

    fn close(a: f64, b: f64, tolerance: f64) -> bool {
        (a - b).abs() <= tolerance * b.abs().max(1.0)
    }

    #[test]
    fn a_steady_rate_reads_back_as_itself() {
        let sketch = Sketch::new(Duration::from_secs(10));
        for tick in 0..400u64 {
            sketch.add(42, 50, tick * SECOND / 10);
        }
        let now = 400 * SECOND / 10;
        assert!(
            close(sketch.rate(42, now), 500.0, 0.06),
            "{}",
            sketch.rate(42, now)
        );
        assert_eq!(sketch.rate(43, now), 0.0);
    }

    #[test]
    fn rates_fade_after_traffic_stops() {
        let sketch = Sketch::new(Duration::from_secs(10));
        for tick in 0..100u64 {
            sketch.add(7, 10, tick * SECOND / 10);
        }
        let end = 10 * SECOND;
        let fresh = sketch.rate(7, end);
        assert!(fresh > 50.0, "{fresh}");
        assert!(sketch.rate(7, end + 5 * SECOND) < fresh);
        assert_eq!(sketch.rate(7, end + 21 * SECOND), 0.0);
        sketch.add(9, 1, end + 30 * SECOND);
        assert_eq!(sketch.rate(7, end + 30 * SECOND), 0.0);
    }

    #[test]
    fn collisions_only_ever_overestimate() {
        let sketch = Sketch::new(Duration::from_secs(1));
        for key in 0..20_000u64 {
            sketch.add(key, 1 + key % 3, 0);
        }
        for key in 0..20_000u64 {
            assert!(sketch.rate(key, 0) >= (1 + key % 3) as f64);
        }
    }

    #[test]
    fn concurrent_adds_are_all_counted() {
        let sketch = Arc::new(Sketch::new(Duration::from_secs(3600)));
        let threads: Vec<_> = (0..8)
            .map(|_| {
                let sketch = Arc::clone(&sketch);
                std::thread::spawn(move || {
                    for i in 0..10_000u64 {
                        sketch.add(5, 1, i);
                    }
                })
            })
            .collect();
        for thread in threads {
            thread.join().unwrap();
        }
        assert!(close(sketch.rate(5, 0) * 3600.0, 80_000.0, 1e-9));
    }
}
