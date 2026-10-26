//! Concurrency and scale for the KV index. Ignored by default; run with
//! `make stress` (release build, output shown).

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use inferrail_core::ids::{BlockHash, ReplicaId};
use inferrail_index::KvIndex;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn below(&mut self, n: u64) -> u64 {
        self.next() % n.max(1)
    }
}

fn id(i: u64) -> ReplicaId {
    ReplicaId::new(format!("r{i}")).expect("id")
}

/// A chain sharing its first `shared` blocks with every chain of the same
/// family, the way prompts share a system prompt.
fn chain(family: u64, unique: u64, shared: usize, len: usize) -> Vec<BlockHash> {
    (0..len as u64)
        .map(|i| {
            let owner = if (i as usize) < shared {
                family
            } else {
                family << 32 | unique
            };
            let x = (owner ^ i << 48).wrapping_mul(0x9e37_79b9_7f4a_7c15);
            BlockHash::new(u128::from(x) << 64 | u128::from(x.rotate_left(29) ^ owner))
        })
        .collect()
}

#[test]
#[ignore = "stress"]
fn concurrent_records_lookups_and_forgets_stay_consistent() {
    let index = Arc::new(KvIndex::new());
    let stop = Arc::new(AtomicBool::new(false));
    let records = Arc::new(AtomicU64::new(0));
    let lookups = Arc::new(AtomicU64::new(0));
    let forgets = Arc::new(AtomicU64::new(0));
    let mut threads = Vec::new();
    for t in 0..12u64 {
        let (index, stop, records) = (Arc::clone(&index), Arc::clone(&stop), Arc::clone(&records));
        threads.push(std::thread::spawn(move || {
            let mut rng = Rng(t * 7_919 + 1);
            while !stop.load(Ordering::Relaxed) {
                let replica = rng.below(80);
                let len = 64 + rng.below(400) as usize;
                let hashes = chain(rng.below(32), rng.next(), 64, len);
                let capacity = 256 + rng.below(4_096) as usize;
                index.record(index.slot(&id(replica)), &hashes, capacity);
                records.fetch_add(1, Ordering::Relaxed);
            }
        }));
    }
    let mut readers = Vec::new();
    for t in 0..6u64 {
        let (index, stop, lookups) = (Arc::clone(&index), Arc::clone(&stop), Arc::clone(&lookups));
        readers.push(std::thread::spawn(move || {
            let mut rng = Rng(t * 104_729 + 3);
            let slots: Vec<_> = (0..64).map(|r| index.slot(&id(r))).collect();
            let mut times = Vec::new();
            while !stop.load(Ordering::Relaxed) {
                let hashes = chain(rng.below(32), rng.next(), 64, 512);
                let started = Instant::now();
                let depths = index.longest_prefix(&hashes, &slots);
                times.push(started.elapsed());
                assert!(depths.iter().all(|&d| d <= hashes.len()));
                lookups.fetch_add(1, Ordering::Relaxed);
            }
            times
        }));
    }
    for t in 0..2u64 {
        let (index, stop, forgets) = (Arc::clone(&index), Arc::clone(&stop), Arc::clone(&forgets));
        threads.push(std::thread::spawn(move || {
            let mut rng = Rng(t * 31 + 5);
            while !stop.load(Ordering::Relaxed) {
                let replica = id(64 + rng.below(16));
                if rng.below(2) == 0 {
                    index.forget(&replica);
                } else if let Some(slot) = index.find(&replica) {
                    index.clear(slot);
                }
                forgets.fetch_add(1, Ordering::Relaxed);
                std::thread::sleep(Duration::from_micros(200));
            }
        }));
    }
    std::thread::sleep(Duration::from_secs(5));
    stop.store(true, Ordering::Relaxed);
    for thread in threads {
        thread.join().expect("worker");
    }
    let mut times: Vec<Duration> = readers
        .into_iter()
        .flat_map(|r| r.join().expect("reader"))
        .collect();
    times.sort();
    let at = |q: f64| times[((times.len() - 1) as f64 * q) as usize].as_secs_f64() * 1e6;
    let stats = index.stats();
    println!(
        "5 s on {} cores: {} records, {} lookups (512 blocks, 64 replicas: p50 {:.1} us, p99 {:.1} us, p99.9 {:.1} us), {} forgets or clears; {} blocks on {} replicas, {:.1} MB",
        std::thread::available_parallelism().map_or(1, |n| n.get()),
        records.load(Ordering::Relaxed),
        lookups.load(Ordering::Relaxed),
        at(0.5),
        at(0.99),
        at(0.999),
        forgets.load(Ordering::Relaxed),
        stats.entries,
        stats.replicas.len(),
        stats.bytes as f64 / 1e6,
    );
    index
        .verify()
        .expect("the shared map matches every shadow cache");
    assert!(stats.replicas.len() <= 80);
    assert!(at(0.99) < 1_000.0, "lookups stalled under contention");
    assert!(stats.replicas.iter().all(|r| r.blocks <= r.capacity));
}

#[test]
#[ignore = "stress"]
fn a_million_blocks_cost_bounded_memory_and_lookups_stay_fast() {
    let index = KvIndex::new();
    let slots: Vec<_> = (0..64).map(|r| index.slot(&id(r))).collect();
    let started = Instant::now();
    for (r, slot) in slots.iter().enumerate() {
        for prompt in 0..32u64 {
            let hashes = chain(r as u64 * 1_000 + prompt, 0, 512, 512);
            index.record(*slot, &hashes, 1 << 20);
        }
    }
    let filled = started.elapsed();
    let stats = index.stats();
    assert_eq!(stats.entries, 64 * 32 * 512);
    let per_block = stats.bytes as f64 / stats.entries as f64;
    let probe = chain(5 * 1_000 + 7, 0, 512, 512);
    let mut times: Vec<Duration> = (0..2_000)
        .map(|_| {
            let started = Instant::now();
            let depths = index.longest_prefix(&probe, &slots);
            let took = started.elapsed();
            assert_eq!(depths[5], 512);
            took
        })
        .collect();
    times.sort();
    let p99 = times[times.len() * 99 / 100];
    println!(
        "{} blocks on 64 replicas in {filled:.2?}: {:.0} MB, {per_block:.0} bytes per block, 512 block lookup p50 {:.1} us p99 {:.1} us",
        stats.entries,
        stats.bytes as f64 / 1e6,
        times[times.len() / 2].as_secs_f64() * 1e6,
        p99.as_secs_f64() * 1e6,
    );
    assert!(per_block < 160.0, "{per_block}");
    assert!(p99 < Duration::from_micros(200), "{p99:?}");
    index.verify().expect("consistent");
}
