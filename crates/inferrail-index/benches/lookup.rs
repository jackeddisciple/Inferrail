use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use inferrail_core::ids::{BlockHash, ReplicaId};
use inferrail_index::{KvIndex, Slot};

fn chain(seed: u64, len: usize) -> Vec<BlockHash> {
    (0..len as u64)
        .map(|i| {
            let x = (seed << 32 | i).wrapping_mul(0x9e37_79b9_7f4a_7c15);
            BlockHash::new(u128::from(x) << 64 | u128::from(x.rotate_left(17) ^ i))
        })
        .collect()
}

/// One replica holds the whole prompt, the rest only its first block, so
/// the walk follows a single survivor to the full depth.
fn fleet(replicas: usize, depth: usize) -> (KvIndex, Vec<Slot>, Vec<BlockHash>) {
    let index = KvIndex::new();
    let prompt = chain(1, depth);
    let slots: Vec<Slot> = (0..replicas)
        .map(|i| index.slot(&ReplicaId::new(format!("r{i}")).expect("id")))
        .collect();
    for (i, slot) in slots.iter().enumerate() {
        let held = if i == 0 { &prompt[..] } else { &prompt[..1] };
        index.record(*slot, held, 1 << 20);
        index.record(*slot, &chain(100 + i as u64, 256), 1 << 20);
    }
    (index, slots, prompt)
}

fn lookup(c: &mut Criterion) {
    let mut group = c.benchmark_group("index_lookup_depth");
    for depth in [1usize, 16, 128, 512, 2_000] {
        let (index, slots, prompt) = fleet(64, depth);
        let mut out = vec![0; slots.len()];
        group.throughput(Throughput::Elements(depth as u64));
        group.bench_function(BenchmarkId::new("64_replicas", depth), |b| {
            b.iter(|| index.longest_prefix_into(black_box(&prompt), &slots, &mut out));
        });
    }
    group.finish();
    let mut group = c.benchmark_group("index_lookup_replicas");
    for replicas in [8usize, 64, 256] {
        let (index, slots, prompt) = fleet(replicas, 512);
        let mut out = vec![0; slots.len()];
        group.bench_function(BenchmarkId::new("512_blocks", replicas), |b| {
            b.iter(|| index.longest_prefix_into(black_box(&prompt), &slots, &mut out));
        });
    }
    group.finish();
    let mut group = c.benchmark_group("index_record");
    let (index, slots, prompt) = fleet(8, 512);
    group.throughput(Throughput::Elements(512));
    group.bench_function("refresh_512_blocks", |b| {
        b.iter(|| index.record(slots[0], black_box(&prompt), 1 << 20));
    });
    let mut seed = 1_000;
    group.bench_function("insert_512_new_blocks_at_capacity", |b| {
        b.iter(|| {
            seed += 1;
            index.record(slots[1], black_box(&chain(seed, 512)), 4_096);
        });
    });
    group.finish();
}

criterion_group!(benches, lookup);
criterion_main!(benches);
