//! A5: the full scoring decision for an 8,000 token prompt: the index
//! lookup of every candidate, the candidates assembled, and the cost model
//! with its guards choosing among them.

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use inferrail_core::ids::{BlockHash, ModelId, ReplicaId};
use inferrail_index::{KvIndex, Slot};
use inferrail_policy::{
    Candidate, CostModelPolicy, RouteContext, RoutingPolicy, Settings, SplitMix,
};

const BLOCKS: usize = 512;

fn chain(seed: u64, len: usize) -> Vec<BlockHash> {
    (0..len as u64)
        .map(|i| {
            let x = (seed << 32 | i).wrapping_mul(0x9e37_79b9_7f4a_7c15);
            BlockHash::new(u128::from(x) << 64 | u128::from(x.rotate_left(17) ^ i))
        })
        .collect()
}

/// Every replica holds a different share of the prompt and unrelated
/// history besides, so the lookup walks deep and the candidates differ.
fn fleet(replicas: usize) -> (KvIndex, Vec<ReplicaId>, Vec<Slot>, Vec<BlockHash>) {
    let index = KvIndex::new();
    let prompt = chain(1, BLOCKS);
    let ids: Vec<ReplicaId> = (0..replicas)
        .map(|i| ReplicaId::new(format!("r{i}")).expect("replica id"))
        .collect();
    let slots: Vec<Slot> = ids.iter().map(|id| index.slot(id)).collect();
    for (i, slot) in slots.iter().enumerate() {
        index.record(*slot, &prompt[..(i * 37) % BLOCKS], 1 << 20);
        index.record(*slot, &chain(100 + i as u64, 256), 1 << 20);
    }
    (index, ids, slots, prompt)
}

fn decide(c: &mut Criterion) {
    let model = ModelId::new("bench").expect("model id");
    let mut group = c.benchmark_group("decide_8k_prompt");
    for replicas in [8usize, 64, 256] {
        let (index, ids, slots, prompt) = fleet(replicas);
        let rng = SplitMix::new(3);
        let loads: Vec<(u32, f64, Option<f64>)> = (0..replicas)
            .map(|_| {
                let in_flight = rng.below(32) as u32;
                (
                    in_flight,
                    rng.below(500) as f64 / 1e3,
                    Some(rng.below(100) as f64 / 100.0),
                )
            })
            .collect();
        let policy = CostModelPolicy::new(&Settings::seeded(9));
        let mut depth = vec![0usize; replicas];
        let mut now = 0u64;
        group.bench_function(BenchmarkId::new("cost_model", replicas), |b| {
            b.iter(|| {
                now += 1_000;
                index.longest_prefix_into(black_box(&prompt[..BLOCKS - 1]), &slots, &mut depth);
                let candidates: Vec<Candidate<'_>> = ids
                    .iter()
                    .zip(&depth)
                    .zip(&loads)
                    .map(|((id, &blocks), &(in_flight, queued, kv))| Candidate {
                        cached_tokens: (blocks * 16) as u32,
                        queued_secs: queued,
                        running: in_flight,
                        kv_usage: kv,
                        ..Candidate::new(id, in_flight)
                    })
                    .collect();
                let ctx = RouteContext {
                    prompt_tokens: (BLOCKS * 16) as u32,
                    blocks: &prompt,
                    block_size: 16,
                    now,
                    ..RouteContext::new(&model, &candidates)
                };
                black_box(policy.choose(&ctx))
            });
        });
    }
    group.finish();
}

criterion_group!(benches, decide);
criterion_main!(benches);
