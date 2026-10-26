use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use inferrail_core::config::PolicyKind;
use inferrail_core::ids::{BlockHash, ModelId, ReplicaId};
use inferrail_policy::{Candidate, RouteContext, Settings, SplitMix, build};

fn choose(c: &mut Criterion) {
    let model = ModelId::new("bench").expect("model id");
    let blocks: Vec<BlockHash> = (1..=512u128).map(|i| BlockHash::new(i << 64 | i)).collect();
    let mut group = c.benchmark_group("policy_choose");
    for replicas in [8usize, 64, 256] {
        let ids: Vec<ReplicaId> = (0..replicas)
            .map(|i| ReplicaId::new(format!("r{i}")).expect("replica id"))
            .collect();
        let rng = SplitMix::new(1);
        let candidates: Vec<Candidate<'_>> = ids
            .iter()
            .map(|id| Candidate {
                cached_tokens: rng.below(8_000) as u32,
                queued_secs: rng.below(1_000) as f64 / 1e3,
                running: rng.below(64) as u32,
                kv_usage: Some(rng.below(100) as f64 / 100.0),
                ..Candidate::new(id, rng.below(64) as u32)
            })
            .collect();
        let ctx = RouteContext {
            prompt_tokens: 8_192,
            blocks: &blocks,
            ..RouteContext::new(&model, &candidates)
        };
        for kind in PolicyKind::ALL {
            let policy = build(kind, &Settings::seeded(7));
            group.bench_with_input(BenchmarkId::new(kind.as_str(), replicas), &ctx, |b, ctx| {
                b.iter(|| black_box(policy.choose(black_box(ctx))));
            });
        }
    }
    group.finish();
}

criterion_group!(benches, choose);
criterion_main!(benches);
