use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use inferrail_core::config::PolicyKind;
use inferrail_core::ids::{ModelId, ReplicaId};
use inferrail_policy::{Candidate, RouteContext, SplitMix, build};

fn choose(c: &mut Criterion) {
    let model = ModelId::new("bench").expect("model id");
    let mut group = c.benchmark_group("policy_choose");
    for replicas in [8usize, 64, 256] {
        let ids: Vec<ReplicaId> = (0..replicas)
            .map(|i| ReplicaId::new(format!("r{i}")).expect("replica id"))
            .collect();
        let rng = SplitMix::new(1);
        let candidates: Vec<Candidate<'_>> = ids
            .iter()
            .map(|id| Candidate {
                id,
                in_flight: rng.below(64) as u32,
                cached_tokens: 0,
            })
            .collect();
        let ctx = RouteContext {
            model: &model,
            prompt_tokens: None,
            candidates: &candidates,
        };
        for kind in [
            PolicyKind::RoundRobin,
            PolicyKind::Random,
            PolicyKind::LeastRequest,
            PolicyKind::P2c,
        ] {
            let policy = build(kind, 7);
            group.bench_with_input(BenchmarkId::new(kind.as_str(), replicas), &ctx, |b, ctx| {
                b.iter(|| black_box(policy.choose(black_box(ctx))));
            });
        }
    }
    group.finish();
}

criterion_group!(benches, choose);
criterion_main!(benches);
