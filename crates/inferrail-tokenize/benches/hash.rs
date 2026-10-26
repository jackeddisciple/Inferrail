use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use inferrail_tokenize::hash::{BlockHasher, Salt};

fn hash(c: &mut Criterion) {
    let salt = Salt::new("Qwen/Qwen3-8B", None, None);
    let mut group = c.benchmark_group("block_hash");
    for block_size in [16usize, 32] {
        let tokens: Vec<u32> = (0..8_192u32)
            .map(|i| i.wrapping_mul(2_654_435_761) % 151_936)
            .collect();
        let blocks = tokens.len() / block_size;
        let mut hasher = BlockHasher::new(block_size);
        let mut out = Vec::with_capacity(blocks);
        group.throughput(Throughput::Elements(blocks as u64));
        group.bench_function(BenchmarkId::new("8k_tokens", block_size), |b| {
            b.iter(|| {
                out.clear();
                hasher.hash_into(black_box(&tokens), salt, &mut out);
                black_box(out.len())
            });
        });
    }
    group.finish();
    c.bench_function("salt", |b| {
        b.iter(|| Salt::new(black_box("Qwen/Qwen3-8B"), black_box(Some("adapter")), None));
    });
}

criterion_group!(benches, hash);
criterion_main!(benches);
