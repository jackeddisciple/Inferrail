use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use inferrail_sim::cache::{PrefixCache, block_hashes};
use inferrail_sim::tokenizer;

fn prompt(tokens: usize) -> Vec<u32> {
    (0..tokens as u32)
        .map(|t| t.wrapping_mul(2_654_435_761))
        .collect()
}

fn hashing(c: &mut Criterion) {
    let mut group = c.benchmark_group("block_hashes");
    for tokens in [1_024, 8_192, 32_768] {
        let ids = prompt(tokens);
        group.throughput(Throughput::Elements((tokens / 16) as u64));
        group.bench_with_input(BenchmarkId::from_parameter(tokens), &ids, |b, ids| {
            b.iter(|| block_hashes(black_box(ids), 16))
        });
    }
    group.finish();
}

fn lookup(c: &mut Criterion) {
    let mut group = c.benchmark_group("prefix_cache_match");
    for tokens in [1_024, 8_192, 32_768] {
        let hashes = block_hashes(&prompt(tokens), 16);
        let mut cache = PrefixCache::new(1 << 16);
        cache.insert(&hashes);
        group.throughput(Throughput::Elements(hashes.len() as u64));
        group.bench_with_input(BenchmarkId::from_parameter(tokens), &hashes, |b, hashes| {
            b.iter(|| cache.match_prefix(black_box(hashes)))
        });
    }
    group.finish();
}

fn tokenize(c: &mut Criterion) {
    let text =
        "The router sends each request to the replica that already holds its prefix. ".repeat(400);
    let mut group = c.benchmark_group("sim_tokenize");
    group.throughput(Throughput::Bytes(text.len() as u64));
    group.bench_function("30k_chars", |b| {
        b.iter(|| tokenizer::tokenize(black_box(&text)))
    });
    group.finish();
}

criterion_group!(benches, hashing, lookup, tokenize);
criterion_main!(benches);
