use std::hint::black_box;
use std::path::PathBuf;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};

fn fixture(name: &str) -> String {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures")
        .join(name);
    std::fs::read_to_string(path).expect("fixture")
}

fn bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("parse");
    for name in ["vllm/metrics.txt", "sglang/metrics.txt"] {
        let page = fixture(name);
        group.throughput(Throughput::Bytes(page.len() as u64));
        group.bench_function(name, |b| b.iter(|| inferrail_load::parse(black_box(&page))));
    }
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
