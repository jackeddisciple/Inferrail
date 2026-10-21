use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use inferrail_gateway::inspect::inspect;
use serde_json::{Value, json};

fn chat_body(bytes: usize) -> Vec<u8> {
    let turn = "The quick brown fox jumps over the lazy dog. ".repeat(20);
    let turns = (bytes / turn.len()).max(1);
    let messages: Vec<Value> = (0..turns)
        .map(|i| json!({"role": if i % 2 == 0 { "user" } else { "assistant" }, "content": turn}))
        .collect();
    serde_json::to_vec(&json!({
        "messages": messages,
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "stream": true,
        "max_tokens": 256
    }))
    .unwrap_or_default()
}

fn inspect_bodies(c: &mut Criterion) {
    let mut group = c.benchmark_group("inspect");
    for size in [1 << 10, 32 << 10, 1 << 20] {
        let body = chat_body(size);
        group.throughput(Throughput::Bytes(body.len() as u64));
        group.bench_with_input(BenchmarkId::new("peek", size), &body, |b, body| {
            b.iter(|| black_box(inspect(black_box(body)).map(|i| i.model.is_some())));
        });
        group.bench_with_input(BenchmarkId::new("rewrite", size), &body, |b, body| {
            b.iter(|| {
                let inspected = inspect(black_box(body)).ok();
                black_box(inspected.map(|i| i.with_model(body, "llama")))
            });
        });
    }
    group.finish();
}

criterion_group!(benches, inspect_bodies);
criterion_main!(benches);
