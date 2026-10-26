use std::hint::black_box;
use std::io::Read;
use std::path::{Path, PathBuf};

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use inferrail_tokenize::{Endpoint, ModelTokenizer, TokenizerOptions};

const WORDS: [&str; 16] = [
    "the", "router", "keeps", "every", "prefix", "warm", "while", "replicas", "share", "load",
    "and", "latency", "stays", "low", "under", "pressure",
];

fn model_dir(family: &str) -> PathBuf {
    let source = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/tokenizers")
        .join(family);
    let dir = std::env::temp_dir().join(format!("inferrail-bench-{family}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("scratch dir");
    let mut json = Vec::new();
    flate2::read::GzDecoder::new(
        std::fs::File::open(source.join("tokenizer.json.gz")).expect("fixture"),
    )
    .read_to_end(&mut json)
    .expect("gunzip");
    std::fs::write(dir.join("tokenizer.json"), json).expect("write");
    std::fs::copy(
        source.join("tokenizer_config.json"),
        dir.join("tokenizer_config.json"),
    )
    .expect("copy");
    dir
}

/// A system prompt and a conversation that render to about `tokens`
/// tokens; only the final question depends on `seed`.
fn chat_body(tokens: usize, seed: usize) -> String {
    let text = |n: usize, k: usize| -> String {
        (0..n)
            .map(|i| WORDS[(i * 7 + k * 13) % WORDS.len()])
            .collect::<Vec<_>>()
            .join(" ")
    };
    let mut messages = vec![serde_json::json!({"role": "system", "content": text(tokens / 2, 0)})];
    let turns = 4;
    for turn in 0..turns {
        messages.push(
            serde_json::json!({"role": "user", "content": text(tokens / (4 * turns), turn + 1)}),
        );
        messages.push(serde_json::json!({"role": "assistant", "content": text(tokens / (4 * turns), turn + 2)}));
    }
    messages.push(serde_json::json!({"role": "user", "content": format!("question {seed}: {}", text(24, seed))}));
    serde_json::json!({"model": "m", "messages": messages}).to_string()
}

fn tokenize(c: &mut Criterion) {
    let dir = model_dir("qwen3");
    let load = |cache_tokens| {
        ModelTokenizer::load(&TokenizerOptions {
            source: dir.to_string_lossy().into_owned(),
            cache_tokens,
            ..TokenizerOptions::default()
        })
        .expect("tokenizer")
    };
    let (cold, warm) = (load(0), load(1 << 22));
    let mut group = c.benchmark_group("tokenize_chat");
    for target in [1_000usize, 8_000, 32_000] {
        let bodies: Vec<String> = (0..64).map(|seed| chat_body(target, seed)).collect();
        let tokens = cold
            .encode(Endpoint::Chat, bodies[0].as_bytes())
            .expect("encode")
            .tokens
            .len();
        group.throughput(Throughput::Elements(tokens as u64));
        group.bench_with_input(
            BenchmarkId::new("qwen3_cold", tokens),
            &bodies[0],
            |b, body| {
                b.iter(|| {
                    cold.encode(Endpoint::Chat, black_box(body.as_bytes()))
                        .expect("encode")
                });
            },
        );
        for body in &bodies {
            warm.encode(Endpoint::Chat, body.as_bytes())
                .expect("encode");
        }
        let mut next = 0;
        group.bench_with_input(
            BenchmarkId::new("qwen3_new_question", tokens),
            &bodies,
            |b, bodies| {
                b.iter(|| {
                    next = (next + 1) % bodies.len();
                    warm.encode(Endpoint::Chat, black_box(bodies[next].as_bytes()))
                        .expect("encode")
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new("qwen3_render", tokens),
            &bodies[0],
            |b, body| {
                b.iter(|| {
                    cold.render_chat(black_box(body.as_bytes()))
                        .expect("render")
                });
            },
        );
    }
    group.finish();
    let sim = ModelTokenizer::sim();
    let mut group = c.benchmark_group("tokenize_chat_sim");
    for target in [1_000usize, 8_000, 32_000] {
        let body = chat_body(target, 1);
        let tokens = sim
            .encode(Endpoint::Chat, body.as_bytes())
            .expect("encode")
            .tokens
            .len();
        group.throughput(Throughput::Elements(tokens as u64));
        group.bench_with_input(BenchmarkId::new("sim", tokens), &body, |b, body| {
            b.iter(|| {
                sim.encode(Endpoint::Chat, black_box(body.as_bytes()))
                    .expect("encode")
            });
        });
    }
    group.finish();
    let _ = std::fs::remove_dir_all(&dir);
}

criterion_group!(benches, tokenize);
criterion_main!(benches);
