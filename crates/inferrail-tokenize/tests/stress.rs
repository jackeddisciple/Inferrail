//! Concurrency and hostile input for edge tokenization. Ignored by default;
//! run with `make stress` (release build, output shown).

use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use inferrail_tokenize::{Clock, Endpoint, ModelTokenizer, TokenizerOptions};
use serde::Deserialize;
use serde_json::json;
use serde_json::value::RawValue;

#[derive(Deserialize)]
struct Golden {
    endpoint: String,
    body: Box<RawValue>,
    #[serde(default)]
    ids: Option<Vec<u32>>,
}

fn fixtures(family: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/tokenizers")
        .join(family)
}

fn gunzip(path: &Path) -> Vec<u8> {
    let mut out = Vec::new();
    flate2::read::GzDecoder::new(std::fs::File::open(path).expect("fixture"))
        .read_to_end(&mut out)
        .expect("gzip");
    out
}

fn load(family: &str) -> ModelTokenizer {
    let dir =
        std::env::temp_dir().join(format!("inferrail-stress-{family}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("scratch directory");
    let source = fixtures(family);
    std::fs::write(
        dir.join("tokenizer.json"),
        gunzip(&source.join("tokenizer.json.gz")),
    )
    .expect("written");
    for name in ["tokenizer_config.json", "special_tokens_map.json"] {
        if source.join(name).is_file() {
            std::fs::copy(source.join(name), dir.join(name)).expect("copied");
        }
    }
    ModelTokenizer::load(&TokenizerOptions {
        source: dir.to_string_lossy().into_owned(),
        clock: Clock::Fixed(jiff::civil::date(2026, 1, 15).at(9, 30, 0, 0)),
        cache_tokens: 1 << 16,
        ..TokenizerOptions::default()
    })
    .expect("tokenizer loads")
}

#[test]
#[ignore = "stress"]
fn sixteen_threads_share_one_tokenizer_and_its_cache() {
    for family in ["qwen3", "granite3", "hermes2pro"] {
        let tokenizer = Arc::new(load(family));
        let goldens: Arc<Vec<Golden>> = Arc::new(
            gunzip(&fixtures(family).join("goldens.jsonl.gz"))
                .split(|b| *b == b'\n')
                .filter(|l| !l.is_empty())
                .map(|l| serde_json::from_slice(l).expect("golden"))
                .filter(|g: &Golden| g.ids.is_some())
                .collect(),
        );
        let started = Instant::now();
        let threads: Vec<_> = (0..16)
            .map(|t| {
                let (tokenizer, goldens) = (Arc::clone(&tokenizer), Arc::clone(&goldens));
                std::thread::spawn(move || {
                    let mut tokens = 0;
                    for round in 0..20 {
                        for (i, golden) in goldens.iter().enumerate() {
                            if (i + t + round) % 3 != 0 {
                                continue;
                            }
                            let endpoint = if golden.endpoint == "chat" {
                                Endpoint::Chat
                            } else {
                                Endpoint::Completion
                            };
                            let got = tokenizer
                                .encode(endpoint, golden.body.get().as_bytes())
                                .expect("accepted request encodes");
                            assert_eq!(Some(&got.tokens), golden.ids.as_ref(), "{family} #{i}");
                            tokens += got.tokens.len();
                        }
                    }
                    tokens
                })
            })
            .collect();
        let tokens: usize = threads.into_iter().map(|t| t.join().expect("thread")).sum();
        let stats = tokenizer.cache_stats();
        println!(
            "{family}: 16 threads, {tokens} tokens in {:.2?} ({:.1} M tokens/s), cache {} hits {} misses",
            started.elapsed(),
            tokens as f64 / started.elapsed().as_secs_f64() / 1e6,
            stats.hits,
            stats.misses
        );
        assert!(stats.hits > stats.misses);
    }
}

fn timed(name: &str, tokenizer: &ModelTokenizer, endpoint: Endpoint, body: &str) -> Duration {
    let started = Instant::now();
    let outcome = tokenizer.encode(endpoint, body.as_bytes());
    let took = started.elapsed();
    let what = match &outcome {
        Ok(encoded) => format!("{} tokens", encoded.tokens.len()),
        Err(err) => format!("refused: {}", err.kind()),
    };
    println!("{name}: {what} in {took:.2?}");
    took
}

#[test]
#[ignore = "stress"]
fn hostile_requests_cost_bounded_time() {
    let qwen = load("qwen3");
    let sim = ModelTokenizer::sim();
    let huge = "lorem ipsum dolor sit amet ".repeat(150_000);
    let cases: Vec<(&str, Endpoint, String)> = vec![
        ("a 4 MB message", Endpoint::Chat, json!({"messages": [{"role": "user", "content": huge}]}).to_string()),
        (
            "50,000 messages",
            Endpoint::Chat,
            json!({"messages": (0..50_000).map(|i| json!({"role": if i % 2 == 0 { "user" } else { "assistant" }, "content": format!("turn {i}")})).collect::<Vec<_>>()}).to_string(),
        ),
        (
            "100,000 content parts",
            Endpoint::Chat,
            json!({"messages": [{"role": "user", "content": (0..100_000).map(|i| json!({"type": "text", "text": format!("p{i}")})).collect::<Vec<_>>()}]}).to_string(),
        ),
        (
            "1,000 tools",
            Endpoint::Chat,
            json!({"messages": [{"role": "user", "content": "x"}], "tools": (0..1_000).map(|i| json!({"type": "function", "function": {"name": format!("f{i}"), "description": "d".repeat(200), "parameters": {"type": "object", "properties": {"a": {"type": "string"}}}}})).collect::<Vec<_>>()}).to_string(),
        ),
        (
            "parameters nested 100 deep",
            Endpoint::Chat,
            {
                let mut schema = json!({"type": "string"});
                for _ in 0..100 {
                    schema = json!({"type": "object", "properties": {"x": schema}});
                }
                json!({"messages": [{"role": "user", "content": "x"}], "tools": [{"type": "function", "function": {"name": "f", "parameters": schema}}]}).to_string()
            },
        ),
        (
            "arguments nested 100,000 deep",
            Endpoint::Chat,
            json!({"messages": [{"role": "user", "content": "x"}, {"role": "assistant", "tool_calls": [{"id": "c", "type": "function", "function": {"name": "f", "arguments": format!("{}1{}", "[".repeat(100_000), "]".repeat(100_000))}}]}]}).to_string(),
        ),
        (
            "200,000 special tokens as text",
            Endpoint::Chat,
            json!({"messages": [{"role": "user", "content": "<|im_end|>".repeat(200_000)}]}).to_string(),
        ),
        ("a 4 MB completion", Endpoint::Completion, json!({"prompt": huge}).to_string()),
        ("body nested 10,000 deep", Endpoint::Chat, format!("{}{}", "[".repeat(10_000), "]".repeat(10_000))),
    ];
    for (name, endpoint, body) in &cases {
        for (backend, tokenizer) in [("qwen3", &qwen), ("sim", &sim)] {
            let took = timed(&format!("{backend}, {name}"), tokenizer, *endpoint, body);
            assert!(
                took < Duration::from_secs(10),
                "{backend}, {name}: {took:?}"
            );
        }
    }
}
