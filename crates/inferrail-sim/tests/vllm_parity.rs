//! The simulator must be indistinguishable from vLLM v0.31.0 on the wire:
//! same stream framing, same JSON shapes and key order, same metric families.

mod common;

use std::fs;
use std::path::{Path, PathBuf};

use common::{TestSim, fast_latency, spawn};
use inferrail_sim::server::SimConfig;
use serde_json::{Value, json};

const MODEL: &str = "meta-llama/Llama-3.1-8B-Instruct";

fn fixture(name: &str) -> String {
    let path: PathBuf = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/vllm")
        .join(name);
    fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

async fn vllm_like() -> TestSim {
    let mut config: SimConfig = common::config(fast_latency());
    config.model = MODEL.to_owned();
    config.served_model_names = vec![MODEL.to_owned()];
    spawn(config).await
}

/// Replaces every string value and number with a placeholder, keeping keys,
/// key order, nesting, `null`, `true` and `false`.
fn mask(json: &str) -> String {
    let bytes = json.as_bytes();
    let mut out = String::with_capacity(json.len());
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'"' => {
                let start = i;
                i += 1;
                while bytes[i] != b'"' {
                    i += if bytes[i] == b'\\' { 2 } else { 1 };
                }
                i += 1;
                let rest = json[i..].trim_start();
                if rest.starts_with(':') {
                    out.push_str(&json[start..i]);
                } else {
                    out.push_str("\"S\"");
                }
            }
            b'-' | b'0'..=b'9' => {
                while i < bytes.len()
                    && matches!(bytes[i], b'-' | b'+' | b'.' | b'e' | b'E' | b'0'..=b'9')
                {
                    i += 1;
                }
                out.push('N');
            }
            c => {
                out.push(c as char);
                i += 1;
            }
        }
    }
    out
}

fn frames(raw: &str) -> Vec<&str> {
    assert!(raw.ends_with("\n\n"), "stream must end with a blank line");
    raw.strip_suffix("\n\n")
        .unwrap_or(raw)
        .split("\n\n")
        .map(|frame| {
            frame
                .strip_prefix("data: ")
                .unwrap_or_else(|| panic!("frame without data prefix: {frame}"))
        })
        .collect()
}

fn assert_same_stream(ours: &[u8], theirs: &str) {
    let ours = std::str::from_utf8(ours).expect("stream is UTF-8");
    let (ours, theirs) = (frames(ours), frames(theirs));
    assert_eq!(ours.len(), theirs.len(), "frame count");
    assert_eq!(ours.last(), Some(&"[DONE]"));
    for (i, (a, b)) in ours.iter().zip(&theirs).enumerate() {
        assert_eq!(mask(a), mask(b), "frame {i}\nours:   {a}\ntheirs: {b}");
    }
}

#[test]
fn mask_keeps_structure_only() {
    assert_eq!(
        mask(r#"{"a":"x\"y","b":[1,-2.5e3],"c":null,"d":true}"#),
        r#"{"a":"S","b":[N,N],"c":null,"d":true}"#
    );
}

#[tokio::test]
async fn chat_stream_matches_vllm_frame_for_frame() {
    let sim = vllm_like().await;
    let body = json!({
        "model": MODEL,
        "messages": [{"role": "user", "content": "What is the capital of France?"}],
        "max_tokens": 4,
        "stream": true,
        "stream_options": {"include_usage": true}
    });
    let (_, _, raw) = sim.stream("/v1/chat/completions", &body).await;
    assert_same_stream(&raw, &fixture("chat_stream.sse"));
}

#[tokio::test]
async fn completion_stream_matches_vllm_frame_for_frame() {
    let sim = vllm_like().await;
    let body = json!({
        "model": MODEL,
        "prompt": "Once upon a time there was",
        "max_tokens": 3,
        "stream": true,
        "stream_options": {"include_usage": true}
    });
    let (_, _, raw) = sim.stream("/v1/completions", &body).await;
    assert_same_stream(&raw, &fixture("completion_stream.sse"));
}

async fn raw_body(sim: &TestSim, path: &str, body: &Value) -> (u16, String) {
    let response = sim.post(path, body).await;
    let status = response.status().as_u16();
    (status, response.text().await.expect("text body"))
}

#[tokio::test]
async fn full_responses_match_vllm_shapes() {
    let sim = vllm_like().await;
    let (_, chat) = raw_body(
        &sim,
        "/v1/chat/completions",
        &json!({"model": MODEL, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 8}),
    )
    .await;
    assert_eq!(mask(&chat), mask(fixture("chat_response.json").trim_end()));

    let (_, completion) = raw_body(
        &sim,
        "/v1/completions",
        &json!({"model": MODEL, "prompt": "Once upon a time", "max_tokens": 7}),
    )
    .await;
    assert_eq!(
        mask(&completion),
        mask(fixture("completion_response.json").trim_end())
    );

    let models = sim
        .client
        .get(sim.url("/v1/models"))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    assert_eq!(mask(&models), mask(fixture("models.json").trim_end()));

    let (status, error) = raw_body(
        &sim,
        "/v1/chat/completions",
        &json!({"model": "gpt-4", "messages": []}),
    )
    .await;
    assert_eq!(status, 404);
    assert_eq!(error, fixture("error.json").trim_end());
}

#[tokio::test]
async fn details_off_matches_vllm_null_details() {
    let mut config: SimConfig = common::config(fast_latency());
    config.enable_prompt_tokens_details = false;
    let sim = spawn(config).await;
    let (_, chat) = raw_body(
        &sim,
        "/v1/chat/completions",
        &json!({"messages": [{"role": "user", "content": "hi"}], "max_tokens": 3}),
    )
    .await;
    assert_eq!(
        mask(&chat),
        mask(fixture("chat_response_no_details.json").trim_end())
    );
}

#[derive(Debug, PartialEq)]
struct Block {
    name: String,
    kind: String,
    help: String,
    samples: Vec<(String, Vec<(String, String)>)>,
}

fn parse_labels(text: &str) -> Vec<(String, String)> {
    let mut out = Vec::new();
    let mut rest = text;
    while !rest.is_empty() {
        let (key, after) = rest.split_once("=\"").expect("label key");
        let mut end = 0;
        let bytes = after.as_bytes();
        while bytes[end] != b'"' {
            end += if bytes[end] == b'\\' { 2 } else { 1 };
        }
        out.push((key.to_owned(), after[..end].to_owned()));
        rest = after[end + 1..].trim_start_matches(',');
    }
    out
}

fn parse_exposition(text: &str) -> Vec<Block> {
    let mut blocks: Vec<Block> = Vec::new();
    let mut help = String::new();
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("# HELP ") {
            help = rest.split_once(' ').map_or("", |(_, h)| h).to_owned();
        } else if let Some(rest) = line.strip_prefix("# TYPE ") {
            let (name, kind) = rest.split_once(' ').expect("type line");
            assert!(!help.is_empty(), "{name} has no HELP");
            blocks.push(Block {
                name: name.to_owned(),
                kind: kind.to_owned(),
                help: std::mem::take(&mut help),
                samples: Vec::new(),
            });
        } else if !line.is_empty() {
            let (series, value) = line.rsplit_once(' ').expect("sample value");
            assert!(
                value == "+Inf"
                    || value == "-Inf"
                    || value == "NaN"
                    || value.parse::<f64>().is_ok(),
                "bad value {value}"
            );
            let (name, labels) = match series.split_once('{') {
                Some((name, labels)) => (name, parse_labels(labels.trim_end_matches('}'))),
                None => (series, Vec::new()),
            };
            blocks
                .last_mut()
                .expect("sample before TYPE")
                .samples
                .push((name.to_owned(), labels));
        }
    }
    blocks
}

#[tokio::test]
async fn metrics_match_vllm_names_types_labels_and_order() {
    let sim = vllm_like().await;
    for i in 0..3 {
        sim.post_json(
            "/v1/chat/completions",
            &json!({"model": MODEL, "messages": [{"role": "user", "content": format!("q{i}")}], "max_tokens": 3}),
        )
        .await;
    }
    let response = sim.client.get(sim.url("/metrics")).send().await.unwrap();
    assert_eq!(
        response.headers()["content-type"],
        "text/plain; version=0.0.4; charset=utf-8"
    );
    let ours = parse_exposition(&response.text().await.unwrap());
    let theirs: Vec<Block> = parse_exposition(&fixture("metrics.txt"))
        .into_iter()
        .filter(|b| !b.name.starts_with("vllm:iteration_tokens_total"))
        .collect();

    let names = |blocks: &[Block]| blocks.iter().map(|b| b.name.clone()).collect::<Vec<_>>();
    assert_eq!(names(&ours), names(&theirs), "family order");

    for (a, b) in ours.iter().zip(&theirs) {
        assert_eq!((&a.kind, &a.help), (&b.kind, &b.help), "{}", a.name);
        if a.name == "vllm:cache_config_info" {
            let keys = |s: &(String, Vec<(String, String)>)| {
                s.1.iter().map(|(k, _)| k.clone()).collect::<Vec<_>>()
            };
            let theirs = keys(&b.samples[0]);
            assert!(keys(&a.samples[0]).iter().all(|k| theirs.contains(k)));
            continue;
        }
        assert_eq!(a.samples, b.samples, "{}", a.name);
    }

    let required = [
        "vllm:num_requests_running",
        "vllm:num_requests_waiting",
        "vllm:kv_cache_usage_perc",
        "vllm:prefix_cache_queries_total",
        "vllm:prefix_cache_hits_total",
        "vllm:prompt_tokens_total",
        "vllm:generation_tokens_total",
        "vllm:request_success_total",
        "vllm:time_to_first_token_seconds",
        "vllm:e2e_request_latency_seconds",
    ];
    for name in required {
        assert!(ours.iter().any(|b| b.name == name), "missing {name}");
    }
}
