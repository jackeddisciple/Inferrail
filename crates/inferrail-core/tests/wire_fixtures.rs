use std::fs;
use std::path::{Path, PathBuf};

use inferrail_core::openai::{
    ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse, CompletionChunk,
    CompletionRequest, CompletionResponse, ErrorResponse, ModelList,
};
use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::Value;

fn fixtures_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures")
}

fn payloads(dir: &str) -> Vec<(String, String)> {
    let mut files: Vec<PathBuf> = fs::read_dir(fixtures_root().join(dir))
        .expect("fixture directory exists")
        .map(|entry| entry.expect("readable directory entry").path())
        .filter(|path| {
            matches!(
                path.extension().and_then(|e| e.to_str()),
                Some("json" | "jsonl")
            )
        })
        .collect();
    files.sort();
    let mut out = Vec::new();
    for path in files {
        let name = path
            .file_name()
            .expect("fixture has a file name")
            .to_string_lossy()
            .into_owned();
        let text = fs::read_to_string(&path).expect("fixture is readable UTF-8");
        if name.ends_with(".jsonl") {
            for (i, line) in text.lines().filter(|l| !l.trim().is_empty()).enumerate() {
                out.push((format!("{name}:{}", i + 1), line.to_owned()));
            }
        } else {
            out.push((name, text));
        }
    }
    out
}

/// Canonical form: compact JSON with object keys sorted.
fn canonical(value: &Value) -> String {
    serde_json::to_string(value).expect("a Value always serializes")
}

fn assert_round_trip<T: Serialize + DeserializeOwned + PartialEq + std::fmt::Debug>(
    label: &str,
    text: &str,
    exact: bool,
) {
    let original: Value = serde_json::from_str(text).expect("fixture is valid JSON");
    let typed: T = serde_json::from_str(text).unwrap_or_else(|e| panic!("{label}: {e}"));
    let rendered = serde_json::to_string(&typed).expect("typed value serializes");
    let back: Value = serde_json::from_str(&rendered).expect("rendered JSON parses");
    assert_eq!(
        canonical(&original),
        canonical(&back),
        "{label} changed in a round trip"
    );
    assert_eq!(
        serde_json::from_str::<T>(&rendered).expect("rendered JSON parses as the same type"),
        typed,
        "{label} is not idempotent"
    );
    if exact {
        assert_eq!(rendered, text.trim_end(), "{label} is not byte exact");
    }
}

fn check(label: &str, text: &str, exact: bool) {
    let kind = label.split(['.', ':']).next().unwrap_or(label);
    match kind {
        k if k.starts_with("chat_request") => {
            assert_round_trip::<ChatCompletionRequest>(label, text, exact)
        }
        k if k.starts_with("completion_request") => {
            assert_round_trip::<CompletionRequest>(label, text, exact)
        }
        k if k.starts_with("chat_response") => {
            assert_round_trip::<ChatCompletionResponse>(label, text, exact)
        }
        k if k.starts_with("chat_chunks") => {
            assert_round_trip::<ChatCompletionChunk>(label, text, exact)
        }
        k if k.starts_with("completion_response") => {
            assert_round_trip::<CompletionResponse>(label, text, exact)
        }
        k if k.starts_with("completion_chunks") => {
            assert_round_trip::<CompletionChunk>(label, text, exact)
        }
        k if k.starts_with("models") => assert_round_trip::<ModelList>(label, text, exact),
        k if k.starts_with("error") => assert_round_trip::<ErrorResponse>(label, text, exact),
        other => panic!("no wire type mapped for fixture {other}"),
    }
}

#[test]
fn openai_payloads_round_trip_canonically() {
    let fixtures = payloads("openai");
    assert!(fixtures.len() >= 14, "expected the full OpenAI fixture set");
    for (label, text) in &fixtures {
        check(label, text, false);
    }
}

#[test]
fn vllm_payloads_round_trip_byte_for_byte() {
    let fixtures = payloads("vllm");
    assert!(fixtures.len() >= 15, "expected the full vLLM fixture set");
    for (label, text) in &fixtures {
        check(label, text, true);
    }
}

#[test]
fn canonical_comparison_catches_lost_fields() {
    let text = r#"{"messages":[],"model":"m","stream":null}"#;
    let mut typed: ChatCompletionRequest = serde_json::from_str(text).unwrap();
    typed.stream = inferrail_core::openai::Nullable::Absent;
    let back: Value = serde_json::to_value(&typed).unwrap();
    assert_ne!(
        canonical(&serde_json::from_str(text).unwrap()),
        canonical(&back)
    );
}

#[test]
fn canonical_comparison_catches_number_drift() {
    let original: Value = serde_json::from_str(r#"{"t":1}"#).unwrap();
    let drifted: Value = serde_json::from_str(r#"{"t":1.0}"#).unwrap();
    assert_ne!(canonical(&original), canonical(&drifted));
}
