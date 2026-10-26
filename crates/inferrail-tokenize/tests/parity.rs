//! Golden parity: for every request vLLM v0.31.0 accepts, the router
//! computes exactly the token ids the engine computes. Requests the engine
//! rejects carry no expectation, since nothing is ever cached for them.

use std::io::Read;
use std::path::{Path, PathBuf};

use inferrail_tokenize::{Clock, Endpoint, ModelTokenizer, Parity, TokenizerOptions};
use serde::Deserialize;
use serde_json::value::RawValue;

#[derive(Deserialize)]
struct Golden {
    id: u64,
    endpoint: String,
    body: Box<RawValue>,
    #[serde(default)]
    text: Option<String>,
    #[serde(default)]
    ids: Option<Vec<u32>>,
}

fn fixtures(family: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/tokenizers")
        .join(family)
}

fn gunzip(path: &Path) -> Vec<u8> {
    let file = std::fs::File::open(path).expect("fixture exists");
    let mut out = Vec::new();
    flate2::read::GzDecoder::new(file)
        .read_to_end(&mut out)
        .expect("fixture is gzip");
    out
}

/// The committed files laid out as a model directory.
fn model_dir(family: &str) -> PathBuf {
    let dir =
        std::env::temp_dir().join(format!("inferrail-parity-{family}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("scratch directory");
    let source = fixtures(family);
    let json = gunzip(&source.join("tokenizer.json.gz"));
    std::fs::write(dir.join("tokenizer.json"), json).expect("tokenizer written");
    for name in ["tokenizer_config.json", "special_tokens_map.json"] {
        if source.join(name).is_file() {
            std::fs::copy(source.join(name), dir.join(name)).expect("config copied");
        }
    }
    dir
}

fn goldens(family: &str) -> Vec<Golden> {
    gunzip(&fixtures(family).join("goldens.jsonl.gz"))
        .split(|b| *b == b'\n')
        .filter(|line| !line.is_empty())
        .map(|line| serde_json::from_slice(line).expect("golden line"))
        .collect()
}

fn check(family: &str) {
    let dir = model_dir(family);
    let goldens = goldens(family);
    for cache_tokens in [0, 1 << 20] {
        let tokenizer = ModelTokenizer::load(&TokenizerOptions {
            source: dir.to_string_lossy().into_owned(),
            clock: Clock::Fixed(jiff::civil::date(2026, 1, 15).at(9, 30, 0, 0)),
            cache_tokens,
            ..TokenizerOptions::default()
        })
        .expect("tokenizer loads");
        assert_eq!(tokenizer.parity(), Parity::Verified);
        for pass in ["cold", "warm"] {
            compare(
                family,
                &tokenizer,
                &goldens,
                &format!("{pass}, cache of {cache_tokens}"),
            );
        }
        if cache_tokens > 0 {
            assert!(
                tokenizer.cache_stats().hits > 0,
                "{family}: the cache was never used"
            );
        }
    }
    std::fs::remove_dir_all(&dir).expect("scratch removed");
}

fn compare(family: &str, tokenizer: &ModelTokenizer, goldens: &[Golden], label: &str) {
    let (mut checked, mut failures) = (0, Vec::new());
    for golden in goldens {
        let Some(want) = &golden.ids else { continue };
        checked += 1;
        let endpoint = match golden.endpoint.as_str() {
            "chat" => Endpoint::Chat,
            _ => Endpoint::Completion,
        };
        let body = golden.body.get().as_bytes();
        match tokenizer.encode(endpoint, body) {
            Ok(got) if &got.tokens == want => {}
            Ok(got) => {
                let at = got
                    .tokens
                    .iter()
                    .zip(want)
                    .take_while(|(a, b)| a == b)
                    .count();
                let text = match (&golden.text, tokenizer.render_chat(body)) {
                    (Some(want), Ok(text)) if &text != want => {
                        let c = text
                            .chars()
                            .zip(want.chars())
                            .take_while(|(a, b)| a == b)
                            .count();
                        let near = |s: &str| {
                            s.chars()
                                .skip(c.saturating_sub(30))
                                .take(80)
                                .collect::<String>()
                        };
                        format!(
                            "text differs at char {c}: got {:?} want {:?}",
                            near(&text),
                            near(want)
                        )
                    }
                    _ => "same text".to_owned(),
                };
                failures.push(format!(
                    "#{} {}: {} vs {} tokens, first difference at {at}; {text}",
                    golden.id,
                    golden.endpoint,
                    got.tokens.len(),
                    want.len()
                ));
            }
            Err(err) => failures.push(format!("#{} {}: {err}", golden.id, golden.endpoint)),
        }
    }
    assert!(checked >= 200, "{family}: only {checked} goldens with ids");
    assert!(
        failures.is_empty(),
        "{family} ({label}): {} of {checked} requests differ\n{}",
        failures.len(),
        failures.join("\n")
    );
}

#[test]
fn qwen3_matches_transformers() {
    check("qwen3");
}

#[test]
fn granite3_matches_transformers() {
    check("granite3");
}

#[test]
fn hermes2pro_matches_transformers() {
    check("hermes2pro");
}
