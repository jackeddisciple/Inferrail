//! Whatever a client sends, tokenizing it returns ids or an error and never
//! panics, for the simulator's tokenizer and a real one.

use std::io::Read;
use std::path::Path;
use std::sync::OnceLock;

use inferrail_tokenize::{Endpoint, ModelTokenizer, TokenizerOptions};
use proptest::prelude::*;
use serde_json::{Value, json};

fn qwen3() -> &'static ModelTokenizer {
    static TOKENIZER: OnceLock<ModelTokenizer> = OnceLock::new();
    TOKENIZER.get_or_init(|| {
        let source =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tokenizers/qwen3");
        let dir = std::env::temp_dir().join(format!("inferrail-robust-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("scratch directory");
        let mut json = Vec::new();
        flate2::read::GzDecoder::new(
            std::fs::File::open(source.join("tokenizer.json.gz")).expect("fixture"),
        )
        .read_to_end(&mut json)
        .expect("gzip");
        std::fs::write(dir.join("tokenizer.json"), json).expect("tokenizer written");
        std::fs::copy(
            source.join("tokenizer_config.json"),
            dir.join("tokenizer_config.json"),
        )
        .expect("config copied");
        ModelTokenizer::load(&TokenizerOptions {
            source: dir.to_string_lossy().into_owned(),
            ..TokenizerOptions::default()
        })
        .expect("tokenizer loads")
    })
}

fn value() -> impl Strategy<Value = Value> {
    let leaf = prop_oneof![
        Just(Value::Null),
        any::<bool>().prop_map(Value::from),
        any::<i64>().prop_map(Value::from),
        (-1e300f64..1e300).prop_map(Value::from),
        "\\PC{0,30}".prop_map(Value::from),
        prop::sample::select(vec![
            "<|im_start|>",
            "<|im_end|>",
            "</think>",
            "{{ x }}",
            "{% if %}"
        ])
        .prop_map(Value::from),
    ];
    leaf.prop_recursive(4, 48, 6, |inner| {
        prop_oneof![
            prop::collection::vec(inner.clone(), 0..6).prop_map(Value::Array),
            prop::collection::btree_map(
                prop::sample::select(vec![
                    "role",
                    "content",
                    "tool_calls",
                    "function",
                    "arguments",
                    "name",
                    "type",
                    "text",
                    "tools",
                    "parameters",
                    "reasoning",
                    "tool_call_id",
                    "x",
                ])
                .prop_map(str::to_owned),
                inner,
                0..6,
            )
            .prop_map(|m| Value::Object(m.into_iter().collect())),
        ]
    })
}

fn message() -> impl Strategy<Value = Value> {
    (
        prop::sample::select(vec![
            "system",
            "user",
            "assistant",
            "tool",
            "developer",
            "other",
        ]),
        value(),
        prop::option::of(value()),
    )
        .prop_map(|(role, content, calls)| {
            let mut m = json!({"role": role, "content": content});
            if let Some(calls) = calls {
                m["tool_calls"] = calls;
            }
            m
        })
}

fn body() -> impl Strategy<Value = Value> {
    (
        prop::collection::vec(message(), 0..6),
        prop::option::of(value()),
        prop::option::of(value()),
        any::<bool>(),
        any::<bool>(),
    )
        .prop_map(|(messages, tools, kwargs, add, cont)| {
            let mut b = json!({"model": "m", "messages": messages, "add_generation_prompt": add, "continue_final_message": cont});
            if let Some(tools) = tools {
                b["tools"] = tools;
            }
            if let Some(kwargs) = kwargs {
                b["chat_template_kwargs"] = kwargs;
            }
            b
        })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(400))]

    #[test]
    fn random_chat_bodies_never_panic(body in body()) {
        let bytes = body.to_string();
        for tokenizer in [qwen3(), &ModelTokenizer::sim()] {
            if let Ok(encoded) = tokenizer.encode(Endpoint::Chat, bytes.as_bytes()) {
                prop_assert!(encoded.tokens.len() <= 8 * bytes.len() + 64);
            }
        }
    }

    #[test]
    fn random_completion_bodies_never_panic(prompt in value(), special in any::<Option<bool>>()) {
        let mut body = json!({"model": "m", "prompt": prompt});
        if let Some(special) = special {
            body["add_special_tokens"] = json!(special);
        }
        for tokenizer in [qwen3(), &ModelTokenizer::sim()] {
            let _ = tokenizer.encode(Endpoint::Completion, body.to_string().as_bytes());
        }
    }

    #[test]
    fn arbitrary_bytes_never_panic(bytes in prop::collection::vec(any::<u8>(), 0..512)) {
        for endpoint in [Endpoint::Chat, Endpoint::Completion] {
            let _ = qwen3().encode(endpoint, &bytes);
            let _ = ModelTokenizer::sim().encode(endpoint, &bytes);
        }
    }
}
