//! Whatever a wire type accepts, it must give back unchanged.

use inferrail_core::openai::{
    ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse, CompletionRequest,
    ErrorResponse, ModelList, Usage,
};
use proptest::prelude::*;
use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::{Map, Number, Value, json};

fn leaf() -> impl Strategy<Value = Value> {
    prop_oneof![
        Just(Value::Null),
        any::<bool>().prop_map(Value::Bool),
        any::<i64>().prop_map(Value::from),
        any::<u64>().prop_map(Value::from),
        any::<f64>()
            .prop_filter_map("finite", Number::from_f64)
            .prop_map(Value::Number),
        "\\PC{0,12}".prop_map(Value::String),
    ]
}

fn any_json() -> impl Strategy<Value = Value> {
    leaf().prop_recursive(3, 24, 4, |inner| {
        prop_oneof![
            proptest::collection::vec(inner.clone(), 0..4).prop_map(Value::Array),
            proptest::collection::btree_map("[a-z_]{1,8}", inner, 0..4)
                .prop_map(|m| Value::Object(m.into_iter().collect())),
        ]
    })
}

fn message() -> impl Strategy<Value = Value> {
    let role = prop_oneof![
        3 => Just(json!("system")),
        3 => Just(json!("user")),
        3 => Just(json!("assistant")),
        1 => Just(json!("tool")),
        1 => "[a-z]{1,6}".prop_map(Value::String),
        1 => any_json(),
    ];
    let content = prop_oneof![
        6 => "\\PC{0,20}".prop_map(Value::String),
        1 => Just(Value::Null),
        2 => proptest::collection::vec(
            ("\\PC{0,10}", prop::bool::ANY).prop_map(|(t, image)| if image {
                json!({"type": "image_url", "image_url": {"url": t}})
            } else {
                json!({"type": "text", "text": t})
            }),
            0..3
        )
        .prop_map(Value::Array),
        1 => any_json(),
    ];
    (
        role,
        content,
        proptest::option::of("[a-z]{1,8}".prop_map(Value::String)),
    )
        .prop_map(|(role, content, extra)| {
            let mut m = Map::new();
            m.insert("role".into(), role);
            m.insert("content".into(), content);
            if let Some(extra) = extra {
                m.insert("name".into(), extra);
            }
            Value::Object(m)
        })
}

fn sometimes(
    key: &'static str,
    value: impl Strategy<Value = Value>,
) -> impl Strategy<Value = Option<(String, Value)>> {
    proptest::option::of(
        prop_oneof![3 => value, 1 => any_json()].prop_map(move |v| (key.to_owned(), v)),
    )
}

fn chat_request() -> impl Strategy<Value = Value> {
    (
        proptest::collection::vec(message(), 0..4),
        (
            sometimes("model", "[a-z/-]{1,12}".prop_map(Value::String)),
            sometimes("max_tokens", (0u64..5000).prop_map(Value::from)),
            sometimes(
                "temperature",
                prop_oneof![Just(json!(1)), Just(json!(0.7)), Just(json!(0.0))],
            ),
            sometimes("stream", any::<bool>().prop_map(Value::Bool)),
            sometimes("stream_options", Just(json!({"include_usage": true}))),
            sometimes(
                "stop",
                prop_oneof![Just(json!("\n")), Just(json!(["a", "b"]))],
            ),
            sometimes("seed", any::<i64>().prop_map(Value::from)),
            sometimes(
                "tools",
                Just(json!([{"type": "function", "function": {"name": "f"}}])),
            ),
        ),
        proptest::collection::btree_map("x_[a-z]{1,6}", any_json(), 0..3),
    )
        .prop_map(|(messages, known, unknown)| {
            let mut m = Map::new();
            m.insert("messages".into(), Value::Array(messages));
            let (a, b, c, d, e, f, g, h) = known;
            for (k, v) in [a, b, c, d, e, f, g, h].into_iter().flatten() {
                m.insert(k, v);
            }
            m.extend(unknown);
            Value::Object(m)
        })
}

fn completion_request() -> impl Strategy<Value = Value> {
    let prompt = prop_oneof![
        "\\PC{0,20}".prop_map(Value::String),
        proptest::collection::vec(any::<u32>(), 0..6).prop_map(|v| json!(v)),
        proptest::collection::vec(proptest::collection::vec(any::<u32>(), 0..4), 0..3)
            .prop_map(|v| json!(v)),
        proptest::collection::vec("\\PC{0,6}", 0..3).prop_map(|v| json!(v)),
        any_json(),
    ];
    (
        proptest::option::of(prompt),
        sometimes("logprobs", (0u32..5).prop_map(Value::from)),
        sometimes("echo", any::<bool>().prop_map(Value::Bool)),
        proptest::collection::btree_map("x_[a-z]{1,6}", any_json(), 0..3),
    )
        .prop_map(|(prompt, logprobs, echo, unknown)| {
            let mut m = Map::new();
            if let Some(prompt) = prompt {
                m.insert("prompt".into(), prompt);
            }
            for (k, v) in [logprobs, echo].into_iter().flatten() {
                m.insert(k, v);
            }
            m.extend(unknown);
            Value::Object(m)
        })
}

fn usage() -> impl Strategy<Value = Value> {
    (
        any::<u32>(),
        any::<u32>(),
        prop_oneof![Just(Value::Null), any::<u32>().prop_map(Value::from)],
        prop_oneof![
            Just(Value::Null),
            Just(json!({})),
            any::<u16>().prop_map(|n| json!({"cached_tokens": n})),
            any::<u16>().prop_map(|n| json!({"cached_tokens": n, "created_cache_tokens": null, "multimodal_tokens": null})),
        ],
    )
        .prop_map(|(p, t, c, d)| json!({"prompt_tokens": p, "total_tokens": t, "completion_tokens": c, "prompt_tokens_details": d}))
}

fn check<T: Serialize + DeserializeOwned + PartialEq + std::fmt::Debug>(
    value: &Value,
) -> Result<bool, TestCaseError> {
    let text = value.to_string();
    let Ok(typed) = serde_json::from_str::<T>(&text) else {
        return Ok(false);
    };
    let rendered = serde_json::to_string(&typed).expect("serializes");
    let back: Value = serde_json::from_str(&rendered).expect("valid JSON");
    prop_assert_eq!(&back, value, "lost or changed data");
    prop_assert_eq!(
        serde_json::from_str::<T>(&rendered).expect("reparses"),
        typed
    );
    Ok(true)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(2_000))]

    #[test]
    fn chat_requests_round_trip_whenever_accepted(value in chat_request()) {
        check::<ChatCompletionRequest>(&value)?;
    }

    #[test]
    fn completion_requests_round_trip_whenever_accepted(value in completion_request()) {
        check::<CompletionRequest>(&value)?;
    }

    #[test]
    fn usage_round_trips_whenever_accepted(value in usage()) {
        prop_assert!(check::<Usage>(&value)?);
    }

    #[test]
    fn arbitrary_json_never_breaks_any_wire_type(value in any_json()) {
        check::<ChatCompletionRequest>(&value)?;
        check::<CompletionRequest>(&value)?;
        check::<ChatCompletionResponse>(&value)?;
        check::<ChatCompletionChunk>(&value)?;
        check::<ModelList>(&value)?;
        check::<ErrorResponse>(&value)?;
        check::<Usage>(&value)?;
    }
}

#[test]
fn generator_produces_mostly_accepted_requests() {
    use proptest::strategy::ValueTree;
    use proptest::test_runner::TestRunner;
    let mut runner = TestRunner::deterministic();
    let strategy = chat_request();
    let accepted = (0..500)
        .filter(|_| {
            let value = strategy.new_tree(&mut runner).expect("tree").current();
            serde_json::from_value::<ChatCompletionRequest>(value).is_ok()
        })
        .count();
    assert!(
        accepted > 100,
        "only {accepted} of 500 generated requests parsed"
    );
}
