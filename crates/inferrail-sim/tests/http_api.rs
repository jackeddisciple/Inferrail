mod common;

use common::{chat, config, fast_latency, spawn};
use inferrail_core::openai::{
    ChatCompletionChunk, ChatCompletionResponse, CompletionChunk, CompletionResponse,
    ErrorResponse, ModelList,
};
use inferrail_sim::latency::LatencyModel;
use serde_json::{Value, json};

#[tokio::test]
async fn health_version_and_unknown_routes() {
    let sim = spawn(config(LatencyModel::ZERO)).await;
    let health = sim.client.get(sim.url("/health")).send().await.unwrap();
    assert_eq!(health.status(), 200);
    assert!(health.bytes().await.unwrap().is_empty());

    let version: Value = sim
        .client
        .get(sim.url("/version"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert!(
        version["version"]
            .as_str()
            .unwrap()
            .starts_with("0.31.0+inferrail.sim.")
    );

    let missing = sim.client.get(sim.url("/v2/nothing")).send().await.unwrap();
    assert_eq!(missing.status(), 404);
    assert_eq!(
        missing.json::<Value>().await.unwrap(),
        json!({"detail": "Not Found"})
    );
}

#[tokio::test]
async fn models_lists_every_served_name() {
    let sim = spawn(config(LatencyModel::ZERO)).await;
    let list: ModelList = sim
        .client
        .get(sim.url("/v1/models"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let ids: Vec<_> = list.data.iter().map(|m| m.id.as_str()).collect();
    assert_eq!(ids, ["sim/llama-8b", "llama"]);
    assert!(list.data.iter().all(|m| m.owned_by == "vllm"));
    assert_eq!(list.data[0].max_model_len.value(), Some(&32_768));
}

#[tokio::test]
async fn chat_completion_returns_vllm_shaped_body() {
    let sim = spawn(config(fast_latency())).await;
    let response = sim.post("/v1/chat/completions", &chat("hello", 7)).await;
    assert_eq!(response.status(), 200);
    assert_eq!(response.headers()["content-type"], "application/json");
    let raw: Value = response.json().await.unwrap();
    for key in [
        "prompt_logprobs",
        "kv_transfer_params",
        "service_tier",
        "metrics",
    ] {
        assert!(raw[key].is_null(), "{key}");
    }
    let body: ChatCompletionResponse = serde_json::from_value(raw).unwrap();
    assert!(body.id.starts_with("chatcmpl-"));
    assert_eq!(body.object, "chat.completion");
    let choice = &body.choices[0];
    assert_eq!(choice.finish_reason.value().unwrap().as_str(), "length");
    assert!(!choice.message.text().unwrap().is_empty());
    let usage = body.usage.value().unwrap();
    assert_eq!(usage.completion_tokens(), 7);
    assert_eq!(usage.total_tokens, usage.prompt_tokens + 7);
    assert_eq!(usage.cached_tokens(), Some(0));
}

#[tokio::test]
async fn aliases_are_echoed_and_missing_model_uses_the_default() {
    let sim = spawn(config(LatencyModel::ZERO)).await;
    let mut body = chat("x", 1);
    body["model"] = json!("llama");
    let (_, reply) = sim.post_json("/v1/chat/completions", &body).await;
    assert_eq!(reply["model"], "llama");
    body.as_object_mut().unwrap().remove("model");
    let (_, reply) = sim.post_json("/v1/chat/completions", &body).await;
    assert_eq!(reply["model"], "sim/llama-8b");
}

#[tokio::test]
async fn completions_accept_text_and_token_prompts() {
    let sim = spawn(config(LatencyModel::ZERO)).await;
    let (status, raw) = sim
        .post_json(
            "/v1/completions",
            &json!({"prompt": "Once upon", "max_tokens": 3}),
        )
        .await;
    assert_eq!(status, 200);
    let body: CompletionResponse = serde_json::from_value(raw).unwrap();
    assert!(body.id.starts_with("cmpl-"));
    assert_eq!(body.object, "text_completion");
    assert_eq!(body.usage.value().unwrap().prompt_tokens, 3);

    for prompt in [
        json!([128000, 791, 6864]),
        json!([[128000, 791, 6864]]),
        json!(["one"]),
    ] {
        let (status, raw) = sim
            .post_json(
                "/v1/completions",
                &json!({"prompt": prompt, "max_tokens": 2}),
            )
            .await;
        assert_eq!(status, 200, "{raw}");
        assert_eq!(raw["usage"]["completion_tokens"], 2);
    }
}

#[tokio::test]
async fn default_output_length_ends_with_stop() {
    let mut cfg = config(LatencyModel::ZERO);
    cfg.default_max_tokens = 9;
    let sim = spawn(cfg).await;
    let mut body = chat("x", 1);
    body.as_object_mut().unwrap().remove("max_tokens");
    let (_, reply) = sim.post_json("/v1/chat/completions", &body).await;
    assert_eq!(reply["choices"][0]["finish_reason"], "stop");
    assert_eq!(reply["usage"]["completion_tokens"], 9);
}

#[tokio::test]
async fn max_completion_tokens_takes_precedence() {
    let sim = spawn(config(LatencyModel::ZERO)).await;
    let mut body = chat("x", 10);
    body["max_completion_tokens"] = json!(4);
    let (_, reply) = sim.post_json("/v1/chat/completions", &body).await;
    assert_eq!(reply["usage"]["completion_tokens"], 4);
}

#[tokio::test]
async fn seeds_make_output_deterministic_and_request_ids_are_honoured() {
    let sim = spawn(config(LatencyModel::ZERO)).await;
    let mut body = chat("x", 12);
    body["seed"] = json!(42);
    let (_, a) = sim.post_json("/v1/chat/completions", &body).await;
    let (_, b) = sim.post_json("/v1/chat/completions", &body).await;
    assert_eq!(
        a["choices"][0]["message"]["content"],
        b["choices"][0]["message"]["content"]
    );
    assert_ne!(a["id"], b["id"]);

    let reply: Value = sim
        .client
        .post(sim.url("/v1/chat/completions"))
        .header("x-request-id", "trace-123")
        .json(&chat("x", 1))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(reply["id"], "chatcmpl-trace-123");
}

#[tokio::test]
async fn chat_stream_follows_vllm_chunk_order() {
    let sim = spawn(config(fast_latency())).await;
    let mut body = chat("stream me", 4);
    body["stream"] = json!(true);
    body["stream_options"] = json!({"include_usage": true});
    let response = sim.post("/v1/chat/completions", &body).await;
    assert_eq!(
        response.headers()["content-type"],
        "text/event-stream; charset=utf-8"
    );
    drop(response);

    let (events, _, _) = sim.stream("/v1/chat/completions", &body).await;
    assert_eq!(events.len(), 7);
    assert!(events.last().unwrap().is_done());
    let chunks: Vec<ChatCompletionChunk> = events[..6]
        .iter()
        .map(|e| serde_json::from_str(&e.data).unwrap())
        .collect();
    assert!(chunks.iter().all(|c| c.id == chunks[0].id));
    assert_eq!(
        chunks[0].choices[0].delta.role.value().unwrap().as_str(),
        "assistant"
    );
    assert_eq!(chunks[0].choices[0].delta.content.value().unwrap(), "");
    for chunk in &chunks[1..4] {
        assert!(chunk.choices[0].finish_reason.is_null());
    }
    assert_eq!(
        chunks[4].choices[0].finish_reason.value().unwrap().as_str(),
        "length"
    );
    assert!(chunks[5].choices.is_empty());
    let usage = chunks[5].usage.value().unwrap();
    assert_eq!(usage.completion_tokens(), 4);
    assert_eq!(usage.cached_tokens(), Some(0));
}

#[tokio::test]
async fn stream_without_usage_ends_after_the_finish_chunk() {
    let sim = spawn(config(LatencyModel::ZERO)).await;
    let mut body = chat("x", 2);
    body["stream"] = json!(true);
    let (events, _, _) = sim.stream("/v1/chat/completions", &body).await;
    assert_eq!(events.len(), 4);
    assert!(events[3].is_done());
    assert!(!events[2].data.contains("usage"));
}

#[tokio::test]
async fn completion_stream_has_no_role_chunk() {
    let sim = spawn(config(LatencyModel::ZERO)).await;
    let body = json!({
        "prompt": "abc", "max_tokens": 3, "stream": true,
        "stream_options": {"include_usage": true}
    });
    let (events, _, _) = sim.stream("/v1/completions", &body).await;
    assert_eq!(events.len(), 5);
    let chunks: Vec<CompletionChunk> = events[..4]
        .iter()
        .map(|e| serde_json::from_str(&e.data).unwrap())
        .collect();
    assert!(chunks[..3].iter().all(|c| c.choices.len() == 1));
    assert_eq!(
        chunks[2].choices[0].finish_reason.value().unwrap().as_str(),
        "length"
    );
    assert_eq!(chunks[3].usage.value().unwrap().completion_tokens(), 3);
}

async fn expect_error(sim: &common::TestSim, path: &str, body: Value, status: u16, needle: &str) {
    let response = sim
        .client
        .post(sim.url(path))
        .header("content-type", "application/json")
        .body(body.to_string())
        .send()
        .await
        .expect("request sent");
    assert_eq!(response.status().as_u16(), status, "{body}");
    let error: ErrorResponse = response.json().await.expect("error body");
    assert_eq!(error.status(), Some(status));
    assert!(
        error.error.message.contains(needle),
        "{}",
        error.error.message
    );
}

#[tokio::test]
async fn invalid_requests_get_vllm_errors() {
    let sim = spawn(config(LatencyModel::ZERO)).await;
    let chat_path = "/v1/chat/completions";
    expect_error(
        &sim,
        chat_path,
        json!({"model": "gpt-9", "messages": []}),
        404,
        "`gpt-9` does not exist",
    )
    .await;
    expect_error(
        &sim,
        chat_path,
        json!({"messages": "nope"}),
        400,
        "Invalid request",
    )
    .await;
    expect_error(
        &sim,
        chat_path,
        json!({"messages": [], "n": 2}),
        400,
        "n other than 1",
    )
    .await;
    expect_error(
        &sim,
        chat_path,
        json!({"messages": [], "max_tokens": 0}),
        400,
        "at least 1",
    )
    .await;
    expect_error(
        &sim,
        chat_path,
        json!({"messages": [], "max_tokens": 40000}),
        400,
        "too large: 40000",
    )
    .await;
    let huge = common::long_text(40_000, "w");
    expect_error(
        &sim,
        chat_path,
        json!({"messages": [{"role": "user", "content": huge}]}),
        400,
        "maximum context length",
    )
    .await;

    let completion = "/v1/completions";
    expect_error(&sim, completion, json!({"max_tokens": 1}), 400, "prompt").await;
    expect_error(
        &sim,
        completion,
        json!({"prompt": []}),
        400,
        "one prompt per request",
    )
    .await;
    expect_error(
        &sim,
        completion,
        json!({"prompt": ["a", "b"]}),
        400,
        "got 2",
    )
    .await;
    expect_error(
        &sim,
        completion,
        json!({"prompt": [[]]}),
        400,
        "cannot be empty",
    )
    .await;
    expect_error(
        &sim,
        completion,
        json!({"prompt": [128256]}),
        400,
        "out of vocabulary",
    )
    .await;

    let raw = sim
        .client
        .post(sim.url(chat_path))
        .body("{not json")
        .send()
        .await
        .unwrap();
    assert_eq!(raw.status(), 400);
}

#[tokio::test]
async fn prompt_details_are_optional_like_vllm() {
    let mut cfg = config(LatencyModel::ZERO);
    cfg.enable_prompt_tokens_details = false;
    let sim = spawn(cfg).await;
    let (_, reply) = sim.post_json("/v1/chat/completions", &chat("x", 1)).await;
    assert!(reply["usage"]["prompt_tokens_details"].is_null());
}

#[tokio::test]
async fn stream_options_require_streaming_like_vllm() {
    let sim = spawn(config(LatencyModel::ZERO)).await;
    for (path, mut body) in [
        ("/v1/chat/completions", chat("x", 1)),
        ("/v1/completions", json!({"prompt": "x", "max_tokens": 1})),
    ] {
        body["stream_options"] = json!({"include_usage": false});
        let (status, error) = sim.post_json(path, &body).await;
        assert_eq!(status, 400, "{path}");
        assert_eq!(error["error"]["param"], "stream_options");
        assert_eq!(
            error["error"]["message"],
            "Stream options can only be defined when `stream=True`."
        );

        body["stream_options"] = json!({});
        assert_eq!(
            sim.post_json(path, &body).await.0,
            200,
            "empty options are falsy"
        );
        body["stream_options"] = Value::Null;
        assert_eq!(
            sim.post_json(path, &body).await.0,
            200,
            "null options are falsy"
        );
    }
}

#[tokio::test]
async fn features_that_change_the_response_shape_are_refused() {
    let sim = spawn(config(LatencyModel::ZERO)).await;
    let mut chat_logprobs = chat("x", 1);
    chat_logprobs["logprobs"] = json!(true);
    let mut chat_prompt_logprobs = chat("x", 1);
    chat_prompt_logprobs["prompt_logprobs"] = json!(1);
    let cases = [
        ("/v1/chat/completions", chat_logprobs, "logprobs"),
        (
            "/v1/chat/completions",
            chat_prompt_logprobs,
            "prompt_logprobs",
        ),
        (
            "/v1/completions",
            json!({"prompt": "x", "logprobs": 0}),
            "logprobs",
        ),
        (
            "/v1/completions",
            json!({"prompt": "x", "echo": true}),
            "echo",
        ),
        (
            "/v1/completions",
            json!({"prompt": "x", "best_of": 2}),
            "best_of",
        ),
    ];
    for (path, body, param) in cases {
        let (status, error) = sim.post_json(path, &body).await;
        assert_eq!(status, 400, "{body}");
        assert_eq!(error["error"]["param"], param);
        assert!(
            error["error"]["message"]
                .as_str()
                .unwrap()
                .contains("not supported")
        );
    }

    let mut harmless = chat("x", 1);
    harmless["logprobs"] = json!(false);
    harmless["stop"] = json!(["\n"]);
    harmless["prompt_logprobs"] = Value::Null;
    assert_eq!(
        sim.post_json("/v1/chat/completions", &harmless).await.0,
        200
    );
    let ok = json!({"prompt": "x", "max_tokens": 1, "echo": false, "best_of": 1, "logprobs": null});
    assert_eq!(sim.post_json("/v1/completions", &ok).await.0, 200);
}

#[tokio::test]
async fn wrong_methods_get_a_json_405() {
    let sim = spawn(config(LatencyModel::ZERO)).await;
    for (method, path) in [
        (reqwest::Method::GET, "/v1/chat/completions"),
        (reqwest::Method::DELETE, "/v1/models"),
        (reqwest::Method::GET, "/reset_prefix_cache"),
    ] {
        let response = sim
            .client
            .request(method, sim.url(path))
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), 405, "{path}");
        assert_eq!(
            response.json::<Value>().await.unwrap(),
            json!({"detail": "Method Not Allowed"})
        );
    }
}

#[tokio::test]
async fn invalid_configs_are_refused_up_front() {
    use inferrail_sim::server::{Sim, SimConfig};
    let mut bad = SimConfig::default();
    bad.engine.block_size = 0;
    assert!(Sim::new(bad).is_err());
    let names = SimConfig {
        served_model_names: vec![String::new()],
        ..SimConfig::default()
    };
    assert!(Sim::new(names).is_err());
    assert!(Sim::new(SimConfig::default()).is_ok());
}

#[tokio::test]
async fn oversized_or_empty_request_ids_fall_back_to_generated_ones() {
    let sim = spawn(config(LatencyModel::ZERO)).await;
    for header in ["x".repeat(300), String::new()] {
        let reply: Value = sim
            .client
            .post(sim.url("/v1/chat/completions"))
            .header("x-request-id", header.as_str())
            .json(&chat("x", 1))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        let id = reply["id"].as_str().unwrap();
        assert_eq!(id.len(), "chatcmpl-".len() + 32, "{id}");
    }
}

#[tokio::test]
async fn unicode_prompts_stream_valid_text() {
    let sim = spawn(config(LatencyModel::ZERO)).await;
    let mut body = chat("こんにちは 🌍 naïve café Ω≈ç√ \u{0}\u{7f} עברית", 24);
    body["stream"] = json!(true);
    let (events, _, raw) = sim.stream("/v1/chat/completions", &body).await;
    assert!(String::from_utf8(raw).is_ok());
    assert!(events.last().unwrap().is_done());
    let (_, full) = sim
        .post_json("/v1/chat/completions", &chat("🌍🌍🌍", 3))
        .await;
    assert!(full["usage"]["prompt_tokens"].as_u64().unwrap() > 3);
}
