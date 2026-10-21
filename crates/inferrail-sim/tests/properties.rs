//! Randomized end to end checks: every request the simulator accepts must get
//! a response that obeys the vLLM contract, and the books must balance.

mod common;

use std::sync::OnceLock;
use std::time::Duration;

use common::{TestSim, config, spawn};
use futures::StreamExt;
use inferrail_core::openai::sse::SseDecoder;
use inferrail_core::openai::{ChatCompletionChunk, ChatCompletionResponse};
use inferrail_sim::latency::LatencyModel;
use proptest::prelude::*;
use serde_json::{Value, json};
use tokio::runtime::Runtime;

fn runtime() -> &'static Runtime {
    static RT: OnceLock<Runtime> = OnceLock::new();
    RT.get_or_init(|| Runtime::new().expect("runtime"))
}

fn shared_sim() -> &'static TestSim {
    static SIM: OnceLock<TestSim> = OnceLock::new();
    SIM.get_or_init(|| {
        runtime().block_on(async {
            let mut cfg = config(LatencyModel::ZERO);
            cfg.default_max_tokens = 7;
            spawn(cfg).await
        })
    })
}

#[derive(Debug, Clone)]
struct Case {
    messages: Vec<(&'static str, String)>,
    model: Option<&'static str>,
    max_tokens: Option<u64>,
    stream: bool,
    include_usage: Option<bool>,
    seed: Option<i64>,
    tools: bool,
}

impl Case {
    fn body(&self) -> Value {
        let messages: Vec<Value> = self
            .messages
            .iter()
            .map(|(role, content)| json!({"role": role, "content": content}))
            .collect();
        let mut body = json!({"messages": messages, "stream": self.stream});
        if let Some(model) = self.model {
            body["model"] = json!(model);
        }
        if let Some(n) = self.max_tokens {
            body["max_tokens"] = json!(n);
        }
        if let Some(include) = self.include_usage {
            body["stream_options"] = json!({"include_usage": include});
        }
        if let Some(seed) = self.seed {
            body["seed"] = json!(seed);
        }
        if self.tools {
            body["tools"] = json!([{"type": "function", "function": {"name": "lookup"}}]);
        }
        body
    }
}

fn case() -> impl Strategy<Value = Case> {
    let message = (
        prop::sample::select(vec!["system", "user", "assistant", "tool", "developer"]),
        "\\PC{0,120}",
    );
    (
        proptest::collection::vec(message, 0..5),
        prop::sample::select(vec![
            Some("sim/llama-8b"),
            Some("llama"),
            None,
            Some("not-served"),
        ]),
        proptest::option::of(1u64..48),
        any::<bool>(),
        proptest::option::of(any::<bool>()),
        proptest::option::of(any::<i64>()),
        any::<bool>(),
    )
        .prop_map(
            |(messages, model, max_tokens, stream, include_usage, seed, tools)| Case {
                messages,
                model,
                max_tokens,
                stream,
                include_usage,
                seed,
                tools,
            },
        )
}

async fn check_case(sim: &TestSim, case: &Case) -> Result<(), String> {
    let response = sim
        .client
        .post(sim.url("/v1/chat/completions"))
        .json(&case.body())
        .send()
        .await
        .map_err(|e| e.to_string())?;
    let status = response.status().as_u16();
    if case.include_usage.is_some() && !case.stream {
        return if status == 400 {
            Ok(())
        } else {
            Err(format!("expected 400, got {status}"))
        };
    }
    if case.model == Some("not-served") {
        return if status == 404 {
            Ok(())
        } else {
            Err(format!("expected 404, got {status}"))
        };
    }
    if status != 200 {
        return Err(format!(
            "unexpected status {status}: {}",
            response.text().await.unwrap_or_default()
        ));
    }
    let expected_tokens = case.max_tokens.unwrap_or(7);
    let expected_finish = if case.max_tokens.is_some() {
        "length"
    } else {
        "stop"
    };
    let expected_model = case.model.unwrap_or("sim/llama-8b");

    if !case.stream {
        let body: ChatCompletionResponse = response.json().await.map_err(|e| e.to_string())?;
        let usage = body.usage.value().ok_or("missing usage")?;
        let choice = &body.choices[0];
        if usage.completion_tokens() != expected_tokens
            || usage.total_tokens != usage.prompt_tokens + expected_tokens
            || choice.finish_reason.value().map(|f| f.as_str()) != Some(expected_finish)
            || body.model != expected_model
            || choice.message.text().is_none_or(|t| t.is_empty())
        {
            return Err(format!("inconsistent response: {body:?}"));
        }
        return Ok(());
    }

    if response.headers()["content-type"] != "text/event-stream; charset=utf-8" {
        return Err("wrong content type".into());
    }
    let mut decoder = SseDecoder::default();
    let mut events = Vec::new();
    let mut bytes = response.bytes_stream();
    while let Some(chunk) = bytes.next().await {
        decoder
            .push(&chunk.map_err(|e| e.to_string())?, &mut events)
            .map_err(|e| e.to_string())?;
    }
    if !decoder.is_idle() || !events.last().is_some_and(|e| e.is_done()) {
        return Err("stream did not end with [DONE]".into());
    }
    let chunks: Vec<ChatCompletionChunk> = events[..events.len() - 1]
        .iter()
        .map(|e| serde_json::from_str(&e.data))
        .collect::<Result<_, _>>()
        .map_err(|e| e.to_string())?;
    let with_usage = case.include_usage == Some(true);
    let token_chunks = chunks.len() - 1 - usize::from(with_usage);
    if token_chunks as u64 != expected_tokens {
        return Err(format!(
            "{token_chunks} token chunks, wanted {expected_tokens}"
        ));
    }
    if chunks[0].choices[0].delta.role.value().map(|r| r.as_str()) != Some("assistant") {
        return Err("first chunk must carry the role".into());
    }
    if chunks
        .iter()
        .any(|c| c.id != chunks[0].id || c.model != expected_model)
    {
        return Err("chunks disagree on id or model".into());
    }
    let finishes: Vec<_> = chunks
        .iter()
        .flat_map(|c| &c.choices)
        .filter_map(|c| c.finish_reason.value().map(|f| f.as_str().to_owned()))
        .collect();
    if finishes != [expected_finish] {
        return Err(format!("finish reasons {finishes:?}"));
    }
    if with_usage {
        let usage = chunks
            .last()
            .and_then(|c| c.usage.value())
            .ok_or("missing usage chunk")?;
        if usage.completion_tokens() != expected_tokens {
            return Err("usage disagrees with streamed tokens".into());
        }
    } else if chunks.iter().any(|c| !c.usage.is_absent()) {
        return Err("usage sent without include_usage".into());
    }
    Ok(())
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(400))]

    #[test]
    fn every_accepted_chat_request_obeys_the_vllm_contract(case in case()) {
        let sim = shared_sim();
        let result = runtime().block_on(check_case(sim, &case));
        prop_assert!(result.is_ok(), "{:?}: {}", case.body(), result.unwrap_err());
    }
}

#[test]
fn streamed_and_full_responses_carry_the_same_text() {
    let sim = shared_sim();
    runtime().block_on(async {
        for seed in [0i64, 1, -5, i64::MAX] {
            for (path, mut body, text_of) in [
                (
                    "/v1/chat/completions",
                    json!({"messages": [{"role": "user", "content": "same"}], "max_tokens": 40}),
                    "/choices/0/message/content",
                ),
                (
                    "/v1/completions",
                    json!({"prompt": "same", "max_tokens": 40}),
                    "/choices/0/text",
                ),
            ] {
                body["seed"] = json!(seed);
                let (_, full) = sim.post_json(path, &body).await;
                body["stream"] = json!(true);
                body["stream_options"] = json!({"include_usage": true});
                let (events, _, _) = sim.stream(path, &body).await;
                let streamed: String = events[..events.len() - 1]
                    .iter()
                    .filter_map(|e| serde_json::from_str::<Value>(&e.data).ok())
                    .filter_map(|v| {
                        v.pointer("/choices/0/delta/content")
                            .or_else(|| v.pointer("/choices/0/text"))
                            .and_then(Value::as_str)
                            .map(str::to_owned)
                    })
                    .collect();
                assert_eq!(
                    full.pointer(text_of).and_then(Value::as_str),
                    Some(streamed.as_str()),
                    "{path} seed {seed}"
                );
                let usage: Value = serde_json::from_str(&events[events.len() - 2].data).unwrap();
                assert_eq!(
                    usage["usage"]["completion_tokens"],
                    full["usage"]["completion_tokens"]
                );
                assert_eq!(
                    usage["usage"]["prompt_tokens"],
                    full["usage"]["prompt_tokens"]
                );
            }
        }
    });
}

#[test]
fn histograms_and_counters_stay_consistent_after_random_traffic() {
    runtime().block_on(async {
        let sim = spawn(config(LatencyModel::ZERO)).await;
        let mut tasks = Vec::new();
        for i in 0..200u64 {
            let client = sim.client.clone();
            let url = sim.url("/v1/chat/completions");
            let body = json!({
                "messages": [{"role": "user", "content": format!("{} {}", "shared prefix ".repeat(20), i % 7)}],
                "max_tokens": 1 + i % 13,
                "stream": i % 2 == 0,
            });
            tasks.push(tokio::spawn(async move {
                let response = client.post(url).json(&body).send().await.unwrap();
                if i % 11 == 0 {
                    drop(response);
                } else {
                    response.bytes().await.unwrap();
                }
            }));
        }
        for task in tasks {
            task.await.unwrap();
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
        let text = sim.client.get(sim.url("/metrics")).send().await.unwrap().text().await.unwrap();
        let value = |line: &str| -> f64 { line.rsplit_once(' ').unwrap().1.parse().unwrap() };
        let mut previous: Option<(String, f64)> = None;
        let mut families = 0;
        for line in text.lines().filter(|l| !l.starts_with('#')) {
            let name = line.split('{').next().unwrap();
            if let Some(family) = name.strip_suffix("_bucket") {
                if let Some((prev_family, prev)) = &previous
                    && prev_family == family
                {
                    assert!(value(line) >= *prev, "{family} buckets decrease");
                }
                previous = Some((family.to_owned(), value(line)));
            } else if let Some(family) = name.strip_suffix("_count") {
                let (bucket_family, inf) = previous.take().expect("count follows buckets");
                assert_eq!(bucket_family, family);
                assert_eq!(value(line), inf, "{family} count differs from +Inf bucket");
                families += 1;
            }
        }
        assert_eq!(families, 15);
        let metrics = sim.sim.metrics();
        assert_eq!(metrics.finished("length") + metrics.finished("abort"), 200);
        let snap = sim.sim.engine().snapshot();
        assert_eq!((snap.running, snap.waiting, snap.kv_blocks_used), (0, 0, 0));
    });
}
