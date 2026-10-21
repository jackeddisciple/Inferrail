//! End to end behaviour over HTTP: caching, queueing, cancellation and the
//! accounting that the router will later depend on.

mod common;

use std::time::{Duration, Instant};

use common::{TestSim, chat, config, long_text, spawn};
use futures::StreamExt;
use inferrail_sim::engine::EngineSnapshot;
use inferrail_sim::latency::LatencyModel;
use serde_json::{Value, json};

fn prefill_heavy() -> LatencyModel {
    LatencyModel {
        prefill_token_secs: 150e-6,
        prefill_pair_secs: 0.0,
        prefill_fixed_secs: 2e-3,
        decode_token_secs: 5e-3,
    }
}

fn shared_prompt_request(system: &str, question: &str) -> Value {
    json!({
        "model": "sim/llama-8b",
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": question}
        ],
        "max_tokens": 4,
        "stream": true,
        "stream_options": {"include_usage": true}
    })
}

fn usage_of(events: &[inferrail_core::openai::sse::SseEvent]) -> Value {
    let usage = &events[events.len() - 2];
    serde_json::from_str::<Value>(&usage.data).expect("usage chunk")["usage"].clone()
}

async fn wait_for(
    sim: &TestSim,
    limit: Duration,
    done: impl Fn(&EngineSnapshot) -> bool,
) -> Duration {
    let started = Instant::now();
    loop {
        let snap = sim.sim.engine().snapshot();
        if done(&snap) {
            return started.elapsed();
        }
        assert!(started.elapsed() < limit, "condition not reached: {snap:?}");
        tokio::time::sleep(Duration::from_millis(2)).await;
    }
}

#[tokio::test]
async fn shared_system_prompt_hits_cache_and_cuts_ttft() {
    let sim = spawn(config(prefill_heavy())).await;
    let system = long_text(1_500, "policy");
    let (cold, cold_ttft, _) = sim
        .stream(
            "/v1/chat/completions",
            &shared_prompt_request(&system, "first question"),
        )
        .await;
    let (warm, warm_ttft, _) = sim
        .stream(
            "/v1/chat/completions",
            &shared_prompt_request(&system, "a different one"),
        )
        .await;

    let cold_usage = usage_of(&cold);
    let warm_usage = usage_of(&warm);
    let prompt = warm_usage["prompt_tokens"].as_u64().unwrap();
    let cached = warm_usage["prompt_tokens_details"]["cached_tokens"]
        .as_u64()
        .unwrap();
    assert_eq!(cold_usage["prompt_tokens_details"]["cached_tokens"], 0);
    assert!(cached >= prompt - 32, "cached {cached} of {prompt}");
    assert_eq!(cached % 16, 0);
    assert!(
        warm_ttft * 4 < cold_ttft,
        "warm {warm_ttft:?} should be far below cold {cold_ttft:?}"
    );
}

#[tokio::test]
async fn later_turns_reuse_earlier_turns() {
    let sim = spawn(config(LatencyModel::ZERO)).await;
    let mut messages = vec![json!({"role": "user", "content": long_text(300, "turn")})];
    let mut previous_prompt = 0u64;
    for turn in 0..4 {
        let (_, reply) = sim
            .post_json(
                "/v1/chat/completions",
                &json!({"messages": messages, "max_tokens": 20}),
            )
            .await;
        let prompt = reply["usage"]["prompt_tokens"].as_u64().unwrap();
        let cached = reply["usage"]["prompt_tokens_details"]["cached_tokens"]
            .as_u64()
            .unwrap();
        if turn > 0 {
            assert!(
                cached >= previous_prompt / 16 * 16,
                "turn {turn}: {cached} < {previous_prompt}"
            );
        }
        previous_prompt = prompt;
        messages.push(
            json!({"role": "assistant", "content": reply["choices"][0]["message"]["content"]}),
        );
        messages.push(json!({"role": "user", "content": format!("follow up {turn}")}));
    }
}

#[tokio::test]
async fn reset_prefix_cache_makes_the_next_request_cold() {
    let sim = spawn(config(LatencyModel::ZERO)).await;
    let body = chat(&long_text(200, "x"), 1);
    sim.post_json("/v1/chat/completions", &body).await;
    let (_, warm) = sim.post_json("/v1/chat/completions", &body).await;
    assert!(
        warm["usage"]["prompt_tokens_details"]["cached_tokens"]
            .as_u64()
            .unwrap()
            > 0
    );
    let reset = sim
        .client
        .post(sim.url("/reset_prefix_cache"))
        .send()
        .await
        .unwrap();
    assert_eq!(reset.status(), 200);
    let (_, cold) = sim.post_json("/v1/chat/completions", &body).await;
    assert_eq!(cold["usage"]["prompt_tokens_details"]["cached_tokens"], 0);
}

#[tokio::test]
async fn concurrent_prefills_queue_behind_each_other() {
    let sim = spawn(config(prefill_heavy())).await;
    let mut tasks = Vec::new();
    for i in 0..5 {
        let client = sim.client.clone();
        let url = sim.url("/v1/chat/completions");
        let body = chat(&long_text(400, &format!("q{i}x")), 1);
        tasks.push(tokio::spawn(async move {
            let started = Instant::now();
            client
                .post(url)
                .json(&body)
                .send()
                .await
                .unwrap()
                .bytes()
                .await
                .unwrap();
            started.elapsed()
        }));
    }
    let mut times = Vec::new();
    for task in tasks {
        times.push(task.await.unwrap());
    }
    times.sort();
    let one = times[0];
    assert!(
        times[4] > one * 3,
        "fifth request should wait for the others: {times:?}"
    );
}

#[tokio::test]
async fn stream_disconnect_frees_the_slot_within_100ms() {
    let latency = LatencyModel {
        decode_token_secs: 20e-3,
        ..LatencyModel::ZERO
    };
    let sim = spawn(config(latency)).await;
    let mut body = chat("long answer please", 2_000);
    body["stream"] = json!(true);
    let response = sim.post("/v1/chat/completions", &body).await;
    let mut stream = response.bytes_stream();
    stream.next().await.expect("first chunk").expect("ok");
    assert_eq!(sim.sim.engine().snapshot().running, 1);
    assert!(sim.sim.engine().snapshot().kv_blocks_used > 0);

    drop(stream);
    let freed = wait_for(&sim, Duration::from_secs(2), |s| s.running == 0).await;
    assert!(freed < Duration::from_millis(100), "took {freed:?}");
    assert_eq!(sim.sim.engine().snapshot().kv_blocks_used, 0);
    assert_eq!(sim.sim.metrics().finished("abort"), 1);
}

#[tokio::test]
async fn disconnect_while_queued_withdraws_the_request() {
    let latency = LatencyModel {
        prefill_token_secs: 1e-3,
        ..LatencyModel::ZERO
    };
    let sim = spawn(config(latency)).await;
    let blocker = {
        let client = sim.client.clone();
        let url = sim.url("/v1/chat/completions");
        let body = chat(&long_text(800, "b"), 1);
        tokio::spawn(async move { client.post(url).json(&body).send().await })
    };
    wait_for(&sim, Duration::from_secs(2), |s| s.running == 1).await;

    let mut queued = chat("short", 1);
    queued["stream"] = json!(true);
    let response = sim.post("/v1/chat/completions", &queued).await;
    wait_for(&sim, Duration::from_secs(1), |s| s.waiting == 1).await;
    drop(response);
    let withdrawn = wait_for(&sim, Duration::from_secs(1), |s| s.waiting == 0).await;
    assert!(withdrawn < Duration::from_millis(100), "took {withdrawn:?}");
    blocker.abort();
}

#[tokio::test]
async fn non_streaming_disconnect_frees_the_slot() {
    let latency = LatencyModel {
        decode_token_secs: 20e-3,
        ..LatencyModel::ZERO
    };
    let sim = spawn(config(latency)).await;
    let client = reqwest::Client::builder()
        .timeout(Duration::from_millis(150))
        .build()
        .unwrap();
    let result = client
        .post(sim.url("/v1/chat/completions"))
        .json(&chat("x", 1_000))
        .send()
        .await;
    assert!(result.is_err(), "request should time out");
    let freed = wait_for(&sim, Duration::from_secs(2), |s| s.running == 0).await;
    assert!(freed < Duration::from_millis(100), "took {freed:?}");
}

#[tokio::test]
async fn sequence_cap_is_never_exceeded_under_load() {
    let mut cfg = config(LatencyModel {
        decode_token_secs: 2e-3,
        ..LatencyModel::ZERO
    });
    cfg.engine.max_num_seqs = 4;
    let sim = spawn(cfg).await;
    let mut tasks = Vec::new();
    for i in 0..24 {
        let client = sim.client.clone();
        let url = sim.url("/v1/chat/completions");
        let body = chat(&format!("req {i}"), 10);
        tasks.push(tokio::spawn(async move {
            client.post(url).json(&body).send().await.unwrap().status()
        }));
    }
    let mut peak = 0;
    let mut saw_waiting = false;
    while tasks.iter().any(|t| !t.is_finished()) {
        let snap = sim.sim.engine().snapshot();
        peak = peak.max(snap.running);
        saw_waiting |= snap.waiting > 0;
        tokio::time::sleep(Duration::from_millis(1)).await;
    }
    for task in tasks {
        assert_eq!(task.await.unwrap(), 200);
    }
    assert!(peak <= 4, "peak running {peak}");
    assert!(saw_waiting, "excess requests should wait");
}

#[tokio::test]
async fn metrics_agree_with_responses() {
    let sim = spawn(config(LatencyModel::ZERO)).await;
    let (mut prompt, mut generated, mut cached) = (0, 0, 0);
    for i in 0..20 {
        let body = chat(&long_text(50 + i * 3, "m"), 5 + i as u64);
        let (_, reply) = sim.post_json("/v1/chat/completions", &body).await;
        prompt += reply["usage"]["prompt_tokens"].as_u64().unwrap();
        generated += reply["usage"]["completion_tokens"].as_u64().unwrap();
        cached += reply["usage"]["prompt_tokens_details"]["cached_tokens"]
            .as_u64()
            .unwrap();
    }
    let text = sim
        .client
        .get(sim.url("/metrics"))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    let value = |series: &str| -> f64 {
        text.lines()
            .find(|l| l.starts_with(series))
            .and_then(|l| l.rsplit_once(' '))
            .map(|(_, v)| v.parse().unwrap())
            .unwrap_or_else(|| panic!("missing {series}"))
    };
    assert_eq!(value("vllm:prompt_tokens_total{") as u64, prompt);
    assert_eq!(value("vllm:prefix_cache_queries_total{") as u64, prompt);
    assert_eq!(value("vllm:prefix_cache_hits_total{") as u64, cached);
    assert_eq!(value("vllm:generation_tokens_total{") as u64, generated);
    assert_eq!(
        value("vllm:request_success_total{engine=\"0\",finished_reason=\"length\"") as u64,
        20
    );
    assert_eq!(value("vllm:time_to_first_token_seconds_count{") as u64, 20);
    assert_eq!(value("vllm:num_requests_running{"), 0.0);
    assert_eq!(value("vllm:kv_cache_usage_perc{"), 0.0);
}

#[tokio::test]
async fn hundreds_of_concurrent_streams_complete_cleanly() {
    let sim = spawn(config(LatencyModel {
        prefill_token_secs: 5e-6,
        decode_token_secs: 1e-3,
        ..LatencyModel::ZERO
    }))
    .await;
    let mut tasks = Vec::new();
    for i in 0..300 {
        let client = sim.client.clone();
        let url = sim.url("/v1/chat/completions");
        let mut body = chat(&format!("stream {i}"), 16);
        body["stream"] = json!(true);
        tasks.push(tokio::spawn(async move {
            let raw = client
                .post(url)
                .json(&body)
                .send()
                .await
                .unwrap()
                .bytes()
                .await
                .unwrap();
            let text = String::from_utf8(raw.to_vec()).unwrap();
            (
                text.matches("data: ").count(),
                text.ends_with("data: [DONE]\n\n"),
            )
        }));
    }
    for task in tasks {
        let (frames, done) = task.await.unwrap();
        assert_eq!(frames, 18);
        assert!(done);
    }
    let snap = sim.sim.engine().snapshot();
    assert_eq!((snap.running, snap.waiting, snap.kv_blocks_used), (0, 0, 0));
}

#[tokio::test]
async fn a_request_finishes_at_its_last_token_not_at_done() {
    use axum::body::Body;
    use axum::http::Request;
    use tower::ServiceExt;

    let sim = inferrail_sim::server::Sim::new(config(LatencyModel {
        decode_token_secs: 2e-3,
        ..LatencyModel::ZERO
    }))
    .unwrap();
    let mut body = chat("finish me", 6);
    body["stream"] = json!(true);
    body["stream_options"] = json!({"include_usage": true});
    let request = Request::post("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let response = sim.router().oneshot(request).await.unwrap();
    let mut frames = response.into_body().into_data_stream();
    loop {
        let frame = frames.next().await.expect("more frames").expect("ok");
        if String::from_utf8_lossy(&frame).contains("\"finish_reason\":\"length\"") {
            break;
        }
    }
    let snap = sim.engine().snapshot();
    assert_eq!(
        (snap.running, snap.kv_blocks_used),
        (0, 0),
        "slot held after last token"
    );
    assert_eq!(sim.metrics().finished("length"), 1);
    drop(frames);
    assert_eq!(sim.metrics().finished("abort"), 0);
}
