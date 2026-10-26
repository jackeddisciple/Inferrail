mod common;

use std::time::Duration;

use common::*;
use inferrail_core::config::Config;
use inferrail_sim::latency::LatencyModel;
use inferrail_sim::server::SimConfig;
use serde_json::{Value, json};

const WORDS: [&str; 16] = [
    "cache", "router", "prefix", "token", "replica", "latency", "block", "engine", "queue",
    "decode", "prefill", "memory", "warm", "stream", "batch", "fleet",
];

fn text(seed: u64, words: usize) -> String {
    (0..words as u64)
        .map(|i| WORDS[((seed.wrapping_mul(2_654_435_761) ^ i.wrapping_mul(40_503)) % 16) as usize])
        .collect::<Vec<_>>()
        .join(" ")
}

fn sim(blocks: usize, latency: LatencyModel) -> SimConfig {
    let mut config = sim_config(&[MODEL], latency);
    config.enable_prompt_tokens_details = true;
    config.engine.num_gpu_blocks = blocks;
    config
}

async fn fleet_of(n: usize, blocks: usize, latency: LatencyModel) -> Vec<Upstream> {
    let mut sims = Vec::new();
    for _ in 0..n {
        sims.push(Upstream::spawn(sim(blocks, latency)).await);
    }
    sims
}

fn routed(sims: &[Upstream], extra: &str) -> Config {
    let refs: Vec<&Upstream> = sims.iter().collect();
    let mut config = fleet(&urls(&refs), &format!("{FAST_HEALTH}{extra}"));
    config.models[0].tokenizer = Some("builtin:sim".to_owned());
    config
}

fn shared_prompt(system: u64, question: u64) -> Value {
    json!({
        "model": MODEL,
        "messages": [
            {"role": "system", "content": format!("System {system}: {}", text(system, 300))},
            {"role": "user", "content": text(10_000 + question, 12)},
        ],
        "max_tokens": 1,
    })
}

/// Prompt and cached tokens of the engine's answer, and who served it.
async fn send(gw: &TestGateway, body: &Value) -> (u64, u64, String) {
    let response = gw.post("/v1/chat/completions", body).await;
    assert_eq!(response.status(), 200);
    let replica = response.headers()["x-inferrail-replica"]
        .to_str()
        .expect("a replica id")
        .to_owned();
    let answer: Value = response.json().await.expect("a JSON answer");
    let usage = &answer["usage"];
    (
        usage["prompt_tokens"].as_u64().expect("prompt tokens"),
        usage["prompt_tokens_details"]["cached_tokens"]
            .as_u64()
            .unwrap_or(0),
        replica,
    )
}

#[tokio::test]
async fn explain_shows_why_a_replica_won() {
    let sims = fleet_of(3, 4_096, LatencyModel::ZERO).await;
    let mut config = routed(
        &sims,
        "routing:\n  default_policy: cost_model\n  seed: 9\n  hot_prefix:\n    enabled: false\nload:\n  enabled: false\n",
    );
    config.models[0].aliases = vec!["alias".to_owned()];
    let gw = TestGateway::start(config).await;
    let admin = serve_admin(&gw.gateway).await;
    let (_, _, warm) = send(&gw, &shared_prompt(1, 1)).await;
    gw.wait_idle().await;
    let mut body = shared_prompt(1, 2);
    body["model"] = json!("alias");
    let response = gw
        .client
        .post(format!("{admin}/admin/explain/v1/chat/completions"))
        .json(&body)
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    assert_eq!(response.headers()["content-type"], "application/json");
    let report = response.text().await.unwrap();
    let explained: Value = serde_json::from_str(&report).unwrap();
    assert_eq!(explained["winner"], warm.as_str());
    assert_eq!(explained["decided_by"], "prefill");
    insta::assert_snapshot!(report.replace(&warm, "WARM"));
    assert_eq!(
        sims.iter().map(Upstream::hits).sum::<u64>(),
        1,
        "explaining sends nothing"
    );

    let explain = |path: &str, body: Value| {
        let (client, url) = (gw.client.clone(), format!("{admin}/admin/explain{path}"));
        async move {
            let response = client.post(url).json(&body).send().await.unwrap();
            (
                response.status().as_u16(),
                response.json::<Value>().await.unwrap_or_default(),
            )
        }
    };
    let completion = json!({"model": MODEL, "prompt": text(3, 200), "max_tokens": 1});
    let (status, report) = explain("/v1/completions", completion).await;
    assert_eq!(status, 200);
    assert_eq!(report["tokenized"], true);
    assert_eq!(
        explain(
            "/v1/chat/completions",
            json!({"model": "ghost", "messages": []})
        )
        .await
        .0,
        404
    );
    assert_eq!(explain("/v1/chat/completions", json!([1])).await.0, 400);
    let broken = json!({"model": MODEL, "messages": "not a list"});
    let (status, report) = explain("/v1/chat/completions", broken).await;
    assert_eq!(status, 200);
    assert_eq!(report["unpredicted"], "invalid");
    assert_eq!(report["tokenized"], false);
    let get = gw
        .client
        .get(format!("{admin}/admin/explain/v1/chat/completions"))
        .send()
        .await
        .unwrap();
    assert_eq!(get.status(), 405);
}

#[tokio::test]
async fn explain_marks_replicas_that_were_not_offered() {
    let sims = fleet_of(2, 4_096, LatencyModel::ZERO).await;
    let mut config = routed(&sims, "routing:\n  default_policy: least_request\n");
    config.models[0].tokenizer = None;
    config.replicas[1].max_concurrency = Some(1);
    let gw = TestGateway::start(config).await;
    let admin = serve_admin(&gw.gateway).await;
    let held = gw.replica("r1").try_acquire(1).unwrap();
    let report: Value = gw
        .client
        .post(format!("{admin}/admin/explain/v1/chat/completions"))
        .json(&chat("hello", 1))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(report["policy"], "least_request");
    assert_eq!(report["unpredicted"], "unconfigured");
    assert_eq!(report["winner"], "r0");
    let rows = report["replicas"].as_array().unwrap();
    assert_eq!(rows[0]["offered"], true);
    assert_eq!(rows[1]["offered"], false);
    assert_eq!(rows[1]["unavailable"], "full");
    assert_eq!(rows[1]["in_flight"], 1);
    sims[0].kill();
    wait_until(Duration::from_secs(3), || {
        !gw.replica("r0").health().probe_up()
    })
    .await;
    let report: Value = gw
        .client
        .post(format!("{admin}/admin/explain/v1/chat/completions"))
        .json(&chat("hello", 1))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(report["replicas"][0]["unavailable"], "unhealthy");
    assert_eq!(report["winner"], Value::Null);
    drop(held);
}

#[tokio::test]
async fn an_oversized_explain_request_gets_a_413_not_a_reset() {
    let sims = fleet_of(1, 4_096, LatencyModel::ZERO).await;
    let mut config = routed(&sims, "");
    config.server.max_body_bytes = 256 * 1024;
    let gw = TestGateway::start(config).await;
    let admin = serve_admin(&gw.gateway).await;
    let url = format!("{admin}/admin/explain/v1/chat/completions");
    for _ in 0..3 {
        let response = gw
            .client
            .post(&url)
            .json(&chat(&"x ".repeat(512 * 1024), 1))
            .send()
            .await
            .expect("an answer rather than a reset");
        assert_eq!(response.status(), 413);
        let body: Value = response.json().await.unwrap();
        assert_eq!(body["error"]["type"], "RequestTooLargeError");
    }
    let response = gw
        .client
        .post(&url)
        .json(&chat(&"x ".repeat(100 * 1024), 1))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
}
