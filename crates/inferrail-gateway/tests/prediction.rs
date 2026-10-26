mod common;

use std::time::Duration;

use common::*;
use inferrail_core::config::Config;
use inferrail_sim::latency::LatencyModel;
use inferrail_sim::server::SimConfig;
use serde_json::{Value, json};

const WORDS: [&str; 24] = [
    "cache", "router", "prefix", "token", "replica", "latency", "block", "engine", "queue",
    "decode", "prefill", "memory", "warm", "request", "stream", "model", "batch", "first", "load",
    "hit", "miss", "fleet", "signal", "budget",
];

/// Deterministic text of `words` words that depends only on `seed`.
fn text(seed: u64, words: usize) -> String {
    (0..words as u64)
        .map(|i| WORDS[((seed.wrapping_mul(2_654_435_761) ^ i.wrapping_mul(40_503)) % 24) as usize])
        .collect::<Vec<_>>()
        .join(" ")
}

fn sim(blocks: usize) -> SimConfig {
    let mut config = sim_config(&[MODEL], LatencyModel::ZERO);
    config.enable_prompt_tokens_details = true;
    config.engine.num_gpu_blocks = blocks;
    config
}

fn tokenized(mut config: Config) -> Config {
    config.models[0].tokenizer = Some("builtin:sim".to_owned());
    config
}

fn header(response: &reqwest::Response, name: &str) -> Option<u64> {
    response.headers().get(name)?.to_str().ok()?.parse().ok()
}

/// One request's prediction next to what the engine reported.
#[derive(Debug, PartialEq)]
struct Seen {
    prompt: (Option<u64>, u64),
    cached: (Option<u64>, u64),
}

async fn send(gw: &TestGateway, body: &Value) -> Seen {
    let response = gw.post("/v1/chat/completions", body).await;
    assert_eq!(response.status(), 200);
    let (prompt, cached) = (
        header(&response, "x-inferrail-prompt-tokens"),
        header(&response, "x-inferrail-cached-tokens"),
    );
    let answer: Value = response.json().await.expect("a JSON answer");
    let usage = &answer["usage"];
    Seen {
        prompt: (prompt, usage["prompt_tokens"].as_u64().expect("usage")),
        cached: (
            cached,
            usage["prompt_tokens_details"]["cached_tokens"]
                .as_u64()
                .unwrap_or(0),
        ),
    }
}

fn shared_prompt(system: u64, question: u64) -> Value {
    json!({
        "model": MODEL,
        "messages": [
            {"role": "system", "content": text(system, 400)},
            {"role": "user", "content": text(1_000 + question, 12)},
        ],
        "max_tokens": 1,
    })
}

#[tokio::test]
async fn predictions_equal_what_the_engines_cached() {
    let sims = [
        Upstream::spawn(sim(300)).await,
        Upstream::spawn(sim(300)).await,
        Upstream::spawn(sim(300)).await,
    ];
    let mut config = tokenized(fleet(
        &urls(&[&sims[0], &sims[1], &sims[2]]),
        &format!("{FAST_HEALTH}routing:\n  default_policy: random\n  seed: 7\n"),
    ));
    for replica in &mut config.replicas {
        replica.kv_capacity_blocks = Some(300);
    }
    let gw = TestGateway::start(config).await;
    let (mut hits, mut exact) = (0, 0);
    for i in 0..300u64 {
        let seen = send(&gw, &shared_prompt(i * 7 % 13, i)).await;
        assert_eq!(seen.prompt.0, Some(seen.prompt.1), "#{i}");
        assert_eq!(seen.cached.0, Some(seen.cached.1), "#{i}: {seen:?}");
        exact += 1;
        hits += u64::from(seen.cached.1 > 0);
    }
    assert_eq!(exact, 300);
    assert!(hits > 100, "only {hits} cache hits");
    for sim in &sims {
        assert!(sim.snapshot().cached_blocks <= 300);
    }
    let page = gw.gateway.render_metrics();
    assert!(
        page.contains("inferrail_tokenized_requests_total{model=\"model-a\",result=\"ok\"} 300"),
        "{page}"
    );
    for name in [
        "inferrail_prompt_tokens_total",
        "inferrail_predicted_cached_tokens_total",
        "inferrail_tokenize_seconds",
    ] {
        assert!(page.contains(name), "{name} missing");
    }
}

#[tokio::test]
async fn streams_and_completions_carry_predictions() {
    let upstream = Upstream::spawn(sim(1_000)).await;
    let gw = TestGateway::start(tokenized(fleet(&urls(&[&upstream]), FAST_HEALTH))).await;
    let body = streaming(shared_prompt(1, 1));
    for round in 0..2 {
        let response = gw.post("/v1/chat/completions", &body).await;
        assert_eq!(response.status(), 200);
        let cached = header(&response, "x-inferrail-cached-tokens").unwrap();
        let prompt = header(&response, "x-inferrail-prompt-tokens").unwrap();
        assert!(prompt > 400);
        assert_eq!(cached > 0, round == 1, "round {round}: {cached}");
        response.bytes().await.unwrap();
    }
    let completion = json!({"model": MODEL, "prompt": text(5, 300), "max_tokens": 1});
    for round in 0..2 {
        let response = gw.post("/v1/completions", &completion).await;
        let cached = header(&response, "x-inferrail-cached-tokens").unwrap();
        let answer: Value = response.json().await.unwrap();
        let usage = &answer["usage"];
        assert_eq!(
            cached,
            usage["prompt_tokens_details"]["cached_tokens"]
                .as_u64()
                .unwrap_or(0)
        );
        assert_eq!(cached > 0, round == 1);
    }
    let ids = json!({"model": MODEL, "prompt": (0..100).collect::<Vec<u32>>(), "max_tokens": 1});
    let response = gw.post("/v1/completions", &ids).await;
    assert_eq!(header(&response, "x-inferrail-prompt-tokens"), Some(100));
}

#[tokio::test]
async fn a_tokenizer_that_cannot_load_degrades_to_plain_routing() {
    let upstream = Upstream::spawn(sim(1_000)).await;
    let mut config = fleet(&urls(&[&upstream]), FAST_HEALTH);
    config.models[0].tokenizer = Some("/definitely/not/a/model".to_owned());
    let warnings = inferrail_gateway::Gateway::check(&config).unwrap();
    assert_eq!(warnings.len(), 1);
    assert!(
        warnings[0].contains("routes without cache predictions"),
        "{warnings:?}"
    );
    let gw = TestGateway::start(config).await;
    let response = gw.post("/v1/chat/completions", &shared_prompt(1, 1)).await;
    assert_eq!(response.status(), 200);
    assert!(
        response
            .headers()
            .get("x-inferrail-prompt-tokens")
            .is_none()
    );
    assert!(
        response
            .headers()
            .get("x-inferrail-cached-tokens")
            .is_none()
    );
    let page = gw.gateway.render_metrics();
    assert!(page.contains("result=\"unavailable\"} 1"), "{page}");
    let plain = Upstream::spawn(sim(1_000)).await;
    let gw = TestGateway::start(fleet(&urls(&[&plain]), FAST_HEALTH)).await;
    let response = gw.post("/v1/chat/completions", &shared_prompt(1, 1)).await;
    assert!(
        response
            .headers()
            .get("x-inferrail-prompt-tokens")
            .is_none()
    );
    assert!(
        !gw.gateway
            .render_metrics()
            .contains("inferrail_tokenized_requests_total")
    );
}

#[tokio::test]
async fn requests_the_tokenizer_cannot_follow_still_route() {
    let upstream = Upstream::spawn(sim(1_000)).await;
    let gw = TestGateway::start(tokenized(fleet(&urls(&[&upstream]), FAST_HEALTH))).await;
    let batch = json!({"model": MODEL, "prompt": ["a", "b"], "max_tokens": 1});
    let response = gw.post("/v1/completions", &batch).await;
    assert!(
        response
            .headers()
            .get("x-inferrail-prompt-tokens")
            .is_none()
    );
    let (status, _) = gw
        .post_json(
            "/v1/chat/completions",
            &json!({"model": MODEL, "messages": "nope"}),
        )
        .await;
    assert_eq!(status, 400);
    let page = gw.gateway.render_metrics();
    assert!(page.contains("result=\"unsupported\"} 1"), "{page}");
    assert!(page.contains("result=\"invalid\"} 1"), "{page}");
}

#[tokio::test]
async fn the_decision_log_shows_each_candidates_prediction() {
    let sims = [
        Upstream::spawn(sim(1_000)).await,
        Upstream::spawn(sim(1_000)).await,
    ];
    let path = std::env::temp_dir().join(format!(
        "inferrail-predictions-{}.jsonl",
        std::process::id()
    ));
    let _ = std::fs::remove_file(&path);
    let gw = TestGateway::start(tokenized(fleet(
        &urls(&[&sims[0], &sims[1]]),
        &format!(
            "{FAST_HEALTH}routing:\n  default_policy: round_robin\nobservability:\n  decision_log:\n    path: {}\n    sample_rate: 1.0\n",
            path.display()
        ),
    )))
    .await;
    let body = shared_prompt(9, 9);
    let first = send(&gw, &body).await;
    let second = send(&gw, &body).await;
    assert_eq!(
        second.cached,
        (Some(0), 0),
        "round robin moved to the cold replica"
    );
    let mut lines = Vec::new();
    wait_until(Duration::from_secs(3), || {
        lines = std::fs::read_to_string(&path)
            .unwrap_or_default()
            .lines()
            .filter_map(|l| serde_json::from_str::<Value>(l).ok())
            .collect();
        lines.len() >= 2
    })
    .await;
    let _ = std::fs::remove_file(&path);
    let last = &lines[1];
    assert_eq!(last["prompt_tokens"].as_u64(), first.prompt.0);
    let cached: Vec<u64> = last["candidates"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| c["cached_tokens"].as_u64().unwrap())
        .collect();
    assert!(
        cached.iter().any(|&c| c > 0) && cached.contains(&0),
        "{cached:?}"
    );
}
