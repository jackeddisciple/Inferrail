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
    config.max_model_len = config.max_model_len.min(blocks * 16);
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
        "inferrail_replica_index_blocks",
        "inferrail_index_entries",
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
async fn capacity_is_learned_from_the_engine() {
    let upstream = Upstream::spawn(sim(60)).await;
    let gw = TestGateway::start(tokenized(fleet(
        &urls(&[&upstream]),
        &format!("{FAST_HEALTH}load:\n  scrape_interval: 20ms\n  scrape_timeout: 20ms\n  max_age: 100ms\n"),
    )))
    .await;
    let replica = gw.replica("r0");
    wait_until(Duration::from_secs(3), || {
        replica.engine_load().latest().is_some()
    })
    .await;
    for i in 0..60u64 {
        let seen = send(&gw, &shared_prompt(i % 3, i)).await;
        assert_eq!(seen.cached.0, Some(seen.cached.1), "#{i}: {seen:?}");
    }
    let page = gw.gateway.render_metrics();
    assert!(
        page.contains("inferrail_replica_index_capacity_blocks{replica=\"r0\"} 60"),
        "{page}"
    );
}

#[tokio::test]
async fn an_engine_that_restarts_is_forgotten() {
    let upstream = Upstream::spawn(sim(1_000)).await;
    let addr = upstream.addr;
    let gw = TestGateway::start(tokenized(fleet(
        &urls(&[&upstream]),
        &format!("{FAST_HEALTH}load:\n  scrape_interval: 20ms\n  scrape_timeout: 20ms\n  max_age: 100ms\n"),
    )))
    .await;
    let replica = gw.replica("r0");
    let body = shared_prompt(3, 3);
    send(&gw, &body).await;
    assert!(send(&gw, &body).await.cached.0.unwrap() > 0);
    let before = replica.engine_load().latest().unwrap().at;
    wait_until(Duration::from_secs(3), || {
        replica
            .engine_load()
            .latest()
            .is_some_and(|s| s.at > before && s.load.prefix_hits > 0)
    })
    .await;
    upstream.kill();
    let listener = loop {
        match tokio::net::TcpListener::bind(addr).await {
            Ok(listener) => break listener,
            Err(_) => tokio::time::sleep(Duration::from_millis(10)).await,
        }
    };
    let _fresh = Upstream::spawn_on(listener, sim(1_000));
    let waited = wait_until(Duration::from_secs(5), || {
        gw.gateway
            .render_metrics()
            .contains("inferrail_index_resets_total{replica=\"r0\"} 1")
    })
    .await;
    assert!(
        waited < Duration::from_secs(5),
        "the restart was never noticed"
    );
    wait_until(Duration::from_secs(5), || {
        replica.is_routable(gw.gateway.clock().now())
    })
    .await;
    let after = send(&gw, &body).await;
    assert_eq!(after.cached, (Some(0), 0));
    let again = send(&gw, &body).await;
    assert_eq!(again.cached.0, Some(again.cached.1));
    assert!(again.cached.1 > 0);
}

#[tokio::test]
async fn prompts_the_engine_never_computed_are_not_predicted() {
    let upstream = Upstream::spawn(sim(1_000)).await;
    let gw = TestGateway::start(tokenized(fleet(&urls(&[&upstream]), FAST_HEALTH))).await;
    let mut refused = shared_prompt(4, 4);
    refused["max_tokens"] = json!(1 << 30);
    let response = gw.post("/v1/chat/completions", &refused).await;
    assert_eq!(response.status(), 400);
    let after = send(&gw, &shared_prompt(4, 4)).await;
    assert_eq!(
        after.cached,
        (Some(0), 0),
        "a refused prompt left blocks behind"
    );
    let again = send(&gw, &shared_prompt(4, 5)).await;
    assert_eq!(again.cached.0, Some(again.cached.1));
    assert!(again.cached.1 > 0);

    let sims = [
        Upstream::spawn(sim(1_000)).await,
        Upstream::spawn(sim(1_000)).await,
    ];
    let gw = TestGateway::start(tokenized(fleet(
        &urls(&[&sims[0], &sims[1]]),
        &format!(
            "{FAST_HEALTH}routing:\n  default_policy: round_robin\nretry:\n  max_attempts: 2\n"
        ),
    )))
    .await;
    sims[0].set_request_fault(Fault::Status(503));
    let body = shared_prompt(6, 6);
    let response = gw.post("/v1/chat/completions", &body).await;
    assert_eq!(response.status(), 200);
    assert_eq!(response.headers()["x-inferrail-replica"], "r1");
    response.bytes().await.unwrap();
    sims[0].set_request_fault(Fault::None);
    let mut landed = std::collections::HashSet::new();
    for _ in 0..4 {
        let response = gw.post("/v1/chat/completions", &body).await;
        landed.insert(
            response.headers()["x-inferrail-replica"]
                .to_str()
                .unwrap()
                .to_owned(),
        );
        let predicted = header(&response, "x-inferrail-cached-tokens");
        let answer: Value = response.json().await.unwrap();
        let cached = answer["usage"]["prompt_tokens_details"]["cached_tokens"]
            .as_u64()
            .unwrap_or(0);
        assert_eq!(predicted, Some(cached));
    }
    assert_eq!(landed.len(), 2, "{landed:?}");
}

#[tokio::test]
async fn without_the_index_only_the_prompt_is_reported() {
    let upstream = Upstream::spawn(sim(1_000)).await;
    let path = std::env::temp_dir().join(format!(
        "inferrail-unpredicted-{}.jsonl",
        std::process::id()
    ));
    let _ = std::fs::remove_file(&path);
    let gw = TestGateway::start(tokenized(fleet(
        &urls(&[&upstream]),
        &format!(
            "{FAST_HEALTH}index:\n  enabled: false\nobservability:\n  decision_log:\n    path: {}\n    sample_rate: 1.0\n",
            path.display()
        ),
    )))
    .await;
    for _ in 0..2 {
        let response = gw.post("/v1/chat/completions", &shared_prompt(2, 2)).await;
        assert!(header(&response, "x-inferrail-prompt-tokens").unwrap() > 400);
        assert!(
            response
                .headers()
                .get("x-inferrail-cached-tokens")
                .is_none()
        );
        response.bytes().await.unwrap();
    }
    let page = gw.gateway.render_metrics();
    assert!(page.contains("inferrail_prompt_tokens_total{"), "{page}");
    assert!(
        !page.contains("inferrail_predicted_cached_tokens_total{"),
        "{page}"
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
    for line in &lines {
        assert!(line["prompt_tokens"].as_u64().unwrap() > 400);
        let candidates = line["candidates"].as_array().unwrap();
        assert!(
            candidates.iter().all(|c| c["cached_tokens"].is_null()),
            "{line}"
        );
    }
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
