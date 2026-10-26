mod common;

use std::time::Duration;

use common::*;
use futures::StreamExt;
use inferrail_core::config::{Config, PolicyKind};
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

/// The share of prompt tokens served from cache over 300 requests for 36
/// system prompts drawn at random, after 100 that warm the caches. Each
/// engine holds about fourteen of them, so only routing on the prefix keeps
/// them all warm.
async fn hit_rate(policy: PolicyKind) -> f64 {
    let sims = fleet_of(3, 560, LatencyModel::ZERO).await;
    let mut config = routed(
        &sims,
        &format!("routing:\n  default_policy: {policy}\n  seed: 3\n"),
    );
    for replica in &mut config.replicas {
        replica.kv_capacity_blocks = Some(560);
    }
    let gw = TestGateway::start(config).await;
    let (mut prompt, mut cached) = (0, 0);
    let draws = inferrail_policy::SplitMix::new(17);
    for i in 0..400u64 {
        let system = draws.below(36) as u64;
        let (p, c, _) = send(&gw, &shared_prompt(system, i)).await;
        if i >= 100 {
            prompt += p;
            cached += c;
        }
    }
    cached as f64 / prompt as f64
}

#[tokio::test]
async fn cache_aware_policies_keep_shared_prompts_where_they_are_cached() {
    let blind = hit_rate(PolicyKind::RoundRobin).await;
    assert!(blind < 0.45, "round robin {blind:.2}");
    for policy in [
        PolicyKind::PrefixHash,
        PolicyKind::KvAware,
        PolicyKind::CostModel,
    ] {
        let aware = hit_rate(policy).await;
        assert!(aware >= 0.85, "{policy} {aware:.2} against {blind:.2}");
    }
}

#[tokio::test]
async fn sessions_stick_to_their_replica_until_it_is_overloaded() {
    let sims = fleet_of(3, 4_096, decode(25.0)).await;
    let gw = TestGateway::start(routed(
        &sims,
        "routing:\n  default_policy: cost_model\n  session_affinity: true\n  seed: 5\n",
    ))
    .await;
    let session = |seed: u64| {
        let gw = &gw;
        async move {
            let response = gw
                .client
                .post(gw.url("/v1/chat/completions"))
                .header("x-session-id", "conversation-1")
                .json(&chat(&text(seed, 20), 1))
                .send()
                .await
                .unwrap();
            assert_eq!(response.status(), 200);
            response.headers()["x-inferrail-replica"]
                .to_str()
                .unwrap()
                .to_owned()
        }
    };
    let home = session(1).await;
    for seed in 2..12 {
        assert_eq!(session(seed).await, home, "#{seed}");
    }
    let mut held = Vec::new();
    for seed in 20..22 {
        let response = gw
            .client
            .post(gw.url("/v1/chat/completions"))
            .header("x-session-id", "conversation-1")
            .json(&streaming(chat(&text(seed, 20), 80)))
            .send()
            .await
            .unwrap();
        assert_eq!(response.headers()["x-inferrail-replica"], home.as_str());
        let mut body = response.bytes_stream();
        body.next().await.unwrap().unwrap();
        held.push(body);
    }
    let moved = session(30).await;
    assert_ne!(
        moved, home,
        "an overloaded replica must not keep its sessions"
    );
    drop(held);
    gw.wait_idle().await;
    assert_eq!(
        session(31).await,
        moved,
        "the session follows its new replica"
    );

    let user = |seed: u64| {
        let gw = &gw;
        async move {
            let mut body = chat(&text(seed, 20), 1);
            body["user"] = json!("alice");
            send(gw, &body).await.2
        }
    };
    let alice = user(40).await;
    for seed in 41..46 {
        assert_eq!(user(seed).await, alice);
    }
    let page = gw.gateway.render_metrics();
    let steered = inferrail_load::parse::samples(&page)
        .find(|s| {
            s.name == "inferrail_routing_steered_total"
                && s.label("reason").as_deref() == Some("session")
        })
        .map(|s| s.value)
        .unwrap();
    assert!(steered >= 15.0, "{steered}");
    let guarded = inferrail_load::parse::samples(&page)
        .find(|s| {
            s.name == "inferrail_routing_guards_total"
                && s.label("guard").as_deref() == Some("bounded_load")
        })
        .map(|s| s.value)
        .unwrap();
    assert_eq!(guarded, 1.0);
}

#[tokio::test]
async fn without_affinity_sessions_are_routed_on_cost_alone() {
    let sims = fleet_of(3, 4_096, LatencyModel::ZERO).await;
    let gw = TestGateway::start(routed(
        &sims,
        "routing:\n  default_policy: cost_model\n  seed: 5\n",
    ))
    .await;
    let mut seen = std::collections::BTreeSet::new();
    for seed in 0..30 {
        let response = gw
            .client
            .post(gw.url("/v1/chat/completions"))
            .header("x-session-id", "conversation-1")
            .json(&chat(&text(seed, 20), 1))
            .send()
            .await
            .unwrap();
        seen.insert(response.headers()["x-inferrail-replica"].clone());
    }
    assert_eq!(seen.len(), 3);
}

fn gauge(page: &str, name: &str, replica: &str) -> f64 {
    inferrail_load::parse::samples(page)
        .find(|s| s.name == name && s.label("replica").as_deref() == Some(replica))
        .map(|s| s.value)
        .unwrap_or_else(|| panic!("no {name} for {replica}"))
}

#[tokio::test]
async fn queued_prefill_is_estimated_while_it_waits_and_drains_after() {
    let latency = LatencyModel {
        prefill_token_secs: 200e-6,
        prefill_pair_secs: 0.0,
        prefill_fixed_secs: 1e-3,
        decode_token_secs: 1e-3,
    };
    let sims = fleet_of(1, 4_096, latency).await;
    let gw = TestGateway::start(routed(
        &sims,
        "routing:\n  default_policy: cost_model\n  cost_model:\n    prefill_token_secs: 0.0002\n    prefill_pair_secs: 0\n    prefill_fixed_secs: 0.001\n",
    ))
    .await;
    let bodies: Vec<Value> = (0..3).map(|i| chat(&text(100 + i, 1_000), 1)).collect();
    let waiting: Vec<_> = bodies
        .iter()
        .map(|body| {
            let (client, url, body) = (
                gw.client.clone(),
                gw.url("/v1/chat/completions"),
                body.clone(),
            );
            tokio::spawn(async move { client.post(url).json(&body).send().await.unwrap().status() })
        })
        .collect();
    wait_until(Duration::from_secs(2), || {
        gw.replica("r0").prefilling() == 3
    })
    .await;
    let queued = gauge(
        &gw.gateway.render_metrics(),
        "inferrail_replica_queued_seconds",
        "r0",
    );
    assert!((0.3..2.0).contains(&queued), "{queued}");
    let admin = serve_admin(&gw.gateway).await;
    let explained: Value = gw
        .client
        .post(format!("{admin}/admin/explain/v1/chat/completions"))
        .json(&chat(&text(7, 1_000), 1))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let estimate = &explained["replicas"][0]["estimate"];
    assert!(
        estimate["queue_secs"].as_f64().unwrap() > 0.2,
        "{explained}"
    );
    let prefill = estimate["prefill_secs"].as_f64().unwrap();
    let tokens = explained["prompt_tokens"].as_f64().unwrap();
    assert!(
        (prefill - (tokens * 200e-6 + 1e-3)).abs() < 1e-9,
        "{explained}"
    );
    for task in waiting {
        assert_eq!(task.await.unwrap(), 200);
    }
    gw.wait_idle().await;
    let page = gw.gateway.render_metrics();
    assert_eq!(gauge(&page, "inferrail_replica_queued_seconds", "r0"), 0.0);
    assert!(
        page.contains(
            "inferrail_estimated_ttft_seconds_count{model=\"model-a\",policy=\"cost_model\"} 3"
        ),
        "{page}"
    );
}
