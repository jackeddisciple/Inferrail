//! The cache aware policies and their guards under load. Ignored by default;
//! run with `make stress` (release build, output shown).

mod common;

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use common::*;
use inferrail_core::config::{Config, PolicyKind};
use inferrail_load::parse::samples;
use inferrail_sim::latency::LatencyModel;
use inferrail_sim::server::SimConfig;
use serde_json::{Value, json};

const WORDS: [&str; 24] = [
    "cache", "router", "prefix", "token", "replica", "latency", "block", "engine", "queue",
    "decode", "prefill", "memory", "warm", "request", "stream", "model", "batch", "first", "load",
    "hit", "miss", "fleet", "signal", "budget",
];

fn text(seed: u64, words: usize) -> String {
    (0..words as u64)
        .map(|i| WORDS[((seed.wrapping_mul(2_654_435_761) ^ i.wrapping_mul(40_503)) % 24) as usize])
        .collect::<Vec<_>>()
        .join(" ")
}

fn sim(latency: LatencyModel) -> SimConfig {
    let mut config = sim_config(&[MODEL], latency);
    config.enable_prompt_tokens_details = true;
    config.engine.max_num_seqs = 4_096;
    config.engine.max_num_batched_tokens = 8_192;
    config
}

fn routed(sims: &[Upstream], extra: &str) -> Config {
    let refs: Vec<&Upstream> = sims.iter().collect();
    let mut config = fleet(&urls(&refs), &format!("{FAST_HEALTH}{extra}"));
    config.models[0].tokenizer = Some("builtin:sim".to_owned());
    config
}

fn shared_prompt(system: u64, question: u64, words: usize) -> Value {
    json!({
        "model": MODEL,
        "messages": [
            {"role": "system", "content": format!("System {system}: {}", text(system, words))},
            {"role": "user", "content": text(1_000_000 + question, 16)},
        ],
        "max_tokens": 1,
    })
}

/// A quantile of a histogram on the metrics page, from its buckets.
fn quantile(page: &str, name: &str, q: f64) -> f64 {
    let mut buckets: Vec<(f64, f64)> = samples(page)
        .filter(|s| s.name == format!("{name}_bucket"))
        .filter_map(|s| {
            let le = s.label("le")?;
            let bound = if le == "+Inf" {
                f64::INFINITY
            } else {
                le.parse().ok()?
            };
            Some((bound, s.value))
        })
        .collect();
    buckets.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut merged: Vec<(f64, f64)> = Vec::new();
    for (bound, count) in buckets {
        match merged.last_mut() {
            Some(last) if last.0 == bound => last.1 += count,
            _ => merged.push((bound, count)),
        }
    }
    let total = merged.last().map_or(0.0, |b| b.1);
    merged
        .iter()
        .find(|b| b.1 >= q * total)
        .map_or(f64::NAN, |b| b.0)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
#[ignore = "stress"]
async fn every_policy_adds_little_over_eight_replicas() {
    let mut sims = Vec::new();
    for _ in 0..8 {
        sims.push(Upstream::spawn(sim(LatencyModel::ZERO)).await);
    }
    let client = reqwest::Client::new();
    for rate in [2_000u64, 5_000] {
        let total = 10_000;
        let body = |i: u64| shared_prompt(i % 16, i, 200);
        let direct = open_loop(
            &client,
            format!("{}/v1/chat/completions", sims[0].url()),
            rate,
            total,
            body,
        )
        .await;
        assert_eq!(direct.errors, 0);
        let [d50, d99, _] = summary(direct.service);
        for policy in [
            PolicyKind::RoundRobin,
            PolicyKind::PrefixHash,
            PolicyKind::KvAware,
            PolicyKind::CostModel,
        ] {
            let gw = TestGateway::start(routed(
                &sims,
                &format!("routing:\n  default_policy: {policy}\n  session_affinity: true\n"),
            ))
            .await;
            let run = open_loop(&client, gw.url("/v1/chat/completions"), rate, total, body).await;
            assert_eq!(run.errors, 0, "{policy}");
            let [p50, p99, _] = summary(run.service);
            let page = gw.gateway.render_metrics();
            let decision = quantile(&page, "inferrail_routing_decision_seconds", 0.99);
            println!(
                "{policy} over 8 replicas at {rate}/s x {total}: added p50 {:.3} ms p99 {:.3} ms, decision p99 under {:.0} us",
                ms(p50.saturating_sub(d50)),
                ms(p99.saturating_sub(d99)),
                decision * 1e6,
            );
            assert!(
                p99.saturating_sub(d99) < Duration::from_millis(20),
                "{policy}"
            );
            assert!(decision <= 2.5e-4, "{policy}: {decision}");
        }
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
#[ignore = "stress"]
async fn hot_prompts_and_sessions_under_load_spread_stick_and_explain() {
    let latency = LatencyModel {
        prefill_token_secs: 20e-6,
        prefill_pair_secs: 0.0,
        prefill_fixed_secs: 1e-3,
        decode_token_secs: 2e-3,
        decode_seq_secs: 0.0,
    };
    let mut sims = Vec::new();
    for _ in 0..4 {
        sims.push(Upstream::spawn(sim(latency)).await);
    }
    let gw = Arc::new(
        TestGateway::start(routed(
            &sims,
            "routing:\n  default_policy: cost_model\n  session_affinity: true\n  seed: 2\n  cost_model:\n    prefill_token_secs: 0.00002\n    prefill_pair_secs: 0\n    prefill_fixed_secs: 0.001\n",
        ))
        .await,
    );
    let admin = serve_admin(&gw.gateway).await;
    let weights: Vec<f64> = (1..=16).map(|k| 1.0 / f64::from(k).powf(1.1)).collect();
    let total_weight: f64 = weights.iter().sum();
    let draws = inferrail_policy::SplitMix::new(7);
    let total = 4_000u64;
    let interval = Duration::from_micros(1_000_000 / 400);
    let started = tokio::time::Instant::now();
    let mut tasks = Vec::new();
    for i in 0..total {
        tokio::time::sleep_until(started + interval * i as u32).await;
        let mut pick = (draws.next_u64() >> 11) as f64 / (1u64 << 53) as f64 * total_weight;
        let system = weights
            .iter()
            .position(|w| {
                pick -= w;
                pick < 0.0
            })
            .unwrap_or(0) as u64;
        let session = format!("user-{}", draws.below(300));
        let shared = Arc::clone(&gw);
        tasks.push(tokio::spawn(async move {
            let gw = shared;
            let mut body = shared_prompt(system, i, 300);
            body["max_tokens"] = json!(20);
            let response = gw
                .client
                .post(gw.url("/v1/chat/completions"))
                .header("x-session-id", &session)
                .json(&body)
                .send()
                .await
                .expect("sent");
            let status = response.status().as_u16();
            let replica = response
                .headers()
                .get("x-inferrail-replica")
                .and_then(|v| v.to_str().ok())
                .unwrap_or_default()
                .to_owned();
            let _ = response.bytes().await;
            (status, session, replica)
        }));
        if i % 200 == 0 {
            let (client, url) = (
                gw.client.clone(),
                format!("{admin}/admin/explain/v1/chat/completions"),
            );
            tasks.push(tokio::spawn(async move {
                let status = client
                    .post(url)
                    .json(&shared_prompt(0, i, 300))
                    .send()
                    .await
                    .expect("sent")
                    .status()
                    .as_u16();
                (status, String::new(), String::new())
            }));
        }
    }
    let mut per_replica: HashMap<String, u64> = HashMap::new();
    let mut per_session: HashMap<String, HashMap<String, u64>> = HashMap::new();
    for task in tasks {
        let (status, session, replica) = task.await.expect("task");
        assert_eq!(status, 200);
        if !session.is_empty() {
            *per_replica.entry(replica.clone()).or_default() += 1;
            *per_session
                .entry(session)
                .or_default()
                .entry(replica)
                .or_default() += 1;
        }
    }
    let busiest = *per_replica.values().max().expect("replicas") as f64 / total as f64;
    let stayed: u64 = per_session
        .values()
        .map(|r| r.values().max().copied().unwrap_or(0))
        .sum();
    let page = gw.gateway.render_metrics();
    let count = |name: &str, label: &str, value: &str| {
        samples(&page)
            .filter(|s| s.name == name && s.label(label).as_deref() == Some(value))
            .map(|s| s.value)
            .sum::<f64>()
    };
    println!(
        "4,000 requests at 400/s, Zipf over 16 system prompts, 300 sessions: busiest replica took {:.1}%, {:.1}% of each session's turns stayed on its main replica, {} sticky decisions, {} bounded load guards",
        busiest * 100.0,
        stayed as f64 / total as f64 * 100.0,
        count("inferrail_routing_steered_total", "reason", "session"),
        count("inferrail_routing_guards_total", "guard", "bounded_load"),
    );
    assert!(busiest < 0.4, "{per_replica:?}");
    assert!(stayed as f64 / total as f64 > 0.8);
    gw.wait_idle().await;
    for sim in &sims {
        assert_eq!(sim.snapshot().running, 0);
    }
}
