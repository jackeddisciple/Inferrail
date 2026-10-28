//! The control plane under load: every policy in shadow with its log on at
//! 5,000 req/s, and the config patched over the admin API every 20 ms while
//! the admin views are read. Ignored by default; run with `make stress`.

mod common;

use std::time::{Duration, Instant};

use common::*;
use inferrail_core::config::PolicyKind;
use serde_json::{Value, json};

const SHADOWS: [PolicyKind; 6] = [
    PolicyKind::RoundRobin,
    PolicyKind::Random,
    PolicyKind::LeastRequest,
    PolicyKind::P2c,
    PolicyKind::PrefixHash,
    PolicyKind::KvAware,
];

fn shared(i: u64) -> Value {
    let system: String = (0..200)
        .map(|w| format!("rule{} ", (i % 8) * 1000 + w))
        .collect();
    json!({
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": format!("question {i}")},
        ],
        "max_tokens": 4,
    })
}

fn total(page: &str, family: &str, label: &str) -> f64 {
    page.lines()
        .filter(|l| l.starts_with(family) && l.contains(label))
        .map(|l| {
            let (_, value) = l.rsplit_once(' ').expect("a sample");
            value.parse::<f64>().expect("a number")
        })
        .sum()
}

fn logged(log: &std::path::Path) -> usize {
    let text = std::fs::read_to_string(log).unwrap_or_default();
    if !text.ends_with('\n') {
        return 0;
    }
    text.lines()
        .filter(|l| l.contains("\"request_id\""))
        .count()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
#[ignore = "stress"]
async fn every_policy_in_shadow_stays_off_the_request_path() {
    let sims = [
        Upstream::zero(&[MODEL]).await,
        Upstream::zero(&[MODEL]).await,
        Upstream::zero(&[MODEL]).await,
    ];
    let client = reqwest::Client::new();
    let (rate, requests) = (5_000, 10_000);
    let mut config = fleet(&urls(&[&sims[0], &sims[1], &sims[2]]), "");
    config.models[0].tokenizer = Some("builtin:sim".to_owned());

    let plain = TestGateway::start(config.clone()).await;
    let off = open_loop(
        &client,
        plain.url("/v1/chat/completions"),
        rate,
        requests,
        shared,
    )
    .await;
    drop(plain);

    let dir = std::env::temp_dir().join(format!("inferrail-stress-shadow-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let log = dir.join("shadow.jsonl");
    let _ = std::fs::remove_file(&log);
    config.routing.shadow = SHADOWS.to_vec();
    config.observability.shadow_log = Some(log.clone());
    let shadowed = TestGateway::start(config).await;
    let on = open_loop(
        &client,
        shadowed.url("/v1/chat/completions"),
        rate,
        requests,
        shared,
    )
    .await;
    assert_eq!(off.errors + on.errors, 0);

    let settled = wait_until(Duration::from_secs(30), || {
        let page = shadowed.gateway.render_metrics();
        let dropped = total(&page, "inferrail_shadow_dropped_total", "");
        SHADOWS.iter().all(|p| {
            let label = format!("policy=\"{}\"", p.as_str());
            total(&page, "inferrail_shadow_decisions_total", &label) + dropped == requests as f64
        })
    })
    .await;
    let page = shadowed.gateway.render_metrics();
    let dropped = total(&page, "inferrail_shadow_dropped_total", "") as u64;
    let seen = requests - dropped;
    wait_until(Duration::from_secs(10), || logged(&log) == seen as usize).await;
    let bytes = std::fs::metadata(&log).unwrap().len();
    let replayed = Instant::now();
    let file = std::fs::File::open(&log).unwrap();
    let report = inferrail_gateway::replay_shadow(std::io::BufReader::new(file)).unwrap();
    let replay_secs = replayed.elapsed().as_secs_f64();
    assert!(report.matched(), "{:#?}", report.mismatches);
    assert_eq!(report.decisions, seen);
    for policy in SHADOWS {
        assert_eq!(report.policies[policy.as_str()], (seen, seen), "{policy}");
    }

    let [off50, off99, _] = summary(off.service);
    let [on50, on99, _] = summary(on.service);
    println!(
        "{} policies in shadow at {rate}/s x {requests}: service p50/p99 {:.3}/{:.3} ms without, \
         {:.3}/{:.3} ms with; {seen} shadowed and {dropped} dropped, settled {:.0} ms after the \
         last response; log {:.1} MiB replayed in {replay_secs:.2} s, every pick matched",
        SHADOWS.len(),
        ms(off50),
        ms(off99),
        ms(on50),
        ms(on99),
        ms(settled),
        bytes as f64 / (1 << 20) as f64,
    );
    assert!(
        on50 <= off50 + Duration::from_micros(250),
        "shadows slowed the median request"
    );
    std::fs::remove_dir_all(dir).unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
#[ignore = "stress"]
async fn the_config_patched_every_twenty_milliseconds_under_load() {
    let sims = [
        Upstream::zero(&[MODEL]).await,
        Upstream::zero(&[MODEL]).await,
        Upstream::zero(&[MODEL]).await,
        Upstream::zero(&[MODEL]).await,
    ];
    let all = urls(&[&sims[0], &sims[1], &sims[2], &sims[3]]);
    let mut config = fleet(&all[..3], "");
    config.models[0].tokenizer = Some("builtin:sim".to_owned());
    let gw = TestGateway::start(config).await;
    let admin = serve_admin(&gw.gateway).await;
    let client = reqwest::Client::new();
    let (rate, requests) = (2_000, 10_000);
    let deadline = Instant::now() + Duration::from_millis(requests * 1_000 / rate);

    let patcher = {
        let (client, admin, all) = (client.clone(), admin.clone(), all.clone());
        tokio::spawn(async move {
            let policies = [
                "cost_model",
                "least_request",
                "kv_aware",
                "prefix_hash",
                "p2c",
            ];
            let mut timings = Vec::new();
            let mut step = 0usize;
            while Instant::now() < deadline {
                let replicas: Vec<Value> = all[..3 + step % 2]
                    .iter()
                    .enumerate()
                    .map(|(i, url)| json!({"id": format!("r{i}"), "url": url, "models": [MODEL]}))
                    .collect();
                let split = if step.is_multiple_of(3) {
                    json!({"cost_model": 70, "least_request": 30})
                } else {
                    Value::Null
                };
                let patch = json!({
                    "routing": {
                        "default_policy": policies[step % policies.len()],
                        "epsilon": 0.5 + (step % 10) as f64 / 10.0,
                        "split": split,
                        "shadow": [policies[(step + 1) % policies.len()]],
                    },
                    "replicas": replicas,
                });
                let sent = Instant::now();
                let response = client
                    .patch(format!("{admin}/admin/config"))
                    .json(&patch)
                    .send()
                    .await
                    .expect("patch");
                let status = response.status();
                let text = response.text().await.expect("reply");
                assert_eq!(status, 200, "{text}");
                timings.push(sent.elapsed());
                step += 1;
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
            timings
        })
    };
    let reader = {
        let (client, admin) = (client.clone(), admin.clone());
        tokio::spawn(async move {
            let mut reads = 0u64;
            while Instant::now() < deadline {
                for path in [
                    "/admin/config",
                    "/admin/index",
                    "/admin/replicas",
                    "/admin/shadow",
                ] {
                    let response = client
                        .get(format!("{admin}{path}"))
                        .send()
                        .await
                        .expect("read");
                    assert_eq!(response.status(), 200, "{path}");
                    response.json::<Value>().await.expect("json");
                    reads += 1;
                }
            }
            reads
        })
    };
    let run = open_loop(
        &client,
        gw.url("/v1/chat/completions"),
        rate,
        requests,
        shared,
    )
    .await;
    let timings = patcher.await.expect("patcher");
    let reads = reader.await.expect("reader");
    gw.wait_idle().await;

    assert_eq!(run.errors, 0);
    let page = gw.gateway.render_metrics();
    let patches = timings.len() as f64;
    assert_eq!(
        total(
            &page,
            "inferrail_config_reloads_total",
            "result=\"applied\""
        ),
        patches
    );
    assert_eq!(total(&page, "inferrail_config_reloads_total", ""), patches);
    assert_eq!(
        total(&page, "inferrail_config_generation", ""),
        patches + 1.0
    );
    let failed: f64 = page
        .lines()
        .filter(|l| l.starts_with("inferrail_requests_total") && !l.contains("status=\"2xx\""))
        .map(|l| l.rsplit_once(' ').unwrap().1.parse::<f64>().unwrap())
        .sum();
    assert_eq!(failed, 0.0);
    let [p50, p99, _] = summary(run.latencies);
    let [patch50, patch99, _] = summary(timings);
    println!(
        "{patches} patches every 20 ms and {reads} admin reads under {rate}/s x {requests}: \
         errors 0, request p50/p99 {:.3}/{:.3} ms, patch p50/p99 {:.3}/{:.3} ms",
        ms(p50),
        ms(p99),
        ms(patch50),
        ms(patch99),
    );
}
