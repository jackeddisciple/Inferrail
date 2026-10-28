//! Shadow mode: candidate policies choose for every routed request without
//! routing any, their picks are measured against the policy that routed,
//! and the log of what they saw replays offline to the same picks.

mod common;

use std::process::Command;
use std::time::Duration;

use common::{FAST_HEALTH, MODEL, TestGateway, Upstream, fleet, serve_admin, urls, wait_until};
use inferrail_core::config::{Config, PolicyKind};
use serde_json::{Value, json};

const BIN: &str = env!("CARGO_BIN_EXE_inferrail-gateway");

fn temp_log(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("inferrail-shadow-{name}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("temp dir");
    let path = dir.join("shadow.jsonl");
    let _ = std::fs::remove_file(&path);
    path
}

fn shadowed(sims: &[&Upstream], shadows: &[PolicyKind], log: Option<&std::path::Path>) -> Config {
    let mut config = fleet(
        &urls(sims),
        &format!("{FAST_HEALTH}routing:\n  seed: 11\n  session_affinity: true\n"),
    );
    config.models[0].tokenizer = Some("builtin:sim".to_owned());
    config.routing.shadow = shadows.to_vec();
    config.observability.shadow_log = log.map(ToOwned::to_owned);
    config
}

fn body(i: u64) -> Value {
    let system: String = (0..300)
        .map(|w| format!("rule{} ", (i % 4) * 1000 + w))
        .collect();
    json!({
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": format!("question {i}")},
        ],
        "max_tokens": 2,
    })
}

async fn traffic(gw: &TestGateway, range: std::ops::Range<u64>) {
    let mut tasks = Vec::new();
    for i in range {
        let (client, url) = (gw.client.clone(), gw.url("/v1/chat/completions"));
        tasks.push(tokio::spawn(async move {
            let mut request = client.post(url).json(&body(i));
            if i % 3 == 0 {
                request = request.header("x-session-id", format!("chat-{}", i % 7));
            }
            let response = request.send().await.expect("sent");
            assert_eq!(response.status(), 200);
            response.bytes().await.expect("body");
        }));
    }
    for task in tasks {
        task.await.expect("request");
    }
}

/// Decisions written so far, counting only whole lines.
fn decision_lines(log: &std::path::Path) -> usize {
    let text = std::fs::read_to_string(log).unwrap_or_default();
    let whole = text.rfind('\n').map_or("", |end| &text[..end]);
    whole
        .lines()
        .filter(|l| l.contains("\"request_id\""))
        .count()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn shadow_logs_match_an_offline_replay() {
    let sims = [
        Upstream::zero(&[MODEL]).await,
        Upstream::zero(&[MODEL]).await,
        Upstream::zero(&[MODEL]).await,
    ];
    let log = temp_log("replay");
    let shadows = [
        PolicyKind::CostModel,
        PolicyKind::KvAware,
        PolicyKind::PrefixHash,
        PolicyKind::RoundRobin,
        PolicyKind::Random,
        PolicyKind::LeastRequest,
        PolicyKind::P2c,
    ];
    let config = shadowed(&[&sims[0], &sims[1], &sims[2]], &shadows[..5], Some(&log));
    let gw = TestGateway::start(config.clone()).await;
    traffic(&gw, 0..150).await;
    let mut retuned = config.clone();
    retuned.routing.epsilon = 0.5;
    retuned.routing.shadow = shadows.to_vec();
    gw.gateway.reload(retuned).expect("reload");
    traffic(&gw, 150..300).await;
    wait_until(Duration::from_secs(10), || {
        decision_lines(&log) == 300
            && std::fs::read_to_string(&log).is_ok_and(|t| t.ends_with('\n'))
    })
    .await;

    let file = std::fs::File::open(&log).unwrap();
    let report = inferrail_gateway::replay_shadow(std::io::BufReader::new(file)).unwrap();
    assert!(report.matched(), "{:#?}", report.mismatches);
    assert_eq!((report.decisions, report.generations), (300, 2));
    for policy in &shadows[..5] {
        assert_eq!(report.policies[policy.as_str()], (300, 300), "{policy}");
    }
    for policy in &shadows[5..] {
        assert_eq!(report.policies[policy.as_str()], (150, 150), "{policy}");
    }

    let output = Command::new(BIN)
        .arg("--replay-shadow")
        .arg(&log)
        .output()
        .expect("run the binary");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let printed: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(printed["decisions"], 300);

    let text = std::fs::read_to_string(&log).unwrap();
    let mut tampered: Vec<String> = text.lines().map(str::to_owned).collect();
    let line = tampered
        .iter()
        .position(|l| l.contains("\"request_id\""))
        .unwrap();
    let mut first: Value = serde_json::from_str(&tampered[line]).unwrap();
    let candidates = first["candidates"].as_array().unwrap().len();
    let cost = first["shadow"]
        .as_array_mut()
        .unwrap()
        .iter_mut()
        .find(|p| p["policy"] == "round_robin")
        .unwrap();
    let other = (0..candidates)
        .map(|i| format!("r{i}"))
        .find(|r| cost["replica"] != json!(r))
        .unwrap();
    cost["replica"] = json!(other);
    tampered[line] = first.to_string();
    let report =
        inferrail_gateway::replay_shadow(std::io::Cursor::new(tampered.join("\n"))).unwrap();
    assert!(!report.matched());
    std::fs::write(&log, tampered.join("\n")).unwrap();
    let output = Command::new(BIN)
        .arg("--replay-shadow")
        .arg(&log)
        .output()
        .unwrap();
    assert!(!output.status.success());
    std::fs::remove_dir_all(log.parent().unwrap()).unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn shadows_are_measured_against_the_routing_policy() {
    let sims = [
        Upstream::zero(&[MODEL]).await,
        Upstream::zero(&[MODEL]).await,
    ];
    let mut config = shadowed(
        &[&sims[0], &sims[1]],
        &[PolicyKind::CostModel, PolicyKind::RoundRobin],
        None,
    );
    config.routing.default_policy = PolicyKind::RoundRobin;
    let gw = TestGateway::start(config).await;
    let admin = serve_admin(&gw.gateway).await;
    traffic(&gw, 0..120).await;
    let tallied = |gw: &TestGateway| {
        gw.gateway
            .render_metrics()
            .lines()
            .filter(|l| l.starts_with("inferrail_shadow_decisions_total"))
            .map(|l| l.rsplit_once(' ').unwrap().1.parse::<f64>().unwrap())
            .sum::<f64>()
    };
    wait_until(Duration::from_secs(5), || tallied(&gw) == 240.0).await;
    let shown: Value = reqwest::get(format!("{admin}/admin/shadow"))
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let rows = shown["policies"].as_array().unwrap();
    assert_eq!(rows.len(), 2);
    let row = |name: &str| rows.iter().find(|r| r["policy"] == name).unwrap();
    let cost = row("cost_model");
    assert_eq!(cost["decisions"], 120);
    assert!(cost["mean_gain_ms"].as_f64().unwrap() > 0.0, "{cost}");
    let (faster, slower) = (
        cost["faster_share"].as_f64().unwrap(),
        cost["slower_share"].as_f64().unwrap(),
    );
    assert!(faster > 0.0 && faster + slower <= 1.0, "{cost}");
    assert!(cost["agreement"].as_f64().unwrap() < 1.0);
    let rotation = row("round_robin");
    assert!((0.0..=1.0).contains(&rotation["agreement"].as_f64().unwrap()));
    let page = gw.gateway.render_metrics();
    for line in [
        "inferrail_shadow_estimated_ttft_seconds_total{model=\"model-a\",policy=\"cost_model\",pick=\"active\"}",
        "inferrail_shadow_estimated_ttft_seconds_total{model=\"model-a\",policy=\"cost_model\",pick=\"shadow\"}",
        "inferrail_shadow_dropped_total 0",
    ] {
        assert!(page.contains(line), "{line} missing");
    }
    let routed: f64 = page
        .lines()
        .filter(|l| l.starts_with("inferrail_routing_decisions_total") && l.contains("round_robin"))
        .map(|l| l.rsplit_once(' ').unwrap().1.parse::<f64>().unwrap())
        .sum();
    assert_eq!(routed, 120.0, "shadows must not route");
}

#[tokio::test]
async fn without_shadows_nothing_is_captured() {
    let upstream = Upstream::zero(&[MODEL]).await;
    let log = temp_log("none");
    let gw = TestGateway::start(shadowed(&[&upstream], &[], Some(&log))).await;
    traffic(&gw, 0..10).await;
    tokio::time::sleep(Duration::from_millis(300)).await;
    assert_eq!(decision_lines(&log), 0);
    let page = gw.gateway.render_metrics();
    assert!(!page.contains("inferrail_shadow_decisions_total{"));
    std::fs::remove_dir_all(log.parent().unwrap()).unwrap();
}
