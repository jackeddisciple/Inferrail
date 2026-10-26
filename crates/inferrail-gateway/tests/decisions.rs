mod common;

use std::collections::{BTreeSet, HashMap};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use common::*;
use inferrail_gateway::{BuildError, Gateway};
use serde_json::Value;

fn log_path(name: &str) -> PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    std::env::temp_dir().join(format!(
        "inferrail-decisions-{name}-{}-{nanos}.jsonl",
        std::process::id()
    ))
}

fn decision_config(path: &Path, rate: f64) -> String {
    format!(
        "{FAST_HEALTH}routing:\n  default_policy: least_request\nobservability:\n  decision_log:\n    path: {}\n    sample_rate: {rate}\n",
        path.display()
    )
}

async fn lines(path: &Path, want: usize) -> Vec<Value> {
    let started = Instant::now();
    loop {
        let text = std::fs::read_to_string(path).unwrap_or_default();
        let complete = &text[..text.rfind('\n').map_or(0, |end| end + 1)];
        let lines: Vec<Value> = complete
            .lines()
            .map(|l| serde_json::from_str(l).expect("one JSON object per line"))
            .collect();
        if lines.len() >= want || started.elapsed() > Duration::from_secs(3) {
            return lines;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
}

const SECRET: &str = "the launch code is swordfish";

#[tokio::test]
async fn every_decision_is_logged_with_candidates_and_never_the_prompt() {
    let sims = [
        Upstream::zero(&[MODEL]).await,
        Upstream::zero(&[MODEL]).await,
        Upstream::zero(&[MODEL]).await,
    ];
    let path = log_path("all");
    let gw = TestGateway::start(fleet(
        &urls(&[&sims[0], &sims[1], &sims[2]]),
        &decision_config(&path, 1.0),
    ))
    .await;
    let mut ids = BTreeSet::new();
    for i in 0..20 {
        let response = gw
            .post("/v1/chat/completions", &chat(&format!("{SECRET} {i}"), 2))
            .await;
        assert_eq!(response.status(), 200);
        ids.insert(
            response.headers()["x-request-id"]
                .to_str()
                .unwrap()
                .to_owned(),
        );
        response.bytes().await.unwrap();
    }
    let logged = lines(&path, 20).await;
    let raw = std::fs::read_to_string(&path).unwrap();
    let _ = std::fs::remove_file(&path);
    assert_eq!(logged.len(), 20);
    assert!(!raw.contains("swordfish"));
    let keys: BTreeSet<&str> = logged[0]
        .as_object()
        .unwrap()
        .keys()
        .map(String::as_str)
        .collect();
    assert_eq!(
        keys,
        BTreeSet::from([
            "attempt",
            "candidates",
            "chosen",
            "decision_ns",
            "model",
            "policy",
            "prompt_tokens",
            "request_id",
            "ts_ms"
        ])
    );
    assert!(logged.iter().all(|l| l["prompt_tokens"].is_null()));
    let mut seen = BTreeSet::new();
    for line in &logged {
        assert_eq!(line["model"], MODEL);
        assert_eq!(line["policy"], "least_request");
        assert_eq!(line["attempt"], 0);
        let candidates = line["candidates"].as_array().unwrap();
        assert_eq!(candidates.len(), 3);
        let chosen = line["chosen"].as_str().unwrap();
        let winner = candidates.iter().find(|c| c["replica"] == chosen).unwrap();
        assert_eq!(winner["eligible"], true);
        assert_eq!(winner["healthy"], true);
        let least = candidates
            .iter()
            .map(|c| c["in_flight"].as_u64().unwrap())
            .min()
            .unwrap();
        assert_eq!(winner["in_flight"].as_u64().unwrap(), least);
        seen.insert(line["request_id"].as_str().unwrap().to_owned());
    }
    assert_eq!(seen, ids);
    let page = gw.gateway.render_metrics();
    assert!(
        page.contains("inferrail_decision_log_records_total 20"),
        "{page}"
    );
    assert!(page.contains("inferrail_decision_log_dropped_total 0"));
}

#[tokio::test]
async fn retries_are_logged_as_numbered_attempts_of_one_request() {
    let failing = Upstream::zero(&[MODEL]).await;
    failing.set_request_fault(Fault::Status(503));
    let healthy = Upstream::zero(&[MODEL]).await;
    let path = log_path("retries");
    let gw = TestGateway::start(fleet(
        &urls(&[&failing, &healthy]),
        &decision_config(&path, 1.0).replace("least_request", "round_robin"),
    ))
    .await;
    for _ in 0..2 {
        let (status, _) = gw.post_json("/v1/chat/completions", &chat("x", 1)).await;
        assert_eq!(status, 200);
    }
    let logged = lines(&path, 3).await;
    let _ = std::fs::remove_file(&path);
    let mut by_request: HashMap<String, Vec<(u64, String, bool)>> = HashMap::new();
    for line in &logged {
        let tried = line["candidates"]
            .as_array()
            .unwrap()
            .iter()
            .any(|c| c["tried"] == true);
        by_request
            .entry(line["request_id"].as_str().unwrap().to_owned())
            .or_default()
            .push((
                line["attempt"].as_u64().unwrap(),
                line["chosen"].as_str().unwrap().to_owned(),
                tried,
            ));
    }
    let retried: Vec<_> = by_request.values().filter(|v| v.len() == 2).collect();
    assert!(!retried.is_empty(), "{by_request:?}");
    for attempts in retried {
        assert_eq!(
            attempts,
            &vec![(0, "r0".to_owned(), false), (1, "r1".to_owned(), true)]
        );
    }
}

#[tokio::test]
async fn sampling_keeps_or_drops_whole_requests() {
    let sim = Upstream::zero(&[MODEL]).await;
    let path = log_path("sampled");
    let gw = TestGateway::start(fleet(&urls(&[&sim]), &decision_config(&path, 0.3))).await;
    for i in 0..300 {
        let (status, _) = gw
            .post_json(
                "/v1/completions",
                &serde_json::json!({"model": MODEL, "prompt": format!("{i}"), "max_tokens": 1}),
            )
            .await;
        assert_eq!(status, 200);
    }
    tokio::time::sleep(Duration::from_millis(400)).await;
    let logged = lines(&path, 1).await;
    let _ = std::fs::remove_file(&path);
    assert!((50..140).contains(&logged.len()), "{}", logged.len());
}

#[tokio::test]
async fn an_unwritable_log_fails_the_build() {
    let sim = Upstream::zero(&[MODEL]).await;
    let config = fleet(
        &urls(&[&sim]),
        &decision_config(&PathBuf::from("/no/such/dir/decisions.jsonl"), 1.0),
    );
    let err = Gateway::new(config).unwrap_err();
    assert!(matches!(err, BuildError::DecisionLog { .. }));
    assert!(err.to_string().contains("/no/such/dir/decisions.jsonl"));
}

#[tokio::test]
async fn no_log_without_a_rate() {
    let sim = Upstream::zero(&[MODEL]).await;
    let path = log_path("off");
    let gw = TestGateway::start(fleet(
        &urls(&[&sim]),
        &format!(
            "{FAST_HEALTH}observability:\n  decision_log:\n    path: {}\n",
            path.display()
        ),
    ))
    .await;
    let (status, _) = gw.post_json("/v1/chat/completions", &chat("x", 1)).await;
    assert_eq!(status, 200);
    tokio::time::sleep(Duration::from_millis(300)).await;
    assert!(!path.exists());
}

#[tokio::test]
async fn candidates_are_logged_as_the_policy_saw_them() {
    let sim = Upstream::zero(&[MODEL]).await;
    let path = log_path("before");
    let text = decision_config(&path, 1.0);
    let mut config = fleet(&urls(&[&sim]), &text);
    config.replicas[0].max_concurrency = Some(1);
    let gw = TestGateway::start(config).await;
    let (status, _) = gw.post_json("/v1/chat/completions", &chat("x", 1)).await;
    assert_eq!(status, 200);
    let logged = lines(&path, 1).await;
    let _ = std::fs::remove_file(&path);
    let winner = &logged[0]["candidates"][0];
    assert_eq!(logged[0]["chosen"], "r0");
    assert_eq!(winner["eligible"], true, "{winner}");
    assert_eq!(winner["in_flight"], 0, "{winner}");
    assert_eq!(winner["in_flight_tokens"], 0, "{winner}");
}
