//! The gateway binary: config checking, serving and a clean exit on SIGTERM.

mod common;

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use common::*;

const BIN: &str = env!("CARGO_BIN_EXE_inferrail-gateway");

fn repo(path: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join(path)
}

fn temp_config(name: &str, text: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!(
        "inferrail-{name}-{}-{}.yaml",
        std::process::id(),
        Instant::now().elapsed().as_nanos()
    ));
    std::fs::write(&path, text).expect("write config");
    path
}

#[test]
fn check_accepts_shipped_configs_and_rejects_bad_ones() {
    for (config, summary) in [
        (
            "deploy/inferrail.example.yaml",
            "2 models, 3 replicas, 1 DNS groups",
        ),
        ("deploy/local.yaml", "1 models, 3 replicas, 0 DNS groups"),
        (
            "deploy/compose/gateway.yaml",
            "1 models, 3 replicas, 0 DNS groups",
        ),
    ] {
        let out = Command::new(BIN)
            .args(["--check", "--config"])
            .arg(repo(config))
            .output()
            .expect("run");
        assert!(
            out.status.success(),
            "{}",
            String::from_utf8_lossy(&out.stderr)
        );
        assert!(String::from_utf8_lossy(&out.stdout).contains(summary));
    }
    let bad = temp_config(
        "bad",
        "models: [{name: m}]\nreplicas: [{id: r, url: 'ftp://x', models: [m]}]\n",
    );
    let out = Command::new(BIN)
        .arg("--check")
        .arg("-c")
        .arg(&bad)
        .output()
        .expect("run");
    let _ = std::fs::remove_file(&bad);
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("scheme must be http or https"));

    let https = temp_config(
        "https",
        "models: [{name: m}]\nreplicas: [{id: r, url: 'https://engine.internal:8443', models: [m]}]\n",
    );
    let out = Command::new(BIN)
        .arg("--check")
        .arg("-c")
        .arg(&https)
        .output()
        .expect("run");
    let _ = std::fs::remove_file(&https);
    assert!(
        !out.status.success(),
        "check must refuse what startup refuses"
    );
    assert!(String::from_utf8_lossy(&out.stderr).contains("uses https"));

    let out = Command::new(BIN)
        .args(["--check", "-c", "/no/such/file.yaml"])
        .output()
        .expect("run");
    assert!(!out.status.success());
}

#[cfg(unix)]
#[tokio::test]
async fn serves_and_exits_cleanly_on_sigterm() {
    let sim = Upstream::zero(&[MODEL]).await;
    let free = || {
        let probe = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        probe.local_addr().unwrap().port()
    };
    let (port, admin) = (free(), free());
    let text = format!(
        "server:\n  listen: 127.0.0.1:{port}\n  admin_listen: 127.0.0.1:{admin}\nmodels:\n  - name: {MODEL}\nreplicas:\n  - id: r0\n    url: {}\n    models: [{MODEL}]\n",
        sim.url()
    );
    let path = temp_config("serve", &text);
    let mut child = Command::new(BIN)
        .arg("-c")
        .arg(&path)
        .env("INFERRAIL_LOG", "warn")
        .stdout(Stdio::null())
        .spawn()
        .expect("spawn");
    let client = reqwest::Client::new();
    let base = format!("http://127.0.0.1:{port}");
    let started = Instant::now();
    while client.get(format!("{base}/health")).send().await.is_err() {
        assert!(
            started.elapsed() < Duration::from_secs(10),
            "gateway never came up"
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    let reply: serde_json::Value = client
        .post(format!("{base}/v1/chat/completions"))
        .json(&chat("hello", 3))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(reply["usage"]["completion_tokens"], 3);
    let metrics = client
        .get(format!("http://127.0.0.1:{admin}/metrics"))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    assert!(metrics.contains("inferrail_requests_total{"), "{metrics}");
    let status = Command::new("kill")
        .arg(child.id().to_string())
        .status()
        .unwrap();
    assert!(status.success());
    let started = Instant::now();
    let exit = loop {
        if let Some(exit) = child.try_wait().unwrap() {
            break exit;
        }
        assert!(
            started.elapsed() < Duration::from_secs(10),
            "no exit after SIGTERM"
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    };
    let _ = std::fs::remove_file(&path);
    assert!(exit.success(), "{exit:?}");
}

#[cfg(unix)]
#[tokio::test]
async fn exports_spans_over_otlp_when_configured() {
    use std::sync::{Arc, Mutex};

    let received: Arc<Mutex<Vec<(String, usize)>>> = Arc::default();
    let sink = Arc::clone(&received);
    let collector = axum::Router::new().route(
        "/v1/traces",
        axum::routing::post(move |headers: hyper::HeaderMap, body: axum::body::Bytes| {
            let sink = Arc::clone(&sink);
            async move {
                let kind = headers
                    .get("content-type")
                    .and_then(|v| v.to_str().ok())
                    .unwrap_or_default()
                    .to_owned();
                sink.lock().unwrap().push((kind, body.len()));
            }
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let otlp = format!("http://{}/v1/traces", listener.local_addr().unwrap());
    tokio::spawn(async move { axum::serve(listener, collector).await });

    let sim = Upstream::zero(&[MODEL]).await;
    let free = || {
        let probe = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        probe.local_addr().unwrap().port()
    };
    let (port, admin) = (free(), free());
    let text = format!(
        "server:\n  listen: 127.0.0.1:{port}\n  admin_listen: 127.0.0.1:{admin}\nmodels:\n  - name: {MODEL}\nreplicas:\n  - id: r0\n    url: {}\n    models: [{MODEL}]\nobservability:\n  otlp_endpoint: {otlp}\n  service_name: gateway-under-test\n",
        sim.url()
    );
    let path = temp_config("otlp", &text);
    let mut child = Command::new(BIN)
        .arg("-c")
        .arg(&path)
        .env("INFERRAIL_LOG", "warn")
        .stdout(Stdio::null())
        .spawn()
        .expect("spawn");
    let client = reqwest::Client::new();
    let base = format!("http://127.0.0.1:{port}");
    let started = Instant::now();
    while client.get(format!("{base}/health")).send().await.is_err() {
        assert!(
            started.elapsed() < Duration::from_secs(10),
            "gateway never came up"
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    let reply = client
        .post(format!("{base}/v1/chat/completions"))
        .json(&chat("traced", 2))
        .send()
        .await
        .unwrap();
    assert_eq!(reply.status(), 200);
    reply.bytes().await.unwrap();
    Command::new("kill")
        .arg(child.id().to_string())
        .status()
        .unwrap();
    let started = Instant::now();
    while received.lock().unwrap().is_empty() {
        assert!(
            started.elapsed() < Duration::from_secs(10),
            "no spans exported"
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    let exit = child.wait().unwrap();
    let _ = std::fs::remove_file(&path);
    assert!(exit.success(), "{exit:?}");
    let batches = received.lock().unwrap().clone();
    assert!(
        batches
            .iter()
            .all(|(kind, len)| kind == "application/x-protobuf" && *len > 0),
        "{batches:?}"
    );
}

#[test]
fn check_leaves_no_trace_but_still_refuses_an_unwritable_decision_log() {
    let dir = std::env::temp_dir().join(format!("inferrail-check-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("dir");
    let log = dir.join("decisions.jsonl");
    let config = |path: &Path| {
        temp_config(
            "decisions",
            &format!(
                "models: [{{name: m}}]\nreplicas: [{{id: r, url: 'http://h:1', models: [m]}}]\nobservability:\n  decision_log: {{path: '{}', sample_rate: 1.0}}\n",
                path.display()
            ),
        )
    };
    let good = config(&log);
    let out = Command::new(BIN)
        .args(["--check", "-c"])
        .arg(&good)
        .output()
        .expect("run");
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(!log.exists(), "--check created the decision log");

    let missing = config(&dir.join("no/such/dir/decisions.jsonl"));
    let out = Command::new(BIN)
        .args(["--check", "-c"])
        .arg(&missing)
        .output()
        .expect("run");
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("cannot open the decision log"));

    let is_dir = config(&dir);
    let out = Command::new(BIN)
        .args(["--check", "-c"])
        .arg(&is_dir)
        .output()
        .expect("run");
    assert!(!out.status.success());
    for path in [good, missing, is_dir] {
        let _ = std::fs::remove_file(path);
    }
    let _ = std::fs::remove_dir_all(&dir);
}
