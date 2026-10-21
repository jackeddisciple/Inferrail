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
    let port = {
        let probe = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        probe.local_addr().unwrap().port()
    };
    let text = format!(
        "server:\n  listen: 127.0.0.1:{port}\nmodels:\n  - name: {MODEL}\nreplicas:\n  - id: r0\n    url: {}\n    models: [{MODEL}]\n",
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
