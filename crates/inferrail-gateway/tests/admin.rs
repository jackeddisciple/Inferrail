mod common;

use std::collections::BTreeSet;
use std::time::Duration;

use common::*;
use futures::StreamExt;
use inferrail_core::config::PolicyKind;
use inferrail_core::ids::ModelId;
use inferrail_load::parse::{name, sample};
use inferrail_sim::latency::LatencyModel;
use serde_json::json;

/// Every metric family and label key set on the page, without values, so
/// renames and label changes show up in review.
fn shape(page: &str) -> String {
    let mut lines = BTreeSet::new();
    for line in page.lines() {
        if let Some(rest) = line.strip_prefix("# TYPE ") {
            lines.insert(format!("TYPE {rest}"));
        } else if let Some(s) = sample(line) {
            let mut keys: Vec<_> = s.labels().map(|(k, _)| k).filter(|k| *k != "le").collect();
            keys.sort_unstable();
            lines.insert(format!("{}{{{}}}", s.name, keys.join(",")));
        } else {
            assert!(
                line == "# EOF" || line.starts_with("# HELP ") || name(line).is_none(),
                "unexpected line {line:?}"
            );
        }
    }
    lines.into_iter().collect::<Vec<_>>().join("\n")
}

#[tokio::test]
async fn metric_names_and_labels() {
    let sims = [
        Upstream::spawn(sim_config(
            &[MODEL],
            LatencyModel {
                decode_token_secs: 0.02,
                ..LatencyModel::ZERO
            },
        ))
        .await,
        Upstream::zero(&[MODEL]).await,
    ];
    let mut config = fleet(
        &urls(&[&sims[0], &sims[1]]),
        &format!("{FAST_HEALTH}routing:\n  default_policy: round_robin\nload:\n  max_age: 5s\n"),
    );
    config.models[0].tokenizer = Some("builtin:sim".to_owned());
    for (name, policy) in [
        ("model-b", PolicyKind::CostModel),
        ("model-c", PolicyKind::PrefixHash),
    ] {
        let mut other = config.models[0].clone();
        other.name = ModelId::new(name).unwrap();
        other.policy = Some(policy);
        for replica in &mut config.replicas {
            replica.models.push(other.name.clone());
        }
        config.models.push(other);
    }
    config.models[0].shadow = Some(vec![PolicyKind::CostModel]);
    let gw = TestGateway::start(config).await;
    let admin = serve_admin(&gw.gateway).await;
    gw.stream(&streaming(chat("one", 3))).await;
    let (status, _) = gw.post_json("/v1/chat/completions", &chat("two", 3)).await;
    assert_eq!(status, 200);
    let bad = json!({"model": MODEL, "messages": [], "max_tokens": 1, "n": 2});
    assert_eq!(gw.post_json("/v1/chat/completions", &bad).await.0, 400);
    let ghost = json!({"model": "ghost", "messages": []});
    assert_eq!(gw.post_json("/v1/chat/completions", &ghost).await.0, 404);
    sims[1].set_request_fault(Fault::Status(503));
    for i in 0..8 {
        assert_eq!(
            gw.post_json(
                "/v1/completions",
                &json!({"model": MODEL, "prompt": format!("p{i}"), "max_tokens": 1})
            )
            .await
            .0,
            200
        );
    }
    let response = gw
        .post("/v1/chat/completions", &streaming(chat("bye", 100)))
        .await;
    let mut body = response.bytes_stream();
    body.next().await.unwrap().unwrap();
    drop(body);
    gw.wait_idle().await;
    wait_until(Duration::from_secs(2), || {
        ["r0", "r1"]
            .iter()
            .all(|id| gw.replica(id).engine_load().latest().is_some())
    })
    .await;
    assert!(
        gw.replica("r1")
            .health()
            .is_ejected(gw.gateway.clock().now())
    );
    wait_until(Duration::from_secs(2), || {
        gw.gateway
            .render_metrics()
            .contains("inferrail_shadow_estimated_ttft_seconds_total{")
    })
    .await;

    let response = gw
        .client
        .get(format!("{admin}/metrics"))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    assert_eq!(
        response.headers()["content-type"],
        "application/openmetrics-text; version=1.0.0; charset=utf-8"
    );
    let page = response.text().await.unwrap();
    insta::assert_snapshot!(shape(&page));
}

#[tokio::test]
async fn metrics_live_on_the_admin_port_only() {
    let sim = Upstream::zero(&[MODEL]).await;
    let gw = TestGateway::start(fleet(&urls(&[&sim]), FAST_HEALTH)).await;
    let admin = serve_admin(&gw.gateway).await;
    let client = &gw.client;
    assert_eq!(
        client
            .get(gw.url("/metrics"))
            .send()
            .await
            .unwrap()
            .status(),
        404
    );
    assert_eq!(
        client
            .get(format!("{admin}/health"))
            .send()
            .await
            .unwrap()
            .status(),
        200
    );
    assert_eq!(
        client
            .get(format!("{admin}/v1/models"))
            .send()
            .await
            .unwrap()
            .status(),
        404
    );
    assert_eq!(
        client
            .post(format!("{admin}/metrics"))
            .send()
            .await
            .unwrap()
            .status(),
        405
    );
    let page = client
        .get(format!("{admin}/metrics"))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    assert!(page.ends_with("# EOF\n"));
    assert!(page.contains("inferrail_replica_healthy{replica=\"r0\"} 1"));
}

#[tokio::test]
async fn the_admin_listener_stops_on_shutdown() {
    let sim = Upstream::zero(&[MODEL]).await;
    let gw = TestGateway::start(fleet(&urls(&[&sim]), FAST_HEALTH)).await;
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let (stop, stopped) = tokio::sync::oneshot::channel::<()>();
    let task = tokio::spawn(gw.gateway.clone().serve_admin(listener, async move {
        let _ = stopped.await;
    }));
    let url = format!("http://{addr}/health");
    assert_eq!(gw.client.get(&url).send().await.unwrap().status(), 200);
    stop.send(()).unwrap();
    task.await.unwrap().unwrap();
    assert!(reqwest::Client::new().get(&url).send().await.is_err());
}
