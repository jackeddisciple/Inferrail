mod common;

use std::time::Duration;

use common::*;
use futures::StreamExt;
use inferrail_load::parse::samples;
use inferrail_sim::latency::LatencyModel;
use serde_json::json;

fn sum(page: &str, name: &str, labels: &[(&str, &str)]) -> f64 {
    samples(page)
        .filter(|s| {
            s.name == name
                && labels
                    .iter()
                    .all(|(k, v)| s.label(k).as_deref() == Some(*v))
        })
        .map(|s| s.value)
        .sum()
}

#[tokio::test]
async fn requests_ttft_and_decisions_are_counted_per_replica() {
    let sims = [
        Upstream::zero(&[MODEL]).await,
        Upstream::zero(&[MODEL]).await,
    ];
    let gw = TestGateway::start(fleet(
        &urls(&[&sims[0], &sims[1]]),
        &format!("{FAST_HEALTH}routing:\n  default_policy: round_robin\n"),
    ))
    .await;
    for i in 0..6 {
        gw.stream(&streaming(chat(&format!("stream {i}"), 3))).await;
        let (status, _) = gw.post_json("/v1/chat/completions", &chat("full", 3)).await;
        assert_eq!(status, 200);
    }
    let (status, _) = gw
        .post_json(
            "/v1/chat/completions",
            &json!({"model": MODEL, "messages": [], "max_tokens": 1, "n": 2}),
        )
        .await;
    assert_eq!(status, 400);
    gw.wait_idle().await;
    let page = gw.gateway.render_metrics();
    let requests = "inferrail_requests_total";
    assert_eq!(sum(&page, requests, &[("status", "2xx")]), 12.0);
    assert_eq!(sum(&page, requests, &[("status", "4xx")]), 1.0);
    for replica in ["r0", "r1"] {
        let route = [("policy", "round_robin"), ("replica", replica)];
        assert!(
            sum(&page, "inferrail_routing_decisions_total", &route) >= 6.0,
            "{replica}"
        );
    }
    assert_eq!(sum(&page, "inferrail_ttft_seconds_count", &[]), 6.0);
    assert_eq!(
        sum(&page, "inferrail_request_duration_seconds_count", &[]),
        13.0
    );
    assert_eq!(sum(&page, "inferrail_request_errors_total", &[]), 0.0);
    assert_eq!(
        sum(
            &page,
            "inferrail_model_policy",
            &[("model", MODEL), ("policy", "round_robin")]
        ),
        1.0
    );
}

#[tokio::test]
async fn retries_disconnects_and_gateway_errors_are_counted() {
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
    sims[1].set_request_fault(Fault::Status(503));
    let gw = TestGateway::start(fleet(
        &urls(&[&sims[0], &sims[1]]),
        &format!("{FAST_HEALTH}routing:\n  default_policy: round_robin\n"),
    ))
    .await;
    for _ in 0..4 {
        let (status, _) = gw.post_json("/v1/chat/completions", &chat("x", 1)).await;
        assert_eq!(status, 200);
    }
    let response = gw
        .post("/v1/chat/completions", &streaming(chat("leave early", 200)))
        .await;
    let mut bytes = response.bytes_stream();
    bytes.next().await.unwrap().unwrap();
    drop(bytes);
    gw.wait_idle().await;
    let (status, _) = gw
        .post_json(
            "/v1/chat/completions",
            &json!({"model": "ghost", "messages": []}),
        )
        .await;
    assert_eq!(status, 404);
    let missing = gw.client.get(gw.url("/nowhere")).send().await.unwrap();
    assert_eq!(missing.status(), 404);

    let page = gw.gateway.render_metrics();
    assert!(
        sum(
            &page,
            "inferrail_upstream_retries_total",
            &[("model", MODEL)]
        ) >= 1.0
    );
    assert_eq!(
        sum(
            &page,
            "inferrail_client_disconnects_total",
            &[("model", MODEL)]
        ),
        1.0
    );
    let cancelled = [("replica", "r0"), ("status", "cancelled")];
    assert_eq!(sum(&page, "inferrail_requests_total", &cancelled), 1.0);
    let ghost = [("model", ""), ("reason", "NotFoundError")];
    assert_eq!(sum(&page, "inferrail_request_errors_total", &ghost), 2.0);
    let unknown = [("model", ""), ("replica", ""), ("status", "4xx")];
    assert_eq!(sum(&page, "inferrail_requests_total", &unknown), 1.0);
}

#[tokio::test]
async fn replica_gauges_show_gateway_and_engine_load() {
    let sims = [
        Upstream::zero(&[MODEL]).await,
        Upstream::zero(&[MODEL]).await,
    ];
    let gw = TestGateway::start(fleet(&urls(&[&sims[0], &sims[1]]), FAST_HEALTH)).await;
    for _ in 0..3 {
        let (status, _) = gw
            .post_json(
                "/v1/completions",
                &json!({"model": MODEL, "prompt": "hi there", "max_tokens": 2}),
            )
            .await;
        assert_eq!(status, 200);
    }
    let r0 = gw.replica("r0");
    let r1 = gw.replica("r1");
    wait_until(Duration::from_secs(2), || {
        [&r0, &r1].iter().all(|r| {
            r.engine_load()
                .latest()
                .is_some_and(|s| s.load.prefix_queries > 0 || r.peak_in_flight() == 0)
        })
    })
    .await;
    let page = gw.gateway.render_metrics();
    for replica in ["r0", "r1"] {
        let labels = [("replica", replica)];
        assert_eq!(sum(&page, "inferrail_replica_healthy", &labels), 1.0);
        assert_eq!(sum(&page, "inferrail_replica_ejected", &labels), 0.0);
        assert_eq!(sum(&page, "inferrail_replica_in_flight", &labels), 0.0);
        assert_eq!(sum(&page, "inferrail_replica_load_stale", &labels), 0.0);
        assert!(sum(&page, "inferrail_replica_load_age_seconds", &labels) < 0.25);
        assert_eq!(sum(&page, "inferrail_replica_engine_running", &labels), 0.0);
        assert!(
            sum(
                &page,
                "inferrail_load_scrapes_total",
                &[("replica", replica), ("result", "ok")]
            ) >= 1.0
        );
    }
    assert!(
        sum(
            &page,
            "inferrail_replica_engine_prefix_cache_queries_total",
            &[]
        ) > 0.0
    );
    sims[1].kill();
    wait_until(Duration::from_secs(2), || {
        let page = gw.gateway.render_metrics();
        sum(&page, "inferrail_replica_load_stale", &[("replica", "r1")]) == 1.0
            && sum(
                &page,
                "inferrail_load_scrapes_total",
                &[("replica", "r1"), ("result", "unreachable")],
            ) >= 1.0
    })
    .await;
}
