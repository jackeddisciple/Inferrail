//! Per replica concurrency and pending caps: a stuck or saturated replica
//! sheds load to the others instead of making clients wait.

mod common;

use std::time::{Duration, Instant};

use common::*;
use futures::StreamExt;

fn capped(urls: &[String], caps: &[&str], extra: &str) -> inferrail_core::config::Config {
    let mut text = format!("models:\n  - name: {MODEL}\nreplicas:\n");
    for (i, (url, cap)) in urls.iter().zip(caps).enumerate() {
        text.push_str(&format!(
            "  - id: r{i}\n    url: {url}\n    models: [{MODEL}]\n{cap}"
        ));
    }
    text.push_str(extra);
    config(&text)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_stuck_replica_stops_absorbing_requests_at_its_pending_cap() {
    let sims = [
        Upstream::zero(&[MODEL]).await,
        Upstream::zero(&[MODEL]).await,
    ];
    sims[0].set_request_fault(Fault::Hang);
    let gw = std::sync::Arc::new(
        TestGateway::start(capped(
            &[sims[0].url(), sims[1].url()],
            &["    max_pending: 4\n", ""],
            "server:\n  request_timeout: 1500ms\nupstream:\n  connect_timeout: 500ms\nrouting:\n  default_policy: round_robin\nhealth:\n  unhealthy_threshold: 100\n",
        ))
        .await,
    );
    let mut tasks = Vec::new();
    for i in 0..40 {
        let gw = std::sync::Arc::clone(&gw);
        tasks.push(tokio::spawn(async move {
            let started = Instant::now();
            let body = if i % 2 == 0 {
                chat("x", 2)
            } else {
                streaming(chat("x", 2))
            };
            let response = gw.post("/v1/chat/completions", &body).await;
            let status = response.status().as_u16();
            let _ = response.bytes().await;
            (status, started.elapsed())
        }));
        tokio::time::sleep(Duration::from_millis(2)).await;
    }
    let r0 = gw.replica("r0");
    wait_until(Duration::from_secs(1), || r0.pending() == 4).await;
    let mut fast = 0;
    let mut timed_out = 0;
    for task in tasks {
        let (status, took) = task.await.unwrap();
        match status {
            200 => {
                assert!(took < Duration::from_millis(500), "{took:?}");
                fast += 1;
            }
            504 => timed_out += 1,
            other => panic!("unexpected {other}"),
        }
    }
    assert_eq!((fast, timed_out), (36, 4));
    assert_eq!(sims[0].hits(), 4);
    assert_eq!(r0.peak_in_flight(), 4);
    gw.wait_idle().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrency_caps_hold_under_a_burst_of_streams() {
    let sims = [
        Upstream::spawn(sim_config(&[MODEL], decode(2.0))).await,
        Upstream::spawn(sim_config(&[MODEL], decode(2.0))).await,
        Upstream::spawn(sim_config(&[MODEL], decode(2.0))).await,
    ];
    let cap = "    max_concurrency: 20\n";
    let gw = TestGateway::start(capped(
        &sims.iter().map(Upstream::url).collect::<Vec<_>>(),
        &[cap, cap, cap],
        "",
    ))
    .await;
    let mut tasks = Vec::new();
    for i in 0..500 {
        let client = gw.client.clone();
        let url = gw.url("/v1/chat/completions");
        let body = streaming(chat(&format!("burst {i}"), 40));
        tasks.push(tokio::spawn(async move {
            let response = client.post(url).json(&body).send().await.expect("sent");
            let status = response.status().as_u16();
            let retry_after = response.headers().contains_key("retry-after");
            let mut stream = response.bytes_stream();
            while let Some(chunk) = stream.next().await {
                chunk.expect("chunk");
            }
            (status, retry_after)
        }));
    }
    let (mut ok, mut shed) = (0, 0);
    for task in tasks {
        match task.await.unwrap() {
            (200, _) => ok += 1,
            (503, true) => shed += 1,
            other => panic!("unexpected {other:?}"),
        }
    }
    assert_eq!(ok + shed, 500);
    assert!(ok >= 60, "{ok}");
    for id in ["r0", "r1", "r2"] {
        let replica = gw.replica(id);
        assert!(
            replica.peak_in_flight() <= 20,
            "{id}: {}",
            replica.peak_in_flight()
        );
    }
    gw.wait_idle().await;
}

#[tokio::test]
async fn a_full_replica_is_skipped_and_a_full_fleet_answers_fast() {
    let sims = [
        Upstream::spawn(sim_config(&[MODEL], decode(5.0))).await,
        Upstream::spawn(sim_config(&[MODEL], decode(5.0))).await,
    ];
    let cap = "    max_concurrency: 1\n";
    let gw = TestGateway::start(capped(
        &[sims[0].url(), sims[1].url()],
        &[cap, cap],
        "routing:\n  default_policy: round_robin\n",
    ))
    .await;
    let hold = gw
        .post("/v1/chat/completions", &streaming(chat("hold", 400)))
        .await;
    let held = hold.headers()["x-inferrail-replica"]
        .to_str()
        .unwrap()
        .to_owned();
    for _ in 0..5 {
        let response = gw.post("/v1/chat/completions", &chat("x", 1)).await;
        assert_eq!(response.status(), 200);
        assert_ne!(response.headers()["x-inferrail-replica"], held.as_str());
    }
    let other = gw
        .post("/v1/chat/completions", &streaming(chat("hold", 400)))
        .await;
    let started = Instant::now();
    let response = gw.post("/v1/chat/completions", &chat("x", 1)).await;
    let took = started.elapsed();
    assert_eq!(response.status(), 503);
    assert_eq!(response.headers()["retry-after"], "1");
    let body: serde_json::Value = response.json().await.unwrap();
    assert!(
        body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("at capacity")
    );
    assert!(took < Duration::from_millis(100), "{took:?}");
    drop((hold, other));
    gw.wait_idle().await;
}
