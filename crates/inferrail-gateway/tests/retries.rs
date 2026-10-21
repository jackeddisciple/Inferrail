//! Retries happen only before the first response byte, on another replica,
//! and never more than the configured number of attempts.

mod common;

use std::time::{Duration, Instant};

use common::*;
use futures::StreamExt;
use proptest::prelude::*;
use serde_json::{Value, json};

const NO_EJECTION: &str = "health:\n  unhealthy_threshold: 1000\n";

async fn three() -> [Upstream; 3] {
    [
        Upstream::zero(&[MODEL]).await,
        Upstream::zero(&[MODEL]).await,
        Upstream::zero(&[MODEL]).await,
    ]
}

fn all(sims: &[Upstream]) -> Vec<String> {
    sims.iter().map(Upstream::url).collect()
}

#[tokio::test]
async fn refused_closed_and_unavailable_replicas_cause_zero_errors() {
    let sims = three().await;
    sims[0].kill();
    sims[1].set_fault(Fault::Close);
    let gw = TestGateway::start(fleet(
        &all(&sims),
        &format!("routing:\n  default_policy: round_robin\n{NO_EJECTION}"),
    ))
    .await;
    for i in 0..30 {
        if i == 15 {
            sims[1].set_fault(Fault::Status(503));
        }
        let body = if i % 2 == 0 {
            chat("x", 2)
        } else {
            streaming(chat("x", 2))
        };
        let response = gw.post("/v1/chat/completions", &body).await;
        assert_eq!(response.status(), 200, "request {i}");
        assert_eq!(response.headers()["x-inferrail-replica"], "r2");
        let _ = response.bytes().await.unwrap();
    }
    assert_eq!(sims[2].hits(), 30);
    gw.wait_idle().await;
}

#[tokio::test]
async fn attempts_are_bounded_and_the_last_answer_is_returned() {
    let sims = three().await;
    for sim in &sims {
        sim.set_fault(Fault::Status(503));
    }
    let gw = TestGateway::start(fleet(
        &all(&sims),
        &format!("retry:\n  max_attempts: 2\n  backoff: 40ms\n{NO_EJECTION}"),
    ))
    .await;
    for round in 1..=5u64 {
        let started = Instant::now();
        let (status, body) = gw.post_json("/v1/chat/completions", &chat("x", 1)).await;
        assert_eq!(status, 503);
        assert_eq!(body["error"]["message"], "injected");
        assert!(started.elapsed() >= Duration::from_millis(40));
        let hits: u64 = sims.iter().map(Upstream::hits).sum();
        assert_eq!(hits, 2 * round);
    }
}

#[tokio::test]
async fn transport_failures_everywhere_give_a_bad_gateway() {
    let sims = three().await;
    for sim in &sims {
        sim.set_fault(Fault::Close);
    }
    let gw = TestGateway::start(fleet(&all(&sims), NO_EJECTION)).await;
    let (status, body) = gw
        .post_json("/v1/completions", &json!({"prompt": "x"}))
        .await;
    assert_eq!(status, 502);
    assert_eq!(body["error"]["type"], "BadGatewayError");
    let hits: u64 = sims.iter().map(Upstream::hits).sum();
    assert_eq!(hits, 3);
}

#[tokio::test]
async fn internal_errors_and_client_errors_are_not_retried() {
    let sims = three().await;
    sims[0].set_fault(Fault::Status(500));
    let gw = TestGateway::start(fleet(
        &all(&sims),
        &format!("routing:\n  default_policy: round_robin\n{NO_EJECTION}"),
    ))
    .await;
    let mut statuses = Vec::new();
    for _ in 0..3 {
        statuses.push(
            gw.post("/v1/chat/completions", &chat("x", 1))
                .await
                .status()
                .as_u16(),
        );
    }
    statuses.sort_unstable();
    assert_eq!(statuses, [200, 200, 500]);
    assert_eq!(sims.iter().map(Upstream::hits).sum::<u64>(), 3);

    let mut bad = chat("x", 1);
    bad["n"] = json!(3);
    for _ in 0..3 {
        let (status, _) = gw.post_json("/v1/chat/completions", &bad).await;
        assert!(status == 400 || status == 500, "{status}");
    }
    assert_eq!(sims.iter().map(Upstream::hits).sum::<u64>(), 6);
}

#[tokio::test]
async fn a_stream_that_dies_midway_is_never_retried() {
    let sims = [
        Upstream::spawn(sim_config(&[MODEL], decode(5.0))).await,
        Upstream::spawn(sim_config(&[MODEL], decode(5.0))).await,
    ];
    let gw = TestGateway::start(fleet(&all(&sims), NO_EJECTION)).await;
    let response = gw
        .post("/v1/chat/completions", &streaming(chat("long", 500)))
        .await;
    let replica = response.headers()["x-inferrail-replica"]
        .to_str()
        .unwrap()
        .to_owned();
    let victim = &sims[usize::from(replica == "r1")];
    let mut stream = response.bytes_stream();
    stream.next().await.expect("first chunk").expect("ok");
    victim.kill();
    let mut failed = false;
    while let Some(chunk) = stream.next().await {
        if chunk.is_err() {
            failed = true;
        }
    }
    assert!(failed, "client must see the broken stream");
    assert_eq!(sims.iter().map(Upstream::hits).sum::<u64>(), 1);
    gw.wait_idle().await;
}

#[tokio::test]
async fn a_hanging_replica_times_out_once_the_deadline_passes() {
    let sim = Upstream::zero(&[MODEL]).await;
    sim.set_fault(Fault::Hang);
    let gw = TestGateway::start(fleet(
        &[sim.url()],
        "server:\n  request_timeout: 300ms\nupstream:\n  connect_timeout: 100ms\n",
    ))
    .await;
    let started = Instant::now();
    let (status, body) = gw.post_json("/v1/chat/completions", &chat("x", 1)).await;
    let took = started.elapsed();
    assert_eq!(status, 504, "{body}");
    assert!(
        took >= Duration::from_millis(290) && took < Duration::from_secs(2),
        "{took:?}"
    );
    assert_eq!(sim.hits(), 1);
    gw.wait_idle().await;
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Behaviour {
    Good,
    Refuses,
    Closes,
    Unavailable,
    Fails,
}

fn behaviour() -> impl Strategy<Value = Behaviour> {
    prop_oneof![
        3 => Just(Behaviour::Good),
        1 => Just(Behaviour::Refuses),
        1 => Just(Behaviour::Closes),
        1 => Just(Behaviour::Unavailable),
        1 => Just(Behaviour::Fails),
    ]
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 40, ..ProptestConfig::default() })]

    #[test]
    fn success_exactly_when_a_good_replica_is_within_reach(
        fleet_behaviour in prop::collection::vec(behaviour(), 1..5),
        attempts in 1u32..5,
        seed: u64,
        stream: bool,
    ) {
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let mut sims = Vec::new();
            for b in &fleet_behaviour {
                let sim = Upstream::zero(&[MODEL]).await;
                match b {
                    Behaviour::Good => {}
                    Behaviour::Refuses => sim.kill(),
                    Behaviour::Closes => sim.set_fault(Fault::Close),
                    Behaviour::Unavailable => sim.set_fault(Fault::Status(503)),
                    Behaviour::Fails => sim.set_fault(Fault::Status(500)),
                }
                sims.push(sim);
            }
            let gw = TestGateway::start(fleet(
                &all(&sims),
                &format!(
                    "routing:\n  default_policy: random\n  seed: {seed}\nretry:\n  max_attempts: {attempts}\n  backoff: 1ms\n{NO_EJECTION}"
                ),
            ))
            .await;
            let count = |b: Behaviour| fleet_behaviour.iter().filter(|&&x| x == b).count();
            let retryable = count(Behaviour::Refuses) + count(Behaviour::Closes) + count(Behaviour::Unavailable);
            for _ in 0..4 {
                let before: u64 = sims.iter().map(Upstream::hits).sum();
                let body = if stream { streaming(chat("x", 2)) } else { chat("x", 2) };
                let response = gw.post("/v1/chat/completions", &body).await;
                let status = response.status().as_u16();
                let text = response.text().await.unwrap();
                let hits = sims.iter().map(Upstream::hits).sum::<u64>() - before;
                prop_assert!(hits <= u64::from(attempts));
                if count(Behaviour::Fails) == 0 && count(Behaviour::Good) > 0
                    && attempts as usize > retryable
                {
                    prop_assert_eq!(status, 200, "{}", text);
                }
                if count(Behaviour::Good) == 0 {
                    prop_assert!(status >= 500, "{}", status);
                }
                if status == 200 && !stream {
                    let value: Value = serde_json::from_str(&text).unwrap();
                    prop_assert_eq!(&value["usage"]["completion_tokens"], &json!(2));
                }
            }
            Ok(())
        })?;
    }
}
