//! Active probes and passive ejection take bad replicas out of rotation
//! quickly and bring them back once they recover.

mod common;

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use common::*;
use inferrail_control::Clock;

async fn fleet_of(n: usize) -> Vec<Upstream> {
    let mut sims = Vec::new();
    for _ in 0..n {
        sims.push(Upstream::zero(&[MODEL]).await);
    }
    sims
}

fn all(sims: &[Upstream]) -> Vec<String> {
    sims.iter().map(Upstream::url).collect()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_replica_killed_under_load_is_ejected_fast_with_zero_errors() {
    let sims = fleet_of(3).await;
    let gw = Arc::new(TestGateway::start(fleet(&all(&sims), FAST_HEALTH)).await);
    let errors = Arc::new(AtomicU64::new(0));
    let sent = Arc::new(AtomicU64::new(0));
    let started = Instant::now();
    let mut tasks = Vec::new();
    for worker in 0..8 {
        let (gw, errors, sent) = (Arc::clone(&gw), Arc::clone(&errors), Arc::clone(&sent));
        tasks.push(tokio::spawn(async move {
            while started.elapsed() < Duration::from_millis(1_500) {
                let body = if worker % 2 == 0 {
                    chat("x", 2)
                } else {
                    streaming(chat("x", 2))
                };
                let ok = match gw
                    .client
                    .post(gw.url("/v1/chat/completions"))
                    .json(&body)
                    .send()
                    .await
                {
                    Ok(r) if r.status() == 200 => r.bytes().await.is_ok(),
                    _ => false,
                };
                sent.fetch_add(1, Ordering::Relaxed);
                if !ok {
                    errors.fetch_add(1, Ordering::Relaxed);
                }
            }
        }));
    }
    tokio::time::sleep(Duration::from_millis(400)).await;
    let killed_at = Instant::now();
    sims[1].kill();
    let r1 = gw.replica("r1");
    let clock = Clock::default();
    let ejected_in = wait_until(Duration::from_secs(2), || !r1.is_routable(clock.now())).await;
    for task in tasks {
        task.await.unwrap();
    }
    assert!(ejected_in < Duration::from_secs(2), "{ejected_in:?}");
    assert_eq!(
        errors.load(Ordering::Relaxed),
        0,
        "of {}",
        sent.load(Ordering::Relaxed)
    );
    assert!(sent.load(Ordering::Relaxed) > 200);
    assert!(
        !r1.health().probe_up(),
        "probes should agree after {:?}",
        killed_at.elapsed()
    );
    gw.wait_idle().await;
}

#[tokio::test]
async fn an_idle_dead_replica_is_marked_down_by_probes() {
    let sims = fleet_of(2).await;
    let gw = TestGateway::start(fleet(&all(&sims), FAST_HEALTH)).await;
    let r0 = gw.replica("r0");
    tokio::time::sleep(Duration::from_millis(120)).await;
    assert!(r0.health().probe_up());
    sims[0].kill();
    let took = wait_until(Duration::from_secs(2), || !r0.health().probe_up()).await;
    assert!(took < Duration::from_millis(500), "{took:?}");
    for _ in 0..10 {
        let response = gw.post("/v1/chat/completions", &chat("x", 1)).await;
        assert_eq!(response.headers()["x-inferrail-replica"], "r1");
    }
}

#[tokio::test]
async fn a_recovered_replica_is_readmitted_and_serves_again() {
    let sims = fleet_of(2).await;
    let gw = TestGateway::start(fleet(
        &all(&sims),
        &format!("{FAST_HEALTH}routing:\n  default_policy: round_robin\n"),
    ))
    .await;
    let r0 = gw.replica("r0");
    sims[0].set_fault(Fault::Status(503));
    wait_until(Duration::from_secs(2), || !r0.health().probe_up()).await;
    let before = sims[0].hits();
    for _ in 0..6 {
        assert_eq!(
            gw.post("/v1/chat/completions", &chat("x", 1))
                .await
                .status(),
            200
        );
    }
    assert_eq!(sims[0].hits(), before, "no traffic while down");

    sims[0].set_fault(Fault::None);
    let back = wait_until(Duration::from_secs(2), || r0.health().probe_up()).await;
    assert!(back < Duration::from_millis(500), "{back:?}");
    for _ in 0..6 {
        assert_eq!(
            gw.post("/v1/chat/completions", &chat("x", 1))
                .await
                .status(),
            200
        );
    }
    assert_eq!(sims[0].hits(), before + 3);

    sims[0].kill();
    wait_until(Duration::from_secs(2), || !r0.health().probe_up()).await;
    sims[0].restart().await;
    wait_until(Duration::from_secs(2), || r0.health().probe_up()).await;
    let replicas: Vec<String> = futures::future::join_all((0..4).map(|_| async {
        let response = gw.post("/v1/chat/completions", &chat("x", 1)).await;
        response.headers()["x-inferrail-replica"]
            .to_str()
            .unwrap()
            .to_owned()
    }))
    .await;
    assert!(replicas.iter().any(|r| r == "r0"), "{replicas:?}");
}

#[tokio::test]
async fn a_replica_that_keeps_failing_requests_backs_off_exponentially() {
    let sims = fleet_of(2).await;
    let health = "health:\n  interval: 50ms\n  timeout: 40ms\n  unhealthy_threshold: 2\n  ejection_backoff: 100ms\n  max_ejection_backoff: 400ms\n";
    let gw = TestGateway::start(fleet(
        &all(&sims),
        &format!("{health}routing:\n  default_policy: round_robin\n"),
    ))
    .await;
    sims[0].set_request_fault(Fault::Status(503));
    let r0 = gw.replica("r0");
    let clock = Clock::default();
    let mut windows = Vec::new();
    let mut last_until = 0;
    let started = Instant::now();
    while windows.len() < 4 && started.elapsed() < Duration::from_secs(5) {
        assert_eq!(
            gw.post("/v1/chat/completions", &chat("x", 1))
                .await
                .status(),
            200
        );
        let until = r0.health().ejected_until();
        if until != last_until && r0.health().is_ejected(clock.now()) {
            windows.push(until);
            last_until = until;
        }
        tokio::time::sleep(Duration::from_millis(2)).await;
    }
    assert!(r0.health().probe_up(), "probes still pass");
    assert_eq!(windows.len(), 4, "{windows:?}");
    let gaps: Vec<u64> = windows
        .windows(2)
        .map(|w| (w[1] - w[0]) / 1_000_000)
        .collect();
    assert!(
        gaps[0] >= 200 && gaps[1] >= 400 && gaps[2] >= 400,
        "{gaps:?}"
    );
    assert!(gaps[1] > gaps[0] + 100, "doubles: {gaps:?}");
    assert!(gaps[2] < 700, "capped at 400ms: {gaps:?}");
}

#[tokio::test]
async fn outlier_ejection_caps_errors_from_a_replica_that_answers_500() {
    let sims = fleet_of(2).await;
    let gw = TestGateway::start(fleet(
        &all(&sims),
        &format!("{FAST_HEALTH}routing:\n  default_policy: round_robin\n"),
    ))
    .await;
    sims[1].set_request_fault(Fault::Status(500));
    let mut errors = 0;
    for _ in 0..40 {
        if gw
            .post("/v1/chat/completions", &chat("x", 1))
            .await
            .status()
            != 200
        {
            errors += 1;
        }
    }
    assert_eq!(errors, 2, "unhealthy_threshold is 2");
}
