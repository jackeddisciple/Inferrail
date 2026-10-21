//! Active probes and passive ejection take bad replicas out of rotation
//! quickly and bring them back once they recover.

mod common;

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
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
    let killed = Arc::new(AtomicBool::new(false));
    let after_kill = Arc::new(AtomicU64::new(0));
    let in_flight = Arc::new(AtomicU64::new(0));
    let sent = Arc::new(AtomicU64::new(0));
    let started = Instant::now();
    let mut tasks = Vec::new();
    for worker in 0..8 {
        let (gw, killed, after_kill, in_flight, sent) = (
            Arc::clone(&gw),
            Arc::clone(&killed),
            Arc::clone(&after_kill),
            Arc::clone(&in_flight),
            Arc::clone(&sent),
        );
        tasks.push(tokio::spawn(async move {
            while started.elapsed() < Duration::from_millis(1_500) {
                let body = if worker % 2 == 0 {
                    chat("x", 2)
                } else {
                    streaming(chat("x", 2))
                };
                let unstarted = killed.load(Ordering::SeqCst);
                sent.fetch_add(1, Ordering::Relaxed);
                let request = gw.client.post(gw.url("/v1/chat/completions")).json(&body);
                let ok = match request.send().await {
                    Ok(r) if r.status() == 200 => r.bytes().await.is_ok(),
                    _ => false,
                };
                if !ok {
                    let bucket = if unstarted { &after_kill } else { &in_flight };
                    bucket.fetch_add(1, Ordering::Relaxed);
                }
            }
        }));
    }
    tokio::time::sleep(Duration::from_millis(400)).await;
    killed.store(true, Ordering::SeqCst);
    sims[1].kill();
    let r1 = gw.replica("r1");
    let clock = Clock::default();
    let ejected_in = wait_until(Duration::from_secs(2), || !r1.is_routable(clock.now())).await;
    for task in tasks {
        task.await.unwrap();
    }
    let sent = sent.load(Ordering::Relaxed);
    assert!(ejected_in < Duration::from_secs(2), "{ejected_in:?}");
    assert_eq!(after_kill.load(Ordering::Relaxed), 0, "of {sent}");
    assert!(
        in_flight.load(Ordering::Relaxed) <= 8,
        "at most one in flight per worker"
    );
    assert!(sent > 200);
    assert!(!r1.health().probe_up());
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
    let mut windows: Vec<u64> = Vec::new();
    let mut last_until = 0;
    let started = Instant::now();
    while windows.len() < 4 && started.elapsed() < Duration::from_secs(10) {
        assert_eq!(
            gw.post("/v1/chat/completions", &chat("x", 1))
                .await
                .status(),
            200
        );
        let (until, now) = (r0.health().ejected_until(), clock.now());
        if until != last_until && now < until {
            windows.push((until - now) / 1_000_000);
            last_until = until;
        }
        tokio::time::sleep(Duration::from_millis(2)).await;
    }
    assert!(r0.health().probe_up(), "probes still pass");
    assert_eq!(windows.len(), 4, "{windows:?}");
    assert!(windows.iter().all(|&ms| ms <= 400), "capped: {windows:?}");
    assert!(windows[0] <= 100, "starts at the base: {windows:?}");
    assert!(windows[3] > windows[0], "grows: {windows:?}");
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

fn ejected(gw: &TestGateway) -> usize {
    let now = gw.gateway.clock().now();
    gw.gateway
        .registry()
        .load()
        .replicas()
        .iter()
        .filter(|r| r.health().is_ejected(now))
        .count()
}

#[tokio::test]
async fn replicas_already_down_do_not_count_toward_the_cap() {
    let sims = fleet_of(4).await;
    let gw = TestGateway::start(fleet(
        &all(&sims),
        &format!("{FAST_HEALTH}routing:\n  default_policy: round_robin\n"),
    ))
    .await;
    sims[0].kill();
    sims[1].kill();
    let (r0, r1) = (gw.replica("r0"), gw.replica("r1"));
    wait_until(Duration::from_secs(2), || {
        !r0.health().probe_up() && !r1.health().probe_up()
    })
    .await;
    sims[2].set_request_fault(Fault::Status(500));
    sims[3].set_request_fault(Fault::Status(500));
    for _ in 0..20 {
        let _ = gw.post("/v1/chat/completions", &chat("poison", 1)).await;
    }
    sims[2].set_request_fault(Fault::None);
    sims[3].set_request_fault(Fault::None);
    let (status, body) = gw
        .post_json("/v1/chat/completions", &chat("legit", 1))
        .await;
    assert_eq!(status, 200, "{body}");
}

#[tokio::test]
async fn poison_requests_cannot_eject_the_whole_pool() {
    let sims = fleet_of(3).await;
    let gw = TestGateway::start(fleet(
        &all(&sims),
        &format!("{FAST_HEALTH}routing:\n  default_policy: round_robin\n"),
    ))
    .await;
    for sim in &sims {
        sim.set_request_fault(Fault::Status(500));
    }
    for _ in 0..30 {
        let status = gw
            .post("/v1/chat/completions", &chat("poison", 1))
            .await
            .status();
        assert_eq!(status, 500);
        assert!(ejected(&gw) <= 1, "half of three is one");
    }
    assert_eq!(ejected(&gw), 1);
    for sim in &sims {
        sim.set_request_fault(Fault::None);
    }
    let (status, body) = gw
        .post_json("/v1/chat/completions", &chat("legit", 1))
        .await;
    assert_eq!(status, 200, "{body}");
}

#[tokio::test]
async fn a_sole_replica_is_never_ejected_by_failed_requests() {
    let sims = fleet_of(1).await;
    let gw = TestGateway::start(fleet(&all(&sims), FAST_HEALTH)).await;
    sims[0].set_request_fault(Fault::Status(500));
    for _ in 0..10 {
        assert_eq!(
            gw.post("/v1/chat/completions", &chat("x", 1))
                .await
                .status(),
            500
        );
    }
    assert_eq!(ejected(&gw), 0);
    sims[0].set_request_fault(Fault::None);
    assert_eq!(
        gw.post("/v1/chat/completions", &chat("x", 1))
            .await
            .status(),
        200
    );
}

#[tokio::test]
async fn the_ejection_cap_is_configurable() {
    for (percent, cap) in [(0, 0), (100, 2)] {
        let sims = fleet_of(2).await;
        let gw = TestGateway::start(fleet(
            &all(&sims),
            &format!(
                "{FAST_HEALTH}  max_ejection_percent: {percent}\nrouting:\n  default_policy: round_robin\n"
            ),
        ))
        .await;
        for sim in &sims {
            sim.set_request_fault(Fault::Status(500));
        }
        for _ in 0..12 {
            let _ = gw.post("/v1/chat/completions", &chat("x", 1)).await;
        }
        assert_eq!(ejected(&gw), cap, "max_ejection_percent {percent}");
    }
}
