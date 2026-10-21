//! Heavy load, chaos and soak tests for the gateway. Ignored by default; run
//! with `make stress` (release build, output shown).

mod common;

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use common::*;
use futures::StreamExt;
use hyper::body::Bytes;
use inferrail_sim::latency::LatencyModel;
use serde_json::{Value, json};

fn percentile(sorted: &[Duration], p: f64) -> Duration {
    if sorted.is_empty() {
        return Duration::ZERO;
    }
    sorted[((sorted.len() - 1) as f64 * p).round() as usize]
}

fn summary(mut latencies: Vec<Duration>) -> [Duration; 3] {
    latencies.sort();
    [
        percentile(&latencies, 0.50),
        percentile(&latencies, 0.99),
        percentile(&latencies, 0.999),
    ]
}

fn rss_kib() -> u64 {
    std::process::Command::new("ps")
        .args(["-o", "rss=", "-p", &std::process::id().to_string()])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(0)
}

/// Deterministic xorshift so failures replay exactly.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn below(&mut self, n: u64) -> u64 {
        self.next() % n.max(1)
    }
}

async fn sims(n: usize, latency: LatencyModel) -> Vec<Upstream> {
    let mut out = Vec::new();
    for _ in 0..n {
        let mut config = sim_config(&[MODEL], latency);
        config.engine.max_num_seqs = 4_096;
        out.push(Upstream::spawn(config).await);
    }
    out
}

async fn drained(sims: &[Upstream], gw: &TestGateway) {
    let started = Instant::now();
    loop {
        let busy = sims.iter().any(|s| {
            let snap = s.snapshot();
            snap.running != 0 || snap.waiting != 0 || snap.kv_blocks_used != 0
        });
        if !busy && gw.in_flight() == 0 {
            break;
        }
        assert!(started.elapsed() < Duration::from_secs(5), "never drained");
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    let fleet = gw.gateway.registry().load();
    assert!(fleet.replicas().iter().all(|r| r.pending() == 0));
}

#[derive(Debug, Default)]
struct Run {
    latencies: Vec<Duration>,
    service: Vec<Duration>,
    lag: Vec<Duration>,
    errors: u64,
}

/// Open loop: request `i` is due at `start + i / rate` whatever happened to
/// earlier ones; latency counts from the due time, so stalls are not hidden.
async fn open_loop(
    client: &reqwest::Client,
    url: String,
    rate: u64,
    total: u64,
    body: impl Fn(u64) -> Value,
) -> Run {
    let interval = Duration::from_nanos(1_000_000_000 / rate);
    let started = tokio::time::Instant::now();
    let mut tasks = Vec::with_capacity(total as usize);
    let mut lag = Vec::with_capacity(total as usize);
    for i in 0..total {
        let due = started + interval * i as u32;
        tokio::time::sleep_until(due).await;
        lag.push(tokio::time::Instant::now() - due);
        let (client, url, body) = (client.clone(), url.clone(), body(i));
        tasks.push(tokio::spawn(async move {
            let sent = tokio::time::Instant::now();
            let ok = match client.post(url).json(&body).send().await {
                Ok(r) if r.status() == 200 => r.bytes().await.is_ok(),
                _ => false,
            };
            let done = tokio::time::Instant::now();
            (ok, done - due, done - sent)
        }));
    }
    let mut run = Run {
        lag,
        ..Run::default()
    };
    for task in tasks {
        let (ok, latency, service) = task.await.expect("task");
        if !ok {
            run.errors += 1;
        }
        run.latencies.push(latency);
        run.service.push(service);
    }
    run
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
#[ignore = "stress"]
async fn two_thousand_concurrent_streams_over_three_sims() {
    let sims = sims(3, decode(2.0)).await;
    let gw = TestGateway::start(fleet(&urls(&sims.iter().collect::<Vec<_>>()), "")).await;
    let started = Instant::now();
    let mut tasks = Vec::new();
    for i in 0..2_000 {
        let (client, url) = (gw.client.clone(), gw.url("/v1/chat/completions"));
        let body = streaming(chat(&format!("stream number {i}"), 32));
        tasks.push(tokio::spawn(async move {
            let begun = Instant::now();
            let text = client.post(url).json(&body).send().await?.text().await?;
            Ok::<_, reqwest::Error>((begun.elapsed(), text))
        }));
    }
    let mut latencies = Vec::new();
    for task in tasks {
        let (elapsed, text) = task.await.unwrap().expect("stream completed");
        assert_eq!(text.matches("data: ").count(), 35, "{text}");
        assert!(text.ends_with("data: [DONE]\n\n"));
        latencies.push(elapsed);
    }
    let [p50, p99, _] = summary(latencies);
    println!(
        "2000 streams via gateway: wall {:.2?}, p50 {p50:.2?}, p99 {p99:.2?}, hits {:?}",
        started.elapsed(),
        sims.iter().map(Upstream::hits).collect::<Vec<_>>()
    );
    drained(&sims, &gw).await;
    let finished: u64 = sims
        .iter()
        .map(|s| s.sim.metrics().finished("length"))
        .sum();
    assert_eq!(finished, 2_000);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
#[ignore = "stress"]
async fn open_loop_overhead_against_a_direct_sim() {
    let sim = Upstream::zero(&[MODEL]).await;
    let gw = TestGateway::start(fleet(&[sim.url()], "")).await;
    let client = reqwest::Client::new();
    for rate in [2_000u64, 5_000] {
        let total = 10_000;
        let body = |i: u64| chat(&format!("overhead {i}"), 4);
        let direct = open_loop(
            &client,
            format!("{}/v1/chat/completions", sim.url()),
            rate,
            total,
            body,
        )
        .await;
        let via = open_loop(&client, gw.url("/v1/chat/completions"), rate, total, body).await;
        assert_eq!(direct.errors + via.errors, 0);
        let [d50, d99, d999] = summary(direct.service);
        let [g50, g99, g999] = summary(via.service);
        let [_, due99, _] = summary(via.latencies);
        let [_, lag99, _] = summary(via.lag);
        println!(
            "open loop {rate}/s x {total}: direct p50 {:.3} p99 {:.3} p99.9 {:.3} ms | gateway p50 {:.3} p99 {:.3} p99.9 {:.3} ms | added p50 {:.3} p99 {:.3} p99.9 {:.3} ms | gateway p99 from due time {:.3} ms, generator lag p99 {:.3} ms",
            ms(d50),
            ms(d99),
            ms(d999),
            ms(g50),
            ms(g99),
            ms(g999),
            ms(g50.saturating_sub(d50)),
            ms(g99.saturating_sub(d99)),
            ms(g999.saturating_sub(d999)),
            ms(due99),
            ms(lag99),
        );
        assert!(g99.saturating_sub(d99) < Duration::from_millis(20));
    }
    drained(std::slice::from_ref(&sim), &gw).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
#[ignore = "stress"]
async fn a_sim_killed_and_restarted_under_open_loop_load() {
    let sims = sims(3, decode(1.0)).await;
    let gw = Arc::new(
        TestGateway::start(fleet(&urls(&sims.iter().collect::<Vec<_>>()), FAST_HEALTH)).await,
    );
    let unstarted = Arc::new(AtomicU64::new(0));
    let broken = Arc::new(AtomicU64::new(0));
    let rate = 1_000u64;
    let total = 6_000u64;
    let interval = Duration::from_nanos(1_000_000_000 / rate);
    let started = tokio::time::Instant::now();
    let mut tasks = Vec::new();
    let mut killed_at = None;
    let mut restarted_at = None;
    let mut ejected_at = None;
    let r1 = gw.replica("r1");
    let clock = gw.gateway.clock();
    for i in 0..total {
        tokio::time::sleep_until(started + interval * i as u32).await;
        if killed_at.is_some() && ejected_at.is_none() && !r1.is_routable(clock.now()) {
            ejected_at = Some(Instant::now());
        }
        if i == 1_500 {
            sims[1].kill();
            killed_at = Some(Instant::now());
        }
        if i == 3_500 {
            sims[1].restart().await;
            restarted_at = Some(Instant::now());
        }
        let (gw, unstarted, broken) =
            (Arc::clone(&gw), Arc::clone(&unstarted), Arc::clone(&broken));
        let after_kill = killed_at.is_some();
        tasks.push(tokio::spawn(async move {
            let fail = |counter: &AtomicU64| {
                let bucket = if after_kill { counter } else { &broken };
                bucket.fetch_add(1, Ordering::Relaxed);
            };
            let body = if i.is_multiple_of(2) {
                chat("x", 4)
            } else {
                streaming(chat("x", 4))
            };
            let response = match gw
                .client
                .post(gw.url("/v1/chat/completions"))
                .json(&body)
                .send()
                .await
            {
                Ok(r) if r.status() == 200 => r,
                _ => {
                    fail(&unstarted);
                    return None;
                }
            };
            let replica = response.headers()["x-inferrail-replica"]
                .to_str()
                .unwrap_or("")
                .to_owned();
            if response.bytes().await.is_err() {
                assert_eq!(replica, "r1", "only the killed replica may break streams");
                fail(&unstarted);
            }
            Some((Instant::now(), replica))
        }));
    }
    let mut served_again = None;
    for task in tasks {
        if let Some((at, replica)) = task.await.unwrap()
            && replica == "r1"
            && restarted_at.is_some_and(|r| at > r)
        {
            served_again = Some(served_again.map_or(at, |s: Instant| s.min(at)));
        }
    }
    let recovery = served_again.zip(restarted_at).map(|(s, r)| s - r);
    let ejected = killed_at.zip(ejected_at).map(|(k, e)| e - k);
    println!(
        "kill under load: {total} requests at {rate}/s, errors for requests sent after the kill {}, broken in flight requests {}, ejected {ejected:?} after the kill, back in rotation {recovery:?} after restart",
        unstarted.load(Ordering::Relaxed),
        broken.load(Ordering::Relaxed),
    );
    assert_eq!(unstarted.load(Ordering::Relaxed), 0);
    assert!(recovery.is_some_and(|r| r < Duration::from_secs(2)));
    drained(&sims, &gw).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
#[ignore = "stress"]
async fn a_flapping_replica_under_load_costs_no_client_errors() {
    let sims = sims(3, LatencyModel::ZERO).await;
    let gw = TestGateway::start(fleet(&urls(&sims.iter().collect::<Vec<_>>()), FAST_HEALTH)).await;
    let load = {
        let (client, url) = (gw.client.clone(), gw.url("/v1/chat/completions"));
        tokio::spawn(async move {
            open_loop(&client, url, 1_000, 4_000, |i| {
                if i % 2 == 0 {
                    chat("x", 2)
                } else {
                    streaming(chat("x", 2))
                }
            })
            .await
        })
    };
    let started = Instant::now();
    let mut bad = false;
    let mut flips = 0;
    while started.elapsed() < Duration::from_secs(4) {
        bad = !bad;
        flips += 1;
        sims[0].set_request_fault(if bad { Fault::Status(503) } else { Fault::None });
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    sims[0].set_request_fault(Fault::None);
    let run = load.await.unwrap();
    let [p50, p99, _] = summary(run.latencies);
    println!(
        "flapping replica: {flips} flips, 4000 requests at 1000/s, errors {}, p50 {:.3} ms, p99 {:.3} ms, hits {:?}",
        run.errors,
        ms(p50),
        ms(p99),
        sims.iter().map(Upstream::hits).collect::<Vec<_>>()
    );
    assert_eq!(run.errors, 0);
    drained(&sims, &gw).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
#[ignore = "stress"]
async fn a_disconnect_storm_leaks_nothing() {
    let sims = sims(3, decode(2.0)).await;
    let gw = TestGateway::start(fleet(&urls(&sims.iter().collect::<Vec<_>>()), "")).await;
    let mut rng = Rng(0x5eed);
    let mut tasks = Vec::new();
    let mut dropped = 0u64;
    for i in 0..1_000 {
        let drop_after = (rng.below(4) != 0).then(|| rng.below(20) as usize + 1);
        dropped += u64::from(drop_after.is_some());
        let (client, url) = (gw.client.clone(), gw.url("/v1/chat/completions"));
        let body = streaming(chat(&format!("storm {i}"), 500));
        tasks.push(tokio::spawn(async move {
            let response = client.post(url).json(&body).send().await.expect("sent");
            let mut stream = response.bytes_stream();
            let mut chunks = 0;
            while let Some(chunk) = stream.next().await {
                chunk.expect("chunk");
                chunks += 1;
                if drop_after.is_some_and(|n| chunks >= n) {
                    return;
                }
            }
        }));
    }
    for task in tasks {
        task.await.unwrap();
    }
    drained(&sims, &gw).await;
    let aborts: u64 = sims.iter().map(|s| s.sim.metrics().finished("abort")).sum();
    let completed: u64 = sims
        .iter()
        .map(|s| s.sim.metrics().finished("length"))
        .sum();
    println!(
        "disconnect storm: 1000 streams, {dropped} dropped, {aborts} aborts, {completed} completed, engines and gateway fully drained"
    );
    assert_eq!(aborts, dropped);
    assert_eq!(completed, 1_000 - dropped);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
#[ignore = "stress"]
async fn hostile_requests_never_get_a_gateway_failure() {
    let sims = sims(2, LatencyModel::ZERO).await;
    let gw = TestGateway::start(fleet(
        &urls(&sims.iter().collect::<Vec<_>>()),
        "server:\n  max_body_bytes: 65536\n",
    ))
    .await;
    let mut rng = Rng(42);
    let mut statuses = std::collections::BTreeMap::new();
    for i in 0..3_000u64 {
        let method = ["POST", "GET", "PUT", "DELETE"][rng.below(4) as usize];
        let path = [
            "/v1/chat/completions",
            "/v1/completions",
            "/v1/models",
            "/health",
            "/v1/chat/completions?x=%00",
            "/../../etc/passwd",
        ][rng.below(6) as usize];
        let body: Vec<u8> = match rng.below(7) {
            0 => (0..rng.below(2_000)).map(|_| rng.next() as u8).collect(),
            1 => b"{".repeat(rng.below(5_000) as usize),
            2 => json!({"model": MODEL, "messages": "not a list"})
                .to_string()
                .into_bytes(),
            3 => json!({"model": format!("ghost-{i}"), "messages": []})
                .to_string()
                .into_bytes(),
            4 => vec![b' '; 70_000],
            5 => json!({"model": [1, 2], "prompt": "x"})
                .to_string()
                .into_bytes(),
            _ => json!({"model": MODEL, "prompt": "fine", "max_tokens": rng.below(5) + 1})
                .to_string()
                .into_bytes(),
        };
        let mut request = gw
            .client
            .request(
                reqwest::Method::from_bytes(method.as_bytes()).unwrap(),
                gw.url(path),
            )
            .body(body);
        if rng.below(3) == 0 {
            request = request.header("x-request-id", "x".repeat(rng.below(600) as usize));
        }
        let status = request.send().await.expect("answered").status().as_u16();
        *statuses.entry(status).or_insert(0u32) += 1;
        assert!(status < 500 || status == 503, "gateway failure {status}");
    }
    println!("hostile requests: 3000 sent, statuses {statuses:?}");
    assert!(!statuses.contains_key(&503));
    let (status, _) = gw
        .post_json("/v1/chat/completions", &chat("still fine", 2))
        .await;
    assert_eq!(status, 200);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
#[ignore = "stress"]
async fn a_long_soak_keeps_memory_flat() {
    let sims = sims(3, LatencyModel::ZERO).await;
    let gw = TestGateway::start(fleet(&urls(&sims.iter().collect::<Vec<_>>()), "")).await;
    let send = |i: u64| {
        let (client, url) = (gw.client.clone(), gw.url("/v1/chat/completions"));
        async move {
            let body = if i.is_multiple_of(2) {
                chat(&format!("soak {i}"), 8)
            } else {
                streaming(chat(&format!("soak {i}"), 8))
            };
            let response = client.post(url).json(&body).send().await.expect("sent");
            assert_eq!(response.status(), 200);
            response.bytes().await.expect("body");
        }
    };
    for batch in 0..20u64 {
        futures::future::join_all((0..250).map(|i| send(batch * 250 + i))).await;
    }
    let warm = rss_kib();
    let started = Instant::now();
    for batch in 0..140u64 {
        futures::future::join_all((0..250).map(|i| send(10_000 + batch * 250 + i))).await;
    }
    let grown = rss_kib().saturating_sub(warm);
    println!(
        "soak: 35000 requests in {:.2?}, resident memory grew {} KiB after warm up",
        started.elapsed(),
        grown
    );
    assert!(grown < 32 * 1024, "grew {grown} KiB");
    drained(&sims, &gw).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
#[ignore = "stress"]
async fn slow_readers_do_not_make_the_gateway_buffer() {
    static CHUNK: [u8; 64 * 1024] = [b'a'; 64 * 1024];
    let raw = RawUpstream::spawn().await;
    raw.script(Script {
        headers: vec![("content-type", "text/event-stream".to_owned())],
        chunks: vec![Bytes::from_static(&CHUNK); 512],
        ..Script::default()
    });
    let gw = TestGateway::start(fleet(&[raw.url()], "")).await;
    let before = rss_kib();
    let mut streams = Vec::new();
    let mut total = 0usize;
    for _ in 0..20 {
        let response = gw
            .post("/v1/chat/completions", &streaming(chat("slow", 1)))
            .await;
        let mut stream = response.bytes_stream();
        total += stream.next().await.expect("first").expect("ok").len();
        streams.push(stream);
    }
    tokio::time::sleep(Duration::from_secs(2)).await;
    let held = rss_kib().saturating_sub(before);
    for mut stream in streams {
        while let Some(chunk) = stream.next().await {
            total += chunk.expect("chunk").len();
        }
    }
    println!(
        "slow readers: 20 streams of {} MiB in total, {} KiB held while clients paused",
        total / (1 << 20),
        held
    );
    assert_eq!(total, 20 * 512 * 64 * 1024);
    assert!((held as usize) < total / 1024 / 8, "held {held} KiB");
    gw.wait_idle().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
#[ignore = "stress"]
async fn bodies_in_tiny_chunks_cost_about_their_size() {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let raw = RawUpstream::spawn().await;
    let gw = TestGateway::start(fleet(&[raw.url()], "server:\n  max_body_bytes: 16777216\n")).await;
    let addr = gw.base.trim_start_matches("http://").to_owned();
    let chunks = 8 << 20;
    let block = b"1\r\n \r\n".repeat(64 * 1024);
    let before = rss_kib();
    let mut socket = tokio::net::TcpStream::connect(addr).await.unwrap();
    socket
        .write_all(b"POST /v1/completions HTTP/1.1\r\nHost: x\r\nTransfer-Encoding: chunked\r\n\r\n1\r\n{\r\n")
        .await
        .unwrap();
    for _ in 0..chunks / (64 * 1024) {
        socket.write_all(&block).await.unwrap();
    }
    tokio::time::sleep(Duration::from_millis(500)).await;
    let held = rss_kib().saturating_sub(before);
    socket.write_all(b"1\r\n}\r\n0\r\n\r\n").await.unwrap();
    let mut answer = vec![0u8; 64];
    let n = socket.read(&mut answer).await.unwrap();
    assert!(answer[..n].starts_with(b"HTTP/1.1 200"));
    let seen = raw.take();
    let inserted = format!("{{\"model\":\"{MODEL}\"");
    assert!(seen[0].body.starts_with(inserted.as_bytes()));
    assert_eq!(seen[0].body.len(), chunks + 1 + inserted.len());
    println!(
        "tiny chunks: an 8 MiB body in one byte chunks held {} KiB while arriving",
        held
    );
    assert!(held < 32 * 1024, "held {held} KiB for an 8 MiB body");
}
