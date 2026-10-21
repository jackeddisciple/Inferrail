//! Heavy load, abuse and soak tests. Ignored by default; run with
//! `make stress` (release build, output shown).

mod common;

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use common::{TestSim, chat, config, long_text, spawn};
use futures::StreamExt;
use inferrail_sim::latency::LatencyModel;
use serde_json::json;

fn percentile(sorted: &[Duration], p: f64) -> Duration {
    if sorted.is_empty() {
        return Duration::ZERO;
    }
    let rank = ((sorted.len() - 1) as f64 * p).round() as usize;
    sorted[rank]
}

fn report(name: &str, mut latencies: Vec<Duration>, wall: Duration) {
    latencies.sort();
    println!(
        "{name}: n={} wall={wall:.2?} rate={:.0}/s p50={:.2?} p99={:.2?} max={:.2?}",
        latencies.len(),
        latencies.len() as f64 / wall.as_secs_f64(),
        percentile(&latencies, 0.50),
        percentile(&latencies, 0.99),
        latencies.last().copied().unwrap_or_default(),
    );
}

async fn assert_drained(sim: &TestSim) {
    let started = Instant::now();
    loop {
        let snap = sim.sim.engine().snapshot();
        if snap.running == 0 && snap.waiting == 0 && snap.kv_blocks_used == 0 {
            return;
        }
        assert!(
            started.elapsed() < Duration::from_secs(5),
            "never drained: {snap:?}"
        );
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
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

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
#[ignore = "stress"]
async fn two_thousand_concurrent_streams() {
    let sim = spawn(config(LatencyModel {
        prefill_token_secs: 2e-6,
        decode_token_secs: 2e-3,
        ..LatencyModel::ZERO
    }))
    .await;
    let started = Instant::now();
    let mut tasks = Vec::new();
    for i in 0..2_000 {
        let client = sim.client.clone();
        let url = sim.url("/v1/chat/completions");
        let mut body = chat(&format!("stream number {i}"), 32);
        body["stream"] = json!(true);
        body["stream_options"] = json!({"include_usage": true});
        tasks.push(tokio::spawn(async move {
            let begun = Instant::now();
            let raw = client.post(url).json(&body).send().await?.bytes().await?;
            let text = String::from_utf8_lossy(&raw).into_owned();
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
    report("2000 concurrent streams", latencies, started.elapsed());
    assert_drained(&sim).await;
    assert_eq!(sim.sim.metrics().finished("length"), 2_000);
    assert_eq!(sim.sim.metrics().finished("abort"), 0);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
#[ignore = "stress"]
async fn open_loop_ten_thousand_requests() {
    let sim = spawn(config(LatencyModel::ZERO)).await;
    let rate = 2_000u64;
    let total = 10_000u64;
    let interval = Duration::from_nanos(1_000_000_000 / rate);
    let errors = Arc::new(AtomicU64::new(0));
    let started = tokio::time::Instant::now();
    let mut tasks = Vec::with_capacity(total as usize);
    let mut send_lag = Vec::with_capacity(total as usize);
    for i in 0..total {
        let scheduled = started + interval * i as u32;
        tokio::time::sleep_until(scheduled).await;
        send_lag.push(tokio::time::Instant::now() - scheduled);
        let client = sim.client.clone();
        let url = sim.url("/v1/chat/completions");
        let errors = Arc::clone(&errors);
        let body = chat(&format!("open loop {i}"), 8);
        tasks.push(tokio::spawn(async move {
            let begun = Instant::now();
            let ok = match client.post(url).json(&body).send().await {
                Ok(r) if r.status() == 200 => r.bytes().await.is_ok(),
                _ => false,
            };
            if !ok {
                errors.fetch_add(1, Ordering::Relaxed);
            }
            begun.elapsed()
        }));
    }
    let mut latencies = Vec::new();
    for task in tasks {
        latencies.push(task.await.unwrap());
    }
    let wall = started.elapsed();
    report("open loop 2000/s", latencies, wall);
    send_lag.sort();
    let lag_p99 = percentile(&send_lag, 0.99);
    println!("open loop send lag p99={lag_p99:.2?}");
    assert_eq!(errors.load(Ordering::Relaxed), 0);
    assert!(
        lag_p99 < Duration::from_millis(50),
        "generator fell behind: {lag_p99:?}"
    );
    assert!(wall < Duration::from_secs(10), "{wall:?}");
    assert_drained(&sim).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
#[ignore = "stress"]
async fn disconnect_storm_leaves_nothing_behind() {
    let sim = spawn(config(LatencyModel {
        prefill_token_secs: 20e-6,
        decode_token_secs: 5e-3,
        ..LatencyModel::ZERO
    }))
    .await;
    let mut rng = Rng(0x5eed);
    let mut tasks = Vec::new();
    for i in 0..1_000 {
        let client = sim.client.clone();
        let url = sim.url("/v1/chat/completions");
        let mut body = chat(&long_text(20 + (i % 50), "storm"), 200);
        body["stream"] = json!(true);
        let cut = Duration::from_millis(rng.below(400));
        let keep = rng.below(4) == 0;
        tasks.push(tokio::spawn(async move {
            let response = client.post(url).json(&body).send().await.expect("sent");
            let mut stream = response.bytes_stream();
            if keep {
                while let Some(chunk) = stream.next().await {
                    chunk.expect("chunk");
                }
                return false;
            }
            let _ =
                tokio::time::timeout(cut, async { while stream.next().await.is_some() {} }).await;
            true
        }));
    }
    let mut dropped = 0;
    for task in tasks {
        dropped += u64::from(task.await.unwrap());
    }
    assert_drained(&sim).await;
    let metrics = sim.sim.metrics();
    let finished = metrics.finished("abort") + metrics.finished("length");
    println!(
        "disconnect storm: dropped={dropped} aborted={} finished={finished}",
        metrics.finished("abort")
    );
    assert_eq!(finished, 1_000);
    assert!(metrics.finished("abort") > 0);
    assert!(metrics.finished("abort") <= dropped);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
#[ignore = "stress"]
async fn malformed_bodies_never_cause_server_errors() {
    let sim = spawn(config(LatencyModel::ZERO)).await;
    let mut rng = Rng(0xbad_c0de);
    let mut bodies: Vec<Vec<u8>> = vec![
        b"".to_vec(),
        b"{".to_vec(),
        b"null".to_vec(),
        b"[]".to_vec(),
        "[".repeat(100_000).into_bytes(),
        format!("{{\"messages\":{}}}", "[".repeat(5_000)).into_bytes(),
        br#"{"messages":[{"role":"user","content":"x"}],"max_tokens":1e400}"#.to_vec(),
        br#"{"messages":[{"role":"user","content":"x"}],"max_tokens":18446744073709551616}"#
            .to_vec(),
        br#"{"messages":[{"role":"user","content":"\ud800"}]}"#.to_vec(),
        br#"{"messages":[{"role":7}]}"#.to_vec(),
        br#"{"prompt":[4294967296]}"#.to_vec(),
        br#"{"prompt":{"a":1}}"#.to_vec(),
        br#"{"messages":[],"stream":"true"}"#.to_vec(),
        br#"{"messages":[],"stream_options":[]}"#.to_vec(),
        vec![0xff, 0xfe, 0x00, 0x7b],
    ];
    let valid =
        json!({"messages": [{"role": "user", "content": "ok"}], "max_tokens": 2}).to_string();
    for _ in 0..3_000 {
        let mut body = valid.clone().into_bytes();
        match rng.below(3) {
            0 => body.truncate(rng.below(body.len() as u64) as usize),
            1 => {
                let at = rng.below(body.len() as u64) as usize;
                body[at] = rng.next() as u8;
            }
            _ => body = (0..rng.below(256)).map(|_| rng.next() as u8).collect(),
        }
        bodies.push(body);
    }
    let mut server_errors = 0;
    let mut accepted = 0;
    for (i, body) in bodies.into_iter().enumerate() {
        let path = if i % 2 == 0 {
            "/v1/chat/completions"
        } else {
            "/v1/completions"
        };
        let response = sim
            .client
            .post(sim.url(path))
            .header("content-type", "application/json")
            .body(body)
            .send()
            .await
            .expect("server answered");
        let status = response.status().as_u16();
        let _ = response.bytes().await;
        if status >= 500 {
            server_errors += 1;
        }
        if status == 200 {
            accepted += 1;
        }
    }
    println!("malformed bodies: accepted={accepted} server_errors={server_errors}");
    assert_eq!(server_errors, 0);
    let health = sim.client.get(sim.url("/health")).send().await.unwrap();
    assert_eq!(health.status(), 200);
    assert_drained(&sim).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
#[ignore = "stress"]
async fn long_prompts_near_the_context_limit() {
    let sim = spawn(config(LatencyModel {
        prefill_token_secs: 1e-6,
        ..LatencyModel::ZERO
    }))
    .await;
    let base = long_text(14_000, "ctx");
    let mut tasks = Vec::new();
    for i in 0..64 {
        let client = sim.client.clone();
        let url = sim.url("/v1/chat/completions");
        let body = chat(&format!("{base} question {i}"), 4);
        tasks.push(tokio::spawn(async move {
            let reply: serde_json::Value =
                client.post(url).json(&body).send().await?.json().await?;
            Ok::<_, reqwest::Error>(reply)
        }));
    }
    let mut cached_total = 0;
    for task in tasks {
        let reply = task.await.unwrap().expect("long prompt served");
        let prompt = reply["usage"]["prompt_tokens"].as_u64().unwrap();
        assert!(prompt > 24_000 && prompt <= 32_768, "{prompt}");
        cached_total += reply["usage"]["prompt_tokens_details"]["cached_tokens"]
            .as_u64()
            .unwrap();
    }
    println!("long prompts: cached tokens across 64 requests = {cached_total}");
    assert!(cached_total > 0);
    assert_drained(&sim).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
#[ignore = "stress"]
async fn memory_stays_flat_over_a_soak() {
    let sim = spawn(config(LatencyModel::ZERO)).await;
    let run = |count: usize, offset: usize| {
        let sim_client = sim.client.clone();
        let url = sim.url("/v1/chat/completions");
        async move {
            let mut tasks = Vec::new();
            for i in 0..count {
                let client = sim_client.clone();
                let url = url.clone();
                let mut body = chat(&format!("soak {} {}", offset + i, long_text(20, "s")), 16);
                body["stream"] = json!(i % 2 == 0);
                tasks.push(tokio::spawn(async move {
                    client
                        .post(url)
                        .json(&body)
                        .send()
                        .await
                        .unwrap()
                        .bytes()
                        .await
                        .unwrap();
                }));
                if tasks.len() == 256 {
                    for task in tasks.drain(..) {
                        task.await.unwrap();
                    }
                }
            }
            for task in tasks {
                task.await.unwrap();
            }
        }
    };
    run(5_000, 0).await;
    let before = rss_kib();
    run(30_000, 5_000).await;
    let after = rss_kib();
    println!("soak: rss before={before} KiB after={after} KiB");
    assert_drained(&sim).await;
    if before > 0 {
        assert!(
            after < before + 64 * 1024,
            "rss grew from {before} to {after} KiB"
        );
    }
}
