//! Edge tokenization and the KV index under load. Ignored by default; run
//! with `make stress` (release build, output shown).

mod common;

use std::sync::Arc;
use std::time::{Duration, Instant};

use common::*;
use inferrail_core::config::Config;
use inferrail_sim::latency::LatencyModel;
use inferrail_sim::server::SimConfig;
use serde_json::{Value, json};

const WORDS: [&str; 24] = [
    "cache", "router", "prefix", "token", "replica", "latency", "block", "engine", "queue",
    "decode", "prefill", "memory", "warm", "request", "stream", "model", "batch", "first", "load",
    "hit", "miss", "fleet", "signal", "budget",
];

fn text(seed: u64, words: usize) -> String {
    (0..words as u64)
        .map(|i| WORDS[((seed.wrapping_mul(2_654_435_761) ^ i.wrapping_mul(40_503)) % 24) as usize])
        .collect::<Vec<_>>()
        .join(" ")
}

fn sim(blocks: usize, latency: LatencyModel) -> SimConfig {
    let mut config = sim_config(&[MODEL], latency);
    config.enable_prompt_tokens_details = true;
    config.engine.num_gpu_blocks = blocks;
    config.engine.max_num_seqs = 4_096;
    config.engine.max_num_batched_tokens = 8_192;
    config.max_model_len = (1 << 20).min(blocks * 16);
    config
}

fn tokenized(mut config: Config) -> Config {
    config.models[0].tokenizer = Some("builtin:sim".to_owned());
    config
}

fn shared_prompt(system: u64, question: u64, words: usize) -> Value {
    json!({
        "model": MODEL,
        "messages": [
            {"role": "system", "content": text(system, words)},
            {"role": "user", "content": text(1_000_000 + question, 16)},
        ],
        "max_tokens": 1,
    })
}

fn header(response: &reqwest::Response, name: &str) -> Option<u64> {
    response.headers().get(name)?.to_str().ok()?.parse().ok()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
#[ignore = "stress"]
async fn open_loop_overhead_with_tokenization_and_the_index() {
    let upstream = Upstream::spawn(sim(1 << 16, LatencyModel::ZERO)).await;
    let plain = TestGateway::start(fleet(&[upstream.url()], "")).await;
    let gw = TestGateway::start(tokenized(fleet(&[upstream.url()], ""))).await;
    let client = reqwest::Client::new();
    for (words, rates) in [(200usize, [2_000u64, 5_000]), (2_000, [1_000, 2_000])] {
        for rate in rates {
            let total = 10_000;
            let body = |i: u64| shared_prompt(i % 16, i, words);
            let direct = open_loop(
                &client,
                format!("{}/v1/chat/completions", upstream.url()),
                rate,
                total,
                body,
            )
            .await;
            let without = open_loop(
                &client,
                plain.url("/v1/chat/completions"),
                rate,
                total,
                body,
            )
            .await;
            let with = open_loop(&client, gw.url("/v1/chat/completions"), rate, total, body).await;
            assert_eq!(direct.errors + without.errors + with.errors, 0);
            let [d50, d99, _] = summary(direct.service);
            let [p50, p99, _] = summary(without.service);
            let [t50, t99, _] = summary(with.service);
            println!(
                "{words} word system prompts, {rate}/s x {total}: direct p50 {:.3} p99 {:.3} ms | gateway added p50 {:.3} p99 {:.3} ms | with tokenization and the index added p50 {:.3} p99 {:.3} ms",
                ms(d50),
                ms(d99),
                ms(p50.saturating_sub(d50)),
                ms(p99.saturating_sub(d99)),
                ms(t50.saturating_sub(d50)),
                ms(t99.saturating_sub(d99)),
            );
            assert!(t99.saturating_sub(d99) < Duration::from_millis(20));
        }
    }
    let page = gw.gateway.render_metrics();
    assert!(
        !page.contains("result=\"busy\"") && !page.contains("result=\"slow\""),
        "{page}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
#[ignore = "stress"]
async fn predictions_stay_accurate_under_concurrency() {
    let latency = LatencyModel {
        prefill_token_secs: 5e-6,
        decode_token_secs: 1e-3,
        ..LatencyModel::ZERO
    };
    let mut sims = Vec::new();
    for _ in 0..3 {
        sims.push(Upstream::spawn(sim(2_000, latency)).await);
    }
    let gw = Arc::new(
        TestGateway::start(tokenized(fleet(
            &urls(&sims.iter().collect::<Vec<_>>()),
            "routing:\n  default_policy: random\n  seed: 3\n",
        )))
        .await,
    );
    let total = 3_000u64;
    let interval = Duration::from_micros(1_000_000 / 400);
    let started = tokio::time::Instant::now();
    let mut tasks = Vec::new();
    for i in 0..total {
        tokio::time::sleep_until(started + interval * i as u32).await;
        let gw = Arc::clone(&gw);
        tasks.push(tokio::spawn(async move {
            let response = gw
                .post("/v1/chat/completions", &shared_prompt(i * 7 % 24, i, 300))
                .await;
            assert_eq!(response.status(), 200);
            let (prompt, cached) = (
                header(&response, "x-inferrail-prompt-tokens"),
                header(&response, "x-inferrail-cached-tokens"),
            );
            let answer: Value = response.json().await.unwrap();
            let usage = &answer["usage"];
            (
                prompt,
                usage["prompt_tokens"].as_u64().unwrap(),
                cached,
                usage["prompt_tokens_details"]["cached_tokens"]
                    .as_u64()
                    .unwrap_or(0),
            )
        }));
    }
    let (mut prompt_exact, mut exact, mut over, mut under, mut error) = (0, 0, 0, 0, 0u64);
    for task in tasks {
        let (prompt, real_prompt, cached, real_cached) = task.await.unwrap();
        prompt_exact += u64::from(prompt == Some(real_prompt));
        let cached = cached.expect("a prediction");
        match cached.cmp(&real_cached) {
            std::cmp::Ordering::Equal => exact += 1,
            std::cmp::Ordering::Greater => over += 1,
            std::cmp::Ordering::Less => under += 1,
        }
        error += cached.abs_diff(real_cached);
    }
    let share = |n: u64| n as f64 / total as f64;
    println!(
        "{total} requests at 400/s over 3 simulators: prompt tokens exact {:.3}, cached tokens exact {:.3}, over {:.3}, under {:.3}, mean error {:.1} tokens",
        share(prompt_exact),
        share(exact),
        share(over),
        share(under),
        error as f64 / total as f64
    );
    assert_eq!(prompt_exact, total);
    assert!(share(exact) >= 0.9, "{:.3}", share(exact));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
#[ignore = "stress"]
async fn long_prompts_through_a_tiny_pool_never_wait_or_fail() {
    let upstream = Upstream::spawn(sim(1 << 16, LatencyModel::ZERO)).await;
    let gw = Arc::new(
        TestGateway::start(tokenized(fleet(
            &[upstream.url()],
            "tokenizer:\n  workers: 1\n  queue: 2\n  max_wait: 5ms\n",
        )))
        .await,
    );
    let rss = rss_kib();
    let started = Instant::now();
    let mut tasks = Vec::new();
    for i in 0..400u64 {
        let gw = Arc::clone(&gw);
        tasks.push(tokio::spawn(async move {
            let body = shared_prompt(i, i, 20_000);
            let sent = Instant::now();
            let response = gw.post("/v1/chat/completions", &body).await;
            assert_eq!(response.status(), 200);
            let predicted = response.headers().contains_key("x-inferrail-prompt-tokens");
            response.bytes().await.unwrap();
            (predicted, sent.elapsed())
        }));
    }
    let mut predicted = 0;
    let mut slowest = Duration::ZERO;
    for task in tasks {
        let (yes, took) = task.await.unwrap();
        predicted += u32::from(yes);
        slowest = slowest.max(took);
    }
    let page = gw.gateway.render_metrics();
    let count = |result: &str| -> u64 {
        page.lines()
            .find(|l| {
                l.starts_with("inferrail_tokenized_requests_total")
                    && l.contains(&format!("result=\"{result}\""))
            })
            .and_then(|l| l.rsplit(' ').next()?.parse().ok())
            .unwrap_or(0)
    };
    println!(
        "400 concurrent 20,000 word prompts through one worker: {predicted} predicted, {} busy, {} slow, slowest request {slowest:.2?}, wall {:.2?}, RSS grew {} MB",
        count("busy"),
        count("slow"),
        started.elapsed(),
        rss_kib().saturating_sub(rss) / 1024
    );
    assert!(predicted > 0);
    assert!(count("busy") + count("slow") > 0);
    assert_eq!(u64::from(predicted), count("ok"));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
#[ignore = "stress"]
async fn a_long_soak_with_the_index_keeps_memory_bounded() {
    let mut sims = Vec::new();
    for _ in 0..3 {
        sims.push(Upstream::spawn(sim(4_096, LatencyModel::ZERO)).await);
    }
    let gw = TestGateway::start(tokenized(fleet(
        &urls(&sims.iter().collect::<Vec<_>>()),
        "",
    )))
    .await;
    let client = reqwest::Client::new();
    let warm = open_loop(&client, gw.url("/v1/chat/completions"), 2_000, 5_000, |i| {
        shared_prompt(i, i, 150)
    })
    .await;
    assert_eq!(warm.errors, 0);
    let rss = rss_kib();
    let run = open_loop(
        &client,
        gw.url("/v1/chat/completions"),
        2_000,
        40_000,
        |i| shared_prompt(10_000 + i, i, 150),
    )
    .await;
    assert_eq!(run.errors, 0);
    let grown = rss_kib().saturating_sub(rss) / 1024;
    let page = gw.gateway.render_metrics();
    let entries: f64 = page
        .lines()
        .find(|l| l.starts_with("inferrail_index_entries "))
        .and_then(|l| l.rsplit(' ').next()?.parse().ok())
        .unwrap();
    println!("40,000 distinct prompts: index holds {entries} blocks, RSS grew {grown} MB");
    assert!(entries <= (3 * 4_096) as f64);
    assert!(grown < 64, "{grown} MB");
}
