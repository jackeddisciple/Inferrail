//! Randomized schedules on a paused clock: arrivals, prompt sharing,
//! preemption and cancellations must never break the engine's invariants.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use inferrail_sim::engine::{Engine, EngineConfig};
use inferrail_sim::latency::LatencyModel;
use proptest::prelude::*;
use tokio::time::Instant;

#[derive(Debug, Clone)]
struct Arrival {
    gap_ms: u64,
    prefix: usize,
    suffix: usize,
    tokens: usize,
    cancel_after_ms: Option<u64>,
}

fn arrival() -> impl Strategy<Value = Arrival> {
    (
        0u64..40,
        0usize..3,
        0usize..200,
        0usize..12,
        prop::option::weighted(0.25, 0u64..300),
    )
        .prop_map(
            |(gap_ms, prefix, suffix, tokens, cancel_after_ms)| Arrival {
                gap_ms,
                prefix,
                suffix,
                tokens,
                cancel_after_ms,
            },
        )
}

#[derive(Debug, Clone, Copy)]
struct Done {
    arrival: Instant,
    scheduled: Instant,
    first_token: Instant,
    prompt: usize,
    cached: usize,
}

fn prompt(prefix: usize, suffix: usize, salt: u64) -> Vec<u32> {
    let mut tokens: Vec<u32> = (0..160).map(|i| (prefix as u32) * 1_000 + i).collect();
    tokens.extend((0..suffix as u32).map(|i| (salt as u32).wrapping_mul(7_919).wrapping_add(i)));
    tokens
}

async fn run_schedule(arrivals: Vec<Arrival>, cap: usize) -> Result<(), TestCaseError> {
    let config = EngineConfig {
        block_size: 16,
        num_gpu_blocks: 64,
        max_num_seqs: cap,
        max_num_batched_tokens: 128,
        enable_prefix_caching: true,
        latency: LatencyModel {
            prefill_token_secs: 100e-6,
            prefill_pair_secs: 1e-9,
            prefill_fixed_secs: 1e-3,
            decode_token_secs: 5e-3,
            decode_seq_secs: 0.0,
        },
    };
    let engine = Engine::start(config);
    let peak = Arc::new(AtomicU64::new(0));
    let watcher = {
        let engine = engine.clone();
        let peak = Arc::clone(&peak);
        tokio::spawn(async move {
            loop {
                peak.fetch_max(engine.snapshot().running, Ordering::Relaxed);
                tokio::time::sleep(Duration::from_micros(500)).await;
            }
        })
    };

    let mut tasks = Vec::new();
    for (i, a) in arrivals.iter().cloned().enumerate() {
        tokio::time::sleep(Duration::from_millis(a.gap_ms)).await;
        let engine = engine.clone();
        tasks.push(tokio::spawn(async move {
            let tokens = prompt(a.prefix, a.suffix, i as u64);
            let output = (0..=a.tokens as u32).map(|t| 90_000 + t).collect();
            let work = async {
                let mut seq = engine
                    .submit(&tokens, output)
                    .admitted()
                    .await
                    .expect("engine alive");
                let first_token = seq.next_token().await.expect("first token");
                let mut generated = 1;
                while seq.next_token().await.is_some() {
                    generated += 1;
                }
                assert_eq!(generated, a.tokens + 1);
                Done {
                    arrival: seq.arrival,
                    scheduled: seq.scheduled,
                    first_token,
                    prompt: seq.prompt_tokens,
                    cached: seq.cached_tokens,
                }
            };
            match a.cancel_after_ms {
                Some(ms) => tokio::time::timeout(Duration::from_millis(ms), work)
                    .await
                    .ok(),
                None => Some(work.await),
            }
        }));
    }
    let mut finished = Vec::new();
    for task in tasks {
        if let Some(done) = task.await.expect("task") {
            finished.push(done);
        }
    }
    let drained = Instant::now();
    while engine.snapshot().running + engine.snapshot().waiting > 0 {
        prop_assert!(
            drained.elapsed() < Duration::from_secs(10),
            "aborts never freed"
        );
        tokio::time::sleep(Duration::from_millis(1)).await;
    }
    watcher.abort();

    let snap = engine.snapshot();
    prop_assert_eq!((snap.running, snap.waiting, snap.kv_blocks_used), (0, 0, 0));
    prop_assert!(
        peak.load(Ordering::Relaxed) <= cap as u64,
        "running exceeded the cap"
    );
    prop_assert!(snap.cached_blocks <= 64);
    prop_assert!(snap.prefix_hits <= snap.prefix_queries);

    finished.sort_by_key(|d| d.arrival);
    for pair in finished.windows(2) {
        prop_assert!(
            pair[0].scheduled <= pair[1].scheduled,
            "admission is not FIFO"
        );
    }
    for d in &finished {
        prop_assert!(d.arrival <= d.scheduled && d.scheduled <= d.first_token);
        prop_assert_eq!(d.cached % 16, 0);
        prop_assert!(d.cached <= (d.prompt - 1) / 16 * 16);
    }
    Ok(())
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(300))]

    #[test]
    fn random_schedules_keep_engine_invariants(
        arrivals in proptest::collection::vec(arrival(), 1..40),
        cap in 1usize..6,
    ) {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_time()
            .start_paused(true)
            .build()
            .expect("runtime");
        rt.block_on(run_schedule(arrivals, cap))?;
    }
}
