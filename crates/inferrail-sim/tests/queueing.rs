//! Queueing on a paused clock: in a burst, more work ahead of the newest
//! request never gets it its first token sooner, and under Poisson arrivals
//! TTFT stays flat at low load and climbs steeply once they pass what the
//! engine can serve, the knee every policy comparison is read against.
//!
//! One by one, only the newest request of a burst is held to this. An older
//! one shares its last prefill step with chunks of later prompts, and one
//! arriving mid step waits for that step to end, so work ahead of either
//! can move its first token earlier by part of a step, as it can on vLLM.

use std::time::Duration;

use inferrail_sim::engine::{Engine, EngineConfig};
use inferrail_sim::latency::LatencyModel;
use proptest::prelude::*;
use tokio::time::Instant;

#[derive(Debug, Clone)]
struct Job {
    at: Duration,
    prompt: Vec<u32>,
    output: usize,
}

fn unique(seed: u64, len: usize) -> Vec<u32> {
    let mut x = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    (0..len)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x % 120_000) as u32
        })
        .collect()
}

fn engine() -> EngineConfig {
    EngineConfig {
        num_gpu_blocks: 16_384,
        latency: LatencyModel::GPU_24GB_8B,
        ..EngineConfig::default()
    }
}

/// Every job's TTFT, in the order given; jobs must be sorted by `at`.
async fn ttfts(config: EngineConfig, jobs: Vec<Job>) -> Vec<Duration> {
    let engine = Engine::start(config);
    let start = Instant::now();
    let mut runs = Vec::with_capacity(jobs.len());
    for job in jobs {
        tokio::time::sleep_until(start + job.at).await;
        let output = (0..job.output as u32).map(|t| 100_000 + t).collect();
        let ticket = engine.submit(&job.prompt, output);
        runs.push(tokio::spawn(async move {
            let mut seq = ticket.admitted().await.expect("admitted");
            let first = seq.next_token().await.expect("first token");
            while seq.next_token().await.is_some() {}
            first - seq.arrival
        }));
    }
    let mut out = Vec::with_capacity(runs.len());
    for run in runs {
        out.push(run.await.expect("run"));
    }
    out
}

fn block_on<F: std::future::Future>(future: F) -> F::Output {
    tokio::runtime::Builder::new_current_thread()
        .enable_time()
        .start_paused(true)
        .build()
        .expect("runtime")
        .block_on(future)
}

fn job() -> impl Strategy<Value = (usize, usize)> {
    (16usize..3_000, 1usize..40)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    #[test]
    fn more_work_ahead_never_brings_the_newest_first_token_sooner(
        jobs in prop::collection::vec(job(), 1..20),
        extra in job(),
        at in any::<prop::sample::Index>(),
    ) {
        let mut jobs: Vec<Job> = jobs
            .into_iter()
            .enumerate()
            .map(|(i, (len, output))| Job {
                at: Duration::ZERO,
                prompt: unique(i as u64, len),
                output,
            })
            .collect();
        let probe = jobs.len() - 1;
        let alone = block_on(ttfts(engine(), jobs.clone()))[probe];
        let (len, output) = extra;
        jobs.insert(at.index(probe + 1), Job { at: Duration::ZERO, prompt: unique(9_999, len), output });
        let crowded = block_on(ttfts(engine(), jobs))[probe + 1];
        prop_assert!(crowded >= alone, "{crowded:?} < {alone:?}");
    }
}

/// Poisson arrivals at `rate` per second of prompts between 500 and 2,500
/// tokens with 64 output tokens; the same requests at every rate.
fn poisson(rate: f64, count: usize) -> Vec<Job> {
    let mut x: u64 = 0x2545_f491_4f6c_dd1d;
    let mut uniform = move || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        (x >> 11) as f64 / (1u64 << 53) as f64
    };
    let mut at = 0.0;
    (0..count)
        .map(|i| {
            at += -(1.0 - uniform()).ln() / rate;
            let len = 500 + (uniform() * 2_000.0) as usize;
            Job {
                at: Duration::from_secs_f64(at),
                prompt: unique(i as u64, len),
                output: 64,
            }
        })
        .collect()
}

fn median(mut values: Vec<Duration>) -> Duration {
    values.sort();
    values[values.len() / 2]
}

#[test]
fn ttft_has_a_knee_at_the_engines_capacity() {
    let rates = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let medians: Vec<Duration> = rates
        .iter()
        .map(|&rate| median(block_on(ttfts(engine(), poisson(rate, 600)))))
        .collect();
    for pair in medians.windows(2) {
        assert!(pair[1] >= pair[0], "TTFT fell as load rose: {medians:?}");
    }
    let (low, mid, high) = (medians[0], medians[2], medians[5]);
    assert!(
        mid < low * 2,
        "TTFT already climbing at half load: {medians:?}"
    );
    assert!(high > mid * 10, "no knee past capacity: {medians:?}");
}

#[test]
fn decode_steps_stretch_with_the_batch_and_with_prefill() {
    let config = engine();
    let latency = config.latency;
    let alone = Job {
        at: Duration::ZERO,
        prompt: unique(1, 100),
        output: 2,
    };
    let crowd: Vec<Job> = (0..64)
        .map(|i| Job {
            at: Duration::ZERO,
            prompt: unique(10 + i, 100),
            output: 200,
        })
        .collect();
    let mut with_crowd = crowd.clone();
    with_crowd.push(Job {
        at: Duration::from_millis(500),
        ..alone.clone()
    });
    let single = block_on(ttfts(config.clone(), vec![alone]))[0];
    assert_eq!(single, latency.prefill(100, 0));
    let crowded = *block_on(ttfts(config, with_crowd)).last().expect("probe");
    let decode = Duration::from_secs_f64(latency.decode_secs(64));
    assert!(
        crowded >= latency.prefill(100, 0) + decode,
        "a prefill next to 64 decodes took {crowded:?}"
    );
}

#[test]
fn a_dry_pool_preempts_but_every_request_finishes() {
    let config = EngineConfig {
        num_gpu_blocks: 200,
        ..engine()
    };
    let jobs: Vec<Job> = (0..150u64)
        .map(|i| Job {
            at: Duration::from_millis(i * 7),
            prompt: unique(i % 9, 200 + (i as usize % 7) * 150),
            output: 50 + (i as usize % 5) * 60,
        })
        .collect();
    let result = block_on(async {
        let engine = Engine::start(config);
        let start = Instant::now();
        let mut runs = Vec::new();
        for job in jobs {
            tokio::time::sleep_until(start + job.at).await;
            let output: Vec<u32> = (0..job.output as u32).collect();
            let ticket = engine.submit(&job.prompt, output);
            runs.push(tokio::spawn(async move {
                let mut seq = ticket.admitted().await.expect("admitted");
                let mut last = None;
                while let Some(at) = seq.next_token().await {
                    assert!(last.is_none_or(|l| at >= l), "tokens out of order");
                    last = Some(at);
                }
                (seq.generated(), job.output, seq.preemptions())
            }));
        }
        let mut preempted = 0;
        for run in runs {
            let (generated, wanted, preemptions) = run.await.expect("run");
            assert_eq!(generated, wanted);
            preempted += u64::from(preemptions);
        }
        (preempted, engine.snapshot())
    });
    let (preempted, snap) = result;
    assert!(preempted > 0, "the pool never ran dry");
    assert_eq!(snap.preemptions, preempted);
    assert_eq!((snap.running, snap.waiting, snap.kv_blocks_used), (0, 0, 0));
    assert!(snap.cached_blocks <= 200);
}
