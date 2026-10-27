//! Decision cost and concurrency. Ignored by default; run with `make
//! stress` (release build, output shown).

use std::sync::Arc;
use std::time::{Duration, Instant};

use inferrail_core::config::PolicyKind;
use inferrail_core::ids::{BlockHash, ModelId, ReplicaId};
use inferrail_index::{KvIndex, Slot};
use inferrail_policy::{Candidate, RouteContext, Settings, SplitMix, build};

const BLOCKS: usize = 512;

fn chain(seed: u64, len: usize) -> Vec<BlockHash> {
    (0..len as u64)
        .map(|i| {
            let x = (seed << 32 | i).wrapping_mul(0x9e37_79b9_7f4a_7c15);
            BlockHash::new(u128::from(x) << 64 | u128::from(x.rotate_left(17) ^ i))
        })
        .collect()
}

fn percentile(sorted: &[Duration], p: f64) -> Duration {
    sorted[((sorted.len() - 1) as f64 * p).round() as usize]
}

/// A5: the full scoring decision for an 8,000 token prompt, timed one by
/// one: the index lookup for every candidate, the candidates assembled and
/// the cost model choosing.
#[test]
#[ignore = "stress"]
fn a5_full_decision_stays_under_200_microseconds_at_64_replicas() {
    let model = ModelId::new("m").expect("model");
    for replicas in [8usize, 64, 256] {
        let index = KvIndex::new();
        let prompt = chain(1, BLOCKS);
        let ids: Vec<ReplicaId> = (0..replicas)
            .map(|i| ReplicaId::new(format!("r{i}")).expect("id"))
            .collect();
        let slots: Vec<Slot> = ids.iter().map(|id| index.slot(id)).collect();
        for (i, slot) in slots.iter().enumerate() {
            index.record(*slot, &prompt[..(i * 37) % BLOCKS], 1 << 20);
            index.record(*slot, &chain(100 + i as u64, 256), 1 << 20);
        }
        let rng = SplitMix::new(5);
        let policy = build(PolicyKind::CostModel, &Settings::seeded(1));
        let mut depth = vec![0usize; replicas];
        let mut samples = Vec::with_capacity(20_000);
        for round in 0..20_000u64 {
            let started = Instant::now();
            index.longest_prefix_into(&prompt[..BLOCKS - 1], &slots, &mut depth);
            let candidates: Vec<Candidate<'_>> = ids
                .iter()
                .zip(&depth)
                .map(|(id, &blocks)| Candidate {
                    cached_tokens: (blocks * 16) as u32,
                    queued_secs: rng.below(500) as f64 / 1e3,
                    running: rng.below(32) as u32,
                    kv_usage: Some(rng.below(100) as f64 / 100.0),
                    ..Candidate::new(id, rng.below(32) as u32)
                })
                .collect();
            let ctx = RouteContext {
                prompt_tokens: (BLOCKS * 16) as u32,
                blocks: &prompt,
                block_size: 16,
                session: Some(round % 64),
                now: round * 1_000_000,
                ..RouteContext::new(&model, &candidates)
            };
            let choice = policy.choose(&ctx).expect("a replica");
            samples.push(started.elapsed());
            assert!(choice.index < replicas);
        }
        samples.sort();
        let (p50, p99, p999) = (
            percentile(&samples, 0.5),
            percentile(&samples, 0.99),
            percentile(&samples, 0.999),
        );
        println!(
            "A5 {replicas} replicas, 8,192 token prompt: p50 {:.1} us, p99 {:.1} us, p99.9 {:.1} us",
            p50.as_secs_f64() * 1e6,
            p99.as_secs_f64() * 1e6,
            p999.as_secs_f64() * 1e6,
        );
        if replicas <= 64 {
            assert!(p99 < Duration::from_micros(200), "{p99:?}");
        }
    }
}

/// Sixteen threads routing through one policy of every kind at once, with
/// sessions and hot prefixes in play: every choice is a valid index and
/// the shared sketch and session table never block or tear.
#[test]
#[ignore = "stress"]
fn every_policy_is_safe_to_share_between_threads() {
    for kind in PolicyKind::ALL {
        let policy: Arc<dyn inferrail_policy::RoutingPolicy> = Arc::from(build(
            kind,
            &Settings {
                session_affinity: true,
                ..Settings::seeded(3)
            },
        ));
        let started = Instant::now();
        let threads: Vec<_> = (0..16u64)
            .map(|t| {
                let policy = Arc::clone(&policy);
                std::thread::spawn(move || {
                    let model = ModelId::new("m").expect("model");
                    let ids: Vec<ReplicaId> = (0..32)
                        .map(|i| ReplicaId::new(format!("r{i}")).expect("id"))
                        .collect();
                    let rng = SplitMix::new(t);
                    let prompts: Vec<Vec<BlockHash>> = (0..8).map(|p| chain(p, 64)).collect();
                    let mut decisions = 0u64;
                    for round in 0..50_000u64 {
                        let n = 1 + rng.below(32);
                        let candidates: Vec<Candidate<'_>> = ids[..n]
                            .iter()
                            .map(|id| Candidate {
                                cached_tokens: rng.below(1_024) as u32,
                                queued_secs: rng.below(100) as f64 / 1e3,
                                kv_usage: Some(rng.below(100) as f64 / 100.0),
                                ..Candidate::new(id, rng.below(16) as u32)
                            })
                            .collect();
                        let ctx = RouteContext {
                            prompt_tokens: 1_030,
                            blocks: &prompts[rng.below(8)],
                            block_size: 16,
                            session: Some(rng.below(1_000) as u64),
                            now: round * 100_000,
                            ..RouteContext::new(&model, &candidates)
                        };
                        let choice = policy.choose(&ctx).expect("a replica");
                        assert!(choice.index < n);
                        if round % 1_000 == 0 {
                            assert_eq!(policy.explain(&ctx).candidates.len(), n);
                        }
                        decisions += 1;
                    }
                    decisions
                })
            })
            .collect();
        let decisions: u64 = threads.into_iter().map(|t| t.join().expect("thread")).sum();
        let took = started.elapsed();
        println!(
            "{kind}: {decisions} decisions from 16 threads in {:.2} s, {:.0} ns each",
            took.as_secs_f64(),
            took.as_nanos() as f64 / decisions as f64 * 16.0,
        );
    }
}
