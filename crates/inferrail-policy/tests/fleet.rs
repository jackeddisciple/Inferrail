//! Every policy routing the same seeded traffic over a simulated fleet:
//! each replica prefills one prompt at a time at the simulator's speed,
//! holds a prefix cache of whole blocks evicted least recently used, and
//! decodes in parallel. The router sees what a gateway sees: requests in
//! flight, queued prefill, KV usage and the prefix each replica holds.

use std::collections::{BTreeMap, HashMap};

use inferrail_core::config::PolicyKind;
use inferrail_core::ids::{BlockHash, ModelId, ReplicaId};
use inferrail_policy::{Candidate, CostModel, RouteContext, Settings, SplitMix, build};

const BLOCK: u32 = 16;
const DECODE_SECS: f64 = 0.02;

/// A prompt as its blocks, chained so a shared prefix shares hashes.
#[derive(Debug, Clone)]
struct Prompt {
    tokens: u32,
    blocks: Vec<BlockHash>,
    output: u32,
}

fn chain(parts: &[(u64, u32)]) -> Vec<BlockHash> {
    let mut out = Vec::new();
    let mut h = 0u64;
    let mut position = 0u32;
    for &(seed, tokens) in parts {
        for _ in 0..tokens / BLOCK {
            h = h.rotate_left(13) ^ seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ u64::from(position);
            h = h.wrapping_mul(0xbf58_476d_1ce4_e5b9);
            out.push(BlockHash::new(u128::from(h) << 64 | u128::from(position)));
            position += 1;
        }
    }
    out
}

struct Lru {
    capacity: usize,
    tick: u64,
    used: HashMap<BlockHash, u64>,
    order: BTreeMap<u64, BlockHash>,
}

impl Lru {
    fn new(capacity: usize) -> Self {
        Self {
            capacity,
            tick: 0,
            used: HashMap::new(),
            order: BTreeMap::new(),
        }
    }

    fn held(&self, blocks: &[BlockHash]) -> usize {
        blocks
            .iter()
            .take_while(|b| self.used.contains_key(b))
            .count()
    }

    fn touch(&mut self, blocks: &[BlockHash]) {
        for block in blocks.iter().rev() {
            self.tick += 1;
            if let Some(old) = self.used.insert(*block, self.tick) {
                self.order.remove(&old);
            }
            self.order.insert(self.tick, *block);
        }
        while self.used.len() > self.capacity {
            let (_, oldest) = self.order.pop_first().expect("an entry");
            self.used.remove(&oldest);
        }
    }
}

struct Replica {
    busy_until: f64,
    running: Vec<(f64, u32)>,
    cache: Lru,
}

#[derive(Debug, Default)]
struct Outcome {
    ttft: Vec<f64>,
    prompt: u64,
    cached: u64,
    share: Vec<u64>,
}

impl Outcome {
    fn hit_rate(&self) -> f64 {
        self.cached as f64 / self.prompt as f64
    }

    fn mean_ttft(&self) -> f64 {
        self.ttft.iter().sum::<f64>() / self.ttft.len() as f64
    }

    fn p99_ttft(&self) -> f64 {
        let mut sorted = self.ttft.clone();
        sorted.sort_by(f64::total_cmp);
        sorted[(sorted.len() - 1) * 99 / 100]
    }

    fn max_share(&self) -> f64 {
        let total: u64 = self.share.iter().sum();
        *self.share.iter().max().expect("replicas") as f64 / total as f64
    }
}

/// Routes `prompts` arriving at `rate` a second with `kind` over
/// `replicas` replicas of `capacity` blocks each.
fn serve(
    kind: PolicyKind,
    prompts: &[Prompt],
    rate: f64,
    replicas: usize,
    capacity: usize,
) -> Outcome {
    serve_with(
        kind,
        Settings::seeded(11),
        prompts,
        rate,
        replicas,
        capacity,
    )
}

fn serve_with(
    kind: PolicyKind,
    settings: Settings,
    prompts: &[Prompt],
    rate: f64,
    replicas: usize,
    capacity: usize,
) -> Outcome {
    let truth = CostModel::default();
    let model = ModelId::new("m").expect("model");
    let ids: Vec<ReplicaId> = (0..replicas)
        .map(|i| ReplicaId::new(format!("r{i}")).expect("id"))
        .collect();
    let policy = build(kind, &settings);
    let mut fleet: Vec<Replica> = (0..replicas)
        .map(|_| Replica {
            busy_until: 0.0,
            running: Vec::new(),
            cache: Lru::new(capacity),
        })
        .collect();
    let arrivals = SplitMix::new(5);
    let mut now = 0.0;
    let mut outcome = Outcome {
        share: vec![0; replicas],
        ..Outcome::default()
    };
    for (n, prompt) in prompts.iter().enumerate() {
        let uniform = (arrivals.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
        now += -(1.0 - uniform).ln() / rate;
        let cacheable = ((prompt.tokens - 1) / BLOCK) as usize;
        for replica in &mut fleet {
            replica.running.retain(|&(end, _)| end > now);
        }
        let held: Vec<usize> = fleet
            .iter()
            .map(|r| {
                r.cache
                    .held(&prompt.blocks[..cacheable.min(prompt.blocks.len())])
            })
            .collect();
        let candidates: Vec<Candidate<'_>> = fleet
            .iter()
            .zip(&ids)
            .zip(&held)
            .map(|((r, id), &blocks)| {
                let used: u32 = r.running.iter().map(|&(_, b)| b).sum();
                Candidate {
                    cached_tokens: blocks as u32 * BLOCK,
                    queued_secs: (r.busy_until - now).max(0.0),
                    running: r.running.len() as u32,
                    kv_usage: Some(f64::from(used) / capacity as f64),
                    ..Candidate::new(id, r.running.len() as u32)
                }
            })
            .collect();
        let ctx = RouteContext {
            prompt_tokens: prompt.tokens,
            blocks: &prompt.blocks,
            block_size: BLOCK,
            now: (now * 1e9) as u64,
            ..RouteContext::new(&model, &candidates)
        };
        let i = policy.choose(&ctx).expect("a replica").index;
        let replica = &mut fleet[i];
        let cached = held[i] as u32 * BLOCK;
        let start = replica.busy_until.max(now);
        replica.busy_until = start + truth.prefill_secs(prompt.tokens - cached, cached);
        let end = replica.busy_until + f64::from(prompt.output) * DECODE_SECS;
        replica
            .running
            .push((end, prompt.tokens.div_ceil(BLOCK) + prompt.output / BLOCK));
        replica.cache.touch(&prompt.blocks);
        if n >= prompts.len() / 5 {
            outcome.ttft.push(replica.busy_until - now);
            outcome.prompt += u64::from(prompt.tokens);
            outcome.cached += u64::from(cached);
            outcome.share[i] += 1;
        }
    }
    outcome
}

fn between(rng: &SplitMix, low: u32, high: u32) -> u32 {
    low + rng.below((high - low + 1) as usize) as u32
}

/// W1: `systems` long system prompts shared by every user, short questions,
/// each system prompt drawn with Zipf weight `skew` (zero for uniform).
fn shared(count: usize, systems: u64, skew: f64, seed: u64) -> Vec<Prompt> {
    let rng = SplitMix::new(seed);
    let lengths: Vec<u32> = (0..systems)
        .map(|_| between(&rng, 2_000, 8_000) / BLOCK * BLOCK)
        .collect();
    let weights: Vec<f64> = (1..=systems).map(|k| 1.0 / (k as f64).powf(skew)).collect();
    let total: f64 = weights.iter().sum();
    (0..count)
        .map(|i| {
            let mut pick = (rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64 * total;
            let system = weights
                .iter()
                .position(|w| {
                    pick -= w;
                    pick < 0.0
                })
                .unwrap_or(0) as u64;
            let question = between(&rng, 50, 500);
            let tokens = lengths[system as usize] + question;
            Prompt {
                tokens,
                blocks: chain(&[
                    (system + 1, lengths[system as usize]),
                    (1_000_000 + i as u64, question),
                ]),
                output: between(&rng, 32, 128),
            }
        })
        .collect()
}

/// W6: prompts that share nothing.
fn unique(count: usize, seed: u64) -> Vec<Prompt> {
    let rng = SplitMix::new(seed);
    (0..count)
        .map(|i| {
            let tokens = between(&rng, 100, 4_000);
            Prompt {
                tokens,
                blocks: chain(&[(2_000_000 + i as u64 + seed * 100_000, tokens)]),
                output: between(&rng, 32, 128),
            }
        })
        .collect()
}

fn interleave(a: Vec<Prompt>, b: Vec<Prompt>, seed: u64) -> Vec<Prompt> {
    let rng = SplitMix::new(seed);
    let (mut a, mut b) = (a.into_iter(), b.into_iter());
    let mut out = Vec::new();
    loop {
        let next = if rng.below(2) == 0 {
            a.next().or_else(|| b.next())
        } else {
            b.next().or_else(|| a.next())
        };
        match next {
            Some(prompt) => out.push(prompt),
            None => return out,
        }
    }
}

const CACHE_AWARE: [PolicyKind; 3] = [
    PolicyKind::PrefixHash,
    PolicyKind::KvAware,
    PolicyKind::CostModel,
];

#[test]
fn cache_aware_policies_hit_eighty_percent_of_shared_prompts() {
    let prompts = shared(2_000, 32, 0.0, 1);
    let blind = serve(PolicyKind::RoundRobin, &prompts, 6.0, 3, 4_096);
    assert!(
        blind.hit_rate() < 0.5,
        "round robin {:.2}",
        blind.hit_rate()
    );
    for kind in CACHE_AWARE {
        let aware = serve(kind, &prompts, 6.0, 3, 4_096);
        assert!(aware.hit_rate() >= 0.8, "{kind} {:.2}", aware.hit_rate());
        assert!(aware.mean_ttft() < blind.mean_ttft() / 2.0, "{kind}");
    }
}

#[test]
fn the_cost_model_beats_kv_aware_on_a_mixed_workload() {
    let prompts = interleave(shared(1_500, 32, 0.0, 2), unique(1_500, 3), 4);
    let kv = serve(PolicyKind::KvAware, &prompts, 9.0, 4, 4_096);
    let cost = serve(PolicyKind::CostModel, &prompts, 9.0, 4, 4_096);
    let least = serve(PolicyKind::LeastRequest, &prompts, 9.0, 4, 4_096);
    assert!(
        cost.p99_ttft() < kv.p99_ttft() * 0.9,
        "p99 {:.3} vs {:.3}",
        cost.p99_ttft(),
        kv.p99_ttft()
    );
    assert!(
        cost.mean_ttft() < kv.mean_ttft(),
        "mean {:.3} vs {:.3}",
        cost.mean_ttft(),
        kv.mean_ttft()
    );
    assert!(cost.mean_ttft() < least.mean_ttft() / 2.0);
}

#[test]
fn zipf_traffic_makes_no_hotspot_and_keeps_its_hits() {
    let prompts = shared(3_000, 32, 1.1, 5);
    let pinned = serve(PolicyKind::PrefixHash, &prompts, 12.0, 4, 4_096);
    let cost = serve(PolicyKind::CostModel, &prompts, 12.0, 4, 4_096);
    assert!(
        pinned.max_share() > 0.35,
        "prefix_hash {:.2}",
        pinned.max_share()
    );
    assert!(
        cost.max_share() < 0.32,
        "cost_model put {:.2} of requests on one replica",
        cost.max_share()
    );
    assert!(cost.hit_rate() >= pinned.hit_rate() - 0.02);
    assert!(cost.p99_ttft() < pinned.p99_ttft());
}

#[test]
fn a_hot_system_prompt_melts_prefix_hash_but_not_the_cost_model() {
    let prompts = hot(4_000, 9, 4_000);
    let pinned = serve(PolicyKind::PrefixHash, &prompts, 16.0, 4, 4_096);
    let cost = serve(PolicyKind::CostModel, &prompts, 16.0, 4, 4_096);
    assert!(pinned.max_share() > 0.55 && pinned.p99_ttft() > 5.0);
    assert!(cost.max_share() < 0.3, "{:.2}", cost.max_share());
    assert!(cost.p99_ttft() < 0.5, "{:.3}", cost.p99_ttft());
    assert!(cost.hit_rate() >= pinned.hit_rate() - 0.01);
}

/// One system prompt every other request shares, with long questions, among
/// requests that share nothing.
fn hot(count: usize, seed: u64, system: u32) -> Vec<Prompt> {
    let rng = SplitMix::new(seed);
    (0..count)
        .map(|i| {
            if rng.below(2) == 0 {
                let question = between(&rng, 300, 1_500);
                Prompt {
                    tokens: system + question,
                    blocks: chain(&[(7, system), (3_000_000 + i as u64, question)]),
                    output: between(&rng, 32, 128),
                }
            } else {
                let tokens = between(&rng, 100, 2_000);
                Prompt {
                    tokens,
                    blocks: chain(&[(4_000_000 + i as u64, tokens)]),
                    output: between(&rng, 32, 128),
                }
            }
        })
        .collect()
}

/// Conversations that all open with one system prompt: each turn's prompt
/// is the system prompt, every earlier question and reply, and a new
/// question, and turns of `live` conversations interleave.
fn conversations(count: usize, live: usize, seed: u64) -> Vec<Prompt> {
    let rng = SplitMix::new(seed);
    let mut talks: Vec<Vec<(u64, u32)>> = Vec::new();
    let mut part = 5_000_000u64;
    let mut out = Vec::new();
    while out.len() < count {
        if talks.len() < live {
            talks.push(vec![(9, 1_024)]);
        }
        let k = rng.below(talks.len());
        let (question, reply) = (between(&rng, 3, 20) * BLOCK, between(&rng, 2, 8) * BLOCK);
        talks[k].push((part, question));
        let tokens: u32 = talks[k].iter().map(|&(_, t)| t).sum();
        out.push(Prompt {
            tokens,
            blocks: chain(&talks[k]),
            output: reply,
        });
        talks[k].push((part + 1, reply));
        part += 2;
        if tokens > 6_000 {
            talks.swap_remove(k);
        }
    }
    out
}

#[test]
fn conversations_keep_their_history_when_their_system_prompt_turns_hot() {
    let prompts = conversations(4_000, 40, 3);
    let cold = Settings {
        hot_prefix: None,
        ..Settings::seeded(11)
    };
    for rate in [20.0, 30.0] {
        let plain = serve_with(
            PolicyKind::CostModel,
            cold.clone(),
            &prompts,
            rate,
            4,
            4_096,
        );
        let hot = serve(PolicyKind::CostModel, &prompts, rate, 4, 4_096);
        assert!(
            hot.hit_rate() >= plain.hit_rate() - 0.01,
            "at {rate}/s hit rate {:.3} vs {:.3} without replication",
            hot.hit_rate(),
            plain.hit_rate()
        );
        assert!(
            hot.p99_ttft() <= plain.p99_ttft() * 1.1,
            "at {rate}/s p99 {:.3} vs {:.3} without replication",
            hot.p99_ttft(),
            plain.p99_ttft()
        );
    }
}
