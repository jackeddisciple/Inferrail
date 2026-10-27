//! KV events as a router consumes them: the stream describes the engine's
//! cache exactly, and every stored block's hash can be rebuilt from the
//! token ids it carries.

mod common;

use std::collections::HashMap;
use std::time::Duration;

use common::{TestSim, config, fast_latency, spawn};
use inferrail_events::{EventBatch, KvEvent, Published, Subscriber, replay};
use inferrail_sim::cache::extend_hashes;
use inferrail_sim::events::KvEventsConfig;
use serde_json::json;

async fn publishing(blocks: usize) -> (TestSim, Subscriber) {
    let mut cfg = config(fast_latency());
    cfg.engine.num_gpu_blocks = blocks;
    cfg.max_model_len = blocks * 16;
    cfg.kv_events = KvEventsConfig {
        enable_kv_cache_events: true,
        endpoint: "tcp://127.0.0.1:0".to_owned(),
        replay_endpoint: Some("tcp://127.0.0.1:0".to_owned()),
        topic: "kv-events".to_owned(),
        ..KvEventsConfig::default()
    };
    let sim = spawn(cfg).await;
    let addr = sim.sim.kv_events_addr().expect("publishing");
    let sub = Subscriber::connect(addr, b"kv").await.expect("subscribed");
    tokio::time::sleep(Duration::from_millis(50)).await;
    (sim, sub)
}

async fn complete(sim: &TestSim, prompt: &[u32], max_tokens: u64) {
    let body = json!({"model": "llama", "prompt": prompt, "max_tokens": max_tokens});
    let (status, reply) = sim.post_json("/v1/completions", &body).await;
    assert_eq!(status, 200, "{reply}");
}

/// Everything published until the publisher has been quiet for a while.
async fn drain(sub: &mut Subscriber) -> Vec<Published> {
    let mut out = Vec::new();
    while let Ok(got) = tokio::time::timeout(Duration::from_millis(300), sub.recv()).await {
        out.push(got.expect("a message"));
    }
    out
}

fn tokens(seed: u32, len: usize) -> Vec<u32> {
    (0..len as u32)
        .map(|i| (i.wrapping_mul(2_654_435_761) ^ seed) % 120_000)
        .collect()
}

/// Applies events to a multiset of hashes, checking each stored block
/// against its tokens and its parent.
#[derive(Default)]
struct Mirror {
    held: HashMap<u64, usize>,
    ever: HashMap<u64, ()>,
}

impl Mirror {
    fn apply(&mut self, event: &KvEvent) {
        match event {
            KvEvent::BlockStored(e) => {
                assert_eq!(e.token_ids.len(), e.block_hashes.len() * 16);
                assert_eq!(e.block_size, 16);
                assert_eq!(e.medium.as_deref(), Some("GPU"));
                let mut chain: Vec<u64> = e.parent_block_hash.into_iter().collect();
                if let Some(parent) = e.parent_block_hash {
                    assert!(self.ever.contains_key(&parent), "unknown parent {parent:x}");
                }
                let mut prefix = vec![0u32; chain.len() * 16];
                prefix.extend(&e.token_ids);
                extend_hashes(&prefix, 16, &mut chain);
                assert_eq!(&chain[chain.len() - e.block_hashes.len()..], e.block_hashes);
                for &hash in &e.block_hashes {
                    *self.held.entry(hash).or_default() += 1;
                    self.ever.insert(hash, ());
                }
            }
            KvEvent::BlockRemoved(e) => {
                for hash in &e.block_hashes {
                    let count = self
                        .held
                        .get_mut(hash)
                        .expect("removed a block never stored");
                    *count -= 1;
                    if *count == 0 {
                        self.held.remove(hash);
                    }
                }
            }
            KvEvent::AllBlocksCleared => self.held.clear(),
        }
    }

    fn blocks(&self) -> usize {
        self.held.values().sum()
    }
}

fn check_sequence(messages: &[Published], first: u64) {
    for (i, message) in messages.iter().enumerate() {
        assert_eq!(message.seq, first + i as u64, "gap in the sequence");
        assert_eq!(message.topic, b"kv-events");
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn events_describe_the_cache_exactly() {
    let (sim, mut sub) = publishing(96).await;
    let shared = tokens(1, 400);
    for round in 0..30u32 {
        let mut prompt = if round % 3 == 0 {
            tokens(1_000 + round, 300)
        } else {
            shared.clone()
        };
        prompt.extend(tokens(2_000 + round, 37));
        complete(&sim, &prompt, 20 + u64::from(round % 4) * 30).await;
    }
    let messages = drain(&mut sub).await;
    check_sequence(&messages, 0);
    let mut mirror = Mirror::default();
    let (mut stored, mut removed) = (0, 0);
    for message in &messages {
        let batch = EventBatch::decode(&message.payload).expect("a vLLM batch");
        assert_eq!(batch.data_parallel_rank, Some(0));
        assert!(batch.ts > 1.7e9);
        for event in &batch.events {
            match event {
                KvEvent::BlockStored(_) => stored += 1,
                KvEvent::BlockRemoved(_) => removed += 1,
                KvEvent::AllBlocksCleared => {}
            }
            mirror.apply(event);
        }
    }
    assert!(
        stored > 30 && removed > 0,
        "{stored} stored, {removed} removed"
    );
    assert_eq!(mirror.blocks(), sim.sim.engine().snapshot().cached_blocks);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_traffic_keeps_the_stream_exact() {
    let (sim, mut sub) = publishing(160).await;
    let mut tasks = Vec::new();
    for i in 0..60u32 {
        let client = sim.client.clone();
        let url = sim.url("/v1/completions");
        let mut prompt = tokens(i % 4, 200 + (i as usize % 5) * 40);
        prompt.extend(tokens(500 + i, 9));
        tasks.push(tokio::spawn(async move {
            let body = json!({"model": "llama", "prompt": prompt, "max_tokens": 40});
            let reply = client.post(url).json(&body).send().await.expect("sent");
            assert_eq!(reply.status(), 200);
            reply.bytes().await.expect("body");
        }));
    }
    for task in tasks {
        task.await.expect("request");
    }
    let messages = drain(&mut sub).await;
    check_sequence(&messages, 0);
    let mut mirror = Mirror::default();
    for message in &messages {
        for event in EventBatch::decode(&message.payload).unwrap().events {
            mirror.apply(&event);
        }
    }
    let snap = sim.sim.engine().snapshot();
    assert_eq!(mirror.blocks(), snap.cached_blocks);
    assert_eq!((snap.running, snap.kv_blocks_used), (0, 0));
}

#[tokio::test]
async fn prompts_are_stored_with_their_own_token_ids() {
    let (sim, mut sub) = publishing(256).await;
    let prompt = tokens(7, 70);
    complete(&sim, &prompt, 1).await;
    let messages = drain(&mut sub).await;
    let events = EventBatch::decode(&messages[0].payload).unwrap().events;
    let KvEvent::BlockStored(first) = &events[0] else {
        panic!("{events:?}")
    };
    assert_eq!(first.parent_block_hash, None);
    assert_eq!(first.token_ids, prompt[..64]);
}

#[tokio::test]
async fn resets_clear_everything_and_replay_returns_the_history() {
    let (sim, mut sub) = publishing(256).await;
    complete(&sim, &tokens(3, 200), 5).await;
    let reset = sim
        .client
        .post(sim.url("/reset_prefix_cache"))
        .send()
        .await
        .unwrap();
    assert_eq!(reset.status(), 200);
    complete(&sim, &tokens(4, 100), 5).await;
    let live = drain(&mut sub).await;
    let cleared = live
        .iter()
        .flat_map(|m| EventBatch::decode(&m.payload).unwrap().events)
        .filter(|e| *e == KvEvent::AllBlocksCleared)
        .count();
    assert_eq!(cleared, 1);
    let addr = sim.sim.kv_replay_addr().unwrap();
    assert_eq!(replay(addr, 0).await.unwrap(), live);
    assert_eq!(replay(addr, 1).await.unwrap(), live[1..]);
}

#[tokio::test]
async fn without_prefix_caching_nothing_is_published() {
    let mut cfg = config(fast_latency());
    cfg.engine.enable_prefix_caching = false;
    cfg.kv_events = KvEventsConfig {
        enable_kv_cache_events: true,
        endpoint: "tcp://127.0.0.1:0".to_owned(),
        ..KvEventsConfig::default()
    };
    let sim = spawn(cfg).await;
    let mut sub = Subscriber::connect(sim.sim.kv_events_addr().unwrap(), b"")
        .await
        .unwrap();
    tokio::time::sleep(Duration::from_millis(50)).await;
    complete(&sim, &tokens(1, 300), 30).await;
    assert!(drain(&mut sub).await.is_empty());
}
