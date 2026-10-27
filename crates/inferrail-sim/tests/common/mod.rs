#![allow(dead_code, unreachable_pub)]

use std::time::{Duration, Instant};

use futures::StreamExt;
use inferrail_core::openai::sse::{SseDecoder, SseEvent};
use inferrail_sim::engine::EngineConfig;
use inferrail_sim::latency::LatencyModel;
use inferrail_sim::server::{Sim, SimConfig};
use serde_json::Value;
use tokio::net::TcpListener;

pub struct TestSim {
    pub base: String,
    pub sim: Sim,
    pub client: reqwest::Client,
}

pub fn fast_latency() -> LatencyModel {
    LatencyModel {
        prefill_token_secs: 200e-6,
        prefill_pair_secs: 0.0,
        prefill_fixed_secs: 1e-3,
        decode_token_secs: 2e-3,
        decode_seq_secs: 0.0,
    }
}

pub fn config(latency: LatencyModel) -> SimConfig {
    SimConfig {
        model: "sim/llama-8b".to_owned(),
        served_model_names: vec!["sim/llama-8b".to_owned(), "llama".to_owned()],
        enable_prompt_tokens_details: true,
        engine: EngineConfig {
            latency,
            ..EngineConfig::default()
        },
        ..SimConfig::default()
    }
}

pub async fn spawn(config: SimConfig) -> TestSim {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr = listener.local_addr().expect("local addr");
    let sim = Sim::new(config).expect("valid simulator config");
    tokio::spawn(sim.clone().serve(listener, std::future::pending()));
    TestSim {
        base: format!("http://{addr}"),
        sim,
        client: reqwest::Client::new(),
    }
}

impl TestSim {
    pub fn url(&self, path: &str) -> String {
        format!("{}{path}", self.base)
    }

    pub async fn post(&self, path: &str, body: &Value) -> reqwest::Response {
        self.client
            .post(self.url(path))
            .json(body)
            .send()
            .await
            .expect("request sent")
    }

    pub async fn post_json(&self, path: &str, body: &Value) -> (u16, Value) {
        let response = self.post(path, body).await;
        let status = response.status().as_u16();
        (status, response.json().await.expect("json body"))
    }

    /// Streams a request and returns the events with the time to first byte.
    pub async fn stream(&self, path: &str, body: &Value) -> (Vec<SseEvent>, Duration, Vec<u8>) {
        let started = Instant::now();
        let response = self.post(path, body).await;
        assert_eq!(response.status(), 200);
        let mut bytes = response.bytes_stream();
        let mut decoder = SseDecoder::default();
        let mut events = Vec::new();
        let mut raw = Vec::new();
        let mut first = None;
        while let Some(chunk) = bytes.next().await {
            let chunk = chunk.expect("stream chunk");
            first.get_or_insert_with(|| started.elapsed());
            raw.extend_from_slice(&chunk);
            decoder.push(&chunk, &mut events).expect("valid sse");
        }
        assert!(decoder.is_idle(), "stream ended mid event");
        (events, first.unwrap_or_default(), raw)
    }
}

pub fn chat(content: &str, max_tokens: u64) -> Value {
    serde_json::json!({
        "model": "sim/llama-8b",
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens
    })
}

pub fn long_text(words: usize, salt: &str) -> String {
    (0..words).map(|i| format!("{salt}{i} ")).collect()
}
