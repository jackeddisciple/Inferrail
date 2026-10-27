//! Mock inference engine that speaks the vLLM HTTP API and metrics.

pub mod cache;
pub mod engine;
pub mod latency;
pub mod metrics;
pub mod pool;
pub mod server;

pub use inferrail_tokenize::sim as tokenizer;
