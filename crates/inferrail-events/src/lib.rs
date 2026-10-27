//! Engine KV events: vLLM's event batches in msgpack, and the ZeroMQ wire
//! protocol they travel over, spoken directly so no libzmq is needed.

pub mod publisher;
pub mod vllm;
pub mod zmtp;

pub use publisher::{Published, Publisher, PublisherConfig, Subscriber, replay};
pub use vllm::{BlockRemoved, BlockStored, DecodeError, EventBatch, KvEvent, MEDIUM_GPU};
