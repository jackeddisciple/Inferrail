//! Chat templates, tokenization and chained block hashing.
//!
//! The router must compute the same token ids the engine computes, or no
//! block hash ever matches. For Hugging Face models that means rendering
//! chat requests exactly as vLLM v0.31.0 does with transformers 5.17 and
//! tokenizing the result with the same pipeline.

mod chat;
mod error;
pub mod hash;
mod hf;
mod json;
mod model;
mod python;
mod segments;
pub mod sim;
mod source;
mod template;

pub use error::TokenizeError;
pub use hf::Parity;
pub use model::{DEFAULT_CACHE_TOKENS, Encoded, Endpoint, ModelTokenizer, TokenizerOptions};
pub use segments::CacheStats;
pub use source::{SIM, Source, resolve};
pub use template::{Clock, ContentFormat};
