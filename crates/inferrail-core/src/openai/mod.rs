//! OpenAI compatible wire types that round trip without losing fields.

mod chat;
mod common;
mod completion;
mod error;
mod models;
mod nullable;
mod response;
pub mod sse;
mod usage;

pub use chat::{ChatCompletionRequest, ChatMessage, ContentPart, MessageContent};
pub use common::{
    Extra, FinishReason, FunctionCall, FunctionDefinition, Role, Stop, StreamOptions, Tool,
    ToolCall,
};
pub use completion::{CompletionRequest, Prompt, PromptRef};
pub use error::{ErrorInfo, ErrorResponse};
pub use models::{ModelCard, ModelList, ModelPermission};
pub use nullable::Nullable;
pub use response::{
    CHAT_COMPLETION_CHUNK_OBJECT, CHAT_COMPLETION_OBJECT, ChatChoice, ChatChunkChoice,
    ChatCompletionChunk, ChatCompletionResponse, CompletionChoice, CompletionChunk,
    CompletionChunkChoice, CompletionResponse, Delta, DeltaFunctionCall, DeltaToolCall,
    TEXT_COMPLETION_OBJECT,
};
pub use usage::{CompletionTokensDetails, PromptTokensDetails, Usage};
