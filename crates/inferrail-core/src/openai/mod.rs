//! OpenAI compatible wire types that round trip without losing fields.

mod chat;
mod common;
mod nullable;

pub use chat::{ChatCompletionRequest, ChatMessage, ContentPart, MessageContent};
pub use common::{
    Extra, FinishReason, FunctionCall, FunctionDefinition, Role, Stop, StreamOptions, Tool,
    ToolCall,
};
pub use nullable::Nullable;
