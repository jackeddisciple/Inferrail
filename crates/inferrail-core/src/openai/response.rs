use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::{ChatMessage, Extra, FinishReason, Nullable, Role, Usage};

pub const CHAT_COMPLETION_OBJECT: &str = "chat.completion";
pub const CHAT_COMPLETION_CHUNK_OBJECT: &str = "chat.completion.chunk";
pub const TEXT_COMPLETION_OBJECT: &str = "text_completion";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    #[serde(default, skip_serializing_if = "Nullable::is_absent")]
    pub service_tier: Nullable<String>,
    #[serde(default, skip_serializing_if = "Nullable::is_absent")]
    pub system_fingerprint: Nullable<String>,
    #[serde(default, skip_serializing_if = "Nullable::is_absent")]
    pub usage: Nullable<Usage>,
    #[serde(default, skip_serializing_if = "Nullable::is_absent")]
    pub prompt_logprobs: Nullable<Value>,
    #[serde(default, skip_serializing_if = "Nullable::is_absent")]
    pub prompt_token_ids: Nullable<Vec<u32>>,
    #[serde(default, skip_serializing_if = "Nullable::is_absent")]
    pub prompt_text: Nullable<String>,
    #[serde(default, skip_serializing_if = "Nullable::is_absent")]
    pub kv_transfer_params: Nullable<Value>,
    #[serde(default, skip_serializing_if = "Nullable::is_absent")]
    pub ec_transfer_params: Nullable<Value>,
    #[serde(default, skip_serializing_if = "Nullable::is_absent")]
    pub metrics: Nullable<Value>,
    #[serde(flatten)]
    pub extra: Extra,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChatChoice {
    pub index: u32,
    pub message: ChatMessage,
    #[serde(default, skip_serializing_if = "Nullable::is_absent")]
    pub logprobs: Nullable<Value>,
    #[serde(default, skip_serializing_if = "Nullable::is_absent")]
    pub finish_reason: Nullable<FinishReason>,
    #[serde(default, skip_serializing_if = "Nullable::is_absent")]
    pub stop_reason: Nullable<Value>,
    #[serde(default, skip_serializing_if = "Nullable::is_absent")]
    pub token_ids: Nullable<Vec<u32>>,
    #[serde(default, skip_serializing_if = "Nullable::is_absent")]
    pub routed_experts: Nullable<String>,
    #[serde(flatten)]
    pub extra: Extra,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatChunkChoice>,
    #[serde(default, skip_serializing_if = "Nullable::is_absent")]
    pub usage: Nullable<Usage>,
    #[serde(default, skip_serializing_if = "Nullable::is_absent")]
    pub system_fingerprint: Nullable<String>,
    #[serde(default, skip_serializing_if = "Nullable::is_absent")]
    pub prompt_token_ids: Nullable<Vec<u32>>,
    #[serde(default, skip_serializing_if = "Nullable::is_absent")]
    pub prompt_text: Nullable<String>,
    #[serde(default, skip_serializing_if = "Nullable::is_absent")]
    pub service_tier: Nullable<String>,
    #[serde(default, skip_serializing_if = "Nullable::is_absent")]
    pub metrics: Nullable<Value>,
    #[serde(flatten)]
    pub extra: Extra,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChatChunkChoice {
    pub index: u32,
    pub delta: Delta,
    #[serde(default, skip_serializing_if = "Nullable::is_absent")]
    pub logprobs: Nullable<Value>,
    #[serde(default, skip_serializing_if = "Nullable::is_absent")]
    pub finish_reason: Nullable<FinishReason>,
    #[serde(default, skip_serializing_if = "Nullable::is_absent")]
    pub stop_reason: Nullable<Value>,
    #[serde(default, skip_serializing_if = "Nullable::is_absent")]
    pub token_ids: Nullable<Vec<u32>>,
    #[serde(flatten)]
    pub extra: Extra,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Delta {
    #[serde(default, skip_serializing_if = "Nullable::is_absent")]
    pub role: Nullable<Role>,
    #[serde(default, skip_serializing_if = "Nullable::is_absent")]
    pub content: Nullable<String>,
    #[serde(default, skip_serializing_if = "Nullable::is_absent")]
    pub reasoning: Nullable<String>,
    #[serde(default, skip_serializing_if = "Nullable::is_absent")]
    pub refusal: Nullable<String>,
    #[serde(default, skip_serializing_if = "Nullable::is_absent")]
    pub tool_calls: Nullable<Vec<DeltaToolCall>>,
    #[serde(flatten)]
    pub extra: Extra,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DeltaToolCall {
    pub index: u32,
    #[serde(default, skip_serializing_if = "Nullable::is_absent")]
    pub id: Nullable<String>,
    #[serde(rename = "type", default, skip_serializing_if = "Nullable::is_absent")]
    pub kind: Nullable<String>,
    #[serde(default, skip_serializing_if = "Nullable::is_absent")]
    pub function: Nullable<DeltaFunctionCall>,
    #[serde(flatten)]
    pub extra: Extra,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct DeltaFunctionCall {
    #[serde(default, skip_serializing_if = "Nullable::is_absent")]
    pub name: Nullable<String>,
    #[serde(default, skip_serializing_if = "Nullable::is_absent")]
    pub arguments: Nullable<String>,
    #[serde(flatten)]
    pub extra: Extra,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<CompletionChoice>,
    #[serde(default, skip_serializing_if = "Nullable::is_absent")]
    pub service_tier: Nullable<String>,
    #[serde(default, skip_serializing_if = "Nullable::is_absent")]
    pub system_fingerprint: Nullable<String>,
    #[serde(default, skip_serializing_if = "Nullable::is_absent")]
    pub usage: Nullable<Usage>,
    #[serde(default, skip_serializing_if = "Nullable::is_absent")]
    pub kv_transfer_params: Nullable<Value>,
    #[serde(default, skip_serializing_if = "Nullable::is_absent")]
    pub ec_transfer_params: Nullable<Value>,
    #[serde(default, skip_serializing_if = "Nullable::is_absent")]
    pub metrics: Nullable<Value>,
    #[serde(flatten)]
    pub extra: Extra,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompletionChoice {
    pub index: u32,
    pub text: String,
    #[serde(default, skip_serializing_if = "Nullable::is_absent")]
    pub logprobs: Nullable<Value>,
    #[serde(default, skip_serializing_if = "Nullable::is_absent")]
    pub finish_reason: Nullable<FinishReason>,
    #[serde(default, skip_serializing_if = "Nullable::is_absent")]
    pub stop_reason: Nullable<Value>,
    #[serde(default, skip_serializing_if = "Nullable::is_absent")]
    pub token_ids: Nullable<Vec<u32>>,
    #[serde(default, skip_serializing_if = "Nullable::is_absent")]
    pub prompt_logprobs: Nullable<Value>,
    #[serde(default, skip_serializing_if = "Nullable::is_absent")]
    pub prompt_token_ids: Nullable<Vec<u32>>,
    #[serde(default, skip_serializing_if = "Nullable::is_absent")]
    pub routed_experts: Nullable<String>,
    #[serde(flatten)]
    pub extra: Extra,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompletionChunk {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<CompletionChunkChoice>,
    #[serde(default, skip_serializing_if = "Nullable::is_absent")]
    pub usage: Nullable<Usage>,
    #[serde(default, skip_serializing_if = "Nullable::is_absent")]
    pub system_fingerprint: Nullable<String>,
    #[serde(default, skip_serializing_if = "Nullable::is_absent")]
    pub metrics: Nullable<Value>,
    #[serde(flatten)]
    pub extra: Extra,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompletionChunkChoice {
    pub index: u32,
    pub text: String,
    #[serde(default, skip_serializing_if = "Nullable::is_absent")]
    pub logprobs: Nullable<Value>,
    #[serde(default, skip_serializing_if = "Nullable::is_absent")]
    pub finish_reason: Nullable<FinishReason>,
    #[serde(default, skip_serializing_if = "Nullable::is_absent")]
    pub stop_reason: Nullable<Value>,
    #[serde(default, skip_serializing_if = "Nullable::is_absent")]
    pub prompt_token_ids: Nullable<Vec<u32>>,
    #[serde(default, skip_serializing_if = "Nullable::is_absent")]
    pub token_ids: Nullable<Vec<u32>>,
    #[serde(flatten)]
    pub extra: Extra,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn chunk_keeps_null_versus_absent() {
        let raw = r#"{"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"content":"hi"},"logprobs":null,"finish_reason":null,"token_ids":null}]}"#;
        let chunk: ChatCompletionChunk = serde_json::from_str(raw).unwrap();
        assert!(chunk.choices[0].finish_reason.is_null());
        assert!(chunk.choices[0].stop_reason.is_absent());
        assert_eq!(serde_json::to_string(&chunk).unwrap(), raw);
    }

    #[test]
    fn usage_only_chunk_has_empty_choices() {
        let chunk: ChatCompletionChunk = serde_json::from_value(json!({
            "id": "chatcmpl-1", "object": "chat.completion.chunk", "created": 1, "model": "m",
            "choices": [], "usage": {"prompt_tokens": 3, "total_tokens": 5, "completion_tokens": 2}
        }))
        .unwrap();
        assert!(chunk.choices.is_empty());
        assert_eq!(chunk.usage.value().map(Usage::completion_tokens), Some(2));
    }

    #[test]
    fn delta_tool_call_fragments_parse() {
        let delta: Delta = serde_json::from_value(json!({
            "tool_calls": [{"index": 0, "function": {"arguments": "{\"ci"}}]
        }))
        .unwrap();
        let call = &delta.tool_calls.value().unwrap()[0];
        assert!(call.id.is_absent());
        assert_eq!(
            call.function
                .value()
                .and_then(|f| f.arguments.value())
                .map(String::as_str),
            Some("{\"ci")
        );
    }

    #[test]
    fn stop_reason_accepts_token_id_or_string() {
        for stop in [json!(128009), json!("</s>"), json!(null)] {
            let choice: CompletionChoice = serde_json::from_value(json!({
                "index": 0, "text": "", "finish_reason": "stop", "stop_reason": stop.clone()
            }))
            .unwrap();
            assert_eq!(serde_json::to_value(&choice).unwrap()["stop_reason"], stop);
        }
    }

    #[test]
    fn response_requires_core_fields() {
        assert!(
            serde_json::from_value::<ChatCompletionResponse>(json!({
                "object": "chat.completion", "created": 1, "model": "m", "choices": []
            }))
            .is_err()
        );
    }
}
