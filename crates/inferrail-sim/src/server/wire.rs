//! Builds response bodies field for field as vLLM serializes them.

use inferrail_core::openai::{
    CHAT_COMPLETION_CHUNK_OBJECT, CHAT_COMPLETION_OBJECT, ChatChoice, ChatChunkChoice,
    ChatCompletionChunk, ChatCompletionResponse, ChatMessage, CompletionChoice, CompletionChunk,
    CompletionChunkChoice, CompletionResponse, Delta, Extra, FinishReason, MessageContent,
    ModelCard, ModelList, ModelPermission, Nullable, PromptTokensDetails, Role,
    TEXT_COMPLETION_OBJECT, Usage,
};

#[derive(Debug, Clone, Copy)]
pub(crate) struct Meta<'a> {
    pub(crate) id: &'a str,
    pub(crate) created: u64,
    pub(crate) model: &'a str,
}

fn chunk(meta: Meta<'_>, choices: Vec<ChatChunkChoice>) -> ChatCompletionChunk {
    ChatCompletionChunk {
        id: meta.id.to_owned(),
        object: CHAT_COMPLETION_CHUNK_OBJECT.to_owned(),
        created: meta.created,
        model: meta.model.to_owned(),
        choices,
        usage: Nullable::Absent,
        system_fingerprint: Nullable::Absent,
        prompt_token_ids: Nullable::Absent,
        prompt_text: Nullable::Absent,
        service_tier: Nullable::Absent,
        metrics: Nullable::Absent,
        extra: Extra::new(),
    }
}

pub(crate) fn chat_role_chunk(meta: Meta<'_>) -> ChatCompletionChunk {
    let mut out = chunk(
        meta,
        vec![ChatChunkChoice {
            index: 0,
            delta: Delta {
                role: Nullable::Value(Role::Assistant),
                content: Nullable::Value(String::new()),
                ..Delta::default()
            },
            logprobs: Nullable::Null,
            finish_reason: Nullable::Null,
            stop_reason: Nullable::Absent,
            token_ids: Nullable::Absent,
            extra: Extra::new(),
        }],
    );
    out.prompt_token_ids = Nullable::Null;
    out.prompt_text = Nullable::Null;
    out
}

pub(crate) fn chat_token_chunk(
    meta: Meta<'_>,
    text: &str,
    finish: Option<&FinishReason>,
) -> ChatCompletionChunk {
    chunk(
        meta,
        vec![ChatChunkChoice {
            index: 0,
            delta: Delta {
                content: Nullable::Value(text.to_owned()),
                ..Delta::default()
            },
            logprobs: Nullable::Null,
            finish_reason: Nullable::null_or(finish.cloned()),
            stop_reason: if finish.is_some() {
                Nullable::Null
            } else {
                Nullable::Absent
            },
            token_ids: Nullable::Null,
            extra: Extra::new(),
        }],
    )
}

pub(crate) fn chat_usage_chunk(meta: Meta<'_>, usage: Usage) -> ChatCompletionChunk {
    let mut out = chunk(meta, Vec::new());
    out.usage = Nullable::Value(usage);
    out
}

pub(crate) fn completion_token_chunk(
    meta: Meta<'_>,
    text: &str,
    finish: Option<&FinishReason>,
) -> CompletionChunk {
    CompletionChunk {
        id: meta.id.to_owned(),
        object: TEXT_COMPLETION_OBJECT.to_owned(),
        created: meta.created,
        model: meta.model.to_owned(),
        choices: vec![CompletionChunkChoice {
            index: 0,
            text: text.to_owned(),
            logprobs: Nullable::Null,
            finish_reason: Nullable::null_or(finish.cloned()),
            stop_reason: Nullable::Null,
            prompt_token_ids: Nullable::Null,
            token_ids: Nullable::Null,
            extra: Extra::new(),
        }],
        usage: Nullable::Absent,
        system_fingerprint: Nullable::Absent,
        metrics: Nullable::Absent,
        extra: Extra::new(),
    }
}

pub(crate) fn completion_usage_chunk(meta: Meta<'_>, usage: Usage) -> CompletionChunk {
    CompletionChunk {
        id: meta.id.to_owned(),
        object: TEXT_COMPLETION_OBJECT.to_owned(),
        created: meta.created,
        model: meta.model.to_owned(),
        choices: Vec::new(),
        usage: Nullable::Value(usage),
        system_fingerprint: Nullable::Absent,
        metrics: Nullable::Absent,
        extra: Extra::new(),
    }
}

/// Usage as the final stream chunk carries it, with `None` fields dropped.
pub(crate) fn stream_usage(prompt: u64, completion: u64, cached: Option<u64>) -> Usage {
    Usage {
        prompt_tokens_details: match cached {
            Some(cached) => Nullable::Value(PromptTokensDetails {
                cached_tokens: Nullable::Value(cached),
                ..PromptTokensDetails::default()
            }),
            None => Nullable::Absent,
        },
        ..Usage::new(prompt, completion)
    }
}

/// Usage as a full response carries it, with every field dumped.
pub(crate) fn full_usage(prompt: u64, completion: u64, cached: Option<u64>) -> Usage {
    Usage {
        prompt_tokens_details: match cached {
            Some(cached) => Nullable::Value(PromptTokensDetails {
                cached_tokens: Nullable::Value(cached),
                created_cache_tokens: Nullable::Null,
                multimodal_tokens: Nullable::Null,
                ..PromptTokensDetails::default()
            }),
            None => Nullable::Null,
        },
        completion_tokens_details: Nullable::Null,
        ..Usage::new(prompt, completion)
    }
}

pub(crate) fn chat_response(
    meta: Meta<'_>,
    text: String,
    finish: FinishReason,
    usage: Usage,
) -> ChatCompletionResponse {
    let mut message = ChatMessage::new(Role::Assistant, String::new());
    message.content = Nullable::Value(MessageContent::Text(text));
    message.refusal = Nullable::Null;
    message.annotations = Nullable::Null;
    message.audio = Nullable::Null;
    message.function_call = Nullable::Null;
    message.reasoning = Nullable::Null;
    ChatCompletionResponse {
        id: meta.id.to_owned(),
        object: CHAT_COMPLETION_OBJECT.to_owned(),
        created: meta.created,
        model: meta.model.to_owned(),
        choices: vec![ChatChoice {
            index: 0,
            message,
            logprobs: Nullable::Null,
            finish_reason: Nullable::Value(finish),
            stop_reason: Nullable::Null,
            token_ids: Nullable::Null,
            routed_experts: Nullable::Null,
            extra: Extra::new(),
        }],
        service_tier: Nullable::Null,
        system_fingerprint: Nullable::Null,
        usage: Nullable::Value(usage),
        prompt_logprobs: Nullable::Null,
        prompt_token_ids: Nullable::Null,
        prompt_text: Nullable::Null,
        kv_transfer_params: Nullable::Null,
        ec_transfer_params: Nullable::Null,
        metrics: Nullable::Null,
        extra: Extra::new(),
    }
}

pub(crate) fn completion_response(
    meta: Meta<'_>,
    text: String,
    finish: FinishReason,
    usage: Usage,
) -> CompletionResponse {
    CompletionResponse {
        id: meta.id.to_owned(),
        object: TEXT_COMPLETION_OBJECT.to_owned(),
        created: meta.created,
        model: meta.model.to_owned(),
        choices: vec![CompletionChoice {
            index: 0,
            text,
            logprobs: Nullable::Null,
            finish_reason: Nullable::Value(finish),
            stop_reason: Nullable::Null,
            token_ids: Nullable::Null,
            prompt_logprobs: Nullable::Null,
            prompt_token_ids: Nullable::Null,
            routed_experts: Nullable::Null,
            extra: Extra::new(),
        }],
        service_tier: Nullable::Null,
        system_fingerprint: Nullable::Null,
        usage: Nullable::Value(usage),
        kv_transfer_params: Nullable::Null,
        ec_transfer_params: Nullable::Null,
        metrics: Nullable::Null,
        extra: Extra::new(),
    }
}

pub(crate) fn model_list(
    names: &[String],
    root: &str,
    created: u64,
    max_model_len: u64,
    permission_id: &str,
) -> ModelList {
    let cards = names
        .iter()
        .map(|name| ModelCard {
            id: name.clone(),
            object: "model".to_owned(),
            created,
            owned_by: "vllm".to_owned(),
            root: Nullable::Value(root.to_owned()),
            parent: Nullable::Null,
            max_model_len: Nullable::Value(max_model_len),
            permission: Nullable::Value(vec![ModelPermission {
                id: format!("modelperm-{permission_id}"),
                object: "model_permission".to_owned(),
                created,
                allow_create_engine: Nullable::Value(false),
                allow_sampling: Nullable::Value(true),
                allow_logprobs: Nullable::Value(true),
                allow_search_indices: Nullable::Value(false),
                allow_view: Nullable::Value(true),
                allow_fine_tuning: Nullable::Value(false),
                organization: Nullable::Value("*".to_owned()),
                group: Nullable::Null,
                is_blocking: Nullable::Value(false),
                extra: Extra::new(),
            }]),
            extra: Extra::new(),
        })
        .collect();
    ModelList::new(cards)
}

#[cfg(test)]
mod tests {
    use super::*;

    const META: Meta<'static> = Meta {
        id: "chatcmpl-1",
        created: 7,
        model: "m",
    };

    #[test]
    fn chat_chunks_match_vllm_bytes() {
        assert_eq!(
            serde_json::to_string(&chat_role_chunk(META)).unwrap(),
            r#"{"id":"chatcmpl-1","object":"chat.completion.chunk","created":7,"model":"m","choices":[{"index":0,"delta":{"role":"assistant","content":""},"logprobs":null,"finish_reason":null}],"prompt_token_ids":null,"prompt_text":null}"#
        );
        assert_eq!(
            serde_json::to_string(&chat_token_chunk(META, " hi", None)).unwrap(),
            r#"{"id":"chatcmpl-1","object":"chat.completion.chunk","created":7,"model":"m","choices":[{"index":0,"delta":{"content":" hi"},"logprobs":null,"finish_reason":null,"token_ids":null}]}"#
        );
        assert_eq!(
            serde_json::to_string(&chat_token_chunk(META, ".", Some(&FinishReason::Length)))
                .unwrap(),
            r#"{"id":"chatcmpl-1","object":"chat.completion.chunk","created":7,"model":"m","choices":[{"index":0,"delta":{"content":"."},"logprobs":null,"finish_reason":"length","stop_reason":null,"token_ids":null}]}"#
        );
    }

    #[test]
    fn usage_shapes_differ_between_stream_and_full_responses() {
        assert_eq!(
            serde_json::to_string(&stream_usage(10, 2, Some(8))).unwrap(),
            r#"{"prompt_tokens":10,"total_tokens":12,"completion_tokens":2,"prompt_tokens_details":{"cached_tokens":8}}"#
        );
        assert_eq!(
            serde_json::to_string(&stream_usage(10, 2, None)).unwrap(),
            r#"{"prompt_tokens":10,"total_tokens":12,"completion_tokens":2}"#
        );
        assert_eq!(
            serde_json::to_string(&full_usage(10, 2, None)).unwrap(),
            r#"{"prompt_tokens":10,"total_tokens":12,"completion_tokens":2,"prompt_tokens_details":null,"completion_tokens_details":null}"#
        );
    }
}
