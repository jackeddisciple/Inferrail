use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::{Extra, Nullable};

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Usage {
    pub prompt_tokens: u64,
    pub total_tokens: u64,
    #[serde(default, skip_serializing_if = "Nullable::is_absent")]
    pub completion_tokens: Nullable<u64>,
    #[serde(default, skip_serializing_if = "Nullable::is_absent")]
    pub prompt_tokens_details: Nullable<PromptTokensDetails>,
    #[serde(default, skip_serializing_if = "Nullable::is_absent")]
    pub completion_tokens_details: Nullable<CompletionTokensDetails>,
    #[serde(flatten)]
    pub extra: Extra,
}

impl Usage {
    pub fn new(prompt_tokens: u64, completion_tokens: u64) -> Self {
        Self {
            prompt_tokens,
            total_tokens: prompt_tokens.saturating_add(completion_tokens),
            completion_tokens: Nullable::Value(completion_tokens),
            ..Self::default()
        }
    }

    pub fn completion_tokens(&self) -> u64 {
        self.completion_tokens.value().copied().unwrap_or(0)
    }

    pub fn cached_tokens(&self) -> Option<u64> {
        self.prompt_tokens_details
            .value()
            .and_then(|d| d.cached_tokens.value().copied())
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct PromptTokensDetails {
    #[serde(default, skip_serializing_if = "Nullable::is_absent")]
    pub cached_tokens: Nullable<u64>,
    #[serde(default, skip_serializing_if = "Nullable::is_absent")]
    pub created_cache_tokens: Nullable<u64>,
    #[serde(default, skip_serializing_if = "Nullable::is_absent")]
    pub multimodal_tokens: Nullable<Value>,
    #[serde(default, skip_serializing_if = "Nullable::is_absent")]
    pub audio_tokens: Nullable<u64>,
    #[serde(flatten)]
    pub extra: Extra,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct CompletionTokensDetails {
    #[serde(default, skip_serializing_if = "Nullable::is_absent")]
    pub reasoning_tokens: Nullable<u64>,
    #[serde(default, skip_serializing_if = "Nullable::is_absent")]
    pub audio_tokens: Nullable<u64>,
    #[serde(default, skip_serializing_if = "Nullable::is_absent")]
    pub accepted_prediction_tokens: Nullable<u64>,
    #[serde(default, skip_serializing_if = "Nullable::is_absent")]
    pub rejected_prediction_tokens: Nullable<u64>,
    #[serde(flatten)]
    pub extra: Extra,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn new_fills_total_and_keeps_vllm_field_order() {
        let usage = Usage::new(12, 5);
        assert_eq!(
            serde_json::to_string(&usage).unwrap(),
            r#"{"prompt_tokens":12,"total_tokens":17,"completion_tokens":5}"#
        );
    }

    #[test]
    fn cached_tokens_reads_details() {
        let usage: Usage = serde_json::from_value(json!({
            "prompt_tokens": 64, "total_tokens": 70, "completion_tokens": 6,
            "prompt_tokens_details": {"cached_tokens": 48}
        }))
        .unwrap();
        assert_eq!(usage.cached_tokens(), Some(48));
        assert_eq!(usage.completion_tokens(), 6);
        assert_eq!(Usage::default().cached_tokens(), None);
    }

    #[test]
    fn null_details_round_trip() {
        let raw = json!({
            "prompt_tokens": 1, "total_tokens": 1, "completion_tokens": null,
            "prompt_tokens_details": null, "completion_tokens_details": null
        });
        let usage: Usage = serde_json::from_value(raw.clone()).unwrap();
        assert_eq!(usage.completion_tokens(), 0);
        assert_eq!(serde_json::to_value(&usage).unwrap(), raw);
    }
}
