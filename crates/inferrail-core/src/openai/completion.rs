use serde::{Deserialize, Serialize};
use serde_json::{Number, Value};

use super::{Extra, Nullable, Stop, StreamOptions};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompletionRequest {
    #[serde(default, skip_serializing_if = "Nullable::is_absent")]
    pub model: Nullable<String>,
    #[serde(default, skip_serializing_if = "Nullable::is_absent")]
    pub prompt: Nullable<Prompt>,
    #[serde(default, skip_serializing_if = "Nullable::is_absent")]
    pub best_of: Nullable<u32>,
    #[serde(default, skip_serializing_if = "Nullable::is_absent")]
    pub echo: Nullable<bool>,
    #[serde(default, skip_serializing_if = "Nullable::is_absent")]
    pub frequency_penalty: Nullable<Number>,
    #[serde(default, skip_serializing_if = "Nullable::is_absent")]
    pub logit_bias: Nullable<Value>,
    #[serde(default, skip_serializing_if = "Nullable::is_absent")]
    pub logprobs: Nullable<u32>,
    #[serde(default, skip_serializing_if = "Nullable::is_absent")]
    pub max_tokens: Nullable<u64>,
    #[serde(default, skip_serializing_if = "Nullable::is_absent")]
    pub n: Nullable<u32>,
    #[serde(default, skip_serializing_if = "Nullable::is_absent")]
    pub presence_penalty: Nullable<Number>,
    #[serde(default, skip_serializing_if = "Nullable::is_absent")]
    pub seed: Nullable<i64>,
    #[serde(default, skip_serializing_if = "Nullable::is_absent")]
    pub stop: Nullable<Stop>,
    #[serde(default, skip_serializing_if = "Nullable::is_absent")]
    pub stream: Nullable<bool>,
    #[serde(default, skip_serializing_if = "Nullable::is_absent")]
    pub stream_options: Nullable<StreamOptions>,
    #[serde(default, skip_serializing_if = "Nullable::is_absent")]
    pub suffix: Nullable<String>,
    #[serde(default, skip_serializing_if = "Nullable::is_absent")]
    pub temperature: Nullable<Number>,
    #[serde(default, skip_serializing_if = "Nullable::is_absent")]
    pub top_p: Nullable<Number>,
    #[serde(default, skip_serializing_if = "Nullable::is_absent")]
    pub user: Nullable<String>,
    #[serde(flatten)]
    pub extra: Extra,
}

impl CompletionRequest {
    pub fn model(&self) -> Option<&str> {
        self.model.value().map(String::as_str)
    }

    pub fn is_stream(&self) -> bool {
        self.stream.value().copied().unwrap_or(false)
    }

    pub fn include_usage(&self) -> bool {
        StreamOptions::include_usage(&self.stream_options)
    }

    pub fn choice_count(&self) -> u32 {
        self.n.value().copied().unwrap_or(1)
    }
}

/// The four prompt shapes the completions API accepts.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Prompt {
    Text(String),
    TextBatch(Vec<String>),
    Tokens(Vec<u32>),
    TokenBatch(Vec<Vec<u32>>),
}

impl Prompt {
    pub fn batch_len(&self) -> usize {
        match self {
            Self::Text(_) | Self::Tokens(_) => 1,
            Self::TextBatch(batch) => batch.len(),
            Self::TokenBatch(batch) => batch.len(),
        }
    }

    /// The prompt when the request carries exactly one.
    pub fn single(&self) -> Option<PromptRef<'_>> {
        match self {
            Self::Text(text) => Some(PromptRef::Text(text)),
            Self::Tokens(tokens) => Some(PromptRef::Tokens(tokens)),
            Self::TextBatch(batch) if batch.len() == 1 => Some(PromptRef::Text(&batch[0])),
            Self::TokenBatch(batch) if batch.len() == 1 => Some(PromptRef::Tokens(&batch[0])),
            Self::TextBatch(_) | Self::TokenBatch(_) => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PromptRef<'a> {
    Text(&'a str),
    Tokens(&'a [u32]),
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn prompt(value: Value) -> Prompt {
        serde_json::from_value(value).unwrap()
    }

    #[test]
    fn every_prompt_shape_parses_and_round_trips() {
        for (raw, len) in [
            (json!("hello"), 1),
            (json!(["a", "b"]), 2),
            (json!([1, 2, 3]), 1),
            (json!([[1, 2], [3]]), 2),
        ] {
            let parsed = prompt(raw.clone());
            assert_eq!(parsed.batch_len(), len);
            assert_eq!(serde_json::to_value(&parsed).unwrap(), raw);
        }
    }

    #[test]
    fn single_unwraps_one_element_batches() {
        assert_eq!(
            prompt(json!(["only"])).single(),
            Some(PromptRef::Text("only"))
        );
        assert_eq!(
            prompt(json!([[7, 8]])).single(),
            Some(PromptRef::Tokens(&[7, 8]))
        );
        assert_eq!(prompt(json!(["a", "b"])).single(), None);
        assert_eq!(prompt(json!([])).single(), None);
    }

    #[test]
    fn mixed_or_negative_prompts_are_rejected() {
        for bad in [
            json!(["a", 1]),
            json!([-1]),
            json!([[1], "a"]),
            json!({"text": "x"}),
        ] {
            assert!(serde_json::from_value::<Prompt>(bad).is_err());
        }
    }

    #[test]
    fn request_helpers_follow_openai_defaults() {
        let req: CompletionRequest = serde_json::from_value(json!({
            "model": "m", "prompt": "x", "stream": true,
            "stream_options": {"include_usage": true}, "n": 2, "min_tokens": 4
        }))
        .unwrap();
        assert_eq!(req.model(), Some("m"));
        assert!(req.is_stream());
        assert!(req.include_usage());
        assert_eq!(req.choice_count(), 2);
        assert_eq!(req.extra["min_tokens"], json!(4));
    }

    #[test]
    fn logprobs_is_an_integer_for_completions() {
        assert!(
            serde_json::from_value::<CompletionRequest>(json!({"prompt": "x", "logprobs": true}))
                .is_err()
        );
    }
}
