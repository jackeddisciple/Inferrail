use std::borrow::Cow;

use serde::{Deserialize, Serialize};
use serde_json::{Number, Value};

use super::{Extra, Nullable, Role, Stop, StreamOptions, Tool, ToolCall};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChatCompletionRequest {
    pub messages: Vec<ChatMessage>,
    #[serde(default, skip_serializing_if = "Nullable::is_absent")]
    pub model: Nullable<String>,
    #[serde(default, skip_serializing_if = "Nullable::is_absent")]
    pub frequency_penalty: Nullable<Number>,
    #[serde(default, skip_serializing_if = "Nullable::is_absent")]
    pub logit_bias: Nullable<Value>,
    #[serde(default, skip_serializing_if = "Nullable::is_absent")]
    pub logprobs: Nullable<bool>,
    #[serde(default, skip_serializing_if = "Nullable::is_absent")]
    pub top_logprobs: Nullable<u32>,
    #[serde(default, skip_serializing_if = "Nullable::is_absent")]
    pub max_tokens: Nullable<u64>,
    #[serde(default, skip_serializing_if = "Nullable::is_absent")]
    pub max_completion_tokens: Nullable<u64>,
    #[serde(default, skip_serializing_if = "Nullable::is_absent")]
    pub n: Nullable<u32>,
    #[serde(default, skip_serializing_if = "Nullable::is_absent")]
    pub presence_penalty: Nullable<Number>,
    #[serde(default, skip_serializing_if = "Nullable::is_absent")]
    pub response_format: Nullable<Value>,
    #[serde(default, skip_serializing_if = "Nullable::is_absent")]
    pub seed: Nullable<i64>,
    #[serde(default, skip_serializing_if = "Nullable::is_absent")]
    pub stop: Nullable<Stop>,
    #[serde(default, skip_serializing_if = "Nullable::is_absent")]
    pub stream: Nullable<bool>,
    #[serde(default, skip_serializing_if = "Nullable::is_absent")]
    pub stream_options: Nullable<StreamOptions>,
    #[serde(default, skip_serializing_if = "Nullable::is_absent")]
    pub temperature: Nullable<Number>,
    #[serde(default, skip_serializing_if = "Nullable::is_absent")]
    pub top_p: Nullable<Number>,
    #[serde(default, skip_serializing_if = "Nullable::is_absent")]
    pub tools: Nullable<Vec<Tool>>,
    #[serde(default, skip_serializing_if = "Nullable::is_absent")]
    pub tool_choice: Nullable<Value>,
    #[serde(default, skip_serializing_if = "Nullable::is_absent")]
    pub parallel_tool_calls: Nullable<bool>,
    #[serde(default, skip_serializing_if = "Nullable::is_absent")]
    pub user: Nullable<String>,
    #[serde(flatten)]
    pub extra: Extra,
}

impl ChatCompletionRequest {
    pub fn model(&self) -> Option<&str> {
        self.model.value().map(String::as_str)
    }

    pub fn is_stream(&self) -> bool {
        self.stream.value().copied().unwrap_or(false)
    }

    pub fn include_usage(&self) -> bool {
        StreamOptions::include_usage(&self.stream_options)
    }

    /// `max_completion_tokens` wins over the legacy `max_tokens`, as in OpenAI.
    pub fn max_output_tokens(&self) -> Option<u64> {
        self.max_completion_tokens
            .value()
            .or(self.max_tokens.value())
            .copied()
    }

    pub fn choice_count(&self) -> u32 {
        self.n.value().copied().unwrap_or(1)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: Role,
    #[serde(default, skip_serializing_if = "Nullable::is_absent")]
    pub content: Nullable<MessageContent>,
    #[serde(default, skip_serializing_if = "Nullable::is_absent")]
    pub name: Nullable<String>,
    #[serde(default, skip_serializing_if = "Nullable::is_absent")]
    pub refusal: Nullable<String>,
    #[serde(default, skip_serializing_if = "Nullable::is_absent")]
    pub annotations: Nullable<Value>,
    #[serde(default, skip_serializing_if = "Nullable::is_absent")]
    pub audio: Nullable<Value>,
    #[serde(default, skip_serializing_if = "Nullable::is_absent")]
    pub function_call: Nullable<Value>,
    #[serde(default, skip_serializing_if = "Nullable::is_absent")]
    pub tool_calls: Nullable<Vec<ToolCall>>,
    #[serde(default, skip_serializing_if = "Nullable::is_absent")]
    pub tool_call_id: Nullable<String>,
    #[serde(default, skip_serializing_if = "Nullable::is_absent")]
    pub reasoning: Nullable<String>,
    #[serde(flatten)]
    pub extra: Extra,
}

impl ChatMessage {
    pub fn new(role: Role, content: impl Into<String>) -> Self {
        Self {
            role,
            content: Nullable::Value(MessageContent::Text(content.into())),
            name: Nullable::Absent,
            refusal: Nullable::Absent,
            annotations: Nullable::Absent,
            audio: Nullable::Absent,
            function_call: Nullable::Absent,
            tool_calls: Nullable::Absent,
            tool_call_id: Nullable::Absent,
            reasoning: Nullable::Absent,
            extra: Extra::new(),
        }
    }

    /// Text content with multi part text joined by newlines, as vLLM renders it.
    pub fn text(&self) -> Option<Cow<'_, str>> {
        self.content.value().map(MessageContent::text)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Parts(Vec<ContentPart>),
}

impl MessageContent {
    pub fn text(&self) -> Cow<'_, str> {
        match self {
            Self::Text(text) => Cow::Borrowed(text),
            Self::Parts(parts) => {
                let mut texts = parts.iter().filter_map(ContentPart::text);
                match (texts.next(), texts.next()) {
                    (None, _) => Cow::Borrowed(""),
                    (Some(only), None) => Cow::Borrowed(only),
                    (Some(first), Some(second)) => {
                        let mut joined = format!("{first}\n{second}");
                        for text in texts {
                            joined.push('\n');
                            joined.push_str(text);
                        }
                        Cow::Owned(joined)
                    }
                }
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContentPart {
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(default, skip_serializing_if = "Nullable::is_absent")]
    pub text: Nullable<String>,
    #[serde(flatten)]
    pub extra: Extra,
}

impl ContentPart {
    pub fn text(&self) -> Option<&str> {
        match self.kind.as_str() {
            "text" | "input_text" => self.text.value().map(String::as_str),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn parse(value: Value) -> ChatCompletionRequest {
        serde_json::from_value(value).unwrap()
    }

    #[test]
    fn minimal_request_has_sane_defaults() {
        let req = parse(json!({"model": "m", "messages": [{"role": "user", "content": "hi"}]}));
        assert_eq!(req.model(), Some("m"));
        assert!(!req.is_stream());
        assert!(!req.include_usage());
        assert_eq!(req.max_output_tokens(), None);
        assert_eq!(req.choice_count(), 1);
        assert_eq!(req.messages[0].text().as_deref(), Some("hi"));
    }

    #[test]
    fn max_completion_tokens_wins_over_max_tokens() {
        let req = parse(json!({
            "model": "m", "messages": [], "max_tokens": 10, "max_completion_tokens": 20
        }));
        assert_eq!(req.max_output_tokens(), Some(20));
        let legacy = parse(json!({"model": "m", "messages": [], "max_tokens": 10}));
        assert_eq!(legacy.max_output_tokens(), Some(10));
    }

    #[test]
    fn sampling_numbers_keep_their_json_form() {
        let raw = json!({"model": "m", "messages": [], "temperature": 1, "top_p": 0.95});
        let req = parse(raw.clone());
        assert_eq!(serde_json::to_string(&req.temperature).unwrap(), "1");
        assert_eq!(serde_json::to_value(&req).unwrap(), raw);
    }

    #[test]
    fn vendor_fields_land_in_extra_and_survive() {
        let raw = json!({
            "model": "m",
            "messages": [{"role": "user", "content": "x"}],
            "top_k": 40,
            "ignore_eos": true,
            "chat_template_kwargs": {"enable_thinking": false}
        });
        let req = parse(raw.clone());
        assert_eq!(req.extra["top_k"], json!(40));
        assert_eq!(serde_json::to_value(&req).unwrap(), raw);
    }

    #[test]
    fn multi_part_content_joins_text_parts_only() {
        let req = parse(
            json!({"model": "m", "messages": [{"role": "user", "content": [
                {"type": "text", "text": "first"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}},
                {"type": "text", "text": "second"},
                {"type": "text", "text": "third"}
            ]}]}),
        );
        assert_eq!(
            req.messages[0].text().as_deref(),
            Some("first\nsecond\nthird")
        );
    }

    #[test]
    fn single_or_no_text_part_borrows() {
        let one = MessageContent::Parts(vec![ContentPart {
            kind: "text".into(),
            text: Nullable::Value("solo".into()),
            extra: Extra::new(),
        }]);
        assert!(matches!(one.text(), Cow::Borrowed("solo")));
        assert_eq!(MessageContent::Parts(vec![]).text(), "");
    }

    #[test]
    fn assistant_tool_call_with_null_content_round_trips() {
        let raw = json!({"model": "m", "messages": [
            {"role": "assistant", "content": null, "tool_calls": [
                {"id": "call_1", "type": "function",
                 "function": {"name": "get_weather", "arguments": "{\"city\":\"Pune\"}"}}
            ]},
            {"role": "tool", "tool_call_id": "call_1", "content": "31C"}
        ]});
        let req = parse(raw.clone());
        assert!(req.messages[0].content.is_null());
        assert_eq!(req.messages[0].text(), None);
        assert_eq!(serde_json::to_value(&req).unwrap(), raw);
    }

    #[test]
    fn missing_messages_is_an_error() {
        assert!(serde_json::from_value::<ChatCompletionRequest>(json!({"model": "m"})).is_err());
    }

    #[test]
    fn wrong_types_are_rejected() {
        for bad in [
            json!({"messages": "hi"}),
            json!({"messages": [], "stream": "yes"}),
            json!({"messages": [], "max_tokens": -1}),
            json!({"messages": [{"content": "no role"}]}),
        ] {
            assert!(serde_json::from_value::<ChatCompletionRequest>(bad).is_err());
        }
    }

    #[test]
    fn constructor_builds_plain_text_message() {
        let msg = ChatMessage::new(Role::System, "be brief");
        assert_eq!(
            serde_json::to_value(&msg).unwrap(),
            json!({"role": "system", "content": "be brief"})
        );
    }
}
