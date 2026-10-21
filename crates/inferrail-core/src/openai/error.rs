use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::{Extra, Nullable};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ErrorResponse {
    pub error: ErrorInfo,
    #[serde(flatten)]
    pub extra: Extra,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ErrorInfo {
    pub message: String,
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(default, skip_serializing_if = "Nullable::is_absent")]
    pub param: Nullable<String>,
    #[serde(default, skip_serializing_if = "Nullable::is_absent")]
    pub code: Nullable<Value>,
    #[serde(flatten)]
    pub extra: Extra,
}

impl ErrorResponse {
    /// An error in the shape vLLM returns: the HTTP status doubles as `code`.
    pub fn new(message: impl Into<String>, kind: impl Into<String>, status: u16) -> Self {
        Self {
            error: ErrorInfo {
                message: message.into(),
                kind: kind.into(),
                param: Nullable::Null,
                code: Nullable::Value(Value::from(status)),
                extra: Extra::new(),
            },
            extra: Extra::new(),
        }
    }

    pub fn bad_request(message: impl Into<String>) -> Self {
        Self::new(message, "BadRequestError", 400)
    }

    pub fn not_found(message: impl Into<String>) -> Self {
        Self::new(message, "NotFoundError", 404)
    }

    pub fn status(&self) -> Option<u16> {
        self.error
            .code
            .value()
            .and_then(Value::as_u64)
            .and_then(|code| u16::try_from(code).ok())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn matches_vllm_error_shape() {
        assert_eq!(
            serde_json::to_string(&ErrorResponse::not_found("The model `x` does not exist."))
                .unwrap(),
            r#"{"error":{"message":"The model `x` does not exist.","type":"NotFoundError","param":null,"code":404}}"#
        );
        assert_eq!(ErrorResponse::bad_request("nope").status(), Some(400));
    }

    #[test]
    fn openai_string_codes_round_trip() {
        let raw = json!({"error": {
            "message": "The model `gpt-5` does not exist", "type": "invalid_request_error",
            "param": null, "code": "model_not_found"
        }});
        let parsed: ErrorResponse = serde_json::from_value(raw.clone()).unwrap();
        assert_eq!(parsed.status(), None);
        assert_eq!(serde_json::to_value(&parsed).unwrap(), raw);
    }
}
