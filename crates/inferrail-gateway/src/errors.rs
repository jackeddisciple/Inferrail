//! Errors the gateway answers itself, in the OpenAI error shape vLLM uses.

use hyper::StatusCode;
use inferrail_core::openai::{ErrorResponse, Nullable};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GatewayError {
    status: StatusCode,
    kind: &'static str,
    message: String,
    param: Option<&'static str>,
    retry_after: Option<u32>,
    allow: Option<&'static str>,
}

impl GatewayError {
    fn new(status: StatusCode, kind: &'static str, message: impl Into<String>) -> Self {
        Self {
            status,
            kind,
            message: message.into(),
            param: None,
            retry_after: None,
            allow: None,
        }
    }

    pub fn bad_request(message: impl Into<String>) -> Self {
        Self::new(StatusCode::BAD_REQUEST, "BadRequestError", message)
    }

    pub fn bad_param(message: impl Into<String>, param: &'static str) -> Self {
        Self {
            param: Some(param),
            ..Self::bad_request(message)
        }
    }

    pub fn unknown_model(name: &str) -> Self {
        Self {
            param: Some("model"),
            ..Self::new(
                StatusCode::NOT_FOUND,
                "NotFoundError",
                format!("The model `{name}` does not exist."),
            )
        }
    }

    pub fn not_found() -> Self {
        Self::new(StatusCode::NOT_FOUND, "NotFoundError", "Not Found")
    }

    pub fn method_not_allowed(allow: &'static str) -> Self {
        Self {
            allow: Some(allow),
            ..Self::new(
                StatusCode::METHOD_NOT_ALLOWED,
                "MethodNotAllowedError",
                "Method Not Allowed",
            )
        }
    }

    pub fn too_large(limit: usize) -> Self {
        Self::new(
            StatusCode::PAYLOAD_TOO_LARGE,
            "RequestTooLargeError",
            format!("Request body exceeds the {limit} byte limit"),
        )
    }

    pub fn unavailable(message: impl Into<String>) -> Self {
        Self {
            retry_after: Some(1),
            ..Self::new(
                StatusCode::SERVICE_UNAVAILABLE,
                "ServiceUnavailableError",
                message,
            )
        }
    }

    pub fn bad_gateway(message: impl Into<String>) -> Self {
        Self::new(StatusCode::BAD_GATEWAY, "BadGatewayError", message)
    }

    pub fn timeout(message: impl Into<String>) -> Self {
        Self::new(StatusCode::GATEWAY_TIMEOUT, "GatewayTimeoutError", message)
    }

    pub fn slow_body() -> Self {
        Self::new(
            StatusCode::REQUEST_TIMEOUT,
            "RequestTimeoutError",
            "The request body did not arrive before the request timeout",
        )
    }

    pub fn status(&self) -> StatusCode {
        self.status
    }

    pub fn message(&self) -> &str {
        &self.message
    }

    pub fn retry_after(&self) -> Option<u32> {
        self.retry_after
    }

    pub fn allow(&self) -> Option<&'static str> {
        self.allow
    }

    pub fn to_json(&self) -> Vec<u8> {
        let mut body = ErrorResponse::new(self.message.as_str(), self.kind, self.status.as_u16());
        if let Some(param) = self.param {
            body.error.param = Nullable::Value(param.to_owned());
        }
        serde_json::to_vec(&body).unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{Value, json};

    fn json_of(err: &GatewayError) -> Value {
        serde_json::from_slice(&err.to_json()).unwrap()
    }

    #[test]
    fn unknown_models_match_the_vllm_message() {
        let err = GatewayError::unknown_model("gpt-4");
        assert_eq!(err.status(), StatusCode::NOT_FOUND);
        assert_eq!(
            json_of(&err),
            json!({"error": {"message": "The model `gpt-4` does not exist.",
                "type": "NotFoundError", "param": "model", "code": 404}})
        );
    }

    #[test]
    fn every_constructor_reports_its_status_as_code() {
        for err in [
            GatewayError::bad_request("x"),
            GatewayError::bad_param("x", "model"),
            GatewayError::not_found(),
            GatewayError::method_not_allowed("POST"),
            GatewayError::too_large(10),
            GatewayError::unavailable("x"),
            GatewayError::bad_gateway("x"),
            GatewayError::timeout("x"),
            GatewayError::slow_body(),
        ] {
            let body = json_of(&err);
            assert_eq!(body["error"]["code"], json!(err.status().as_u16()));
            assert!(body["error"]["type"].as_str().unwrap().ends_with("Error"));
        }
        assert_eq!(GatewayError::unavailable("x").retry_after(), Some(1));
        assert_eq!(
            GatewayError::method_not_allowed("POST").allow(),
            Some("POST")
        );
        assert_eq!(
            GatewayError::too_large(10).message(),
            "Request body exceeds the 10 byte limit"
        );
    }
}
