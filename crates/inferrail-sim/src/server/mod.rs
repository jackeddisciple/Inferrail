//! HTTP front end with vLLM's routes, status codes and wire format.

mod wire;

use std::convert::Infallible;
use std::future::Future;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use axum::Router;
use axum::body::{Body, Bytes};
use axum::extract::{DefaultBodyLimit, State};
use axum::http::{HeaderMap, HeaderValue, StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use inferrail_core::openai::sse::{DONE_FRAME, data_frame};
use inferrail_core::openai::{
    ChatCompletionRequest, CompletionRequest, ErrorResponse, FinishReason, PromptRef,
};
use serde::Serialize;
use tokio::net::TcpListener;
use xxhash_rust::xxh3::xxh3_128;

use crate::engine::{Engine, EngineConfig, Sequence};
use crate::tokenizer;
use wire::Meta;

pub const EMULATED_VLLM_VERSION: &str = "0.31.0";
pub const VOCAB_SIZE: u32 = 128_256;
const MAX_BODY_BYTES: usize = 64 * 1024 * 1024;

#[derive(Debug, Clone)]
pub struct SimConfig {
    pub model: String,
    pub served_model_names: Vec<String>,
    pub max_model_len: usize,
    pub default_max_tokens: usize,
    pub enable_prompt_tokens_details: bool,
    pub seed: u64,
    pub engine: EngineConfig,
}

impl Default for SimConfig {
    fn default() -> Self {
        Self {
            model: "inferrail/sim-8b".to_owned(),
            served_model_names: Vec::new(),
            max_model_len: 32_768,
            default_max_tokens: 256,
            enable_prompt_tokens_details: false,
            seed: 0,
            engine: EngineConfig::default(),
        }
    }
}

impl SimConfig {
    pub fn served_names(&self) -> Vec<String> {
        if self.served_model_names.is_empty() {
            vec![self.model.clone()]
        } else {
            self.served_model_names.clone()
        }
    }
}

#[derive(Debug)]
struct AppState {
    config: SimConfig,
    names: Vec<String>,
    engine: Engine,
    started_unix: u64,
    requests: AtomicU64,
}

impl AppState {
    fn next_hex(&self) -> String {
        let n = self.requests.fetch_add(1, Ordering::Relaxed);
        let key = [self.config.seed.to_le_bytes(), n.to_le_bytes()].concat();
        format!("{:032x}", xxh3_128(&key))
    }
}

/// A running simulator: one engine behind the vLLM HTTP API.
#[derive(Debug, Clone)]
pub struct Sim {
    state: Arc<AppState>,
}

impl Sim {
    /// Starts the engine; must be called inside a Tokio runtime.
    pub fn new(config: SimConfig) -> Self {
        let engine = Engine::start(config.engine.clone());
        let names = config.served_names();
        Self {
            state: Arc::new(AppState {
                names,
                engine,
                started_unix: unix_now(),
                requests: AtomicU64::new(0),
                config,
            }),
        }
    }

    pub fn engine(&self) -> &Engine {
        &self.state.engine
    }

    pub fn config(&self) -> &SimConfig {
        &self.state.config
    }

    pub fn router(&self) -> Router {
        Router::new()
            .route("/v1/chat/completions", post(chat_completions))
            .route("/v1/completions", post(completions))
            .route("/v1/models", get(models))
            .route("/health", get(health))
            .route("/version", get(version))
            .route("/reset_prefix_cache", post(reset_prefix_cache))
            .fallback(not_found)
            .layer(DefaultBodyLimit::max(MAX_BODY_BYTES))
            .with_state(Arc::clone(&self.state))
    }

    pub async fn serve(
        self,
        listener: TcpListener,
        shutdown: impl Future<Output = ()> + Send + 'static,
    ) -> std::io::Result<()> {
        axum::serve(listener, self.router())
            .with_graceful_shutdown(shutdown)
            .await
    }
}

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

#[derive(Debug)]
struct ApiError {
    status: StatusCode,
    body: Box<ErrorResponse>,
}

impl ApiError {
    fn bad_request(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            body: Box::new(ErrorResponse::bad_request(message)),
        }
    }

    fn not_found(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::NOT_FOUND,
            body: Box::new(ErrorResponse::not_found(message)),
        }
    }

    fn internal(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            body: Box::new(ErrorResponse::new(message, "InternalServerError", 500)),
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        json_response(self.status, &self.body)
    }
}

fn json_response(status: StatusCode, value: &impl Serialize) -> Response {
    match serde_json::to_vec(value) {
        Ok(bytes) => (
            status,
            [(
                header::CONTENT_TYPE,
                HeaderValue::from_static("application/json"),
            )],
            bytes,
        )
            .into_response(),
        Err(_) => StatusCode::INTERNAL_SERVER_ERROR.into_response(),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Endpoint {
    Chat,
    Completion,
}

#[derive(Debug)]
struct Plan {
    endpoint: Endpoint,
    id: String,
    created: u64,
    model: String,
    prompt: Vec<u32>,
    max_tokens: usize,
    finish: FinishReason,
    stream: bool,
    include_usage: bool,
    seed: u64,
}

impl Plan {
    fn meta(&self) -> Meta<'_> {
        Meta {
            id: &self.id,
            created: self.created,
            model: &self.model,
        }
    }
}

#[derive(Debug)]
struct Common<'a> {
    model: Option<&'a str>,
    max_tokens: Option<u64>,
    n: u32,
    stream: bool,
    include_usage: bool,
    seed: Option<i64>,
}

fn parse<T: serde::de::DeserializeOwned>(body: &[u8]) -> Result<T, ApiError> {
    serde_json::from_slice(body).map_err(|e| ApiError::bad_request(format!("Invalid request: {e}")))
}

fn plan(
    state: &AppState,
    headers: &HeaderMap,
    endpoint: Endpoint,
    common: Common<'_>,
    prompt: Vec<u32>,
) -> Result<Plan, ApiError> {
    let model = match common.model {
        None => state.names[0].clone(),
        Some(name) if state.names.iter().any(|n| n == name) => name.to_owned(),
        Some(name) => {
            return Err(ApiError::not_found(format!(
                "The model `{name}` does not exist."
            )));
        }
    };
    if common.n != 1 {
        return Err(ApiError::bad_request(
            "n other than 1 is not supported by inferrail-sim",
        ));
    }
    if prompt.is_empty() {
        return Err(ApiError::bad_request("The decoder prompt cannot be empty"));
    }
    let max_len = state.config.max_model_len;
    if prompt.len() > max_len {
        return Err(ApiError::bad_request(format!(
            "This model's maximum context length is {max_len} tokens. However, your request has {} input tokens. Please reduce the length of the input messages.",
            prompt.len()
        )));
    }
    let room = max_len - prompt.len();
    let (max_tokens, finish) = match common.max_tokens {
        Some(0) => return Err(ApiError::bad_request("max_tokens must be at least 1")),
        Some(asked) if asked > room as u64 => {
            return Err(ApiError::bad_request(format!(
                "'max_tokens' or 'max_completion_tokens' is too large: {asked}. This model's maximum context length is {max_len} tokens and your request has {} input tokens ({asked} > {max_len} - {}).",
                prompt.len(),
                prompt.len()
            )));
        }
        Some(asked) => (asked as usize, FinishReason::Length),
        None if room == 0 => {
            return Err(ApiError::bad_request(format!(
                "This model's maximum context length is {max_len} tokens and the prompt leaves no room to generate."
            )));
        }
        None if state.config.default_max_tokens < room => {
            (state.config.default_max_tokens.max(1), FinishReason::Stop)
        }
        None => (room, FinishReason::Length),
    };
    let base = headers
        .get("x-request-id")
        .and_then(|v| v.to_str().ok())
        .filter(|v| !v.is_empty() && v.len() <= 256)
        .map_or_else(|| state.next_hex(), str::to_owned);
    let prefix = match endpoint {
        Endpoint::Chat => "chatcmpl",
        Endpoint::Completion => "cmpl",
    };
    let id = format!("{prefix}-{base}");
    let seed = match common.seed {
        Some(seed) => seed as u64,
        None => xxh3_128(id.as_bytes()) as u64,
    };
    Ok(Plan {
        endpoint,
        id,
        created: unix_now(),
        model,
        prompt,
        max_tokens,
        finish,
        stream: common.stream,
        include_usage: common.include_usage,
        seed,
    })
}

async fn chat_completions(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let request: ChatCompletionRequest = match parse(&body) {
        Ok(request) => request,
        Err(err) => return err.into_response(),
    };
    let add_generation_prompt = request
        .extra
        .get("add_generation_prompt")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(true);
    let prompt = tokenizer::render_chat(
        &request.messages,
        request.tools.value().map(Vec::as_slice),
        add_generation_prompt,
    );
    let common = Common {
        model: request.model(),
        max_tokens: request.max_output_tokens(),
        n: request.choice_count(),
        stream: request.is_stream(),
        include_usage: request.include_usage(),
        seed: request.seed.value().copied(),
    };
    match plan(&state, &headers, Endpoint::Chat, common, prompt) {
        Ok(plan) => respond(state, plan).await,
        Err(err) => err.into_response(),
    }
}

async fn completions(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let request: CompletionRequest = match parse(&body) {
        Ok(request) => request,
        Err(err) => return err.into_response(),
    };
    let prompt = match request.prompt.value().map(|p| (p.batch_len(), p.single())) {
        None => {
            return ApiError::bad_request("Either prompt or prompt_embeds must be provided")
                .into_response();
        }
        Some((_, Some(PromptRef::Text(text)))) => tokenizer::encode_prompt(text),
        Some((_, Some(PromptRef::Tokens(tokens)))) => {
            if let Some(bad) = tokens.iter().find(|&&t| t >= VOCAB_SIZE) {
                return ApiError::bad_request(format!("Token id {bad} is out of vocabulary"))
                    .into_response();
            }
            tokens.to_vec()
        }
        Some((len, None)) => {
            return ApiError::bad_request(format!(
                "inferrail-sim serves one prompt per request, got {len}"
            ))
            .into_response();
        }
    };
    let common = Common {
        model: request.model(),
        max_tokens: request.max_tokens.value().copied(),
        n: request.choice_count(),
        stream: request.is_stream(),
        include_usage: request.include_usage(),
        seed: request.seed.value().copied(),
    };
    match plan(&state, &headers, Endpoint::Completion, common, prompt) {
        Ok(plan) => respond(state, plan).await,
        Err(err) => err.into_response(),
    }
}

async fn respond(state: Arc<AppState>, plan: Plan) -> Response {
    if plan.stream {
        stream_response(state, plan)
    } else {
        full_response(state, plan)
            .await
            .unwrap_or_else(IntoResponse::into_response)
    }
}

fn cached_detail(state: &AppState, seq: &Sequence) -> Option<u64> {
    state
        .config
        .enable_prompt_tokens_details
        .then_some(seq.cached_tokens as u64)
}

async fn full_response(state: Arc<AppState>, plan: Plan) -> Result<Response, ApiError> {
    let mut seq = state
        .engine
        .submit(&plan.prompt)
        .admitted()
        .await
        .map_err(|e| ApiError::internal(e.to_string()))?;
    let mut text = String::new();
    for (_, piece) in tokenizer::generate(plan.seed, plan.max_tokens) {
        seq.next_token().await;
        text.push_str(&piece);
    }
    let usage = wire::full_usage(
        plan.prompt.len() as u64,
        seq.generated() as u64,
        cached_detail(&state, &seq),
    );
    let meta = plan.meta();
    Ok(match plan.endpoint {
        Endpoint::Chat => json_response(
            StatusCode::OK,
            &wire::chat_response(meta, text, plan.finish.clone(), usage),
        ),
        Endpoint::Completion => json_response(
            StatusCode::OK,
            &wire::completion_response(meta, text, plan.finish.clone(), usage),
        ),
    })
}

fn frame(value: &impl Serialize) -> Bytes {
    serde_json::to_vec(value)
        .map(|json| Bytes::from(data_frame(&json)))
        .unwrap_or_default()
}

fn stream_response(state: Arc<AppState>, plan: Plan) -> Response {
    let body = async_stream::stream! {
        let admitted = state.engine.submit(&plan.prompt).admitted().await;
        let mut seq = match admitted {
            Ok(seq) => seq,
            Err(err) => {
                yield Ok::<Bytes, Infallible>(frame(&ApiError::internal(err.to_string()).body));
                yield Ok(Bytes::from_static(DONE_FRAME));
                return;
            }
        };
        let meta = plan.meta();
        let pieces = tokenizer::generate(plan.seed, plan.max_tokens);
        let last = pieces.len().saturating_sub(1);
        for (i, (_, piece)) in pieces.iter().enumerate() {
            seq.next_token().await;
            let finish = (i == last).then_some(&plan.finish);
            match plan.endpoint {
                Endpoint::Chat => {
                    if i == 0 {
                        yield Ok(frame(&wire::chat_role_chunk(meta)));
                    }
                    yield Ok(frame(&wire::chat_token_chunk(meta, piece, finish)));
                }
                Endpoint::Completion => {
                    yield Ok(frame(&wire::completion_token_chunk(meta, piece, finish)));
                }
            }
        }
        if plan.include_usage {
            let usage = wire::stream_usage(
                plan.prompt.len() as u64,
                seq.generated() as u64,
                cached_detail(&state, &seq),
            );
            yield Ok(match plan.endpoint {
                Endpoint::Chat => frame(&wire::chat_usage_chunk(meta, usage)),
                Endpoint::Completion => frame(&wire::completion_usage_chunk(meta, usage)),
            });
        }
        drop(seq);
        yield Ok(Bytes::from_static(DONE_FRAME));
    };
    (
        [(
            header::CONTENT_TYPE,
            HeaderValue::from_static("text/event-stream; charset=utf-8"),
        )],
        Body::from_stream(body),
    )
        .into_response()
}

async fn models(State(state): State<Arc<AppState>>) -> Response {
    let permission = format!("{:016x}", xxh3_128(state.config.model.as_bytes()) as u64);
    json_response(
        StatusCode::OK,
        &wire::model_list(
            &state.names,
            &state.config.model,
            state.started_unix,
            state.config.max_model_len as u64,
            &permission,
        ),
    )
}

async fn health() -> StatusCode {
    StatusCode::OK
}

async fn version() -> Response {
    json_response(
        StatusCode::OK,
        &serde_json::json!({
            "version": format!("{EMULATED_VLLM_VERSION}+inferrail.sim.{}", env!("CARGO_PKG_VERSION"))
        }),
    )
}

async fn reset_prefix_cache(State(state): State<Arc<AppState>>) -> StatusCode {
    state.engine.reset_prefix_cache();
    StatusCode::OK
}

async fn not_found() -> Response {
    json_response(
        StatusCode::NOT_FOUND,
        &serde_json::json!({"detail": "Not Found"}),
    )
}
