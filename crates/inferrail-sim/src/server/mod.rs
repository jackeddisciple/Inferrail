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
    ChatCompletionRequest, CompletionRequest, ErrorResponse, FinishReason, Nullable, PromptRef,
    StreamOptions,
};
use serde::Serialize;
use tokio::net::TcpListener;
use xxhash_rust::xxh3::xxh3_128;

use crate::engine::{Engine, EngineConfig, EngineError, Sequence};
use crate::metrics::{self, Metrics};
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

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("invalid simulator config: {0}")]
pub struct InvalidConfig(String);

impl SimConfig {
    pub fn validate(&self) -> Result<(), InvalidConfig> {
        if self.max_model_len == 0 {
            return Err(InvalidConfig("max_model_len must be positive".to_owned()));
        }
        let kv_tokens = self.engine.kv_tokens();
        if kv_tokens < self.max_model_len {
            return Err(InvalidConfig(format!(
                "to serve at least one request with the model's max seq len ({}), the KV cache needs {} blocks but has {}; raise num_gpu_blocks or lower max_model_len",
                self.max_model_len,
                self.max_model_len.div_ceil(self.engine.block_size.max(1)),
                self.engine.num_gpu_blocks
            )));
        }
        if let Some(bad) = self
            .served_names()
            .iter()
            .find(|n| n.is_empty() || n.chars().any(char::is_whitespace))
        {
            return Err(InvalidConfig(format!(
                "served model name {bad:?} is not usable"
            )));
        }
        self.engine.validate().map_err(InvalidConfig)
    }

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
    metrics: Arc<Metrics>,
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
    pub fn new(config: SimConfig) -> Result<Self, InvalidConfig> {
        config.validate()?;
        let engine = Engine::start(config.engine.clone());
        let names = config.served_names();
        let metrics = Arc::new(Metrics::new(
            names[0].clone(),
            config.max_model_len,
            &config.engine,
        ));
        Ok(Self {
            state: Arc::new(AppState {
                names,
                engine,
                metrics,
                started_unix: unix_now(),
                requests: AtomicU64::new(0),
                config,
            }),
        })
    }

    pub fn engine(&self) -> &Engine {
        &self.state.engine
    }

    pub fn config(&self) -> &SimConfig {
        &self.state.config
    }

    pub fn metrics(&self) -> &Metrics {
        &self.state.metrics
    }

    pub fn router(&self) -> Router {
        Router::new()
            .route("/v1/chat/completions", post(chat_completions))
            .route("/v1/completions", post(completions))
            .route("/v1/models", get(models))
            .route("/health", get(health))
            .route("/metrics", get(prometheus))
            .route("/version", get(version))
            .route("/reset_prefix_cache", post(reset_prefix_cache))
            .fallback(not_found)
            .method_not_allowed_fallback(method_not_allowed)
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

    fn bad_param(message: impl Into<String>, param: &str) -> Self {
        let mut body = ErrorResponse::bad_request(message);
        body.error.param = Nullable::Value(param.to_owned());
        Self {
            status: StatusCode::BAD_REQUEST,
            body: Box::new(body),
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

    fn engine(err: EngineError) -> Self {
        match err {
            EngineError::Stopped => Self::internal(err.to_string()),
            EngineError::TooLong | EngineError::NoOutput => Self::bad_request(err.to_string()),
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

/// vLLM rejects a non empty `stream_options` unless `stream` is true.
fn check_stream_options(stream: bool, options: &Nullable<StreamOptions>) -> Result<(), ApiError> {
    let set = options.value().is_some_and(|o| {
        !o.include_usage.is_absent() || !o.continuous_usage_stats.is_absent() || !o.extra.is_empty()
    });
    if set && !stream {
        return Err(ApiError::bad_param(
            "Stream options can only be defined when `stream=True`.",
            "stream_options",
        ));
    }
    Ok(())
}

fn unsupported(param: &str) -> ApiError {
    ApiError::bad_param(
        format!("`{param}` is not supported by inferrail-sim"),
        param,
    )
}

fn check_prompt_logprobs(extra: &inferrail_core::openai::Extra) -> Result<(), ApiError> {
    match extra.get("prompt_logprobs") {
        Some(value) if !value.is_null() => Err(unsupported("prompt_logprobs")),
        _ => Ok(()),
    }
}

fn check_chat(request: &ChatCompletionRequest) -> Result<(), ApiError> {
    check_stream_options(request.is_stream(), &request.stream_options)?;
    if request.logprobs.value() == Some(&true) {
        return Err(unsupported("logprobs"));
    }
    check_prompt_logprobs(&request.extra)
}

fn check_completion(request: &CompletionRequest) -> Result<(), ApiError> {
    check_stream_options(request.is_stream(), &request.stream_options)?;
    if request.logprobs.is_value() {
        return Err(unsupported("logprobs"));
    }
    if request.echo.value() == Some(&true) {
        return Err(unsupported("echo"));
    }
    if request.best_of.value().is_some_and(|&n| n > 1) {
        return Err(unsupported("best_of"));
    }
    check_prompt_logprobs(&request.extra)
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
    if let Err(err) = check_chat(&request) {
        return err.into_response();
    }
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
    if let Err(err) = check_completion(&request) {
        return err.into_response();
    }
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

fn submit(state: &AppState, plan: &Plan) -> (crate::engine::Ticket, Vec<String>) {
    let (ids, pieces) = tokenizer::generate(plan.seed, plan.max_tokens)
        .into_iter()
        .unzip();
    (state.engine.submit(&plan.prompt, ids), pieces)
}

async fn full_response(state: Arc<AppState>, plan: Plan) -> Result<Response, ApiError> {
    let mut tracker = state.metrics.tracker(plan.max_tokens);
    let (ticket, pieces) = submit(&state, &plan);
    let mut seq = match ticket.admitted().await {
        Ok(seq) => seq,
        Err(err) => {
            tracker.finish(&FinishReason::Error);
            return Err(ApiError::engine(err));
        }
    };
    tracker.admitted(&seq);
    let mut text = String::new();
    for piece in &pieces {
        let Some(at) = seq.next_token().await else {
            tracker.finish(&FinishReason::Error);
            return Err(ApiError::engine(EngineError::Stopped));
        };
        tracker.token(at, seq.preemptions());
        text.push_str(piece);
    }
    tracker.finish(&plan.finish);
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

fn token_frames(
    plan: &Plan,
    index: usize,
    piece: &str,
    finish: Option<&FinishReason>,
) -> Vec<Bytes> {
    let meta = plan.meta();
    match plan.endpoint {
        Endpoint::Chat if index == 0 => vec![
            frame(&wire::chat_role_chunk(meta)),
            frame(&wire::chat_token_chunk(meta, piece, finish)),
        ],
        Endpoint::Chat => vec![frame(&wire::chat_token_chunk(meta, piece, finish))],
        Endpoint::Completion => vec![frame(&wire::completion_token_chunk(meta, piece, finish))],
    }
}

/// The request finishes, and frees its slot, when the engine emits the last
/// token; trailing frames only wait on the client.
fn stream_response(state: Arc<AppState>, plan: Plan) -> Response {
    let body = async_stream::stream! {
        let mut tracker = state.metrics.tracker(plan.max_tokens);
        let (ticket, pieces) = submit(&state, &plan);
        let mut seq = match ticket.admitted().await {
            Ok(seq) => seq,
            Err(err) => {
                tracker.finish(&FinishReason::Error);
                yield Ok::<Bytes, Infallible>(frame(&ApiError::engine(err).body));
                yield Ok(Bytes::from_static(DONE_FRAME));
                return;
            }
        };
        tracker.admitted(&seq);
        let Some((last, head)) = pieces.split_last() else {
            return;
        };
        for (i, piece) in head.iter().enumerate() {
            let Some(at) = seq.next_token().await else {
                tracker.finish(&FinishReason::Error);
                return;
            };
            tracker.token(at, seq.preemptions());
            for bytes in token_frames(&plan, i, piece, None) {
                yield Ok(bytes);
            }
        }
        let Some(at) = seq.next_token().await else {
            tracker.finish(&FinishReason::Error);
            return;
        };
        tracker.token(at, seq.preemptions());
        tracker.finish(&plan.finish);
        let usage = wire::stream_usage(
            plan.prompt.len() as u64,
            seq.generated() as u64,
            cached_detail(&state, &seq),
        );
        drop(seq);
        for bytes in token_frames(&plan, head.len(), last, Some(&plan.finish)) {
            yield Ok(bytes);
        }
        if plan.include_usage {
            let meta = plan.meta();
            yield Ok(match plan.endpoint {
                Endpoint::Chat => frame(&wire::chat_usage_chunk(meta, usage)),
                Endpoint::Completion => frame(&wire::completion_usage_chunk(meta, usage)),
            });
        }
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

async fn prometheus(State(state): State<Arc<AppState>>) -> Response {
    let body = state.metrics.render(&state.engine.snapshot());
    (
        [(
            header::CONTENT_TYPE,
            HeaderValue::from_static(metrics::CONTENT_TYPE),
        )],
        body,
    )
        .into_response()
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
    state.engine.reset_prefix_cache().await;
    StatusCode::OK
}

async fn method_not_allowed() -> Response {
    json_response(
        StatusCode::METHOD_NOT_ALLOWED,
        &serde_json::json!({"detail": "Method Not Allowed"}),
    )
}

async fn not_found() -> Response {
    json_response(
        StatusCode::NOT_FOUND,
        &serde_json::json!({"detail": "Not Found"}),
    )
}
