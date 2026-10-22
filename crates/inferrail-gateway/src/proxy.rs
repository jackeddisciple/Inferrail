use std::convert::Infallible;
use std::sync::Arc;
use std::time::Duration;

use http_body_util::{BodyExt, Full, Limited};
use hyper::body::{Bytes, Incoming};
use hyper::header::{self, HeaderMap, HeaderValue};
use hyper::http::request::Parts;
use hyper::{Method, Request, Response, StatusCode};
use inferrail_control::{Lease, Nanos, Replica};
use inferrail_core::ids::{ModelId, ReplicaId};
use tokio::time::{Instant, sleep, timeout, timeout_at};
use tracing::field::Empty;
use tracing::{Instrument, Span};

use crate::body::{Recording, ResponseBody, UpstreamBody};
use crate::decision_log::{self, Decision};
use crate::errors::GatewayError;
use crate::headers::{self, REPLICA};
use crate::inspect::{InspectError, inspect};
use crate::metrics::Routed;
use crate::request_id::REQUEST_ID;
use crate::telemetry::{self, SPAN_TARGET};
use crate::{Shared, models, route};

pub(crate) async fn handle(
    shared: Arc<Shared>,
    request: Request<Incoming>,
) -> Result<Response<ResponseBody>, Infallible> {
    let arrival = std::time::Instant::now();
    let id = shared.ids.resolve(request.headers());
    let version = request.version();
    let span = tracing::info_span!(
        target: SPAN_TARGET,
        "request",
        otel.kind = "server",
        http.request.method = %request.method(),
        url.path = %request.uri().path(),
        http.response.status_code = Empty,
        gen_ai.request.model = Empty,
        inferrail.request_id = Empty,
        inferrail.attempts = Empty,
        inferrail.outcome = Empty,
    );
    telemetry::adopt_parent(&span, request.headers());
    let mut response = match dispatch(&shared, request, &id, arrival)
        .instrument(span.clone())
        .await
    {
        Ok(response) => response,
        Err(err) => error_response(&err),
    };
    if !span.is_disabled() {
        span.record("http.response.status_code", response.status().as_u16());
        span.record("inferrail.request_id", id.to_str().unwrap_or_default());
    }
    *response.version_mut() = version;
    response.headers_mut().insert(REQUEST_ID, id);
    Ok(response)
}

enum Route {
    Proxy,
    Local(Result<Local, GatewayError>),
}

enum Local {
    Models,
    Health,
}

fn route_of(method: &Method, path: &str) -> Route {
    let read = method == Method::GET || method == Method::HEAD;
    let local = match path {
        "/v1/chat/completions" | "/v1/completions" if method == Method::POST => {
            return Route::Proxy;
        }
        "/v1/chat/completions" | "/v1/completions" => Err(GatewayError::method_not_allowed("POST")),
        "/v1/models" if read => Ok(Local::Models),
        "/health" if read => Ok(Local::Health),
        "/v1/models" | "/health" => Err(GatewayError::method_not_allowed("GET, HEAD")),
        _ => Err(GatewayError::not_found()),
    };
    Route::Local(local)
}

async fn dispatch(
    shared: &Arc<Shared>,
    request: Request<Incoming>,
    id: &HeaderValue,
    arrival: std::time::Instant,
) -> Result<Response<ResponseBody>, GatewayError> {
    match route_of(request.method(), request.uri().path()) {
        Route::Proxy => proxy(shared, request, id, arrival).await,
        Route::Local(local) => {
            if !expects_continue(request.headers()) {
                drain(request.into_body(), shared.config.server.max_body_bytes).await;
            }
            let local = local.inspect_err(|err| shared.metrics.rejected(None, err, false))?;
            match local {
                Local::Models => Ok(models::list(shared)),
                Local::Health => Ok(Response::new(ResponseBody::empty())),
            }
        }
    }
}

/// A client that sent `Expect: 100-continue` holds its body back until told
/// to send it; answering without reading the body declines it cleanly.
fn expects_continue(headers: &HeaderMap) -> bool {
    headers
        .get(header::EXPECT)
        .is_some_and(|v| v.as_bytes().eq_ignore_ascii_case(b"100-continue"))
}

const DRAIN_FLOOR: usize = 1 << 20;
const DRAIN_TIME: Duration = Duration::from_secs(2);

/// Reads and discards what is left of a request body, so a client that is
/// still uploading gets the answer rather than a connection reset. Bounded
/// in bytes and time; past either bound the connection simply closes.
async fn drain(mut body: Incoming, limit: usize) {
    let mut left = limit.max(DRAIN_FLOOR);
    let discard = async {
        while let Some(Ok(frame)) = body.frame().await {
            let len = frame.data_ref().map_or(0, Bytes::len);
            if len >= left {
                break;
            }
            left -= len;
        }
    };
    let _ = timeout(DRAIN_TIME, discard).await;
}

pub(crate) fn error_response(err: &GatewayError) -> Response<ResponseBody> {
    let mut response = Response::new(ResponseBody::full(err.to_json()));
    *response.status_mut() = err.status();
    let headers = response.headers_mut();
    headers.insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("application/json"),
    );
    if let Some(seconds) = err.retry_after() {
        headers.insert(header::RETRY_AFTER, HeaderValue::from(seconds));
    }
    if let Some(allow) = err.allow() {
        headers.insert(header::ALLOW, HeaderValue::from_static(allow));
    }
    response
}

/// What is known about a proxied request so far, for its metrics.
#[derive(Debug, Default)]
struct Seen {
    model: Option<ModelId>,
    replica: Option<ReplicaId>,
    attempts: u32,
}

async fn proxy(
    shared: &Arc<Shared>,
    request: Request<Incoming>,
    id: &HeaderValue,
    arrival: std::time::Instant,
) -> Result<Response<ResponseBody>, GatewayError> {
    let mut seen = Seen::default();
    let result = attempt(shared, request, id, arrival, &mut seen).await;
    if seen.attempts > 0 {
        Span::current().record("inferrail.attempts", seen.attempts);
    }
    match &result {
        Ok(response) => {
            if let (ResponseBody::Full(_), Some(model)) = (response.body(), &seen.model) {
                shared
                    .metrics
                    .answered(model, seen.replica.as_ref(), response.status());
            }
        }
        Err(err) => shared.metrics.rejected(seen.model.as_ref(), err, true),
    }
    result
}

async fn attempt(
    shared: &Arc<Shared>,
    request: Request<Incoming>,
    id: &HeaderValue,
    arrival: std::time::Instant,
    seen: &mut Seen,
) -> Result<Response<ResponseBody>, GatewayError> {
    let deadline = Instant::now() + shared.config.server.request_timeout;
    let (parts, body) = request.into_parts();
    let limit = shared.config.server.max_body_bytes;
    let body = timeout_at(deadline, read_body(&parts.headers, body, limit))
        .await
        .map_err(|_| GatewayError::slow_body())??;
    let (model, body) = resolve_model(shared, body)?;
    Span::current().record("gen_ai.request.model", model.as_str());
    seen.model = Some(model.clone());
    if shared.config.logging.log_prompts {
        tracing::debug!(request_id = ?id, body = %String::from_utf8_lossy(&body), "request body");
    }
    let policy = shared.policy(&model)?;
    let tokens = estimate_tokens(body.len());
    let sampled = shared
        .decisions
        .as_ref()
        .filter(|log| log.sampled(id.as_bytes()));
    let mut tried: Vec<ReplicaId> = Vec::new();
    let mut last: Option<Result<Response<ResponseBody>, GatewayError>> = None;
    for attempt in 0..shared.config.retry.max_attempts {
        if attempt > 0 {
            let pause = backoff(shared.config.retry.backoff, attempt);
            if Instant::now() + pause >= deadline {
                break;
            }
            sleep(pause).await;
        }
        seen.attempts = attempt + 1;
        let route_span = tracing::info_span!(
            target: SPAN_TARGET,
            "route",
            inferrail.policy = policy.kind().as_str(),
            inferrail.attempt = attempt,
            inferrail.candidates = Empty,
            inferrail.replica = Empty,
            inferrail.replica.in_flight = Empty,
            inferrail.engine.running = Empty,
            inferrail.engine.waiting = Empty,
            inferrail.engine.kv_usage = Empty,
            inferrail.load.age_ms = Empty,
            inferrail.decision_ns = Empty,
        );
        let now = shared.clock.now();
        let started = std::time::Instant::now();
        let selected = route_span
            .in_scope(|| route::select(&shared.registry, policy, &model, now, &tried, tokens));
        let took = started.elapsed();
        let (lease, candidates) = match selected {
            Ok(selected) => selected,
            Err(err) => return last.unwrap_or(Err(err)),
        };
        let routed = Routed {
            model: model.clone(),
            policy: policy.kind(),
            replica: lease.replica().id().clone(),
        };
        shared.metrics.decided(&routed, took);
        if let Some(log) = sampled {
            log.record(Decision {
                ts_ms: decision_log::unix_ms(),
                request_id: id.to_str().unwrap_or_default().to_owned(),
                model: model.clone(),
                policy: policy.kind().as_str(),
                attempt,
                chosen: routed.replica.clone(),
                decision_ns: u64::try_from(took.as_nanos()).unwrap_or(u64::MAX),
                candidates: decision_log::candidates(
                    &shared.registry.load(),
                    &model,
                    now,
                    &tried,
                    shared.config.load.max_age,
                ),
            });
        }
        if !route_span.is_disabled() {
            describe(&route_span, shared, &lease, candidates, took);
        }
        drop(route_span);
        if attempt > 0 {
            shared.metrics.retried(&model);
        }
        tried.push(routed.replica.clone());
        seen.replica = Some(routed.replica.clone());
        let upstream_span = tracing::info_span!(
            target: SPAN_TARGET,
            "upstream",
            otel.kind = "client",
            server.address = lease.replica().endpoint().origin(),
            inferrail.replica = routed.replica.as_str(),
            http.response.status_code = Empty,
            error.type = Empty,
        );
        let recording = Recording::new(routed, arrival, Span::current());
        let forwarded = forward(shared, lease, recording, &parts, body.clone(), id, deadline)
            .instrument(upstream_span)
            .await;
        match forwarded {
            Attempt::Done(response) => return Ok(response),
            Attempt::Fatal(err) => return Err(err),
            Attempt::Retry(failure) => {
                tracing::debug!(request_id = ?id, attempt, "retrying on another replica");
                last = Some(failure);
            }
        }
    }
    last.unwrap_or_else(|| {
        Err(GatewayError::timeout(
            "no response before the request timeout",
        ))
    })
}

/// Puts the decision on its span: how many replicas were offered, which one
/// won, and its load at that moment.
fn describe(span: &Span, shared: &Shared, lease: &Lease, candidates: usize, took: Duration) {
    let replica = lease.replica();
    let load = replica.load(Instant::now(), shared.config.load.max_age);
    span.record("inferrail.candidates", candidates);
    span.record("inferrail.replica", replica.id().as_str());
    span.record("inferrail.replica.in_flight", load.in_flight);
    span.record("inferrail.decision_ns", took.as_nanos() as u64);
    if let Some(engine) = load.engine {
        span.record("inferrail.engine.running", engine.running);
        span.record("inferrail.engine.waiting", engine.waiting);
        span.record("inferrail.engine.kv_usage", engine.kv_usage);
    }
    if let Some(age) = load.age {
        span.record("inferrail.load.age_ms", age.as_secs_f64() * 1e3);
    }
}

/// Prompt tokens a body of `len` bytes holds, at about four bytes a token,
/// until requests are tokenized at the edge.
fn estimate_tokens(len: usize) -> u32 {
    u32::try_from(len.div_ceil(4)).unwrap_or(u32::MAX)
}

fn backoff(base: Duration, attempt: u32) -> Duration {
    base.saturating_mul(1 << attempt.saturating_sub(1).min(16))
}

/// Collects the body without keeping an allocation per frame: one frame is
/// kept as it arrived, more are copied into a single buffer, so memory stays
/// proportional to the body however finely the client chunks it.
async fn read_body(
    headers: &HeaderMap,
    mut body: Incoming,
    limit: usize,
) -> Result<Bytes, GatewayError> {
    let declared = headers
        .get(header::CONTENT_LENGTH)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse::<u64>().ok());
    if declared.is_some_and(|len| len > limit as u64) {
        if !expects_continue(headers) {
            drain(body, limit).await;
        }
        return Err(GatewayError::too_large(limit));
    }
    let mut first: Option<Bytes> = None;
    let mut joined: Vec<u8> = Vec::new();
    while let Some(frame) = body.frame().await {
        let frame =
            frame.map_err(|_| GatewayError::bad_request("could not read the request body"))?;
        let Ok(data) = frame.into_data() else {
            continue;
        };
        if data.is_empty() {
            continue;
        }
        let held = first.as_ref().map_or(joined.len(), Bytes::len);
        if held + data.len() > limit {
            drain(body, limit).await;
            return Err(GatewayError::too_large(limit));
        }
        match first.take() {
            None if joined.is_empty() => first = Some(data),
            previous => {
                if let Some(previous) = previous {
                    let expected = declared.map_or(0, |len| len as usize).min(limit);
                    joined.reserve(expected.max(previous.len() + data.len()));
                    joined.extend_from_slice(&previous);
                }
                joined.extend_from_slice(&data);
            }
        }
    }
    Ok(first.unwrap_or_else(|| Bytes::from(joined)))
}

/// Resolves the requested name to a served model. The body is only
/// rewritten when the engine would otherwise see a different name.
fn resolve_model(shared: &Shared, body: Bytes) -> Result<(ModelId, Bytes), GatewayError> {
    let (model, rewritten) = {
        let inspected = inspect(&body).map_err(|err| match err {
            InspectError::ModelNotString => GatewayError::bad_param(err.to_string(), "model"),
            _ => GatewayError::bad_request(err.to_string()),
        })?;
        match inspected.model.as_deref() {
            Some(name) => {
                let model = shared
                    .names
                    .get(name)
                    .ok_or_else(|| GatewayError::unknown_model(name))?
                    .clone();
                let rewrite =
                    (model.as_str() != name).then(|| inspected.with_model(&body, model.as_str()));
                (model, rewrite)
            }
            None => {
                let [only] = shared.config.models.as_slice() else {
                    return Err(GatewayError::bad_param(
                        "`model` is required when the gateway serves more than one model",
                        "model",
                    ));
                };
                let rewrite = inspected.with_model(&body, only.name.as_str());
                (only.name.clone(), Some(rewrite))
            }
        }
    };
    Ok((model, rewritten.map_or(body, Bytes::from)))
}

fn upstream_request(
    replica: &Replica,
    parts: &Parts,
    body: Bytes,
    id: &HeaderValue,
) -> Result<Request<Full<Bytes>>, GatewayError> {
    let path = parts.uri.path_and_query().map_or("/", |pq| pq.as_str());
    let uri = replica
        .endpoint()
        .uri_for(path)
        .parse()
        .map_err(|_| GatewayError::bad_gateway("invalid upstream address"))?;
    let mut request = Request::new(Full::new(body));
    *request.method_mut() = parts.method.clone();
    *request.uri_mut() = uri;
    let headers = request.headers_mut();
    headers.clone_from(&parts.headers);
    headers::for_upstream(headers);
    headers.insert(REQUEST_ID, id.clone());
    telemetry::propagate(headers);
    Ok(request)
}

enum Attempt {
    Done(Response<ResponseBody>),
    Retry(Result<Response<ResponseBody>, GatewayError>),
    Fatal(GatewayError),
}

const MAX_ERROR_BODY: usize = 64 * 1024;

fn is_retryable(status: StatusCode) -> bool {
    matches!(
        status,
        StatusCode::BAD_GATEWAY | StatusCode::SERVICE_UNAVAILABLE | StatusCode::GATEWAY_TIMEOUT
    )
}

/// One try against one replica. Nothing reaches the client until this
/// returns `Done`, which is what makes retrying the other outcomes safe.
async fn forward(
    shared: &Arc<Shared>,
    mut lease: Lease,
    recording: Recording,
    parts: &Parts,
    body: Bytes,
    id: &HeaderValue,
    deadline: Instant,
) -> Attempt {
    let replica = Arc::clone(lease.replica());
    let request = match upstream_request(&replica, parts, body, id) {
        Ok(request) => request,
        Err(err) => return Attempt::Fatal(err),
    };
    let went_down = replica.health().went_down().notified();
    tokio::pin!(went_down);
    went_down.as_mut().enable();
    let waiting = async {
        if !replica.health().probe_up() {
            return None;
        }
        tokio::select! {
            result = timeout_at(deadline, shared.client.request(request)) => Some(result),
            () = &mut went_down => None,
        }
    };
    let Some(answer) = waiting.await else {
        Span::current().record("error.type", "replica_down");
        tracing::debug!(replica = %replica.id(), "replica failed its health checks while the request waited");
        return Attempt::Retry(Err(GatewayError::bad_gateway(format!(
            "replica {} failed its health checks before responding",
            replica.id()
        ))));
    };
    let response = match answer {
        Ok(Ok(response)) => response,
        Ok(Err(err)) => {
            Span::current().record("error.type", "connection");
            record_failure(shared, &replica);
            tracing::debug!(replica = %replica.id(), error = %err, "upstream request failed");
            return Attempt::Retry(Err(GatewayError::bad_gateway(format!(
                "replica {} failed before responding",
                replica.id()
            ))));
        }
        Err(_) => {
            Span::current().record("error.type", "timeout");
            record_failure(shared, &replica);
            return Attempt::Fatal(GatewayError::timeout(
                "no response before the request timeout",
            ));
        }
    };
    lease.responded();
    let status = response.status();
    Span::current().record("http.response.status_code", status.as_u16());
    if status.is_server_error() {
        record_failure(shared, &replica);
    } else {
        replica.health().record_success();
    }
    let (mut head, incoming) = response.into_parts();
    headers::for_downstream(&mut head.headers);
    if let Ok(value) = HeaderValue::from_str(replica.id().as_str()) {
        head.headers.insert(REPLICA, value);
    }
    if is_retryable(status) {
        drop(lease);
        let buffered = timeout(
            Duration::from_secs(1),
            Limited::new(incoming, MAX_ERROR_BODY).collect(),
        )
        .await;
        return Attempt::Retry(match buffered {
            Ok(Ok(collected)) => Ok(Response::from_parts(
                head,
                ResponseBody::full(collected.to_bytes()),
            )),
            _ => Err(GatewayError::bad_gateway(format!(
                "replica {} answered {status}",
                replica.id()
            ))),
        });
    }
    let stream = head
        .headers
        .get(header::CONTENT_TYPE)
        .is_some_and(|v| v.as_bytes().starts_with(b"text/event-stream"));
    let recording = recording.responded(status, stream);
    let body = UpstreamBody::new(incoming, lease, Arc::clone(shared), recording);
    Attempt::Done(Response::from_parts(head, ResponseBody::Upstream(body)))
}

/// Counts a failed request against the replica and ejects it once the streak
/// is long enough, unless that would push the share of ejected replicas
/// among those passing probes, for any model it serves, past
/// `health.max_ejection_percent`. Evidence from
/// failed requests can come from the requests themselves, so it must never
/// be able to take a whole pool out of rotation.
pub(crate) fn record_failure(shared: &Shared, replica: &Replica) {
    let now = shared.clock.now();
    if !replica.health().record_failure(now, &shared.health) {
        return;
    }
    let _deciding = shared
        .ejecting
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if !ejection_allowed(shared, replica, now) {
        tracing::debug!(replica = %replica.id(), "not ejecting; too much of the pool is already ejected");
        return;
    }
    if let Some(ejection) = replica.health().eject(now, &shared.health) {
        shared.metrics.ejected(replica.id());
        tracing::warn!(
            replica = %replica.id(),
            seconds = ejection.length.as_secs_f64(),
            consecutive = ejection.consecutive,
            "ejecting replica after consecutive failures"
        );
    }
}

fn ejection_allowed(shared: &Shared, replica: &Replica, now: Nanos) -> bool {
    let percent = shared.config.health.max_ejection_percent as usize;
    let fleet = shared.registry.load();
    replica.models().iter().all(|model| {
        let up: Vec<_> = fleet
            .serving(model)
            .iter()
            .filter(|r| r.health().probe_up())
            .collect();
        let ejected = up.iter().filter(|r| r.health().is_ejected(now)).count();
        (ejected + 1) * 100 <= percent * up.len()
    })
}
