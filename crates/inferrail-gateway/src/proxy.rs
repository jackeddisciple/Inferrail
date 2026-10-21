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

use crate::body::{ResponseBody, UpstreamBody};
use crate::errors::GatewayError;
use crate::headers::{self, REPLICA};
use crate::inspect::{InspectError, inspect};
use crate::request_id::REQUEST_ID;
use crate::{Shared, models, route};

pub(crate) async fn handle(
    shared: Arc<Shared>,
    request: Request<Incoming>,
) -> Result<Response<ResponseBody>, Infallible> {
    let id = shared.ids.resolve(request.headers());
    let version = request.version();
    let mut response = match dispatch(&shared, request, &id).await {
        Ok(response) => response,
        Err(err) => error_response(&err),
    };
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
    let local = match path {
        "/v1/chat/completions" | "/v1/completions" if method == Method::POST => {
            return Route::Proxy;
        }
        "/v1/chat/completions" | "/v1/completions" => Err(GatewayError::method_not_allowed("POST")),
        "/v1/models" if method == Method::GET => Ok(Local::Models),
        "/health" if method == Method::GET => Ok(Local::Health),
        "/v1/models" | "/health" => Err(GatewayError::method_not_allowed("GET")),
        _ => Err(GatewayError::not_found()),
    };
    Route::Local(local)
}

async fn dispatch(
    shared: &Arc<Shared>,
    request: Request<Incoming>,
    id: &HeaderValue,
) -> Result<Response<ResponseBody>, GatewayError> {
    match route_of(request.method(), request.uri().path()) {
        Route::Proxy => proxy(shared, request, id).await,
        Route::Local(local) => {
            drain(request.into_body(), shared.config.server.max_body_bytes).await;
            match local? {
                Local::Models => Ok(models::list(shared)),
                Local::Health => Ok(Response::new(ResponseBody::empty())),
            }
        }
    }
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

async fn proxy(
    shared: &Arc<Shared>,
    request: Request<Incoming>,
    id: &HeaderValue,
) -> Result<Response<ResponseBody>, GatewayError> {
    let deadline = Instant::now() + shared.config.server.request_timeout;
    let (parts, body) = request.into_parts();
    let limit = shared.config.server.max_body_bytes;
    let body = timeout_at(deadline, read_body(&parts.headers, body, limit))
        .await
        .map_err(|_| GatewayError::slow_body())??;
    let (model, body) = resolve_model(shared, body)?;
    if shared.config.logging.log_prompts {
        tracing::debug!(request_id = ?id, body = %String::from_utf8_lossy(&body), "request body");
    }
    let policy = shared.policy(&model)?;
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
        let now = shared.clock.now();
        let lease = match route::select(&shared.registry, policy, &model, now, &tried) {
            Ok(lease) => lease,
            Err(err) => return last.unwrap_or(Err(err)),
        };
        tried.push(lease.replica().id().clone());
        match forward(shared, lease, &parts, body.clone(), id, deadline).await {
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

fn backoff(base: Duration, attempt: u32) -> Duration {
    base.saturating_mul(1 << attempt.saturating_sub(1).min(16))
}

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
        drain(body, limit).await;
        return Err(GatewayError::too_large(limit));
    }
    let mut chunks: Vec<Bytes> = Vec::new();
    let mut total = 0;
    while let Some(frame) = body.frame().await {
        let frame =
            frame.map_err(|_| GatewayError::bad_request("could not read the request body"))?;
        if let Ok(data) = frame.into_data() {
            total += data.len();
            if total > limit {
                drain(body, limit).await;
                return Err(GatewayError::too_large(limit));
            }
            chunks.push(data);
        }
    }
    Ok(match chunks.len() {
        0 => Bytes::new(),
        1 => chunks.swap_remove(0),
        _ => Bytes::from(chunks.concat()),
    })
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
        tracing::debug!(replica = %replica.id(), "replica failed its health checks while the request waited");
        return Attempt::Retry(Err(GatewayError::bad_gateway(format!(
            "replica {} failed its health checks before responding",
            replica.id()
        ))));
    };
    let response = match answer {
        Ok(Ok(response)) => response,
        Ok(Err(err)) => {
            record_failure(shared, &replica);
            tracing::debug!(replica = %replica.id(), error = %err, "upstream request failed");
            return Attempt::Retry(Err(GatewayError::bad_gateway(format!(
                "replica {} failed before responding",
                replica.id()
            ))));
        }
        Err(_) => {
            record_failure(shared, &replica);
            return Attempt::Fatal(GatewayError::timeout(
                "no response before the request timeout",
            ));
        }
    };
    lease.responded();
    let status = response.status();
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
    let body = UpstreamBody::new(incoming, lease, Arc::clone(shared));
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
