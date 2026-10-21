use std::convert::Infallible;
use std::sync::Arc;

use http_body_util::{BodyExt, Full, LengthLimitError, Limited};
use hyper::body::{Bytes, Incoming};
use hyper::header::{self, HeaderMap, HeaderValue};
use hyper::http::request::Parts;
use hyper::{Method, Request, Response};
use inferrail_control::{Lease, Replica};
use inferrail_core::ids::ModelId;
use tokio::time::{Instant, timeout_at};

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

async fn dispatch(
    shared: &Arc<Shared>,
    request: Request<Incoming>,
    id: &HeaderValue,
) -> Result<Response<ResponseBody>, GatewayError> {
    let method = request.method();
    match request.uri().path() {
        "/v1/chat/completions" | "/v1/completions" if method == Method::POST => {
            proxy(shared, request, id).await
        }
        "/v1/chat/completions" | "/v1/completions" => Err(GatewayError::method_not_allowed("POST")),
        "/v1/models" if method == Method::GET => Ok(models::list(shared)),
        "/health" if method == Method::GET => Ok(Response::new(ResponseBody::empty())),
        "/v1/models" | "/health" => Err(GatewayError::method_not_allowed("GET")),
        _ => Err(GatewayError::not_found()),
    }
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
    let body = read_body(&parts.headers, body, shared.config.server.max_body_bytes).await?;
    let (model, body) = resolve_model(shared, body)?;
    if shared.config.logging.log_prompts {
        tracing::debug!(request_id = ?id, body = %String::from_utf8_lossy(&body), "request body");
    }
    let lease = route::select(
        &shared.registry,
        shared.policy(&model)?,
        &model,
        shared.clock.now(),
        &[],
    )?;
    forward(shared, lease, &parts, body, id, deadline).await
}

async fn read_body(
    headers: &HeaderMap,
    body: Incoming,
    limit: usize,
) -> Result<Bytes, GatewayError> {
    let declared = headers
        .get(header::CONTENT_LENGTH)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse::<u64>().ok());
    if declared.is_some_and(|len| len > limit as u64) {
        return Err(GatewayError::too_large(limit));
    }
    match Limited::new(body, limit).collect().await {
        Ok(collected) => Ok(collected.to_bytes()),
        Err(err) if err.is::<LengthLimitError>() => Err(GatewayError::too_large(limit)),
        Err(_) => Err(GatewayError::bad_request("could not read the request body")),
    }
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

async fn forward(
    shared: &Arc<Shared>,
    mut lease: Lease,
    parts: &Parts,
    body: Bytes,
    id: &HeaderValue,
    deadline: Instant,
) -> Result<Response<ResponseBody>, GatewayError> {
    let replica = Arc::clone(lease.replica());
    let request = upstream_request(&replica, parts, body, id)?;
    let response = match timeout_at(deadline, shared.client.request(request)).await {
        Ok(Ok(response)) => response,
        Ok(Err(err)) => {
            record_failure(shared, &replica);
            tracing::debug!(replica = %replica.id(), error = %err, "upstream request failed");
            return Err(GatewayError::bad_gateway(format!(
                "replica {} failed before responding",
                replica.id()
            )));
        }
        Err(_) => {
            record_failure(shared, &replica);
            return Err(GatewayError::timeout(
                "no response before the request timeout",
            ));
        }
    };
    lease.responded();
    if response.status().is_server_error() {
        record_failure(shared, &replica);
    } else {
        replica.health().record_success();
    }
    let (mut head, incoming) = response.into_parts();
    headers::for_downstream(&mut head.headers);
    if let Ok(value) = HeaderValue::from_str(replica.id().as_str()) {
        head.headers.insert(REPLICA, value);
    }
    let body = UpstreamBody::new(incoming, lease, Arc::clone(shared));
    Ok(Response::from_parts(head, ResponseBody::Upstream(body)))
}

pub(crate) fn record_failure(shared: &Shared, replica: &Replica) {
    if let Some(ejection) = replica
        .health()
        .record_failure(shared.clock.now(), &shared.health)
    {
        tracing::warn!(
            replica = %replica.id(),
            seconds = ejection.length.as_secs_f64(),
            consecutive = ejection.consecutive,
            "ejecting replica after consecutive failures"
        );
    }
}
