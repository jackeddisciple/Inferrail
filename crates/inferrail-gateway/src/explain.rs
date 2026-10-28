//! `/admin/explain`: what the model's policy would do with a request now,
//! and why, without routing it or changing any state.

use std::sync::Arc;

use axum::body::Bytes;
use axum::extract::Request;
use axum::http::{HeaderMap, HeaderValue, StatusCode, header};
use axum::response::{IntoResponse, Response};
use inferrail_core::config::PolicyKind;
use inferrail_core::ids::{ModelId, ReplicaId};
use inferrail_policy::{Estimate, Exclusion, Guarded, HotPrefix, Steer, Term};
use inferrail_tokenize::Endpoint;
use serde::Serialize;
use tokio::time::{Instant, timeout};

use crate::Shared;
use crate::errors::GatewayError;
use crate::proxy::{Resolved, estimate_tokens, read_body, resolve_model, session_key};
use crate::route;

#[derive(Debug, Serialize)]
struct Report {
    model: ModelId,
    policy: &'static str,
    rule: &'static str,
    prompt_tokens: u32,
    tokenized: bool,
    /// Why the KV index made no prediction, when it made none.
    unpredicted: Option<&'static str>,
    blocks: usize,
    session: bool,
    /// The replica the policy would pick, unless chance decides.
    winner: Option<ReplicaId>,
    decided_by: Option<Term>,
    steer: Option<Steer>,
    guarded: Guarded,
    hot_prefix: Option<HotPrefix>,
    replicas: Vec<Row>,
}

#[derive(Debug, Serialize)]
struct Row {
    replica: ReplicaId,
    offered: bool,
    /// Why it was not offered: unhealthy, ejected or full.
    unavailable: Option<&'static str>,
    in_flight: u32,
    queued_secs: f64,
    running: u32,
    kv_usage: Option<f64>,
    cached_tokens: Option<u32>,
    excluded: Option<Exclusion>,
    owner: bool,
    score: Option<f64>,
    estimate: Option<Estimate>,
}

/// Reads the body as the client port does, within the request timeout and
/// its size limit, draining what is left of one too large so the client
/// gets the error rather than a reset.
pub(crate) async fn handle(shared: Arc<Shared>, endpoint: Endpoint, request: Request) -> Response {
    let (parts, body) = request.into_parts();
    let server = &shared.config.server;
    let read = timeout(
        server.request_timeout,
        read_body(&parts.headers, body, server.max_body_bytes),
    );
    let explained = match read.await {
        Ok(Ok(body)) => match asked_policy(parts.uri.query()) {
            Ok(asked) => report(&shared, endpoint, &parts.headers, body, asked).await,
            Err(err) => Err(err),
        },
        Ok(Err(err)) => Err(err),
        Err(_) => Err(GatewayError::slow_body()),
    };
    let (status, body) = match explained {
        Ok(report) => (
            StatusCode::OK,
            serde_json::to_vec_pretty(&report).unwrap_or_default(),
        ),
        Err(err) => (err.status(), err.to_json()),
    };
    (
        status,
        [(
            header::CONTENT_TYPE,
            HeaderValue::from_static("application/json"),
        )],
        body,
    )
        .into_response()
}

/// The policy named by `?policy=`, to explain one arm of a split or a
/// shadow policy rather than the one the request would get.
fn asked_policy(query: Option<&str>) -> Result<Option<PolicyKind>, GatewayError> {
    let Some(name) = query
        .into_iter()
        .flat_map(|q| q.split('&'))
        .find_map(|pair| pair.strip_prefix("policy="))
    else {
        return Ok(None);
    };
    PolicyKind::ALL
        .into_iter()
        .find(|p| p.as_str() == name)
        .map(Some)
        .ok_or_else(|| GatewayError::bad_param(format!("unknown policy {name:?}"), "policy"))
}

async fn report(
    shared: &Shared,
    endpoint: Endpoint,
    headers: &HeaderMap,
    body: Bytes,
    asked: Option<PolicyKind>,
) -> Result<Report, GatewayError> {
    let live = shared.live();
    let Resolved { model, body, user } = resolve_model(shared, &live, body)?;
    let session = session_key(headers, user.as_deref());
    let routing = live.routing(&model)?;
    let built = match asked {
        Some(kind) => routing.find(kind).ok_or_else(|| {
            GatewayError::bad_param(format!("`{model}` is not routed by {kind}"), "policy")
        })?,
        None => session.map_or_else(|| routing.main(), |key| routing.pick(key)),
    };
    let policy = built.policy.as_ref();
    let (prepared, _) = shared.prompts.prepare(&model, endpoint, &body).await;
    let (prompt, skipped) = match prepared {
        Ok(prompt) => (Some(prompt), None),
        Err(skip) => (None, Some(skip.label())),
    };
    let unpredicted =
        skipped.or((prompt.is_some() && shared.index.is_none()).then_some("index_off"));
    let tokens = prompt
        .as_ref()
        .map_or_else(|| estimate_tokens(body.len()), |p| p.tokens);
    let request = route::Request {
        model: &model,
        tokens,
        prompt: prompt.as_ref(),
        index: shared.index.as_deref(),
        session,
        cost: live.cost(&model),
        max_age: shared.config.load.max_age,
        explain: true,
    };
    let now = shared.clock.now();
    let fleet = shared.registry.load();
    let (considered, candidates, predicted, explanation) =
        route::explain(fleet.serving(&model), policy, request, now);
    let instant = Instant::now();
    let replicas = considered
        .iter()
        .map(|seen| {
            let scored = seen.offered.map(|i| explanation.candidates[i]);
            let cached = seen.offered.and_then(|i| predicted.as_ref().map(|p| p[i]));
            let view = seen.offered.map_or_else(
                || route::candidate(seen.replica, 0, now, instant, request.max_age),
                |i| candidates[i],
            );
            Row {
                replica: seen.replica.id().clone(),
                offered: seen.offered.is_some(),
                unavailable: seen.unavailable,
                in_flight: view.in_flight,
                queued_secs: view.queued_secs,
                running: view.running,
                kv_usage: view.kv_usage,
                cached_tokens: cached,
                excluded: scored.and_then(|s| s.excluded),
                owner: scored.is_some_and(|s| s.owner),
                score: scored.and_then(|s| s.score),
                estimate: scored.and_then(|s| s.estimate),
            }
        })
        .collect();
    Ok(Report {
        policy: policy.kind().as_str(),
        rule: explanation.rule,
        prompt_tokens: tokens,
        tokenized: prompt.is_some(),
        unpredicted,
        blocks: prompt.as_ref().map_or(0, |p| p.hashes.len()),
        session: request.session.is_some(),
        winner: explanation.winner.map(|i| candidates[i].id.clone()),
        decided_by: explanation.decided_by,
        steer: explanation.steer,
        guarded: explanation.guarded,
        hot_prefix: explanation.hot_prefix,
        replicas,
        model,
    })
}
