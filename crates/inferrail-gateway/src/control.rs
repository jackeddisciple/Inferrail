//! The admin API: the config in effect and live changes to it, a reload of
//! the config file, and what the gateway knows of the fleet and its cache.

use std::sync::Arc;

use axum::body::Bytes;
use axum::extract::{Request, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use inferrail_control::Origin;
use inferrail_core::config::Config;
use inferrail_core::ids::{ModelId, ReplicaId};
use serde::Serialize;
use serde_json::{Value, json};
use tokio::time::Instant;

use crate::proxy::read_body;
use crate::{Gateway, ReloadError, Reloaded, Shared};

const MAX_PATCH: usize = 1 << 20;

fn reply(status: StatusCode, body: &impl Serialize) -> Response {
    (
        status,
        [("content-type", "application/json")],
        serde_json::to_vec_pretty(body).unwrap_or_default(),
    )
        .into_response()
}

fn failure(
    status: StatusCode,
    kind: &str,
    message: impl Into<String>,
    paths: &[String],
) -> Response {
    let mut error = json!({"message": message.into(), "type": kind});
    if !paths.is_empty() {
        error["paths"] = json!(paths);
    }
    reply(status, &json!({ "error": error }))
}

fn reloaded(result: Result<Reloaded, ReloadError>) -> Response {
    match result {
        Ok(done) => reply(
            StatusCode::OK,
            &json!({"generation": done.generation, "changed": done.changed}),
        ),
        Err(ReloadError::Restart(paths)) => failure(
            StatusCode::CONFLICT,
            "restart_required",
            "only a restart applies changes to these sections; nothing was applied",
            &paths,
        ),
        Err(err @ (ReloadError::Invalid(_) | ReloadError::Https(_))) => failure(
            StatusCode::BAD_REQUEST,
            "invalid_config",
            err.to_string(),
            &[],
        ),
        Err(err @ ReloadError::Load(_)) => failure(
            StatusCode::BAD_REQUEST,
            "unreadable_config",
            err.to_string(),
            &[],
        ),
    }
}

pub(crate) async fn config(State(shared): State<Arc<Shared>>) -> Response {
    let live = shared.live();
    reply(
        StatusCode::OK,
        &json!({"generation": live.generation, "config": live.config.as_ref()}),
    )
}

/// RFC 7386: objects merge key by key, `null` removes a key, anything else
/// replaces what was there.
fn merge(target: &mut Value, patch: Value) {
    match patch {
        Value::Object(changes) => {
            if !target.is_object() {
                *target = Value::Object(serde_json::Map::new());
            }
            if let Value::Object(fields) = target {
                for (key, value) in changes {
                    if value.is_null() {
                        fields.remove(&key);
                    } else {
                        merge(fields.entry(key).or_insert(Value::Null), value);
                    }
                }
            }
        }
        other => *target = other,
    }
}

/// Applies a JSON merge patch to the config in effect, as a reload does.
/// The file on disk is left alone; its next edit replaces the patch.
pub(crate) async fn patch(State(shared): State<Arc<Shared>>, request: Request) -> Response {
    let (parts, body) = request.into_parts();
    let body: Bytes = match read_body(&parts.headers, body, MAX_PATCH).await {
        Ok(body) => body,
        Err(err) => return failure(err.status(), "bad_request", err.message(), &[]),
    };
    let patch: Value = match serde_json::from_slice(&body) {
        Ok(patch) => patch,
        Err(err) => {
            shared.metrics.reloaded("rejected");
            return failure(
                StatusCode::BAD_REQUEST,
                "bad_request",
                format!("the body is not JSON: {err}"),
                &[],
            );
        }
    };
    let current = shared.live();
    let mut merged = match serde_json::to_value(current.config.as_ref()) {
        Ok(merged) => merged,
        Err(err) => {
            return failure(
                StatusCode::INTERNAL_SERVER_ERROR,
                "internal",
                err.to_string(),
                &[],
            );
        }
    };
    merge(&mut merged, patch);
    let next: Config = match serde_json::from_value(merged) {
        Ok(next) => next,
        Err(err) => {
            shared.metrics.reloaded("rejected");
            return failure(
                StatusCode::BAD_REQUEST,
                "invalid_config",
                format!("the patched config does not parse: {err}"),
                &[],
            );
        }
    };
    reloaded(Gateway::from_shared(shared).reload(next))
}

/// Rereads and applies the config file the gateway was started from.
pub(crate) async fn reload(State(shared): State<Arc<Shared>>) -> Response {
    let Some(path) = shared.source() else {
        return failure(
            StatusCode::NOT_FOUND,
            "no_config_file",
            "this gateway was not started from a config file",
            &[],
        );
    };
    let gateway = Gateway::from_shared(shared);
    let done = tokio::task::spawn_blocking(move || gateway.reload_from(&path)).await;
    match done {
        Ok(result) => reloaded(result),
        Err(err) => failure(
            StatusCode::INTERNAL_SERVER_ERROR,
            "internal",
            err.to_string(),
            &[],
        ),
    }
}

#[derive(Debug, Serialize)]
struct IndexRow {
    replica: ReplicaId,
    blocks: usize,
    capacity: usize,
    fill: f64,
}

pub(crate) async fn index(State(shared): State<Arc<Shared>>) -> Response {
    let Some(index) = &shared.index else {
        return reply(
            StatusCode::OK,
            &json!({"enabled": false, "mode": null, "entries": 0, "bytes": 0, "replicas": []}),
        );
    };
    let stats = index.stats();
    let replicas: Vec<IndexRow> = stats
        .replicas
        .into_iter()
        .map(|r| IndexRow {
            fill: if r.capacity == 0 {
                0.0
            } else {
                r.blocks as f64 / r.capacity as f64
            },
            replica: r.replica,
            blocks: r.blocks,
            capacity: r.capacity,
        })
        .collect();
    reply(
        StatusCode::OK,
        &json!({
            "enabled": true,
            "mode": "approximate",
            "entries": stats.entries,
            "bytes": stats.bytes,
            "bytes_per_entry": if stats.entries == 0 { 0.0 } else { stats.bytes as f64 / stats.entries as f64 },
            "replicas": replicas,
        }),
    )
}

#[derive(Debug, Serialize)]
struct EngineRow {
    running: u32,
    waiting: u32,
    kv_usage: f64,
    kv_capacity_tokens: Option<u64>,
    age_ms: f64,
    fresh: bool,
}

#[derive(Debug, Serialize)]
struct ReplicaRow {
    id: ReplicaId,
    url: String,
    origin: String,
    models: Vec<ModelId>,
    routable: bool,
    probe_up: bool,
    ejected: bool,
    consecutive_failures: u32,
    in_flight: u32,
    pending: u32,
    prefilling: u32,
    queued_secs: f64,
    max_concurrency: Option<u32>,
    engine: Option<EngineRow>,
}

pub(crate) async fn replicas(State(shared): State<Arc<Shared>>) -> Response {
    let fleet = shared.registry.load();
    let now = shared.clock.now();
    let instant = Instant::now();
    let max_age = shared.config.load.max_age;
    let rows: Vec<ReplicaRow> = fleet
        .replicas()
        .iter()
        .map(|replica| {
            let health = replica.health();
            ReplicaRow {
                id: replica.id().clone(),
                url: replica.endpoint().url().to_string(),
                origin: match &replica.spec().origin {
                    Origin::Static => "static".to_owned(),
                    Origin::Dns(group) => format!("dns:{group}"),
                },
                models: replica.models().to_vec(),
                routable: replica.is_routable(now),
                probe_up: health.probe_up(),
                ejected: health.is_ejected(now),
                consecutive_failures: health.consecutive_failures(),
                in_flight: replica.in_flight(),
                pending: replica.pending(),
                prefilling: replica.prefilling(),
                queued_secs: replica.queued(now).as_secs_f64(),
                max_concurrency: replica.limits().max_concurrency,
                engine: replica.engine_load().latest().map(|s| EngineRow {
                    running: s.load.running,
                    waiting: s.load.waiting,
                    kv_usage: s.load.kv_usage,
                    kv_capacity_tokens: s.load.kv_capacity_tokens,
                    age_ms: s.age(instant).as_secs_f64() * 1e3,
                    fresh: s.age(instant) <= max_age,
                }),
            }
        })
        .collect();
    reply(StatusCode::OK, &json!({"replicas": rows}))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merge_patches_follow_rfc_7386() {
        let mut doc = json!({"a": {"b": 1, "c": [1, 2]}, "d": "x"});
        merge(
            &mut doc,
            json!({"a": {"b": 2, "c": [3]}, "d": null, "e": {"f": true}}),
        );
        assert_eq!(doc, json!({"a": {"b": 2, "c": [3]}, "e": {"f": true}}));
        let mut scalar = json!(1);
        merge(&mut scalar, json!({"x": 1}));
        assert_eq!(scalar, json!({"x": 1}));
        merge(&mut scalar, json!([1]));
        assert_eq!(scalar, json!([1]));
    }
}
