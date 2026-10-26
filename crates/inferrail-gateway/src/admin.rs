//! The admin listener: Prometheus metrics, a liveness check and routing
//! explanations, kept off the port clients use.

use std::sync::Arc;

use axum::Router;
use axum::body::Bytes;
use axum::extract::DefaultBodyLimit;
use axum::http::{HeaderMap, header};
use axum::response::IntoResponse;
use axum::routing::{get, post};
use inferrail_tokenize::Endpoint;

use crate::{Shared, explain};

pub(crate) const CONTENT_TYPE: &str = "application/openmetrics-text; version=1.0.0; charset=utf-8";

pub(crate) fn router(shared: Arc<Shared>) -> Router {
    let limit = shared.config.server.max_body_bytes;
    let explains = |endpoint: Endpoint| {
        let shared = Arc::clone(&shared);
        post(move |headers: HeaderMap, body: Bytes| {
            explain::handle(Arc::clone(&shared), endpoint, headers, body)
        })
    };
    let metrics = Arc::clone(&shared);
    Router::new()
        .route(
            "/metrics",
            get(move || async move {
                (
                    [(header::CONTENT_TYPE, CONTENT_TYPE)],
                    metrics.metrics.render(),
                )
                    .into_response()
            }),
        )
        .route("/health", get(|| async {}))
        .route(
            "/admin/explain/v1/chat/completions",
            explains(Endpoint::Chat),
        )
        .route(
            "/admin/explain/v1/completions",
            explains(Endpoint::Completion),
        )
        .layer(DefaultBodyLimit::max(limit))
}
