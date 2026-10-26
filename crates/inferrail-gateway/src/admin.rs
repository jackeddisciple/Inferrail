//! The admin listener: Prometheus metrics, a liveness check and routing
//! explanations, kept off the port clients use.

use std::sync::Arc;

use axum::Router;
use axum::extract::Request;
use axum::http::header;
use axum::response::IntoResponse;
use axum::routing::{get, post};
use inferrail_tokenize::Endpoint;

use crate::{Shared, explain};

pub(crate) const CONTENT_TYPE: &str = "application/openmetrics-text; version=1.0.0; charset=utf-8";

pub(crate) fn router(shared: Arc<Shared>) -> Router {
    let explains = |endpoint: Endpoint| {
        let shared = Arc::clone(&shared);
        post(move |request: Request| explain::handle(Arc::clone(&shared), endpoint, request))
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
}
