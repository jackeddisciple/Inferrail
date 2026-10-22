//! The admin listener: Prometheus metrics and a liveness check, kept off
//! the port clients use.

use std::sync::Arc;

use axum::Router;
use axum::http::header;
use axum::response::IntoResponse;
use axum::routing::get;

use crate::Shared;

pub(crate) const CONTENT_TYPE: &str = "application/openmetrics-text; version=1.0.0; charset=utf-8";

pub(crate) fn router(shared: Arc<Shared>) -> Router {
    Router::new()
        .route(
            "/metrics",
            get(move || async move {
                (
                    [(header::CONTENT_TYPE, CONTENT_TYPE)],
                    shared.metrics.render(),
                )
                    .into_response()
            }),
        )
        .route("/health", get(|| async {}))
}
