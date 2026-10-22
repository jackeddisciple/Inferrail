//! Logs and traces. Routing spans use their own target, which the log
//! output never shows and which only an OTLP exporter switches on, so with
//! no endpoint configured they cost the hot path nothing.

use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use hyper::header::{HeaderMap, HeaderName, HeaderValue};
use inferrail_core::config::ObservabilityConfig;
use opentelemetry::global;
use opentelemetry::propagation::{Extractor, Injector};
use opentelemetry::trace::TracerProvider as _;
use opentelemetry_otlp::WithExportConfig;
use opentelemetry_sdk::Resource;
use opentelemetry_sdk::error::OTelSdkResult;
use opentelemetry_sdk::propagation::TraceContextPropagator;
use opentelemetry_sdk::trace::{Sampler, SdkTracerProvider, SpanData, SpanExporter};
use tracing::Span;
use tracing_opentelemetry::OpenTelemetrySpanExt;
use tracing_subscriber::filter::{EnvFilter, LevelFilter, Targets};
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{Layer, Registry};
use url::Url;

/// The target of every span that describes a request's route.
pub const SPAN_TARGET: &str = "inferrail::span";

#[derive(Debug, thiserror::Error)]
pub enum TelemetryError {
    #[error("invalid log filter: {0}")]
    Filter(#[from] tracing_subscriber::filter::ParseError),
    #[error("cannot build the OTLP exporter: {0}")]
    Exporter(#[from] opentelemetry_otlp::ExporterBuildError),
    #[error("cannot install the subscriber: {0}")]
    Install(#[from] tracing_subscriber::util::TryInitError),
}

/// Flushes pending spans when dropped.
#[derive(Debug)]
pub struct Telemetry {
    provider: Option<SdkTracerProvider>,
}

impl Drop for Telemetry {
    fn drop(&mut self) {
        if let Some(provider) = self.provider.take()
            && let Err(err) = provider.shutdown()
        {
            tracing::debug!(error = %err, "trace exporter did not shut down cleanly");
        }
    }
}

/// The log filter for `level` with routing spans left out.
pub fn log_filter(level: &str) -> Result<EnvFilter, TelemetryError> {
    Ok(EnvFilter::try_new(format!("{level},{SPAN_TARGET}=off"))?)
}

/// A layer that exports routing spans through `provider`.
pub fn span_layer<S>(provider: &SdkTracerProvider) -> impl Layer<S> + Send + Sync
where
    S: tracing::Subscriber + for<'a> tracing_subscriber::registry::LookupSpan<'a> + Send + Sync,
{
    tracing_opentelemetry::layer()
        .with_tracer(provider.tracer("inferrail-gateway"))
        .with_filter(Targets::new().with_target(SPAN_TARGET, LevelFilter::TRACE))
}

/// An exporter that says when trace export starts failing and when it
/// recovers, once each, instead of dropping spans in silence.
#[derive(Debug)]
pub struct Reported<E> {
    inner: E,
    failing: AtomicBool,
}

impl<E> Reported<E> {
    pub fn new(inner: E) -> Self {
        Self {
            inner,
            failing: AtomicBool::new(false),
        }
    }
}

impl<E: SpanExporter> SpanExporter for Reported<E> {
    async fn export(&self, batch: Vec<SpanData>) -> OTelSdkResult {
        let spans = batch.len();
        let result = self.inner.export(batch).await;
        match &result {
            Err(err) if !self.failing.swap(true, Ordering::Relaxed) => {
                tracing::warn!(error = %err, spans, "cannot export traces; spans are dropped until the collector answers");
            }
            Ok(()) if self.failing.swap(false, Ordering::Relaxed) => {
                tracing::info!("trace export recovered");
            }
            _ => {}
        }
        result
    }

    fn shutdown_with_timeout(&self, timeout: Duration) -> OTelSdkResult {
        self.inner.shutdown_with_timeout(timeout)
    }

    fn force_flush(&self) -> OTelSdkResult {
        self.inner.force_flush()
    }

    fn set_resource(&mut self, resource: &Resource) {
        self.inner.set_resource(resource);
    }
}

/// The URL spans are posted to. A bare collector address gets the standard
/// `/v1/traces` path, as `OTEL_EXPORTER_OTLP_ENDPOINT` does; a URL with a
/// path is used as given.
pub fn traces_endpoint(endpoint: &Url) -> String {
    if endpoint.path() == "/" {
        let mut url = endpoint.clone();
        url.set_path("/v1/traces");
        return url.into();
    }
    endpoint.as_str().to_owned()
}

/// Installs logging at `level` and, when an OTLP endpoint is configured, a
/// batch exporter for routing spans with W3C trace context propagation.
pub fn init(level: &str, config: &ObservabilityConfig) -> Result<Telemetry, TelemetryError> {
    let provider = match &config.otlp_endpoint {
        Some(endpoint) => {
            let exporter = opentelemetry_otlp::SpanExporter::builder()
                .with_http()
                .with_endpoint(traces_endpoint(endpoint))
                .build()?;
            let sampler = Sampler::ParentBased(Box::new(Sampler::TraceIdRatioBased(
                config.trace_sample_ratio,
            )));
            global::set_text_map_propagator(TraceContextPropagator::new());
            Some(
                SdkTracerProvider::builder()
                    .with_batch_exporter(Reported::new(exporter))
                    .with_sampler(sampler)
                    .with_resource(
                        Resource::builder()
                            .with_service_name(config.service_name.clone())
                            .build(),
                    )
                    .build(),
            )
        }
        None => None,
    };
    let fmt = tracing_subscriber::fmt::layer().with_filter(log_filter(level)?);
    let spans = provider.as_ref().map(span_layer::<Registry>);
    tracing_subscriber::registry()
        .with(spans)
        .with(fmt)
        .try_init()?;
    Ok(Telemetry { provider })
}

struct Headers<'a>(&'a HeaderMap);

impl Extractor for Headers<'_> {
    fn get(&self, key: &str) -> Option<&str> {
        self.0.get(key).and_then(|v| v.to_str().ok())
    }

    fn keys(&self) -> Vec<&str> {
        self.0.keys().map(HeaderName::as_str).collect()
    }
}

struct HeadersMut<'a>(&'a mut HeaderMap);

impl Injector for HeadersMut<'_> {
    fn set(&mut self, key: &str, value: String) {
        if let (Ok(name), Ok(value)) = (
            HeaderName::from_bytes(key.as_bytes()),
            HeaderValue::from_str(&value),
        ) {
            self.0.insert(name, value);
        }
    }
}

/// Continues the client's trace, if it sent one.
pub(crate) fn adopt_parent(span: &Span, headers: &HeaderMap) {
    if span.is_disabled() {
        return;
    }
    let parent = global::get_text_map_propagator(|p| p.extract(&Headers(headers)));
    let _ = span.set_parent(parent);
}

/// Passes the current span's trace on to the engine.
pub(crate) fn propagate(headers: &mut HeaderMap) {
    let span = Span::current();
    if span.is_disabled() {
        return;
    }
    let context = span.context();
    global::get_text_map_propagator(|p| p.inject_context(&context, &mut HeadersMut(headers)));
}

#[cfg(test)]
mod tests {
    use super::*;
    use opentelemetry_sdk::error::OTelSdkError;
    use std::sync::{Arc, Mutex};

    #[derive(Debug, Clone, Default)]
    struct Flaky(Arc<Mutex<Vec<bool>>>);

    impl SpanExporter for Flaky {
        async fn export(&self, _batch: Vec<SpanData>) -> OTelSdkResult {
            match self.0.lock().unwrap().remove(0) {
                true => Ok(()),
                false => Err(OTelSdkError::InternalFailure("refused".into())),
            }
        }
    }

    #[derive(Clone, Default)]
    struct Captured(Arc<Mutex<Vec<u8>>>);

    impl std::io::Write for Captured {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(buf);
            Ok(buf.len())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    #[tokio::test]
    async fn export_failures_and_recovery_are_logged_once_each() {
        let log = Captured::default();
        let writer = log.clone();
        let subscriber = tracing_subscriber::fmt()
            .with_writer(move || writer.clone())
            .with_ansi(false)
            .finish();
        let _guard = tracing::subscriber::set_default(subscriber);
        let script = Flaky::default();
        *script.0.lock().unwrap() = vec![true, false, false, false, true, true, false];
        let reported = Reported::new(script);
        let mut results = Vec::new();
        for _ in 0..7 {
            results.push(reported.export(Vec::new()).await.is_ok());
        }
        assert_eq!(results, [true, false, false, false, true, true, false]);
        let text = String::from_utf8(log.0.lock().unwrap().clone()).unwrap();
        assert_eq!(text.matches("cannot export traces").count(), 2, "{text}");
        assert_eq!(text.matches("trace export recovered").count(), 1, "{text}");
    }

    #[test]
    fn bare_collector_addresses_get_the_traces_path() {
        for (given, used) in [
            ("http://collector:4318", "http://collector:4318/v1/traces"),
            ("http://collector:4318/", "http://collector:4318/v1/traces"),
            (
                "http://collector:4318/v1/traces",
                "http://collector:4318/v1/traces",
            ),
            (
                "https://otel.example/custom/path",
                "https://otel.example/custom/path",
            ),
        ] {
            assert_eq!(traces_endpoint(&given.parse().unwrap()), used, "{given}");
        }
    }
}
