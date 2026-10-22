//! Logs and traces. Routing spans use their own target, which the log
//! output never shows and which only an OTLP exporter switches on, so with
//! no endpoint configured they cost the hot path nothing.

use hyper::header::{HeaderMap, HeaderName, HeaderValue};
use inferrail_core::config::ObservabilityConfig;
use opentelemetry::global;
use opentelemetry::propagation::{Extractor, Injector};
use opentelemetry::trace::TracerProvider as _;
use opentelemetry_otlp::WithExportConfig;
use opentelemetry_sdk::Resource;
use opentelemetry_sdk::propagation::TraceContextPropagator;
use opentelemetry_sdk::trace::{Sampler, SdkTracerProvider};
use tracing::Span;
use tracing_opentelemetry::OpenTelemetrySpanExt;
use tracing_subscriber::filter::{EnvFilter, LevelFilter, Targets};
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{Layer, Registry};

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

/// Installs logging at `level` and, when an OTLP endpoint is configured, a
/// batch exporter for routing spans with W3C trace context propagation.
pub fn init(level: &str, config: &ObservabilityConfig) -> Result<Telemetry, TelemetryError> {
    let provider = match &config.otlp_endpoint {
        Some(endpoint) => {
            let exporter = opentelemetry_otlp::SpanExporter::builder()
                .with_http()
                .with_endpoint(endpoint.as_str())
                .build()?;
            let sampler = Sampler::ParentBased(Box::new(Sampler::TraceIdRatioBased(
                config.trace_sample_ratio,
            )));
            global::set_text_map_propagator(TraceContextPropagator::new());
            Some(
                SdkTracerProvider::builder()
                    .with_batch_exporter(exporter)
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
