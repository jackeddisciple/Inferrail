//! Routing spans as an OTLP backend would receive them.

mod common;

use std::io::Write;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use common::*;
use inferrail_gateway::telemetry::{self, log_filter};
use opentelemetry::trace::SpanKind;
use opentelemetry_sdk::propagation::TraceContextPropagator;
use opentelemetry_sdk::trace::{InMemorySpanExporter, SdkTracerProvider, SpanData};
use tracing::subscriber::DefaultGuard;
use tracing_subscriber::Layer;
use tracing_subscriber::layer::SubscriberExt;

const TRACEPARENT: &str = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01";

struct Traced {
    exporter: InMemorySpanExporter,
    provider: SdkTracerProvider,
    _guard: DefaultGuard,
}

impl Traced {
    fn install() -> Self {
        let exporter = InMemorySpanExporter::default();
        let provider = SdkTracerProvider::builder()
            .with_simple_exporter(exporter.clone())
            .build();
        opentelemetry::global::set_text_map_propagator(TraceContextPropagator::new());
        let subscriber = tracing_subscriber::registry().with(telemetry::span_layer(&provider));
        Self {
            _guard: tracing::subscriber::set_default(subscriber),
            exporter,
            provider,
        }
    }

    fn spans(&self) -> Vec<SpanData> {
        self.provider.force_flush().expect("flush");
        self.exporter.get_finished_spans().expect("spans")
    }
}

fn attr(span: &SpanData, key: &str) -> Option<String> {
    span.attributes
        .iter()
        .find(|kv| kv.key.as_str() == key)
        .map(|kv| kv.value.to_string())
}

fn named<'a>(spans: &'a [SpanData], name: &str) -> Vec<&'a SpanData> {
    spans
        .iter()
        .filter(|s| s.name == name)
        .filter(|s| name != "request" || attr(s, "url.path").as_deref() != Some("/health"))
        .collect()
}

#[tokio::test(flavor = "current_thread")]
async fn spans_carry_routing_attributes() {
    let traced = Traced::install();
    let sims = [
        Upstream::zero(&[MODEL]).await,
        Upstream::zero(&[MODEL]).await,
    ];
    let mut config = fleet(
        &urls(&[&sims[0], &sims[1]]),
        &format!("{FAST_HEALTH}routing:\n  default_policy: round_robin\nload:\n  max_age: 5s\n"),
    );
    config.models[0].tokenizer = Some("builtin:sim".to_owned());
    let gw = TestGateway::start(config).await;
    wait_until(Duration::from_secs(2), || {
        ["r0", "r1"]
            .iter()
            .all(|id| gw.replica(id).engine_load().latest().is_some())
    })
    .await;
    let events = gw.stream(&streaming(chat("trace me", 4))).await;
    assert!(!events.is_empty());
    gw.wait_idle().await;

    let spans = traced.spans();
    let [request] = named(&spans, "request")[..] else {
        panic!("one request span: {spans:#?}")
    };
    let [route] = named(&spans, "route")[..] else {
        panic!("one route span")
    };
    let [upstream] = named(&spans, "upstream")[..] else {
        panic!("one upstream span")
    };
    assert_eq!(request.span_kind, SpanKind::Server);
    assert_eq!(
        attr(request, "http.request.method").as_deref(),
        Some("POST")
    );
    assert_eq!(
        attr(request, "url.path").as_deref(),
        Some("/v1/chat/completions")
    );
    assert_eq!(
        attr(request, "http.response.status_code").as_deref(),
        Some("200")
    );
    assert_eq!(
        attr(request, "gen_ai.request.model").as_deref(),
        Some(MODEL)
    );
    assert_eq!(attr(request, "inferrail.attempts").as_deref(), Some("1"));
    assert_eq!(
        attr(request, "inferrail.request_id").map(|id| id.len()),
        Some(32)
    );

    let trace = request.span_context.trace_id();
    for child in [route, upstream] {
        assert_eq!(child.span_context.trace_id(), trace);
        assert_eq!(child.parent_span_id, request.span_context.span_id());
    }
    assert_eq!(
        attr(route, "inferrail.policy").as_deref(),
        Some("round_robin")
    );
    assert_eq!(attr(route, "inferrail.attempt").as_deref(), Some("0"));
    assert_eq!(attr(route, "inferrail.candidates").as_deref(), Some("2"));
    let replica = attr(route, "inferrail.replica").unwrap();
    assert!(replica == "r0" || replica == "r1", "{replica}");
    assert_eq!(
        attr(route, "inferrail.replica.in_flight").as_deref(),
        Some("1")
    );
    assert!(
        attr(route, "inferrail.decision_ns")
            .unwrap()
            .parse::<u64>()
            .unwrap()
            > 0
    );
    assert_eq!(
        attr(route, "inferrail.engine.running").as_deref(),
        Some("0")
    );
    assert!(attr(route, "inferrail.engine.kv_usage").is_some());
    let prompt: u64 = attr(route, "inferrail.prompt_tokens")
        .unwrap()
        .parse()
        .unwrap();
    assert!(prompt > 4, "{prompt}");
    assert_eq!(attr(route, "inferrail.cached_tokens").as_deref(), Some("0"));
    assert!(
        attr(route, "inferrail.load.age_ms")
            .unwrap()
            .parse::<f64>()
            .unwrap()
            < 5000.0
    );

    assert_eq!(upstream.span_kind, SpanKind::Client);
    assert_eq!(attr(upstream, "inferrail.replica"), Some(replica.clone()));
    let index: usize = replica[1..].parse().unwrap();
    assert_eq!(attr(upstream, "server.address"), Some(sims[index].url()));
    assert_eq!(
        attr(upstream, "http.response.status_code").as_deref(),
        Some("200")
    );
    assert!(request.end_time >= upstream.end_time);
}

#[tokio::test(flavor = "current_thread")]
async fn retries_show_as_attempts_and_client_traces_are_continued() {
    let traced = Traced::install();
    let failing = Upstream::zero(&[MODEL]).await;
    failing.set_request_fault(Fault::Status(503));
    let raw = RawUpstream::spawn().await;
    let gw = TestGateway::start(fleet(
        &[failing.url(), raw.url()],
        &format!("{FAST_HEALTH}routing:\n  default_policy: round_robin\n"),
    ))
    .await;
    for _ in 0..2 {
        let response = gw
            .client
            .post(gw.url("/v1/chat/completions"))
            .header("traceparent", TRACEPARENT)
            .json(&chat("hello", 1))
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), 200);
        response.bytes().await.unwrap();
    }
    gw.wait_idle().await;

    let spans = traced.spans();
    let requests = named(&spans, "request");
    assert_eq!(requests.len(), 2);
    for request in &requests {
        assert_eq!(
            request.span_context.trace_id().to_string(),
            "4bf92f3577b34da6a3ce929d0e0e4736"
        );
        assert_eq!(request.parent_span_id.to_string(), "00f067aa0ba902b7");
    }
    let retried = requests
        .iter()
        .find(|r| attr(r, "inferrail.attempts").as_deref() == Some("2"))
        .expect("round robin sends one of two requests to the failing replica first");
    let children = |name: &str| -> Vec<&SpanData> {
        named(&spans, name)
            .into_iter()
            .filter(|s| s.parent_span_id == retried.span_context.span_id())
            .collect()
    };
    let attempts: Vec<_> = children("route")
        .iter()
        .filter_map(|s| attr(s, "inferrail.attempt"))
        .collect();
    assert_eq!(attempts, ["0", "1"]);
    assert!(
        children("route")
            .iter()
            .all(|s| attr(s, "inferrail.prompt_tokens").is_none()),
        "a model without a tokenizer has no prompt attributes"
    );
    let upstreams = children("upstream");
    let statuses: Vec<_> = upstreams
        .iter()
        .filter_map(|s| attr(s, "http.response.status_code"))
        .collect();
    assert_eq!(statuses, ["503", "200"]);

    let sent: Vec<String> = raw
        .take()
        .iter()
        .map(|r| r.headers["traceparent"].to_str().unwrap().to_owned())
        .collect();
    let expected = format!(
        "00-4bf92f3577b34da6a3ce929d0e0e4736-{}-01",
        upstreams[1].span_context.span_id()
    );
    assert!(sent.contains(&expected), "{sent:?} lacks {expected}");
}

#[derive(Clone, Default)]
struct Captured(Arc<Mutex<Vec<u8>>>);

impl Write for Captured {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0.lock().expect("buffer").extend_from_slice(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

#[tokio::test(flavor = "current_thread")]
async fn without_an_exporter_logs_stay_clean_and_trace_headers_pass_through() {
    let captured = Captured::default();
    let writer = captured.clone();
    let fmt = tracing_subscriber::fmt::layer()
        .with_writer(move || writer.clone())
        .with_filter(log_filter("trace").unwrap());
    let _guard = tracing::subscriber::set_default(tracing_subscriber::registry().with(fmt));
    let raw = RawUpstream::spawn().await;
    let gw = TestGateway::start(fleet(&[raw.url()], FAST_HEALTH)).await;
    let response = gw
        .client
        .post(gw.url("/v1/chat/completions"))
        .header("traceparent", TRACEPARENT)
        .json(&chat("quiet", 1))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    response.bytes().await.unwrap();
    gw.wait_idle().await;
    assert_eq!(raw.take()[0].headers["traceparent"], TRACEPARENT);
    let log = String::from_utf8(captured.0.lock().unwrap().clone()).unwrap();
    assert!(!log.is_empty());
    for noise in ["inferrail.policy", "route{", "upstream{", "request{"] {
        assert!(!log.contains(noise), "{noise} in {log}");
    }
}
