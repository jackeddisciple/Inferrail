//! Gateway overhead with every observability feature on: metrics, the
//! decision log at full sampling and trace export to an unreachable OTLP
//! endpoint. Its own test binary, because tracing installs a global
//! subscriber. Ignored by default; run with `make stress`.

mod common;

use std::time::Duration;

use common::*;
use inferrail_core::config::ObservabilityConfig;
use inferrail_gateway::telemetry;

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
#[ignore = "stress"]
async fn overhead_with_metrics_decision_log_and_tracing_on() {
    let sim = Upstream::zero(&[MODEL]).await;
    let client = reqwest::Client::new();
    let rate = 5_000;
    let total = 15_000;
    let body = |i: u64| chat(&format!("observed {i}"), 4);
    let direct_url = format!("{}/v1/chat/completions", sim.url());

    let plain = TestGateway::start(fleet(&[sim.url()], "")).await;
    let direct = open_loop(&client, direct_url.clone(), rate, total, body).await;
    let off = open_loop(
        &client,
        plain.url("/v1/chat/completions"),
        rate,
        total,
        body,
    )
    .await;
    drop(plain);

    let log = std::env::temp_dir().join(format!("inferrail-stress-{}.jsonl", std::process::id()));
    let mut observability = ObservabilityConfig {
        otlp_endpoint: Some("http://127.0.0.1:9/v1/traces".parse().unwrap()),
        trace_sample_ratio: 1.0,
        ..ObservabilityConfig::default()
    };
    observability.decision_log.path = Some(log.clone());
    observability.decision_log.sample_rate = 1.0;
    let _telemetry = telemetry::init("warn", &observability).expect("telemetry");
    let mut config = fleet(&[sim.url()], "");
    config.observability = observability;
    let observed = TestGateway::start(config).await;
    let direct_again = open_loop(&client, direct_url, rate, total, body).await;
    let on = open_loop(
        &client,
        observed.url("/v1/chat/completions"),
        rate,
        total,
        body,
    )
    .await;
    assert_eq!(
        direct.errors + off.errors + direct_again.errors + on.errors,
        0
    );

    let added = |gateway: Run, direct: Run| {
        let [g50, g99, g999] = summary(gateway.service);
        let [d50, d99, d999] = summary(direct.service);
        [
            g50.saturating_sub(d50),
            g99.saturating_sub(d99),
            g999.saturating_sub(d999),
        ]
    };
    let [off50, off99, off999] = added(off, direct);
    let [on50, on99, on999] = added(on, direct_again);
    tokio::time::sleep(Duration::from_millis(500)).await;
    let page = observed.gateway.render_metrics();
    let count = |name: &str| -> f64 {
        inferrail_load::parse::samples(&page)
            .filter(|s| s.name == name)
            .map(|s| s.value)
            .sum()
    };
    let (records, dropped) = (
        count("inferrail_decision_log_records_total"),
        count("inferrail_decision_log_dropped_total"),
    );
    let lines = std::fs::read_to_string(&log)
        .map(|t| t.lines().count())
        .unwrap_or(0);
    let _ = std::fs::remove_file(&log);
    println!(
        "{rate}/s x {total}: added p50/p99/p99.9 {:.3}/{:.3}/{:.3} ms with metrics only, {:.3}/{:.3}/{:.3} ms with decision log and tracing; {records} decisions logged, {dropped} dropped, {lines} lines on disk",
        ms(off50),
        ms(off99),
        ms(off999),
        ms(on50),
        ms(on99),
        ms(on999),
    );
    assert_eq!(records + dropped, total as f64);
    assert!(lines as f64 >= records * 0.99);
    assert!(on99 < Duration::from_millis(20));
}
