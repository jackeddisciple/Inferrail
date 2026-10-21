//! A client that goes away frees the engine at once, shutdown drains
//! politely but within its bound, and prompts stay out of the logs.

mod common;

use std::io::Write;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use common::*;
use futures::StreamExt;
use inferrail_sim::latency::LatencyModel;
use serde_json::json;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

async fn wait_engine(
    sim: &Upstream,
    done: impl Fn(&inferrail_sim::engine::EngineSnapshot) -> bool,
) -> Duration {
    wait_until(Duration::from_secs(3), || done(&sim.snapshot())).await
}

#[tokio::test]
async fn a_stream_disconnect_frees_the_engine_slot_within_100ms() {
    let sim = Upstream::spawn(sim_config(&[MODEL], decode(20.0))).await;
    let gw = TestGateway::start(fleet(&[sim.url()], "")).await;
    let response = gw
        .post(
            "/v1/chat/completions",
            &streaming(chat("long answer", 2_000)),
        )
        .await;
    let mut stream = response.bytes_stream();
    stream.next().await.expect("first chunk").expect("ok");
    assert_eq!(sim.snapshot().running, 1);
    assert_eq!(gw.in_flight(), 1);
    drop(stream);
    let freed = wait_engine(&sim, |s| s.running == 0).await;
    assert!(freed < Duration::from_millis(100), "took {freed:?}");
    assert_eq!(sim.snapshot().kv_blocks_used, 0);
    assert_eq!(sim.sim.metrics().finished("abort"), 1);
    gw.wait_idle().await;
}

#[tokio::test]
async fn a_queued_request_is_withdrawn_when_its_client_leaves() {
    let latency = LatencyModel {
        prefill_token_secs: 1e-3,
        ..LatencyModel::ZERO
    };
    let sim = Upstream::spawn(sim_config(&[MODEL], latency)).await;
    let gw = TestGateway::start(fleet(&[sim.url()], "")).await;
    let blocker = {
        let (client, url) = (gw.client.clone(), gw.url("/v1/chat/completions"));
        let prompt: String = (0..800).map(|i| format!("b{i} ")).collect();
        let body = chat(&prompt, 1);
        tokio::spawn(async move { client.post(url).json(&body).send().await })
    };
    wait_engine(&sim, |s| s.running == 1).await;
    let queued = gw
        .post("/v1/chat/completions", &streaming(chat("short", 1)))
        .await;
    wait_engine(&sim, |s| s.waiting == 1).await;
    drop(queued);
    let withdrawn = wait_engine(&sim, |s| s.waiting == 0).await;
    assert!(withdrawn < Duration::from_millis(100), "took {withdrawn:?}");
    blocker.abort();
}

#[tokio::test]
async fn a_client_that_gives_up_on_a_full_response_frees_the_slot() {
    let sim = Upstream::spawn(sim_config(&[MODEL], decode(20.0))).await;
    let gw = TestGateway::start(fleet(&[sim.url()], "")).await;
    let impatient = reqwest::Client::builder()
        .timeout(Duration::from_millis(150))
        .build()
        .unwrap();
    let result = impatient
        .post(gw.url("/v1/chat/completions"))
        .json(&chat("x", 1_000))
        .send()
        .await;
    assert!(result.is_err(), "request should time out");
    let gave_up = Instant::now();
    wait_engine(&sim, |s| s.running == 0).await;
    let freed = gave_up.elapsed();
    assert!(freed < Duration::from_millis(100), "took {freed:?}");
    gw.wait_idle().await;
    assert_eq!(gw.replica("r0").pending(), 0);
}

#[tokio::test]
async fn shutdown_finishes_running_streams_and_refuses_new_connections() {
    let sim = Upstream::spawn(sim_config(&[MODEL], decode(4.0))).await;
    let mut gw = TestGateway::start(fleet(&[sim.url()], "")).await;
    let response = gw
        .post("/v1/chat/completions", &streaming(chat("story", 100)))
        .await;
    let mut stream = response.bytes_stream();
    let mut body = stream.next().await.expect("first").expect("ok").to_vec();
    gw.signal_shutdown();
    tokio::time::sleep(Duration::from_millis(50)).await;
    let fresh = reqwest::Client::new();
    assert!(fresh.get(gw.url("/health")).send().await.is_err());
    while let Some(chunk) = stream.next().await {
        body.extend_from_slice(&chunk.expect("stream finishes during the drain"));
    }
    assert!(
        String::from_utf8(body)
            .unwrap()
            .ends_with("data: [DONE]\n\n")
    );
    gw.join().await.unwrap();
}

#[tokio::test]
async fn the_drain_is_bounded_by_the_shutdown_timeout() {
    let sim = Upstream::spawn(sim_config(&[MODEL], decode(20.0))).await;
    let mut gw =
        TestGateway::start(fleet(&[sim.url()], "server:\n  shutdown_timeout: 200ms\n")).await;
    let response = gw
        .post("/v1/chat/completions", &streaming(chat("endless", 5_000)))
        .await;
    let mut stream = response.bytes_stream();
    stream.next().await.expect("first").expect("ok");
    let started = Instant::now();
    gw.signal_shutdown();
    gw.join().await.unwrap();
    let took = started.elapsed();
    assert!(
        took >= Duration::from_millis(190) && took < Duration::from_secs(2),
        "{took:?}"
    );
    while let Some(chunk) = stream.next().await {
        if chunk.is_err() {
            break;
        }
    }
    wait_engine(&sim, |s| s.running == 0).await;
}

#[tokio::test]
async fn connections_that_never_finish_their_headers_are_closed() {
    let sim = Upstream::zero(&[MODEL]).await;
    let gw = TestGateway::start(fleet(
        &[sim.url()],
        "server:\n  header_read_timeout: 200ms\n",
    ))
    .await;
    let addr = gw.base.trim_start_matches("http://").to_owned();
    let mut socket = tokio::net::TcpStream::connect(addr).await.unwrap();
    socket
        .write_all(b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\n")
        .await
        .unwrap();
    let started = Instant::now();
    let mut buf = Vec::new();
    let read = tokio::time::timeout(Duration::from_secs(3), socket.read_to_end(&mut buf)).await;
    assert!(read.is_ok(), "connection left open");
    let took = started.elapsed();
    assert!(took >= Duration::from_millis(150), "{took:?}");
    assert!(sim.hits() == 0);
}

#[derive(Clone, Default)]
struct Capture(Arc<Mutex<Vec<u8>>>);

impl Write for Capture {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0.lock().expect("log buffer").extend_from_slice(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

async fn captured_logs(log_prompts: bool) -> String {
    let capture = Capture::default();
    let sink = capture.clone();
    let subscriber = tracing_subscriber::fmt()
        .with_max_level(tracing::Level::TRACE)
        .with_ansi(false)
        .with_writer(move || sink.clone())
        .finish();
    let _guard = tracing::subscriber::set_default(subscriber);
    let sims = [
        Upstream::zero(&[MODEL]).await,
        Upstream::zero(&[MODEL]).await,
    ];
    sims[0].kill();
    let gw = TestGateway::start(fleet(
        &[sims[0].url(), sims[1].url()],
        &format!(
            "routing:\n  default_policy: round_robin\nlogging:\n  log_prompts: {log_prompts}\n"
        ),
    ))
    .await;
    for stream in [false, true, false, true] {
        let mut body = chat("SECRET-PROMPT-7f3a", 2);
        body["stream"] = json!(stream);
        let response = gw.post("/v1/chat/completions", &body).await;
        assert_eq!(response.status(), 200);
        let _ = response.bytes().await;
    }
    drop(gw);
    String::from_utf8(capture.0.lock().expect("log buffer").clone()).expect("utf8 logs")
}

#[tokio::test]
async fn prompts_never_reach_the_logs_unless_enabled() {
    let logs = captured_logs(false).await;
    assert!(logs.contains("retrying on another replica"), "{logs}");
    assert!(!logs.contains("SECRET-PROMPT-7f3a"), "{logs}");

    let logs = captured_logs(true).await;
    assert!(logs.contains("SECRET-PROMPT-7f3a"));
}

#[tokio::test]
async fn a_body_that_never_finishes_arriving_times_out() {
    let sim = Upstream::zero(&[MODEL]).await;
    let gw = TestGateway::start(fleet(
        &[sim.url()],
        "server:\n  request_timeout: 300ms\nupstream:\n  connect_timeout: 100ms\n",
    ))
    .await;
    let addr = gw.base.trim_start_matches("http://").to_owned();
    let mut socket = tokio::net::TcpStream::connect(addr).await.unwrap();
    socket
        .write_all(
            b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Length: 100\r\n\r\n{\"model",
        )
        .await
        .unwrap();
    let started = Instant::now();
    let mut answer = Vec::new();
    let read = tokio::time::timeout(Duration::from_secs(3), socket.read_to_end(&mut answer)).await;
    let took = started.elapsed();
    assert!(read.is_ok(), "the gateway kept waiting for the body");
    let answer = String::from_utf8_lossy(&answer);
    assert!(answer.starts_with("HTTP/1.1 408"), "{answer}");
    assert!(answer.contains("RequestTimeoutError"), "{answer}");
    assert!(
        took >= Duration::from_millis(250) && took < Duration::from_secs(2),
        "{took:?}"
    );
    assert_eq!(sim.hits(), 0);
}
