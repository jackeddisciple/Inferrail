#![allow(dead_code, unreachable_pub)]

use std::convert::Infallible;
use std::net::SocketAddr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use axum::body::Body;
use futures::StreamExt;
use http_body_util::BodyExt;
use hyper::body::{Bytes, Incoming};
use hyper::header::HeaderMap;
use hyper::service::service_fn;
use hyper::{Request, Response, Version};
use hyper_util::rt::{TokioExecutor, TokioIo};
use hyper_util::server::conn::auto;
use inferrail_core::config::Config;
use inferrail_core::openai::sse::{SseDecoder, SseEvent};
use inferrail_gateway::Gateway;
use inferrail_sim::engine::{EngineConfig, EngineSnapshot};
use inferrail_sim::latency::LatencyModel;
use inferrail_sim::server::{Sim, SimConfig};
use serde_json::{Value, json};
use tokio::net::TcpListener;
use tokio::sync::oneshot;
use tokio::task::{AbortHandle, JoinHandle};
use tower::ServiceExt;

pub const MODEL: &str = "model-a";

pub const FAST_HEALTH: &str = "health:\n  interval: 50ms\n  timeout: 40ms\n  unhealthy_threshold: 2\n  healthy_threshold: 1\n  ejection_backoff: 300ms\n  max_ejection_backoff: 3s\n";

#[derive(Debug, Clone, PartialEq)]
pub enum Fault {
    None,
    Status(u16),
    Hang,
    Close,
    Delay(Duration),
}

type Tasks = Arc<Mutex<Vec<AbortHandle>>>;

/// An in process sim behind its own accept loop, so it can be killed for
/// real and made to misbehave on demand.
pub struct Upstream {
    pub sim: Sim,
    pub addr: SocketAddr,
    fault: Arc<Mutex<Fault>>,
    request_fault: Arc<Mutex<Fault>>,
    hits: Arc<AtomicU64>,
    tasks: Tasks,
}

pub fn sim_config(names: &[&str], latency: LatencyModel) -> SimConfig {
    SimConfig {
        model: names[0].to_owned(),
        served_model_names: names.iter().map(|n| (*n).to_owned()).collect(),
        engine: EngineConfig {
            latency,
            ..EngineConfig::default()
        },
        ..SimConfig::default()
    }
}

pub fn decode(token_ms: f64) -> LatencyModel {
    LatencyModel {
        decode_token_secs: token_ms / 1e3,
        ..LatencyModel::ZERO
    }
}

impl Upstream {
    pub async fn spawn(config: SimConfig) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let upstream = Self {
            addr: listener.local_addr().expect("addr"),
            sim: Sim::new(config).expect("sim config"),
            fault: Arc::new(Mutex::new(Fault::None)),
            request_fault: Arc::new(Mutex::new(Fault::None)),
            hits: Arc::new(AtomicU64::new(0)),
            tasks: Arc::default(),
        };
        upstream.start(listener);
        upstream
    }

    pub async fn zero(names: &[&str]) -> Self {
        Self::spawn(sim_config(names, LatencyModel::ZERO)).await
    }

    fn start(&self, listener: TcpListener) {
        let router = self.sim.router();
        let (fault, request_fault, hits, tasks) = (
            Arc::clone(&self.fault),
            Arc::clone(&self.request_fault),
            Arc::clone(&self.hits),
            Arc::clone(&self.tasks),
        );
        let conns = Arc::clone(&self.tasks);
        let accept = tokio::spawn(async move {
            loop {
                let Ok((stream, _)) = listener.accept().await else {
                    continue;
                };
                let (router, fault, request_fault, hits) = (
                    router.clone(),
                    Arc::clone(&fault),
                    Arc::clone(&request_fault),
                    Arc::clone(&hits),
                );
                let service = service_fn(move |request: Request<Incoming>| {
                    let (router, hits) = (router.clone(), Arc::clone(&hits));
                    let path = request.uri().path();
                    let inference =
                        path.starts_with("/v1/chat") || path.starts_with("/v1/completions");
                    let mut fault = fault.lock().expect("fault").clone();
                    if inference && fault == Fault::None {
                        fault = request_fault.lock().expect("fault").clone();
                    }
                    async move {
                        if inference {
                            hits.fetch_add(1, Ordering::SeqCst);
                        }
                        match fault {
                            Fault::None => {}
                            Fault::Delay(delay) => tokio::time::sleep(delay).await,
                            Fault::Status(code) => {
                                let body = json!({"error": {"message": "injected", "type": "InjectedError", "param": null, "code": code}});
                                return Ok(Response::builder()
                                    .status(code)
                                    .header("content-type", "application/json")
                                    .body(Body::from(body.to_string()))
                                    .expect("response"));
                            }
                            Fault::Hang => std::future::pending::<()>().await,
                            Fault::Close => return Err(std::io::Error::other("injected close")),
                        }
                        let response = router
                            .oneshot(request.map(Body::new))
                            .await
                            .unwrap_or_else(|never: Infallible| match never {});
                        Ok::<_, std::io::Error>(response)
                    }
                });
                let conn = tokio::spawn(async move {
                    let _ = auto::Builder::new(TokioExecutor::new())
                        .serve_connection(TokioIo::new(stream), service)
                        .await;
                });
                conns.lock().expect("tasks").push(conn.abort_handle());
            }
        });
        tasks.lock().expect("tasks").push(accept.abort_handle());
    }

    pub fn url(&self) -> String {
        format!("http://{}", self.addr)
    }

    pub fn set_fault(&self, fault: Fault) {
        *self.fault.lock().expect("fault") = fault;
    }

    /// A fault on inference routes only; `/health` keeps answering.
    pub fn set_request_fault(&self, fault: Fault) {
        *self.request_fault.lock().expect("fault") = fault;
    }

    pub fn hits(&self) -> u64 {
        self.hits.load(Ordering::SeqCst)
    }

    /// Drops the listener and every open connection, like a crashed process.
    pub fn kill(&self) {
        for task in self.tasks.lock().expect("tasks").drain(..) {
            task.abort();
        }
    }

    pub async fn restart(&self) {
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            match TcpListener::bind(self.addr).await {
                Ok(listener) => return self.start(listener),
                Err(err) if Instant::now() < deadline => {
                    let _ = err;
                    tokio::time::sleep(Duration::from_millis(20)).await;
                }
                Err(err) => panic!("cannot rebind {}: {err}", self.addr),
            }
        }
    }

    pub fn snapshot(&self) -> EngineSnapshot {
        self.sim.engine().snapshot()
    }
}

impl Drop for Upstream {
    fn drop(&mut self) {
        self.kill();
    }
}

#[derive(Debug, Clone)]
pub struct Recorded {
    pub method: String,
    pub uri: String,
    pub version: Version,
    pub headers: HeaderMap,
    pub body: Bytes,
}

#[derive(Debug, Clone)]
pub struct Script {
    pub status: u16,
    pub headers: Vec<(&'static str, String)>,
    pub chunks: Vec<Bytes>,
    pub pause: Duration,
}

impl Default for Script {
    fn default() -> Self {
        Self {
            status: 200,
            headers: vec![("content-type", "application/json".to_owned())],
            chunks: vec![Bytes::from_static(b"{}")],
            pause: Duration::ZERO,
        }
    }
}

/// A scripted upstream that records what it receives and answers with
/// exactly the chunks it is given.
pub struct RawUpstream {
    pub addr: SocketAddr,
    model_fetches: Arc<AtomicU64>,
    recorded: Arc<Mutex<Vec<Recorded>>>,
    script: Arc<Mutex<Script>>,
    task: AbortHandle,
}

impl RawUpstream {
    pub async fn spawn() -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let addr = listener.local_addr().expect("addr");
        let recorded: Arc<Mutex<Vec<Recorded>>> = Arc::default();
        let script: Arc<Mutex<Script>> = Arc::default();
        let model_fetches: Arc<AtomicU64> = Arc::default();
        let (rec, scr, fetches) = (
            Arc::clone(&recorded),
            Arc::clone(&script),
            Arc::clone(&model_fetches),
        );
        let task = tokio::spawn(async move {
            loop {
                let Ok((stream, _)) = listener.accept().await else {
                    continue;
                };
                let (rec, scr, fetches) =
                    (Arc::clone(&rec), Arc::clone(&scr), Arc::clone(&fetches));
                let service = service_fn(move |request: Request<Incoming>| {
                    let (rec, scr, fetches) =
                        (Arc::clone(&rec), Arc::clone(&scr), Arc::clone(&fetches));
                    async move {
                        match request.uri().path() {
                            "/health" => {
                                return Ok::<_, Infallible>(Response::new(Body::empty()));
                            }
                            "/v1/models" => {
                                fetches.fetch_add(1, Ordering::SeqCst);
                                return Ok(Response::new(Body::from("not a model list")));
                            }
                            _ => {}
                        }
                        let (parts, body) = request.into_parts();
                        let body = body
                            .collect()
                            .await
                            .map(|c| c.to_bytes())
                            .unwrap_or_default();
                        rec.lock().expect("recorded").push(Recorded {
                            method: parts.method.to_string(),
                            uri: parts.uri.to_string(),
                            version: parts.version,
                            headers: parts.headers,
                            body,
                        });
                        let script = scr.lock().expect("script").clone();
                        let pause = script.pause;
                        let chunks =
                            futures::stream::iter(script.chunks).then(move |chunk| async move {
                                if !pause.is_zero() {
                                    tokio::time::sleep(pause).await;
                                }
                                tokio::task::yield_now().await;
                                Ok::<_, Infallible>(chunk)
                            });
                        let mut response = Response::builder().status(script.status);
                        for (name, value) in &script.headers {
                            response = response.header(*name, value);
                        }
                        Ok(response.body(Body::from_stream(chunks)).expect("response"))
                    }
                });
                tokio::spawn(async move {
                    let _ = auto::Builder::new(TokioExecutor::new())
                        .serve_connection(TokioIo::new(stream), service)
                        .await;
                });
            }
        });
        Self {
            addr,
            model_fetches,
            recorded,
            script,
            task: task.abort_handle(),
        }
    }

    pub fn url(&self) -> String {
        format!("http://{}", self.addr)
    }

    pub fn script(&self, script: Script) {
        *self.script.lock().expect("script") = script;
    }

    pub fn model_fetches(&self) -> u64 {
        self.model_fetches.load(Ordering::SeqCst)
    }

    pub fn take(&self) -> Vec<Recorded> {
        std::mem::take(&mut *self.recorded.lock().expect("recorded"))
    }
}

impl Drop for RawUpstream {
    fn drop(&mut self) {
        self.task.abort();
    }
}

pub struct TestGateway {
    pub base: String,
    pub gateway: Gateway,
    pub client: reqwest::Client,
    stop: Option<oneshot::Sender<()>>,
    task: Option<JoinHandle<std::io::Result<()>>>,
}

pub fn config(text: &str) -> Config {
    Config::from_yaml_str(text).unwrap_or_else(|err| panic!("{err}\n{text}"))
}

/// A config with one model served by every given URL as replicas r0, r1...
pub fn fleet(urls: &[String], extra: &str) -> Config {
    let mut text = format!("models:\n  - name: {MODEL}\nreplicas:\n");
    for (i, url) in urls.iter().enumerate() {
        text.push_str(&format!(
            "  - id: r{i}\n    url: {url}\n    models: [{MODEL}]\n"
        ));
    }
    text.push_str(extra);
    config(&text)
}

pub fn urls(upstreams: &[&Upstream]) -> Vec<String> {
    upstreams.iter().map(|u| u.url()).collect()
}

impl TestGateway {
    pub async fn start(config: Config) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let base = format!("http://{}", listener.local_addr().expect("addr"));
        let gateway = Gateway::new(config).expect("gateway");
        let (stop, stopped) = oneshot::channel::<()>();
        let task = tokio::spawn(gateway.clone().serve(listener, async move {
            let _ = stopped.await;
        }));
        Self {
            base,
            gateway,
            client: reqwest::Client::new(),
            stop: Some(stop),
            task: Some(task),
        }
    }

    pub fn url(&self, path: &str) -> String {
        format!("{}{path}", self.base)
    }

    pub async fn post(&self, path: &str, body: &Value) -> reqwest::Response {
        self.client
            .post(self.url(path))
            .json(body)
            .send()
            .await
            .expect("request sent")
    }

    pub async fn post_json(&self, path: &str, body: &Value) -> (u16, Value) {
        let response = self.post(path, body).await;
        let status = response.status().as_u16();
        (status, response.json().await.expect("json body"))
    }

    pub async fn stream(&self, body: &Value) -> Vec<SseEvent> {
        let response = self.post("/v1/chat/completions", body).await;
        assert_eq!(response.status(), 200);
        let mut decoder = SseDecoder::default();
        let mut events = Vec::new();
        let mut bytes = response.bytes_stream();
        while let Some(chunk) = bytes.next().await {
            decoder
                .push(&chunk.expect("chunk"), &mut events)
                .expect("sse");
        }
        events
    }

    pub fn replica(&self, id: &str) -> std::sync::Arc<inferrail_control::Replica> {
        std::sync::Arc::clone(
            self.gateway
                .registry()
                .load()
                .get(id)
                .unwrap_or_else(|| panic!("no replica {id}")),
        )
    }

    pub fn in_flight(&self) -> u32 {
        self.gateway
            .registry()
            .load()
            .replicas()
            .iter()
            .map(|r| r.in_flight())
            .sum()
    }

    pub async fn wait_idle(&self) {
        let started = Instant::now();
        while self.in_flight() != 0 {
            assert!(
                started.elapsed() < Duration::from_secs(5),
                "in flight never drained: {}",
                self.in_flight()
            );
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    }

    pub async fn shutdown(mut self) -> std::io::Result<()> {
        if let Some(stop) = self.stop.take() {
            let _ = stop.send(());
        }
        match self.task.take() {
            Some(task) => task.await.expect("serve task"),
            None => Ok(()),
        }
    }
}

impl Drop for TestGateway {
    fn drop(&mut self) {
        if let Some(task) = &self.task {
            task.abort();
        }
    }
}

pub fn chat(content: &str, max_tokens: u64) -> Value {
    json!({
        "model": MODEL,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens
    })
}

pub fn streaming(mut body: Value) -> Value {
    body["stream"] = json!(true);
    body["stream_options"] = json!({"include_usage": true});
    body
}

pub async fn wait_until(limit: Duration, mut done: impl FnMut() -> bool) -> Duration {
    let started = Instant::now();
    while !done() {
        assert!(
            started.elapsed() < limit,
            "condition not met within {limit:?}"
        );
        tokio::time::sleep(Duration::from_millis(2)).await;
    }
    started.elapsed()
}
