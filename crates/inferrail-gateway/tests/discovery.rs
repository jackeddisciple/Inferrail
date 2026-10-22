//! Replicas discovered through DNS join and leave without disturbing
//! traffic or the state of members that stay.

mod common;

use std::io;
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use common::*;
use futures::StreamExt;
use inferrail_control::{Resolution, Resolve};
use inferrail_gateway::Gateway;
use tokio::net::TcpListener;

#[derive(Debug, Default)]
struct FakeDns(Mutex<Option<Vec<SocketAddr>>>);

impl FakeDns {
    fn answer(&self, sims: &[&Upstream]) {
        *self.0.lock().expect("dns") = Some(sims.iter().map(|s| s.addr).collect());
    }

    fn fail(&self) {
        *self.0.lock().expect("dns") = None;
    }
}

impl Resolve for FakeDns {
    fn resolve<'a>(&'a self, host: &'a str, port: u16) -> Resolution<'a> {
        assert_eq!((host, port), ("pool.test", 8000));
        let answer = self.0.lock().expect("dns").clone();
        Box::pin(async move { answer.ok_or_else(|| io::Error::other("SERVFAIL")) })
    }
}

async fn start(text: &str, dns: Arc<FakeDns>) -> TestGateway {
    let gateway = Gateway::with_resolver(config(text), dns).expect("gateway");
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    TestGateway::serve(gateway, listener).await
}

fn pool(extra: &str) -> String {
    format!(
        "models:\n  - name: {MODEL}\ndiscovery:\n  dns:\n    - name: pool\n      url: http://pool.test:8000\n      models: [{MODEL}]\n      refresh: 50ms\n{extra}"
    )
}

fn member(sim: &Upstream) -> String {
    format!("pool-127.0.0.1-{}", sim.addr.port())
}

async fn served_by(gw: &TestGateway, n: usize) -> Vec<String> {
    let mut seen = Vec::new();
    for _ in 0..n {
        let response = gw.post("/v1/chat/completions", &chat("x", 1)).await;
        assert_eq!(response.status(), 200);
        seen.push(
            response.headers()["x-inferrail-replica"]
                .to_str()
                .expect("ascii")
                .to_owned(),
        );
    }
    seen.sort();
    seen.dedup();
    seen
}

#[tokio::test]
async fn members_join_and_leave_while_survivors_keep_their_state() {
    let sims = [
        Upstream::zero(&[MODEL]).await,
        Upstream::zero(&[MODEL]).await,
        Upstream::zero(&[MODEL]).await,
    ];
    let dns = Arc::new(FakeDns::default());
    dns.answer(&[&sims[0], &sims[1]]);
    let gw = start(
        &pool("routing:\n  default_policy: round_robin\n"),
        Arc::clone(&dns),
    )
    .await;
    let mut first = vec![member(&sims[0]), member(&sims[1])];
    first.sort();
    assert_eq!(served_by(&gw, 6).await, first);

    let survivor = gw.replica(&member(&sims[1]));
    let held = survivor.try_acquire(0).unwrap();
    dns.answer(&[&sims[1], &sims[2]]);
    let registry = gw.gateway.registry();
    let joined = wait_until(Duration::from_secs(2), || {
        registry.load().get(&member(&sims[2])).is_some()
    })
    .await;
    assert!(joined < Duration::from_millis(500), "{joined:?}");
    assert!(registry.load().get(&member(&sims[0])).is_none());
    let kept = gw.replica(&member(&sims[1]));
    assert!(Arc::ptr_eq(&kept, &survivor));
    assert_eq!(kept.in_flight(), 1);
    drop(held);

    let before = sims[0].hits();
    let mut second = vec![member(&sims[1]), member(&sims[2])];
    second.sort();
    assert_eq!(served_by(&gw, 6).await, second);
    assert_eq!(sims[0].hits(), before);
}

#[tokio::test]
async fn a_leaving_member_finishes_its_streams() {
    let sims = [
        Upstream::spawn(sim_config(&[MODEL], decode(2.0))).await,
        Upstream::spawn(sim_config(&[MODEL], decode(2.0))).await,
    ];
    let dns = Arc::new(FakeDns::default());
    dns.answer(&[&sims[0]]);
    let gw = start(&pool(""), Arc::clone(&dns)).await;
    let response = gw
        .post("/v1/chat/completions", &streaming(chat("long", 100)))
        .await;
    assert_eq!(
        response.headers()["x-inferrail-replica"],
        member(&sims[0]).as_str()
    );
    dns.answer(&[&sims[1]]);
    let registry = gw.gateway.registry();
    wait_until(Duration::from_secs(2), || {
        registry.load().get(&member(&sims[0])).is_none()
    })
    .await;
    let mut body = Vec::new();
    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.next().await {
        body.extend_from_slice(&chunk.expect("stream survives removal"));
    }
    let text = String::from_utf8(body).unwrap();
    assert!(text.ends_with("data: [DONE]\n\n"));
    assert_eq!(text.matches("data: ").count(), 100 + 3);
    assert_eq!(served_by(&gw, 3).await, [member(&sims[1])]);
}

#[tokio::test]
async fn lookup_failures_keep_members_and_an_empty_answer_drains_the_pool() {
    let sim = Upstream::zero(&[MODEL]).await;
    let dns = Arc::new(FakeDns::default());
    dns.answer(&[&sim]);
    let gw = start(&pool(""), Arc::clone(&dns)).await;
    dns.fail();
    tokio::time::sleep(Duration::from_millis(250)).await;
    assert_eq!(served_by(&gw, 3).await, [member(&sim)]);

    *dns.0.lock().unwrap() = Some(Vec::new());
    let registry = gw.gateway.registry();
    wait_until(Duration::from_secs(2), || {
        registry.load().replicas().is_empty()
    })
    .await;
    let (status, body) = gw.post_json("/v1/chat/completions", &chat("x", 1)).await;
    assert_eq!(status, 503, "{body}");
    assert!(
        body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("no healthy replica")
    );
}

#[tokio::test]
async fn static_replicas_and_groups_mix_and_ids_never_collide() {
    let sims = [
        Upstream::zero(&[MODEL]).await,
        Upstream::zero(&[MODEL]).await,
    ];
    let dns = Arc::new(FakeDns::default());
    dns.answer(&[&sims[0], &sims[1]]);
    let colliding = member(&sims[1]);
    let text = pool(&format!(
        "replicas:\n  - id: {colliding}\n    url: {}\n    models: [{MODEL}]\n",
        sims[1].url()
    ));
    let gw = start(&text, dns).await;
    let registry = gw.gateway.registry();
    let fleet = registry.load();
    let mut ids: Vec<&str> = fleet.replicas().iter().map(|r| r.id().as_str()).collect();
    ids.sort_unstable();
    let mut want = vec![colliding.clone(), member(&sims[0])];
    want.sort();
    assert_eq!(ids, want);
}

#[tokio::test]
async fn the_system_resolver_discovers_a_literal_address() {
    let sim = Upstream::zero(&[MODEL]).await;
    let text = format!(
        "models:\n  - name: {MODEL}\ndiscovery:\n  dns:\n    - name: local\n      url: http://127.0.0.1:{}\n      models: [{MODEL}]\n",
        sim.addr.port()
    );
    let gw = TestGateway::start(config(&text)).await;
    let id = format!("local-127.0.0.1-{}", sim.addr.port());
    assert_eq!(served_by(&gw, 2).await, [id]);
}

#[derive(Debug)]
struct SlowDns(SocketAddr);

impl Resolve for SlowDns {
    fn resolve<'a>(&'a self, _host: &'a str, _port: u16) -> Resolution<'a> {
        Box::pin(async move {
            tokio::time::sleep(Duration::from_millis(400)).await;
            Ok(vec![self.0])
        })
    }
}

#[tokio::test]
async fn groups_are_resolved_together_at_startup() {
    let sim = Upstream::zero(&[MODEL]).await;
    let mut text = format!("models:\n  - name: {MODEL}\ndiscovery:\n  dns:\n");
    for name in ["a", "b", "c", "d"] {
        text.push_str(&format!(
            "    - name: pool-{name}\n      url: http://{name}.test:8000\n      models: [{MODEL}]\n      refresh: 1m\n"
        ));
    }
    let started = std::time::Instant::now();
    let gateway = Gateway::with_resolver(config(&text), Arc::new(SlowDns(sim.addr))).unwrap();
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let gw = TestGateway::serve(gateway, listener).await;
    let took = started.elapsed();
    assert_eq!(gw.gateway.registry().load().replicas().len(), 4);
    assert!(
        took < Duration::from_millis(1_000),
        "four lookups took {took:?}"
    );
}
