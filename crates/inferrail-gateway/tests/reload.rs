//! Live config reloads: under load nothing is dropped, policies whose
//! settings stay keep their state, sections only a restart can change are
//! refused whole, and the config file is followed as it is edited.

mod common;

use std::time::{Duration, Instant};

use common::{
    FAST_HEALTH, MODEL, TestGateway, Upstream, chat, decode, fleet, sim_config, streaming, urls,
    wait_until,
};
use futures::StreamExt;
use inferrail_core::config::{Config, PolicyKind};
use inferrail_gateway::{Gateway, ReloadError};

fn metric(gw: &TestGateway, line: &str) -> f64 {
    gw.gateway
        .render_metrics()
        .lines()
        .find(|l| l.starts_with(line))
        .and_then(|l| l.rsplit_once(' '))
        .map_or(0.0, |(_, v)| v.parse().expect("number"))
}

async fn replica_of(gw: &TestGateway, i: usize) -> String {
    let response = gw
        .post("/v1/chat/completions", &chat(&format!("q{i}"), 1))
        .await;
    assert_eq!(response.status(), 200);
    let replica = response.headers()["x-inferrail-replica"]
        .to_str()
        .expect("ascii")
        .to_owned();
    response.bytes().await.expect("body");
    replica
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn reloads_under_load_drop_no_request() {
    let sims = [
        Upstream::spawn(sim_config(&[MODEL], decode(4.0))).await,
        Upstream::spawn(sim_config(&[MODEL], decode(4.0))).await,
        Upstream::spawn(sim_config(&[MODEL], decode(4.0))).await,
        Upstream::spawn(sim_config(&[MODEL], decode(4.0))).await,
    ];
    let all = urls(&[&sims[0], &sims[1], &sims[2], &sims[3]]);
    let base = fleet(&all[..3], FAST_HEALTH);
    let gw = TestGateway::start(base.clone()).await;
    let mut tasks = Vec::new();
    for i in 0..400 {
        let (client, url) = (gw.client.clone(), gw.url("/v1/chat/completions"));
        let body = if i % 2 == 0 {
            streaming(chat(&format!("stream {i}"), 30))
        } else {
            chat(&format!("full {i}"), 20)
        };
        tasks.push(tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(i * 8)).await;
            let response = client.post(url).json(&body).send().await.expect("sent");
            let status = response.status().as_u16();
            let mut bytes = response.bytes_stream();
            let mut text = Vec::new();
            while let Some(chunk) = bytes.next().await {
                text.extend_from_slice(&chunk.expect("chunk"));
            }
            (status, String::from_utf8_lossy(&text).into_owned())
        }));
    }
    let policies = [
        PolicyKind::CostModel,
        PolicyKind::LeastRequest,
        PolicyKind::RoundRobin,
        PolicyKind::KvAware,
    ];
    let mut generation = 1;
    for step in 0..30usize {
        tokio::time::sleep(Duration::from_millis(100)).await;
        let mut next = if step % 3 == 2 {
            fleet(&all, FAST_HEALTH)
        } else {
            base.clone()
        };
        next.routing.default_policy = policies[step % policies.len()];
        next.routing.epsilon = 0.5 + step as f64 / 10.0;
        let done = gw.gateway.reload(next).expect("reload");
        assert!(done.generation > generation);
        generation = done.generation;
    }
    for task in tasks {
        let (status, text) = task.await.expect("request");
        assert_eq!(status, 200, "{text}");
        assert!(
            text.ends_with("data: [DONE]\n\n") || text.contains("\"finish_reason\":\"length\""),
            "{text}"
        );
    }
    gw.wait_idle().await;
    assert_eq!(
        metric(&gw, "inferrail_config_reloads_total{result=\"applied\"}"),
        30.0
    );
    assert_eq!(
        metric(&gw, "inferrail_config_generation"),
        generation as f64
    );
    assert_eq!(
        metric(
            &gw,
            "inferrail_requests_total{model=\"model-a\",replica=\"r3\",status=\"5xx\"}"
        ),
        0.0
    );
    let page = gw.gateway.render_metrics();
    let last = policies[29 % policies.len()].as_str();
    assert!(page.contains(&format!(
        "inferrail_model_policy{{model=\"model-a\",policy=\"{last}\"}} 1"
    )));
}

#[tokio::test]
async fn unchanged_policies_keep_their_state_across_reloads() {
    let sims = [
        Upstream::zero(&[MODEL]).await,
        Upstream::zero(&[MODEL]).await,
        Upstream::zero(&[MODEL]).await,
    ];
    let config = fleet(
        &urls(&[&sims[0], &sims[1], &sims[2]]),
        &format!("{FAST_HEALTH}routing:\n  default_policy: round_robin\n  seed: 3\n"),
    );
    let plain = TestGateway::start(config.clone()).await;
    let mut want = Vec::new();
    for i in 0..6 {
        want.push(replica_of(&plain, i).await);
    }
    let gw = TestGateway::start(config.clone()).await;
    let mut got = Vec::new();
    for i in 0..4 {
        got.push(replica_of(&gw, i).await);
    }
    let mut unrelated = config.clone();
    unrelated.retry.max_attempts = 5;
    let done = gw.gateway.reload(unrelated).expect("reload");
    assert_eq!(
        (done.generation, done.changed.as_slice()),
        (2, &["retry".to_owned()][..])
    );
    for i in 4..6 {
        got.push(replica_of(&gw, i).await);
    }
    assert_eq!(got, want, "the rotation restarted");

    let same = gw
        .gateway
        .reload(gw.gateway.config().as_ref().clone())
        .expect("reload");
    assert_eq!((same.generation, same.changed.len()), (2, 0));
    let mut retuned = config.clone();
    retuned.routing.seed = Some(4);
    gw.gateway.reload(retuned).expect("reload");
    assert_eq!(
        metric(&gw, "inferrail_config_reloads_total{result=\"unchanged\"}"),
        1.0
    );
    assert_eq!(metric(&gw, "inferrail_config_generation"), 3.0);
}

#[tokio::test]
async fn restart_only_changes_are_refused_whole() {
    let upstream = Upstream::zero(&[MODEL]).await;
    let config = fleet(&urls(&[&upstream]), FAST_HEALTH);
    let gw = TestGateway::start(config.clone()).await;
    let mut next = config.clone();
    next.routing.default_policy = PolicyKind::P2c;
    next.server.request_timeout = Duration::from_secs(3);
    next.models[0].tokenizer = Some("builtin:sim".into());
    match gw.gateway.reload(next) {
        Err(ReloadError::Restart(paths)) => assert_eq!(paths, ["server", "models[model-a]"]),
        other => panic!("{other:?}"),
    }
    assert_eq!(
        gw.gateway.config().routing.default_policy,
        PolicyKind::CostModel
    );
    let mut invalid = config.clone();
    invalid.routing.epsilon = -1.0;
    assert!(matches!(
        gw.gateway.reload(invalid),
        Err(ReloadError::Invalid(_))
    ));
    let mut tls = config.clone();
    tls.replicas[0].url = "https://10.0.0.1:8000".parse().unwrap();
    assert!(matches!(gw.gateway.reload(tls), Err(ReloadError::Https(_))));
    assert_eq!(
        metric(&gw, "inferrail_config_reloads_total{result=\"rejected\"}"),
        3.0
    );
    assert_eq!(gw.gateway.config().as_ref(), &config);
    assert_eq!(replica_of(&gw, 0).await, "r0");
}

#[tokio::test]
async fn static_replicas_join_and_leave_while_streams_finish() {
    let sims = [
        Upstream::spawn(sim_config(&[MODEL], decode(10.0))).await,
        Upstream::zero(&[MODEL]).await,
    ];
    let both = urls(&[&sims[0], &sims[1]]);
    let gw = TestGateway::start(fleet(&both[..1], FAST_HEALTH)).await;
    let long = gw
        .post("/v1/chat/completions", &streaming(chat("long", 60)))
        .await;
    assert_eq!(long.headers()["x-inferrail-replica"], "r0");
    let mut moved = fleet(&both, FAST_HEALTH);
    moved.replicas.remove(0);
    let done = gw.gateway.reload(moved).expect("reload");
    assert_eq!(done.changed, ["replicas"]);
    assert!(gw.gateway.registry().load().get("r0").is_none());
    for i in 0..5 {
        assert_eq!(replica_of(&gw, i).await, "r1");
    }
    let text = long.text().await.expect("the stream on r0 finished");
    assert!(text.ends_with("data: [DONE]\n\n"));
    assert_eq!(sims[0].snapshot().running, 0);
}

#[tokio::test]
async fn aliases_change_live() {
    let upstream = Upstream::zero(&[MODEL]).await;
    let config = fleet(&urls(&[&upstream]), FAST_HEALTH);
    let gw = TestGateway::start(config.clone()).await;
    let mut body = chat("hi", 1);
    body["model"] = "nickname".into();
    assert_eq!(gw.post("/v1/chat/completions", &body).await.status(), 404);
    let mut named = config.clone();
    named.models[0].aliases = vec!["nickname".into()];
    gw.gateway.reload(named).expect("reload");
    assert_eq!(gw.post("/v1/chat/completions", &body).await.status(), 200);
    let models: serde_json::Value = gw
        .client
        .get(gw.url("/v1/models"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(models["data"][1]["id"], "nickname");
}

#[tokio::test]
async fn the_config_file_is_followed_as_it_is_edited() {
    let upstream = Upstream::zero(&[MODEL]).await;
    let dir = std::env::temp_dir().join(format!("inferrail-reload-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("gateway.yaml");
    let config = fleet(&urls(&[&upstream]), FAST_HEALTH);
    std::fs::write(&path, config.to_yaml_string().unwrap()).unwrap();
    let gateway = Gateway::new(Config::load(&path).unwrap()).unwrap();
    let watcher = tokio::spawn(
        gateway
            .clone()
            .watch_config(path.clone(), Duration::from_millis(20)),
    );
    let policy = |g: &Gateway| g.config().routing.default_policy;

    let mut next = config.clone();
    next.routing.default_policy = PolicyKind::LeastRequest;
    std::fs::write(&path, next.to_yaml_string().unwrap()).unwrap();
    let took = wait_until(Duration::from_secs(3), || {
        policy(&gateway) == PolicyKind::LeastRequest
    })
    .await;
    assert!(took < Duration::from_secs(1), "{took:?}");

    std::fs::write(&path, "models: [oops").unwrap();
    let mut restart = next.clone();
    restart.health.interval = Duration::from_secs(7);
    tokio::time::sleep(Duration::from_millis(150)).await;
    std::fs::write(&path, restart.to_yaml_string().unwrap()).unwrap();
    tokio::time::sleep(Duration::from_millis(150)).await;
    assert_eq!(policy(&gateway), PolicyKind::LeastRequest);
    assert_eq!(gateway.config().health.interval, config.health.interval);
    let page = gateway.render_metrics();
    assert!(
        page.contains("inferrail_config_reloads_total{result=\"rejected\"} 2"),
        "{page}"
    );

    let started = Instant::now();
    let done = gateway.reload_from(&path);
    assert!(matches!(done, Err(ReloadError::Restart(_))));
    assert!(gateway.reload_from(&dir.join("missing.yaml")).is_err());
    assert!(started.elapsed() < Duration::from_secs(1));
    watcher.abort();
    std::fs::remove_dir_all(&dir).unwrap();
}
