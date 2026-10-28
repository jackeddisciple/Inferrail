//! The admin API: the config in effect, live changes by merge patch, a
//! reload of the config file, and the fleet and index as the gateway sees
//! them.

mod common;

use std::time::Duration;

use common::{
    FAST_HEALTH, MODEL, TestGateway, Upstream, chat, decode, fleet, serve_admin, sim_config, urls,
};
use futures::StreamExt;
use inferrail_core::config::PolicyKind;
use serde_json::{Value, json};

async fn get(admin: &str, path: &str) -> (u16, Value) {
    let response = reqwest::get(format!("{admin}{path}")).await.expect("sent");
    let status = response.status().as_u16();
    (status, response.json().await.expect("json"))
}

async fn send(
    method: reqwest::Method,
    admin: &str,
    path: &str,
    body: Option<Value>,
) -> (u16, Value) {
    let client = reqwest::Client::new();
    let mut request = client.request(method, format!("{admin}{path}"));
    if let Some(body) = body {
        request = request.json(&body);
    }
    let response = request.send().await.expect("sent");
    let status = response.status().as_u16();
    (status, response.json().await.expect("json"))
}

async fn patch(admin: &str, body: Value) -> (u16, Value) {
    send(reqwest::Method::PATCH, admin, "/admin/config", Some(body)).await
}

fn tokenized(sims: &[&Upstream]) -> inferrail_core::config::Config {
    let mut config = fleet(&urls(sims), FAST_HEALTH);
    config.models[0].tokenizer = Some("builtin:sim".to_owned());
    config
}

#[tokio::test]
async fn the_config_is_read_and_patched_live() {
    let sims = [
        Upstream::zero(&[MODEL]).await,
        Upstream::zero(&[MODEL]).await,
    ];
    let gw = TestGateway::start(fleet(&urls(&[&sims[0], &sims[1]]), FAST_HEALTH)).await;
    let admin = serve_admin(&gw.gateway).await;
    let (status, shown) = get(&admin, "/admin/config").await;
    assert_eq!(status, 200);
    assert_eq!(shown["generation"], 1);
    assert_eq!(shown["config"]["routing"]["default_policy"], "cost_model");
    assert_eq!(shown["config"]["server"]["reload_interval"], "2s");

    let (status, done) = patch(
        &admin,
        json!({"routing": {"default_policy": "least_request", "kv_aware": {"load_weight": 3}}}),
    )
    .await;
    assert_eq!(status, 200, "{done}");
    assert_eq!(done, json!({"generation": 2, "changed": ["routing"]}));
    assert_eq!(
        gw.gateway.config().routing.default_policy,
        PolicyKind::LeastRequest
    );
    assert_eq!(gw.gateway.config().routing.kv_aware.load_weight, 3.0);
    assert_eq!(gw.gateway.config().routing.kv_aware.cache_weight, 1.0);

    let (status, err) = patch(&admin, json!({"server": {"max_body_bytes": 10}})).await;
    assert_eq!(status, 409, "{err}");
    assert_eq!(err["error"]["type"], "restart_required");
    assert_eq!(err["error"]["paths"], json!(["server"]));
    let (status, err) = patch(&admin, json!({"routing": {"epsilon": -2}})).await;
    assert_eq!(
        (status, err["error"]["type"].as_str()),
        (400, Some("invalid_config"))
    );
    let (status, err) = patch(&admin, json!({"routing": {"epsilonn": 1}})).await;
    assert_eq!(status, 400, "{err}");
    assert!(
        err["error"]["message"]
            .as_str()
            .unwrap()
            .contains("unknown field")
    );
    let raw = reqwest::Client::new()
        .patch(format!("{admin}/admin/config"))
        .body("{not json")
        .send()
        .await
        .unwrap();
    assert_eq!(raw.status(), 400);
    let (status, err) = patch(&admin, json!({"replicas": null})).await;
    assert_eq!(status, 400, "{err}");

    let (_, shown) = get(&admin, "/admin/config").await;
    assert_eq!(shown["generation"], 2);
    let (status, done) = patch(&admin, json!({"retry": {"max_attempts": 3}})).await;
    assert_eq!((status, done["changed"].clone()), (200, json!([])));
    let page = gw.gateway.render_metrics();
    assert!(
        page.contains("inferrail_config_reloads_total{result=\"rejected\"} 5"),
        "{page}"
    );
}

#[tokio::test]
async fn patching_replicas_moves_traffic_without_dropping_any() {
    let sims = [
        Upstream::spawn(sim_config(&[MODEL], decode(3.0))).await,
        Upstream::spawn(sim_config(&[MODEL], decode(3.0))).await,
    ];
    let both = urls(&[&sims[0], &sims[1]]);
    let gw = TestGateway::start(fleet(&both[..1], FAST_HEALTH)).await;
    let admin = serve_admin(&gw.gateway).await;
    let mut tasks = Vec::new();
    for i in 0..120u64 {
        let (client, url) = (gw.client.clone(), gw.url("/v1/chat/completions"));
        tasks.push(tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(i * 10)).await;
            let response = client
                .post(url)
                .json(&chat(&format!("p{i}"), 20))
                .send()
                .await
                .expect("sent");
            let status = response.status().as_u16();
            let replica = response.headers()["x-inferrail-replica"]
                .to_str()
                .unwrap()
                .to_owned();
            let mut body = response.bytes_stream();
            while let Some(chunk) = body.next().await {
                chunk.expect("chunk");
            }
            (status, replica)
        }));
    }
    tokio::time::sleep(Duration::from_millis(400)).await;
    let joined = json!({"replicas": [
        {"id": "r0", "url": both[0], "models": [MODEL]},
        {"id": "r1", "url": both[1], "models": [MODEL]},
    ]});
    assert_eq!(patch(&admin, joined).await.0, 200);
    tokio::time::sleep(Duration::from_millis(400)).await;
    let moved = json!({"replicas": [{"id": "r1", "url": both[1], "models": [MODEL]}]});
    assert_eq!(patch(&admin, moved).await.0, 200);
    let mut seen = std::collections::BTreeMap::new();
    for task in tasks {
        let (status, replica) = task.await.expect("request");
        assert_eq!(status, 200);
        *seen.entry(replica).or_insert(0) += 1;
    }
    assert!(seen["r0"] > 10 && seen["r1"] > 10, "{seen:?}");
}

#[tokio::test]
async fn reload_rereads_the_file_it_started_from() {
    let upstream = Upstream::zero(&[MODEL]).await;
    let config = fleet(&urls(&[&upstream]), FAST_HEALTH);
    let gw = TestGateway::start(config.clone()).await;
    let admin = serve_admin(&gw.gateway).await;
    let (status, err) = send(reqwest::Method::POST, &admin, "/admin/reload", None).await;
    assert_eq!(
        (status, err["error"]["type"].as_str()),
        (404, Some("no_config_file"))
    );

    let dir = std::env::temp_dir().join(format!("inferrail-admin-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("gateway.yaml");
    let mut next = config.clone();
    next.routing.default_policy = PolicyKind::P2c;
    std::fs::write(&path, next.to_yaml_string().unwrap()).unwrap();
    gw.gateway.set_config_path(path.clone());
    let (status, done) = send(reqwest::Method::POST, &admin, "/admin/reload", None).await;
    assert_eq!((status, done["generation"].as_u64()), (200, Some(2)));
    assert_eq!(gw.gateway.config().routing.default_policy, PolicyKind::P2c);
    std::fs::write(&path, "models: [").unwrap();
    let (status, err) = send(reqwest::Method::POST, &admin, "/admin/reload", None).await;
    assert_eq!(
        (status, err["error"]["type"].as_str()),
        (400, Some("unreadable_config"))
    );
    std::fs::remove_dir_all(&dir).unwrap();
    let (status, err) = send(reqwest::Method::POST, &admin, "/admin/reload", None).await;
    assert_eq!(status, 400, "{err}");
}

#[tokio::test]
async fn the_fleet_and_the_index_are_shown_as_routing_sees_them() {
    let sims = [
        Upstream::zero(&[MODEL]).await,
        Upstream::zero(&[MODEL]).await,
    ];
    let gw = TestGateway::start(tokenized(&[&sims[0], &sims[1]])).await;
    let admin = serve_admin(&gw.gateway).await;
    let (_, empty) = get(&admin, "/admin/index").await;
    assert_eq!(
        (empty["enabled"].as_bool(), empty["entries"].as_u64()),
        (Some(true), Some(0))
    );
    let long: String = (0..400).map(|i| format!("word{i} ")).collect();
    for i in 0..4 {
        let (status, _) = gw
            .post_json("/v1/chat/completions", &chat(&format!("{long}{i}"), 1))
            .await;
        assert_eq!(status, 200);
    }
    let (status, index) = get(&admin, "/admin/index").await;
    assert_eq!(status, 200);
    assert_eq!(index["mode"], "approximate");
    assert!(index["entries"].as_u64().unwrap() > 20, "{index}");
    let held: u64 = index["replicas"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| r["blocks"].as_u64().unwrap())
        .sum();
    assert!(held >= index["entries"].as_u64().unwrap());
    assert!(index["bytes_per_entry"].as_f64().unwrap() > 0.0);

    sims[1].kill();
    let (status, fleet) = get(&admin, "/admin/replicas").await;
    assert_eq!(status, 200);
    let rows = fleet["replicas"].as_array().unwrap();
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0]["id"], "r0");
    assert_eq!(rows[0]["origin"], "static");
    assert_eq!(rows[0]["models"], json!([MODEL]));
    for key in ["routable", "probe_up", "in_flight", "queued_secs", "engine"] {
        assert!(rows[0].get(key).is_some(), "{key} missing: {}", rows[0]);
    }
    common::wait_until(Duration::from_secs(3), || {
        !gw.replica("r1").health().probe_up()
    })
    .await;
    let (_, fleet) = get(&admin, "/admin/replicas").await;
    assert_eq!(fleet["replicas"][1]["probe_up"], false);
    assert_eq!(fleet["replicas"][1]["routable"], false);
}

#[tokio::test]
async fn without_an_index_the_index_says_so() {
    let upstream = Upstream::zero(&[MODEL]).await;
    let gw = TestGateway::start(fleet(&urls(&[&upstream]), FAST_HEALTH)).await;
    let admin = serve_admin(&gw.gateway).await;
    let (status, index) = get(&admin, "/admin/index").await;
    assert_eq!(status, 200);
    assert_eq!(index["enabled"], false);
}
