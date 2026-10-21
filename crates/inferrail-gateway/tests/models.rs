//! `/v1/models` lists configured models with the engines' own metadata,
//! straight from memory.

mod common;

use std::time::Duration;

use common::*;
use serde_json::Value;

async fn listing(gw: &TestGateway) -> Value {
    let response = gw
        .client
        .get(gw.url("/v1/models"))
        .send()
        .await
        .expect("sent");
    assert_eq!(response.status(), 200);
    assert_eq!(response.headers()["content-type"], "application/json");
    response.json().await.expect("json")
}

#[tokio::test]
async fn engine_cards_are_merged_in_config_order_with_aliases() {
    let mut config = sim_config(&[MODEL], inferrail_sim::latency::LatencyModel::ZERO);
    config.model = "org/base-8b".to_owned();
    config.max_model_len = 8_192;
    let sims = [
        Upstream::spawn(config.clone()).await,
        Upstream::spawn(config).await,
    ];
    let text = format!(
        "models:\n  - name: other\n  - name: {MODEL}\n    aliases: [friendly, short]\nreplicas:\n  - id: r0\n    url: {}\n    models: [{MODEL}, other]\n  - id: r1\n    url: {}\n    models: [{MODEL}]\n{FAST_HEALTH}",
        sims[0].url(),
        sims[1].url()
    );
    let gw = TestGateway::start(common::config(&text)).await;
    let r0 = gw.replica("r0");
    wait_until(Duration::from_secs(2), || r0.cards().is_some()).await;
    let list = listing(&gw).await;
    assert_eq!(list["object"], "list");
    let ids: Vec<&str> = list["data"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| c["id"].as_str().unwrap())
        .collect();
    assert_eq!(ids, ["other", MODEL, "friendly", "short"]);
    let data = &list["data"];
    assert_eq!(data[0]["owned_by"], "inferrail");
    assert_eq!(data[0]["root"], "other");
    for card in &data.as_array().unwrap()[1..] {
        assert_eq!(card["owned_by"], "vllm");
        assert_eq!(card["root"], "org/base-8b");
        assert_eq!(card["max_model_len"], 8_192);
    }

    sims[0].kill();
    sims[1].kill();
    wait_until(Duration::from_secs(2), || !r0.health().probe_up()).await;
    let list = listing(&gw).await;
    assert_eq!(list["data"][1]["max_model_len"], 8_192, "last known cards");
}

#[tokio::test]
async fn listing_never_calls_an_engine() {
    let raw = RawUpstream::spawn().await;
    let gw = TestGateway::start(fleet(
        &[raw.url()],
        "health:\n  interval: 10s\n  timeout: 1s\n",
    ))
    .await;
    wait_until(Duration::from_secs(2), || raw.model_fetches() > 0).await;
    let before = raw.model_fetches();
    for _ in 0..50 {
        let list = listing(&gw).await;
        assert_eq!(list["data"][0]["id"], MODEL);
        assert_eq!(list["data"][0]["owned_by"], "inferrail");
    }
    assert_eq!(raw.model_fetches(), before);
    assert!(raw.take().is_empty());
}
