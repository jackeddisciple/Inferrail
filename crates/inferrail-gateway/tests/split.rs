//! A/B splits: each policy routes its share of requests, a conversation
//! keeps to one policy, the share moves live, and metrics and headers tell
//! the arms apart.

mod common;

use std::collections::HashMap;

use common::{FAST_HEALTH, MODEL, TestGateway, Upstream, chat, fleet, serve_admin, urls};
use inferrail_core::config::PolicyKind;
use serde_json::json;

async fn policy_of(gw: &TestGateway, i: u64, session: Option<&str>) -> String {
    let mut request = gw
        .client
        .post(gw.url("/v1/chat/completions"))
        .json(&chat(&format!("q{i}"), 1));
    if let Some(session) = session {
        request = request.header("x-session-id", session);
    }
    let response = request.send().await.expect("sent");
    assert_eq!(response.status(), 200);
    let policy = response.headers()["x-inferrail-policy"]
        .to_str()
        .expect("ascii")
        .to_owned();
    response.bytes().await.expect("body");
    policy
}

async fn split_fleet(extra: &str) -> (Vec<Upstream>, TestGateway) {
    let sims = vec![
        Upstream::zero(&[MODEL]).await,
        Upstream::zero(&[MODEL]).await,
        Upstream::zero(&[MODEL]).await,
    ];
    let config = fleet(
        &urls(&[&sims[0], &sims[1], &sims[2]]),
        &format!("{FAST_HEALTH}{extra}"),
    );
    let gw = TestGateway::start(config).await;
    (sims, gw)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn each_policy_routes_its_share() {
    let (_sims, gw) =
        split_fleet("routing:\n  split: {round_robin: 75, least_request: 25}\n").await;
    let mut counts: HashMap<String, u32> = HashMap::new();
    for i in 0..800 {
        *counts.entry(policy_of(&gw, i, None).await).or_default() += 1;
    }
    assert!((540..660).contains(&counts["round_robin"]), "{counts:?}");
    assert!((140..260).contains(&counts["least_request"]), "{counts:?}");
    let page = gw.gateway.render_metrics();
    let decided = |policy: &str| -> u32 {
        page.lines()
            .filter(|l| {
                l.starts_with("inferrail_routing_decisions_total")
                    && l.contains(&format!("policy=\"{policy}\""))
            })
            .map(|l| l.rsplit_once(' ').unwrap().1.parse::<u32>().unwrap())
            .sum()
    };
    assert_eq!(decided("round_robin"), counts["round_robin"]);
    assert_eq!(decided("least_request"), counts["least_request"]);
    assert!(
        page.contains(
            "inferrail_model_policy_percent{model=\"model-a\",policy=\"round_robin\"} 75"
        )
    );
    assert!(
        page.contains(
            "inferrail_model_policy_percent{model=\"model-a\",policy=\"least_request\"} 25"
        )
    );
}

#[tokio::test]
async fn a_conversation_keeps_to_one_policy() {
    let (_sims, gw) = split_fleet("routing:\n  split: {p2c: 50, cost_model: 50}\n").await;
    let mut arms = HashMap::new();
    for session in 0..40 {
        let name = format!("conversation-{session}");
        let first = policy_of(&gw, 0, Some(&name)).await;
        for turn in 1..5 {
            assert_eq!(policy_of(&gw, turn, Some(&name)).await, first, "{name}");
        }
        *arms.entry(first).or_insert(0) += 1;
    }
    assert_eq!(arms.len(), 2, "every conversation took one arm: {arms:?}");
}

#[tokio::test]
async fn the_split_moves_live_and_models_can_split_on_their_own() {
    let (_sims, gw) = split_fleet("routing:\n  split: {round_robin: 50, random: 50}\n").await;
    let mut all_random = gw.gateway.config().as_ref().clone();
    all_random.routing.split = [(PolicyKind::Random, 100)].into();
    let done = gw.gateway.reload(all_random.clone()).unwrap();
    assert_eq!(done.changed, ["routing"]);
    for i in 0..30 {
        assert_eq!(policy_of(&gw, i, None).await, "random");
    }
    let mut own = all_random;
    own.models[0].split = Some([(PolicyKind::KvAware, 100)].into());
    gw.gateway.reload(own.clone()).unwrap();
    assert_eq!(policy_of(&gw, 99, None).await, "kv_aware");
    own.models[0].split = None;
    own.models[0].policy = Some(PolicyKind::P2c);
    gw.gateway.reload(own).unwrap();
    assert_eq!(policy_of(&gw, 100, None).await, "p2c");
    let page = gw.gateway.render_metrics();
    assert!(page.contains("inferrail_model_policy_percent{model=\"model-a\",policy=\"p2c\"} 100"));
    assert!(!page.contains("policy=\"kv_aware\"} 100"));
}

#[tokio::test]
async fn explain_names_the_arm_it_explains() {
    let (_sims, gw) = split_fleet("routing:\n  split: {least_request: 30, cost_model: 70}\n").await;
    let admin = serve_admin(&gw.gateway).await;
    let client = reqwest::Client::new();
    let explain = |query: &'static str| {
        let client = client.clone();
        let url = format!("{admin}/admin/explain/v1/chat/completions{query}");
        async move {
            let response = client.post(url).json(&chat("hi", 1)).send().await.unwrap();
            let status = response.status().as_u16();
            (status, response.json::<serde_json::Value>().await.unwrap())
        }
    };
    let (status, main) = explain("").await;
    assert_eq!((status, main["policy"].as_str()), (200, Some("cost_model")));
    let (_, other) = explain("?policy=least_request").await;
    assert_eq!(other["policy"], "least_request");
    let (status, missing) = explain("?policy=p2c").await;
    assert_eq!(status, 400, "{missing}");
    let (status, unknown) = explain("?policy=fastest").await;
    assert_eq!(status, 400, "{unknown}");
    assert_eq!(unknown["error"]["param"], json!("policy"));
}
