//! The gateway against three in process sims: routing, pass through,
//! request ids, aliases and errors.

mod common;

use std::collections::HashMap;
use std::time::Duration;

use common::*;
use futures::StreamExt;
use hyper::body::Bytes;
use serde_json::{Value, json};

fn without_created(mut value: Value) -> Value {
    if let Some(object) = value.as_object_mut() {
        object.remove("created");
    }
    value
}

fn normalize(raw: &[u8], stream: bool) -> Vec<Value> {
    if !stream {
        return vec![without_created(serde_json::from_slice(raw).expect("json"))];
    }
    let text = std::str::from_utf8(raw).expect("utf8");
    assert!(text.ends_with("data: [DONE]\n\n"), "{text}");
    text.split("\n\n")
        .filter(|frame| !frame.is_empty())
        .map(|frame| {
            let data = frame.strip_prefix("data: ").expect("data frame");
            if data == "[DONE]" {
                Value::from("[DONE]")
            } else {
                without_created(serde_json::from_str(data).expect("chunk json"))
            }
        })
        .collect()
}

async fn send(
    client: &reqwest::Client,
    url: String,
    id: &str,
    body: &Value,
) -> (u16, String, Bytes) {
    let response = client
        .post(url)
        .header("x-request-id", id)
        .json(body)
        .send()
        .await
        .expect("sent");
    let status = response.status().as_u16();
    let kind = response
        .headers()
        .get("content-type")
        .map(|v| v.to_str().expect("ascii").to_owned())
        .unwrap_or_default();
    (status, kind, response.bytes().await.expect("body"))
}

#[tokio::test]
async fn responses_match_a_direct_request_to_a_sim() {
    let fleet_sims = [
        Upstream::zero(&[MODEL]).await,
        Upstream::zero(&[MODEL]).await,
        Upstream::zero(&[MODEL]).await,
    ];
    let reference = Upstream::zero(&[MODEL]).await;
    let gw = TestGateway::start(fleet(
        &urls(&fleet_sims.iter().collect::<Vec<_>>()),
        "routing:\n  default_policy: round_robin\n",
    ))
    .await;
    let client = reqwest::Client::new();
    let cases = [
        ("/v1/chat/completions", chat("tell me a story", 12), false),
        (
            "/v1/chat/completions",
            streaming(chat("stream a story", 12)),
            true,
        ),
        (
            "/v1/completions",
            json!({"model": MODEL, "prompt": "Once upon", "max_tokens": 7}),
            false,
        ),
        (
            "/v1/completions",
            json!({"model": MODEL, "prompt": [1, 2, 3], "max_tokens": 5, "stream": true}),
            true,
        ),
    ];
    for (i, (path, body, stream)) in cases.iter().enumerate() {
        for round in 0..3 {
            let id = format!("case-{i}-{round}");
            let via = send(&client, gw.url(path), &id, body).await;
            let direct = send(&client, format!("{}{path}", reference.url()), &id, body).await;
            assert_eq!(via.0, 200, "{}", String::from_utf8_lossy(&via.2));
            assert_eq!((via.0, &via.1), (direct.0, &direct.1));
            assert_eq!(
                normalize(&via.2, *stream),
                normalize(&direct.2, *stream),
                "{path}"
            );
        }
    }
    let hits: Vec<u64> = fleet_sims.iter().map(Upstream::hits).collect();
    assert_eq!(hits, [4, 4, 4]);
    gw.wait_idle().await;
}

#[tokio::test]
async fn request_ids_are_kept_or_minted_and_reach_the_engine() {
    let sim = Upstream::zero(&[MODEL]).await;
    let gw = TestGateway::start(fleet(&urls(&[&sim]), "")).await;
    let response = gw
        .client
        .post(gw.url("/v1/chat/completions"))
        .header("x-request-id", "client-abc")
        .json(&chat("hi", 2))
        .send()
        .await
        .unwrap();
    assert_eq!(response.headers()["x-request-id"], "client-abc");
    assert_eq!(response.headers()["x-inferrail-replica"], "r0");
    let body: Value = response.json().await.unwrap();
    assert_eq!(body["id"], "chatcmpl-client-abc");

    let response = gw.post("/v1/chat/completions", &chat("hi", 2)).await;
    let minted = response.headers()["x-request-id"]
        .to_str()
        .unwrap()
        .to_owned();
    assert_eq!(minted.len(), 32);
    let body: Value = response.json().await.unwrap();
    assert_eq!(body["id"], format!("chatcmpl-{minted}"));

    let response = gw.client.get(gw.url("/nope")).send().await.unwrap();
    assert_eq!(response.status(), 404);
    assert_eq!(response.headers()["x-request-id"].len(), 32);
}

#[tokio::test]
async fn aliases_and_missing_models_are_resolved_for_the_engine() {
    let sim = Upstream::zero(&[MODEL]).await;
    let text = format!(
        "models:\n  - name: {MODEL}\n    aliases: [friendly]\nreplicas:\n  - id: r0\n    url: {}\n    models: [{MODEL}]\n",
        sim.url()
    );
    let gw = TestGateway::start(config(&text)).await;
    let mut body = chat("hi", 2);
    body["model"] = json!("friendly");
    let (status, reply) = gw.post_json("/v1/chat/completions", &body).await;
    assert_eq!(status, 200, "{reply}");
    assert_eq!(reply["model"], MODEL);

    for missing in [
        json!({"messages": [{"role": "user", "content": "x"}], "max_tokens": 1}),
        json!({"model": null, "messages": [{"role": "user", "content": "x"}], "max_tokens": 1}),
    ] {
        let (status, reply) = gw.post_json("/v1/chat/completions", &missing).await;
        assert_eq!(status, 200, "{reply}");
        assert_eq!(reply["model"], MODEL);
    }
}

#[tokio::test]
async fn a_missing_model_is_ambiguous_with_two_models() {
    let sim = Upstream::zero(&["a", "b"]).await;
    let text = format!(
        "models:\n  - name: a\n  - name: b\nreplicas:\n  - id: r0\n    url: {}\n    models: [a, b]\n",
        sim.url()
    );
    let gw = TestGateway::start(config(&text)).await;
    let (status, reply) = gw
        .post_json("/v1/chat/completions", &json!({"messages": []}))
        .await;
    assert_eq!(status, 400);
    assert_eq!(reply["error"]["param"], "model");
    assert_eq!(sim.hits(), 0);
}

#[tokio::test]
async fn each_model_uses_its_own_policy() {
    let sims = [
        Upstream::spawn(sim_config(&["even", "busy"], decode(5.0))).await,
        Upstream::spawn(sim_config(&["even", "busy"], decode(5.0))).await,
        Upstream::spawn(sim_config(&["even", "busy"], decode(5.0))).await,
    ];
    let mut text = String::from(
        "models:\n  - name: even\n    policy: round_robin\n  - name: busy\n    policy: least_request\nreplicas:\n",
    );
    for (i, sim) in sims.iter().enumerate() {
        text.push_str(&format!(
            "  - id: r{i}\n    url: {}\n    models: [even, busy]\n",
            sim.url()
        ));
    }
    let gw = TestGateway::start(config(&text)).await;
    let mut served: HashMap<String, u32> = HashMap::new();
    for _ in 0..30 {
        let mut body = chat("x", 1);
        body["model"] = json!("even");
        let response = gw.post("/v1/chat/completions", &body).await;
        let replica = response.headers()["x-inferrail-replica"]
            .to_str()
            .unwrap()
            .to_owned();
        *served.entry(replica).or_default() += 1;
    }
    assert_eq!(served.values().copied().collect::<Vec<_>>(), [10, 10, 10]);

    let mut long = streaming(chat("hold the slot", 400));
    long["model"] = json!("busy");
    let hold = gw.post("/v1/chat/completions", &long).await;
    let held = hold.headers()["x-inferrail-replica"]
        .to_str()
        .unwrap()
        .to_owned();
    let mut body = chat("x", 1);
    body["model"] = json!("busy");
    for _ in 0..20 {
        let response = gw.post("/v1/chat/completions", &body).await;
        assert_ne!(response.headers()["x-inferrail-replica"], held.as_str());
    }
    drop(hold);
}

#[tokio::test]
async fn gateway_errors_are_openai_shaped() {
    let sim = Upstream::zero(&[MODEL]).await;
    let gw = TestGateway::start(fleet(&urls(&[&sim]), "server:\n  max_body_bytes: 4096\n")).await;
    let (status, reply) = gw
        .post_json(
            "/v1/chat/completions",
            &json!({"model": "gpt-4", "messages": []}),
        )
        .await;
    assert_eq!(status, 404);
    assert_eq!(
        reply["error"]["message"],
        "The model `gpt-4` does not exist."
    );

    for (body, param) in [
        (&b"{not json"[..], Value::Null),
        (b"[1, 2]", Value::Null),
        (b"{\"model\": 7}", json!("model")),
    ] {
        let response = gw
            .client
            .post(gw.url("/v1/chat/completions"))
            .header("content-type", "application/json")
            .body(body.to_vec())
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), 400);
        let reply: Value = response.json().await.unwrap();
        assert_eq!(reply["error"]["type"], "BadRequestError");
        assert_eq!(reply["error"]["param"], param);
    }

    let big = chat(&"x".repeat(5_000), 1);
    let (status, reply) = gw.post_json("/v1/chat/completions", &big).await;
    assert_eq!(status, 413, "{reply}");
    let chunks = futures::stream::iter(
        (0..10).map(|_| Ok::<_, std::io::Error>(Bytes::from(vec![b' '; 1_000]))),
    );
    let response = gw
        .client
        .post(gw.url("/v1/completions"))
        .body(reqwest::Body::wrap_stream(chunks))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 413);

    let response = gw
        .client
        .get(gw.url("/v1/chat/completions"))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 405);
    assert_eq!(response.headers()["allow"], "POST");
    let response = gw.client.post(gw.url("/v1/models")).send().await.unwrap();
    assert_eq!(response.status(), 405);
    assert_eq!(response.headers()["allow"], "GET");
    let response = gw
        .client
        .get(gw.url("/v1/embeddings"))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 404);
    let response = gw.client.get(gw.url("/health")).send().await.unwrap();
    assert_eq!(response.status(), 200);
    assert_eq!(sim.hits(), 0);
}

#[tokio::test]
async fn engine_errors_pass_through_untouched() {
    let sim = Upstream::zero(&[MODEL]).await;
    let gw = TestGateway::start(fleet(&urls(&[&sim]), "")).await;
    let mut body = chat("x", 1);
    body["n"] = json!(2);
    let client = reqwest::Client::new();
    let via = send(&client, gw.url("/v1/chat/completions"), "e1", &body).await;
    let direct = send(
        &client,
        format!("{}/v1/chat/completions", sim.url()),
        "e1",
        &body,
    )
    .await;
    assert_eq!(via.0, 400);
    assert_eq!(via, direct);
}

#[tokio::test]
async fn hop_by_hop_headers_never_cross_and_bodies_arrive_unchanged() {
    let raw = RawUpstream::spawn().await;
    raw.script(Script {
        headers: vec![
            ("content-type", "application/json".to_owned()),
            ("connection", "x-upstream-hop".to_owned()),
            ("x-upstream-hop", "secret".to_owned()),
            ("x-engine", "kept".to_owned()),
        ],
        chunks: vec![Bytes::from_static(b"{\"ok\":true}")],
        ..Script::default()
    });
    let gw = TestGateway::start(fleet(&[raw.url()], "")).await;
    let body = format!(
        "{{ \"messages\" : [ {{\"role\":\"user\",\"content\":\"caf\\u00e9 \u{1f600}\"}} ],\n  \"model\":\"{MODEL}\",  \"top_k\" : 40 , \"temperature\": 1.0 }}"
    );
    let response = gw
        .client
        .post(gw.url("/v1/chat/completions?api-version=2"))
        .header("content-type", "application/json")
        .header("authorization", "Bearer secret-key")
        .header("connection", "x-client-hop")
        .header("x-client-hop", "1")
        .header("te", "trailers")
        .header("x-inferrail-replica", "spoofed")
        .header("x-custom", "kept")
        .body(body.clone())
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    let headers = response.headers();
    assert_eq!(headers["x-engine"], "kept");
    assert!(headers.get("x-upstream-hop").is_none());
    assert_eq!(headers["x-inferrail-replica"], "r0");
    assert_eq!(response.text().await.unwrap(), "{\"ok\":true}");

    let seen = raw.take();
    assert_eq!(seen.len(), 1);
    let request = &seen[0];
    assert_eq!(request.body, body.as_bytes());
    assert_eq!(request.uri, "/v1/chat/completions?api-version=2");
    assert_eq!(request.method, "POST");
    assert_eq!(request.headers["authorization"], "Bearer secret-key");
    assert_eq!(request.headers["x-custom"], "kept");
    assert_eq!(
        request.headers["content-length"],
        body.len().to_string().as_str()
    );
    assert_eq!(request.headers["x-request-id"].len(), 32);
    for gone in ["x-client-hop", "te", "x-inferrail-replica"] {
        assert!(request.headers.get(gone).is_none(), "{gone} crossed");
    }
    assert_eq!(request.headers["host"], raw.addr.to_string().as_str());
}

#[tokio::test]
async fn http2_works_inbound_and_upstream() {
    let raw = RawUpstream::spawn().await;
    let gw = TestGateway::start(fleet(&[raw.url()], "upstream:\n  http2: true\n")).await;
    let h2: hyper_util::client::legacy::Client<_, http_body_util::Full<Bytes>> =
        hyper_util::client::legacy::Client::builder(hyper_util::rt::TokioExecutor::new())
            .http2_only(true)
            .build_http();
    let request = hyper::Request::post(gw.url("/v1/chat/completions"))
        .header("content-type", "application/json")
        .body(http_body_util::Full::new(Bytes::from(
            chat("x", 1).to_string(),
        )))
        .unwrap();
    let response = h2.request(request).await.unwrap();
    assert_eq!(response.version(), hyper::Version::HTTP_2);
    assert_eq!(response.status(), 200);
    let body = http_body_util::BodyExt::collect(response.into_body())
        .await
        .unwrap()
        .to_bytes();
    assert_eq!(body, "{}");
    let seen = raw.take();
    assert_eq!(seen[0].version, hyper::Version::HTTP_2);
}

#[tokio::test]
async fn hundreds_of_concurrent_streams_complete_and_drain() {
    let sims = [
        Upstream::spawn(sim_config(&[MODEL], decode(1.0))).await,
        Upstream::spawn(sim_config(&[MODEL], decode(1.0))).await,
        Upstream::spawn(sim_config(&[MODEL], decode(1.0))).await,
    ];
    let gw = TestGateway::start(fleet(&urls(&sims.iter().collect::<Vec<_>>()), "")).await;
    let mut tasks = Vec::new();
    for i in 0..300 {
        let client = gw.client.clone();
        let url = gw.url("/v1/chat/completions");
        let body = streaming(chat(&format!("stream {i}"), 16));
        tasks.push(tokio::spawn(async move {
            let response = client.post(url).json(&body).send().await.expect("sent");
            assert_eq!(response.status(), 200);
            let mut frames = 0;
            let mut stream = response.bytes_stream();
            let mut text = Vec::new();
            while let Some(chunk) = stream.next().await {
                text.extend_from_slice(&chunk.expect("chunk"));
            }
            for frame in String::from_utf8(text).unwrap().split("\n\n") {
                if frame.starts_with("data: ") {
                    frames += 1;
                }
            }
            frames
        }));
    }
    for task in tasks {
        assert_eq!(task.await.unwrap(), 16 + 3);
    }
    gw.wait_idle().await;
    let hits: u64 = sims.iter().map(Upstream::hits).sum();
    assert_eq!(hits, 300);
    assert!(sims.iter().all(|s| s.hits() > 50), "uneven spread");
    tokio::time::sleep(Duration::from_millis(20)).await;
    assert!(sims.iter().all(|s| s.snapshot().running == 0));
}
