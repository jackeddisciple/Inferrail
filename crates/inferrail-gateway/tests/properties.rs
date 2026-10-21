//! Randomized checks through a live gateway: bytes in equal bytes out, in
//! both directions, and no input makes the gateway itself fail.

mod common;

use std::sync::OnceLock;
use std::time::Duration;

use common::*;
use hyper::body::Bytes;
use proptest::prelude::*;
use serde_json::{Value, json};
use tokio::runtime::Runtime;

struct Env {
    raw: RawUpstream,
    gw: TestGateway,
}

fn runtime() -> &'static Runtime {
    static RT: OnceLock<Runtime> = OnceLock::new();
    RT.get_or_init(|| Runtime::new().expect("runtime"))
}

fn env() -> &'static Env {
    static ENV: OnceLock<Env> = OnceLock::new();
    ENV.get_or_init(|| {
        runtime().block_on(async {
            let raw = RawUpstream::spawn().await;
            let gw = TestGateway::start(fleet(&[raw.url()], "server:\n  max_body_bytes: 262144\n"))
                .await;
            Env { raw, gw }
        })
    })
}

fn split(payload: &[u8], mut cuts: Vec<usize>) -> Vec<Bytes> {
    cuts.retain(|&c| c > 0 && c < payload.len());
    cuts.sort_unstable();
    cuts.dedup();
    let mut chunks = Vec::new();
    let mut start = 0;
    for cut in cuts.into_iter().chain([payload.len()]) {
        if cut > start {
            chunks.push(Bytes::copy_from_slice(&payload[start..cut]));
            start = cut;
        }
    }
    chunks
}

fn json_value() -> impl Strategy<Value = Value> {
    let leaf = prop_oneof![
        Just(Value::Null),
        any::<bool>().prop_map(Value::from),
        any::<i64>().prop_map(Value::from),
        any::<f64>()
            .prop_filter("finite", |f| f.is_finite())
            .prop_map(Value::from),
        "\\PC{0,40}".prop_map(Value::from),
    ];
    leaf.prop_recursive(3, 32, 6, |inner| {
        prop_oneof![
            prop::collection::vec(inner.clone(), 0..6).prop_map(Value::Array),
            prop::collection::btree_map("\\PC{1,10}", inner, 0..6)
                .prop_map(|m| Value::Object(m.into_iter().collect())),
        ]
    })
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 96, ..ProptestConfig::default() })]

    #[test]
    fn streamed_bytes_arrive_unchanged_under_any_chunking(
        payload in prop::collection::vec(any::<u8>(), 0..40_000),
        cuts in prop::collection::vec(0usize..40_000, 0..48),
        pause_us in prop_oneof![Just(0u64), 1u64..400],
        sse: bool,
    ) {
        let env = env();
        let kind = if sse { "text/event-stream; charset=utf-8" } else { "application/octet-stream" };
        env.raw.script(Script {
            status: 200,
            headers: vec![("content-type", kind.to_owned())],
            chunks: split(&payload, cuts),
            pause: Duration::from_micros(pause_us),
        });
        let (status, got) = runtime().block_on(async {
            let response = env.gw.post("/v1/chat/completions", &streaming(chat("x", 1))).await;
            (response.status().as_u16(), response.bytes().await.expect("body"))
        });
        prop_assert_eq!(status, 200);
        prop_assert_eq!(got.as_ref(), payload.as_slice());
        env.raw.take();
    }

    #[test]
    fn request_bodies_reach_the_engine_byte_for_byte(
        fields in prop::collection::btree_map("[a-z_]{1,12}", json_value(), 0..8),
        pretty: bool,
        padding in "[ \\t\\n\\r]{0,4}",
    ) {
        let env = env();
        env.raw.script(Script::default());
        let mut object: serde_json::Map<String, Value> =
            fields.into_iter().filter(|(k, _)| k != "model").collect();
        object.insert("model".to_owned(), json!(MODEL));
        let value = Value::Object(object);
        let text = if pretty {
            serde_json::to_string_pretty(&value).unwrap()
        } else {
            serde_json::to_string(&value).unwrap()
        };
        let body = format!("{padding}{text}{padding}");
        let status = runtime().block_on(async {
            env.gw
                .client
                .post(env.gw.url("/v1/completions"))
                .body(body.clone())
                .send()
                .await
                .expect("sent")
                .status()
                .as_u16()
        });
        prop_assert_eq!(status, 200);
        let seen = env.raw.take();
        prop_assert_eq!(seen.len(), 1);
        prop_assert_eq!(seen[0].body.as_ref(), body.as_bytes());
    }

    #[test]
    fn chunked_request_bodies_arrive_whole(
        fields in prop::collection::btree_map("[a-z_]{1,12}", json_value(), 0..8),
        cuts in prop::collection::vec(1usize..64, 1..400),
    ) {
        let env = env();
        env.raw.script(Script::default());
        let mut object: serde_json::Map<String, Value> =
            fields.into_iter().filter(|(k, _)| k != "model").collect();
        object.insert("model".to_owned(), json!(MODEL));
        let body = serde_json::to_vec(&Value::Object(object)).unwrap();
        let mut chunks = Vec::new();
        let mut rest = body.as_slice();
        for cut in cuts.iter().cycle() {
            if rest.is_empty() {
                break;
            }
            let (head, tail) = rest.split_at((*cut).min(rest.len()));
            chunks.push(Ok::<_, std::io::Error>(Bytes::copy_from_slice(head)));
            rest = tail;
        }
        let status = runtime().block_on(async {
            env.gw
                .client
                .post(env.gw.url("/v1/completions"))
                .body(reqwest::Body::wrap_stream(futures::stream::iter(chunks)))
                .send()
                .await
                .expect("sent")
                .status()
                .as_u16()
        });
        prop_assert_eq!(status, 200);
        let seen = env.raw.take();
        prop_assert_eq!(seen.len(), 1);
        prop_assert_eq!(seen[0].body.as_ref(), body.as_slice());
    }

    #[test]
    fn no_request_makes_the_gateway_fail(
        method in prop::sample::select(vec!["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"]),
        path in prop::sample::select(vec![
            "/v1/chat/completions", "/v1/completions", "/v1/models", "/health", "/", "/v1",
            "/v1/chat/completions/", "/V1/MODELS", "/v1/embeddings", "/metrics", "/v1/completions?x=1",
        ]),
        body in prop_oneof![
            prop::collection::vec(any::<u8>(), 0..300),
            json_value().prop_map(|v| serde_json::to_vec(&v).unwrap()),
            json_value().prop_map(|v| serde_json::to_vec(&json!({"model": MODEL, "x": v})).unwrap()),
        ],
        request_id in prop::option::of("[ -~]{0,300}"),
    ) {
        let env = env();
        env.raw.script(Script::default());
        let (status, has_id) = runtime().block_on(async {
            let method = reqwest::Method::from_bytes(method.as_bytes()).unwrap();
            let mut request = env.gw.client.request(method, env.gw.url(path)).body(body);
            if let Some(id) = request_id {
                request = request.header("x-request-id", id);
            }
            let response = request.send().await.expect("sent");
            (response.status().as_u16(), response.headers().contains_key("x-request-id"))
        });
        prop_assert!(status < 500, "status {}", status);
        prop_assert!(has_id);
        env.raw.take();
    }
}
