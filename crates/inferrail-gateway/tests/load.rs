mod common;

use std::time::Duration;

use common::*;
use futures::StreamExt;
use inferrail_sim::latency::LatencyModel;

fn slow_prefill() -> LatencyModel {
    LatencyModel {
        prefill_fixed_secs: 0.3,
        decode_token_secs: 0.05,
        ..LatencyModel::ZERO
    }
}

#[tokio::test]
async fn prompt_tokens_count_while_in_flight_and_prefill_ends_at_the_first_byte() {
    let sim = Upstream::spawn(sim_config(&[MODEL], slow_prefill())).await;
    let gw = TestGateway::start(fleet(&urls(&[&sim]), FAST_HEALTH)).await;
    let replica = gw.replica("r0");
    let body = streaming(chat(&"a long prompt ".repeat(40), 8));
    let expected = serde_json::to_vec(&body).unwrap().len().div_ceil(4) as u64;

    let request = {
        let (client, url, body) = (
            gw.client.clone(),
            gw.url("/v1/chat/completions"),
            body.clone(),
        );
        tokio::spawn(async move {
            let response = client.post(url).json(&body).send().await.unwrap();
            let mut chunks = response.bytes_stream();
            while let Some(chunk) = chunks.next().await {
                chunk.unwrap();
            }
        })
    };
    wait_until(Duration::from_secs(2), || replica.in_flight_tokens() > 0).await;
    assert_eq!(replica.in_flight_tokens(), expected);
    assert_eq!(replica.prefill_tokens(), expected);

    wait_until(Duration::from_secs(2), || replica.prefill_tokens() == 0).await;
    assert_eq!(replica.in_flight_tokens(), expected);

    request.await.unwrap();
    gw.wait_idle().await;
    assert_eq!(
        (replica.in_flight_tokens(), replica.prefill_tokens()),
        (0, 0)
    );
}

#[tokio::test]
async fn a_client_that_leaves_returns_its_tokens() {
    let sim = Upstream::spawn(sim_config(&[MODEL], slow_prefill())).await;
    let gw = TestGateway::start(fleet(&urls(&[&sim]), FAST_HEALTH)).await;
    let replica = gw.replica("r0");
    let request = {
        let (client, url) = (gw.client.clone(), gw.url("/v1/chat/completions"));
        tokio::spawn(async move { client.post(url).json(&chat("hello", 50)).send().await })
    };
    wait_until(Duration::from_secs(2), || replica.prefill_tokens() > 0).await;
    request.abort();
    wait_until(Duration::from_secs(2), || {
        replica.in_flight_tokens() == 0 && replica.prefill_tokens() == 0
    })
    .await;
    gw.wait_idle().await;
}
