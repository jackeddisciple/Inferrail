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

const MAX_AGE: Duration = Duration::from_millis(250);

fn age(replica: &inferrail_control::Replica) -> Option<Duration> {
    replica
        .engine_load()
        .latest()
        .map(|s| s.age(tokio::time::Instant::now()))
}

async fn until_scraped(gw: &TestGateway, ids: &[&str]) {
    let replicas: Vec<_> = ids.iter().map(|id| gw.replica(id)).collect();
    wait_until(Duration::from_secs(3), || {
        replicas.iter().all(|r| r.engine_load().latest().is_some())
    })
    .await;
}

#[tokio::test]
async fn every_snapshot_stays_under_250_ms_old() {
    let sims = [
        Upstream::zero(&[MODEL]).await,
        Upstream::zero(&[MODEL]).await,
        Upstream::zero(&[MODEL]).await,
    ];
    let gw = TestGateway::start(fleet(&urls(&[&sims[0], &sims[1], &sims[2]]), FAST_HEALTH)).await;
    until_scraped(&gw, &["r0", "r1", "r2"]).await;
    let replicas: Vec<_> = ["r0", "r1", "r2"].iter().map(|id| gw.replica(id)).collect();
    let mut worst = Duration::ZERO;
    let started = std::time::Instant::now();
    while started.elapsed() < Duration::from_secs(3) {
        for replica in &replicas {
            let load = replica.load(tokio::time::Instant::now(), MAX_AGE);
            assert!(
                !load.is_stale(),
                "{} went stale: {:?}",
                replica.id(),
                load.age
            );
            worst = worst.max(load.age.unwrap());
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    assert!(worst < MAX_AGE, "worst age {worst:?}");
}

#[tokio::test]
async fn engine_queues_show_up_in_the_snapshot() {
    let mut config = sim_config(&[MODEL], slow_prefill());
    config.engine.max_num_seqs = 1;
    let sim = Upstream::spawn(config).await;
    let gw = TestGateway::start(fleet(&urls(&[&sim]), FAST_HEALTH)).await;
    let replica = gw.replica("r0");
    let requests: Vec<_> = (0..3)
        .map(|i| {
            let (client, url) = (gw.client.clone(), gw.url("/v1/chat/completions"));
            tokio::spawn(async move {
                client
                    .post(url)
                    .json(&chat(&format!("queue {i}"), 4))
                    .send()
                    .await
            })
        })
        .collect();
    wait_until(Duration::from_secs(2), || {
        replica
            .load(tokio::time::Instant::now(), MAX_AGE)
            .engine
            .is_some_and(|e| e.running == 1 && e.waiting == 2)
    })
    .await;
    let load = replica.load(tokio::time::Instant::now(), MAX_AGE);
    assert_eq!(load.in_flight, 3);
    assert!(load.engine.unwrap().kv_usage > 0.0);
    for request in requests {
        assert_eq!(request.await.unwrap().unwrap().status(), 200);
    }
    wait_until(Duration::from_secs(2), || {
        replica
            .load(tokio::time::Instant::now(), MAX_AGE)
            .engine
            .is_some_and(|e| e.running == 0 && e.waiting == 0 && e.prefix_queries > 0)
    })
    .await;
}

#[tokio::test]
async fn a_killed_engine_goes_stale_while_the_others_stay_fresh() {
    let sims = [
        Upstream::zero(&[MODEL]).await,
        Upstream::zero(&[MODEL]).await,
    ];
    let gw = TestGateway::start(fleet(&urls(&[&sims[0], &sims[1]]), FAST_HEALTH)).await;
    until_scraped(&gw, &["r0", "r1"]).await;
    let (r0, r1) = (gw.replica("r0"), gw.replica("r1"));
    sims[1].kill();
    wait_until(Duration::from_secs(2), || {
        r1.load(tokio::time::Instant::now(), MAX_AGE).is_stale()
    })
    .await;
    for i in 0..20 {
        let (status, _) = gw
            .post_json("/v1/chat/completions", &chat(&format!("still {i}"), 2))
            .await;
        assert_eq!(status, 200);
        assert!(age(&r0).unwrap() < MAX_AGE);
    }
    sims[1].restart().await;
    wait_until(Duration::from_secs(3), || {
        !r1.load(tokio::time::Instant::now(), MAX_AGE).is_stale()
    })
    .await;
}

#[derive(Debug, Clone, Copy)]
enum Page {
    Garbage,
    Hang,
    Huge,
    Trickle,
}

async fn hostile(page: Page) -> String {
    use axum::body::Body;
    use axum::routing::get;
    let metrics = move || async move {
        match page {
            Page::Garbage => Body::from("hello\nworld {\n# nothing here\n"),
            Page::Hang => {
                std::future::pending::<()>().await;
                Body::empty()
            }
            Page::Huge => Body::from_stream(futures::stream::iter(
                (0..50).map(|_| Ok::<_, std::io::Error>(vec![b'#'; 1 << 20])),
            )),
            Page::Trickle => Body::from_stream(futures::stream::unfold((), |()| async {
                tokio::time::sleep(Duration::from_millis(10)).await;
                Some((Ok::<_, std::io::Error>(b"v".to_vec()), ()))
            })),
        }
    };
    let app = axum::Router::new()
        .route("/health", get(|| async { "" }))
        .route("/metrics", get(metrics));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind");
    let url = format!("http://{}", listener.local_addr().expect("addr"));
    tokio::spawn(async move { axum::serve(listener, app).await });
    url
}

#[tokio::test]
async fn hostile_metrics_pages_never_publish_and_never_slow_the_rest() {
    let sim = Upstream::zero(&[MODEL]).await;
    let mut text = format!(
        "models:\n  - name: {MODEL}\n  - name: other\nreplicas:\n  - id: good\n    url: {}\n    models: [{MODEL}]\n",
        sim.url()
    );
    for (i, page) in [Page::Garbage, Page::Hang, Page::Huge, Page::Trickle]
        .into_iter()
        .enumerate()
    {
        text.push_str(&format!(
            "  - id: bad{i}\n    url: {}\n    models: [other]\n",
            hostile(page).await
        ));
    }
    let gw = TestGateway::start(config(&format!("{text}{FAST_HEALTH}"))).await;
    until_scraped(&gw, &["good"]).await;
    let good = gw.replica("good");
    let started = std::time::Instant::now();
    while started.elapsed() < Duration::from_millis(1500) {
        assert!(age(&good).unwrap() < MAX_AGE, "{:?}", age(&good));
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    for i in 0..4 {
        assert!(
            gw.replica(&format!("bad{i}"))
                .engine_load()
                .latest()
                .is_none()
        );
    }
    let (status, _) = gw.post_json("/v1/chat/completions", &chat("fine", 2)).await;
    assert_eq!(status, 200);
}

#[tokio::test]
async fn scraping_can_be_turned_off() {
    let sim = Upstream::zero(&[MODEL]).await;
    let gw = TestGateway::start(fleet(
        &urls(&[&sim]),
        &format!("{FAST_HEALTH}load:\n  enabled: false\n"),
    ))
    .await;
    tokio::time::sleep(Duration::from_millis(300)).await;
    let load = gw.replica("r0").load(tokio::time::Instant::now(), MAX_AGE);
    assert!(load.is_stale());
    assert_eq!(load.age, None);
}
