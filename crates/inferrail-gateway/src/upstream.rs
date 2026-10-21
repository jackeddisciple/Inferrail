use std::sync::Arc;
use std::time::Duration;

use http_body_util::{BodyExt, Full, Limited};
use hyper::Request;
use hyper::body::Bytes;
use hyper_util::client::legacy::connect::HttpConnector;
use hyper_util::rt::{TokioExecutor, TokioTimer};
use inferrail_control::{Probe, Replica};
use inferrail_core::config::UpstreamConfig;
use inferrail_core::openai::ModelList;

pub(crate) type Client = hyper_util::client::legacy::Client<HttpConnector, Full<Bytes>>;

pub(crate) fn client(config: &UpstreamConfig) -> Client {
    let mut http = HttpConnector::new();
    http.set_nodelay(true);
    http.set_connect_timeout(Some(config.connect_timeout));
    let mut builder = hyper_util::client::legacy::Client::builder(TokioExecutor::new());
    builder
        .pool_timer(TokioTimer::new())
        .pool_idle_timeout(config.pool_idle_timeout)
        .pool_max_idle_per_host(config.pool_max_idle_per_host)
        .http2_only(config.http2);
    builder.build(http)
}

const MAX_MODEL_LIST: usize = 1 << 20;

/// `GET /health`, passing on any 2xx within the timeout. A healthy replica
/// also has its `/v1/models` cards refreshed every `cards_every`.
#[derive(Debug, Clone)]
pub(crate) struct HttpProbe {
    pub(crate) client: Client,
    pub(crate) cards_every: Duration,
}

impl HttpProbe {
    async fn get(&self, replica: &Replica, path: &str, limit: usize) -> Option<(bool, Bytes)> {
        let request = Request::get(replica.endpoint().uri_for(path))
            .body(Full::default())
            .ok()?;
        let response = self.client.request(request).await.ok()?;
        let ok = response.status().is_success();
        let body = Limited::new(response.into_body(), limit)
            .collect()
            .await
            .ok()?;
        Some((ok, body.to_bytes()))
    }

    async fn refresh_cards(&self, replica: &Replica) {
        if let Some((true, body)) = self.get(replica, "/v1/models", MAX_MODEL_LIST).await {
            match serde_json::from_slice::<ModelList>(&body) {
                Ok(list) => replica.set_cards(list.data),
                Err(err) => {
                    tracing::debug!(replica = %replica.id(), error = %err, "unreadable model list");
                }
            }
        }
    }
}

impl Probe for HttpProbe {
    async fn probe(&self, replica: Arc<Replica>, timeout: Duration) -> bool {
        let healthy = matches!(
            tokio::time::timeout(timeout, self.get(&replica, "/health", 4096)).await,
            Ok(Some((true, _)))
        );
        let stale = replica
            .cards()
            .is_none_or(|c| c.fetched.elapsed() >= self.cards_every);
        if healthy && stale {
            let _ = tokio::time::timeout(timeout, self.refresh_cards(&replica)).await;
        }
        healthy
    }
}
