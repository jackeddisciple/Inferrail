use std::sync::Arc;
use std::time::Duration;

use http_body_util::{BodyExt, Full, Limited};
use hyper::Request;
use hyper::body::Bytes;
use hyper_util::client::legacy::connect::HttpConnector;
use hyper_util::rt::{TokioExecutor, TokioTimer};
use inferrail_control::{Probe, Replica};
use inferrail_core::config::UpstreamConfig;

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

/// `GET /health`, passing on any 2xx within the timeout.
#[derive(Debug, Clone)]
pub(crate) struct HttpProbe {
    pub(crate) client: Client,
}

impl Probe for HttpProbe {
    async fn probe(&self, replica: Arc<Replica>, timeout: Duration) -> bool {
        let Ok(request) = Request::get(replica.endpoint().uri_for("/health")).body(Full::default())
        else {
            return false;
        };
        let check = async {
            let response = self.client.request(request).await.ok()?;
            let ok = response.status().is_success();
            Limited::new(response.into_body(), 4096)
                .collect()
                .await
                .ok()?;
            Some(ok)
        };
        matches!(tokio::time::timeout(timeout, check).await, Ok(Some(true)))
    }
}
