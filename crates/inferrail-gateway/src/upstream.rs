use http_body_util::Full;
use hyper::body::Bytes;
use hyper_util::client::legacy::connect::HttpConnector;
use hyper_util::rt::{TokioExecutor, TokioTimer};
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
