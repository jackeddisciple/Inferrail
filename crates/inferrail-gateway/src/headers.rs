//! Header hygiene for a transparent proxy: hop by hop headers never cross.

use hyper::header::{self, HeaderMap, HeaderName};

const HOP_BY_HOP: [HeaderName; 7] = [
    header::CONNECTION,
    HeaderName::from_static("keep-alive"),
    header::PROXY_AUTHENTICATE,
    header::PROXY_AUTHORIZATION,
    header::TE,
    header::TRAILER,
    header::TRANSFER_ENCODING,
];

pub const REPLICA: HeaderName = HeaderName::from_static("x-inferrail-replica");
/// The policy that routed the request, which an A/B split makes worth knowing.
pub const POLICY: HeaderName = HeaderName::from_static("x-inferrail-policy");
/// The prompt's length in tokens, when the gateway tokenized it.
pub const PROMPT_TOKENS: HeaderName = HeaderName::from_static("x-inferrail-prompt-tokens");
/// Prompt tokens the KV index predicted the chosen replica had cached; absent
/// when the index made no prediction.
pub const CACHED_TOKENS: HeaderName = HeaderName::from_static("x-inferrail-cached-tokens");
/// Names the conversation a request belongs to, for session affinity.
pub const SESSION_ID: HeaderName = HeaderName::from_static("x-session-id");

pub fn strip_hop_by_hop(headers: &mut HeaderMap) {
    let listed: Vec<HeaderName> = headers
        .get_all(header::CONNECTION)
        .iter()
        .filter_map(|v| v.to_str().ok())
        .flat_map(|v| v.split(','))
        .filter_map(|name| HeaderName::from_bytes(name.trim().as_bytes()).ok())
        .collect();
    for name in listed.iter().chain(HOP_BY_HOP.iter()) {
        headers.remove(name);
    }
    headers.remove(header::UPGRADE);
    headers.remove("proxy-connection");
}

/// Client headers as the engine should see them. Host and Content-Length
/// are left to the client connection, since the body may have been
/// rewritten, and Expect is moot because the body is already in hand.
pub fn for_upstream(headers: &mut HeaderMap) {
    strip_hop_by_hop(headers);
    headers.remove(header::HOST);
    headers.remove(header::CONTENT_LENGTH);
    headers.remove(header::EXPECT);
    headers.remove(&REPLICA);
}

pub fn for_downstream(headers: &mut HeaderMap) {
    strip_hop_by_hop(headers);
}

#[cfg(test)]
mod tests {
    use super::*;
    use hyper::header::HeaderValue;

    fn map(pairs: &[(&'static str, &'static str)]) -> HeaderMap {
        let mut headers = HeaderMap::new();
        for (name, value) in pairs {
            headers.append(*name, HeaderValue::from_static(value));
        }
        headers
    }

    #[test]
    fn hop_by_hop_and_connection_listed_headers_are_removed() {
        let mut headers = map(&[
            ("connection", "keep-alive, X-Secret-Hop"),
            ("connection", "upgrade"),
            ("keep-alive", "timeout=5"),
            ("x-secret-hop", "1"),
            ("te", "trailers"),
            ("trailer", "x-checksum"),
            ("transfer-encoding", "chunked"),
            ("upgrade", "h2c"),
            ("proxy-authorization", "Basic abc"),
            ("proxy-connection", "keep-alive"),
            ("authorization", "Bearer key"),
            ("content-type", "application/json"),
            ("x-request-id", "abc"),
        ]);
        strip_hop_by_hop(&mut headers);
        let mut left: Vec<&str> = headers.keys().map(HeaderName::as_str).collect();
        left.sort_unstable();
        assert_eq!(left, ["authorization", "content-type", "x-request-id"]);
    }

    #[test]
    fn upstream_requests_drop_host_length_and_spoofed_replica() {
        let mut headers = map(&[
            ("host", "gateway:8080"),
            ("content-length", "12"),
            ("expect", "100-continue"),
            ("x-inferrail-replica", "spoofed"),
            ("accept", "text/event-stream"),
        ]);
        for_upstream(&mut headers);
        assert_eq!(headers.len(), 1);
        assert!(headers.contains_key("accept"));
    }

    #[test]
    fn downstream_keeps_content_length() {
        let mut headers = map(&[
            ("content-length", "42"),
            ("content-type", "application/json"),
            ("connection", "close"),
        ]);
        for_downstream(&mut headers);
        assert_eq!(headers.len(), 2);
    }
}
