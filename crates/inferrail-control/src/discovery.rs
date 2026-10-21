use std::future::Future;
use std::io;
use std::net::SocketAddr;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use inferrail_core::config::DnsGroupConfig;
use inferrail_core::ids::ReplicaId;
use tokio::time::{Instant, MissedTickBehavior};
use url::{Host, Url};

use crate::fleet::{MembershipChange, Registry};
use crate::replica::{Limits, Origin, ReplicaSpec};

pub type Resolution<'a> = Pin<Box<dyn Future<Output = io::Result<Vec<SocketAddr>>> + Send + 'a>>;

pub trait Resolve: Send + Sync + std::fmt::Debug + 'static {
    fn resolve<'a>(&'a self, host: &'a str, port: u16) -> Resolution<'a>;
}

/// The operating system resolver, off the async threads.
#[derive(Debug, Clone, Copy, Default)]
pub struct SystemResolver;

impl Resolve for SystemResolver {
    fn resolve<'a>(&'a self, host: &'a str, port: u16) -> Resolution<'a> {
        Box::pin(async move { Ok(tokio::net::lookup_host((host, port)).await?.collect()) })
    }
}

fn host_of(url: &Url) -> Option<String> {
    Some(match url.host()? {
        Host::Domain(domain) => domain.to_owned(),
        Host::Ipv4(ip) => ip.to_string(),
        Host::Ipv6(ip) => ip.to_string(),
    })
}

/// `{group}-{ip}-{port}`, with characters a replica id cannot hold replaced
/// by `_` and the group part shortened if the whole would be too long.
pub fn member_id(group: &ReplicaId, addr: SocketAddr) -> Option<ReplicaId> {
    let suffix: String = format!("-{}-{}", addr.ip(), addr.port())
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-') {
                c
            } else {
                '_'
            }
        })
        .collect();
    let room = ReplicaId::MAX_LEN.saturating_sub(suffix.len());
    let name = group.as_str();
    ReplicaId::new(format!("{}{suffix}", &name[..name.len().min(room)])).ok()
}

/// One replica per distinct address, in address order.
pub fn members(group: &DnsGroupConfig, addrs: &[SocketAddr]) -> Vec<ReplicaSpec> {
    let mut addrs = addrs.to_vec();
    addrs.sort_unstable();
    addrs.dedup();
    addrs
        .into_iter()
        .filter_map(|addr| {
            let mut url = group.url.clone();
            url.set_ip_host(addr.ip()).ok()?;
            url.set_port(Some(addr.port())).ok()?;
            Some(ReplicaSpec {
                id: member_id(&group.name, addr)?,
                url,
                models: group.models.clone(),
                origin: Origin::Dns(group.name.clone()),
                limits: Limits {
                    max_concurrency: group.max_concurrency,
                    max_pending: group.max_pending,
                },
                kv_capacity_blocks: group.kv_capacity_blocks,
            })
        })
        .collect()
}

/// Resolves the group once and applies the answer. A failed lookup keeps the
/// last known members; a successful empty answer removes them all.
pub async fn refresh(
    registry: &Registry,
    group: &DnsGroupConfig,
    resolver: &dyn Resolve,
    timeout: Duration,
) -> Option<MembershipChange> {
    let host = host_of(&group.url)?;
    let port = group.url.port_or_known_default()?;
    let answer = tokio::time::timeout(timeout, resolver.resolve(&host, port)).await;
    match answer {
        Ok(Ok(addrs)) => {
            let change = registry.set_group(&group.name, members(group, &addrs));
            if !change.is_empty() {
                tracing::info!(
                    group = %group.name,
                    added = ?change.added,
                    removed = ?change.removed,
                    rejected = ?change.rejected,
                    "DNS membership changed"
                );
            }
            Some(change)
        }
        Ok(Err(err)) => {
            tracing::warn!(group = %group.name, %host, error = %err, "DNS lookup failed; keeping the last known members");
            None
        }
        Err(_) => {
            tracing::warn!(group = %group.name, %host, "DNS lookup timed out; keeping the last known members");
            None
        }
    }
}

/// Re-resolves the group every `refresh`, starting one period from now.
pub async fn run_dns(registry: Arc<Registry>, group: DnsGroupConfig, resolver: Arc<dyn Resolve>) {
    let period = group.refresh;
    let mut ticker = tokio::time::interval_at(Instant::now() + period, period);
    ticker.set_missed_tick_behavior(MissedTickBehavior::Delay);
    loop {
        ticker.tick().await;
        refresh(
            &registry,
            &group,
            resolver.as_ref(),
            period.max(Duration::from_secs(1)),
        )
        .await;
    }
}

#[cfg(test)]
mod tests {
    use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};
    use std::sync::Mutex;

    use super::*;
    use inferrail_core::ids::ModelId;

    fn group(url: &str) -> DnsGroupConfig {
        DnsGroupConfig {
            name: ReplicaId::new("pool").unwrap(),
            url: url.parse().unwrap(),
            models: vec![ModelId::new("m").unwrap()],
            refresh: Duration::from_secs(1),
            kv_capacity_blocks: Some(100),
            max_concurrency: Some(8),
            max_pending: Some(2),
        }
    }

    fn v4(last: u8, port: u16) -> SocketAddr {
        SocketAddr::new(IpAddr::V4(Ipv4Addr::new(10, 0, 0, last)), port)
    }

    #[test]
    fn member_ids_are_valid_and_unique_per_address() {
        let name = ReplicaId::new("pool").unwrap();
        assert_eq!(
            member_id(&name, v4(7, 8000)).unwrap().as_str(),
            "pool-10.0.0.7-8000"
        );
        let v6 = SocketAddr::new(IpAddr::V6(Ipv6Addr::LOCALHOST), 9000);
        assert_eq!(member_id(&name, v6).unwrap().as_str(), "pool-__1-9000");
        let long = ReplicaId::new("p".repeat(128)).unwrap();
        let id = member_id(&long, v6).unwrap();
        assert_eq!(id.as_str().len(), ReplicaId::MAX_LEN);
        assert!(id.as_str().ends_with("-__1-9000"));
    }

    #[test]
    fn members_rewrite_the_host_and_keep_everything_else() {
        let specs = members(
            &group("http://vllm.svc:8000/engine/"),
            &[v4(2, 8000), v4(1, 8000), v4(2, 8000), v4(1, 9000)],
        );
        let urls: Vec<&str> = specs.iter().map(|s| s.url.as_str()).collect();
        assert_eq!(
            urls,
            [
                "http://10.0.0.1:8000/engine/",
                "http://10.0.0.1:9000/engine/",
                "http://10.0.0.2:8000/engine/"
            ]
        );
        let spec = &specs[0];
        assert_eq!(spec.origin, Origin::Dns(ReplicaId::new("pool").unwrap()));
        assert_eq!(spec.limits.max_pending, Some(2));
        assert_eq!(spec.kv_capacity_blocks, Some(100));
    }

    #[derive(Debug)]
    struct Fake(Mutex<Option<Vec<SocketAddr>>>);

    impl Resolve for Fake {
        fn resolve<'a>(&'a self, host: &'a str, port: u16) -> Resolution<'a> {
            assert_eq!((host, port), ("vllm.svc", 8000));
            let answer = self.0.lock().unwrap().clone();
            Box::pin(async move { answer.ok_or_else(|| io::Error::other("no answer")) })
        }
    }

    #[tokio::test]
    async fn failures_keep_members_and_empty_answers_clear_them() {
        let registry = Registry::new(vec![]);
        let fake = Fake(Mutex::new(Some(vec![v4(1, 8000), v4(2, 8000)])));
        let g = group("http://vllm.svc:8000");
        let timeout = Duration::from_secs(1);
        let change = refresh(&registry, &g, &fake, timeout).await.unwrap();
        assert_eq!(change.added.len(), 2);
        *fake.0.lock().unwrap() = None;
        assert!(refresh(&registry, &g, &fake, timeout).await.is_none());
        assert_eq!(registry.load().replicas().len(), 2);
        *fake.0.lock().unwrap() = Some(vec![]);
        let change = refresh(&registry, &g, &fake, timeout).await.unwrap();
        assert_eq!(change.removed.len(), 2);
        assert!(registry.load().replicas().is_empty());
    }

    #[tokio::test]
    async fn the_system_resolver_handles_literals() {
        let addrs = SystemResolver.resolve("127.0.0.1", 80).await.unwrap();
        assert_eq!(addrs, [SocketAddr::from(([127, 0, 0, 1], 80))]);
        assert_eq!(host_of(&"http://[::1]:1".parse().unwrap()).unwrap(), "::1");
    }
}
