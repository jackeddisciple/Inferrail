use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use arc_swap::ArcSwapOption;
use inferrail_core::config::{Config, ReplicaConfig};
use inferrail_core::ids::{ModelId, ReplicaId};
use inferrail_core::openai::ModelCard;
use tokio::time::Instant;
use url::Url;

use crate::health::{Health, Nanos};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Origin {
    Static,
    Dns(ReplicaId),
}

/// Where requests for a replica go: scheme, host and port, plus any path
/// prefix the replica URL carries.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Endpoint {
    url: Url,
    origin: String,
    prefix: String,
}

impl Endpoint {
    pub fn new(url: &Url) -> Self {
        Self {
            origin: url.origin().ascii_serialization(),
            prefix: url.path().trim_end_matches('/').to_owned(),
            url: url.clone(),
        }
    }

    pub fn url(&self) -> &Url {
        &self.url
    }

    pub fn origin(&self) -> &str {
        &self.origin
    }

    pub fn is_https(&self) -> bool {
        self.url.scheme() == "https"
    }

    pub fn uri_for(&self, path_and_query: &str) -> String {
        let mut uri =
            String::with_capacity(self.origin.len() + self.prefix.len() + path_and_query.len());
        uri.push_str(&self.origin);
        uri.push_str(&self.prefix);
        uri.push_str(path_and_query);
        uri
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Limits {
    pub max_concurrency: Option<u32>,
    pub max_pending: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReplicaSpec {
    pub id: ReplicaId,
    pub url: Url,
    pub models: Vec<ModelId>,
    pub origin: Origin,
    pub limits: Limits,
    pub kv_capacity_blocks: Option<u64>,
}

impl From<&ReplicaConfig> for ReplicaSpec {
    fn from(config: &ReplicaConfig) -> Self {
        Self {
            id: config.id.clone(),
            url: config.url.clone(),
            models: config.models.clone(),
            origin: Origin::Static,
            limits: Limits {
                max_concurrency: config.max_concurrency,
                max_pending: config.max_pending,
            },
            kv_capacity_blocks: config.kv_capacity_blocks,
        }
    }
}

impl ReplicaSpec {
    pub fn statics(config: &Config) -> Vec<Self> {
        config.replicas.iter().map(Self::from).collect()
    }
}

/// One engine replica. Static facts plus lock free counters that the hot
/// path reads and updates without blocking.
#[derive(Debug)]
pub struct Replica {
    spec: ReplicaSpec,
    endpoint: Endpoint,
    in_flight: AtomicU32,
    pending: AtomicU32,
    peak_in_flight: AtomicU32,
    health: Health,
    cards: ArcSwapOption<Cards>,
}

/// The engine's own `/v1/models` answer, as last fetched.
#[derive(Debug)]
pub struct Cards {
    pub fetched: Instant,
    pub cards: Vec<ModelCard>,
}

impl Replica {
    pub fn new(spec: ReplicaSpec) -> Self {
        Self {
            endpoint: Endpoint::new(&spec.url),
            spec,
            in_flight: AtomicU32::new(0),
            pending: AtomicU32::new(0),
            peak_in_flight: AtomicU32::new(0),
            health: Health::default(),
            cards: ArcSwapOption::empty(),
        }
    }

    pub fn cards(&self) -> Option<Arc<Cards>> {
        self.cards.load_full()
    }

    pub fn set_cards(&self, cards: Vec<ModelCard>) {
        self.cards.store(Some(Arc::new(Cards {
            fetched: Instant::now(),
            cards,
        })));
    }

    pub fn health(&self) -> &Health {
        &self.health
    }

    pub fn is_routable(&self, now: Nanos) -> bool {
        self.health.is_routable(now)
    }

    pub fn id(&self) -> &ReplicaId {
        &self.spec.id
    }

    pub fn spec(&self) -> &ReplicaSpec {
        &self.spec
    }

    pub fn endpoint(&self) -> &Endpoint {
        &self.endpoint
    }

    pub fn models(&self) -> &[ModelId] {
        &self.spec.models
    }

    pub fn serves(&self, model: &ModelId) -> bool {
        self.spec.models.contains(model)
    }

    pub fn limits(&self) -> Limits {
        self.spec.limits
    }

    pub fn in_flight(&self) -> u32 {
        self.in_flight.load(Ordering::Relaxed)
    }

    pub fn pending(&self) -> u32 {
        self.pending.load(Ordering::Relaxed)
    }

    /// The most requests ever in flight at once, as seen by the lease that
    /// reached it.
    pub fn peak_in_flight(&self) -> u32 {
        self.peak_in_flight.load(Ordering::Relaxed)
    }

    pub fn has_room(&self) -> bool {
        let Limits {
            max_concurrency,
            max_pending,
        } = self.spec.limits;
        max_concurrency.is_none_or(|cap| self.in_flight() < cap)
            && max_pending.is_none_or(|cap| self.pending() < cap)
    }

    /// Reserves an in flight slot and a pending slot, or nothing when either
    /// cap is reached. The caps can never be exceeded, even under races.
    pub fn try_acquire(self: &Arc<Self>) -> Option<Lease> {
        let limits = self.spec.limits;
        let in_flight = increment_below(&self.in_flight, limits.max_concurrency)?;
        if increment_below(&self.pending, limits.max_pending).is_none() {
            self.in_flight.fetch_sub(1, Ordering::Relaxed);
            return None;
        }
        self.peak_in_flight.fetch_max(in_flight, Ordering::Relaxed);
        Some(Lease {
            replica: Arc::clone(self),
            pending: true,
        })
    }
}

/// Adds one unless that would pass `cap`; returns the new value.
fn increment_below(counter: &AtomicU32, cap: Option<u32>) -> Option<u32> {
    match cap {
        None => Some(counter.fetch_add(1, Ordering::Relaxed) + 1),
        Some(cap) => counter
            .fetch_update(Ordering::AcqRel, Ordering::Relaxed, |n| {
                (n < cap).then_some(n + 1)
            })
            .ok()
            .map(|n| n + 1),
    }
}

/// A request's claim on a replica: in flight until dropped, pending until
/// the upstream response head arrives.
#[derive(Debug)]
pub struct Lease {
    replica: Arc<Replica>,
    pending: bool,
}

impl Lease {
    pub fn replica(&self) -> &Arc<Replica> {
        &self.replica
    }

    pub fn responded(&mut self) {
        if std::mem::take(&mut self.pending) {
            self.replica.pending.fetch_sub(1, Ordering::Relaxed);
        }
    }
}

impl Drop for Lease {
    fn drop(&mut self) {
        self.responded();
        self.replica.in_flight.fetch_sub(1, Ordering::Relaxed);
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    pub(crate) fn spec(id: &str, url: &str, models: &[&str]) -> ReplicaSpec {
        ReplicaSpec {
            id: ReplicaId::new(id).unwrap(),
            url: url.parse().unwrap(),
            models: models.iter().map(|m| ModelId::new(m).unwrap()).collect(),
            origin: Origin::Static,
            limits: Limits::default(),
            kv_capacity_blocks: None,
        }
    }

    fn capped(concurrency: Option<u32>, pending: Option<u32>) -> Arc<Replica> {
        let mut spec = spec("r0", "http://10.0.0.1:8000", &["m"]);
        spec.limits = Limits {
            max_concurrency: concurrency,
            max_pending: pending,
        };
        Arc::new(Replica::new(spec))
    }

    #[test]
    fn endpoints_join_origin_prefix_and_path() {
        for (url, origin, uri) in [
            (
                "http://10.0.0.1:8000",
                "http://10.0.0.1:8000",
                "http://10.0.0.1:8000/v1/models",
            ),
            (
                "https://gpu-0.internal:8443/vllm/",
                "https://gpu-0.internal:8443",
                "https://gpu-0.internal:8443/vllm/v1/models",
            ),
            (
                "http://[::1]:9000",
                "http://[::1]:9000",
                "http://[::1]:9000/v1/models",
            ),
            ("http://engine", "http://engine", "http://engine/v1/models"),
        ] {
            let endpoint = Endpoint::new(&url.parse().unwrap());
            assert_eq!(endpoint.origin(), origin, "{url}");
            assert_eq!(endpoint.uri_for("/v1/models"), uri, "{url}");
        }
        assert!(Endpoint::new(&"https://h:1".parse().unwrap()).is_https());
    }

    #[test]
    fn leases_balance_both_counters() {
        let replica = capped(None, None);
        let mut a = replica.try_acquire().unwrap();
        let b = replica.try_acquire().unwrap();
        assert_eq!((replica.in_flight(), replica.pending()), (2, 2));
        a.responded();
        a.responded();
        assert_eq!((replica.in_flight(), replica.pending()), (2, 1));
        drop(b);
        assert_eq!((replica.in_flight(), replica.pending()), (1, 0));
        drop(a);
        assert_eq!((replica.in_flight(), replica.pending()), (0, 0));
        assert_eq!(replica.id().as_str(), "r0");
    }

    #[test]
    fn concurrency_cap_refuses_and_reports_no_room() {
        let replica = capped(Some(2), None);
        let first = replica.try_acquire().unwrap();
        let _second = replica.try_acquire().unwrap();
        assert!(!replica.has_room());
        assert!(replica.try_acquire().is_none());
        drop(first);
        assert!(replica.has_room());
        assert!(replica.try_acquire().is_some());
    }

    #[test]
    fn pending_cap_rolls_back_the_in_flight_slot() {
        let replica = capped(Some(10), Some(1));
        let mut first = replica.try_acquire().unwrap();
        assert!(replica.try_acquire().is_none());
        assert_eq!(replica.in_flight(), 1);
        first.responded();
        let _second = replica.try_acquire().unwrap();
        assert_eq!((replica.in_flight(), replica.pending()), (2, 1));
    }

    #[test]
    fn caps_hold_under_sixteen_racing_threads() {
        let replica = capped(Some(4), Some(3));
        let high_in_flight = Arc::new(AtomicU32::new(0));
        let high_pending = Arc::new(AtomicU32::new(0));
        let threads: Vec<_> = (0..16)
            .map(|t| {
                let replica = Arc::clone(&replica);
                let (hi, hp) = (Arc::clone(&high_in_flight), Arc::clone(&high_pending));
                std::thread::spawn(move || {
                    let mut granted = 0u32;
                    for i in 0..20_000u32 {
                        if let Some(mut lease) = replica.try_acquire() {
                            granted += 1;
                            hi.fetch_max(replica.in_flight(), Ordering::Relaxed);
                            hp.fetch_max(replica.pending(), Ordering::Relaxed);
                            if (i + t) % 3 == 0 {
                                lease.responded();
                            }
                            std::hint::spin_loop();
                        }
                    }
                    granted
                })
            })
            .collect();
        let granted: u32 = threads.into_iter().map(|t| t.join().unwrap()).sum();
        assert!(granted > 0);
        assert!(high_in_flight.load(Ordering::Relaxed) <= 4);
        assert!((1..=4).contains(&replica.peak_in_flight()));
        assert!(high_pending.load(Ordering::Relaxed) <= 3);
        assert_eq!((replica.in_flight(), replica.pending()), (0, 0));
    }

    #[test]
    fn cards_are_stored_and_replaced() {
        let replica = capped(None, None);
        assert!(replica.cards().is_none());
        replica.set_cards(Vec::new());
        let first = replica.cards().unwrap();
        assert!(first.cards.is_empty());
        replica.set_cards(Vec::new());
        assert!(!Arc::ptr_eq(&first, &replica.cards().unwrap()));
    }

    #[test]
    fn specs_come_from_static_config() {
        let config = Config::from_yaml_str(
            "models: [{name: m}]\nreplicas: [{id: a, url: 'http://h:1', models: [m], max_concurrency: 3, max_pending: 2}]\n",
        )
        .unwrap();
        let specs = ReplicaSpec::statics(&config);
        assert_eq!(specs.len(), 1);
        assert_eq!(specs[0].origin, Origin::Static);
        assert_eq!(
            specs[0].limits,
            Limits {
                max_concurrency: Some(3),
                max_pending: Some(2)
            }
        );
        let replica = Replica::new(specs[0].clone());
        assert!(replica.serves(&ModelId::new("m").unwrap()));
        assert!(!replica.serves(&ModelId::new("n").unwrap()));
    }
}
