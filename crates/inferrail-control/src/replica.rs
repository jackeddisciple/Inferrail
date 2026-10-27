use std::sync::Arc;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::time::Duration;

use arc_swap::ArcSwapOption;
use inferrail_core::config::{Config, ReplicaConfig};
use inferrail_core::ids::{ModelId, ReplicaId};
use inferrail_core::openai::ModelCard;
use inferrail_load::{LoadCell, ReplicaLoad, Target};
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
    in_flight_tokens: AtomicU64,
    prefill_tokens: AtomicU64,
    prefilling: AtomicU32,
    queue_until: AtomicU64,
    health: Health,
    cards: ArcSwapOption<Cards>,
    engine_load: LoadCell,
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
            in_flight_tokens: AtomicU64::new(0),
            prefill_tokens: AtomicU64::new(0),
            prefilling: AtomicU32::new(0),
            queue_until: AtomicU64::new(0),
            health: Health::default(),
            cards: ArcSwapOption::empty(),
            engine_load: LoadCell::default(),
        }
    }

    /// The engine's own load reports, as published by the collector.
    pub fn engine_load(&self) -> &LoadCell {
        &self.engine_load
    }

    /// The gateway's counters merged with the engine's latest report, which
    /// only counts while it is at most `max_age` old.
    pub fn load(&self, now: Instant, max_age: Duration) -> ReplicaLoad {
        let latest = self.engine_load.latest();
        ReplicaLoad {
            in_flight: self.in_flight(),
            pending: self.pending(),
            in_flight_tokens: self.in_flight_tokens(),
            prefill_tokens: self.prefill_tokens(),
            engine: latest.filter(|s| s.age(now) <= max_age).map(|s| s.load),
            age: latest.map(|s| s.age(now)),
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

    pub fn in_flight_tokens(&self) -> u64 {
        self.in_flight_tokens.load(Ordering::Relaxed)
    }

    pub fn prefill_tokens(&self) -> u64 {
        self.prefill_tokens.load(Ordering::Relaxed)
    }

    /// Requests that have not had a response byte back yet.
    pub fn prefilling(&self) -> u32 {
        self.prefilling.load(Ordering::Relaxed)
    }

    /// Prefill work queued on the replica at `now`, as estimated when each
    /// request was sent and drained in real time. Once every request has
    /// had its first byte back, nothing is queued, whatever was estimated.
    pub fn queued(&self, now: Nanos) -> Duration {
        if self.prefilling() == 0 {
            return Duration::ZERO;
        }
        Duration::from_nanos(self.queue_until.load(Ordering::Relaxed).saturating_sub(now))
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

    /// Reserves an in flight slot and a pending slot for a request of about
    /// `tokens` prompt tokens, or nothing when either cap is reached. The caps
    /// can never be exceeded, even under races.
    pub fn try_acquire(self: &Arc<Self>, tokens: u32) -> Option<Lease> {
        let limits = self.spec.limits;
        let in_flight = increment_below(&self.in_flight, limits.max_concurrency)?;
        if increment_below(&self.pending, limits.max_pending).is_none() {
            self.in_flight.fetch_sub(1, Ordering::Relaxed);
            return None;
        }
        self.peak_in_flight.fetch_max(in_flight, Ordering::Relaxed);
        let tokens = u64::from(tokens);
        self.in_flight_tokens.fetch_add(tokens, Ordering::Relaxed);
        self.prefill_tokens.fetch_add(tokens, Ordering::Relaxed);
        self.prefilling.fetch_add(1, Ordering::Relaxed);
        Some(Lease {
            replica: Arc::clone(self),
            pending: true,
            prefilling: true,
            tokens,
        })
    }
}

impl Target for Replica {
    fn name(&self) -> &str {
        self.spec.id.as_str()
    }

    fn cell(&self) -> &LoadCell {
        &self.engine_load
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
/// the upstream response head arrives, prefilling until its first body byte.
#[derive(Debug)]
pub struct Lease {
    replica: Arc<Replica>,
    pending: bool,
    prefilling: bool,
    tokens: u64,
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

    pub fn first_byte(&mut self) {
        if std::mem::take(&mut self.prefilling) {
            self.replica
                .prefill_tokens
                .fetch_sub(self.tokens, Ordering::Relaxed);
            self.replica.prefilling.fetch_sub(1, Ordering::Relaxed);
        }
    }

    /// Queues `prefill` of work on the replica behind what is already
    /// there, or from `now` when no other request is waiting on it.
    pub fn enqueue(&self, now: Nanos, prefill: Duration) {
        let mine = u32::from(self.prefilling);
        let work = u64::try_from(prefill.as_nanos()).unwrap_or(u64::MAX);
        let _ =
            self.replica
                .queue_until
                .fetch_update(Ordering::AcqRel, Ordering::Acquire, |until| {
                    let alone = self.replica.prefilling() <= mine;
                    let start = if alone { now } else { until.max(now) };
                    Some(start.saturating_add(work))
                });
    }

    pub fn tokens(&self) -> u64 {
        self.tokens
    }
}

impl Drop for Lease {
    fn drop(&mut self) {
        self.responded();
        self.first_byte();
        self.replica
            .in_flight_tokens
            .fetch_sub(self.tokens, Ordering::Relaxed);
        self.replica.in_flight.fetch_sub(1, Ordering::Relaxed);
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use inferrail_load::{Engine, EngineLoad};
    use proptest::prelude::*;

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
        let mut a = replica.try_acquire(0).unwrap();
        let b = replica.try_acquire(0).unwrap();
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
        let first = replica.try_acquire(0).unwrap();
        let _second = replica.try_acquire(0).unwrap();
        assert!(!replica.has_room());
        assert!(replica.try_acquire(0).is_none());
        drop(first);
        assert!(replica.has_room());
        assert!(replica.try_acquire(0).is_some());
    }

    #[test]
    fn pending_cap_rolls_back_the_in_flight_slot() {
        let replica = capped(Some(10), Some(1));
        let mut first = replica.try_acquire(0).unwrap();
        assert!(replica.try_acquire(0).is_none());
        assert_eq!(replica.in_flight(), 1);
        first.responded();
        let _second = replica.try_acquire(0).unwrap();
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
                        if let Some(mut lease) = replica.try_acquire(i % 7) {
                            granted += 1;
                            if (i + t) % 2 == 0 {
                                lease.first_byte();
                            }
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
        assert_eq!(
            (replica.in_flight_tokens(), replica.prefill_tokens()),
            (0, 0)
        );
    }

    #[derive(Debug, Clone)]
    enum Op {
        Acquire(u32),
        Responded(usize),
        FirstByte(usize),
        Drop(usize),
    }

    fn op() -> impl Strategy<Value = Op> {
        prop_oneof![
            (0u32..5000).prop_map(Op::Acquire),
            any::<usize>().prop_map(Op::Responded),
            any::<usize>().prop_map(Op::FirstByte),
            any::<usize>().prop_map(Op::Drop),
        ]
    }

    #[derive(Debug, Default)]
    struct Reference {
        leases: Vec<(u64, bool, bool)>,
    }

    impl Reference {
        fn counters(&self) -> (u32, u32, u64, u64, u32) {
            let in_flight = self.leases.len() as u32;
            let pending = self.leases.iter().filter(|l| l.1).count() as u32;
            let tokens = self.leases.iter().map(|l| l.0).sum();
            let prefill = self.leases.iter().filter(|l| l.2).map(|l| l.0).sum();
            let prefilling = self.leases.iter().filter(|l| l.2).count() as u32;
            (in_flight, pending, tokens, prefill, prefilling)
        }
    }

    fn counters(replica: &Replica) -> (u32, u32, u64, u64, u32) {
        (
            replica.in_flight(),
            replica.pending(),
            replica.in_flight_tokens(),
            replica.prefill_tokens(),
            replica.prefilling(),
        )
    }

    proptest! {
        #[test]
        fn counters_match_a_reference_under_any_lease_sequence(
            ops in prop::collection::vec(op(), 0..80),
            cap in prop::option::of(1u32..6),
        ) {
            let replica = capped(cap, None);
            let mut leases: Vec<Lease> = Vec::new();
            let mut model = Reference::default();
            for op in ops {
                match op {
                    Op::Acquire(tokens) => match replica.try_acquire(tokens) {
                        Some(lease) => {
                            leases.push(lease);
                            model.leases.push((u64::from(tokens), true, true));
                        }
                        None => prop_assert_eq!(Some(model.leases.len() as u32), cap),
                    },
                    Op::Responded(i) if !leases.is_empty() => {
                        let i = i % leases.len();
                        leases[i].responded();
                        model.leases[i].1 = false;
                    }
                    Op::FirstByte(i) if !leases.is_empty() => {
                        let i = i % leases.len();
                        leases[i].first_byte();
                        model.leases[i].2 = false;
                    }
                    Op::Drop(i) if !leases.is_empty() => {
                        let i = i % leases.len();
                        drop(leases.swap_remove(i));
                        model.leases.swap_remove(i);
                    }
                    _ => {}
                }
                prop_assert_eq!(counters(&replica), model.counters());
            }
            drop(leases);
            prop_assert_eq!(counters(&replica), (0, 0, 0, 0, 0));
            prop_assert_eq!(replica.queued(1), Duration::ZERO);
        }
    }

    #[test]
    fn the_prefill_queue_builds_drains_and_empties() {
        let replica = capped(None, None);
        let ms = Duration::from_millis;
        let at = |millis: u64| millis * 1_000_000;
        assert_eq!(replica.queued(at(0)), Duration::ZERO);
        let mut first = replica.try_acquire(10).unwrap();
        first.enqueue(at(100), ms(50));
        assert_eq!(replica.queued(at(100)), ms(50));
        assert_eq!(replica.queued(at(120)), ms(30));
        let second = replica.try_acquire(10).unwrap();
        second.enqueue(at(110), ms(30));
        assert_eq!(replica.queued(at(110)), ms(70));
        assert_eq!(replica.queued(at(180)), Duration::ZERO);
        assert_eq!(replica.queued(at(500)), Duration::ZERO);
        first.first_byte();
        assert_eq!(replica.prefilling(), 1);
        drop(second);
        assert_eq!(replica.prefilling(), 0);

        let mut slow = replica.try_acquire(1).unwrap();
        slow.enqueue(at(1_000), ms(900));
        slow.first_byte();
        assert_eq!(replica.queued(at(1_100)), Duration::ZERO);
        let next = replica.try_acquire(1).unwrap();
        next.enqueue(at(1_200), ms(40));
        assert_eq!(replica.queued(at(1_200)), ms(40));
        drop((first, slow, next));
        assert_eq!(counters(&replica), (0, 0, 0, 0, 0));
    }

    #[test]
    fn requests_queued_at_once_all_count() {
        for _ in 0..2_000 {
            let replica = capped(None, None);
            let start = Arc::new(std::sync::Barrier::new(8));
            let threads: Vec<_> = (0..8)
                .map(|_| {
                    let (replica, start) = (Arc::clone(&replica), Arc::clone(&start));
                    std::thread::spawn(move || {
                        start.wait();
                        let lease = replica.try_acquire(1).unwrap();
                        lease.enqueue(1_000, Duration::from_millis(1));
                        lease
                    })
                })
                .collect();
            let leases: Vec<Lease> = threads.into_iter().map(|t| t.join().unwrap()).collect();
            assert_eq!(replica.queued(1_000), Duration::from_millis(8));
            drop(leases);
        }
    }

    #[tokio::test(start_paused = true)]
    async fn load_merges_counters_with_a_fresh_engine_report() {
        let replica = capped(None, None);
        let max_age = Duration::from_millis(250);
        let mut lease = replica.try_acquire(120).unwrap();
        lease.responded();
        let _other = replica.try_acquire(30).unwrap();
        let load = replica.load(Instant::now(), max_age);
        assert_eq!(
            (
                load.in_flight,
                load.pending,
                load.in_flight_tokens,
                load.prefill_tokens
            ),
            (2, 1, 150, 150)
        );
        assert!(load.is_stale());
        assert_eq!(load.age, None);

        lease.first_byte();
        replica.engine_load().publish(
            EngineLoad {
                engine: Engine::Vllm,
                running: 7,
                waiting: 2,
                kv_usage: 0.4,
                prefix_queries: 10,
                prefix_hits: 5,
                preemptions: 0,
                kv_capacity_tokens: Some(1000),
            },
            Instant::now(),
        );
        tokio::time::advance(Duration::from_millis(100)).await;
        let load = replica.load(Instant::now(), max_age);
        assert_eq!(load.prefill_tokens, 30);
        assert_eq!(load.engine.map(|e| (e.running, e.waiting)), Some((7, 2)));
        assert_eq!(load.age, Some(Duration::from_millis(100)));

        tokio::time::advance(Duration::from_millis(200)).await;
        let load = replica.load(Instant::now(), max_age);
        assert!(load.is_stale());
        assert_eq!(load.age, Some(Duration::from_millis(300)));
        assert_eq!(load.in_flight, 2);
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
