//! The gateway's own Prometheus metrics. Request counters and histograms
//! are recorded as requests flow; replica gauges are read from the fleet
//! when Prometheus scrapes, so they cost the hot path nothing.

use std::collections::{HashMap, HashSet};
use std::fmt::{self, Write as _};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Duration;

use hyper::StatusCode;
use inferrail_control::{Clock, Registry};
use inferrail_core::config::PolicyKind;
use inferrail_core::ids::{ModelId, ReplicaId};
use inferrail_index::KvIndex;
use inferrail_tokenize::ModelTokenizer;
use prometheus_client::collector::Collector;
use prometheus_client::encoding::{
    DescriptorEncoder, EncodeLabelSet, EncodeLabelValue, EncodeMetric, LabelValueEncoder,
};
use prometheus_client::metrics::MetricType;
use prometheus_client::metrics::counter::{ConstCounter, Counter};
use prometheus_client::metrics::family::Family;
use prometheus_client::metrics::gauge::ConstGauge;
use prometheus_client::metrics::histogram::Histogram;
use prometheus_client::registry::Registry as MetricRegistry;

use crate::errors::GatewayError;
use crate::proxy::Prediction;

const TTFT_BUCKETS: [f64; 19] = [
    0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0,
    20.0, 40.0,
];
const DURATION_BUCKETS: [f64; 16] = [
    0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 15.0, 30.0, 60.0, 120.0, 300.0, 600.0,
];
const DECISION_BUCKETS: [f64; 11] = [
    1e-6, 2.5e-6, 5e-6, 1e-5, 2.5e-5, 5e-5, 1e-4, 2.5e-4, 5e-4, 1e-3, 5e-3,
];
const TOKENIZE_BUCKETS: [f64; 12] = [
    1e-5, 2.5e-5, 5e-5, 1e-4, 2.5e-4, 5e-4, 1e-3, 2.5e-3, 5e-3, 1e-2, 2.5e-2, 0.1,
];

/// A label value that is cheap to clone: ids share their string.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Text {
    Model(ModelId),
    Replica(ReplicaId),
    Fixed(&'static str),
}

/// `prometheus-client` writes label values verbatim, and model names may
/// hold quotes and backslashes, so values are escaped here.
impl EncodeLabelValue for Text {
    fn encode(&self, encoder: &mut LabelValueEncoder<'_>) -> fmt::Result {
        let value = match self {
            Self::Model(id) => id.as_str(),
            Self::Replica(id) => id.as_str(),
            Self::Fixed(text) => text,
        };
        if !value.contains(['\\', '"', '\n']) {
            return encoder.write_str(value);
        }
        for c in value.chars() {
            match c {
                '\\' => encoder.write_str("\\\\")?,
                '"' => encoder.write_str("\\\"")?,
                '\n' => encoder.write_str("\\n")?,
                c => encoder.write_char(c)?,
            }
        }
        Ok(())
    }
}

fn model(model: Option<&ModelId>) -> Text {
    model.map_or(Text::Fixed(""), |m| Text::Model(m.clone()))
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, EncodeLabelSet)]
struct RequestLabels {
    model: Text,
    replica: Text,
    status: Text,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, EncodeLabelSet)]
struct ErrorLabels {
    model: Text,
    reason: Text,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, EncodeLabelSet)]
struct RouteLabels {
    model: Text,
    policy: Text,
    replica: Text,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, EncodeLabelSet)]
struct ModelLabels {
    model: Text,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, EncodeLabelSet)]
struct PolicyLabels {
    policy: Text,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, EncodeLabelSet)]
struct ReplicaLabels {
    replica: Text,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, EncodeLabelSet)]
struct TokenizeLabels {
    model: Text,
    result: Text,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, EncodeLabelSet)]
struct ScrapeLabels {
    replica: Text,
    result: Text,
}

type Histograms<S> = Family<S, Histogram, fn() -> Histogram>;

/// A label set that names a replica, kept so it can be removed when the
/// replica leaves the fleet.
#[derive(Debug, Clone)]
enum Series {
    Requests(RequestLabels),
    Route(RouteLabels),
    Ejections(ReplicaLabels),
    Scrapes(ScrapeLabels),
    Resets(ReplicaLabels),
}

/// Which request this is, for the metrics recorded when its body ends.
#[derive(Debug, Clone)]
pub struct Routed {
    pub model: ModelId,
    pub policy: PolicyKind,
    pub replica: ReplicaId,
}

impl Routed {
    fn labels(&self) -> RouteLabels {
        RouteLabels {
            model: Text::Model(self.model.clone()),
            policy: Text::Fixed(self.policy.as_str()),
            replica: Text::Replica(self.replica.clone()),
        }
    }
}

#[derive(Debug)]
pub struct Metrics {
    registry: MetricRegistry,
    fleet: Arc<Registry>,
    requests: Family<RequestLabels, Counter>,
    errors: Family<ErrorLabels, Counter>,
    decisions: Family<RouteLabels, Counter>,
    retries: Family<ModelLabels, Counter>,
    disconnects: Family<ModelLabels, Counter>,
    ejections: Family<ReplicaLabels, Counter>,
    scrapes: Family<ScrapeLabels, Counter>,
    ttft: Histograms<RouteLabels>,
    duration: Histograms<RouteLabels>,
    decision_time: Histograms<PolicyLabels>,
    decision_log: (Counter, Counter),
    tokenize: Family<TokenizeLabels, Counter>,
    tokenize_time: Histograms<ModelLabels>,
    prompt_tokens: Family<ModelLabels, Counter>,
    predicted_tokens: Family<ModelLabels, Counter>,
    index_resets: Family<ReplicaLabels, Counter>,
    series: Mutex<HashMap<ReplicaId, Vec<Series>>>,
}

fn histograms<S: Clone + std::hash::Hash + Eq>(buckets: fn() -> Histogram) -> Histograms<S> {
    Family::new_with_constructor(buckets)
}

impl Metrics {
    pub fn new(fleet: FleetGauges) -> Self {
        let this = Self {
            registry: MetricRegistry::default(),
            fleet: Arc::clone(&fleet.registry),
            requests: Family::default(),
            errors: Family::default(),
            decisions: Family::default(),
            retries: Family::default(),
            disconnects: Family::default(),
            ejections: Family::default(),
            scrapes: Family::default(),
            ttft: histograms(|| Histogram::new(TTFT_BUCKETS)),
            duration: histograms(|| Histogram::new(DURATION_BUCKETS)),
            decision_time: histograms(|| Histogram::new(DECISION_BUCKETS)),
            decision_log: (Counter::default(), Counter::default()),
            tokenize: Family::default(),
            tokenize_time: histograms(|| Histogram::new(TOKENIZE_BUCKETS)),
            prompt_tokens: Family::default(),
            predicted_tokens: Family::default(),
            index_resets: Family::default(),
            series: Mutex::default(),
        };
        let mut registry = MetricRegistry::default();
        registry.register(
            "inferrail_requests",
            "Inference requests by model, replica and outcome: a status class, aborted when the engine broke a started response, or cancelled when the client left.",
            this.requests.clone(),
        );
        registry.register(
            "inferrail_request_errors",
            "Requests the gateway answered itself with an error, by reason.",
            this.errors.clone(),
        );
        registry.register(
            "inferrail_routing_decisions",
            "Replicas chosen by each model's policy, one per attempt.",
            this.decisions.clone(),
        );
        registry.register(
            "inferrail_upstream_retries",
            "Attempts on another replica before the first response byte.",
            this.retries.clone(),
        );
        registry.register(
            "inferrail_client_disconnects",
            "Responses abandoned by the client before they ended.",
            this.disconnects.clone(),
        );
        registry.register(
            "inferrail_ejections",
            "Replicas ejected after consecutive failed requests.",
            this.ejections.clone(),
        );
        registry.register(
            "inferrail_load_scrapes",
            "Engine metrics scrapes by result.",
            this.scrapes.clone(),
        );
        registry.register(
            "inferrail_ttft_seconds",
            "Time from a streamed request's arrival to its first response byte.",
            this.ttft.clone(),
        );
        registry.register(
            "inferrail_request_duration_seconds",
            "Time from a request's arrival to the end of its response.",
            this.duration.clone(),
        );
        registry.register(
            "inferrail_routing_decision_seconds",
            "Time to pick a replica and take a lease on it.",
            this.decision_time.clone(),
        );
        registry.register(
            "inferrail_decision_log_records",
            "Routing decisions queued for the decision log.",
            this.decision_log.0.clone(),
        );
        registry.register(
            "inferrail_decision_log_dropped",
            "Sampled routing decisions dropped because the log writer fell behind.",
            this.decision_log.1.clone(),
        );
        registry.register(
            "inferrail_tokenized_requests",
            "Requests whose prompt the gateway tried to tokenize, by result: ok, or why the request was routed without a cache prediction.",
            this.tokenize.clone(),
        );
        registry.register(
            "inferrail_tokenize_seconds",
            "Time to tokenize and hash a prompt, waiting for a worker included.",
            this.tokenize_time.clone(),
        );
        registry.register(
            "inferrail_prompt_tokens",
            "Prompt tokens of tokenized requests, counted at every routing decision.",
            this.prompt_tokens.clone(),
        );
        registry.register(
            "inferrail_predicted_cached_tokens",
            "Prompt tokens the KV index predicted the chosen replica had cached, at every routing decision.",
            this.predicted_tokens.clone(),
        );
        registry.register(
            "inferrail_index_resets",
            "Times a replica's predicted cache was dropped because its engine restarted.",
            this.index_resets.clone(),
        );
        registry.register_collector(Box::new(fleet));
        Self { registry, ..this }
    }

    /// The counters the decision log updates: records queued, and dropped.
    pub fn decision_counters(&self) -> (Counter, Counter) {
        self.decision_log.clone()
    }

    /// Creates a series that names `replica` and tracks it under the lock
    /// `prune` holds, so no series can appear after its replica was pruned
    /// and then be forgotten.
    fn create(&self, replica: &ReplicaId, series: Series, make: impl FnOnce()) {
        let mut tracked = self.series.lock().unwrap_or_else(PoisonError::into_inner);
        make();
        tracked.entry(replica.clone()).or_default().push(series);
    }

    fn count(&self, family: &Family<RouteLabels, Counter>, labels: &RouteLabels, id: &ReplicaId) {
        if let Some(counter) = family.get(labels) {
            counter.inc();
            return;
        }
        self.create(id, Series::Route(labels.clone()), || {
            family.get_or_create(labels).inc();
        });
    }

    fn observe(&self, family: &Histograms<RouteLabels>, routed: &Routed, value: Duration) {
        let labels = routed.labels();
        if let Some(histogram) = family.get(&labels) {
            histogram.observe(value.as_secs_f64());
            return;
        }
        self.create(&routed.replica, Series::Route(labels.clone()), || {
            family.get_or_create(&labels).observe(value.as_secs_f64());
        });
    }

    pub fn decided(&self, routed: &Routed, took: Duration) {
        self.count(&self.decisions, &routed.labels(), &routed.replica);
        self.decision_time
            .get_or_create(&PolicyLabels {
                policy: Text::Fixed(routed.policy.as_str()),
            })
            .observe(took.as_secs_f64());
    }

    pub fn tokenized(&self, model: &ModelId, result: &'static str, took: Duration) {
        self.tokenize
            .get_or_create(&TokenizeLabels {
                model: Text::Model(model.clone()),
                result: Text::Fixed(result),
            })
            .inc();
        self.tokenize_time
            .get_or_create(&ModelLabels {
                model: Text::Model(model.clone()),
            })
            .observe(took.as_secs_f64());
    }

    pub(crate) fn predicted(&self, model: &ModelId, prediction: Prediction) {
        let labels = ModelLabels {
            model: Text::Model(model.clone()),
        };
        self.prompt_tokens
            .get_or_create(&labels)
            .inc_by(u64::from(prediction.prompt_tokens));
        self.predicted_tokens
            .get_or_create(&labels)
            .inc_by(u64::from(prediction.cached_tokens));
    }

    pub fn index_reset(&self, replica: &ReplicaId) {
        let labels = ReplicaLabels {
            replica: Text::Replica(replica.clone()),
        };
        if let Some(counter) = self.index_resets.get(&labels) {
            counter.inc();
            return;
        }
        self.create(replica, Series::Resets(labels.clone()), || {
            self.index_resets.get_or_create(&labels).inc();
        });
    }

    pub fn retried(&self, model: &ModelId) {
        self.retries
            .get_or_create(&ModelLabels {
                model: Text::Model(model.clone()),
            })
            .inc();
    }

    pub fn first_byte(&self, routed: &Routed, ttft: Duration) {
        self.observe(&self.ttft, routed, ttft);
    }

    fn request(&self, model: Text, replica: Option<&ReplicaId>, status: &'static str) {
        let labels = RequestLabels {
            model,
            replica: replica.map_or(Text::Fixed(""), |r| Text::Replica(r.clone())),
            status: Text::Fixed(status),
        };
        if let Some(counter) = self.requests.get(&labels) {
            counter.inc();
            return;
        }
        match replica {
            Some(replica) => self.create(replica, Series::Requests(labels.clone()), || {
                self.requests.get_or_create(&labels).inc();
            }),
            None => {
                self.requests.get_or_create(&labels).inc();
            }
        }
    }

    /// A proxied response ended: completely, broken by the engine, or
    /// abandoned by the client.
    pub fn finished(&self, routed: &Routed, outcome: Outcome, took: Duration) {
        if let Outcome::Cancelled = outcome {
            self.disconnects
                .get_or_create(&ModelLabels {
                    model: Text::Model(routed.model.clone()),
                })
                .inc();
        }
        self.request(
            Text::Model(routed.model.clone()),
            Some(&routed.replica),
            outcome.as_str(),
        );
        self.observe(&self.duration, routed, took);
    }

    /// An engine answer passed on whole after retries ran out.
    pub fn answered(&self, model: &ModelId, replica: Option<&ReplicaId>, status: StatusCode) {
        self.request(Text::Model(model.clone()), replica, status_class(status));
    }

    /// An error the gateway answered itself.
    pub fn rejected(&self, model_id: Option<&ModelId>, err: &GatewayError, inference: bool) {
        if inference {
            self.request(model(model_id), None, status_class(err.status()));
        }
        self.errors
            .get_or_create(&ErrorLabels {
                model: model(model_id),
                reason: Text::Fixed(err.kind()),
            })
            .inc();
    }

    pub fn ejected(&self, replica: &ReplicaId) {
        let labels = ReplicaLabels {
            replica: Text::Replica(replica.clone()),
        };
        if let Some(counter) = self.ejections.get(&labels) {
            counter.inc();
            return;
        }
        self.create(replica, Series::Ejections(labels.clone()), || {
            self.ejections.get_or_create(&labels).inc();
        });
    }

    pub fn scraped(&self, replica: &ReplicaId, result: &'static str) {
        let labels = ScrapeLabels {
            replica: Text::Replica(replica.clone()),
            result: Text::Fixed(result),
        };
        if let Some(counter) = self.scrapes.get(&labels) {
            counter.inc();
            return;
        }
        self.create(replica, Series::Scrapes(labels.clone()), || {
            self.scrapes.get_or_create(&labels).inc();
        });
    }

    /// Drops every series that names a replica no longer in the fleet, so
    /// discovery churn cannot grow the page without bound.
    pub fn prune(&self) {
        let mut series = self.series.lock().unwrap_or_else(PoisonError::into_inner);
        let fleet = self.fleet.snapshot();
        let live: HashSet<&str> = fleet.replicas().iter().map(|r| r.id().as_str()).collect();
        series.retain(|replica, sets| {
            if live.contains(replica.as_str()) {
                return true;
            }
            for set in sets.drain(..) {
                match set {
                    Series::Requests(labels) => {
                        self.requests.remove(&labels);
                    }
                    Series::Route(labels) => {
                        self.decisions.remove(&labels);
                        self.ttft.remove(&labels);
                        self.duration.remove(&labels);
                    }
                    Series::Ejections(labels) => {
                        self.ejections.remove(&labels);
                    }
                    Series::Scrapes(labels) => {
                        self.scrapes.remove(&labels);
                    }
                    Series::Resets(labels) => {
                        self.index_resets.remove(&labels);
                    }
                }
            }
            false
        });
    }

    /// Prunes every `period`, so departed replicas are forgotten even when
    /// nothing scrapes the page.
    pub async fn run_pruner(self: Arc<Self>, period: Duration) {
        let mut ticker = tokio::time::interval(period);
        ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        loop {
            ticker.tick().await;
            self.prune();
        }
    }

    pub fn render(&self) -> String {
        self.prune();
        let mut out = String::with_capacity(32 * 1024);
        if prometheus_client::encoding::text::encode(&mut out, &self.registry).is_err() {
            out.clear();
        }
        out
    }
}

#[derive(Debug, Clone, Copy)]
pub enum Outcome {
    Complete(StatusCode),
    Aborted,
    Cancelled,
}

impl Outcome {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Complete(status) => status_class(status),
            Self::Aborted => "aborted",
            Self::Cancelled => "cancelled",
        }
    }
}

fn status_class(status: StatusCode) -> &'static str {
    match status.as_u16() {
        100..=199 => "1xx",
        200..=299 => "2xx",
        300..=399 => "3xx",
        400..=499 => "4xx",
        _ => "5xx",
    }
}

/// Replica gauges computed from the fleet at scrape time.
#[derive(Debug)]
pub struct FleetGauges {
    pub registry: Arc<Registry>,
    pub clock: Clock,
    pub max_age: Duration,
    pub policies: Vec<(ModelId, PolicyKind)>,
    pub index: Option<Arc<KvIndex>>,
    pub tokenizers: Vec<(ModelId, Arc<ModelTokenizer>)>,
}

struct Row {
    id: ReplicaId,
    healthy: bool,
    ejected: bool,
    load: inferrail_load::ReplicaLoad,
}

type Gauge = (&'static str, &'static str, fn(&Row) -> Option<f64>);

const GAUGES: [Gauge; 11] = [
    (
        "inferrail_replica_healthy",
        "1 while the replica passes its health checks.",
        |r| Some(f64::from(u8::from(r.healthy))),
    ),
    (
        "inferrail_replica_ejected",
        "1 while the replica is ejected for failed requests.",
        |r| Some(f64::from(u8::from(r.ejected))),
    ),
    (
        "inferrail_replica_in_flight",
        "Requests the gateway has in flight to the replica.",
        |r| Some(f64::from(r.load.in_flight)),
    ),
    (
        "inferrail_replica_pending",
        "Requests still waiting for the replica's response head.",
        |r| Some(f64::from(r.load.pending)),
    ),
    (
        "inferrail_replica_in_flight_tokens",
        "Estimated prompt tokens in flight to the replica.",
        |r| Some(r.load.in_flight_tokens as f64),
    ),
    (
        "inferrail_replica_prefill_tokens",
        "Estimated prompt tokens still waiting for their first response byte.",
        |r| Some(r.load.prefill_tokens as f64),
    ),
    (
        "inferrail_replica_engine_running",
        "Requests the engine reports running, while its report is fresh.",
        |r| r.load.engine.map(|e| f64::from(e.running)),
    ),
    (
        "inferrail_replica_engine_waiting",
        "Requests the engine reports waiting, while its report is fresh.",
        |r| r.load.engine.map(|e| f64::from(e.waiting)),
    ),
    (
        "inferrail_replica_engine_kv_usage",
        "Share of KV cache the engine reports in use, while its report is fresh.",
        |r| r.load.engine.map(|e| e.kv_usage),
    ),
    (
        "inferrail_replica_load_age_seconds",
        "Age of the engine's latest load report.",
        |r| r.load.age.map(|a| a.as_secs_f64()),
    ),
    (
        "inferrail_replica_load_stale",
        "1 when the engine's load report is missing or too old to use.",
        |r| Some(f64::from(u8::from(r.load.is_stale()))),
    ),
];

type EngineCounter = (
    &'static str,
    &'static str,
    fn(&inferrail_load::EngineLoad) -> u64,
);

const ENGINE_COUNTERS: [EngineCounter; 2] = [
    (
        "inferrail_replica_engine_prefix_cache_queries",
        "Prompt tokens the engine looked up in its prefix cache.",
        |e| e.prefix_queries,
    ),
    (
        "inferrail_replica_engine_prefix_cache_hits",
        "Prompt tokens the engine found in its prefix cache.",
        |e| e.prefix_hits,
    ),
];

impl Collector for FleetGauges {
    fn encode(&self, mut encoder: DescriptorEncoder<'_>) -> fmt::Result {
        let fleet = self.registry.snapshot();
        let now = self.clock.now();
        let instant = tokio::time::Instant::now();
        let rows: Vec<Row> = fleet
            .replicas()
            .iter()
            .map(|r| Row {
                id: r.id().clone(),
                healthy: r.health().probe_up(),
                ejected: r.health().is_ejected(now),
                load: r.load(instant, self.max_age),
            })
            .collect();
        for (name, help, read) in GAUGES {
            let mut metric = encoder.encode_descriptor(name, help, None, MetricType::Gauge)?;
            for row in &rows {
                if let Some(value) = read(row) {
                    let labels = [("replica", row.id.as_str())];
                    ConstGauge::new(value).encode(metric.encode_family(&labels)?)?;
                }
            }
        }
        for (name, help, read) in ENGINE_COUNTERS {
            let mut metric = encoder.encode_descriptor(name, help, None, MetricType::Counter)?;
            for row in &rows {
                if let Some(engine) = &row.load.engine {
                    let labels = [("replica", row.id.as_str())];
                    ConstCounter::new(read(engine)).encode(metric.encode_family(&labels)?)?;
                }
            }
        }
        self.encode_index(&mut encoder, &rows)?;
        self.encode_tokenizers(&mut encoder)?;
        let mut info = encoder.encode_descriptor(
            "inferrail_model_policy",
            "1 for the routing policy each model uses.",
            None,
            MetricType::Gauge,
        )?;
        for (model, policy) in &self.policies {
            let labels = [
                ("model", Text::Model(model.clone())),
                ("policy", Text::Fixed(policy.as_str())),
            ];
            ConstGauge::new(1i64).encode(info.encode_family(&labels)?)?;
        }
        let mut build = encoder.encode_descriptor(
            "inferrail_build_info",
            "1, labelled with the gateway version.",
            None,
            MetricType::Gauge,
        )?;
        let labels = [("version", env!("CARGO_PKG_VERSION"))];
        ConstGauge::new(1i64).encode(build.encode_family(&labels)?)?;
        Ok(())
    }
}

impl FleetGauges {
    fn encode_index(&self, encoder: &mut DescriptorEncoder<'_>, rows: &[Row]) -> fmt::Result {
        let Some(index) = &self.index else {
            return Ok(());
        };
        let stats = index.stats();
        let held: HashMap<&str, (usize, usize)> = stats
            .replicas
            .iter()
            .map(|r| (r.replica.as_str(), (r.blocks, r.capacity)))
            .collect();
        for (name, help, pick) in [
            (
                "inferrail_replica_index_blocks",
                "KV blocks the index predicts the replica holds.",
                (|h: (usize, usize)| h.0) as fn((usize, usize)) -> usize,
            ),
            (
                "inferrail_replica_index_capacity_blocks",
                "KV capacity in blocks the index assumes for the replica.",
                |h: (usize, usize)| h.1,
            ),
        ] {
            let mut metric = encoder.encode_descriptor(name, help, None, MetricType::Gauge)?;
            for row in rows {
                if let Some(&h) = held.get(row.id.as_str()) {
                    let labels = [("replica", row.id.as_str())];
                    ConstGauge::new(pick(h) as i64).encode(metric.encode_family(&labels)?)?;
                }
            }
        }
        for (name, help, value) in [
            (
                "inferrail_index_entries",
                "Distinct KV blocks the index predicts at least one replica holds.",
                stats.entries,
            ),
            (
                "inferrail_index_bytes",
                "Memory the KV index holds.",
                stats.bytes,
            ),
        ] {
            let metric = encoder.encode_descriptor(name, help, None, MetricType::Gauge)?;
            ConstGauge::new(value as i64).encode(metric)?;
        }
        Ok(())
    }

    fn encode_tokenizers(&self, encoder: &mut DescriptorEncoder<'_>) -> fmt::Result {
        if self.tokenizers.is_empty() {
            return Ok(());
        }
        let mut metric = encoder.encode_descriptor(
            "inferrail_tokenize_cache_lookups",
            "Repeated prompt pieces looked up in a model's tokenization cache, by result.",
            None,
            MetricType::Counter,
        )?;
        for (model, tokenizer) in &self.tokenizers {
            let stats = tokenizer.cache_stats();
            for (result, count) in [("hit", stats.hits), ("miss", stats.misses)] {
                let labels = [
                    ("model", Text::Model(model.clone())),
                    ("result", Text::Fixed(result)),
                ];
                ConstCounter::new(count).encode(metric.encode_family(&labels)?)?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use inferrail_control::{Limits, Origin, ReplicaSpec};
    use inferrail_load::parse::samples;

    fn spec(id: &str, group: Option<&str>) -> ReplicaSpec {
        ReplicaSpec {
            id: ReplicaId::new(id).unwrap(),
            url: format!("http://{id}:8000").parse().unwrap(),
            models: vec![ModelId::new("m").unwrap()],
            origin: group.map_or(Origin::Static, |g| Origin::Dns(ReplicaId::new(g).unwrap())),
            limits: Limits::default(),
            kv_capacity_blocks: None,
        }
    }

    fn metrics(fleet: &Arc<Registry>) -> Metrics {
        Metrics::new(FleetGauges {
            registry: Arc::clone(fleet),
            clock: Clock::default(),
            max_age: Duration::from_millis(250),
            policies: vec![(ModelId::new("m").unwrap(), PolicyKind::P2c)],
            index: None,
            tokenizers: Vec::new(),
        })
    }

    fn routed(replica: &str) -> Routed {
        Routed {
            model: ModelId::new("m").unwrap(),
            policy: PolicyKind::P2c,
            replica: ReplicaId::new(replica).unwrap(),
        }
    }

    fn value(page: &str, name: &str, labels: &[(&str, &str)]) -> Option<f64> {
        samples(page)
            .find(|s| {
                s.name == name
                    && labels
                        .iter()
                        .all(|(k, v)| s.label(k).as_deref() == Some(*v))
            })
            .map(|s| s.value)
    }

    #[test]
    fn outcomes_land_in_the_right_series() {
        let fleet = Arc::new(Registry::new(vec![spec("a", None)]));
        let m = metrics(&fleet);
        let a = routed("a");
        m.decided(&a, Duration::from_micros(3));
        m.first_byte(&a, Duration::from_millis(40));
        m.finished(
            &a,
            Outcome::Complete(StatusCode::OK),
            Duration::from_millis(90),
        );
        m.finished(
            &a,
            Outcome::Complete(StatusCode::BAD_REQUEST),
            Duration::from_millis(1),
        );
        m.finished(&a, Outcome::Aborted, Duration::from_millis(5));
        m.finished(&a, Outcome::Cancelled, Duration::from_millis(5));
        m.retried(&a.model);
        m.rejected(None, &GatewayError::not_found(), false);
        m.rejected(Some(&a.model), &GatewayError::unavailable("full"), true);
        m.answered(&a.model, Some(&a.replica), StatusCode::SERVICE_UNAVAILABLE);
        let page = m.render();
        let ok = [("model", "m"), ("replica", "a"), ("status", "2xx")];
        assert_eq!(value(&page, "inferrail_requests_total", &ok), Some(1.0));
        for status in ["4xx", "aborted", "cancelled"] {
            let labels = [("replica", "a"), ("status", status)];
            assert_eq!(
                value(&page, "inferrail_requests_total", &labels),
                Some(1.0),
                "{status}"
            );
        }
        let answered = [("replica", "a"), ("status", "5xx")];
        assert_eq!(
            value(&page, "inferrail_requests_total", &answered),
            Some(1.0)
        );
        let local = [("model", "m"), ("replica", ""), ("status", "5xx")];
        assert_eq!(value(&page, "inferrail_requests_total", &local), Some(1.0));
        let route = [("model", "m"), ("policy", "p2c"), ("replica", "a")];
        assert_eq!(
            value(&page, "inferrail_routing_decisions_total", &route),
            Some(1.0)
        );
        assert_eq!(
            value(&page, "inferrail_ttft_seconds_count", &route),
            Some(1.0)
        );
        assert_eq!(
            value(&page, "inferrail_request_duration_seconds_count", &route),
            Some(4.0)
        );
        assert_eq!(
            value(
                &page,
                "inferrail_client_disconnects_total",
                &[("model", "m")]
            ),
            Some(1.0)
        );
        assert_eq!(
            value(&page, "inferrail_upstream_retries_total", &[("model", "m")]),
            Some(1.0)
        );
        let not_found = [("model", ""), ("reason", "NotFoundError")];
        assert_eq!(
            value(&page, "inferrail_request_errors_total", &not_found),
            Some(1.0)
        );
        let full = [("model", "m"), ("reason", "ServiceUnavailableError")];
        assert_eq!(
            value(&page, "inferrail_request_errors_total", &full),
            Some(1.0)
        );
        assert_eq!(
            value(&page, "inferrail_replica_in_flight", &[("replica", "a")]),
            Some(0.0)
        );
        assert_eq!(
            value(&page, "inferrail_replica_load_stale", &[("replica", "a")]),
            Some(1.0)
        );
        assert_eq!(
            value(
                &page,
                "inferrail_replica_engine_running",
                &[("replica", "a")]
            ),
            None
        );
        assert_eq!(
            value(
                &page,
                "inferrail_model_policy",
                &[("model", "m"), ("policy", "p2c")]
            ),
            Some(1.0)
        );
        assert!(page.contains("inferrail_build_info{version=\""));
        assert!(page.ends_with("# EOF\n"));
    }

    #[test]
    fn replicas_that_leave_take_their_series_with_them() {
        let fleet = Arc::new(Registry::new(vec![spec("static", None)]));
        let group = ReplicaId::new("pool").unwrap();
        let m = metrics(&fleet);
        let before = m.render().len();
        for round in 0..50 {
            let ids: Vec<String> = (0..4).map(|i| format!("pool-{round}-{i}")).collect();
            fleet.set_group(
                &group,
                ids.iter().map(|id| spec(id, Some("pool"))).collect(),
            );
            for id in &ids {
                let r = routed(id);
                m.decided(&r, Duration::from_micros(1));
                m.first_byte(&r, Duration::from_millis(1));
                m.finished(
                    &r,
                    Outcome::Complete(StatusCode::OK),
                    Duration::from_millis(2),
                );
                m.ejected(&r.replica);
                m.scraped(&r.replica, "ok");
            }
            assert!(m.render().contains(&format!("pool-{round}-3")));
        }
        fleet.set_group(&group, Vec::new());
        let page = m.render();
        assert!(!page.contains("pool-"), "{page}");
        assert!(page.len() < before + 4096, "{} vs {before}", page.len());
        assert!(m.series.lock().unwrap().is_empty());
    }

    #[test]
    fn recording_while_pruning_never_strands_a_series() {
        use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
        let fleet = Arc::new(Registry::new(Vec::new()));
        let m = Arc::new(metrics(&fleet));
        let group = ReplicaId::new("pool").unwrap();
        let (stop, generation) = (
            Arc::new(AtomicBool::new(false)),
            Arc::new(AtomicU64::new(0)),
        );
        let workers: Vec<_> = (0..4)
            .map(|w| {
                let (m, stop, generation) =
                    (Arc::clone(&m), Arc::clone(&stop), Arc::clone(&generation));
                std::thread::spawn(move || {
                    while !stop.load(Ordering::Relaxed) {
                        let g = generation.load(Ordering::Relaxed);
                        let r = routed(&format!("pool-{g}-{}", w % 2));
                        m.decided(&r, Duration::from_micros(1));
                        m.finished(
                            &r,
                            Outcome::Complete(StatusCode::OK),
                            Duration::from_millis(1),
                        );
                        m.ejected(&r.replica);
                        m.scraped(&r.replica, "ok");
                    }
                })
            })
            .collect();
        for g in 0..400u64 {
            fleet.set_group(
                &group,
                (0..2)
                    .map(|i| spec(&format!("pool-{g}-{i}"), Some("pool")))
                    .collect(),
            );
            generation.store(g, Ordering::Relaxed);
            m.prune();
        }
        stop.store(true, Ordering::Relaxed);
        for w in workers {
            w.join().unwrap();
        }
        fleet.set_group(&group, Vec::new());
        let page = m.render();
        assert_eq!(page.matches("pool-").count(), 0, "stranded series");
    }

    #[tokio::test(start_paused = true)]
    async fn departed_replicas_are_forgotten_without_a_scrape() {
        let fleet = Arc::new(Registry::new(Vec::new()));
        let m = Arc::new(metrics(&fleet));
        let pruner = tokio::spawn(Arc::clone(&m).run_pruner(Duration::from_secs(10)));
        let group = ReplicaId::new("pool").unwrap();
        for g in 0..200 {
            let id = format!("pool-{g}");
            fleet.set_group(&group, vec![spec(&id, Some("pool"))]);
            let r = routed(&id);
            m.decided(&r, Duration::from_micros(1));
            m.finished(
                &r,
                Outcome::Complete(StatusCode::OK),
                Duration::from_millis(1),
            );
        }
        assert_eq!(m.series.lock().unwrap().len(), 200);
        tokio::time::sleep(Duration::from_secs(11)).await;
        let tracked: Vec<String> = m
            .series
            .lock()
            .unwrap()
            .keys()
            .map(ToString::to_string)
            .collect();
        assert_eq!(tracked, ["pool-199"]);
        pruner.abort();
    }

    #[test]
    fn label_values_are_escaped() {
        let fleet = Arc::new(Registry::new(vec![spec("a", None)]));
        let quoted = ModelId::new(r#"org/we"ird\name"#).unwrap();
        let m = Metrics::new(FleetGauges {
            registry: Arc::clone(&fleet),
            clock: Clock::default(),
            max_age: Duration::from_millis(250),
            policies: vec![(quoted.clone(), PolicyKind::P2c)],
            index: None,
            tokenizers: Vec::new(),
        });
        let routed = Routed {
            model: quoted.clone(),
            policy: PolicyKind::P2c,
            replica: ReplicaId::new("a").unwrap(),
        };
        m.decided(&routed, Duration::from_micros(1));
        m.finished(
            &routed,
            Outcome::Complete(StatusCode::OK),
            Duration::from_millis(1),
        );
        m.rejected(Some(&quoted), &GatewayError::not_found(), true);
        let page = m.render();
        assert!(page.contains(r#"model="org/we\"ird\\name""#), "{page}");
        assert!(!page.contains(r#"we"ird"#), "{page}");
        let models: Vec<String> = samples(&page)
            .filter_map(|s| s.label("model").map(|v| v.into_owned()))
            .filter(|v| !v.is_empty())
            .collect();
        assert!(models.len() >= 5, "{models:?}");
        assert!(models.iter().all(|v| v == quoted.as_str()), "{models:?}");
    }
}
