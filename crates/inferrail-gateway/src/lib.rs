//! OpenAI compatible gateway that routes requests across engine replicas.

use std::future::Future;
use std::hash::{BuildHasher, RandomState};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Duration;

use arc_swap::ArcSwap;
use inferrail_control::discovery::refresh;
use inferrail_control::{
    CheckerConfig, Clock, HealthPolicy, Registry, ReplicaSpec, Resolve, SystemResolver,
    run_checker, run_dns,
};
use inferrail_core::config::{Config, ConfigError, ConfigIssues, PolicyKind};
use inferrail_index::KvIndex;
use inferrail_load::{CollectorConfig, run_collector};
use tokio::net::TcpListener;
use tokio::task::JoinSet;
use xxhash_rust::xxh3::xxh3_64;

mod admin;
mod body;
mod control;
mod decision_log;
pub mod errors;
mod explain;
pub mod headers;
pub mod inspect;
mod live;
pub mod metrics;
mod models;
mod prompts;
mod proxy;
pub mod request_id;
mod route;
mod server;
pub mod telemetry;
mod upstream;

pub use body::ResponseBody;

#[derive(Debug, thiserror::Error)]
pub enum BuildError {
    #[error("invalid config: {0}")]
    Invalid(ConfigIssues),
    #[error("replica {0} uses https; TLS to engines is not supported yet")]
    Https(String),
    #[error("cannot open the decision log {path}: {source}")]
    DecisionLog {
        path: std::path::PathBuf,
        source: std::io::Error,
    },
    #[error("cannot start the tokenizer workers: {0}")]
    Workers(std::io::Error),
}

#[derive(Debug, thiserror::Error)]
pub enum ReloadError {
    #[error("{0}")]
    Load(#[from] ConfigError),
    #[error("invalid config: {0}")]
    Invalid(ConfigIssues),
    #[error("only a restart applies changes to {}", .0.join(", "))]
    Restart(Vec<String>),
    #[error("replica {0} uses https; TLS to engines is not supported yet")]
    Https(String),
}

/// What a reload applied: the config generation now live and the sections
/// that changed, none when the new config was the same.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Reloaded {
    pub generation: u64,
    pub changed: Vec<String>,
}

const PRUNE_EVERY: std::time::Duration = std::time::Duration::from_secs(10);

/// A configured gateway. Cheap to clone; clones share all state.
#[derive(Debug, Clone)]
pub struct Gateway {
    shared: Arc<Shared>,
}

/// The gateway's state. `config` is the config it started with, which
/// holds every section a reload cannot change; `live` holds the rest.
#[derive(Debug)]
pub(crate) struct Shared {
    config: Config,
    live: ArcSwap<live::Live>,
    seed: u64,
    reloading: Mutex<()>,
    source: Mutex<Option<PathBuf>>,
    registry: Arc<Registry>,
    client: upstream::Client,
    ids: request_id::RequestIds,
    clock: Clock,
    health: HealthPolicy,
    started_unix: u64,
    resolver: Arc<dyn Resolve>,
    ejecting: Mutex<()>,
    metrics: Arc<metrics::Metrics>,
    decisions: Option<decision_log::DecisionLog>,
    prompts: prompts::Prompts,
    index: Option<Arc<KvIndex>>,
}

impl Shared {
    pub(crate) fn live(&self) -> Arc<live::Live> {
        self.live.load_full()
    }

    pub(crate) fn source(&self) -> Option<PathBuf> {
        self.source
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }
}

fn check_https(config: &Config) -> Result<(), String> {
    let urls = config
        .replicas
        .iter()
        .map(|r| (r.id.as_str(), &r.url))
        .chain(
            config
                .discovery
                .dns
                .iter()
                .map(|g| (g.name.as_str(), &g.url)),
        );
    for (id, url) in urls {
        if url.scheme() == "https" {
            return Err(id.to_owned());
        }
    }
    Ok(())
}

/// Warns about each model whose policy routes on cached prefixes it will
/// never be told about.
fn warn_blind(config: &Config, prompts: &prompts::Prompts, indexed: bool) {
    for model in &config.models {
        let kind = config.policy_for(model);
        let sees_prefixes =
            prompts.tokenizes(&model.name) && (kind == PolicyKind::PrefixHash || indexed);
        if kind.is_cache_aware() && !sees_prefixes {
            tracing::warn!(
                model = %model.name,
                policy = %kind,
                "the policy routes on cached prefixes, but this model gets no cache predictions; it routes on load alone"
            );
        }
    }
}

impl Gateway {
    /// Builds the gateway; must be called inside a Tokio runtime.
    pub fn new(config: Config) -> Result<Self, BuildError> {
        Self::with_resolver(config, Arc::new(SystemResolver))
    }

    /// Every check startup makes, without opening files for writing or
    /// starting anything: what `--check` runs. Tokenizers that cannot be
    /// loaded do not stop startup, since their models still route, so they
    /// come back as warnings.
    pub fn check(config: &Config) -> Result<Vec<String>, BuildError> {
        Self::check_startup(config)?;
        let (_, failures) = prompts::load_tokenizers(config);
        Ok(failures
            .iter()
            .map(|f| {
                format!(
                    "model {}: {}; it routes without cache predictions",
                    f.model, f.error
                )
            })
            .collect())
    }

    fn check_startup(config: &Config) -> Result<(), BuildError> {
        config.validate().map_err(BuildError::Invalid)?;
        check_https(config).map_err(BuildError::Https)?;
        let log = &config.observability.decision_log;
        if let Some(path) = log.path.as_deref().filter(|_| log.sample_rate > 0.0) {
            decision_log::writable(path).map_err(|source| BuildError::DecisionLog {
                path: path.to_owned(),
                source,
            })?;
        }
        Ok(())
    }

    /// Like [`Gateway::new`] with a custom resolver for DNS discovery.
    pub fn with_resolver(config: Config, resolver: Arc<dyn Resolve>) -> Result<Self, BuildError> {
        Self::check_startup(&config)?;
        let (tokenizers, failures) = prompts::load_tokenizers(&config);
        for failure in &failures {
            tracing::warn!(model = %failure.model, error = %failure.error, "cannot load the tokenizer; routing this model without cache predictions");
        }
        let prompts = prompts::Prompts::new(&config, &tokenizers).map_err(BuildError::Workers)?;
        let index = (config.index.enabled && prompts.any()).then(|| Arc::new(KvIndex::new()));
        let seed = RandomState::new().hash_one("inferrail routing");
        warn_blind(&config, &prompts, index.is_some());
        let live = live::Live::build(config.clone(), seed, None);
        let registry = Arc::new(Registry::new(ReplicaSpec::statics(&config)));
        let clock = Clock::default();
        let metrics = Arc::new(metrics::Metrics::new(metrics::FleetGauges {
            registry: Arc::clone(&registry),
            clock,
            max_age: config.load.max_age,
            policies: Arc::new(ArcSwap::from_pointee(live.policies())),
            index: index.clone(),
            tokenized: config
                .models
                .iter()
                .filter(|m| tokenizers.contains_key(&m.name))
                .map(|m| m.name.clone())
                .collect(),
        }));
        let log = &config.observability.decision_log;
        let decisions = match &log.path {
            Some(path) if log.sample_rate > 0.0 => {
                let (written, dropped) = metrics.decision_counters();
                Some(
                    decision_log::DecisionLog::open(path, log.sample_rate, written, dropped)
                        .map_err(|source| BuildError::DecisionLog {
                            path: path.clone(),
                            source,
                        })?,
                )
            }
            _ => None,
        };
        let shared = Shared {
            decisions,
            registry,
            client: upstream::client(&config.upstream),
            ids: request_id::RequestIds::default(),
            clock,
            metrics,
            health: HealthPolicy::from(&config.health),
            started_unix: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| d.as_secs()),
            resolver,
            ejecting: Mutex::new(()),
            live: ArcSwap::from_pointee(live),
            seed,
            reloading: Mutex::new(()),
            source: Mutex::new(None),
            prompts,
            index,
            config,
        };
        Ok(Self {
            shared: Arc::new(shared),
        })
    }

    pub(crate) fn from_shared(shared: Arc<Shared>) -> Self {
        Self { shared }
    }

    /// The file `/admin/reload` rereads.
    pub fn set_config_path(&self, path: PathBuf) {
        *self
            .shared
            .source
            .lock()
            .unwrap_or_else(PoisonError::into_inner) = Some(path);
    }

    /// The config in effect: the startup one with every reload since.
    pub fn config(&self) -> Arc<Config> {
        Arc::clone(&self.shared.live().config)
    }

    /// Applies `next` without dropping a request: requests already routed
    /// finish with the policies that routed them, and policies whose
    /// settings did not change keep their state. Sections only a restart
    /// can change are refused, leaving the running config as it was.
    pub fn reload(&self, next: Config) -> Result<Reloaded, ReloadError> {
        let shared = &self.shared;
        let _one_at_a_time = shared
            .reloading
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        let applied = Self::apply(shared, next);
        shared.metrics.reloaded(match &applied {
            Ok(r) if r.changed.is_empty() => "unchanged",
            Ok(_) => "applied",
            Err(_) => "rejected",
        });
        applied
    }

    fn apply(shared: &Shared, next: Config) -> Result<Reloaded, ReloadError> {
        next.validate().map_err(ReloadError::Invalid)?;
        let current = shared.live();
        let restart = current.config.restart_changes(&next);
        if !restart.is_empty() {
            return Err(ReloadError::Restart(restart));
        }
        check_https(&next).map_err(ReloadError::Https)?;
        let changed = current.config.changed_sections(&next);
        if changed.is_empty() {
            return Ok(Reloaded {
                generation: current.generation,
                changed,
            });
        }
        warn_blind(&next, &shared.prompts, shared.index.is_some());
        let live = live::Live::build(next, shared.seed, Some(&current));
        if live.config.replicas != current.config.replicas {
            let change = shared
                .registry
                .set_statics(ReplicaSpec::statics(&live.config));
            for id in &change.rejected {
                tracing::warn!(replica = %id, "a discovered replica already has this id; ignoring it");
            }
            tracing::info!(added = ?change.added, removed = ?change.removed, "static replicas changed");
        }
        let generation = live.generation;
        shared.metrics.set_policies(live.policies());
        shared.metrics.set_generation(generation);
        shared.live.store(Arc::new(live));
        tracing::info!(generation, changed = ?changed, "config reloaded");
        Ok(Reloaded {
            generation,
            changed,
        })
    }

    /// Reads, checks and applies the config file at `path`.
    pub fn reload_from(&self, path: &Path) -> Result<Reloaded, ReloadError> {
        let next = match Config::parse_unvalidated(&read(path)?) {
            Ok(next) => next,
            Err(err) => {
                self.shared.metrics.reloaded("rejected");
                return Err(err.into());
            }
        };
        self.reload(next)
    }

    /// Rereads `path` every `interval` and applies it whenever its contents
    /// change and differ from the config in effect, until the returned
    /// future is dropped.
    pub async fn watch_config(self, path: PathBuf, interval: Duration) {
        let mut seen = None;
        let mut ticker = tokio::time::interval(interval);
        ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        loop {
            ticker.tick().await;
            let reading = path.clone();
            let Ok(Ok(text)) =
                tokio::task::spawn_blocking(move || std::fs::read_to_string(reading)).await
            else {
                continue;
            };
            let hash = Some(xxh3_64(text.as_bytes()));
            if hash == seen {
                continue;
            }
            seen = hash;
            let applied = match Config::parse_unvalidated(&text) {
                Ok(next) if next == *self.config() => continue,
                Ok(next) => self.reload(next),
                Err(err) => {
                    self.shared.metrics.reloaded("rejected");
                    Err(err.into())
                }
            };
            if let Err(err) = applied {
                tracing::warn!(path = %path.display(), error = %err, "config change not applied");
            }
        }
    }

    pub fn registry(&self) -> &Registry {
        &self.shared.registry
    }

    /// Serves `GET /metrics`, `GET /health` and `POST /admin/explain/v1/...`
    /// on the admin listener until `shutdown` resolves.
    pub async fn serve_admin(
        self,
        listener: TcpListener,
        shutdown: impl Future<Output = ()> + Send + 'static,
    ) -> std::io::Result<()> {
        axum::serve(listener, admin::router(self.shared))
            .with_graceful_shutdown(shutdown)
            .await
    }

    /// The gateway's Prometheus metrics in OpenMetrics text format.
    pub fn render_metrics(&self) -> String {
        self.shared.metrics.render()
    }

    /// The clock health and ejection deadlines are measured on.
    pub fn clock(&self) -> Clock {
        self.shared.clock
    }

    /// Serves until `shutdown` resolves, then drains open connections for at
    /// most `server.shutdown_timeout`. DNS groups are resolved once before
    /// the first connection; health checks and DNS refreshes run alongside.
    pub async fn serve(
        self,
        listener: TcpListener,
        shutdown: impl Future<Output = ()> + Send + 'static,
    ) -> std::io::Result<()> {
        let shared = self.shared;
        let mut first = JoinSet::new();
        for group in &shared.config.discovery.dns {
            let registry = Arc::clone(&shared.registry);
            let resolver = Arc::clone(&shared.resolver);
            let group = group.clone();
            first.spawn(async move {
                let wait = group.refresh.max(std::time::Duration::from_secs(1));
                refresh(&registry, &group, resolver.as_ref(), wait).await;
            });
        }
        while first.join_next().await.is_some() {}
        let mut background = JoinSet::new();
        for group in &shared.config.discovery.dns {
            background.spawn(run_dns(
                Arc::clone(&shared.registry),
                group.clone(),
                Arc::clone(&shared.resolver),
            ));
        }
        let health = &shared.config.health;
        background.spawn(run_checker(
            Arc::clone(&shared.registry),
            upstream::HttpProbe {
                client: shared.client.clone(),
                cards_every: health.interval.saturating_mul(10),
            },
            CheckerConfig {
                interval: health.interval,
                timeout: health.timeout,
                policy: shared.health,
            },
            shared.clock,
        ));
        background.spawn(Arc::clone(&shared.metrics).run_pruner(PRUNE_EVERY));
        if let Some(index) = &shared.index {
            background.spawn(forget_departed(
                Arc::clone(&shared.registry),
                Arc::clone(index),
                PRUNE_EVERY,
            ));
        }
        if shared.config.load.enabled {
            let registry = Arc::clone(&shared.registry);
            background.spawn(run_collector(
                move || registry.snapshot().replicas().to_vec(),
                upstream::HttpScrape {
                    client: shared.client.clone(),
                    metrics: Arc::clone(&shared.metrics),
                    index: shared.index.clone(),
                },
                CollectorConfig {
                    interval: shared.config.load.scrape_interval,
                    timeout: shared.config.load.scrape_timeout,
                },
            ));
        }
        server::serve(shared, listener, shutdown).await
    }
}

fn read(path: &Path) -> Result<String, ReloadError> {
    std::fs::read_to_string(path).map_err(|source| {
        ReloadError::Load(ConfigError::Io {
            path: path.to_owned(),
            source,
        })
    })
}

/// Drops index state of replicas that left the fleet, every `period`.
async fn forget_departed(
    registry: Arc<Registry>,
    index: Arc<KvIndex>,
    period: std::time::Duration,
) {
    let mut ticker = tokio::time::interval(period);
    ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    loop {
        ticker.tick().await;
        prune_index(&registry, &index);
    }
}

fn prune_index(registry: &Registry, index: &KvIndex) {
    let fleet = registry.snapshot();
    let live: std::collections::HashSet<&str> =
        fleet.replicas().iter().map(|r| r.id().as_str()).collect();
    for id in index.replicas() {
        if !live.contains(id.as_str()) {
            index.forget(&id);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use inferrail_control::{Limits, Origin};
    use inferrail_core::ids::{BlockHash, ModelId, ReplicaId};

    fn member(group: &ReplicaId, id: &str) -> ReplicaSpec {
        ReplicaSpec {
            id: ReplicaId::new(id).unwrap(),
            url: format!("http://{id}.pool:8000").parse().unwrap(),
            models: vec![ModelId::new("m").unwrap()],
            origin: Origin::Dns(group.clone()),
            limits: Limits::default(),
            kv_capacity_blocks: None,
        }
    }

    #[test]
    fn replicas_that_left_the_fleet_are_forgotten() {
        let registry = Registry::new(Vec::new());
        let group = ReplicaId::new("pool").unwrap();
        registry.set_group(&group, vec![member(&group, "a"), member(&group, "b")]);
        let index = KvIndex::new();
        for id in ["a", "b", "never-seen"] {
            let slot = index.slot(&ReplicaId::new(id).unwrap());
            index.record(slot, &[BlockHash::new(1), BlockHash::new(2)], 10);
        }
        registry.set_group(&group, vec![member(&group, "a")]);
        prune_index(&registry, &index);
        assert_eq!(index.replicas(), [ReplicaId::new("a").unwrap()]);
        assert_eq!(index.stats().entries, 2);
        index.verify().unwrap();
    }
}
