//! OpenAI compatible gateway that routes requests across engine replicas.

use std::collections::HashMap;
use std::future::Future;
use std::hash::{BuildHasher, RandomState};
use std::sync::{Arc, Mutex};

use inferrail_control::discovery::refresh;
use inferrail_control::{
    CheckerConfig, Clock, HealthPolicy, Registry, ReplicaSpec, Resolve, SystemResolver,
    run_checker, run_dns,
};
use inferrail_core::config::{Config, ConfigIssues};
use inferrail_core::ids::ModelId;
use inferrail_index::KvIndex;
use inferrail_load::{CollectorConfig, run_collector};
use inferrail_policy::RoutingPolicy;
use tokio::net::TcpListener;
use tokio::task::JoinSet;

mod admin;
mod body;
mod decision_log;
pub mod errors;
pub mod headers;
pub mod inspect;
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

const PRUNE_EVERY: std::time::Duration = std::time::Duration::from_secs(10);

/// A configured gateway. Cheap to clone; clones share all state.
#[derive(Debug, Clone)]
pub struct Gateway {
    shared: Arc<Shared>,
}

#[derive(Debug)]
pub(crate) struct Shared {
    config: Config,
    registry: Arc<Registry>,
    names: HashMap<String, ModelId>,
    policies: HashMap<ModelId, Box<dyn RoutingPolicy>>,
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
    fn policy(&self, model: &ModelId) -> Result<&dyn RoutingPolicy, errors::GatewayError> {
        self.policies
            .get(model)
            .map(Box::as_ref)
            .ok_or_else(|| errors::GatewayError::unknown_model(model.as_str()))
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
                return Err(BuildError::Https(id.to_owned()));
            }
        }
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
        let seed = config
            .routing
            .seed
            .unwrap_or_else(|| RandomState::new().hash_one("inferrail routing"));
        let mut names = HashMap::new();
        let mut policies = HashMap::new();
        for model in &config.models {
            names.insert(model.name.as_str().to_owned(), model.name.clone());
            for alias in &model.aliases {
                names.insert(alias.clone(), model.name.clone());
            }
            let kind = config.policy_for(model);
            if !inferrail_policy::is_native(kind) {
                tracing::warn!(
                    model = %model.name,
                    policy = %kind,
                    "policy needs the KV index, which is not built yet; routing with power of two choices"
                );
            }
            let model_seed = inferrail_policy::mix_seed(seed, model.name.as_str());
            policies.insert(
                model.name.clone(),
                inferrail_policy::build(kind, model_seed),
            );
        }
        let registry = Arc::new(Registry::new(ReplicaSpec::statics(&config)));
        let clock = Clock::default();
        let metrics = Arc::new(metrics::Metrics::new(metrics::FleetGauges {
            registry: Arc::clone(&registry),
            clock,
            max_age: config.load.max_age,
            policies: config
                .models
                .iter()
                .map(|m| (m.name.clone(), config.policy_for(m)))
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
            names,
            policies,
            prompts,
            index,
            config,
        };
        Ok(Self {
            shared: Arc::new(shared),
        })
    }

    pub fn config(&self) -> &Config {
        &self.shared.config
    }

    pub fn registry(&self) -> &Registry {
        &self.shared.registry
    }

    /// Serves `GET /metrics` and `GET /health` on the admin listener until
    /// `shutdown` resolves.
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
        if shared.config.load.enabled {
            let registry = Arc::clone(&shared.registry);
            background.spawn(run_collector(
                move || registry.snapshot().replicas().to_vec(),
                upstream::HttpScrape {
                    client: shared.client.clone(),
                    metrics: Arc::clone(&shared.metrics),
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
