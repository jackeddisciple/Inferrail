//! OpenAI compatible gateway that routes requests across engine replicas.

use std::collections::HashMap;
use std::future::Future;
use std::hash::{BuildHasher, RandomState};
use std::sync::Arc;

use inferrail_control::{CheckerConfig, Clock, HealthPolicy, Registry, ReplicaSpec, run_checker};
use inferrail_core::config::{Config, ConfigIssues};
use inferrail_core::ids::ModelId;
use inferrail_policy::RoutingPolicy;
use tokio::net::TcpListener;

mod body;
pub mod errors;
pub mod headers;
pub mod inspect;
mod models;
mod proxy;
pub mod request_id;
mod route;
mod server;
mod upstream;

pub use body::ResponseBody;

#[derive(Debug, thiserror::Error)]
pub enum BuildError {
    #[error("invalid config: {0}")]
    Invalid(ConfigIssues),
    #[error("replica {0} uses https; TLS to engines is not supported yet")]
    Https(String),
}

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
        let shared = Shared {
            registry: Arc::new(Registry::new(ReplicaSpec::statics(&config))),
            client: upstream::client(&config.upstream),
            ids: request_id::RequestIds::default(),
            clock: Clock::default(),
            health: HealthPolicy::from(&config.health),
            started_unix: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| d.as_secs()),
            names,
            policies,
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

    /// Serves until `shutdown` resolves, then drains open connections for at
    /// most `server.shutdown_timeout`. Health checks run while it serves.
    pub async fn serve(
        self,
        listener: TcpListener,
        shutdown: impl Future<Output = ()> + Send + 'static,
    ) -> std::io::Result<()> {
        let shared = self.shared;
        let health = &shared.config.health;
        let checker = tokio::spawn(run_checker(
            Arc::clone(&shared.registry),
            upstream::HttpProbe {
                client: shared.client.clone(),
            },
            CheckerConfig {
                interval: health.interval,
                timeout: health.timeout,
                policy: shared.health,
            },
            shared.clock,
        ));
        let served = server::serve(shared, listener, shutdown).await;
        checker.abort();
        served
    }
}
