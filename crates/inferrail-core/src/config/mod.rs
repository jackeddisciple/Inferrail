//! Gateway configuration: YAML schema, defaults and loading.

use std::fmt;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::time::Duration;

use serde::{Deserialize, Serialize};
use url::Url;

use crate::ids::{ModelId, ReplicaId};

pub const DEFAULT_BLOCK_SIZE: u32 = 16;
pub const DEFAULT_EPSILON: f64 = 0.25;
pub const DEFAULT_KAPPA: f64 = 0.92;

#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("cannot read config {path}: {source}")]
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("cannot parse config: {0}")]
    Parse(#[from] serde_yaml_ng::Error),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    #[serde(default)]
    pub server: ServerConfig,
    pub models: Vec<ModelConfig>,
    pub replicas: Vec<ReplicaConfig>,
    #[serde(default)]
    pub routing: RoutingConfig,
    #[serde(default)]
    pub health: HealthConfig,
    #[serde(default)]
    pub retry: RetryConfig,
    #[serde(default)]
    pub logging: LoggingConfig,
}

impl Config {
    pub fn from_yaml_str(text: &str) -> Result<Self, ConfigError> {
        Ok(serde_yaml_ng::from_str(text)?)
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Self, ConfigError> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path).map_err(|source| ConfigError::Io {
            path: path.to_owned(),
            source,
        })?;
        Self::from_yaml_str(&text)
    }

    pub fn to_yaml_string(&self) -> Result<String, ConfigError> {
        Ok(serde_yaml_ng::to_string(self)?)
    }

    pub fn model(&self, name: &str) -> Option<&ModelConfig> {
        self.models
            .iter()
            .find(|m| m.name.as_str() == name || m.aliases.iter().any(|a| a == name))
    }

    pub fn policy_for(&self, model: &ModelConfig) -> PolicyKind {
        model.policy.unwrap_or(self.routing.default_policy)
    }

    pub fn replicas_serving<'a>(
        &'a self,
        model: &'a ModelId,
    ) -> impl Iterator<Item = &'a ReplicaConfig> + 'a {
        self.replicas
            .iter()
            .filter(move |r| r.models.contains(model))
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct ServerConfig {
    pub listen: SocketAddr,
    pub admin_listen: SocketAddr,
    #[serde(with = "humantime_serde")]
    pub request_timeout: Duration,
    pub max_body_bytes: usize,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            listen: SocketAddr::from(([0, 0, 0, 0], 8080)),
            admin_listen: SocketAddr::from(([127, 0, 0, 1], 9090)),
            request_timeout: Duration::from_secs(600),
            max_body_bytes: 32 * 1024 * 1024,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelConfig {
    pub name: ModelId,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub aliases: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub policy: Option<PolicyKind>,
    #[serde(default = "default_block_size")]
    pub block_size: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tokenizer: Option<String>,
}

fn default_block_size() -> u32 {
    DEFAULT_BLOCK_SIZE
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReplicaConfig {
    pub id: ReplicaId,
    pub url: Url,
    pub models: Vec<ModelId>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kv_capacity_blocks: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_concurrency: Option<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PolicyKind {
    RoundRobin,
    Random,
    LeastRequest,
    P2c,
    PrefixHash,
    KvAware,
    CostModel,
}

impl PolicyKind {
    pub const ALL: [Self; 7] = [
        Self::RoundRobin,
        Self::Random,
        Self::LeastRequest,
        Self::P2c,
        Self::PrefixHash,
        Self::KvAware,
        Self::CostModel,
    ];

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::RoundRobin => "round_robin",
            Self::Random => "random",
            Self::LeastRequest => "least_request",
            Self::P2c => "p2c",
            Self::PrefixHash => "prefix_hash",
            Self::KvAware => "kv_aware",
            Self::CostModel => "cost_model",
        }
    }

    pub const fn is_cache_aware(self) -> bool {
        matches!(self, Self::PrefixHash | Self::KvAware | Self::CostModel)
    }
}

impl fmt::Display for PolicyKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct RoutingConfig {
    pub default_policy: PolicyKind,
    pub epsilon: f64,
    pub kappa: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
}

impl Default for RoutingConfig {
    fn default() -> Self {
        Self {
            default_policy: PolicyKind::CostModel,
            epsilon: DEFAULT_EPSILON,
            kappa: DEFAULT_KAPPA,
            seed: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct HealthConfig {
    #[serde(with = "humantime_serde")]
    pub interval: Duration,
    #[serde(with = "humantime_serde")]
    pub timeout: Duration,
    pub unhealthy_threshold: u32,
    pub healthy_threshold: u32,
    #[serde(with = "humantime_serde")]
    pub ejection_backoff: Duration,
    #[serde(with = "humantime_serde")]
    pub max_ejection_backoff: Duration,
}

impl Default for HealthConfig {
    fn default() -> Self {
        Self {
            interval: Duration::from_secs(1),
            timeout: Duration::from_millis(500),
            unhealthy_threshold: 3,
            healthy_threshold: 2,
            ejection_backoff: Duration::from_secs(5),
            max_ejection_backoff: Duration::from_secs(60),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct RetryConfig {
    pub max_attempts: u32,
    #[serde(with = "humantime_serde")]
    pub backoff: Duration,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            backoff: Duration::from_millis(25),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct LoggingConfig {
    pub level: String,
    pub log_prompts: bool,
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: "info".to_owned(),
            log_prompts: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MINIMAL: &str = r#"
models:
  - name: meta-llama/Llama-3.1-8B-Instruct
replicas:
  - id: r0
    url: http://10.0.0.10:8000
    models: [meta-llama/Llama-3.1-8B-Instruct]
"#;

    #[test]
    fn minimal_config_gets_safe_defaults() {
        let config = Config::from_yaml_str(MINIMAL).unwrap();
        assert_eq!(config.server, ServerConfig::default());
        assert_eq!(config.routing.default_policy, PolicyKind::CostModel);
        assert_eq!(config.routing.epsilon, 0.25);
        assert_eq!(config.routing.kappa, 0.92);
        assert_eq!(config.models[0].block_size, 16);
        assert!(!config.logging.log_prompts);
        assert_eq!(config.retry.max_attempts, 3);
        assert_eq!(config.health.interval, Duration::from_secs(1));
        assert_eq!(config.replicas[0].url.as_str(), "http://10.0.0.10:8000/");
    }

    #[test]
    fn durations_and_policies_parse_from_text() {
        let config = Config::from_yaml_str(&format!(
            "{MINIMAL}health:\n  interval: 250ms\n  timeout: 2s\nrouting:\n  default_policy: p2c\n"
        ))
        .unwrap();
        assert_eq!(config.health.interval, Duration::from_millis(250));
        assert_eq!(config.health.timeout, Duration::from_secs(2));
        assert_eq!(config.routing.default_policy, PolicyKind::P2c);
        assert_eq!(config.health.unhealthy_threshold, 3);
    }

    #[test]
    fn every_policy_name_parses() {
        for policy in PolicyKind::ALL {
            let parsed: PolicyKind = serde_yaml_ng::from_str(policy.as_str()).unwrap();
            assert_eq!(parsed, policy);
            assert_eq!(policy.to_string(), policy.as_str());
        }
        assert!(serde_yaml_ng::from_str::<PolicyKind>("least_loaded").is_err());
    }

    #[test]
    fn unknown_keys_are_rejected_everywhere() {
        for extra in [
            "typo_section: 1\n",
            "routing:\n  epsilonn: 0.3\n",
            "server:\n  listen_addr: 0.0.0.0:1\n",
        ] {
            let err = Config::from_yaml_str(&format!("{MINIMAL}{extra}")).unwrap_err();
            assert!(err.to_string().contains("unknown field"), "{err}");
        }
    }

    #[test]
    fn invalid_ids_fail_at_parse_time() {
        let text = MINIMAL.replace("id: r0", "id: \"r 0\"");
        assert!(Config::from_yaml_str(&text).is_err());
    }

    #[test]
    fn lookups_resolve_names_aliases_and_policies() {
        let text = MINIMAL.replace(
            "  - name: meta-llama/Llama-3.1-8B-Instruct\n",
            "  - name: meta-llama/Llama-3.1-8B-Instruct\n    aliases: [llama]\n    policy: round_robin\n",
        );
        let config = Config::from_yaml_str(&text).unwrap();
        let model = config.model("llama").unwrap();
        assert_eq!(config.policy_for(model), PolicyKind::RoundRobin);
        assert_eq!(config.replicas_serving(&model.name).count(), 1);
        assert!(config.model("other").is_none());
    }

    #[test]
    fn yaml_round_trip_is_lossless() {
        let config = Config::from_yaml_str(MINIMAL).unwrap();
        let text = config.to_yaml_string().unwrap();
        assert_eq!(Config::from_yaml_str(&text).unwrap(), config);
    }

    #[test]
    fn load_reports_missing_files() {
        let err = Config::load("/definitely/not/here.yaml").unwrap_err();
        assert!(matches!(err, ConfigError::Io { .. }));
        assert!(err.to_string().contains("/definitely/not/here.yaml"));
    }

    #[test]
    fn cache_awareness_is_classified() {
        let aware: Vec<_> = PolicyKind::ALL
            .into_iter()
            .filter(|p| p.is_cache_aware())
            .collect();
        assert_eq!(
            aware,
            [
                PolicyKind::PrefixHash,
                PolicyKind::KvAware,
                PolicyKind::CostModel
            ]
        );
    }
}
