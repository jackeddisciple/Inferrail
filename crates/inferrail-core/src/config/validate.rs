use std::collections::{HashMap, HashSet};
use std::fmt;
use std::time::Duration;

use url::Url;

use super::{Config, ContentFormat, CostModelConfig, ModelConfig, SIM_TOKENIZER};
use crate::ids::ModelId;

pub const MAX_BLOCK_SIZE: u32 = 4096;
pub const MAX_TOKENIZER_WORKERS: usize = 256;
pub const MAX_TOKENIZER_QUEUE: usize = 1 << 20;
pub const MAX_CAPACITY_BLOCKS: u64 = 1 << 32;
pub const MAX_RETRY_ATTEMPTS: u32 = 10;
pub const MIN_SCRAPE_INTERVAL: Duration = Duration::from_millis(10);
pub const MAX_PREFIX_BLOCKS: u32 = 4096;
const MAX_WEIGHT: f64 = 1000.0;
const MAX_COST_SECS: f64 = 60.0;
const MIN_HOT_WINDOW: Duration = Duration::from_millis(100);
const MAX_HOT_WINDOW: Duration = Duration::from_secs(3600);
const LOG_LEVELS: [&str; 5] = ["trace", "debug", "info", "warn", "error"];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConfigIssue {
    pub path: String,
    pub message: String,
}

impl fmt::Display for ConfigIssue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.path, self.message)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConfigIssues(pub Vec<ConfigIssue>);

impl fmt::Display for ConfigIssues {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let noun = if self.0.len() == 1 {
            "problem"
        } else {
            "problems"
        };
        write!(f, "{} {noun}", self.0.len())?;
        for issue in &self.0 {
            write!(f, "\n  {issue}")?;
        }
        Ok(())
    }
}

impl ConfigIssues {
    pub fn paths(&self) -> impl Iterator<Item = &str> {
        self.0.iter().map(|i| i.path.as_str())
    }
}

#[derive(Default)]
struct Collector(Vec<ConfigIssue>);

impl Collector {
    fn push(&mut self, path: impl Into<String>, message: impl Into<String>) {
        self.0.push(ConfigIssue {
            path: path.into(),
            message: message.into(),
        });
    }

    fn positive(&mut self, path: &str, value: Duration) {
        if value.is_zero() {
            self.push(path, "must be greater than zero");
        }
    }
}

fn claim_name<'a>(
    names: &mut HashMap<&'a str, String>,
    name: &'a str,
    at: String,
    owner: &str,
    c: &mut Collector,
) {
    match names.get(name) {
        Some(first) => c.push(at, format!("name {name:?} is already used by {first}")),
        None => {
            names.insert(name, owner.to_owned());
        }
    }
}

fn check_url<'a>(path: &str, url: &'a Url, seen: &mut HashMap<&'a str, String>, c: &mut Collector) {
    let at = format!("{path}.url");
    if !matches!(url.scheme(), "http" | "https") {
        c.push(&*at, "scheme must be http or https");
    }
    if url.host().is_none() {
        c.push(&*at, "must include a host");
    }
    if url.query().is_some() || url.fragment().is_some() {
        c.push(&*at, "must not carry a query or fragment");
    }
    if !url.username().is_empty() || url.password().is_some() {
        c.push(
            &*at,
            "must not embed credentials; they would leak into logs and metrics",
        );
    }
    if let Some(first) = seen.insert(url.as_str(), path.to_owned()) {
        c.push(at, format!("duplicate url, first used by {first}"));
    }
}

fn check_models(path: &str, models: &[ModelId], known: &HashSet<&str>, c: &mut Collector) {
    if models.is_empty() {
        c.push(format!("{path}.models"), "must list at least one model");
    }
    for (j, model) in models.iter().enumerate() {
        if models[..j].contains(model) {
            c.push(
                format!("{path}.models[{j}]"),
                format!("lists {:?} twice", model.as_str()),
            );
        }
        if !known.contains(model.as_str()) {
            c.push(
                format!("{path}.models[{j}]"),
                format!("unknown model {:?}", model.as_str()),
            );
        }
    }
}

fn check_tokenization(path: &str, model: &ModelConfig, c: &mut Collector) {
    let overrides = [
        ("chat_template", model.chat_template.is_some()),
        (
            "chat_template_kwargs",
            !model.chat_template_kwargs.is_empty(),
        ),
        (
            "chat_template_content_format",
            model.chat_template_content_format != ContentFormat::Auto,
        ),
    ];
    match model.tokenizer.as_deref() {
        None => {
            for (field, set) in overrides {
                if set {
                    c.push(format!("{path}.{field}"), "needs a tokenizer");
                }
            }
        }
        Some(spec) if spec.trim().is_empty() || spec.trim() != spec => c.push(
            format!("{path}.tokenizer"),
            "must be a model directory, a Hugging Face repository id or builtin:sim",
        ),
        Some(SIM_TOKENIZER) => {
            for (field, set) in overrides {
                if set {
                    c.push(
                        format!("{path}.{field}"),
                        "does not apply to the simulator's own tokenizer",
                    );
                }
            }
        }
        Some(_) => {}
    }
    if model
        .chat_template
        .as_ref()
        .is_some_and(|p| p.as_os_str().is_empty())
    {
        c.push(format!("{path}.chat_template"), "must not be empty");
    }
    for key in model.chat_template_kwargs.keys() {
        if matches!(key.as_str(), "chat_template" | "tokenize") {
            c.push(
                format!("{path}.chat_template_kwargs.{key}"),
                "is not allowed; vLLM refuses it too",
            );
        }
    }
}

/// Every cost is a time, so finite and never negative; prefill must cost
/// something per token or cached tokens would be worth nothing.
fn check_cost(path: &str, cost: &CostModelConfig, c: &mut Collector) {
    for (name, value) in [
        ("prefill_token_secs", cost.prefill_token_secs),
        ("prefill_pair_secs", cost.prefill_pair_secs),
        ("prefill_fixed_secs", cost.prefill_fixed_secs),
        ("decode_interference_secs", cost.decode_interference_secs),
    ] {
        if !(value.is_finite() && (0.0..=MAX_COST_SECS).contains(&value)) {
            c.push(
                format!("{path}.{name}"),
                format!("must be between 0 and {MAX_COST_SECS} seconds"),
            );
        }
    }
    if cost.prefill_token_secs == 0.0 {
        c.push(
            format!("{path}.prefill_token_secs"),
            "must be greater than zero",
        );
    }
}

fn check_limits(
    path: &str,
    kv_capacity_blocks: Option<u64>,
    max_concurrency: Option<u32>,
    max_pending: Option<u32>,
    c: &mut Collector,
) {
    for (name, value) in [
        ("kv_capacity_blocks", kv_capacity_blocks.map(|v| v == 0)),
        ("max_concurrency", max_concurrency.map(|v| v == 0)),
        ("max_pending", max_pending.map(|v| v == 0)),
    ] {
        if value == Some(true) {
            c.push(format!("{path}.{name}"), "must be greater than zero");
        }
    }
}

impl Config {
    /// Checks cross field rules that the schema alone cannot express.
    pub fn validate(&self) -> Result<(), ConfigIssues> {
        let mut c = Collector::default();
        self.check_server(&mut c);
        self.check_models(&mut c);
        self.check_replicas(&mut c);
        self.check_discovery(&mut c);
        self.check_upstream(&mut c);
        self.check_routing(&mut c);
        self.check_health(&mut c);
        self.check_retry(&mut c);
        self.check_load(&mut c);
        self.check_observability(&mut c);
        self.check_tokenizer(&mut c);
        if self.index.default_capacity_blocks == 0
            || self.index.default_capacity_blocks > MAX_CAPACITY_BLOCKS
        {
            c.push(
                "index.default_capacity_blocks",
                format!("must be between 1 and {MAX_CAPACITY_BLOCKS}"),
            );
        }
        if !LOG_LEVELS.contains(&self.logging.level.as_str()) {
            c.push(
                "logging.level",
                format!("must be one of {}", LOG_LEVELS.join(", ")),
            );
        }
        if c.0.is_empty() {
            Ok(())
        } else {
            Err(ConfigIssues(c.0))
        }
    }

    fn check_server(&self, c: &mut Collector) {
        let server = &self.server;
        if server.listen == server.admin_listen {
            c.push("server.admin_listen", "must differ from server.listen");
        }
        c.positive("server.request_timeout", server.request_timeout);
        if server.max_body_bytes == 0 {
            c.push("server.max_body_bytes", "must be greater than zero");
        }
        c.positive("server.header_read_timeout", server.header_read_timeout);
    }

    fn check_upstream(&self, c: &mut Collector) {
        let upstream = &self.upstream;
        c.positive("upstream.connect_timeout", upstream.connect_timeout);
        c.positive("upstream.pool_idle_timeout", upstream.pool_idle_timeout);
        if upstream.connect_timeout > self.server.request_timeout {
            c.push(
                "upstream.connect_timeout",
                "must not exceed server.request_timeout",
            );
        }
    }

    fn check_models(&self, c: &mut Collector) {
        if self.models.is_empty() {
            c.push("models", "at least one model is required");
        }
        let mut names: HashMap<&str, String> = HashMap::new();
        for (i, model) in self.models.iter().enumerate() {
            let path = format!("models[{i}]");
            let own = model.name.as_str();
            claim_name(&mut names, own, format!("{path}.name"), &path, c);
            for (j, alias) in model.aliases.iter().enumerate() {
                let at = format!("{path}.aliases[{j}]");
                if alias.is_empty() || alias.chars().any(char::is_whitespace) {
                    c.push(at, "must be non empty without whitespace");
                } else if alias == own {
                    c.push(at, "repeats the model name");
                } else {
                    claim_name(&mut names, alias, at, &path, c);
                }
            }
            if model.block_size == 0 || model.block_size > MAX_BLOCK_SIZE {
                c.push(
                    format!("{path}.block_size"),
                    format!("must be between 1 and {MAX_BLOCK_SIZE}"),
                );
            }
            if !self.is_served(&model.name) {
                c.push(format!("{path}.name"), format!("no replica serves {own:?}"));
            }
            check_tokenization(&path, model, c);
        }
    }

    fn check_tokenizer(&self, c: &mut Collector) {
        let tokenizer = &self.tokenizer;
        if tokenizer.workers > MAX_TOKENIZER_WORKERS {
            c.push(
                "tokenizer.workers",
                format!("must be at most {MAX_TOKENIZER_WORKERS}"),
            );
        }
        if tokenizer.queue == 0 || tokenizer.queue > MAX_TOKENIZER_QUEUE {
            c.push(
                "tokenizer.queue",
                format!("must be between 1 and {MAX_TOKENIZER_QUEUE}"),
            );
        }
        c.positive("tokenizer.max_wait", tokenizer.max_wait);
        if tokenizer.max_wait > self.server.request_timeout {
            c.push(
                "tokenizer.max_wait",
                "must not exceed server.request_timeout",
            );
        }
    }

    fn check_replicas(&self, c: &mut Collector) {
        if self.replicas.is_empty() && self.discovery.dns.is_empty() {
            c.push("replicas", "at least one replica or DNS group is required");
        }
        let known = self.known_models();
        let mut urls: HashMap<&str, String> = HashMap::new();
        let mut ids: HashMap<&str, usize> = HashMap::new();
        for (i, replica) in self.replicas.iter().enumerate() {
            let path = format!("replicas[{i}]");
            if let Some(first) = ids.insert(replica.id.as_str(), i) {
                c.push(
                    format!("{path}.id"),
                    format!(
                        "duplicate id {:?}, first used by replicas[{first}]",
                        replica.id.as_str()
                    ),
                );
            }
            check_url(&path, &replica.url, &mut urls, c);
            check_models(&path, &replica.models, &known, c);
            check_limits(
                &path,
                replica.kv_capacity_blocks,
                replica.max_concurrency,
                replica.max_pending,
                c,
            );
        }
    }

    fn check_discovery(&self, c: &mut Collector) {
        let known = self.known_models();
        let mut urls: HashMap<&str, String> = self
            .replicas
            .iter()
            .enumerate()
            .map(|(i, r)| (r.url.as_str(), format!("replicas[{i}]")))
            .collect();
        let mut names: HashMap<&str, usize> = HashMap::new();
        for (i, group) in self.discovery.dns.iter().enumerate() {
            let path = format!("discovery.dns[{i}]");
            let name = group.name.as_str();
            if let Some(first) = names.insert(name, i) {
                c.push(
                    format!("{path}.name"),
                    format!("duplicate name {name:?}, first used by discovery.dns[{first}]"),
                );
            }
            if self.replicas.iter().any(|r| r.id.as_str() == name) {
                c.push(
                    format!("{path}.name"),
                    format!("name {name:?} is already a replica id"),
                );
            }
            check_url(&path, &group.url, &mut urls, c);
            check_models(&path, &group.models, &known, c);
            c.positive(&format!("{path}.refresh"), group.refresh);
            check_limits(
                &path,
                group.kv_capacity_blocks,
                group.max_concurrency,
                group.max_pending,
                c,
            );
        }
    }

    fn known_models(&self) -> HashSet<&str> {
        self.models.iter().map(|m| m.name.as_str()).collect()
    }

    fn check_routing(&self, c: &mut Collector) {
        let routing = &self.routing;
        if !(routing.epsilon.is_finite() && routing.epsilon > 0.0 && routing.epsilon <= 10.0) {
            c.push("routing.epsilon", "must be in (0, 10]");
        }
        if !(routing.kappa.is_finite() && routing.kappa > 0.0 && routing.kappa <= 1.0) {
            c.push("routing.kappa", "must be in (0, 1]");
        }
        for (path, blocks) in [
            ("routing.prefix_hash.blocks", routing.prefix_hash.blocks),
            ("routing.hot_prefix.blocks", routing.hot_prefix.blocks),
        ] {
            if blocks == 0 || blocks > MAX_PREFIX_BLOCKS {
                c.push(path, format!("must be between 1 and {MAX_PREFIX_BLOCKS}"));
            }
        }
        let weights = &routing.kv_aware;
        for (path, weight) in [
            ("routing.kv_aware.cache_weight", weights.cache_weight),
            ("routing.kv_aware.load_weight", weights.load_weight),
        ] {
            if !(weight.is_finite() && (0.0..=MAX_WEIGHT).contains(&weight)) {
                c.push(path, format!("must be between 0 and {MAX_WEIGHT}"));
            }
        }
        if weights.cache_weight == 0.0 && weights.load_weight == 0.0 {
            c.push("routing.kv_aware", "needs a weight above zero");
        }
        check_cost("routing.cost_model", &routing.cost_model, c);
        let hot = &routing.hot_prefix;
        if !(hot.utilization.is_finite() && hot.utilization > 0.0 && hot.utilization <= 1.0) {
            c.push("routing.hot_prefix.utilization", "must be in (0, 1]");
        }
        if !(MIN_HOT_WINDOW..=MAX_HOT_WINDOW).contains(&hot.window) {
            c.push("routing.hot_prefix.window", "must be between 100ms and 1h");
        }
        for (i, model) in self.models.iter().enumerate() {
            if let Some(cost) = &model.cost_model {
                check_cost(&format!("models[{i}].cost_model"), cost, c);
            }
        }
    }

    fn check_health(&self, c: &mut Collector) {
        let health = &self.health;
        c.positive("health.interval", health.interval);
        c.positive("health.timeout", health.timeout);
        c.positive("health.ejection_backoff", health.ejection_backoff);
        if health.timeout > health.interval {
            c.push("health.timeout", "must not exceed health.interval");
        }
        if health.max_ejection_backoff < health.ejection_backoff {
            c.push(
                "health.max_ejection_backoff",
                "must be at least health.ejection_backoff",
            );
        }
        if health.unhealthy_threshold == 0 {
            c.push("health.unhealthy_threshold", "must be at least 1");
        }
        if health.healthy_threshold == 0 {
            c.push("health.healthy_threshold", "must be at least 1");
        }
        if health.max_ejection_percent > 100 {
            c.push("health.max_ejection_percent", "must be at most 100");
        }
    }

    fn check_load(&self, c: &mut Collector) {
        let load = &self.load;
        if load.scrape_interval < MIN_SCRAPE_INTERVAL {
            c.push("load.scrape_interval", "must be at least 10ms");
        }
        c.positive("load.scrape_timeout", load.scrape_timeout);
        if load.scrape_timeout > load.scrape_interval {
            c.push(
                "load.scrape_timeout",
                "must not exceed load.scrape_interval",
            );
        }
        if load.max_age < load.scrape_interval.saturating_add(load.scrape_timeout) {
            c.push(
                "load.max_age",
                "must be at least load.scrape_interval plus load.scrape_timeout, or one slow scrape makes a healthy replica look stale",
            );
        }
    }

    fn check_observability(&self, c: &mut Collector) {
        let otel = &self.observability;
        if let Some(url) = &otel.otlp_endpoint
            && (!matches!(url.scheme(), "http" | "https") || url.host().is_none())
        {
            c.push(
                "observability.otlp_endpoint",
                "must be an http or https URL with a host",
            );
        }
        if otel.service_name.trim().is_empty() {
            c.push("observability.service_name", "must not be empty");
        }
        if !(0.0..=1.0).contains(&otel.trace_sample_ratio) {
            c.push("observability.trace_sample_ratio", "must be in [0, 1]");
        }
        let log = &otel.decision_log;
        if !(0.0..=1.0).contains(&log.sample_rate) {
            c.push(
                "observability.decision_log.sample_rate",
                "must be in [0, 1]",
            );
        } else if log.sample_rate > 0.0 && log.path.is_none() {
            c.push(
                "observability.decision_log.path",
                "is required when sample_rate is above zero",
            );
        }
    }

    fn check_retry(&self, c: &mut Collector) {
        let attempts = self.retry.max_attempts;
        if attempts == 0 || attempts > MAX_RETRY_ATTEMPTS {
            c.push(
                "retry.max_attempts",
                format!("must be between 1 and {MAX_RETRY_ATTEMPTS}"),
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::super::{
        Config, ConfigError, ContentFormat, DnsGroupConfig, ReplicaConfig, SIM_TOKENIZER,
    };
    use super::{
        MAX_BLOCK_SIZE, MAX_CAPACITY_BLOCKS, MAX_PREFIX_BLOCKS, MAX_TOKENIZER_QUEUE,
        MAX_TOKENIZER_WORKERS,
    };
    use crate::ids::{ModelId, ReplicaId};
    use proptest::prelude::*;

    const BASE: &str = r#"
models:
  - name: llama-8b
    aliases: [llama]
  - name: qwen-8b
replicas:
  - id: r0
    url: http://10.0.0.10:8000
    models: [llama-8b, qwen-8b]
  - id: r1
    url: http://10.0.0.11:8000
    models: [llama-8b]
"#;

    fn base() -> Config {
        Config::from_yaml_str(BASE).unwrap()
    }

    fn issues(config: &Config) -> Vec<String> {
        config
            .validate()
            .err()
            .map(|i| i.0.iter().map(ToString::to_string).collect())
            .unwrap_or_default()
    }

    fn only_issue(config: &Config) -> String {
        let found = issues(config);
        assert_eq!(found.len(), 1, "{found:?}");
        found.into_iter().next().unwrap()
    }

    fn replica(id: &str, url: &str, models: &[&str]) -> ReplicaConfig {
        ReplicaConfig {
            id: ReplicaId::new(id).unwrap(),
            url: url.parse().unwrap(),
            models: models.iter().map(|m| ModelId::new(m).unwrap()).collect(),
            kv_capacity_blocks: None,
            max_concurrency: None,
            max_pending: None,
        }
    }

    fn group(name: &str, url: &str, models: &[&str]) -> DnsGroupConfig {
        DnsGroupConfig {
            name: ReplicaId::new(name).unwrap(),
            url: url.parse().unwrap(),
            models: models.iter().map(|m| ModelId::new(m).unwrap()).collect(),
            refresh: Duration::from_secs(2),
            kv_capacity_blocks: None,
            max_concurrency: None,
            max_pending: None,
        }
    }

    #[test]
    fn base_config_is_valid() {
        assert!(base().validate().is_ok());
    }

    #[test]
    fn empty_fleet_is_rejected() {
        let text = "models: []\nreplicas: []\n";
        let Err(ConfigError::Invalid(found)) = Config::from_yaml_str(text) else {
            panic!("expected validation failure");
        };
        assert_eq!(found.paths().collect::<Vec<_>>(), ["models", "replicas"]);
    }

    #[test]
    fn dns_groups_alone_can_serve_every_model() {
        let text = r#"
models:
  - name: llama-8b
discovery:
  dns:
    - name: pool
      url: http://vllm.default.svc:8000
      models: [llama-8b]
"#;
        let config = Config::from_yaml_str(text).unwrap();
        assert!(config.replicas.is_empty());
        assert_eq!(config.discovery.dns[0].refresh, Duration::from_secs(2));
        assert!(config.is_served(&ModelId::new("llama-8b").unwrap()));
    }

    #[test]
    fn a_model_served_only_by_dns_is_not_flagged() {
        let mut config = base();
        config.replicas[0]
            .models
            .retain(|m| m.as_str() != "qwen-8b");
        config
            .discovery
            .dns
            .push(group("qwen-pool", "http://qwen.svc:8000", &["qwen-8b"]));
        assert!(config.validate().is_ok(), "{:?}", issues(&config));
    }

    #[test]
    fn dns_groups_are_validated_like_replicas() {
        let mut config = base();
        let mut bad = group(
            "r0",
            "http://user:pw@pool.svc:8000/?x=1",
            &["ghost", "ghost"],
        );
        bad.refresh = Duration::ZERO;
        bad.max_pending = Some(0);
        bad.max_concurrency = Some(0);
        config.discovery.dns.push(bad);
        config
            .discovery
            .dns
            .push(group("r0", "http://10.0.0.10:8000", &[]));
        let found = issues(&config);
        for needle in [
            "discovery.dns[0].name: name \"r0\" is already a replica id",
            "discovery.dns[0].url: must not carry a query or fragment",
            "discovery.dns[0].url: must not embed credentials",
            "discovery.dns[0].models[0]: unknown model \"ghost\"",
            "discovery.dns[0].models[1]: lists \"ghost\" twice",
            "discovery.dns[0].refresh: must be greater than zero",
            "discovery.dns[0].max_concurrency: must be greater than zero",
            "discovery.dns[0].max_pending: must be greater than zero",
            "discovery.dns[1].name: duplicate name \"r0\", first used by discovery.dns[0]",
            "discovery.dns[1].url: duplicate url, first used by replicas[0]",
            "discovery.dns[1].models: must list at least one model",
        ] {
            assert!(
                found.iter().any(|i| i.starts_with(needle)),
                "missing {needle}: {found:?}"
            );
        }
    }

    #[test]
    fn empty_fleet_message_mentions_dns() {
        let text = "models: [{name: m}]\n";
        let err = Config::from_yaml_str(text).unwrap_err().to_string();
        assert!(
            err.contains("replicas: at least one replica or DNS group is required"),
            "{err}"
        );
    }

    #[test]
    fn upstream_and_server_timeouts_are_checked() {
        let mut config = base();
        config.upstream.connect_timeout = Duration::ZERO;
        config.upstream.pool_idle_timeout = Duration::ZERO;
        config.server.header_read_timeout = Duration::ZERO;
        let found = issues(&config);
        assert_eq!(found.len(), 3, "{found:?}");

        let mut config = base();
        config.server.request_timeout = Duration::from_secs(1);
        config.upstream.connect_timeout = Duration::from_secs(2);
        assert_eq!(
            only_issue(&config),
            "upstream.connect_timeout: must not exceed server.request_timeout"
        );

        let mut config = base();
        config.server.shutdown_timeout = Duration::ZERO;
        config.upstream.pool_max_idle_per_host = 0;
        assert!(config.validate().is_ok());
    }

    #[test]
    fn duplicate_names_and_aliases_are_rejected() {
        let mut config = base();
        config.models[1].aliases = vec!["llama".into()];
        assert!(
            only_issue(&config).starts_with("models[1].aliases[0]: name \"llama\" is already used")
        );

        let mut config = base();
        config.models[1].aliases = vec!["llama-8b".into()];
        assert!(only_issue(&config).contains("already used by models[0]"));
    }

    #[test]
    fn alias_repeating_its_own_name_says_so() {
        let mut config = base();
        config.models[0].aliases = vec!["llama-8b".into()];
        assert_eq!(
            only_issue(&config),
            "models[0].aliases[0]: repeats the model name"
        );
    }

    #[test]
    fn credentials_in_replica_urls_are_rejected() {
        for url in [
            "http://user:secret@10.0.0.9:8000",
            "http://token@10.0.0.9:8000",
        ] {
            let mut config = base();
            config.replicas.push(replica("r9", url, &["llama-8b"]));
            assert!(
                only_issue(&config).contains("must not embed credentials"),
                "{url}"
            );
        }
    }

    #[test]
    fn replica_listing_a_model_twice_is_rejected() {
        let mut config = base();
        config.replicas[1]
            .models
            .push(ModelId::new("llama-8b").unwrap());
        assert_eq!(
            only_issue(&config),
            "replicas[1].models[1]: lists \"llama-8b\" twice"
        );
    }

    #[test]
    fn bad_aliases_are_rejected() {
        let mut config = base();
        config.models[0].aliases = vec!["has space".into()];
        assert!(only_issue(&config).starts_with("models[0].aliases[0]: must be non empty"));
    }

    #[test]
    fn block_size_bounds() {
        for size in [0, MAX_BLOCK_SIZE + 1] {
            let mut config = base();
            config.models[0].block_size = size;
            assert!(only_issue(&config).starts_with("models[0].block_size"));
        }
        let mut config = base();
        config.models[0].block_size = 1;
        assert!(config.validate().is_ok());
    }

    #[test]
    fn unserved_and_unknown_models_are_rejected() {
        let mut config = base();
        config.replicas[0]
            .models
            .retain(|m| m.as_str() != "qwen-8b");
        assert_eq!(
            only_issue(&config),
            "models[1].name: no replica serves \"qwen-8b\""
        );

        let mut config = base();
        config.replicas[1]
            .models
            .push(ModelId::new("mistral").unwrap());
        assert_eq!(
            only_issue(&config),
            "replicas[1].models[1]: unknown model \"mistral\""
        );
    }

    #[test]
    fn replica_ids_and_urls_must_be_unique() {
        let mut config = base();
        config
            .replicas
            .push(replica("r0", "http://10.0.0.12:8000", &["llama-8b"]));
        assert!(only_issue(&config).contains("duplicate id \"r0\", first used by replicas[0]"));

        let mut config = base();
        config
            .replicas
            .push(replica("r2", "http://10.0.0.10:8000", &["llama-8b"]));
        assert!(only_issue(&config).starts_with("replicas[2].url: duplicate url"));
    }

    #[test]
    fn replica_urls_must_be_plain_http_endpoints() {
        for url in [
            "ftp://10.0.0.1/",
            "http://h:1/?x=1",
            "https://h/#frag",
            "unix:/tmp/sock",
        ] {
            let mut config = base();
            config.replicas.push(replica("rx", url, &["llama-8b"]));
            let found = issues(&config);
            assert!(!found.is_empty(), "{url} accepted");
            assert!(
                found.iter().all(|i| i.starts_with("replicas[2].url")),
                "{found:?}"
            );
        }
    }

    #[test]
    fn replica_limits_must_be_positive() {
        let mut config = base();
        config.replicas[0].kv_capacity_blocks = Some(0);
        config.replicas[0].max_concurrency = Some(0);
        config.replicas[0].max_pending = Some(0);
        config.replicas[0].models.clear();
        let found = issues(&config);
        assert!(
            found
                .iter()
                .any(|i| i.starts_with("replicas[0].max_pending"))
        );
        assert!(
            found
                .iter()
                .any(|i| i.starts_with("replicas[0].kv_capacity_blocks"))
        );
        assert!(
            found
                .iter()
                .any(|i| i.starts_with("replicas[0].max_concurrency"))
        );
        assert!(found.iter().any(|i| i.starts_with("replicas[0].models")));
    }

    #[test]
    fn routing_knobs_are_bounded() {
        for epsilon in [0.0, -1.0, 10.5, f64::NAN, f64::INFINITY] {
            let mut config = base();
            config.routing.epsilon = epsilon;
            assert!(
                only_issue(&config).starts_with("routing.epsilon"),
                "{epsilon}"
            );
        }
        for kappa in [0.0, 1.01, f64::NAN] {
            let mut config = base();
            config.routing.kappa = kappa;
            assert!(only_issue(&config).starts_with("routing.kappa"), "{kappa}");
        }
        let mut config = base();
        config.routing.kappa = 1.0;
        config.routing.epsilon = 10.0;
        assert!(config.validate().is_ok());
    }

    #[test]
    fn policy_settings_are_bounded() {
        for blocks in [0, MAX_PREFIX_BLOCKS + 1] {
            let mut config = base();
            config.routing.prefix_hash.blocks = blocks;
            config.routing.hot_prefix.blocks = blocks;
            let found = issues(&config);
            assert_eq!(found.len(), 2, "{found:?}");
            assert!(found[0].starts_with("routing.prefix_hash.blocks"));
            assert!(found[1].starts_with("routing.hot_prefix.blocks"));
        }
        for weight in [-0.1, f64::NAN, f64::INFINITY, 1000.5] {
            let mut config = base();
            config.routing.kv_aware.load_weight = weight;
            assert!(
                only_issue(&config).starts_with("routing.kv_aware.load_weight"),
                "{weight}"
            );
        }
        let mut config = base();
        config.routing.kv_aware.cache_weight = 0.0;
        config.routing.kv_aware.load_weight = 0.0;
        assert_eq!(
            only_issue(&config),
            "routing.kv_aware: needs a weight above zero"
        );
        for utilization in [0.0, 1.5, f64::NAN] {
            let mut config = base();
            config.routing.hot_prefix.utilization = utilization;
            assert!(
                only_issue(&config).starts_with("routing.hot_prefix.utilization"),
                "{utilization}"
            );
        }
        for window in [Duration::from_millis(99), Duration::from_secs(3601)] {
            let mut config = base();
            config.routing.hot_prefix.window = window;
            assert!(only_issue(&config).starts_with("routing.hot_prefix.window"));
        }
        let mut config = base();
        config.routing.prefix_hash.blocks = MAX_PREFIX_BLOCKS;
        config.routing.kv_aware.cache_weight = 0.0;
        config.routing.hot_prefix.utilization = 1.0;
        config.routing.hot_prefix.window = Duration::from_millis(100);
        assert!(config.validate().is_ok(), "{:?}", issues(&config));
    }

    #[test]
    fn cost_models_must_be_finite_times() {
        let mut config = base();
        config.routing.cost_model.prefill_pair_secs = -1e-9;
        config.routing.cost_model.decode_interference_secs = f64::NAN;
        config.routing.cost_model.prefill_fixed_secs = 61.0;
        let found = issues(&config);
        assert_eq!(found.len(), 3, "{found:?}");
        for field in [
            "prefill_pair_secs",
            "decode_interference_secs",
            "prefill_fixed_secs",
        ] {
            assert!(
                found
                    .iter()
                    .any(|i| i.starts_with(&format!("routing.cost_model.{field}"))),
                "{field}: {found:?}"
            );
        }
        let mut config = base();
        config.models[1].cost_model = Some(super::super::CostModelConfig {
            prefill_token_secs: 0.0,
            ..Default::default()
        });
        assert_eq!(
            only_issue(&config),
            "models[1].cost_model.prefill_token_secs: must be greater than zero"
        );
        config.models[1].cost_model = Some(super::super::CostModelConfig {
            prefill_token_secs: 1e-5,
            prefill_pair_secs: 0.0,
            prefill_fixed_secs: 0.0,
            decode_interference_secs: 0.0,
        });
        assert!(config.validate().is_ok());
    }

    #[test]
    fn health_timings_are_consistent() {
        let mut config = base();
        config.health.timeout = Duration::from_secs(5);
        assert_eq!(
            only_issue(&config),
            "health.timeout: must not exceed health.interval"
        );

        let mut config = base();
        config.health.interval = Duration::ZERO;
        config.health.timeout = Duration::ZERO;
        assert_eq!(issues(&config).len(), 2);

        let mut config = base();
        config.health.max_ejection_backoff = Duration::from_secs(1);
        assert!(only_issue(&config).starts_with("health.max_ejection_backoff"));

        let mut config = base();
        config.health.unhealthy_threshold = 0;
        config.health.healthy_threshold = 0;
        assert_eq!(issues(&config).len(), 2);

        let mut config = base();
        config.health.max_ejection_percent = 101;
        assert_eq!(
            only_issue(&config),
            "health.max_ejection_percent: must be at most 100"
        );
        for percent in [0, 100] {
            config.health.max_ejection_percent = percent;
            assert!(config.validate().is_ok());
        }
    }

    #[test]
    fn server_and_retry_and_logging_checks() {
        let mut config = base();
        config.server.admin_listen = config.server.listen;
        config.server.request_timeout = Duration::ZERO;
        config.server.max_body_bytes = 0;
        config.retry.max_attempts = 0;
        config.logging.level = "verbose".into();
        let found = issues(&config);
        for path in [
            "server.admin_listen",
            "server.request_timeout",
            "server.max_body_bytes",
            "retry.max_attempts",
            "logging.level",
        ] {
            assert!(
                found.iter().any(|i| i.starts_with(path)),
                "missing {path}: {found:?}"
            );
        }
    }

    #[test]
    fn load_timings_are_consistent() {
        let mut config = base();
        config.load.scrape_interval = Duration::from_millis(5);
        config.load.scrape_timeout = Duration::from_millis(5);
        config.load.max_age = Duration::from_secs(1);
        assert_eq!(
            only_issue(&config),
            "load.scrape_interval: must be at least 10ms"
        );

        let mut config = base();
        config.load.scrape_timeout = Duration::from_millis(150);
        config.load.max_age = Duration::from_secs(1);
        assert_eq!(
            only_issue(&config),
            "load.scrape_timeout: must not exceed load.scrape_interval"
        );

        let mut config = base();
        config.load.scrape_timeout = Duration::ZERO;
        assert!(only_issue(&config).starts_with("load.scrape_timeout"));

        let mut config = base();
        config.load.max_age = Duration::from_millis(199);
        assert!(only_issue(&config).starts_with("load.max_age"));
        config.load.max_age = Duration::from_millis(200);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn observability_settings_are_checked() {
        let mut config = base();
        config.observability.otlp_endpoint = Some("ftp://collector:4318".parse().unwrap());
        config.observability.service_name = " ".into();
        config.observability.trace_sample_ratio = 1.5;
        config.observability.decision_log.sample_rate = f64::NAN;
        let found = issues(&config);
        assert_eq!(found.len(), 4, "{found:?}");
        for path in [
            "observability.otlp_endpoint",
            "observability.service_name",
            "observability.trace_sample_ratio",
            "observability.decision_log.sample_rate",
        ] {
            assert!(found.iter().any(|i| i.starts_with(path)), "{path}");
        }

        let mut config = base();
        config.observability.decision_log.sample_rate = 0.01;
        assert_eq!(
            only_issue(&config),
            "observability.decision_log.path: is required when sample_rate is above zero"
        );
        config.observability.decision_log.path = Some("/tmp/d.jsonl".into());
        config.observability.otlp_endpoint =
            Some("http://collector:4318/v1/traces".parse().unwrap());
        config.observability.trace_sample_ratio = 0.0;
        assert!(config.validate().is_ok());
    }

    #[test]
    fn tokenizer_settings_are_checked() {
        let mut config = base();
        config.models[0].chat_template = Some("/t.jinja".into());
        config.models[0]
            .chat_template_kwargs
            .insert("x".into(), serde_json::json!(1));
        let found = issues(&config);
        assert_eq!(found.len(), 2, "{found:?}");
        assert!(found.iter().all(|i| i.ends_with("needs a tokenizer")));
        config.models[0].tokenizer = Some("/models/llama".into());
        assert!(config.validate().is_ok());
        config.models[0].tokenizer = Some(SIM_TOKENIZER.into());
        assert_eq!(issues(&config).len(), 2);
        config.models[0].chat_template = None;
        config.models[0].chat_template_kwargs.clear();
        config.models[0].chat_template_content_format = ContentFormat::String;
        assert!(only_issue(&config).contains("simulator"));
        config.models[0].chat_template_content_format = ContentFormat::Auto;
        for bad in ["", " padded", "/x "] {
            config.models[0].tokenizer = Some(bad.into());
            assert!(
                only_issue(&config).starts_with("models[0].tokenizer"),
                "{bad:?}"
            );
        }
        config.models[0].tokenizer = Some("org/model".into());
        for key in ["tokenize", "chat_template"] {
            config.models[0].chat_template_kwargs.clear();
            config.models[0]
                .chat_template_kwargs
                .insert(key.into(), serde_json::json!(true));
            assert!(only_issue(&config).contains(&format!("chat_template_kwargs.{key}")));
        }
        config.models[0].chat_template_kwargs.clear();
        config.models[0].chat_template = Some("".into());
        assert!(only_issue(&config).contains("must not be empty"));
    }

    #[test]
    fn tokenizer_pool_and_index_limits() {
        let mut config = base();
        config.tokenizer.workers = MAX_TOKENIZER_WORKERS + 1;
        config.tokenizer.queue = 0;
        config.tokenizer.max_wait = Duration::ZERO;
        config.index.default_capacity_blocks = 0;
        let found = issues(&config);
        for path in [
            "tokenizer.workers",
            "tokenizer.queue",
            "tokenizer.max_wait",
            "index.default_capacity_blocks",
        ] {
            assert!(
                found.iter().any(|i| i.starts_with(path)),
                "{path}: {found:?}"
            );
        }
        config.tokenizer.workers = MAX_TOKENIZER_WORKERS;
        config.tokenizer.queue = MAX_TOKENIZER_QUEUE;
        config.tokenizer.max_wait = Duration::from_secs(1);
        config.index.default_capacity_blocks = MAX_CAPACITY_BLOCKS;
        assert!(config.validate().is_ok());
        config.tokenizer.max_wait = Duration::from_secs(3600);
        assert!(only_issue(&config).starts_with("tokenizer.max_wait"));
        config.tokenizer.max_wait = Duration::from_secs(1);
        config.index.default_capacity_blocks = MAX_CAPACITY_BLOCKS + 1;
        assert!(only_issue(&config).starts_with("index.default_capacity_blocks"));
    }

    #[test]
    fn all_problems_are_reported_together() {
        let text = r#"
models:
  - name: m
    block_size: 0
replicas:
  - id: a
    url: ftp://x
    models: [ghost]
routing:
  epsilon: 0
  kappa: 2
retry:
  max_attempts: 99
"#;
        let err = Config::from_yaml_str(text).unwrap_err();
        let message = err.to_string();
        assert!(
            message.starts_with("invalid config: 7 problems"),
            "{message}"
        );
        for needle in [
            "models[0].block_size",
            "no replica serves \"m\"",
            "replicas[0].url",
            "unknown model \"ghost\"",
            "routing.epsilon",
            "routing.kappa",
            "retry.max_attempts",
        ] {
            assert!(message.contains(needle), "missing {needle} in {message}");
        }
    }

    proptest! {
        #[test]
        fn arbitrary_yaml_never_panics(text in "(?s).{0,400}") {
            let _ = Config::from_yaml_str(&text);
        }

        #[test]
        fn random_valid_fleets_validate_and_round_trip(
            models in 1usize..5,
            replicas in 1usize..12,
            groups in 0usize..4,
            epsilon in 0.01f64..10.0,
            kappa in 0.01f64..=1.0,
            block_shift in 0u32..8,
            refresh_ms in 1u64..60_000,
            cap in proptest::option::of(1u32..10_000),
            scrape_ms in 10u64..2_000,
            sample in 0.0f64..=1.0,
        ) {
            let mut text = String::from("models:\n");
            for m in 0..models {
                text.push_str(&format!("  - name: model-{m}\n    block_size: {}\n", 1u32 << block_shift));
            }
            text.push_str("replicas:\n");
            for r in 0..replicas.max(models) {
                text.push_str(&format!(
                    "  - id: r{r}\n    url: http://10.1.{}.{}:8000\n    models: [model-{}]\n",
                    r / 250, r % 250, r % models
                ));
            }
            text.push_str("discovery:\n  dns:\n");
            for g in 0..groups {
                text.push_str(&format!(
                    "    - name: pool-{g}\n      url: http://pool-{g}.svc:8000\n      models: [model-{}]\n      refresh: {refresh_ms}ms\n",
                    g % models
                ));
                if let Some(cap) = cap {
                    text.push_str(&format!("      max_pending: {cap}\n"));
                }
            }
            text.push_str(&format!("routing:\n  epsilon: {epsilon}\n  kappa: {kappa}\n"));
            text.push_str(&format!(
                "load:\n  scrape_interval: {scrape_ms}ms\n  scrape_timeout: {scrape_ms}ms\n  max_age: {}ms\n",
                scrape_ms * 2
            ));
            text.push_str(&format!(
                "observability:\n  trace_sample_ratio: {sample}\n  decision_log:\n    path: /tmp/d.jsonl\n    sample_rate: {sample}\n"
            ));
            let config = Config::from_yaml_str(&text).unwrap();
            let again = Config::from_yaml_str(&config.to_yaml_string().unwrap()).unwrap();
            prop_assert_eq!(again, config);
        }
    }
}
