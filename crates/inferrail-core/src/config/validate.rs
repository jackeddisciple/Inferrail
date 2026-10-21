use std::collections::{HashMap, HashSet};
use std::fmt;
use std::time::Duration;

use super::Config;

pub const MAX_BLOCK_SIZE: u32 = 4096;
pub const MAX_RETRY_ATTEMPTS: u32 = 10;
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

impl Config {
    /// Checks cross field rules that the schema alone cannot express.
    pub fn validate(&self) -> Result<(), ConfigIssues> {
        let mut c = Collector::default();
        self.check_server(&mut c);
        self.check_models(&mut c);
        self.check_replicas(&mut c);
        self.check_routing(&mut c);
        self.check_health(&mut c);
        self.check_retry(&mut c);
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
            if self.replicas_serving(&model.name).next().is_none() {
                c.push(format!("{path}.name"), format!("no replica serves {own:?}"));
            }
        }
    }

    fn check_replicas(&self, c: &mut Collector) {
        if self.replicas.is_empty() {
            c.push("replicas", "at least one replica is required");
        }
        let known: HashSet<&str> = self.models.iter().map(|m| m.name.as_str()).collect();
        let mut ids: HashMap<&str, usize> = HashMap::new();
        let mut urls: HashMap<&str, usize> = HashMap::new();
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
            let url = &replica.url;
            if !matches!(url.scheme(), "http" | "https") {
                c.push(format!("{path}.url"), "scheme must be http or https");
            }
            if url.host().is_none() {
                c.push(format!("{path}.url"), "must include a host");
            }
            if url.query().is_some() || url.fragment().is_some() {
                c.push(format!("{path}.url"), "must not carry a query or fragment");
            }
            if !url.username().is_empty() || url.password().is_some() {
                c.push(
                    format!("{path}.url"),
                    "must not embed credentials; they would leak into logs and metrics",
                );
            }
            if let Some(first) = urls.insert(url.as_str(), i) {
                c.push(
                    format!("{path}.url"),
                    format!("duplicate url, first used by replicas[{first}]"),
                );
            }
            if replica.models.is_empty() {
                c.push(format!("{path}.models"), "must list at least one model");
            }
            for (j, model) in replica.models.iter().enumerate() {
                if replica.models[..j].contains(model) {
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
            if replica.kv_capacity_blocks == Some(0) {
                c.push(
                    format!("{path}.kv_capacity_blocks"),
                    "must be greater than zero",
                );
            }
            if replica.max_concurrency == Some(0) {
                c.push(
                    format!("{path}.max_concurrency"),
                    "must be greater than zero",
                );
            }
        }
    }

    fn check_routing(&self, c: &mut Collector) {
        let routing = &self.routing;
        if !(routing.epsilon.is_finite() && routing.epsilon > 0.0 && routing.epsilon <= 10.0) {
            c.push("routing.epsilon", "must be in (0, 10]");
        }
        if !(routing.kappa.is_finite() && routing.kappa > 0.0 && routing.kappa <= 1.0) {
            c.push("routing.kappa", "must be in (0, 1]");
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

    use super::super::{Config, ConfigError, ReplicaConfig};
    use super::MAX_BLOCK_SIZE;
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
        config.replicas[0].models.clear();
        let found = issues(&config);
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
            epsilon in 0.01f64..10.0,
            kappa in 0.01f64..=1.0,
            block_shift in 0u32..8,
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
            text.push_str(&format!("routing:\n  epsilon: {epsilon}\n  kappa: {kappa}\n"));
            let config = Config::from_yaml_str(&text).unwrap();
            let again = Config::from_yaml_str(&config.to_yaml_string().unwrap()).unwrap();
            prop_assert_eq!(again, config);
        }
    }
}
