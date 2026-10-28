//! What a running gateway can take from a new config, and what needs a
//! restart: routing, each model's policy, cost model and aliases, the
//! static replicas, retries and prompt logging change in place; listeners,
//! clients, discovery, health checks, telemetry, tokenizers and the index
//! are built once at startup.

use super::{Config, ModelConfig};

impl Config {
    /// The sections of `next` that differ from `self` where only a restart
    /// would apply them, by config path; empty when `next` can be applied
    /// live.
    pub fn restart_changes(&self, next: &Config) -> Vec<String> {
        let mut out = Vec::new();
        let mut differs = |path: &str, changed: bool| {
            if changed {
                out.push(path.to_owned());
            }
        };
        differs("server", self.server != next.server);
        differs("upstream", self.upstream != next.upstream);
        differs("discovery", self.discovery != next.discovery);
        differs("health", self.health != next.health);
        differs("logging.level", self.logging.level != next.logging.level);
        differs("load", self.load != next.load);
        differs("observability", self.observability != next.observability);
        differs("tokenizer", self.tokenizer != next.tokenizer);
        differs("index", self.index != next.index);
        let names = |c: &Config| c.models.iter().map(|m| m.name.clone()).collect::<Vec<_>>();
        if names(self) != names(next) {
            out.push("models".to_owned());
            return out;
        }
        for (old, new) in self.models.iter().zip(&next.models) {
            if fixed(old) != fixed(new) {
                out.push(format!("models[{}]", old.name));
            }
        }
        out
    }

    /// The sections that differ between the two, by config path.
    pub fn changed_sections(&self, next: &Config) -> Vec<String> {
        let mut out = self.restart_changes(next);
        let mut differs = |path: &str, changed: bool| {
            if changed {
                out.push(path.to_owned());
            }
        };
        differs("routing", self.routing != next.routing);
        differs("replicas", self.replicas != next.replicas);
        differs("retry", self.retry != next.retry);
        differs(
            "logging.log_prompts",
            self.logging.log_prompts != next.logging.log_prompts,
        );
        if !out.iter().any(|p| p.starts_with("models")) {
            for (old, new) in self.models.iter().zip(&next.models) {
                if old != new {
                    out.push(format!("models[{}]", old.name));
                }
            }
        }
        out
    }
}

/// A model's settings with the live ones blanked out.
fn fixed(model: &ModelConfig) -> ModelConfig {
    ModelConfig {
        aliases: Vec::new(),
        policy: None,
        cost_model: None,
        ..model.clone()
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;
    use crate::config::{CostModelConfig, PolicyKind};

    fn base() -> Config {
        Config::from_yaml_str(
            "models:\n  - name: a\n  - name: b\nreplicas:\n  - {id: r0, url: 'http://h:1', models: [a, b]}\n",
        )
        .unwrap()
    }

    #[test]
    fn live_sections_need_no_restart() {
        let old = base();
        let mut new = old.clone();
        new.routing.epsilon = 0.5;
        new.routing.default_policy = PolicyKind::KvAware;
        new.models[0].policy = Some(PolicyKind::RoundRobin);
        new.models[1].aliases = vec!["bee".into()];
        new.models[1].cost_model = Some(CostModelConfig::default());
        new.replicas[0].max_concurrency = Some(4);
        new.retry.max_attempts = 5;
        new.logging.log_prompts = true;
        assert!(old.restart_changes(&new).is_empty());
        assert_eq!(
            old.changed_sections(&new),
            [
                "routing",
                "replicas",
                "retry",
                "logging.log_prompts",
                "models[a]",
                "models[b]"
            ]
        );
        assert!(old.changed_sections(&old).is_empty());
    }

    #[test]
    fn fixed_sections_are_named() {
        let old = base();
        let mut new = old.clone();
        new.server.request_timeout = Duration::from_secs(1);
        new.health.interval = Duration::from_secs(9);
        new.logging.level = "debug".into();
        new.index.enabled = false;
        new.models[1].tokenizer = Some("builtin:sim".into());
        assert_eq!(
            old.restart_changes(&new),
            ["server", "health", "logging.level", "index", "models[b]"]
        );
        let mut renamed = old.clone();
        renamed.models.swap(0, 1);
        assert_eq!(old.restart_changes(&renamed), ["models"]);
        let mut fewer = old.clone();
        fewer.models.pop();
        assert_eq!(old.restart_changes(&fewer), ["models"]);
    }
}
