//! What a config reload replaces: the config, the names clients may use,
//! and each model's policy and cost model. A request reads one snapshot
//! for its whole life, so a reload never changes a decision halfway.

use std::collections::HashMap;
use std::sync::Arc;

use inferrail_core::config::{Config, PolicyKind};
use inferrail_core::ids::ModelId;
use inferrail_policy::{CostModel, RoutingPolicy, Settings};

use crate::errors::GatewayError;

/// A policy with the settings it was built from, so a reload that leaves
/// them alone keeps the policy and what it has learned.
#[derive(Debug, Clone)]
pub(crate) struct Built {
    kind: PolicyKind,
    settings: Settings,
    pub(crate) policy: Arc<dyn RoutingPolicy>,
}

impl Built {
    fn new(kind: PolicyKind, settings: Settings, previous: Option<&Self>) -> Self {
        match previous {
            Some(old) if old.kind == kind && old.settings == settings => old.clone(),
            _ => Self {
                policy: Arc::from(inferrail_policy::build(kind, &settings)),
                kind,
                settings,
            },
        }
    }
}

#[derive(Debug)]
pub(crate) struct ModelRouting {
    pub(crate) policy: Built,
    pub(crate) cost: CostModel,
}

#[derive(Debug)]
pub(crate) struct Live {
    pub(crate) config: Arc<Config>,
    pub(crate) generation: u64,
    names: HashMap<String, ModelId>,
    models: HashMap<ModelId, ModelRouting>,
}

impl Live {
    /// Builds the state for `config`, reusing every policy of `previous`
    /// whose kind and settings are unchanged. Unseeded routing draws from
    /// `seed`, the same for the gateway's whole life.
    pub(crate) fn build(config: Config, seed: u64, previous: Option<&Live>) -> Self {
        let seed = config.routing.seed.unwrap_or(seed);
        let mut names = HashMap::new();
        let mut models = HashMap::new();
        for model in &config.models {
            names.insert(model.name.as_str().to_owned(), model.name.clone());
            for alias in &model.aliases {
                names.insert(alias.clone(), model.name.clone());
            }
            let settings = Settings::new(
                &config,
                model,
                inferrail_policy::mix_seed(seed, model.name.as_str()),
            );
            let old = previous.and_then(|p| p.models.get(&model.name));
            let routing = ModelRouting {
                cost: settings.cost,
                policy: Built::new(config.policy_for(model), settings, old.map(|o| &o.policy)),
            };
            models.insert(model.name.clone(), routing);
        }
        Self {
            config: Arc::new(config),
            generation: previous.map_or(1, |p| p.generation + 1),
            names,
            models,
        }
    }

    pub(crate) fn resolve(&self, name: &str) -> Option<&ModelId> {
        self.names.get(name)
    }

    pub(crate) fn routing(&self, model: &ModelId) -> Result<&ModelRouting, GatewayError> {
        self.models
            .get(model)
            .ok_or_else(|| GatewayError::unknown_model(model.as_str()))
    }

    pub(crate) fn policy(&self, model: &ModelId) -> Result<&dyn RoutingPolicy, GatewayError> {
        self.routing(model).map(|r| r.policy.policy.as_ref())
    }

    pub(crate) fn cost(&self, model: &ModelId) -> CostModel {
        self.models.get(model).map(|r| r.cost).unwrap_or_default()
    }

    /// Each model's policy, for the policy gauge.
    pub(crate) fn policies(&self) -> Vec<(ModelId, PolicyKind)> {
        self.config
            .models
            .iter()
            .map(|m| (m.name.clone(), self.models[&m.name].policy.kind))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(extra: &str) -> Config {
        Config::from_yaml_str(&format!(
            "models:\n  - name: a\n    aliases: [ay]\n  - name: b\nreplicas:\n  - {{id: r, url: 'http://h:1', models: [a, b]}}\n{extra}"
        ))
        .unwrap()
    }

    #[test]
    fn unchanged_policies_survive_a_rebuild() {
        let first = Live::build(config(""), 7, None);
        let a = Arc::clone(
            &first
                .routing(&ModelId::new("a").unwrap())
                .unwrap()
                .policy
                .policy,
        );
        let same = Live::build(config(""), 7, Some(&first));
        assert_eq!(same.generation, 2);
        let kept = &same
            .routing(&ModelId::new("a").unwrap())
            .unwrap()
            .policy
            .policy;
        assert!(Arc::ptr_eq(kept, &a));
        let tuned = Live::build(config("routing:\n  epsilon: 0.5\n"), 7, Some(&same));
        let fresh = &tuned
            .routing(&ModelId::new("a").unwrap())
            .unwrap()
            .policy
            .policy;
        assert!(!Arc::ptr_eq(fresh, &a));
        let seeded = Live::build(config("routing:\n  seed: 8\n"), 7, Some(&first));
        let other = &seeded
            .routing(&ModelId::new("a").unwrap())
            .unwrap()
            .policy
            .policy;
        assert!(!Arc::ptr_eq(other, &a));
    }

    #[test]
    fn names_policies_and_costs_follow_the_config() {
        let live = Live::build(
            config(
                "routing:\n  default_policy: p2c\n  cost_model:\n    prefill_token_secs: 0.001\n",
            ),
            1,
            None,
        );
        assert_eq!(live.resolve("ay").map(ModelId::as_str), Some("a"));
        assert!(live.resolve("c").is_none());
        let b = ModelId::new("b").unwrap();
        assert_eq!(live.policy(&b).unwrap().kind(), PolicyKind::P2c);
        assert_eq!(live.cost(&b).prefill_token_secs, 0.001);
        assert!(live.policy(&ModelId::new("c").unwrap()).is_err());
        assert_eq!(
            live.policies(),
            [
                (ModelId::new("a").unwrap(), PolicyKind::P2c),
                (b, PolicyKind::P2c)
            ]
        );
    }
}
