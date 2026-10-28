//! What a config reload replaces: the config, the names clients may use,
//! and each model's policies and cost model. A request reads one snapshot
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
    pub(crate) kind: PolicyKind,
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

/// One policy of a split, taking the keys below `until` out of 100.
#[derive(Debug)]
pub(crate) struct Arm {
    pub(crate) built: Built,
    pub(crate) percent: u32,
    until: u32,
}

#[derive(Debug)]
pub(crate) struct ModelRouting {
    pub(crate) arms: Vec<Arm>,
    /// Policies that only choose in shadow, never routing a request.
    pub(crate) shadows: Vec<Built>,
    pub(crate) cost: CostModel,
}

impl ModelRouting {
    /// The arm a request routes by: its key picks a percentile, so the
    /// same conversation always lands on the same policy.
    pub(crate) fn pick(&self, key: u64) -> &Built {
        let at = (key % 100) as u32;
        let arm = self.arms.iter().find(|a| at < a.until);
        &arm.unwrap_or(&self.arms[self.arms.len() - 1]).built
    }

    /// The arm with the largest share, which explains requests without a
    /// conversation.
    pub(crate) fn main(&self) -> &Built {
        let mut best = &self.arms[0];
        for arm in &self.arms[1..] {
            if arm.percent > best.percent {
                best = arm;
            }
        }
        &best.built
    }

    pub(crate) fn find(&self, kind: PolicyKind) -> Option<&Built> {
        self.arms.iter().map(|a| &a.built).find(|b| b.kind == kind)
    }

    pub(crate) fn shadow(&self, kind: PolicyKind) -> Option<&Built> {
        self.shadows.iter().find(|b| b.kind == kind)
    }

    pub(crate) fn has_shadows(&self) -> bool {
        !self.shadows.is_empty()
    }
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
            let old = previous.and_then(|p| p.models.get(&model.name));
            let split = config.arms_for(model);
            let mut until = 0;
            let mut arms = Vec::with_capacity(split.len());
            for &(kind, percent) in &split {
                let name = if split.len() == 1 {
                    model.name.as_str().to_owned()
                } else {
                    format!("{}/{kind}", model.name)
                };
                let settings =
                    Settings::new(&config, model, inferrail_policy::mix_seed(seed, &name));
                until += percent;
                arms.push(Arm {
                    built: Built::new(kind, settings, old.and_then(|o| o.find(kind))),
                    percent,
                    until,
                });
            }
            let shadows = config
                .shadows_for(model)
                .iter()
                .map(|&kind| {
                    let name = format!("{}/shadow/{kind}", model.name);
                    let settings =
                        Settings::new(&config, model, inferrail_policy::mix_seed(seed, &name));
                    Built::new(kind, settings, old.and_then(|o| o.shadow(kind)))
                })
                .collect();
            let cost = CostModel::from(config.cost_model_for(model));
            models.insert(
                model.name.clone(),
                ModelRouting {
                    arms,
                    shadows,
                    cost,
                },
            );
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

    pub(crate) fn cost(&self, model: &ModelId) -> CostModel {
        self.models.get(model).map(|r| r.cost).unwrap_or_default()
    }

    /// Each model's policies with the percent of traffic each routes, for
    /// the policy gauges.
    pub(crate) fn policies(&self) -> Vec<(ModelId, PolicyKind, u32)> {
        self.config
            .models
            .iter()
            .flat_map(|m| {
                self.models[&m.name]
                    .arms
                    .iter()
                    .map(|a| (m.name.clone(), a.built.kind, a.percent))
            })
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

    fn policy_of(live: &Live, model: &str) -> Arc<dyn RoutingPolicy> {
        let routing = live.routing(&ModelId::new(model).unwrap()).unwrap();
        Arc::clone(&routing.main().policy)
    }

    #[test]
    fn unchanged_policies_survive_a_rebuild() {
        let first = Live::build(config(""), 7, None);
        let a = policy_of(&first, "a");
        let same = Live::build(config(""), 7, Some(&first));
        assert_eq!(same.generation, 2);
        assert!(Arc::ptr_eq(&policy_of(&same, "a"), &a));
        let tuned = Live::build(config("routing:\n  epsilon: 0.5\n"), 7, Some(&same));
        assert!(!Arc::ptr_eq(&policy_of(&tuned, "a"), &a));
        let seeded = Live::build(config("routing:\n  seed: 8\n"), 7, Some(&first));
        assert!(!Arc::ptr_eq(&policy_of(&seeded, "a"), &a));
    }

    #[test]
    fn a_split_shares_keys_by_percent_and_keeps_its_arms() {
        let split = "routing:\n  split: {cost_model: 70, kv_aware: 30}\n";
        let live = Live::build(config(split), 1, None);
        let routing = live.routing(&ModelId::new("a").unwrap()).unwrap();
        let mut counts = std::collections::HashMap::new();
        for key in 0..10_000u64 {
            let kind = routing.pick(key.wrapping_mul(0x9e37_79b9_7f4a_7c15)).kind;
            *counts.entry(kind).or_insert(0) += 1;
        }
        assert!(
            (6_700..7_300).contains(&counts[&PolicyKind::CostModel]),
            "{counts:?}"
        );
        assert_eq!(routing.pick(5).kind, routing.pick(105).kind);
        assert_eq!(routing.main().kind, PolicyKind::CostModel);
        let kv = Arc::clone(&routing.find(PolicyKind::KvAware).unwrap().policy);
        let moved = Live::build(
            config("routing:\n  split: {cost_model: 50, kv_aware: 50}\n"),
            1,
            Some(&live),
        );
        let after = moved.routing(&ModelId::new("a").unwrap()).unwrap();
        assert!(Arc::ptr_eq(
            &after.find(PolicyKind::KvAware).unwrap().policy,
            &kv
        ));
        assert!(after.find(PolicyKind::P2c).is_none());
        let single = Live::build(config(""), 1, None);
        let alone = single.routing(&ModelId::new("a").unwrap()).unwrap();
        assert_ne!(
            alone.main().settings.seed,
            routing.find(PolicyKind::CostModel).unwrap().settings.seed
        );
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
        assert_eq!(
            live.routing(&b).unwrap().pick(3).policy.kind(),
            PolicyKind::P2c
        );
        assert_eq!(live.cost(&b).prefill_token_secs, 0.001);
        assert!(live.routing(&ModelId::new("c").unwrap()).is_err());
        assert_eq!(
            live.policies(),
            [
                (ModelId::new("a").unwrap(), PolicyKind::P2c, 100),
                (b, PolicyKind::P2c, 100)
            ]
        );
    }
}
