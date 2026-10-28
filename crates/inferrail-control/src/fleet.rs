use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::{Arc, Mutex, PoisonError};

use arc_swap::ArcSwap;
use inferrail_core::ids::{ModelId, ReplicaId};

use crate::replica::{Replica, ReplicaSpec};

/// An immutable view of every routable member, indexed by model.
#[derive(Debug, Default)]
pub struct Fleet {
    replicas: Vec<Arc<Replica>>,
    by_model: HashMap<ModelId, Vec<Arc<Replica>>>,
}

impl Fleet {
    pub fn new(replicas: Vec<Arc<Replica>>) -> Self {
        let mut by_model: HashMap<ModelId, Vec<Arc<Replica>>> = HashMap::new();
        for replica in &replicas {
            for model in replica.models() {
                by_model
                    .entry(model.clone())
                    .or_default()
                    .push(Arc::clone(replica));
            }
        }
        Self { replicas, by_model }
    }

    pub fn replicas(&self) -> &[Arc<Replica>] {
        &self.replicas
    }

    pub fn serving(&self, model: &ModelId) -> &[Arc<Replica>] {
        self.by_model.get(model).map_or(&[], Vec::as_slice)
    }

    pub fn get(&self, id: &str) -> Option<&Arc<Replica>> {
        self.replicas.iter().find(|r| r.id().as_str() == id)
    }
}

#[derive(Debug, Default, PartialEq, Eq)]
pub struct MembershipChange {
    pub added: Vec<ReplicaId>,
    pub removed: Vec<ReplicaId>,
    pub rejected: Vec<ReplicaId>,
}

impl MembershipChange {
    pub fn is_empty(&self) -> bool {
        self.added.is_empty() && self.removed.is_empty() && self.rejected.is_empty()
    }
}

/// Holds the current fleet for lock free readers. Writers (discovery and
/// config reloads) are rare and serialize on a mutex that the hot path
/// never touches.
#[derive(Debug)]
pub struct Registry {
    fleet: ArcSwap<Fleet>,
    members: Mutex<Members>,
}

#[derive(Debug, Default)]
struct Members {
    statics: Vec<Arc<Replica>>,
    groups: BTreeMap<ReplicaId, Vec<Arc<Replica>>>,
}

impl Members {
    fn publish(&self, fleet: &ArcSwap<Fleet>) {
        let all = self
            .statics
            .iter()
            .chain(self.groups.values().flatten())
            .cloned()
            .collect();
        fleet.store(Arc::new(Fleet::new(all)));
    }
}

/// Replaces `previous` with `members`, keeping the replica of every member
/// whose spec is unchanged so health and counters survive, and refusing ids
/// in `taken` or given twice.
fn replace(
    previous: &[Arc<Replica>],
    members: Vec<ReplicaSpec>,
    taken: &HashSet<&str>,
) -> (Vec<Arc<Replica>>, MembershipChange) {
    let mut change = MembershipChange::default();
    let mut next: Vec<Arc<Replica>> = Vec::with_capacity(members.len());
    for spec in members {
        if taken.contains(spec.id.as_str()) || next.iter().any(|r| r.id() == &spec.id) {
            change.rejected.push(spec.id);
            continue;
        }
        match previous.iter().find(|r| r.spec() == &spec) {
            Some(kept) => next.push(Arc::clone(kept)),
            None => {
                change.added.push(spec.id.clone());
                next.push(Arc::new(Replica::new(spec)));
            }
        }
    }
    change.removed = previous
        .iter()
        .filter(|old| !next.iter().any(|r| Arc::ptr_eq(r, old)))
        .map(|old| old.id().clone())
        .collect();
    (next, change)
}

impl Registry {
    pub fn new(statics: Vec<ReplicaSpec>) -> Self {
        let statics: Vec<Arc<Replica>> = statics
            .into_iter()
            .map(|s| Arc::new(Replica::new(s)))
            .collect();
        Self {
            fleet: ArcSwap::from_pointee(Fleet::new(statics.clone())),
            members: Mutex::new(Members {
                statics,
                groups: BTreeMap::new(),
            }),
        }
    }

    pub fn load(&self) -> arc_swap::Guard<Arc<Fleet>> {
        self.fleet.load()
    }

    pub fn snapshot(&self) -> Arc<Fleet> {
        self.fleet.load_full()
    }

    fn members(&self) -> std::sync::MutexGuard<'_, Members> {
        self.members.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Replaces one discovery group's members. Members whose id and URL are
    /// unchanged keep their replica, so health and counters survive.
    pub fn set_group(&self, group: &ReplicaId, members: Vec<ReplicaSpec>) -> MembershipChange {
        let mut all = self.members();
        let previous = all.groups.remove(group).unwrap_or_default();
        let taken: HashSet<&str> = all
            .statics
            .iter()
            .chain(all.groups.values().flatten())
            .map(|r| r.id().as_str())
            .collect();
        let (next, change) = replace(&previous, members, &taken);
        if !next.is_empty() {
            all.groups.insert(group.clone(), next);
        }
        all.publish(&self.fleet);
        change
    }

    /// Replaces the replicas listed in the config, as a reload does. Ones
    /// whose spec is unchanged keep their replica; requests already on a
    /// removed one finish there.
    pub fn set_statics(&self, members: Vec<ReplicaSpec>) -> MembershipChange {
        let mut all = self.members();
        let taken: HashSet<&str> = all
            .groups
            .values()
            .flatten()
            .map(|r| r.id().as_str())
            .collect();
        let (next, change) = replace(&all.statics, members, &taken);
        all.statics = next;
        all.publish(&self.fleet);
        change
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::*;
    use crate::replica::Origin;
    use crate::replica::tests::spec;
    use proptest::prelude::*;

    fn member(group: &str, n: u8) -> ReplicaSpec {
        let mut s = spec(
            &format!("{group}-10.0.0.{n}-8000"),
            &format!("http://10.0.0.{n}:8000"),
            &["m"],
        );
        s.origin = Origin::Dns(ReplicaId::new(group).unwrap());
        s
    }

    fn ids(fleet: &Fleet) -> Vec<&str> {
        fleet.replicas().iter().map(|r| r.id().as_str()).collect()
    }

    #[test]
    fn fleet_indexes_replicas_by_model_in_order() {
        let fleet = Fleet::new(vec![
            Arc::new(Replica::new(spec("a", "http://a:1", &["m", "n"]))),
            Arc::new(Replica::new(spec("b", "http://b:1", &["n"]))),
        ]);
        let names = |model: &str| -> Vec<String> {
            fleet
                .serving(&ModelId::new(model).unwrap())
                .iter()
                .map(|r| r.id().to_string())
                .collect()
        };
        assert_eq!(names("m"), ["a"]);
        assert_eq!(names("n"), ["a", "b"]);
        assert!(names("o").is_empty());
        assert!(fleet.get("b").is_some());
        assert!(fleet.get("c").is_none());
    }

    #[test]
    fn groups_join_leave_and_keep_survivors() {
        let registry = Registry::new(vec![spec("s", "http://s:1", &["m"])]);
        let group = ReplicaId::new("pool").unwrap();
        let change = registry.set_group(&group, vec![member("pool", 1), member("pool", 2)]);
        assert_eq!(change.added.len(), 2);
        assert_eq!(
            ids(&registry.load()),
            ["s", "pool-10.0.0.1-8000", "pool-10.0.0.2-8000"]
        );
        let survivor = Arc::clone(registry.load().get("pool-10.0.0.2-8000").unwrap());
        let _lease = survivor.try_acquire(0).unwrap();

        let change = registry.set_group(&group, vec![member("pool", 2), member("pool", 3)]);
        assert_eq!(
            change.added,
            [ReplicaId::new("pool-10.0.0.3-8000").unwrap()]
        );
        assert_eq!(
            change.removed,
            [ReplicaId::new("pool-10.0.0.1-8000").unwrap()]
        );
        let fleet = registry.snapshot();
        let kept = fleet.get("pool-10.0.0.2-8000").unwrap();
        assert!(Arc::ptr_eq(kept, &survivor));
        assert_eq!(kept.in_flight(), 1);

        let change = registry.set_group(&group, vec![]);
        assert_eq!(change.removed.len(), 2);
        assert_eq!(ids(&registry.load()), ["s"]);
    }

    #[test]
    fn statics_are_replaced_in_place_keeping_survivors() {
        let registry = Registry::new(vec![
            spec("a", "http://a:1", &["m"]),
            spec("b", "http://b:1", &["m"]),
        ]);
        let group = ReplicaId::new("pool").unwrap();
        registry.set_group(&group, vec![member("pool", 1)]);
        let a = Arc::clone(registry.load().get("a").unwrap());
        let change = registry.set_statics(vec![
            spec("a", "http://a:1", &["m"]),
            spec("c", "http://c:1", &["m"]),
            spec("pool-10.0.0.1-8000", "http://x:1", &["m"]),
        ]);
        assert_eq!(change.added, [ReplicaId::new("c").unwrap()]);
        assert_eq!(change.removed, [ReplicaId::new("b").unwrap()]);
        assert_eq!(
            change.rejected,
            [ReplicaId::new("pool-10.0.0.1-8000").unwrap()]
        );
        assert_eq!(ids(&registry.load()), ["a", "c", "pool-10.0.0.1-8000"]);
        assert!(Arc::ptr_eq(registry.load().get("a").unwrap(), &a));
        let change = registry.set_statics(vec![spec("a", "http://a:2", &["m"])]);
        assert_eq!((change.added.len(), change.removed.len()), (1, 2));
        assert!(!Arc::ptr_eq(registry.load().get("a").unwrap(), &a));
        assert_eq!(registry.load().replicas().len(), 2);
    }

    #[test]
    fn colliding_ids_are_rejected() {
        let registry = Registry::new(vec![spec("pool-10.0.0.1-8000", "http://s:1", &["m"])]);
        let change = registry.set_group(
            &ReplicaId::new("pool").unwrap(),
            vec![member("pool", 1), member("pool", 2), member("pool", 2)],
        );
        assert_eq!(change.rejected.len(), 2);
        assert_eq!(change.added.len(), 1);
        assert_eq!(registry.load().replicas().len(), 2);
    }

    #[test]
    fn a_changed_url_replaces_the_member() {
        let registry = Registry::new(vec![]);
        let group = ReplicaId::new("pool").unwrap();
        registry.set_group(&group, vec![member("pool", 1)]);
        let mut moved = member("pool", 1);
        moved.url = "http://10.0.0.1:9000".parse().unwrap();
        let change = registry.set_group(&group, vec![moved]);
        assert_eq!(change.removed, change.added);
        assert_eq!(
            registry.load().replicas()[0].endpoint().origin(),
            "http://10.0.0.1:9000"
        );
    }

    proptest! {
        #[test]
        fn fleet_matches_a_reference_model(
            updates in prop::collection::vec((0u8..3, prop::collection::btree_set(1u8..12, 0..6)), 1..30)
        ) {
            let registry = Registry::new(vec![spec("static", "http://s:1", &["m"])]);
            let mut model: BTreeMap<u8, BTreeSet<u8>> = BTreeMap::new();
            for (group, hosts) in updates {
                let name = format!("g{group}");
                let members = hosts.iter().map(|&h| member(&name, h)).collect();
                registry.set_group(&ReplicaId::new(&name).unwrap(), members);
                model.insert(group, hosts);
                let mut want: BTreeSet<String> = model
                    .iter()
                    .flat_map(|(g, hosts)| hosts.iter().map(move |h| format!("g{g}-10.0.0.{h}-8000")))
                    .collect();
                want.insert("static".to_owned());
                let fleet = registry.load();
                let got: BTreeSet<String> = fleet.replicas().iter().map(|r| r.id().to_string()).collect();
                prop_assert_eq!(&got, &want);
                prop_assert_eq!(fleet.serving(&ModelId::new("m").unwrap()).len(), want.len());
            }
        }
    }
}
