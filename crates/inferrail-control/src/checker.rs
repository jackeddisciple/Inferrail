use std::collections::HashSet;
use std::future::Future;
use std::sync::Arc;
use std::time::Duration;

use inferrail_core::ids::ReplicaId;
use tokio::task::JoinSet;
use tokio::time::MissedTickBehavior;

use crate::fleet::Registry;
use crate::health::{Clock, HealthPolicy};
use crate::replica::Replica;

/// One active health check against one replica, true when it passed.
pub trait Probe: Clone + Send + Sync + 'static {
    fn probe(&self, replica: Arc<Replica>, timeout: Duration) -> impl Future<Output = bool> + Send;
}

#[derive(Debug, Clone, Copy)]
pub struct CheckerConfig {
    pub interval: Duration,
    pub timeout: Duration,
    pub policy: HealthPolicy,
}

/// Probes every member once per interval, all in parallel, and folds the
/// results into each replica's health. Runs until the task is dropped.
pub async fn run_checker<P: Probe>(
    registry: Arc<Registry>,
    probe: P,
    config: CheckerConfig,
    clock: Clock,
) {
    let mut ticker = tokio::time::interval(config.interval);
    ticker.set_missed_tick_behavior(MissedTickBehavior::Delay);
    let mut ejected: HashSet<ReplicaId> = HashSet::new();
    loop {
        ticker.tick().await;
        let fleet = registry.snapshot();
        let mut probes = JoinSet::new();
        for replica in fleet.replicas() {
            let (probe, replica) = (probe.clone(), Arc::clone(replica));
            probes.spawn(async move {
                let ok = probe.probe(Arc::clone(&replica), config.timeout).await;
                (replica, ok)
            });
        }
        while let Some(done) = probes.join_next().await {
            let Ok((replica, ok)) = done else { continue };
            match replica.health().record_probe(ok, &config.policy) {
                Some(true) => {
                    tracing::info!(replica = %replica.id(), "replica passed health checks")
                }
                Some(false) => {
                    tracing::warn!(replica = %replica.id(), "replica failed health checks")
                }
                None => {}
            }
        }
        let now = clock.now();
        for replica in fleet.replicas() {
            let id = replica.id();
            if replica.health().is_ejected(now) {
                ejected.insert(id.clone());
            } else if ejected.remove(id) {
                tracing::info!(replica = %id, "ejection window ended; replica readmitted");
            }
        }
        ejected.retain(|id| fleet.get(id.as_str()).is_some());
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use super::*;
    use crate::replica::tests::spec;

    #[derive(Clone, Default)]
    struct Script(Arc<Mutex<Vec<String>>>);

    impl Probe for Script {
        async fn probe(&self, replica: Arc<Replica>, _timeout: Duration) -> bool {
            !self.0.lock().unwrap().contains(&replica.id().to_string())
        }
    }

    #[tokio::test(start_paused = true)]
    async fn probes_flip_health_after_thresholds() {
        let registry = Arc::new(Registry::new(vec![
            spec("a", "http://a:1", &["m"]),
            spec("b", "http://b:1", &["m"]),
        ]));
        let script = Script::default();
        let config = CheckerConfig {
            interval: Duration::from_millis(100),
            timeout: Duration::from_millis(50),
            policy: HealthPolicy {
                unhealthy_threshold: 3,
                healthy_threshold: 2,
                ejection_backoff: Duration::from_secs(1),
                max_ejection_backoff: Duration::from_secs(1),
            },
        };
        let task = tokio::spawn(run_checker(
            Arc::clone(&registry),
            script.clone(),
            config,
            Clock::default(),
        ));
        let up = |id: &str| registry.load().get(id).unwrap().health().probe_up();
        script.0.lock().unwrap().push("b".to_owned());
        tokio::time::sleep(Duration::from_millis(150)).await;
        assert!(up("b"), "one failure is not enough");
        tokio::time::sleep(Duration::from_millis(200)).await;
        assert!(!up("b"));
        assert!(up("a"));
        script.0.lock().unwrap().clear();
        tokio::time::sleep(Duration::from_millis(100)).await;
        assert!(!up("b"), "one success is not enough");
        tokio::time::sleep(Duration::from_millis(100)).await;
        assert!(up("b"));
        task.abort();
    }
}
