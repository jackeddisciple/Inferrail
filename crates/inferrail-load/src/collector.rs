use std::collections::HashSet;
use std::future::Future;
use std::sync::Arc;
use std::time::Duration;

use tokio::task::JoinSet;
use tokio::time::{Instant, MissedTickBehavior};

use crate::{EngineLoad, LoadCell, ParseError};

/// Something whose engine load can be scraped and published.
pub trait Target: Send + Sync + 'static {
    fn name(&self) -> &str;
    fn cell(&self) -> &LoadCell;
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ScrapeError {
    #[error("no answer within the scrape timeout")]
    Timeout,
    #[error("unreachable: {0}")]
    Unreachable(String),
    #[error("answered {0}")]
    Status(u16),
    #[error("page larger than {0} bytes")]
    TooLarge(usize),
    #[error("page is not UTF-8")]
    NotUtf8,
    #[error(transparent)]
    Parse(#[from] ParseError),
}

impl ScrapeError {
    /// A short stable name, for metric labels.
    pub fn kind(&self) -> &'static str {
        match self {
            Self::Timeout => "timeout",
            Self::Unreachable(_) => "unreachable",
            Self::Status(_) => "status",
            Self::TooLarge(_) => "too_large",
            Self::NotUtf8 | Self::Parse(_) => "unreadable",
        }
    }
}

pub trait Scrape<T: Target>: Clone + Send + Sync + 'static {
    fn scrape(
        &self,
        target: Arc<T>,
    ) -> impl Future<Output = Result<EngineLoad, ScrapeError>> + Send;

    /// Sees every outcome, after the timeout is applied.
    fn observed(&self, _target: &T, _result: &Result<EngineLoad, ScrapeError>) {}
}

#[derive(Debug, Clone, Copy)]
pub struct CollectorConfig {
    pub interval: Duration,
    pub timeout: Duration,
}

/// Scrapes every target once per interval, all in parallel, and publishes
/// each answer with the time its request was sent. A round ends when every
/// scrape has answered or timed out, so a hung engine delays nobody past the
/// timeout. Runs until the task is dropped.
pub async fn run_collector<T, S, F>(targets: F, scrape: S, config: CollectorConfig)
where
    T: Target,
    S: Scrape<T>,
    F: Fn() -> Vec<Arc<T>> + Send,
{
    let mut ticker = tokio::time::interval(config.interval);
    ticker.set_missed_tick_behavior(MissedTickBehavior::Delay);
    let mut failing: HashSet<String> = HashSet::new();
    loop {
        ticker.tick().await;
        let round = targets();
        let mut scrapes = JoinSet::new();
        for target in &round {
            let (scrape, target) = (scrape.clone(), Arc::clone(target));
            scrapes.spawn(async move {
                let sent = Instant::now();
                let result =
                    tokio::time::timeout(config.timeout, scrape.scrape(Arc::clone(&target)))
                        .await
                        .unwrap_or(Err(ScrapeError::Timeout));
                scrape.observed(&target, &result);
                if let Ok(load) = &result {
                    target.cell().publish(*load, sent);
                }
                (target, result.err())
            });
        }
        while let Some(done) = scrapes.join_next().await {
            let Ok((target, error)) = done else { continue };
            let name = target.name();
            match error {
                Some(err) if !failing.contains(name) => {
                    tracing::warn!(replica = %name, error = %err, "cannot read engine load; routing on gateway counts until it recovers");
                    failing.insert(name.to_owned());
                }
                None if failing.remove(name) => {
                    tracing::info!(replica = %name, "engine load readable again");
                }
                _ => {}
            }
        }
        failing.retain(|name| round.iter().any(|t| t.name() == name));
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicU32, Ordering};

    use super::*;
    use crate::Engine;

    #[derive(Debug)]
    struct Node {
        name: String,
        cell: LoadCell,
    }

    impl Target for Node {
        fn name(&self) -> &str {
            &self.name
        }

        fn cell(&self) -> &LoadCell {
            &self.cell
        }
    }

    #[derive(Debug, Clone, Copy)]
    enum Behavior {
        Answer(u32),
        Fail,
        Slow(Duration),
    }

    #[derive(Debug, Clone, Default)]
    struct Script {
        behavior: Arc<Mutex<HashMap<String, Behavior>>>,
        observed: Arc<AtomicU32>,
    }

    impl Script {
        fn set(&self, name: &str, behavior: Behavior) {
            self.behavior
                .lock()
                .unwrap()
                .insert(name.to_owned(), behavior);
        }
    }

    fn load(running: u32) -> EngineLoad {
        EngineLoad {
            engine: Engine::Sglang,
            running,
            waiting: 0,
            kv_usage: 0.1,
            prefix_queries: 0,
            prefix_hits: 0,
            preemptions: 0,
            kv_capacity_tokens: None,
        }
    }

    impl Scrape<Node> for Script {
        async fn scrape(&self, target: Arc<Node>) -> Result<EngineLoad, ScrapeError> {
            let behavior = self.behavior.lock().unwrap().get(&target.name).copied();
            match behavior.unwrap_or(Behavior::Answer(0)) {
                Behavior::Answer(n) => Ok(load(n)),
                Behavior::Fail => Err(ScrapeError::Status(500)),
                Behavior::Slow(pause) => {
                    tokio::time::sleep(pause).await;
                    Ok(load(99))
                }
            }
        }

        fn observed(&self, _target: &Node, _result: &Result<EngineLoad, ScrapeError>) {
            self.observed.fetch_add(1, Ordering::Relaxed);
        }
    }

    fn node(name: &str) -> Arc<Node> {
        Arc::new(Node {
            name: name.to_owned(),
            cell: LoadCell::default(),
        })
    }

    const CONFIG: CollectorConfig = CollectorConfig {
        interval: Duration::from_millis(100),
        timeout: Duration::from_millis(80),
    };

    #[tokio::test(start_paused = true)]
    async fn every_target_is_published_each_interval() {
        let nodes = vec![node("a"), node("b")];
        let script = Script::default();
        script.set("a", Behavior::Answer(3));
        let fleet = nodes.clone();
        let task = tokio::spawn(run_collector(move || fleet.clone(), script.clone(), CONFIG));
        tokio::time::sleep(Duration::from_millis(1)).await;
        for _ in 0..20 {
            for n in &nodes {
                let age = n.cell.latest().unwrap().age(Instant::now());
                assert!(age <= CONFIG.interval, "{} is {age:?} old", n.name);
            }
            tokio::time::sleep(Duration::from_millis(37)).await;
        }
        assert_eq!(nodes[0].cell.latest().unwrap().load.running, 3);
        assert!(script.observed.load(Ordering::Relaxed) >= 14);
        task.abort();
    }

    #[tokio::test(start_paused = true)]
    async fn failures_keep_the_last_sample_until_it_ages_out() {
        let a = node("a");
        let script = Script::default();
        script.set("a", Behavior::Answer(5));
        let fleet = vec![Arc::clone(&a)];
        let task = tokio::spawn(run_collector(move || fleet.clone(), script.clone(), CONFIG));
        tokio::time::sleep(Duration::from_millis(1)).await;
        assert!(
            a.cell
                .fresh(Instant::now(), Duration::from_millis(250))
                .is_some()
        );
        script.set("a", Behavior::Fail);
        tokio::time::sleep(Duration::from_millis(150)).await;
        assert_eq!(a.cell.latest().unwrap().load.running, 5);
        tokio::time::sleep(Duration::from_millis(150)).await;
        assert!(
            a.cell
                .fresh(Instant::now(), Duration::from_millis(250))
                .is_none()
        );
        script.set("a", Behavior::Answer(6));
        tokio::time::sleep(Duration::from_millis(100)).await;
        let fresh = a
            .cell
            .fresh(Instant::now(), Duration::from_millis(250))
            .unwrap();
        assert_eq!(fresh.load.running, 6);
        task.abort();
    }

    #[tokio::test(start_paused = true)]
    async fn a_slow_engine_times_out_without_delaying_the_others() {
        let (a, slow) = (node("a"), node("slow"));
        let script = Script::default();
        script.set("slow", Behavior::Slow(Duration::from_secs(10)));
        let fleet = vec![Arc::clone(&a), Arc::clone(&slow)];
        let task = tokio::spawn(run_collector(move || fleet.clone(), script.clone(), CONFIG));
        let mut worst = Duration::ZERO;
        for _ in 0..50 {
            tokio::time::sleep(Duration::from_millis(13)).await;
            if let Some(sample) = a.cell.latest() {
                worst = worst.max(sample.age(Instant::now()));
            }
        }
        assert!(worst <= CONFIG.interval + CONFIG.timeout, "{worst:?}");
        assert!(slow.cell.latest().is_none());
        task.abort();
    }

    #[tokio::test(start_paused = true)]
    async fn membership_changes_are_followed() {
        let fleet: Arc<Mutex<Vec<Arc<Node>>>> = Arc::new(Mutex::new(vec![node("a")]));
        let script = Script::default();
        let current = Arc::clone(&fleet);
        let task = tokio::spawn(run_collector(
            move || current.lock().unwrap().clone(),
            script.clone(),
            CONFIG,
        ));
        tokio::time::sleep(Duration::from_millis(1)).await;
        let joined = node("b");
        fleet.lock().unwrap().push(Arc::clone(&joined));
        tokio::time::sleep(Duration::from_millis(100)).await;
        assert!(joined.cell.latest().is_some());
        let left = fleet.lock().unwrap().remove(0);
        let last = left.cell.latest().unwrap().at;
        tokio::time::sleep(Duration::from_millis(500)).await;
        assert_eq!(left.cell.latest().unwrap().at, last);
        task.abort();
    }

    #[test]
    fn error_kinds_are_stable_labels() {
        let kinds: Vec<_> = [
            ScrapeError::Timeout,
            ScrapeError::Unreachable("x".into()),
            ScrapeError::Status(503),
            ScrapeError::TooLarge(1),
            ScrapeError::NotUtf8,
            ScrapeError::Parse(ParseError::UnknownSchema),
        ]
        .iter()
        .map(ScrapeError::kind)
        .collect();
        assert_eq!(
            kinds,
            [
                "timeout",
                "unreachable",
                "status",
                "too_large",
                "unreadable",
                "unreadable"
            ]
        );
        assert_eq!(ScrapeError::Status(503).to_string(), "answered 503");
    }
}
