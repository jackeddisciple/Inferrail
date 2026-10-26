//! A sampled JSON lines log of routing decisions, written off the request
//! path: records go through a bounded channel to a writer thread and are
//! dropped, and counted, when it falls behind.

use std::fs::OpenOptions;
use std::io::{self, BufWriter, Write};
use std::path::Path;
use std::sync::mpsc::{Receiver, RecvTimeoutError, SyncSender, TrySendError, sync_channel};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use inferrail_control::{Fleet, Nanos};
use inferrail_core::ids::{ModelId, ReplicaId};
use prometheus_client::metrics::counter::Counter;
use serde::Serialize;
use xxhash_rust::xxh3::xxh3_64;

const CAPACITY: usize = 4096;
const IDLE_FLUSH: Duration = Duration::from_millis(200);

#[derive(Debug, Clone, PartialEq, Serialize)]
pub(crate) struct CandidateState {
    pub(crate) replica: ReplicaId,
    pub(crate) eligible: bool,
    pub(crate) healthy: bool,
    pub(crate) tried: bool,
    pub(crate) in_flight: u32,
    pub(crate) pending: u32,
    pub(crate) in_flight_tokens: u64,
    pub(crate) prefill_tokens: u64,
    pub(crate) engine_running: Option<u32>,
    pub(crate) engine_waiting: Option<u32>,
    pub(crate) engine_kv_usage: Option<f64>,
    pub(crate) load_age_ms: Option<f64>,
    /// Prompt tokens the KV index predicted it had cached, when offered.
    pub(crate) cached_tokens: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub(crate) struct Decision {
    pub(crate) ts_ms: u64,
    pub(crate) request_id: String,
    pub(crate) model: ModelId,
    pub(crate) policy: &'static str,
    pub(crate) attempt: u32,
    pub(crate) chosen: ReplicaId,
    pub(crate) decision_ns: u64,
    pub(crate) prompt_tokens: Option<u32>,
    pub(crate) candidates: Vec<CandidateState>,
}

/// Every replica serving the model as the policy saw it: whether it was
/// offered, and its load.
pub(crate) fn candidates(
    fleet: &Fleet,
    model: &ModelId,
    now: Nanos,
    tried: &[ReplicaId],
    max_age: Duration,
) -> Vec<CandidateState> {
    let instant = tokio::time::Instant::now();
    fleet
        .serving(model)
        .iter()
        .map(|r| {
            let load = r.load(instant, max_age);
            CandidateState {
                replica: r.id().clone(),
                eligible: r.is_routable(now) && r.has_room(),
                healthy: r.health().probe_up(),
                tried: tried.contains(r.id()),
                in_flight: load.in_flight,
                pending: load.pending,
                in_flight_tokens: load.in_flight_tokens,
                prefill_tokens: load.prefill_tokens,
                engine_running: load.engine.map(|e| e.running),
                engine_waiting: load.engine.map(|e| e.waiting),
                engine_kv_usage: load.engine.map(|e| e.kv_usage),
                load_age_ms: load.age.map(|a| a.as_secs_f64() * 1e3),
                cached_tokens: None,
            }
        })
        .collect()
}

/// Whether `path` could be opened for appending, without creating it: its
/// directory exists and it is not itself a directory.
pub(crate) fn writable(path: &Path) -> io::Result<()> {
    let dir = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    if !std::fs::metadata(dir)?.is_dir() {
        return Err(io::Error::new(
            io::ErrorKind::NotADirectory,
            format!("{} is not a directory", dir.display()),
        ));
    }
    match std::fs::metadata(path) {
        Ok(meta) if meta.is_dir() => Err(io::Error::new(
            io::ErrorKind::IsADirectory,
            "the path is a directory",
        )),
        Ok(meta) if meta.permissions().readonly() => Err(io::Error::new(
            io::ErrorKind::PermissionDenied,
            "the file is read only",
        )),
        _ => Ok(()),
    }
}

pub(crate) fn unix_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| u64::try_from(d.as_millis()).unwrap_or(u64::MAX))
}

#[derive(Debug)]
pub(crate) struct DecisionLog {
    sender: SyncSender<Decision>,
    threshold: Option<u64>,
    written: Counter,
    dropped: Counter,
}

impl DecisionLog {
    /// Opens `path` for appending and starts the writer thread.
    pub(crate) fn open(
        path: &Path,
        rate: f64,
        written: Counter,
        dropped: Counter,
    ) -> io::Result<Self> {
        let file = OpenOptions::new().create(true).append(true).open(path)?;
        let (log, receiver) = Self::channel(rate, CAPACITY, written, dropped);
        std::thread::Builder::new()
            .name("decision-log".to_owned())
            .spawn(move || write_all(receiver, BufWriter::new(file)))?;
        Ok(log)
    }

    fn channel(
        rate: f64,
        capacity: usize,
        written: Counter,
        dropped: Counter,
    ) -> (Self, Receiver<Decision>) {
        let (sender, receiver) = sync_channel(capacity);
        let threshold = (rate < 1.0).then_some((rate * u64::MAX as f64) as u64);
        let log = Self {
            sender,
            threshold,
            written,
            dropped,
        };
        (log, receiver)
    }

    /// Sampling follows the request id, so every attempt of one request is
    /// logged or none is.
    pub(crate) fn sampled(&self, request_id: &[u8]) -> bool {
        self.threshold
            .is_none_or(|limit| xxh3_64(request_id) < limit)
    }

    pub(crate) fn record(&self, decision: Decision) {
        match self.sender.try_send(decision) {
            Ok(()) => {
                self.written.inc();
            }
            Err(TrySendError::Full(_) | TrySendError::Disconnected(_)) => {
                self.dropped.inc();
            }
        }
    }
}

fn write_all(receiver: Receiver<Decision>, mut out: impl Write) {
    loop {
        let result = match receiver.recv_timeout(IDLE_FLUSH) {
            Ok(decision) => serde_json::to_writer(&mut out, &decision)
                .map_err(io::Error::from)
                .and_then(|()| out.write_all(b"\n")),
            Err(RecvTimeoutError::Timeout) => out.flush(),
            Err(RecvTimeoutError::Disconnected) => {
                let _ = out.flush();
                return;
            }
        };
        if let Err(err) = result {
            tracing::warn!(error = %err, "cannot write the decision log; stopping it");
            return;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn decision(n: u32) -> Decision {
        Decision {
            ts_ms: 1,
            request_id: format!("req-{n}"),
            model: ModelId::new("m").unwrap(),
            policy: "p2c",
            attempt: n,
            chosen: ReplicaId::new("r0").unwrap(),
            decision_ns: 100,
            prompt_tokens: None,
            candidates: Vec::new(),
        }
    }

    #[test]
    fn a_full_channel_drops_and_counts_instead_of_blocking() {
        let (written, dropped) = (Counter::default(), Counter::default());
        let (log, receiver) = DecisionLog::channel(1.0, 4, written.clone(), dropped.clone());
        let started = std::time::Instant::now();
        for n in 0..10 {
            log.record(decision(n));
        }
        assert!(started.elapsed() < Duration::from_millis(50));
        assert_eq!((written.get(), dropped.get()), (4, 6));
        let kept: Vec<u32> = receiver.try_iter().map(|d| d.attempt).collect();
        assert_eq!(kept, [0, 1, 2, 3]);
        drop(receiver);
        log.record(decision(10));
        assert_eq!(dropped.get(), 7);
    }

    #[test]
    fn sampling_is_stable_per_request_and_close_to_the_rate() {
        let (log, _rx) = DecisionLog::channel(0.25, 1, Counter::default(), Counter::default());
        let ids: Vec<String> = (0..20_000).map(|i| format!("{i:032x}")).collect();
        let kept = ids.iter().filter(|id| log.sampled(id.as_bytes())).count();
        assert!((4_500..5_500).contains(&kept), "{kept}");
        for id in &ids[..100] {
            assert_eq!(log.sampled(id.as_bytes()), log.sampled(id.as_bytes()));
        }
        let (all, _rx) = DecisionLog::channel(1.0, 1, Counter::default(), Counter::default());
        assert!(ids.iter().all(|id| all.sampled(id.as_bytes())));
        let (none, _rx) = DecisionLog::channel(0.0, 1, Counter::default(), Counter::default());
        assert!(!ids.iter().any(|id| none.sampled(id.as_bytes())));
    }

    #[test]
    fn the_writer_emits_one_json_object_per_line_and_flushes_on_close() {
        let (log, receiver) = DecisionLog::channel(1.0, 16, Counter::default(), Counter::default());
        for n in 0..3 {
            log.record(decision(n));
        }
        drop(log);
        let mut out = Vec::new();
        write_all(receiver, &mut out);
        let text = String::from_utf8(out).unwrap();
        let lines: Vec<serde_json::Value> = text
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect();
        assert_eq!(lines.len(), 3);
        assert_eq!(lines[2]["attempt"], 2);
        assert_eq!(lines[0]["model"], "m");
        assert_eq!(lines[0]["chosen"], "r0");
    }
}
