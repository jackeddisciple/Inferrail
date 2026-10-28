//! Shadow mode: candidate policies choose a replica for every routed
//! request without routing it. One worker evaluates them in the order
//! requests were routed, against the candidates exactly as the active
//! policy saw them, so their state evolves only from what they are shown
//! and a log of what they were shown replays to the same choices offline.
//!
//! Each shadow pick is priced by the model's cost model next to the active
//! pick: the estimated gain is the active estimate minus the shadow's, so
//! a policy is judged by what the cost model believes it would save. That
//! is the cost model's view, which favors `cost_model` itself by design.

use std::collections::BTreeMap;
use std::fs::{File, OpenOptions};
use std::io::{self, BufRead, BufWriter, Write};
use std::path::Path;
use std::sync::mpsc::{Receiver, RecvTimeoutError, SyncSender, TrySendError, sync_channel};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Duration;

use inferrail_core::config::{Config, PolicyKind};
use inferrail_core::ids::{BlockHash, ModelId, ReplicaId};
use inferrail_policy::{Candidate, CostModel, Hits, RouteContext};
use serde::{Deserialize, Serialize};

use crate::live::Live;
use crate::metrics::Metrics;
use crate::route::Observed;

const CAPACITY: usize = 8192;
const IDLE_FLUSH: Duration = Duration::from_millis(200);

/// One routed request as the active policy decided it.
#[derive(Debug)]
pub(crate) struct Job {
    pub(crate) ts_ms: u64,
    pub(crate) request_id: String,
    pub(crate) live: Arc<Live>,
    pub(crate) model: ModelId,
    pub(crate) active: PolicyKind,
    pub(crate) chosen: usize,
    pub(crate) now: u64,
    pub(crate) prompt_tokens: u32,
    pub(crate) block_size: u32,
    pub(crate) blocks: Vec<BlockHash>,
    pub(crate) session: Option<u64>,
    pub(crate) seen: Vec<Observed>,
}

#[derive(Debug)]
enum Message {
    Decision(Box<Job>),
    Generation(Arc<Live>),
}

/// What a shadow policy has done so far.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize)]
pub(crate) struct Tally {
    pub(crate) decisions: u64,
    pub(crate) agreed: u64,
    /// Decisions where the cost model priced the shadow pick below the
    /// active one.
    pub(crate) faster: u64,
    pub(crate) slower: u64,
    pub(crate) active_secs: f64,
    pub(crate) shadow_secs: f64,
}

type Tallies = Arc<Mutex<BTreeMap<(ModelId, PolicyKind), Tally>>>;

#[derive(Debug)]
pub(crate) struct Shadow {
    sender: SyncSender<Message>,
    tallies: Tallies,
    metrics: Arc<Metrics>,
}

impl Shadow {
    /// Starts the worker; with `log`, every decision is appended there.
    pub(crate) fn start(log: Option<&Path>, seed: u64, metrics: Arc<Metrics>) -> io::Result<Self> {
        let out = log
            .map(|path| OpenOptions::new().create(true).append(true).open(path))
            .transpose()?
            .map(BufWriter::new);
        let (sender, receiver) = sync_channel(CAPACITY);
        let tallies = Tallies::default();
        let worker = Worker {
            out,
            seed,
            generation: 0,
            tallies: Arc::clone(&tallies),
            metrics: Arc::clone(&metrics),
        };
        std::thread::Builder::new()
            .name("shadow".to_owned())
            .spawn(move || worker.run(receiver))?;
        Ok(Self {
            sender,
            tallies,
            metrics,
        })
    }

    /// Hands a decision to the worker, or drops and counts it when the
    /// worker is behind, so routing never waits on a shadow.
    pub(crate) fn offer(&self, job: Job) {
        match self.sender.try_send(Message::Decision(Box::new(job))) {
            Ok(()) => {}
            Err(TrySendError::Full(_) | TrySendError::Disconnected(_)) => {
                self.metrics.shadow_dropped();
            }
        }
    }

    /// Tells the worker about a new config, so the log records it even
    /// before the next request.
    pub(crate) fn reloaded(&self, live: Arc<Live>) {
        let _ = self.sender.send(Message::Generation(live));
    }

    pub(crate) fn tallies(&self) -> BTreeMap<(ModelId, PolicyKind), Tally> {
        self.tallies
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct Header {
    shadow_log: u32,
    generation: u64,
    seed: u64,
    config: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct Pick {
    policy: PolicyKind,
    replica: Option<ReplicaId>,
    steer: Option<String>,
    estimated_ttft_ms: Option<f64>,
}

#[derive(Debug, Serialize, Deserialize)]
struct Line {
    ts_ms: u64,
    request_id: String,
    model: ModelId,
    generation: u64,
    active: Pick,
    now: u64,
    prompt_tokens: u32,
    block_size: u32,
    session: Option<u64>,
    blocks: String,
    candidates: Vec<Observed>,
    shadow: Vec<Pick>,
}

fn hex(blocks: &[BlockHash]) -> String {
    let mut out = String::with_capacity(blocks.len() * 32);
    for block in blocks {
        out.push_str(&format!("{:032x}", block.get()));
    }
    out
}

fn unhex(text: &str) -> Option<Vec<BlockHash>> {
    if !text.len().is_multiple_of(32) || !text.is_ascii() {
        return None;
    }
    (0..text.len())
        .step_by(32)
        .map(|i| {
            u128::from_str_radix(&text[i..i + 32], 16)
                .ok()
                .map(BlockHash::new)
        })
        .collect()
}

fn header(live: &Live, seed: u64) -> Header {
    Header {
        shadow_log: 1,
        generation: live.generation,
        seed,
        config: live.config.to_yaml_string().unwrap_or_default(),
    }
}

fn context<'a>(
    model: &'a ModelId,
    line: (&'a [BlockHash], u32, u32, Option<u64>, u64),
    candidates: &'a [Candidate<'a>],
) -> RouteContext<'a> {
    let (blocks, prompt_tokens, block_size, session, now) = line;
    RouteContext {
        model,
        prompt_tokens,
        blocks,
        block_size,
        session,
        now,
        candidates,
    }
}

fn estimate(cost: &CostModel, prompt: u32, candidate: &Candidate<'_>) -> f64 {
    cost.estimate(prompt, candidate, Hits::gpu(candidate.cached_tokens))
        .ttft_secs
}

struct Worker {
    out: Option<BufWriter<File>>,
    seed: u64,
    generation: u64,
    tallies: Tallies,
    metrics: Arc<Metrics>,
}

impl Worker {
    fn run(mut self, receiver: Receiver<Message>) {
        loop {
            let result = match receiver.recv_timeout(IDLE_FLUSH) {
                Ok(Message::Generation(live)) => self.generation(&live),
                Ok(Message::Decision(job)) => self.decide(*job),
                Err(RecvTimeoutError::Timeout) => self.out.as_mut().map_or(Ok(()), Write::flush),
                Err(RecvTimeoutError::Disconnected) => {
                    if let Some(out) = self.out.as_mut() {
                        let _ = out.flush();
                    }
                    return;
                }
            };
            if let Err(err) = result {
                tracing::warn!(error = %err, "cannot write the shadow log; stopping it");
                self.out = None;
            }
        }
    }

    fn write(&mut self, value: &impl Serialize) -> io::Result<()> {
        let Some(out) = self.out.as_mut() else {
            return Ok(());
        };
        serde_json::to_writer(&mut *out, value)?;
        out.write_all(b"\n")
    }

    fn generation(&mut self, live: &Live) -> io::Result<()> {
        if live.generation == self.generation {
            return Ok(());
        }
        self.generation = live.generation;
        let header = header(live, self.seed);
        self.write(&header)
    }

    fn decide(&mut self, job: Job) -> io::Result<()> {
        self.generation(&job.live)?;
        let Ok(routing) = job.live.routing(&job.model) else {
            return Ok(());
        };
        let candidates: Vec<Candidate<'_>> = job.seen.iter().map(Observed::view).collect();
        let Some(chosen) = candidates.get(job.chosen) else {
            return Ok(());
        };
        let ctx = context(
            &job.model,
            (
                &job.blocks,
                job.prompt_tokens,
                job.block_size,
                job.session,
                job.now,
            ),
            &candidates,
        );
        let active_secs = estimate(&routing.cost, job.prompt_tokens, chosen);
        let mut picks = Vec::with_capacity(routing.shadows.len());
        for built in &routing.shadows {
            let choice = built.policy.choose(&ctx);
            let picked = choice.and_then(|c| candidates.get(c.index));
            let shadow_secs = picked.map(|c| estimate(&routing.cost, job.prompt_tokens, c));
            if let Some(shadow_secs) = shadow_secs {
                let agreed = choice.is_some_and(|c| c.index == job.chosen);
                let mut tallies = self.tallies.lock().unwrap_or_else(PoisonError::into_inner);
                let tally = tallies.entry((job.model.clone(), built.kind)).or_default();
                tally.decisions += 1;
                tally.agreed += u64::from(agreed);
                tally.faster += u64::from(shadow_secs < active_secs);
                tally.slower += u64::from(shadow_secs > active_secs);
                tally.active_secs += active_secs;
                tally.shadow_secs += shadow_secs;
                drop(tallies);
                self.metrics
                    .shadowed(&job.model, built.kind, agreed, active_secs, shadow_secs);
            }
            picks.push(Pick {
                policy: built.kind,
                replica: picked.map(|c| c.id.clone()),
                steer: choice.and_then(|c| c.steer).map(|s| s.as_str().to_owned()),
                estimated_ttft_ms: shadow_secs.map(|s| s * 1e3),
            });
        }
        if self.out.is_none() {
            return Ok(());
        }
        let line = Line {
            ts_ms: job.ts_ms,
            request_id: job.request_id.clone(),
            model: job.model.clone(),
            generation: job.live.generation,
            active: Pick {
                policy: job.active,
                replica: Some(chosen.id.clone()),
                steer: None,
                estimated_ttft_ms: Some(active_secs * 1e3),
            },
            now: job.now,
            prompt_tokens: job.prompt_tokens,
            block_size: job.block_size,
            session: job.session,
            blocks: hex(&job.blocks),
            candidates: job.seen.clone(),
            shadow: picks,
        };
        self.write(&line)
    }
}

/// How an offline replay of a shadow log compared with what was logged.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct ReplayReport {
    pub decisions: u64,
    pub generations: u64,
    /// Per policy: shadow picks replayed, and how many matched the log.
    pub policies: BTreeMap<String, (u64, u64)>,
    /// The first few picks that differed, for a person to look at.
    pub mismatches: Vec<String>,
}

impl ReplayReport {
    pub fn matched(&self) -> bool {
        self.mismatches.is_empty()
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ReplayError {
    #[error("cannot read the shadow log: {0}")]
    Io(#[from] io::Error),
    #[error("line {line}: {message}")]
    Bad { line: usize, message: String },
}

/// Rebuilds every shadow policy from the configs the log recorded and
/// feeds it the logged requests in order, checking each pick.
pub fn replay(reader: impl BufRead) -> Result<ReplayReport, ReplayError> {
    let mut report = ReplayReport::default();
    let mut live: Option<Live> = None;
    for (n, text) in reader.lines().enumerate() {
        let text = text?;
        if text.trim().is_empty() {
            continue;
        }
        let bad = |message: String| ReplayError::Bad {
            line: n + 1,
            message,
        };
        let value: serde_json::Value =
            serde_json::from_str(&text).map_err(|e| bad(e.to_string()))?;
        if value.get("shadow_log").is_some() {
            let header: Header = serde_json::from_value(value).map_err(|e| bad(e.to_string()))?;
            let config = Config::from_yaml_str(&header.config).map_err(|e| bad(e.to_string()))?;
            live = Some(Live::build(config, header.seed, live.as_ref()));
            report.generations += 1;
            continue;
        }
        let line: Line = serde_json::from_value(value).map_err(|e| bad(e.to_string()))?;
        let Some(current) = &live else {
            return Err(bad("a decision before any config".to_owned()));
        };
        let routing = current
            .routing(&line.model)
            .map_err(|_| bad(format!("unknown model {}", line.model)))?;
        let blocks = unhex(&line.blocks).ok_or_else(|| bad("unreadable blocks".to_owned()))?;
        let candidates: Vec<Candidate<'_>> = line.candidates.iter().map(Observed::view).collect();
        let ctx = context(
            &line.model,
            (
                &blocks,
                line.prompt_tokens,
                line.block_size,
                line.session,
                line.now,
            ),
            &candidates,
        );
        report.decisions += 1;
        for logged in &line.shadow {
            let built = routing.shadow(logged.policy).ok_or_else(|| {
                bad(format!(
                    "{} is not a shadow of {}",
                    logged.policy, line.model
                ))
            })?;
            let choice = built.policy.choose(&ctx);
            let replica = choice
                .and_then(|c| candidates.get(c.index))
                .map(|c| c.id.clone());
            let steer = choice.and_then(|c| c.steer).map(|s| s.as_str().to_owned());
            let entry = report
                .policies
                .entry(logged.policy.as_str().to_owned())
                .or_default();
            entry.0 += 1;
            if replica == logged.replica && steer == logged.steer {
                entry.1 += 1;
            } else if report.mismatches.len() < 20 {
                report.mismatches.push(format!(
                    "line {}: {} picked {replica:?} {steer:?}, the log has {:?} {:?}",
                    n + 1,
                    logged.policy,
                    logged.replica,
                    logged.steer
                ));
            }
        }
    }
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_hashes_survive_the_log() {
        let blocks: Vec<BlockHash> = [0u128, 1, u128::MAX, 0x1234 << 64]
            .into_iter()
            .map(BlockHash::new)
            .collect();
        assert_eq!(unhex(&hex(&blocks)).unwrap(), blocks);
        assert_eq!(unhex("").unwrap(), []);
        assert!(unhex("abc").is_none());
        assert!(unhex(&"g".repeat(32)).is_none());
    }

    #[test]
    fn replay_refuses_logs_it_cannot_follow() {
        let err = replay(io::Cursor::new("{\"ts_ms\": 1}\n")).unwrap_err();
        assert!(err.to_string().starts_with("line 1:"), "{err}");
        let err = replay(io::Cursor::new("not json\n")).unwrap_err();
        assert!(matches!(err, ReplayError::Bad { line: 1, .. }));
        let empty = replay(io::Cursor::new("\n\n")).unwrap();
        assert_eq!((empty.decisions, empty.generations), (0, 0));
        assert!(empty.matched());
    }
}
