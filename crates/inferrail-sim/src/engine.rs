//! The simulated GPU: one FIFO prefill queue on a virtual timeline, a cap on
//! running sequences, and independent per sequence decode.
//!
//! The scheduler plans each prefill to start when the previous one ends, so
//! timer overshoot never accumulates into drift. Every state a request holds
//! (waiting, running, sequence slot, KV blocks) is released by a guard, so a
//! client that disconnects frees its resources the moment its future drops.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Duration;

use tokio::sync::{OwnedSemaphorePermit, Semaphore, TryAcquireError, mpsc, oneshot};
use tokio::time::{Instant, sleep_until};

use crate::cache::{BlockHash, PrefixCache, block_hashes};
use crate::latency::LatencyModel;

#[derive(Debug, Clone, PartialEq)]
pub struct EngineConfig {
    pub block_size: usize,
    pub num_gpu_blocks: usize,
    pub max_num_seqs: usize,
    pub latency: LatencyModel,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            block_size: 16,
            num_gpu_blocks: 4096,
            max_num_seqs: 256,
            latency: LatencyModel::default(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("engine scheduler has stopped")]
pub struct EngineStopped;

#[derive(Debug, Default)]
pub struct EngineCounters {
    pub waiting: AtomicU64,
    pub running: AtomicU64,
    pub kv_blocks_used: AtomicU64,
    pub prefix_queries: AtomicU64,
    pub prefix_hits: AtomicU64,
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct EngineSnapshot {
    pub waiting: u64,
    pub running: u64,
    pub kv_blocks_used: u64,
    pub kv_usage: f64,
    pub cached_blocks: usize,
    pub prefix_queries: u64,
    pub prefix_hits: u64,
}

#[derive(Debug)]
struct Inner {
    config: EngineConfig,
    cache: Mutex<PrefixCache>,
    slots: Arc<Semaphore>,
    counters: EngineCounters,
}

impl Inner {
    fn cache(&self) -> std::sync::MutexGuard<'_, PrefixCache> {
        self.cache.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

#[derive(Debug, Clone)]
pub struct Engine {
    inner: Arc<Inner>,
    jobs: mpsc::UnboundedSender<Job>,
}

#[derive(Debug)]
struct Job {
    hashes: Vec<BlockHash>,
    prompt_len: usize,
    arrival: Instant,
    reply: oneshot::Sender<Admission>,
}

#[derive(Debug)]
struct Admission {
    cached_tokens: usize,
    scheduled: Instant,
    first_token_at: Instant,
    permit: OwnedSemaphorePermit,
}

impl Engine {
    /// Starts the scheduler on the current Tokio runtime.
    pub fn start(config: EngineConfig) -> Self {
        let inner = Arc::new(Inner {
            cache: Mutex::new(PrefixCache::new(config.num_gpu_blocks)),
            slots: Arc::new(Semaphore::new(config.max_num_seqs.max(1))),
            counters: EngineCounters::default(),
            config,
        });
        let (jobs, rx) = mpsc::unbounded_channel();
        tokio::spawn(schedule(Arc::clone(&inner), rx));
        Self { inner, jobs }
    }

    pub fn config(&self) -> &EngineConfig {
        &self.inner.config
    }

    pub fn counters(&self) -> &EngineCounters {
        &self.inner.counters
    }

    /// Queues a prompt. It counts as waiting until the scheduler admits it.
    pub fn submit(&self, prompt: &[u32]) -> Ticket {
        let hashes = block_hashes(prompt, self.inner.config.block_size);
        let waiting = GaugeGuard::new(Arc::clone(&self.inner), |c| &c.waiting);
        let (reply, rx) = oneshot::channel();
        let arrival = Instant::now();
        let job = Job {
            hashes,
            prompt_len: prompt.len(),
            arrival,
            reply,
        };
        let queued = self.jobs.send(job).is_ok();
        Ticket {
            rx,
            queued,
            prompt_len: prompt.len(),
            arrival,
            _waiting: waiting,
            inner: Arc::clone(&self.inner),
        }
    }

    pub fn reset_prefix_cache(&self) {
        self.inner.cache().clear();
    }

    pub fn snapshot(&self) -> EngineSnapshot {
        let c = &self.inner.counters;
        let kv_blocks_used = c.kv_blocks_used.load(Ordering::Relaxed);
        let total = self.inner.config.num_gpu_blocks.max(1) as f64;
        EngineSnapshot {
            waiting: c.waiting.load(Ordering::Relaxed),
            running: c.running.load(Ordering::Relaxed),
            kv_blocks_used,
            kv_usage: (kv_blocks_used as f64 / total).min(1.0),
            cached_blocks: self.inner.cache().len(),
            prefix_queries: c.prefix_queries.load(Ordering::Relaxed),
            prefix_hits: c.prefix_hits.load(Ordering::Relaxed),
        }
    }
}

async fn schedule(inner: Arc<Inner>, mut jobs: mpsc::UnboundedReceiver<Job>) {
    let mut busy_until = Instant::now();
    let block_size = inner.config.block_size;
    let latency = inner.config.latency;
    while let Some(mut job) = jobs.recv().await {
        let permit = match Arc::clone(&inner.slots).try_acquire_owned() {
            Ok(permit) => permit,
            Err(TryAcquireError::Closed) => return,
            Err(TryAcquireError::NoPermits) => {
                tokio::select! {
                    permit = Arc::clone(&inner.slots).acquire_owned() => {
                        let Ok(permit) = permit else { return };
                        busy_until = busy_until.max(Instant::now());
                        permit
                    }
                    () = job.reply.closed() => continue,
                }
            }
        };
        let start = busy_until.max(job.arrival);
        if start > Instant::now() {
            tokio::select! {
                () = sleep_until(start) => {}
                () = job.reply.closed() => continue,
            }
        } else if job.reply.is_closed() {
            continue;
        }

        let cacheable = job.prompt_len.saturating_sub(1) / block_size;
        let cached_tokens = {
            let mut cache = inner.cache();
            let matched = cache.match_prefix(&job.hashes[..cacheable.min(job.hashes.len())]);
            cache.insert(&job.hashes);
            matched * block_size
        };
        let uncached = job.prompt_len - cached_tokens;
        busy_until = start + latency.prefill(uncached as u64, cached_tokens as u64);

        let counters = &inner.counters;
        counters
            .prefix_queries
            .fetch_add(job.prompt_len as u64, Ordering::Relaxed);
        counters
            .prefix_hits
            .fetch_add(cached_tokens as u64, Ordering::Relaxed);

        let _ = job.reply.send(Admission {
            cached_tokens,
            scheduled: start,
            first_token_at: busy_until,
            permit,
        });
    }
}

/// A queued request; dropping it withdraws the request from the queue.
#[derive(Debug)]
pub struct Ticket {
    rx: oneshot::Receiver<Admission>,
    queued: bool,
    prompt_len: usize,
    arrival: Instant,
    _waiting: GaugeGuard,
    inner: Arc<Inner>,
}

impl Ticket {
    pub async fn admitted(self) -> Result<Sequence, EngineStopped> {
        if !self.queued {
            return Err(EngineStopped);
        }
        let admission = self.rx.await.map_err(|_| EngineStopped)?;
        let block_size = self.inner.config.block_size as u64;
        let held_blocks = (self.prompt_len as u64).div_ceil(block_size);
        self.inner
            .counters
            .kv_blocks_used
            .fetch_add(held_blocks, Ordering::Relaxed);
        Ok(Sequence {
            prompt_tokens: self.prompt_len,
            cached_tokens: admission.cached_tokens,
            arrival: self.arrival,
            scheduled: admission.scheduled,
            first_token_at: admission.first_token_at,
            decode_step: self.inner.config.latency.decode_step(),
            generated: 0,
            held_blocks,
            _running: GaugeGuard::new(Arc::clone(&self.inner), |c| &c.running),
            _permit: admission.permit,
            inner: self.inner,
        })
    }
}

/// A running request. Dropping it frees its slot and KV blocks.
#[derive(Debug)]
pub struct Sequence {
    pub prompt_tokens: usize,
    pub cached_tokens: usize,
    pub arrival: Instant,
    pub scheduled: Instant,
    pub first_token_at: Instant,
    decode_step: Duration,
    generated: usize,
    held_blocks: u64,
    _running: GaugeGuard,
    _permit: OwnedSemaphorePermit,
    inner: Arc<Inner>,
}

impl Sequence {
    pub fn generated(&self) -> usize {
        self.generated
    }

    pub fn deadline(&self, index: usize) -> Instant {
        self.first_token_at + self.decode_step.saturating_mul(index as u32)
    }

    /// Waits for the next token's planned time and accounts for its KV slot.
    pub async fn next_token(&mut self) {
        let deadline = self.deadline(self.generated);
        if deadline > Instant::now() {
            sleep_until(deadline).await;
        }
        self.generated += 1;
        let block_size = self.inner.config.block_size as u64;
        let needed = ((self.prompt_tokens + self.generated) as u64).div_ceil(block_size);
        if needed > self.held_blocks {
            self.inner
                .counters
                .kv_blocks_used
                .fetch_add(needed - self.held_blocks, Ordering::Relaxed);
            self.held_blocks = needed;
        }
    }
}

impl Drop for Sequence {
    fn drop(&mut self) {
        self.inner
            .counters
            .kv_blocks_used
            .fetch_sub(self.held_blocks, Ordering::Relaxed);
    }
}

#[derive(Debug)]
struct GaugeGuard {
    inner: Arc<Inner>,
    gauge: fn(&EngineCounters) -> &AtomicU64,
}

impl GaugeGuard {
    fn new(inner: Arc<Inner>, gauge: fn(&EngineCounters) -> &AtomicU64) -> Self {
        gauge(&inner.counters).fetch_add(1, Ordering::Relaxed);
        Self { inner, gauge }
    }
}

impl Drop for GaugeGuard {
    fn drop(&mut self) {
        (self.gauge)(&self.inner.counters).fetch_sub(1, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(max_num_seqs: usize) -> EngineConfig {
        EngineConfig {
            block_size: 16,
            num_gpu_blocks: 1024,
            max_num_seqs,
            latency: LatencyModel {
                prefill_token_secs: 1e-3,
                prefill_pair_secs: 0.0,
                prefill_fixed_secs: 0.0,
                decode_token_secs: 10e-3,
            },
        }
    }

    fn prompt(seed: u32, len: usize) -> Vec<u32> {
        (0..len as u32)
            .map(|i| i.wrapping_mul(31).wrapping_add(seed))
            .collect()
    }

    async fn run(engine: &Engine, tokens: &[u32], output: usize) -> (usize, Duration) {
        let mut seq = engine.submit(tokens).admitted().await.unwrap();
        for _ in 0..output {
            seq.next_token().await;
        }
        (seq.cached_tokens, seq.first_token_at - seq.arrival)
    }

    #[tokio::test(start_paused = true)]
    async fn repeated_prompt_hits_cache_and_cuts_ttft() {
        let engine = Engine::start(config(8));
        let tokens = prompt(1, 1_000);
        let (cold_hit, cold_ttft) = run(&engine, &tokens, 1).await;
        let (warm_hit, warm_ttft) = run(&engine, &tokens, 1).await;
        assert_eq!(cold_hit, 0);
        assert_eq!(warm_hit, 992);
        assert_eq!(cold_ttft, Duration::from_secs(1));
        assert_eq!(warm_ttft, Duration::from_millis(8));
        let snap = engine.snapshot();
        assert_eq!(snap.prefix_queries, 2_000);
        assert_eq!(snap.prefix_hits, 992);
    }

    #[tokio::test(start_paused = true)]
    async fn fully_cached_prompt_still_computes_its_last_block() {
        let engine = Engine::start(config(8));
        let tokens = prompt(2, 64);
        run(&engine, &tokens, 0).await;
        let (hit, _) = run(&engine, &tokens, 0).await;
        assert_eq!(hit, 48);
    }

    #[tokio::test(start_paused = true)]
    async fn prefills_queue_back_to_back() {
        let engine = Engine::start(config(8));
        let tickets: Vec<_> = (0..3)
            .map(|i| engine.submit(&prompt(10 + i, 100)))
            .collect();
        assert_eq!(engine.snapshot().waiting, 3);
        let mut seqs = Vec::new();
        for ticket in tickets {
            seqs.push(ticket.admitted().await.unwrap());
        }
        assert_eq!(seqs[1].scheduled, seqs[0].first_token_at);
        assert_eq!(seqs[2].scheduled, seqs[1].first_token_at);
        assert_eq!(
            seqs[2].first_token_at - seqs[0].arrival,
            Duration::from_millis(300)
        );
        let snap = engine.snapshot();
        assert_eq!((snap.waiting, snap.running), (0, 3));
    }

    #[tokio::test(start_paused = true)]
    async fn ttft_grows_with_queue_depth() {
        let mut previous = Duration::ZERO;
        for depth in 0..6u32 {
            let engine = Engine::start(config(64));
            let queued: Vec<_> = (0..depth)
                .map(|i| engine.submit(&prompt(100 + i, 200)))
                .collect();
            let probe = engine.submit(&prompt(999, 200));
            let mut seqs = Vec::new();
            for ticket in queued {
                seqs.push(ticket.admitted().await.unwrap());
            }
            let probe = probe.admitted().await.unwrap();
            let ttft = probe.first_token_at - probe.arrival;
            assert!(ttft > previous, "depth {depth}: {ttft:?} <= {previous:?}");
            previous = ttft;
        }
    }

    #[tokio::test(start_paused = true)]
    async fn sequence_cap_holds_requests_in_waiting() {
        let engine = Engine::start(config(1));
        let first = engine.submit(&prompt(1, 32)).admitted().await.unwrap();
        let second = engine.submit(&prompt(2, 32));
        tokio::time::sleep(Duration::from_secs(5)).await;
        assert_eq!(engine.snapshot().waiting, 1);
        assert_eq!(engine.snapshot().running, 1);
        let released = Instant::now();
        drop(first);
        let second = second.admitted().await.unwrap();
        assert!(second.scheduled >= released);
        assert_eq!(engine.snapshot().waiting, 0);
    }

    #[tokio::test(start_paused = true)]
    async fn withdrawn_ticket_is_skipped_and_frees_the_gpu() {
        let engine = Engine::start(config(8));
        let busy = engine.submit(&prompt(1, 1_000));
        let abandoned = engine.submit(&prompt(2, 1_000));
        let next = engine.submit(&prompt(3, 100));
        drop(abandoned);
        assert_eq!(engine.snapshot().waiting, 2);
        let busy = busy.admitted().await.unwrap();
        let next = next.admitted().await.unwrap();
        assert_eq!(next.scheduled, busy.first_token_at);
    }

    #[tokio::test(start_paused = true)]
    async fn decode_follows_planned_deadlines_and_tracks_kv_blocks() {
        let engine = Engine::start(config(8));
        let mut seq = engine.submit(&prompt(5, 30)).admitted().await.unwrap();
        assert_eq!(engine.snapshot().kv_blocks_used, 2);
        for _ in 0..3 {
            seq.next_token().await;
        }
        assert_eq!(seq.generated(), 3);
        assert_eq!(Instant::now(), seq.deadline(2));
        assert_eq!(
            seq.deadline(2) - seq.first_token_at,
            Duration::from_millis(20)
        );
        assert_eq!(engine.snapshot().kv_blocks_used, 3);
        drop(seq);
        let snap = engine.snapshot();
        assert_eq!((snap.kv_blocks_used, snap.running, snap.waiting), (0, 0, 0));
    }

    #[tokio::test(start_paused = true)]
    async fn reset_prefix_cache_forgets_everything() {
        let engine = Engine::start(config(8));
        let tokens = prompt(9, 256);
        run(&engine, &tokens, 0).await;
        assert!(engine.snapshot().cached_blocks > 0);
        engine.reset_prefix_cache();
        assert_eq!(engine.snapshot().cached_blocks, 0);
        assert_eq!(run(&engine, &tokens, 0).await.0, 0);
    }

    #[tokio::test]
    async fn zero_latency_mode_never_sleeps() {
        let engine = Engine::start(EngineConfig {
            latency: LatencyModel::ZERO,
            ..EngineConfig::default()
        });
        let started = std::time::Instant::now();
        for i in 0..200 {
            run(&engine, &prompt(i, 512), 32).await;
        }
        assert!(started.elapsed() < Duration::from_secs(2));
    }
}
