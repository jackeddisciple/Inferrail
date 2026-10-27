//! The simulated GPU, scheduled the way vLLM v0.31.0 schedules one. Every
//! step runs one batch within a token budget: running requests first, each
//! decoding one token or computing the next chunk of its prompt, then
//! waiting requests in arrival order while sequences and budget last. All
//! of them share one pool of KV blocks. A running request that needs a
//! block when none is free preempts the newest running request, which gives
//! its blocks back and starts over from the front of the queue.
//!
//! Steps are planned on a virtual timeline: each starts when the previous
//! one ends, so timer overshoot never accumulates into drift. A client that
//! disconnects withdraws its request at once while it waits, and at the
//! next step boundary once it runs, as vLLM aborts between steps.

use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicU8, AtomicU64, Ordering};
use std::time::Duration;

use tokio::sync::{mpsc, oneshot};
use tokio::time::{Instant, sleep_until};

use crate::cache::{BlockHash, block_hashes, extend_hashes};
use crate::latency::{LatencyModel, duration};
use crate::pool::{BlockId, BlockPool};

#[derive(Debug, Clone, PartialEq)]
pub struct EngineConfig {
    pub block_size: usize,
    pub num_gpu_blocks: usize,
    pub max_num_seqs: usize,
    /// Tokens one step may compute, as vLLM's `max_num_batched_tokens`.
    pub max_num_batched_tokens: usize,
    pub enable_prefix_caching: bool,
    pub latency: LatencyModel,
}

impl EngineConfig {
    pub fn validate(&self) -> Result<(), String> {
        if self.block_size == 0 {
            return Err("block_size must be positive".to_owned());
        }
        if self.max_num_seqs == 0 {
            return Err("max_num_seqs must be positive".to_owned());
        }
        if self.max_num_batched_tokens < self.max_num_seqs {
            return Err(format!(
                "max_num_batched_tokens ({}) must be greater than or equal to max_num_seqs ({})",
                self.max_num_batched_tokens, self.max_num_seqs
            ));
        }
        self.latency.validate()
    }

    /// The most tokens the KV cache can hold for one request.
    pub fn kv_tokens(&self) -> usize {
        self.num_gpu_blocks.saturating_mul(self.block_size)
    }
}

impl Default for EngineConfig {
    /// vLLM's defaults for an API server on a GPU under 70 GiB.
    fn default() -> Self {
        Self {
            block_size: 16,
            num_gpu_blocks: 4096,
            max_num_seqs: 256,
            max_num_batched_tokens: 2048,
            enable_prefix_caching: true,
            latency: LatencyModel::default(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum EngineError {
    #[error("engine scheduler has stopped")]
    Stopped,
    #[error("the request needs more KV blocks than the engine has")]
    TooLong,
    #[error("the request asks for no output")]
    NoOutput,
}

#[derive(Debug, Default)]
struct Counters {
    waiting: AtomicU64,
    running: AtomicU64,
    kv_blocks_used: AtomicU64,
    cached_blocks: AtomicU64,
    prefix_queries: AtomicU64,
    prefix_hits: AtomicU64,
    preemptions: AtomicU64,
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct EngineSnapshot {
    pub waiting: u64,
    pub running: u64,
    /// KV blocks requests hold; a block shared by several counts once.
    pub kv_blocks_used: u64,
    pub kv_usage: f64,
    /// Blocks holding a hash, so able to serve a prefix hit.
    pub cached_blocks: usize,
    pub prefix_queries: u64,
    pub prefix_hits: u64,
    pub preemptions: u64,
}

#[derive(Debug)]
struct Shared {
    config: EngineConfig,
    counters: Counters,
}

impl Shared {
    fn add(&self, gauge: fn(&Counters) -> &AtomicU64, n: u64) {
        gauge(&self.counters).fetch_add(n, Ordering::Relaxed);
    }

    fn sub(&self, gauge: fn(&Counters) -> &AtomicU64, n: u64) {
        gauge(&self.counters).fetch_sub(n, Ordering::Relaxed);
    }
}

const WAITING: u8 = 0;
const RUNNING: u8 = 1;
const DONE: u8 = 2;
const CANCELLED: u8 = 4;

/// Where a request is, shared between the engine and its client. The
/// client only ever sets `CANCELLED`; whoever moves a request out of
/// waiting owns the waiting gauge's decrement.
#[derive(Debug)]
struct Status(AtomicU8);

impl Status {
    fn cancelled(&self) -> bool {
        self.0.load(Ordering::Acquire) & CANCELLED != 0
    }

    fn advance(&self, from: u8, to: u8) -> bool {
        self.0
            .compare_exchange(from, to, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
    }

    fn finish(&self) {
        self.0.fetch_or(DONE, Ordering::AcqRel);
    }
}

/// What the engine tells a request's client.
#[derive(Debug)]
enum Emit {
    Scheduled { at: Instant, cached_tokens: usize },
    Token { at: Instant, preemptions: u32 },
}

#[derive(Debug)]
enum Message {
    Submit(Box<Job>),
    Reset(oneshot::Sender<()>),
}

#[derive(Debug)]
struct Job {
    prompt: Vec<u32>,
    hashes: Vec<BlockHash>,
    output: Vec<u32>,
    arrival: Instant,
    status: Arc<Status>,
    emit: mpsc::UnboundedSender<Emit>,
}

#[derive(Debug, Clone)]
pub struct Engine {
    shared: Arc<Shared>,
    messages: mpsc::UnboundedSender<Message>,
}

impl Engine {
    /// Starts the scheduler on the current Tokio runtime.
    pub fn start(config: EngineConfig) -> Self {
        let shared = Arc::new(Shared {
            config,
            counters: Counters::default(),
        });
        let (messages, rx) = mpsc::unbounded_channel();
        tokio::spawn(run(Scheduler::new(Arc::clone(&shared)), rx));
        Self { shared, messages }
    }

    pub fn config(&self) -> &EngineConfig {
        &self.shared.config
    }

    /// Queues a prompt that will generate `output`. It counts as waiting
    /// until the scheduler first runs it.
    pub fn submit(&self, prompt: &[u32], output: Vec<u32>) -> Ticket {
        self.submit_at(prompt, output, Instant::now())
    }

    fn submit_at(&self, prompt: &[u32], output: Vec<u32>, arrival: Instant) -> Ticket {
        let block_size = self.shared.config.block_size;
        let status = Arc::new(Status(AtomicU8::new(WAITING)));
        let (emit, rx) = mpsc::unbounded_channel();
        let held = (prompt.len() + output.len().saturating_sub(1)).div_ceil(block_size);
        let failure = if output.is_empty() {
            Some(EngineError::NoOutput)
        } else if held > self.shared.config.num_gpu_blocks {
            Some(EngineError::TooLong)
        } else {
            None
        };
        self.shared.add(|c| &c.waiting, 1);
        let guard = Guard {
            status: Arc::clone(&status),
            shared: Arc::clone(&self.shared),
        };
        let failure = failure.or_else(|| {
            let job = Job {
                prompt: prompt.to_vec(),
                hashes: block_hashes(prompt, block_size),
                output,
                arrival,
                status,
                emit,
            };
            let sent = self.messages.send(Message::Submit(Box::new(job)));
            sent.is_err().then_some(EngineError::Stopped)
        });
        Ticket {
            rx,
            failure,
            prompt_tokens: prompt.len(),
            arrival,
            guard,
        }
    }

    /// Forgets every cached block, between steps.
    pub async fn reset_prefix_cache(&self) {
        let (done, rx) = oneshot::channel();
        if self.messages.send(Message::Reset(done)).is_ok() {
            let _ = rx.await;
        }
    }

    pub fn snapshot(&self) -> EngineSnapshot {
        let c = &self.shared.counters;
        let load = |a: &AtomicU64| a.load(Ordering::Relaxed);
        let kv_blocks_used = load(&c.kv_blocks_used);
        let total = self.shared.config.num_gpu_blocks.max(1) as f64;
        EngineSnapshot {
            waiting: load(&c.waiting),
            running: load(&c.running),
            kv_blocks_used,
            kv_usage: (kv_blocks_used as f64 / total).min(1.0),
            cached_blocks: load(&c.cached_blocks) as usize,
            prefix_queries: load(&c.prefix_queries),
            prefix_hits: load(&c.prefix_hits),
            preemptions: load(&c.preemptions),
        }
    }
}

/// Withdraws the request when its client goes away.
#[derive(Debug)]
struct Guard {
    status: Arc<Status>,
    shared: Arc<Shared>,
}

impl Drop for Guard {
    fn drop(&mut self) {
        let before = self.status.0.fetch_or(CANCELLED, Ordering::AcqRel);
        if before == WAITING {
            self.shared.sub(|c| &c.waiting, 1);
        }
    }
}

/// A queued request; dropping it withdraws the request.
#[derive(Debug)]
pub struct Ticket {
    rx: mpsc::UnboundedReceiver<Emit>,
    failure: Option<EngineError>,
    prompt_tokens: usize,
    arrival: Instant,
    guard: Guard,
}

impl Ticket {
    /// Waits until the scheduler first runs the request.
    pub async fn admitted(mut self) -> Result<Sequence, EngineError> {
        if let Some(err) = self.failure {
            return Err(err);
        }
        let Some(Emit::Scheduled { at, cached_tokens }) = self.rx.recv().await else {
            return Err(EngineError::Stopped);
        };
        Ok(Sequence {
            prompt_tokens: self.prompt_tokens,
            cached_tokens,
            arrival: self.arrival,
            scheduled: at,
            first_token_at: None,
            generated: 0,
            preemptions: 0,
            rx: self.rx,
            _guard: self.guard,
        })
    }
}

/// A running request. Dropping it aborts the request.
#[derive(Debug)]
pub struct Sequence {
    pub prompt_tokens: usize,
    pub cached_tokens: usize,
    pub arrival: Instant,
    /// When the scheduler first ran it.
    pub scheduled: Instant,
    first_token_at: Option<Instant>,
    generated: usize,
    preemptions: u32,
    rx: mpsc::UnboundedReceiver<Emit>,
    _guard: Guard,
}

impl Sequence {
    pub fn generated(&self) -> usize {
        self.generated
    }

    pub fn first_token_at(&self) -> Option<Instant> {
        self.first_token_at
    }

    /// Times the request went back to the queue to make room for others.
    pub fn preemptions(&self) -> u32 {
        self.preemptions
    }

    /// The time the engine produced the next token, or `None` once every
    /// token is out or the engine stopped. The engine has freed the
    /// request's blocks by the time its last token arrives.
    pub async fn next_token(&mut self) -> Option<Instant> {
        loop {
            match self.rx.recv().await? {
                Emit::Token { at, preemptions } => {
                    self.generated += 1;
                    self.preemptions = preemptions;
                    self.first_token_at.get_or_insert(at);
                    return Some(at);
                }
                Emit::Scheduled { .. } => {}
            }
        }
    }
}

/// A request inside the scheduler: its prompt and the tokens sampled so
/// far, the hashes of its full blocks and the blocks it holds.
#[derive(Debug)]
struct Request {
    tokens: Vec<u32>,
    hashes: Vec<BlockHash>,
    output: Vec<u32>,
    generated: usize,
    blocks: Vec<BlockId>,
    computed: usize,
    cached_upto: usize,
    first_admission: bool,
    preemptions: u32,
    arrival: Instant,
    status: Arc<Status>,
    emit: mpsc::UnboundedSender<Emit>,
}

impl Request {
    fn len(&self) -> usize {
        self.tokens.len()
    }
}

/// One request's share of a step.
#[derive(Debug, Clone, Copy)]
struct Work {
    index: usize,
    tokens: usize,
    context: usize,
    admitted: bool,
}

#[derive(Debug)]
struct Scheduler {
    shared: Arc<Shared>,
    pool: BlockPool,
    running: Vec<Request>,
    waiting: VecDeque<Request>,
    batch: Vec<Work>,
    hit: Vec<BlockId>,
    evicted: Vec<BlockHash>,
}

/// How far behind the wall clock planned steps may fall before the
/// timeline jumps forward rather than replay the backlog.
const MAX_LAG: Duration = Duration::from_millis(5);

/// How long a queue that cannot run yet waits before it is planned again.
const IDLE_RETRY: Duration = Duration::from_millis(1);

impl Scheduler {
    fn new(shared: Arc<Shared>) -> Self {
        let config = &shared.config;
        Self {
            pool: BlockPool::new(config.num_gpu_blocks, config.enable_prefix_caching),
            running: Vec::new(),
            waiting: VecDeque::new(),
            batch: Vec::new(),
            hit: Vec::new(),
            evicted: Vec::new(),
            shared,
        }
    }

    fn config(&self) -> &EngineConfig {
        &self.shared.config
    }

    fn is_idle(&self) -> bool {
        self.running.is_empty() && self.waiting.is_empty()
    }

    fn accept(&mut self, message: Message) {
        match message {
            Message::Submit(job) => {
                let job = *job;
                self.waiting.push_back(Request {
                    tokens: job.prompt,
                    hashes: job.hashes,
                    output: job.output,
                    generated: 0,
                    blocks: Vec::new(),
                    computed: 0,
                    cached_upto: 0,
                    first_admission: true,
                    preemptions: 0,
                    arrival: job.arrival,
                    status: job.status,
                    emit: job.emit,
                });
            }
            Message::Reset(done) => {
                self.pool.reset();
                for request in &mut self.running {
                    request.cached_upto = request.blocks.len();
                }
                self.publish_pool();
                let _ = done.send(());
            }
        }
    }

    /// When the next step starts: after the last one, and never before the
    /// request at the head of the queue arrived. Clients stamp arrivals
    /// before they queue, so on several threads the queue can hold them
    /// out of order; an idle engine still starts with its head.
    fn start(&self, previous_end: Instant) -> Instant {
        if self.config().latency.is_zero() {
            return Instant::now();
        }
        if self.running.is_empty() {
            let head = self.waiting.front().map(|r| r.arrival);
            return head.map_or(previous_end, |at| previous_end.max(at));
        }
        let now = Instant::now();
        if now.saturating_duration_since(previous_end) > MAX_LAG {
            now
        } else {
            previous_end
        }
    }

    fn release(&mut self, request: &Request) {
        self.pool.free(&request.blocks);
    }

    fn drop_cancelled(&mut self) {
        let mut i = 0;
        while i < self.running.len() {
            if self.running[i].status.cancelled() {
                let request = self.running.remove(i);
                self.release(&request);
                request.status.finish();
                self.shared.sub(|c| &c.running, 1);
            } else {
                i += 1;
            }
        }
        self.waiting.retain(|r| !r.status.cancelled());
    }

    fn blocks_for(&self, tokens: usize) -> usize {
        tokens.div_ceil(self.config().block_size)
    }

    /// Hashes the request's full blocks up to `tokens` into the pool, as
    /// vLLM caches blocks when it schedules their computation.
    fn cache_upto(&mut self, index: usize, tokens: usize) {
        let block_size = self.config().block_size;
        let request = &mut self.running[index];
        let full = tokens.min(request.len()) / block_size;
        for i in request.cached_upto..full {
            self.pool.cache(request.blocks[i], request.hashes[i]);
        }
        request.cached_upto = request.cached_upto.max(full);
    }

    /// Puts the newest running request back at the front of the queue.
    fn preempt_last(&mut self) {
        let Some(mut victim) = self.running.pop() else {
            return;
        };
        self.release(&victim);
        victim.blocks.clear();
        victim.computed = 0;
        victim.cached_upto = 0;
        victim.preemptions += 1;
        self.shared.add(|c| &c.preemptions, 1);
        self.shared.sub(|c| &c.running, 1);
        self.shared.add(|c| &c.waiting, 1);
        if victim.status.advance(RUNNING, WAITING) {
            self.waiting.push_front(victim);
        } else {
            self.shared.sub(|c| &c.waiting, 1);
            victim.status.finish();
        }
    }

    /// Picks the step's batch; false when nothing can run.
    fn plan(&mut self, start: Instant) -> bool {
        self.drop_cancelled();
        self.batch.clear();
        let mut budget = self.config().max_num_batched_tokens;
        let mut preempted = false;
        let mut i = 0;
        while i < self.running.len() && budget > 0 {
            let request = &self.running[i];
            let tokens = (request.len() - request.computed).min(budget);
            let context = request.computed;
            let need = self
                .blocks_for(context + tokens)
                .saturating_sub(request.blocks.len());
            while !self.pool.fits(&[], need) && i < self.running.len() {
                self.preempt_last();
                preempted = true;
            }
            if i >= self.running.len() {
                break;
            }
            let request = &mut self.running[i];
            self.pool
                .allocate(need, &mut request.blocks, &mut self.evicted);
            self.cache_upto(i, context + tokens);
            self.batch.push(Work {
                index: i,
                tokens,
                context,
                admitted: false,
            });
            budget -= tokens;
            i += 1;
        }
        if !preempted {
            self.admit(start, &mut budget);
        }
        self.evicted.clear();
        self.publish_pool();
        !self.batch.is_empty()
    }

    fn admit(&mut self, start: Instant, budget: &mut usize) {
        let block_size = self.config().block_size;
        while *budget > 0 && self.running.len() < self.config().max_num_seqs {
            let Some(request) = self.waiting.front() else {
                break;
            };
            if request.arrival > start {
                break;
            }
            let reusable = (request.len() - 1) / block_size;
            self.pool.longest_hit(
                &request.hashes[..reusable.min(request.hashes.len())],
                &mut self.hit,
            );
            let context = self.hit.len() * block_size;
            let tokens = (request.len() - context).min(*budget);
            let need = self.blocks_for(context + tokens) - self.hit.len();
            if !self.pool.fits(&self.hit, need) {
                break;
            }
            let Some(mut request) = self.waiting.pop_front() else {
                break;
            };
            if !request.status.advance(WAITING, RUNNING) {
                continue;
            }
            self.shared.add(|c| &c.running, 1);
            self.shared.sub(|c| &c.waiting, 1);
            self.pool.touch(&self.hit);
            request.blocks.extend_from_slice(&self.hit);
            self.pool
                .allocate(need, &mut request.blocks, &mut self.evicted);
            request.computed = context;
            request.cached_upto = self.hit.len();
            if request.first_admission {
                request.first_admission = false;
                self.shared.add(|c| &c.prefix_queries, request.len() as u64);
                self.shared.add(|c| &c.prefix_hits, context as u64);
                let _ = request.emit.send(Emit::Scheduled {
                    at: start,
                    cached_tokens: context,
                });
            }
            self.running.push(request);
            let index = self.running.len() - 1;
            self.cache_upto(index, context + tokens);
            self.batch.push(Work {
                index,
                tokens,
                context,
                admitted: true,
            });
            *budget -= tokens;
        }
    }

    fn step_secs(&self) -> f64 {
        let latency = &self.config().latency;
        let mut decodes = 0u64;
        let mut secs = 0.0;
        for work in &self.batch {
            let request = &self.running[work.index];
            let decode = !work.admitted
                && request.generated > 0
                && work.tokens == 1
                && work.context + 1 == request.len();
            if decode {
                decodes += 1;
            } else {
                secs += latency.chunk_secs(work.tokens as u64, work.context as u64);
                if work.admitted {
                    secs += latency.prefill_fixed_secs;
                }
            }
        }
        secs + latency.decode_secs(decodes)
    }

    /// Applies the step that ended at `end`: requests whose tokens are all
    /// computed sample their next one, and finished ones let go of their
    /// blocks before their last token goes out.
    fn complete(&mut self, end: Instant) {
        let block_size = self.config().block_size;
        let mut finished = false;
        for k in 0..self.batch.len() {
            let Work { index, tokens, .. } = self.batch[k];
            let request = &mut self.running[index];
            request.computed += tokens;
            if request.computed < request.len() {
                continue;
            }
            let id = request.output[request.generated];
            request.generated += 1;
            request.tokens.push(id);
            extend_hashes(&request.tokens, block_size, &mut request.hashes);
            let at = end.max(request.arrival);
            let preemptions = request.preemptions;
            if request.generated == request.output.len() {
                self.pool.free(&request.blocks);
                request.blocks.clear();
                request.status.finish();
                self.shared.sub(|c| &c.running, 1);
                finished = true;
            }
            let request = &self.running[index];
            if request.emit.send(Emit::Token { at, preemptions }).is_err() {
                request.status.0.fetch_or(CANCELLED, Ordering::AcqRel);
            }
        }
        if finished {
            self.running
                .retain(|r| r.status.0.load(Ordering::Acquire) & DONE == 0);
        }
        self.publish_pool();
    }

    fn publish_pool(&self) {
        let c = &self.shared.counters;
        c.kv_blocks_used
            .store(self.pool.used_blocks() as u64, Ordering::Relaxed);
        c.cached_blocks
            .store(self.pool.cached_blocks() as u64, Ordering::Relaxed);
    }
}

async fn run(mut scheduler: Scheduler, mut messages: mpsc::UnboundedReceiver<Message>) {
    let mut previous_end = Instant::now();
    loop {
        if scheduler.is_idle() {
            match messages.recv().await {
                Some(message) => scheduler.accept(message),
                None => return,
            }
        }
        while let Ok(message) = messages.try_recv() {
            scheduler.accept(message);
        }
        let start = scheduler.start(previous_end);
        if !scheduler.plan(start) {
            if !scheduler.is_idle() {
                let wait = tokio::time::timeout(IDLE_RETRY, messages.recv());
                if let Ok(Some(message)) = wait.await {
                    scheduler.accept(message);
                }
            }
            continue;
        }
        let end = start + duration(scheduler.step_secs());
        if end > Instant::now() {
            sleep_until(end).await;
        } else {
            tokio::task::consume_budget().await;
        }
        scheduler.complete(end);
        previous_end = end;
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
            max_num_batched_tokens: 2048,
            enable_prefix_caching: true,
            latency: LatencyModel {
                prefill_token_secs: 1e-3,
                prefill_pair_secs: 0.0,
                prefill_fixed_secs: 0.0,
                decode_token_secs: 10e-3,
                decode_seq_secs: 0.0,
            },
        }
    }

    fn prompt(seed: u32, len: usize) -> Vec<u32> {
        (0..len as u32)
            .map(|i| i.wrapping_mul(31).wrapping_add(seed))
            .collect()
    }

    fn output(n: usize) -> Vec<u32> {
        (0..n as u32).map(|i| 50_000 + i).collect()
    }

    /// Runs one request to the end: cached tokens, TTFT and token times.
    async fn run(engine: &Engine, tokens: &[u32], n: usize) -> (usize, Duration, Vec<Instant>) {
        let mut seq = engine.submit(tokens, output(n)).admitted().await.unwrap();
        let mut times = Vec::new();
        while let Some(at) = seq.next_token().await {
            times.push(at);
        }
        assert_eq!(times.len(), n);
        (seq.cached_tokens, times[0] - seq.arrival, times)
    }

    #[tokio::test(start_paused = true)]
    async fn repeated_prompt_hits_cache_and_cuts_ttft() {
        let engine = Engine::start(config(8));
        let tokens = prompt(1, 1_000);
        let (cold_hit, cold_ttft, _) = run(&engine, &tokens, 1).await;
        let (warm_hit, warm_ttft, _) = run(&engine, &tokens, 1).await;
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
        run(&engine, &tokens, 1).await;
        let (hit, _, _) = run(&engine, &tokens, 1).await;
        assert_eq!(hit, 48);
    }

    #[tokio::test(start_paused = true)]
    async fn long_prompts_prefill_in_chunks_within_the_budget() {
        let mut cfg = config(8);
        cfg.max_num_batched_tokens = 512;
        cfg.latency.prefill_pair_secs = 1e-7;
        cfg.latency.prefill_fixed_secs = 3e-3;
        let latency = cfg.latency;
        let engine = Engine::start(cfg);
        let (_, ttft, _) = run(&engine, &prompt(3, 1_300), 1).await;
        let want = latency.prefill(1_300, 0).as_secs_f64();
        assert!((ttft.as_secs_f64() - want).abs() < 1e-6, "{ttft:?} {want}");
    }

    #[tokio::test(start_paused = true)]
    async fn prefills_share_steps_and_queue_behind_the_budget() {
        let mut cfg = config(8);
        cfg.max_num_batched_tokens = 256;
        let engine = Engine::start(cfg);
        let tickets: Vec<_> = (0..3)
            .map(|i| engine.submit(&prompt(10 + i, 100), output(1)))
            .collect();
        assert_eq!(engine.snapshot().waiting, 3);
        let mut firsts = Vec::new();
        for ticket in tickets {
            let mut seq = ticket.admitted().await.unwrap();
            firsts.push(seq.next_token().await.unwrap() - seq.arrival);
        }
        assert_eq!(firsts[0], Duration::from_millis(256));
        assert_eq!(firsts[1], Duration::from_millis(256));
        assert_eq!(firsts[2], Duration::from_millis(256 + 44));
    }

    #[tokio::test(start_paused = true)]
    async fn ttft_grows_with_queue_depth() {
        let mut previous = Duration::ZERO;
        for depth in 0..6u32 {
            let mut cfg = config(64);
            cfg.max_num_batched_tokens = 200;
            let engine = Engine::start(cfg);
            let queued: Vec<_> = (0..depth)
                .map(|i| engine.submit(&prompt(100 + i, 200), output(1)))
                .collect();
            let probe = engine.submit(&prompt(999, 200), output(1));
            let mut seqs = Vec::new();
            for ticket in queued {
                seqs.push(ticket.admitted().await.unwrap());
            }
            let mut probe = probe.admitted().await.unwrap();
            let ttft = probe.next_token().await.unwrap() - probe.arrival;
            assert!(ttft > previous, "depth {depth}: {ttft:?} <= {previous:?}");
            previous = ttft;
        }
    }

    #[tokio::test(start_paused = true)]
    async fn decodes_slow_down_while_a_prompt_prefills() {
        let engine = Engine::start(config(8));
        let mut decoding = engine
            .submit(&prompt(1, 32), output(40))
            .admitted()
            .await
            .unwrap();
        let mut times = vec![decoding.next_token().await.unwrap()];
        for _ in 0..4 {
            times.push(decoding.next_token().await.unwrap());
        }
        let _prefill = engine.submit(&prompt(2, 500), output(1));
        for _ in 0..3 {
            times.push(decoding.next_token().await.unwrap());
        }
        let gaps: Vec<Duration> = times.windows(2).map(|w| w[1] - w[0]).collect();
        assert_eq!(gaps[0], Duration::from_millis(10));
        assert!(
            gaps.contains(&Duration::from_millis(510)),
            "no step carried the prefill: {gaps:?}"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn requests_queued_out_of_arrival_order_still_run() {
        let engine = Engine::start(config(8));
        let now = Instant::now();
        let later = engine.submit_at(&prompt(1, 32), output(2), now + Duration::from_millis(3));
        let earlier = engine.submit_at(&prompt(2, 32), output(2), now);
        let run = async {
            let mut later = later.admitted().await.unwrap();
            let mut earlier = earlier.admitted().await.unwrap();
            assert_eq!(later.scheduled, now + Duration::from_millis(3));
            assert_eq!(earlier.scheduled, later.scheduled);
            while later.next_token().await.is_some() {}
            while earlier.next_token().await.is_some() {}
        };
        tokio::time::timeout(Duration::from_secs(5), run)
            .await
            .expect("the engine stalled on its queue");
    }

    #[tokio::test(start_paused = true)]
    async fn sequence_cap_holds_requests_in_waiting() {
        let engine = Engine::start(config(1));
        let mut first = engine
            .submit(&prompt(1, 32), output(1_000))
            .admitted()
            .await
            .unwrap();
        first.next_token().await;
        let second = engine.submit(&prompt(2, 32), output(1));
        tokio::time::sleep(Duration::from_secs(1)).await;
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
        let mut cfg = config(8);
        cfg.max_num_batched_tokens = 1_000;
        let engine = Engine::start(cfg);
        let busy = engine.submit(&prompt(1, 1_000), output(1));
        let abandoned = engine.submit(&prompt(2, 1_000), output(1));
        let next = engine.submit(&prompt(3, 100), output(1));
        drop(abandoned);
        assert_eq!(engine.snapshot().waiting, 2);
        let mut busy = busy.admitted().await.unwrap();
        let mut next = next.admitted().await.unwrap();
        let busy_first = busy.next_token().await.unwrap();
        assert_eq!(next.scheduled, busy_first);
        assert_eq!(
            next.next_token().await.unwrap() - busy_first,
            Duration::from_millis(100)
        );
    }

    #[tokio::test(start_paused = true)]
    async fn decode_follows_planned_steps_and_tracks_kv_blocks() {
        let engine = Engine::start(config(8));
        let mut seq = engine
            .submit(&prompt(5, 30), output(4))
            .admitted()
            .await
            .unwrap();
        let mut times = Vec::new();
        for _ in 0..3 {
            times.push(seq.next_token().await.unwrap());
        }
        assert_eq!(seq.generated(), 3);
        assert_eq!(times[2] - times[0], Duration::from_millis(20));
        assert_eq!(engine.snapshot().kv_blocks_used, 3);
        assert!(seq.next_token().await.is_some());
        assert!(seq.next_token().await.is_none());
        let snap = engine.snapshot();
        assert_eq!((snap.kv_blocks_used, snap.running, snap.waiting), (0, 0, 0));
    }

    #[tokio::test(start_paused = true)]
    async fn generated_blocks_are_cached_for_the_next_turn() {
        let engine = Engine::start(config(8));
        let first = prompt(7, 40);
        run(&engine, &first, 30).await;
        let mut turn = first.clone();
        turn.extend(output(30));
        turn.extend(prompt(8, 20));
        let (hit, _, _) = run(&engine, &turn, 1).await;
        assert_eq!(
            hit, 64,
            "40 prompt and 29 computed output tokens fill 4 blocks"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn running_out_of_blocks_preempts_the_newest_request() {
        let mut cfg = config(8);
        cfg.num_gpu_blocks = 12;
        let engine = Engine::start(cfg);
        let mut old = engine
            .submit(&prompt(1, 64), output(100))
            .admitted()
            .await
            .unwrap();
        old.next_token().await;
        let mut young = engine
            .submit(&prompt(2, 64), output(100))
            .admitted()
            .await
            .unwrap();
        let mut young_times = Vec::new();
        while let Some(at) = young.next_token().await {
            young_times.push(at);
        }
        while old.next_token().await.is_some() {}
        assert_eq!(old.preemptions(), 0);
        assert!(young.preemptions() > 0, "the newer request should yield");
        assert!(engine.snapshot().preemptions > 0);
        assert_eq!(young_times.len(), 100);
        let snap = engine.snapshot();
        assert_eq!((snap.kv_blocks_used, snap.running, snap.waiting), (0, 0, 0));
    }

    #[tokio::test(start_paused = true)]
    async fn requests_larger_than_the_cache_are_refused() {
        let mut cfg = config(8);
        cfg.num_gpu_blocks = 4;
        let engine = Engine::start(cfg);
        let err = engine
            .submit(&prompt(1, 60), output(10))
            .admitted()
            .await
            .unwrap_err();
        assert_eq!(err, EngineError::TooLong);
        assert_eq!(engine.snapshot().waiting, 0);
        assert!(
            engine
                .submit(&prompt(1, 60), output(5))
                .admitted()
                .await
                .is_ok()
        );
        let empty = engine.submit(&prompt(1, 10), Vec::new()).admitted().await;
        assert_eq!(empty.unwrap_err(), EngineError::NoOutput);
    }

    #[tokio::test(start_paused = true)]
    async fn reset_prefix_cache_forgets_everything() {
        let engine = Engine::start(config(8));
        let tokens = prompt(9, 256);
        run(&engine, &tokens, 1).await;
        assert!(engine.snapshot().cached_blocks > 0);
        engine.reset_prefix_cache().await;
        assert_eq!(engine.snapshot().cached_blocks, 0);
        assert_eq!(run(&engine, &tokens, 1).await.0, 0);
    }

    #[tokio::test(start_paused = true)]
    async fn caching_off_never_hits() {
        let mut cfg = config(8);
        cfg.enable_prefix_caching = false;
        let engine = Engine::start(cfg);
        let tokens = prompt(4, 300);
        run(&engine, &tokens, 2).await;
        assert_eq!(run(&engine, &tokens, 2).await.0, 0);
        assert_eq!(engine.snapshot().cached_blocks, 0);
    }

    #[test]
    fn invalid_engine_configs_are_reported() {
        assert!(EngineConfig::default().validate().is_ok());
        for config in [
            EngineConfig {
                block_size: 0,
                ..EngineConfig::default()
            },
            EngineConfig {
                max_num_seqs: 0,
                ..EngineConfig::default()
            },
            EngineConfig {
                max_num_batched_tokens: 100,
                ..EngineConfig::default()
            },
            EngineConfig {
                latency: LatencyModel {
                    decode_token_secs: -1.0,
                    ..LatencyModel::ZERO
                },
                ..EngineConfig::default()
            },
        ] {
            assert!(config.validate().is_err(), "{config:?}");
        }
    }

    #[tokio::test]
    async fn zero_latency_decode_yields_to_other_tasks() {
        let engine = Engine::start(EngineConfig {
            latency: LatencyModel::ZERO,
            ..EngineConfig::default()
        });
        let ticks = Arc::new(AtomicU64::new(0));
        let ticker = {
            let ticks = Arc::clone(&ticks);
            tokio::spawn(async move {
                loop {
                    ticks.fetch_add(1, Ordering::Relaxed);
                    tokio::task::yield_now().await;
                }
            })
        };
        run(&engine, &prompt(1, 64), 20_000).await;
        ticker.abort();
        assert!(
            ticks.load(Ordering::Relaxed) > 10,
            "decode loop starved the runtime"
        );
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
