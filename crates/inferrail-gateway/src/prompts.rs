//! Prompts tokenized and hashed at the edge. Short bodies are tokenized on
//! the request's own task; long ones go to a bounded pool of threads, and a
//! request that finds it full or slow is routed without a cache prediction
//! rather than made to wait.

use std::collections::HashMap;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::mpsc::{Receiver, SyncSender, sync_channel};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::{Duration, Instant};

use hyper::body::Bytes;
use inferrail_control::Replica;
use inferrail_core::config::{Config, ContentFormat as FormatSetting, ModelConfig};
use inferrail_core::ids::{BlockHash, ModelId};
use inferrail_tokenize::hash::{BlockHasher, Salt, cacheable_blocks};
use inferrail_tokenize::{
    CacheStats, ContentFormat, Endpoint, ModelTokenizer, Parity, TokenizeError, TokenizerOptions,
};
use tokio::sync::oneshot;

const INLINE_BYTES: usize = 1024;

/// A request's prompt as the engine will see it, cut into hashed blocks.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Prompt {
    pub(crate) tokens: u32,
    pub(crate) hashes: Vec<BlockHash>,
    /// Leading blocks an engine can serve from cache: it always computes
    /// the last prompt token itself.
    pub(crate) cacheable: usize,
    pub(crate) block_size: usize,
    /// How tokenizing it used the model's cache of repeated pieces.
    pub(crate) cache: CacheStats,
}

impl Prompt {
    pub(crate) fn cached_tokens(&self, blocks: usize) -> u32 {
        u32::try_from(blocks.min(self.cacheable) * self.block_size).unwrap_or(u32::MAX)
    }
}

/// Why a request was routed without its prompt.
#[derive(Debug)]
pub(crate) enum Skip {
    /// The model has no tokenizer configured.
    Unconfigured,
    /// Its tokenizer failed to load at startup.
    Unavailable,
    /// Every worker was busy and the queue full.
    Busy,
    /// The prompt was not ready within `tokenizer.max_wait`.
    Slow,
    Failed(TokenizeError),
}

impl Skip {
    pub(crate) fn label(&self) -> &'static str {
        match self {
            Self::Unconfigured => "unconfigured",
            Self::Unavailable => "unavailable",
            Self::Busy => "busy",
            Self::Slow => "slow",
            Self::Failed(err) => err.kind(),
        }
    }
}

#[derive(Debug)]
enum Loaded {
    None,
    Failed,
    Ready(Arc<ModelTokenizer>),
}

#[derive(Debug)]
struct Model {
    tokenizer: Loaded,
    block_size: usize,
    salt_name: Arc<str>,
}

type Job = Box<dyn FnOnce() + Send>;

#[derive(Debug)]
struct Pool {
    jobs: SyncSender<Job>,
}

impl Pool {
    fn start(workers: usize, queue: usize) -> std::io::Result<Self> {
        let (jobs, receiver) = sync_channel::<Job>(queue);
        let receiver = Arc::new(Mutex::new(receiver));
        for i in 0..workers {
            let receiver = Arc::clone(&receiver);
            std::thread::Builder::new()
                .name(format!("tokenize-{i}"))
                .spawn(move || work(&receiver))?;
        }
        Ok(Self { jobs })
    }

    /// Queues `work`, or gives up at once when the queue is full. Work whose
    /// requester stopped waiting is skipped.
    fn submit<R: Send + 'static>(
        &self,
        work: impl FnOnce() -> R + Send + 'static,
    ) -> Option<oneshot::Receiver<R>> {
        let (reply, ready) = oneshot::channel();
        let job: Job = Box::new(move || {
            if !reply.is_closed() {
                let _ = reply.send(work());
            }
        });
        self.jobs.try_send(job).ok().map(|()| ready)
    }
}

fn work(receiver: &Mutex<Receiver<Job>>) {
    loop {
        let job = receiver
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .recv();
        match job {
            Ok(job) => {
                let _ = catch_unwind(AssertUnwindSafe(job));
            }
            Err(_) => return,
        }
    }
}

/// Every model's tokenizer, loaded once and shared by models configured
/// alike, with the pool that runs long prompts.
#[derive(Debug)]
pub(crate) struct Prompts {
    models: HashMap<ModelId, Model>,
    pool: Pool,
    max_wait: Duration,
}

/// A tokenizer that could not be loaded; its model routes without
/// predictions.
#[derive(Debug)]
pub(crate) struct LoadFailure {
    pub(crate) model: ModelId,
    pub(crate) error: TokenizeError,
}

fn options(model: &ModelConfig, spec: &str, cache_tokens: usize) -> TokenizerOptions {
    TokenizerOptions {
        source: spec.to_owned(),
        chat_template: model.chat_template.clone(),
        content_format: match model.chat_template_content_format {
            FormatSetting::Auto => None,
            FormatSetting::String => Some(ContentFormat::String),
            FormatSetting::Openai => Some(ContentFormat::OpenAi),
        },
        template_kwargs: model
            .chat_template_kwargs
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect(),
        cache_tokens,
        ..TokenizerOptions::default()
    }
}

/// Loads every configured tokenizer; models that share every setting share
/// one instance.
pub(crate) fn load_tokenizers(
    config: &Config,
) -> (HashMap<ModelId, Arc<ModelTokenizer>>, Vec<LoadFailure>) {
    let mut shared: HashMap<String, Arc<ModelTokenizer>> = HashMap::new();
    let mut loaded = HashMap::new();
    let mut failures = Vec::new();
    for model in &config.models {
        let Some(spec) = &model.tokenizer else {
            continue;
        };
        let key = format!(
            "{spec}\0{:?}\0{:?}\0{}",
            model.chat_template,
            model.chat_template_content_format,
            serde_json::to_string(&model.chat_template_kwargs).unwrap_or_default()
        );
        if let Some(tokenizer) = shared.get(&key) {
            loaded.insert(model.name.clone(), Arc::clone(tokenizer));
            continue;
        }
        match ModelTokenizer::load(&options(model, spec, config.tokenizer.cache_tokens)) {
            Ok(tokenizer) => {
                let tokenizer = Arc::new(tokenizer);
                shared.insert(key, Arc::clone(&tokenizer));
                loaded.insert(model.name.clone(), tokenizer);
            }
            Err(error) => failures.push(LoadFailure {
                model: model.name.clone(),
                error,
            }),
        }
    }
    (loaded, failures)
}

impl Prompts {
    pub(crate) fn new(
        config: &Config,
        tokenizers: &HashMap<ModelId, Arc<ModelTokenizer>>,
    ) -> std::io::Result<Self> {
        let models = config
            .models
            .iter()
            .map(|m| {
                let tokenizer = match (tokenizers.get(&m.name), &m.tokenizer) {
                    (Some(tokenizer), _) => Loaded::Ready(Arc::clone(tokenizer)),
                    (None, Some(_)) => Loaded::Failed,
                    (None, None) => Loaded::None,
                };
                if let Loaded::Ready(tokenizer) = &tokenizer {
                    if tokenizer.parity() == Parity::Unverified {
                        tracing::warn!(model = %m.name, "the tokenizer class has no verified parity with the engine; cache predictions may miss");
                    }
                    if !tokenizer.has_chat_template() {
                        tracing::info!(model = %m.name, "the tokenizer has no chat template; only completions get cache predictions");
                    }
                }
                let model = Model {
                    tokenizer,
                    block_size: m.block_size as usize,
                    salt_name: Arc::from(m.name.as_str()),
                };
                (m.name.clone(), model)
            })
            .collect();
        let workers = match config.tokenizer.workers {
            0 => std::thread::available_parallelism().map_or(1, |n| (n.get() / 2).max(1)),
            n => n,
        };
        Ok(Self {
            models,
            pool: Pool::start(workers, config.tokenizer.queue)?,
            max_wait: config.tokenizer.max_wait,
        })
    }

    /// Whether `model`'s prompts are tokenized.
    pub(crate) fn tokenizes(&self, model: &ModelId) -> bool {
        self.models
            .get(model)
            .is_some_and(|m| matches!(m.tokenizer, Loaded::Ready(_)))
    }

    /// Whether any model gets predictions at all.
    pub(crate) fn any(&self) -> bool {
        self.models
            .values()
            .any(|m| matches!(m.tokenizer, Loaded::Ready(_)))
    }

    /// Tokenizes and hashes `body` for `model`, with how long it took.
    pub(crate) async fn prepare(
        &self,
        model: &ModelId,
        endpoint: Endpoint,
        body: &Bytes,
    ) -> (Result<Prompt, Skip>, Duration) {
        let started = Instant::now();
        let Some(entry) = self.models.get(model) else {
            return (Err(Skip::Unconfigured), started.elapsed());
        };
        let tokenizer = match &entry.tokenizer {
            Loaded::Ready(tokenizer) => Arc::clone(tokenizer),
            Loaded::Failed => return (Err(Skip::Unavailable), started.elapsed()),
            Loaded::None => return (Err(Skip::Unconfigured), started.elapsed()),
        };
        let (block_size, name) = (entry.block_size, Arc::clone(&entry.salt_name));
        if body.len() <= INLINE_BYTES {
            let prompt = encode(&tokenizer, endpoint, body, &name, block_size);
            return (prompt.map_err(Skip::Failed), started.elapsed());
        }
        let body = body.clone();
        let Some(ready) = self
            .pool
            .submit(move || encode(&tokenizer, endpoint, &body, &name, block_size))
        else {
            return (Err(Skip::Busy), started.elapsed());
        };
        let outcome = match tokio::time::timeout(self.max_wait, ready).await {
            Ok(Ok(prompt)) => prompt.map_err(Skip::Failed),
            Ok(Err(_)) => Err(Skip::Failed(TokenizeError::Tokenizer(
                "the tokenizer stopped".to_owned(),
            ))),
            Err(_) => Err(Skip::Slow),
        };
        (outcome, started.elapsed())
    }
}

fn encode(
    tokenizer: &ModelTokenizer,
    endpoint: Endpoint,
    body: &[u8],
    model: &str,
    block_size: usize,
) -> Result<Prompt, TokenizeError> {
    let encoded = tokenizer.encode(endpoint, body)?;
    let salt = Salt::new(model, None, encoded.cache_salt.as_deref());
    let hashes = BlockHasher::new(block_size.max(1)).hash(&encoded.tokens, salt);
    let len = encoded.tokens.len();
    Ok(Prompt {
        tokens: u32::try_from(len).unwrap_or(u32::MAX),
        cacheable: cacheable_blocks(len, block_size).min(hashes.len()),
        hashes,
        block_size,
        cache: encoded.cache,
    })
}

/// The replica's KV capacity in blocks: its configured size, else what its
/// engine reports, else the default.
pub(crate) fn capacity(replica: &Replica, block_size: usize, default: u64) -> usize {
    let blocks = replica.spec().kv_capacity_blocks.or_else(|| {
        replica
            .engine_load()
            .latest()
            .and_then(|s| s.load.kv_capacity_tokens)
            .map(|tokens| tokens / block_size.max(1) as u64)
    });
    usize::try_from(blocks.unwrap_or(default)).unwrap_or(usize::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(text: &str) -> Config {
        Config::from_yaml_str(text).unwrap()
    }

    const SIM: &str = "models:\n  - name: m\n    tokenizer: builtin:sim\n  - name: plain\nreplicas:\n  - id: r\n    url: http://127.0.0.1:1\n    models: [m, plain]\ntokenizer:\n  workers: 2\n  queue: 2\n  max_wait: 2s\n";

    fn prompts(text: &str) -> Prompts {
        let config = config(text);
        let (tokenizers, failures) = load_tokenizers(&config);
        assert!(failures.is_empty(), "{failures:?}");
        Prompts::new(&config, &tokenizers).unwrap()
    }

    fn chat(words: usize) -> Bytes {
        let content = "lorem ipsum ".repeat(words);
        Bytes::from(
            serde_json::json!({"messages": [{"role": "user", "content": content}]}).to_string(),
        )
    }

    #[tokio::test]
    async fn short_and_long_prompts_tokenize_the_same_way() {
        let prompts = prompts(SIM);
        let model = ModelId::new("m").unwrap();
        for words in [2, 2_000] {
            let body = chat(words);
            let (prompt, _) = prompts.prepare(&model, Endpoint::Chat, &body).await;
            let prompt = prompt.unwrap();
            let want = ModelTokenizer::sim()
                .encode(Endpoint::Chat, &body)
                .unwrap()
                .tokens;
            assert_eq!(prompt.tokens as usize, want.len());
            assert_eq!(prompt.hashes.len(), want.len() / 16);
            assert_eq!(prompt.cacheable, (want.len() - 1) / 16);
            assert_eq!(
                prompt.cached_tokens(usize::MAX),
                (prompt.cacheable * 16) as u32
            );
        }
        let plain = ModelId::new("plain").unwrap();
        let (skip, _) = prompts.prepare(&plain, Endpoint::Chat, &chat(2)).await;
        assert_eq!(skip.unwrap_err().label(), "unconfigured");
        let (bad, _) = prompts
            .prepare(&model, Endpoint::Chat, &Bytes::from_static(b"{"))
            .await;
        assert_eq!(bad.unwrap_err().label(), "invalid");
        assert!(prompts.any());
    }

    #[tokio::test]
    async fn a_full_pool_routes_without_waiting() {
        let pool = Pool::start(1, 1).unwrap();
        let (hold, release) = std::sync::mpsc::channel::<()>();
        let blocked = pool.submit(move || release.recv().ok()).unwrap();
        let mut queued = Vec::new();
        while let Some(rx) = pool.submit(|| 1) {
            queued.push(rx);
            assert!(queued.len() < 10, "the queue never filled");
        }
        hold.send(()).unwrap();
        assert_eq!(blocked.await.unwrap(), Some(()));
        for rx in queued {
            assert_eq!(rx.await.unwrap(), 1);
        }
    }

    #[tokio::test]
    async fn panicking_work_keeps_the_worker() {
        let pool = Pool::start(1, 4).unwrap();
        let panicked = pool.submit(|| -> u32 { panic!("bad prompt") }).unwrap();
        assert!(panicked.await.is_err());
        assert_eq!(pool.submit(|| 7).unwrap().await.unwrap(), 7);
    }

    #[tokio::test]
    async fn slow_prompts_are_routed_without_waiting_past_max_wait() {
        let prompts = prompts(&SIM.replace("max_wait: 2s", "max_wait: 5ms"));
        let busy: Vec<_> = (0..2)
            .map(|_| {
                prompts
                    .pool
                    .submit(|| std::thread::sleep(Duration::from_millis(300)))
            })
            .collect();
        let (outcome, took) = prompts
            .prepare(&ModelId::new("m").unwrap(), Endpoint::Chat, &chat(2_000))
            .await;
        assert!(
            matches!(outcome, Err(Skip::Slow | Skip::Busy)),
            "{outcome:?}"
        );
        assert!(took < Duration::from_millis(150), "{took:?}");
        assert!(busy.iter().all(Option::is_some));
    }

    #[tokio::test]
    async fn cache_lookups_belong_to_the_request_that_made_them() {
        let dir =
            std::env::temp_dir().join(format!("inferrail-prompt-cache-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let tokenizer = serde_json::json!({
            "version": "1.0", "truncation": null, "padding": null, "added_tokens": [],
            "normalizer": null, "pre_tokenizer": {"type": "Whitespace"}, "post_processor": null,
            "decoder": null,
            "model": {"type": "WordLevel", "vocab": {"<unk>": 0, "hello": 1, "world": 2}, "unk_token": "<unk>"}
        });
        std::fs::write(dir.join("tokenizer.json"), tokenizer.to_string()).unwrap();
        let prompts = prompts(&format!(
            "models:\n  - name: a\n    tokenizer: {d}\n  - name: b\n    tokenizer: {d}\nreplicas:\n  - id: r\n    url: http://127.0.0.1:1\n    models: [a, b]\n",
            d = dir.display()
        ));
        let body = Bytes::from(
            serde_json::json!({"prompt": "hello world ".repeat(40), "add_special_tokens": false})
                .to_string(),
        );
        let mut seen = Vec::new();
        for model in ["a", "b", "a"] {
            let model = ModelId::new(model).unwrap();
            let (prompt, _) = prompts.prepare(&model, Endpoint::Completion, &body).await;
            let cache = prompt.unwrap().cache;
            seen.push((cache.hits, cache.misses));
        }
        assert_eq!(seen, [(0, 1), (1, 0), (1, 0)]);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn failed_tokenizers_are_reported_and_alike_models_share_one() {
        let config = config(
            "models:\n  - name: a\n    tokenizer: builtin:sim\n  - name: b\n    tokenizer: builtin:sim\n  - name: c\n    tokenizer: /no/such/model\nreplicas:\n  - id: r\n    url: http://127.0.0.1:1\n    models: [a, b, c]\n",
        );
        let (tokenizers, failures) = load_tokenizers(&config);
        assert_eq!(tokenizers.len(), 2);
        let prompts = Prompts::new(&config, &tokenizers).unwrap();
        let c = ModelId::new("c").unwrap();
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let (skip, _) =
            runtime.block_on(prompts.prepare(&c, Endpoint::Chat, &Bytes::from_static(b"{}")));
        assert_eq!(skip.unwrap_err().label(), "unavailable");
        let (a, b) = (
            &tokenizers[&ModelId::new("a").unwrap()],
            &tokenizers[&ModelId::new("b").unwrap()],
        );
        assert!(Arc::ptr_eq(a, b));
        assert_eq!(failures.len(), 1);
        assert_eq!(failures[0].model.as_str(), "c");
    }
}
