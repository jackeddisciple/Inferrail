//! Engine adapters: which families of a `/metrics` page describe load, for
//! vLLM v0.31.0 and SGLang v0.5.21, and how replicas inside one engine
//! (data parallel ranks) add up.

use std::borrow::Cow;
use std::fmt;

use crate::parse::{self, Sample};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Engine {
    Vllm,
    Sglang,
}

impl Engine {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Vllm => "vllm",
            Self::Sglang => "sglang",
        }
    }

    fn prefix(self) -> &'static str {
        match self {
            Self::Vllm => "vllm:",
            Self::Sglang => "sglang:",
        }
    }

    /// The label that tells data parallel ranks apart.
    fn rank_label(self) -> &'static str {
        match self {
            Self::Vllm => "engine",
            Self::Sglang => "dp_rank",
        }
    }
}

impl fmt::Display for Engine {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// What one engine reports about its own load, summed over its data
/// parallel ranks. Counters are cumulative since the engine started.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EngineLoad {
    pub engine: Engine,
    pub running: u32,
    pub waiting: u32,
    /// Share of KV cache in use, from 0 to 1, averaged over ranks.
    pub kv_usage: f64,
    pub prefix_queries: u64,
    pub prefix_hits: u64,
    pub preemptions: u64,
    pub kv_capacity_tokens: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ParseError {
    #[error("no vLLM or SGLang metrics on the page")]
    UnknownSchema,
    #[error("{engine} metrics lack {metric}")]
    Missing {
        engine: Engine,
        metric: &'static str,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Field {
    Running,
    Waiting,
    KvUsage,
    PrefixQueries,
    PrefixHits,
    Preemptions,
    Capacity,
}

const FIELDS: usize = 7;

/// How samples of one field combine inside a rank: `Max` for copies that
/// several workers of the same scheduler report, `Sum` for one total split
/// across label values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Within {
    Max,
    Sum,
}

fn field(engine: Engine, name: &str) -> Option<(Field, Within)> {
    use Field::*;
    use Within::*;
    let found = match (engine, name) {
        (Engine::Vllm, "num_requests_running") => (Running, Sum),
        (Engine::Vllm, "num_requests_waiting") => (Waiting, Sum),
        (Engine::Vllm, "kv_cache_usage_perc") => (KvUsage, Sum),
        (Engine::Vllm, "prefix_cache_queries_total") => (PrefixQueries, Sum),
        (Engine::Vllm, "prefix_cache_hits_total") => (PrefixHits, Sum),
        (Engine::Vllm, "num_preemptions_total") => (Preemptions, Sum),
        (Engine::Vllm, "cache_config_info") => (Capacity, Sum),
        (Engine::Sglang, "num_running_reqs") => (Running, Max),
        (Engine::Sglang, "num_queue_reqs") => (Waiting, Max),
        (Engine::Sglang, "token_usage") => (KvUsage, Max),
        (Engine::Sglang, "prompt_tokens_total") => (PrefixQueries, Sum),
        (Engine::Sglang, "cached_tokens_total") => (PrefixHits, Sum),
        (Engine::Sglang, "num_retracted_requests_total") => (Preemptions, Max),
        (Engine::Sglang, "max_total_num_tokens") => (Capacity, Max),
        _ => return None,
    };
    Some(found)
}

#[derive(Debug)]
struct Rank<'a> {
    key: Cow<'a, str>,
    values: [Option<f64>; FIELDS],
}

#[derive(Debug)]
struct Tally<'a> {
    engine: Engine,
    ranks: Vec<Rank<'a>>,
    seen: bool,
}

impl<'a> Tally<'a> {
    fn new(engine: Engine) -> Self {
        Self {
            engine,
            ranks: Vec::new(),
            seen: false,
        }
    }

    fn add(&mut self, sample: &Sample<'a>, field: Field, within: Within) {
        if sample.label("priority").is_some_and(|p| !p.is_empty()) {
            return;
        }
        let Some(value) = self.value_of(field, sample) else {
            return;
        };
        let key = sample
            .label(self.engine.rank_label())
            .unwrap_or(Cow::Borrowed(""));
        let index = match self.ranks.iter().position(|r| r.key == key) {
            Some(i) => i,
            None => {
                self.ranks.push(Rank {
                    key,
                    values: [None; FIELDS],
                });
                self.ranks.len() - 1
            }
        };
        let slot = &mut self.ranks[index].values[field as usize];
        *slot = Some(match (*slot, within) {
            (None, _) => value,
            (Some(old), Within::Max) => old.max(value),
            (Some(old), Within::Sum) => old + value,
        });
    }

    fn value_of(&self, field: Field, sample: &Sample<'_>) -> Option<f64> {
        let value = if field == Field::Capacity && self.engine == Engine::Vllm {
            let blocks: f64 = sample.label("num_gpu_blocks")?.parse().ok()?;
            let size: f64 = sample.label("block_size")?.parse().ok()?;
            blocks * size
        } else {
            sample.value
        };
        (value.is_finite() && value >= 0.0).then_some(value)
    }

    fn total(&self, field: Field) -> Option<f64> {
        self.ranks
            .iter()
            .filter_map(|r| r.values[field as usize])
            .fold(None, |acc, v| Some(acc.unwrap_or(0.0) + v))
    }

    fn mean(&self, field: Field) -> Option<f64> {
        let (sum, n) = self
            .ranks
            .iter()
            .filter_map(|r| r.values[field as usize])
            .fold((0.0, 0u32), |(s, n), v| (s + v, n + 1));
        (n > 0).then(|| sum / f64::from(n))
    }

    fn finish(&self) -> Result<EngineLoad, ParseError> {
        let engine = self.engine;
        let required = |field: Field, metric: &'static str| {
            self.total(field)
                .ok_or(ParseError::Missing { engine, metric })
        };
        let names = match engine {
            Engine::Vllm => [
                "vllm:num_requests_running",
                "vllm:num_requests_waiting",
                "vllm:kv_cache_usage_perc",
            ],
            Engine::Sglang => [
                "sglang:num_running_reqs",
                "sglang:num_queue_reqs",
                "sglang:token_usage",
            ],
        };
        let running = required(Field::Running, names[0])?;
        let waiting = required(Field::Waiting, names[1])?;
        let kv_usage = self.mean(Field::KvUsage).ok_or(ParseError::Missing {
            engine,
            metric: names[2],
        })?;
        let count = |field| self.total(field).map_or(0, to_u64);
        Ok(EngineLoad {
            engine,
            running: to_u32(running),
            waiting: to_u32(waiting),
            kv_usage: kv_usage.clamp(0.0, 1.0),
            prefix_queries: count(Field::PrefixQueries),
            prefix_hits: count(Field::PrefixHits),
            preemptions: count(Field::Preemptions),
            kv_capacity_tokens: self.total(Field::Capacity).map(to_u64),
        })
    }
}

fn to_u64(value: f64) -> u64 {
    value.round() as u64
}

fn to_u32(value: f64) -> u32 {
    value.round() as u32
}

/// Reads an engine's `/metrics` page. The engine is recognized by its metric
/// prefix; a page with neither prefix, or missing a required family, is an
/// error so a schema change never passes for an idle engine.
pub fn parse(text: &str) -> Result<EngineLoad, ParseError> {
    let mut vllm = Tally::new(Engine::Vllm);
    let mut sglang = Tally::new(Engine::Sglang);
    for line in text.lines() {
        let Some(full) = parse::name(line) else {
            continue;
        };
        for tally in [&mut vllm, &mut sglang] {
            if let Some(name) = full.strip_prefix(tally.engine.prefix()) {
                tally.seen = true;
                if let Some((field, within)) = field(tally.engine, name)
                    && let Some(sample) = parse::sample(line)
                {
                    tally.add(&sample, field, within);
                }
            }
        }
    }
    match (vllm.seen, sglang.seen) {
        (true, false) => vllm.finish(),
        (false, true) => sglang.finish(),
        (true, true) => vllm.finish().or_else(|_| sglang.finish()),
        (false, false) => Err(ParseError::UnknownSchema),
    }
}
