//! vLLM v0.31.0 metric families, rendered byte for byte the way Python's
//! prometheus_client renders them on a vLLM server's /metrics endpoint.

use std::fmt::Write as _;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use inferrail_core::openai::FinishReason;
use tokio::time::Instant;

use crate::engine::{EngineConfig, EngineSnapshot, Sequence};

pub const CONTENT_TYPE: &str = "text/plain; version=0.0.4; charset=utf-8";

const REQUEST_LATENCY: &[f64] = &[
    0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0, 120.0, 240.0,
    480.0, 960.0, 1920.0, 7680.0,
];
const TIME_TO_FIRST_TOKEN: &[f64] = &[
    0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0,
    20.0, 40.0, 80.0, 160.0, 640.0, 2560.0,
];
const INTER_TOKEN_LATENCY: &[f64] = &[
    0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0, 20.0,
    40.0, 80.0,
];
const PARAMS_N: &[f64] = &[1.0, 2.0, 5.0, 10.0, 20.0];
const PREEMPTIONS: &[f64] = &[1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 20.0];
const FINISH_REASONS: [&str; 5] = ["stop", "length", "abort", "error", "repetition"];

/// vLLM's 1, 2, 5, 10, 20, 50, ... series capped at the context length.
fn token_buckets(max_model_len: usize) -> Vec<f64> {
    let mut out = Vec::new();
    let mut scale = 1u64;
    loop {
        for m in [1, 2, 5] {
            let value = m * scale;
            if value > max_model_len as u64 {
                return out;
            }
            out.push(value as f64);
        }
        scale *= 10;
    }
}

/// The last bucket counts values above every bound; `_count` is the sum of
/// all buckets, so a scrape can never see a count below a cumulative bucket.
#[derive(Debug)]
struct Histogram {
    bounds: Vec<f64>,
    buckets: Vec<AtomicU64>,
    sum_bits: AtomicU64,
}

impl Histogram {
    fn new(bounds: &[f64]) -> Self {
        Self {
            bounds: bounds.to_vec(),
            buckets: (0..=bounds.len()).map(|_| AtomicU64::new(0)).collect(),
            sum_bits: AtomicU64::new(0f64.to_bits()),
        }
    }

    fn observe(&self, value: f64) {
        let i = self
            .bounds
            .iter()
            .position(|&b| value <= b)
            .unwrap_or(self.bounds.len());
        self.buckets[i].fetch_add(1, Ordering::Relaxed);
        let _ = self
            .sum_bits
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |bits| {
                Some((f64::from_bits(bits) + value).to_bits())
            });
    }

    fn observe_duration(&self, value: Duration) {
        self.observe(value.as_secs_f64());
    }
}

#[derive(Debug)]
pub struct Metrics {
    model_name: String,
    created: f64,
    block_size: usize,
    num_gpu_blocks: usize,
    prefix_caching: bool,
    prompt_tokens: AtomicU64,
    generation_tokens: AtomicU64,
    prompt_tokens_cached: AtomicU64,
    computed_tokens: AtomicU64,
    success: [AtomicU64; 5],
    request_prompt_tokens: Histogram,
    request_generation_tokens: Histogram,
    request_max_num_generation_tokens: Histogram,
    request_params_n: Histogram,
    request_params_max_tokens: Histogram,
    time_to_first_token: Histogram,
    inter_token_latency: Histogram,
    request_time_per_output_token: Histogram,
    e2e_request_latency: Histogram,
    request_queue_time: Histogram,
    request_inference_time: Histogram,
    request_prefill_time: Histogram,
    request_decode_time: Histogram,
    request_num_preemptions: Histogram,
    request_prefill_kv_computed_tokens: Histogram,
}

impl Metrics {
    pub fn new(model_name: impl Into<String>, max_model_len: usize, engine: &EngineConfig) -> Self {
        let tokens = token_buckets(max_model_len);
        let created = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0.0, |d| d.as_secs_f64());
        Self {
            model_name: model_name.into(),
            created,
            block_size: engine.block_size,
            num_gpu_blocks: engine.num_gpu_blocks,
            prefix_caching: engine.enable_prefix_caching,
            prompt_tokens: AtomicU64::new(0),
            generation_tokens: AtomicU64::new(0),
            prompt_tokens_cached: AtomicU64::new(0),
            computed_tokens: AtomicU64::new(0),
            success: Default::default(),
            request_prompt_tokens: Histogram::new(&tokens),
            request_generation_tokens: Histogram::new(&tokens),
            request_max_num_generation_tokens: Histogram::new(&tokens),
            request_params_n: Histogram::new(PARAMS_N),
            request_params_max_tokens: Histogram::new(&tokens),
            time_to_first_token: Histogram::new(TIME_TO_FIRST_TOKEN),
            inter_token_latency: Histogram::new(INTER_TOKEN_LATENCY),
            request_time_per_output_token: Histogram::new(INTER_TOKEN_LATENCY),
            e2e_request_latency: Histogram::new(REQUEST_LATENCY),
            request_queue_time: Histogram::new(REQUEST_LATENCY),
            request_inference_time: Histogram::new(REQUEST_LATENCY),
            request_prefill_time: Histogram::new(REQUEST_LATENCY),
            request_decode_time: Histogram::new(REQUEST_LATENCY),
            request_num_preemptions: Histogram::new(PREEMPTIONS),
            request_prefill_kv_computed_tokens: Histogram::new(&tokens),
        }
    }

    pub fn tracker(self: &Arc<Self>, max_tokens: usize) -> RequestTracker {
        RequestTracker {
            metrics: Arc::clone(self),
            max_tokens,
            admitted: None,
            first: None,
            last: None,
            generated: 0,
            preemptions: 0,
            finished: false,
        }
    }

    pub fn finished(&self, reason: &str) -> u64 {
        FINISH_REASONS
            .iter()
            .position(|r| *r == reason)
            .map_or(0, |i| self.success[i].load(Ordering::Relaxed))
    }

    pub fn render(&self, snap: &EngineSnapshot) -> String {
        let mut w = Writer {
            out: String::with_capacity(48 * 1024),
            model: escape_label(&self.model_name),
            created: go_float(self.created),
        };
        let load = |a: &AtomicU64| a.load(Ordering::Relaxed) as f64;
        w.gauge(
            "vllm:num_requests_running",
            "Number of requests in model execution batches.",
            &[(None, snap.running as f64)],
        );
        w.gauge(
            "vllm:num_requests_waiting",
            "Number of requests waiting to be processed.",
            &[(None, snap.waiting as f64)],
        );
        w.gauge(
            "vllm:num_requests_waiting_by_reason",
            "Number of waiting requests by reason. Reason labels: 'capacity' = waiting for scheduling capacity; 'deferred' = deferred by transient constraints (LoRA budget, KV transfer, blocked status). Sum of all reasons equals vllm:num_requests_waiting.",
            &[
                (Some(("reason", "capacity")), snap.waiting as f64),
                (Some(("reason", "deferred")), 0.0),
            ],
        );
        w.gauge(
            "vllm:engine_sleep_state",
            "Engine sleep state; awake = 0 means engine is sleeping; awake = 1 means engine is awake; weights_offloaded = 1 means sleep level 1; discard_all = 1 means sleep level 2.",
            &[
                (Some(("sleep_state", "awake")), 1.0),
                (Some(("sleep_state", "weights_offloaded")), 0.0),
                (Some(("sleep_state", "discard_all")), 0.0),
            ],
        );
        w.gauge(
            "vllm:kv_cache_usage_perc",
            "KV-cache usage. 1 means 100 percent usage.",
            &[(None, snap.kv_usage)],
        );
        w.counter(
            "vllm:prefix_cache_queries",
            "Prefix cache queries, in terms of number of queried tokens.",
            &[(None, snap.prefix_queries as f64)],
        );
        w.counter(
            "vllm:prefix_cache_hits",
            "Prefix cache hits, in terms of number of cached tokens.",
            &[(None, snap.prefix_hits as f64)],
        );
        w.counter(
            "vllm:external_prefix_cache_queries",
            "External prefix cache queries from KV connector cross-instance cache sharing, in terms of number of queried tokens.",
            &[(None, 0.0)],
        );
        w.counter(
            "vllm:external_prefix_cache_hits",
            "External prefix cache hits from KV connector cross-instance cache sharing, in terms of number of cached tokens.",
            &[(None, 0.0)],
        );
        w.counter(
            "vllm:mm_cache_queries",
            "Multi-modal cache queries, in terms of number of queried items.",
            &[(None, 0.0)],
        );
        w.counter(
            "vllm:mm_cache_hits",
            "Multi-modal cache hits, in terms of number of cached items.",
            &[(None, 0.0)],
        );
        w.counter(
            "vllm:num_preemptions",
            "Cumulative number of preemption from the engine.",
            &[(None, snap.preemptions as f64)],
        );
        w.counter(
            "vllm:prompt_tokens",
            "Number of prefill tokens processed.",
            &[(None, load(&self.prompt_tokens))],
        );
        w.counter(
            "vllm:prompt_tokens_by_source",
            "Number of prompt tokens by source.",
            &[
                (
                    Some(("source", "local_compute")),
                    load(&self.computed_tokens),
                ),
                (
                    Some(("source", "local_cache_hit")),
                    load(&self.prompt_tokens_cached),
                ),
                (Some(("source", "external_kv_transfer")), 0.0),
            ],
        );
        w.counter(
            "vllm:prompt_tokens_cached",
            "Number of cached prompt tokens (local + external).",
            &[(None, load(&self.prompt_tokens_cached))],
        );
        w.counter(
            "vllm:generation_tokens",
            "Number of generation tokens processed.",
            &[(None, load(&self.generation_tokens))],
        );
        let success: Vec<_> = FINISH_REASONS
            .iter()
            .zip(&self.success)
            .map(|(reason, n)| (Some(("finished_reason", *reason)), load(n)))
            .collect();
        w.counter(
            "vllm:request_success",
            "Count of successfully processed requests.",
            &success,
        );
        w.histogram(
            "vllm:request_prompt_tokens",
            "Number of prefill tokens processed.",
            &self.request_prompt_tokens,
        );
        w.histogram(
            "vllm:request_generation_tokens",
            "Number of generation tokens processed.",
            &self.request_generation_tokens,
        );
        w.histogram(
            "vllm:request_max_num_generation_tokens",
            "Histogram of maximum number of requested generation tokens.",
            &self.request_max_num_generation_tokens,
        );
        w.histogram(
            "vllm:request_params_n",
            "Histogram of the n request parameter.",
            &self.request_params_n,
        );
        w.histogram(
            "vllm:request_params_max_tokens",
            "Histogram of the max_tokens request parameter.",
            &self.request_params_max_tokens,
        );
        w.histogram(
            "vllm:time_to_first_token_seconds",
            "Histogram of time to first token in seconds.",
            &self.time_to_first_token,
        );
        w.histogram(
            "vllm:inter_token_latency_seconds",
            "Histogram of inter-token latency in seconds.",
            &self.inter_token_latency,
        );
        w.histogram(
            "vllm:request_time_per_output_token_seconds",
            "Histogram of time_per_output_token_seconds per request.",
            &self.request_time_per_output_token,
        );
        w.histogram(
            "vllm:e2e_request_latency_seconds",
            "Histogram of e2e request latency in seconds.",
            &self.e2e_request_latency,
        );
        w.histogram(
            "vllm:request_queue_time_seconds",
            "Histogram of time spent in WAITING phase for request.",
            &self.request_queue_time,
        );
        w.histogram(
            "vllm:request_inference_time_seconds",
            "Histogram of time spent in RUNNING phase for request.",
            &self.request_inference_time,
        );
        w.histogram(
            "vllm:request_prefill_time_seconds",
            "Histogram of time spent in PREFILL phase for request.",
            &self.request_prefill_time,
        );
        w.histogram(
            "vllm:request_decode_time_seconds",
            "Histogram of time spent in DECODE phase for request.",
            &self.request_decode_time,
        );
        w.histogram(
            "vllm:request_num_preemptions",
            "Histogram of the number of times a request was preempted.",
            &self.request_num_preemptions,
        );
        w.histogram(
            "vllm:request_prefill_kv_computed_tokens",
            "Histogram of new KV tokens computed during prefill (excluding cached tokens).",
            &self.request_prefill_kv_computed_tokens,
        );
        let block_size = self.block_size.to_string();
        let blocks = self.num_gpu_blocks.to_string();
        w.info(
            "vllm:cache_config_info",
            "Information of the LLMEngine CacheConfig",
            &[
                ("block_size", &block_size),
                ("cache_dtype", "auto"),
                (
                    "enable_prefix_caching",
                    if self.prefix_caching { "True" } else { "False" },
                ),
                ("engine", "0"),
                ("gpu_memory_utilization", "0.92"),
                ("num_gpu_blocks", &blocks),
                ("num_gpu_blocks_override", &blocks),
                ("prefix_caching_hash_algo", "xxhash"),
            ],
        );
        w.out
    }
}

type Extra<'a> = Option<(&'a str, &'a str)>;

struct Writer {
    out: String,
    model: String,
    created: String,
}

impl Writer {
    fn labels(&self, extra: &[(&str, &str)]) -> String {
        let mut pairs: Vec<(&str, String)> = vec![
            ("engine", "0".to_owned()),
            ("model_name", self.model.clone()),
        ];
        pairs.extend(extra.iter().map(|(k, v)| (*k, escape_label(v))));
        pairs.sort_by(|a, b| a.0.cmp(b.0));
        let body: Vec<String> = pairs.iter().map(|(k, v)| format!("{k}=\"{v}\"")).collect();
        format!("{{{}}}", body.join(","))
    }

    fn header(&mut self, name: &str, help: &str, kind: &str) {
        let help = help.replace('\\', "\\\\").replace('\n', "\\n");
        let _ = writeln!(self.out, "# HELP {name} {help}");
        let _ = writeln!(self.out, "# TYPE {name} {kind}");
    }

    fn sample(&mut self, name: &str, labels: &str, value: f64) {
        let _ = writeln!(self.out, "{name}{labels} {}", go_float(value));
    }

    fn created(&mut self, name: &str, help: &str, label_sets: &[String]) {
        self.header(&format!("{name}_created"), help, "gauge");
        let created = self.created.clone();
        for labels in label_sets {
            let _ = writeln!(self.out, "{name}_created{labels} {created}");
        }
    }

    fn gauge(&mut self, name: &str, help: &str, rows: &[(Extra<'_>, f64)]) {
        self.header(name, help, "gauge");
        for (extra, value) in rows {
            let labels = self.labels(extra.as_slice());
            self.sample(name, &labels, *value);
        }
    }

    fn info(&mut self, name: &str, help: &str, labels: &[(&str, &str)]) {
        self.header(name, help, "gauge");
        let mut pairs: Vec<(&str, String)> =
            labels.iter().map(|(k, v)| (*k, escape_label(v))).collect();
        pairs.sort_by(|a, b| a.0.cmp(b.0));
        let body: Vec<String> = pairs.iter().map(|(k, v)| format!("{k}=\"{v}\"")).collect();
        self.sample(name, &format!("{{{}}}", body.join(",")), 1.0);
    }

    fn counter(&mut self, name: &str, help: &str, rows: &[(Extra<'_>, f64)]) {
        let total = format!("{name}_total");
        self.header(&total, help, "counter");
        let mut sets = Vec::with_capacity(rows.len());
        for (extra, value) in rows {
            let labels = self.labels(extra.as_slice());
            self.sample(&total, &labels, *value);
            sets.push(labels);
        }
        self.created(name, help, &sets);
    }

    fn histogram(&mut self, name: &str, help: &str, h: &Histogram) {
        self.header(name, help, "histogram");
        let bucket = format!("{name}_bucket");
        let mut cumulative = 0u64;
        for (bound, count) in h.bounds.iter().zip(&h.buckets) {
            cumulative += count.load(Ordering::Relaxed);
            let labels = self.labels(&[("le", &go_float(*bound))]);
            self.sample(&bucket, &labels, cumulative as f64);
        }
        let total = cumulative + h.buckets[h.bounds.len()].load(Ordering::Relaxed);
        let labels = self.labels(&[("le", "+Inf")]);
        self.sample(&bucket, &labels, total as f64);
        let base = self.labels(&[]);
        self.sample(&format!("{name}_count"), &base, total as f64);
        let sum = f64::from_bits(h.sum_bits.load(Ordering::Relaxed));
        self.sample(&format!("{name}_sum"), &base, sum);
        self.created(name, help, &[base]);
    }
}

fn escape_label(value: &str) -> String {
    value
        .replace('\\', "\\\\")
        .replace('\n', "\\n")
        .replace('"', "\\\"")
}

/// Python's `repr(float)`: shortest round trip digits, exponent outside
/// 1e-4 to 1e16.
fn py_repr(value: f64) -> String {
    if value == 0.0 {
        return if value.is_sign_negative() {
            "-0.0"
        } else {
            "0.0"
        }
        .to_owned();
    }
    let sci = format!("{value:e}");
    let Some((mantissa, exponent)) = sci.split_once('e') else {
        return value.to_string();
    };
    let Ok(exponent) = exponent.parse::<i32>() else {
        return value.to_string();
    };
    let (sign, mantissa) = match mantissa.strip_prefix('-') {
        Some(rest) => ("-", rest),
        None => ("", mantissa),
    };
    let digits: String = mantissa.chars().filter(|c| *c != '.').collect();
    if (-4..16).contains(&exponent) {
        let point = exponent + 1;
        if point <= 0 {
            format!(
                "{sign}0.{}{digits}",
                "0".repeat(point.unsigned_abs() as usize)
            )
        } else {
            let point = point as usize;
            if point >= digits.len() {
                format!("{sign}{digits}{}.0", "0".repeat(point - digits.len()))
            } else {
                format!("{sign}{}.{}", &digits[..point], &digits[point..])
            }
        }
    } else {
        let mantissa = if digits.len() > 1 {
            format!("{}.{}", &digits[..1], &digits[1..])
        } else {
            digits
        };
        let exp_sign = if exponent < 0 { '-' } else { '+' };
        format!("{sign}{mantissa}e{exp_sign}{:02}", exponent.unsigned_abs())
    }
}

/// prometheus_client's `floatToGoString`.
pub fn go_float(value: f64) -> String {
    if value.is_nan() {
        return "NaN".to_owned();
    }
    if value == f64::INFINITY {
        return "+Inf".to_owned();
    }
    if value == f64::NEG_INFINITY {
        return "-Inf".to_owned();
    }
    let repr = py_repr(value);
    if value > 0.0
        && let Some(dot) = repr.find('.')
        && dot > 6
    {
        let mut mantissa = format!("{}.{}{}", &repr[..1], &repr[1..dot], &repr[dot + 1..]);
        while mantissa.ends_with(['0', '.']) {
            mantissa.pop();
        }
        return format!("{mantissa}e+{:02}", dot - 1);
    }
    repr
}

/// A request's engine side timeline. Tokens count from when the simulated
/// GPU produced them, as vLLM stamps them in its engine core, not from
/// when the timer woke the stream that sends them.
#[derive(Debug, Clone, Copy)]
struct Admitted {
    arrival: Instant,
    scheduled: Instant,
    prompt: usize,
    cached: usize,
}

/// Records one request's lifecycle; dropping it unfinished counts an abort.
#[derive(Debug)]
pub struct RequestTracker {
    metrics: Arc<Metrics>,
    max_tokens: usize,
    admitted: Option<Admitted>,
    first: Option<Instant>,
    last: Option<Instant>,
    generated: usize,
    preemptions: u32,
    finished: bool,
}

impl RequestTracker {
    pub fn admitted(&mut self, seq: &Sequence) {
        let m = &self.metrics;
        let prompt = seq.prompt_tokens as u64;
        let cached = seq.cached_tokens as u64;
        m.prompt_tokens.fetch_add(prompt, Ordering::Relaxed);
        m.prompt_tokens_cached.fetch_add(cached, Ordering::Relaxed);
        m.computed_tokens
            .fetch_add(prompt - cached, Ordering::Relaxed);
        self.admitted = Some(Admitted {
            arrival: seq.arrival,
            scheduled: seq.scheduled,
            prompt: seq.prompt_tokens,
            cached: seq.cached_tokens,
        });
    }

    /// Counts a token the engine produced at `at`, after `preemptions`
    /// preemptions so far.
    pub fn token(&mut self, at: Instant, preemptions: u32) {
        let m = &self.metrics;
        match (self.last, self.admitted) {
            (Some(last), _) => m.inter_token_latency.observe_duration(at - last),
            (None, Some(admitted)) => m
                .time_to_first_token
                .observe_duration(at - admitted.arrival),
            (None, None) => {}
        }
        self.first.get_or_insert(at);
        self.last = Some(at);
        self.generated += 1;
        self.preemptions = preemptions;
        m.generation_tokens.fetch_add(1, Ordering::Relaxed);
    }

    pub fn finish(mut self, reason: &FinishReason) {
        self.record(reason.as_str());
    }

    fn record(&mut self, reason: &str) {
        self.finished = true;
        let m = &self.metrics;
        if let Some(i) = FINISH_REASONS.iter().position(|r| *r == reason) {
            m.success[i].fetch_add(1, Ordering::Relaxed);
        }
        let Some(a) = self.admitted else { return };
        let now = Instant::now();
        let first = self.first.unwrap_or(now);
        let last = self.last.unwrap_or(now).max(first);
        let end = if reason == "abort" {
            now.max(last)
        } else {
            last
        };
        m.e2e_request_latency.observe_duration(end - a.arrival);
        m.request_queue_time
            .observe_duration(a.scheduled - a.arrival);
        m.request_inference_time.observe_duration(end - a.scheduled);
        m.request_prefill_time.observe_duration(first - a.scheduled);
        m.request_decode_time.observe_duration(last - first);
        if self.generated > 1 {
            let per_token = (last - first).as_secs_f64() / (self.generated - 1) as f64;
            m.request_time_per_output_token.observe(per_token);
        }
        m.request_prompt_tokens.observe(a.prompt as f64);
        m.request_generation_tokens.observe(self.generated as f64);
        m.request_max_num_generation_tokens
            .observe(self.max_tokens as f64);
        m.request_params_n.observe(1.0);
        m.request_params_max_tokens.observe(self.max_tokens as f64);
        m.request_num_preemptions
            .observe(f64::from(self.preemptions));
        m.request_prefill_kv_computed_tokens
            .observe((a.prompt - a.cached) as f64);
    }
}

impl Drop for RequestTracker {
    fn drop(&mut self) {
        if !self.finished {
            self.record("abort");
        }
    }
}

#[cfg(test)]
fn small() -> EngineConfig {
    EngineConfig {
        num_gpu_blocks: 10,
        ..EngineConfig::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn floats_render_like_prometheus_client() {
        let cases: &[(f64, &str)] = &[
            (0.0, "0.0"),
            (-0.0, "-0.0"),
            (1.0, "1.0"),
            (0.5, "0.5"),
            (0.0731, "0.0731"),
            (0.001, "0.001"),
            (1e-05, "1e-05"),
            (1.5e-07, "1.5e-07"),
            (123456.0, "123456.0"),
            (1234567.0, "1.234567e+06"),
            (1000000.0, "1e+06"),
            (100000.0, "100000.0"),
            (20000.0, "20000.0"),
            (1792579709.233452, "1.792579709233452e+09"),
            (1792579709.0, "1.792579709e+09"),
            (1e16, "1e+16"),
            (1.5e16, "1.5e+16"),
            (12345678.9, "1.23456789e+07"),
            (-2.5, "-2.5"),
            (-1234567.0, "-1234567.0"),
            (f64::INFINITY, "+Inf"),
            (f64::NEG_INFINITY, "-Inf"),
            (3.3e-09, "3.3e-09"),
            (0.1 + 0.2, "0.30000000000000004"),
            (999999.9999, "999999.9999"),
            (7680.0, "7680.0"),
            (1e22, "1e+22"),
            (123.456, "123.456"),
        ];
        for (value, want) in cases {
            assert_eq!(go_float(*value), *want, "{value:?}");
        }
        assert_eq!(go_float(f64::NAN), "NaN");
    }

    #[test]
    fn token_buckets_follow_the_one_two_five_series() {
        assert_eq!(token_buckets(100), [1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]);
        assert_eq!(token_buckets(32_768).last(), Some(&20_000.0));
    }

    #[test]
    fn histogram_buckets_are_cumulative_with_inf() {
        let metrics = Metrics::new("m", 100, &small());
        for v in [0.0005, 0.003, 0.003, 9999.0] {
            metrics.time_to_first_token.observe(v);
        }
        let text = metrics.render(&EngineSnapshot::default());
        assert!(text.contains(
            "vllm:time_to_first_token_seconds_bucket{engine=\"0\",le=\"0.001\",model_name=\"m\"} 1.0\n"
        ));
        assert!(text.contains(
            "vllm:time_to_first_token_seconds_bucket{engine=\"0\",le=\"0.005\",model_name=\"m\"} 3.0\n"
        ));
        assert!(text.contains(
            "vllm:time_to_first_token_seconds_bucket{engine=\"0\",le=\"+Inf\",model_name=\"m\"} 4.0\n"
        ));
        assert!(text.contains(
            "vllm:time_to_first_token_seconds_count{engine=\"0\",model_name=\"m\"} 4.0\n"
        ));
    }

    #[test]
    fn counters_get_total_and_created_samples() {
        let text = Metrics::new("m", 100, &small()).render(&EngineSnapshot::default());
        assert!(text.contains("# TYPE vllm:prompt_tokens_total counter\n"));
        assert!(text.contains("# TYPE vllm:prompt_tokens_created gauge\n"));
        assert!(text.contains("vllm:request_success_total{engine=\"0\",finished_reason=\"abort\",model_name=\"m\"} 0.0\n"));
    }

    #[test]
    fn label_values_are_escaped() {
        let text = Metrics::new("a\"b\\c\nd", 10, &small()).render(&EngineSnapshot::default());
        assert!(text.contains("model_name=\"a\\\"b\\\\c\\nd\""));
    }

    #[test]
    fn unfinished_tracker_counts_an_abort() {
        let metrics = Arc::new(Metrics::new("m", 100, &small()));
        drop(metrics.tracker(5));
        metrics.tracker(5).finish(&FinishReason::Length);
        assert_eq!(metrics.finished("abort"), 1);
        assert_eq!(metrics.finished("length"), 1);
        assert_eq!(metrics.finished("stop"), 0);
    }
}

#[cfg(test)]
mod concurrency {
    use super::*;

    #[test]
    fn concurrent_scrapes_never_see_a_torn_histogram() {
        let metrics = Arc::new(Metrics::new("m", 100, &small()));
        let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let writers: Vec<_> = (0..3)
            .map(|_| {
                let metrics = Arc::clone(&metrics);
                let stop = Arc::clone(&stop);
                std::thread::spawn(move || {
                    let mut i = 0u64;
                    while !stop.load(Ordering::Relaxed) {
                        metrics.time_to_first_token.observe((i % 7) as f64 * 0.003);
                        i += 1;
                    }
                })
            })
            .collect();
        let mut torn = 0;
        let mut scrapes = 0;
        let started = std::time::Instant::now();
        while started.elapsed() < Duration::from_secs(1) {
            scrapes += 1;
            let text = metrics.render(&EngineSnapshot::default());
            let mut last = 0.0f64;
            let mut count = None;
            for line in text
                .lines()
                .filter(|l| l.starts_with("vllm:time_to_first_token_seconds_"))
            {
                let value: f64 = line.rsplit_once(' ').unwrap().1.parse().unwrap();
                if line.contains("_bucket") {
                    if value < last {
                        torn += 1;
                    }
                    last = value;
                } else if line.contains("_count") {
                    count = Some(value);
                }
            }
            if count.is_some_and(|c| c != last) {
                torn += 1;
            }
        }
        stop.store(true, Ordering::Relaxed);
        for writer in writers {
            writer.join().unwrap();
        }
        assert!(scrapes > 100);
        assert_eq!(
            torn, 0,
            "{torn} of {scrapes} scrapes saw an inconsistent histogram"
        );
    }
}
