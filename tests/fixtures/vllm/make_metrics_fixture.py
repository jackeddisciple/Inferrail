"""Rebuild metrics.txt: the default vLLM v0.31.0 metric families, rendered by
prometheus_client the way a vLLM server renders /metrics, after a short run.

    uv run --with prometheus_client==0.26.0 python make_metrics_fixture.py > metrics.txt

Names, help text, label names and buckets are copied from
vllm/v1/metrics/loggers.py and vllm/v1/metrics/buckets.py at v0.31.0.
"""

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
MAX_MODEL_LEN = 32768
LABELS = ["model_name", "engine"]
VALUES = {"model_name": MODEL, "engine": "0"}

REQUEST_LATENCY = [
    0.3,
    0.5,
    0.8,
    1.0,
    1.5,
    2.0,
    2.5,
    5.0,
    10.0,
    15.0,
    20.0,
    30.0,
    40.0,
    50.0,
    60.0,
    120.0,
    240.0,
    480.0,
    960.0,
    1920.0,
    7680.0,
]
TTFT = [
    0.001,
    0.005,
    0.01,
    0.02,
    0.04,
    0.06,
    0.08,
    0.1,
    0.25,
    0.5,
    0.75,
    1.0,
    2.5,
    5.0,
    7.5,
    10.0,
    20.0,
    40.0,
    80.0,
    160.0,
    640.0,
    2560.0,
]
ITL = [
    0.01,
    0.025,
    0.05,
    0.075,
    0.1,
    0.15,
    0.2,
    0.3,
    0.4,
    0.5,
    0.75,
    1.0,
    2.5,
    5.0,
    7.5,
    10.0,
    20.0,
    40.0,
    80.0,
]
ITERATION = [1, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
PARAMS_N = [1, 2, 5, 10, 20]
PREEMPTIONS = [1, 2, 3, 4, 5, 10, 20]


def one_two_five(limit):
    out, exp = [], 0
    while True:
        for m in (1, 2, 5):
            value = m * 10**exp
            if value > limit:
                return out
            out.append(value)
        exp += 1


TOKENS = one_two_five(MAX_MODEL_LEN)

r = CollectorRegistry()


def gauge(name, doc, extra=()):
    return Gauge(name, doc, LABELS + list(extra), registry=r, multiprocess_mode="mostrecent")


def counter(name, doc, extra=()):
    return Counter(name, doc, LABELS + list(extra), registry=r)


def hist(name, doc, buckets):
    return Histogram(name, doc, LABELS, buckets=buckets, registry=r)


running = gauge("vllm:num_requests_running", "Number of requests in model execution batches.")
waiting = gauge("vllm:num_requests_waiting", "Number of requests waiting to be processed.")
by_reason = gauge(
    "vllm:num_requests_waiting_by_reason",
    "Number of waiting requests by reason. Reason labels: 'capacity' = waiting for scheduling "
    "capacity; 'deferred' = deferred by transient constraints (LoRA budget, KV transfer, blocked "
    "status). Sum of all reasons equals vllm:num_requests_waiting.",
    ["reason"],
)
sleep = gauge(
    "vllm:engine_sleep_state",
    "Engine sleep state; awake = 0 means engine is sleeping; awake = 1 means engine is awake; "
    "weights_offloaded = 1 means sleep level 1; discard_all = 1 means sleep level 2.",
    ["sleep_state"],
)
kv_usage = gauge("vllm:kv_cache_usage_perc", "KV-cache usage. 1 means 100 percent usage.")
queries = counter(
    "vllm:prefix_cache_queries", "Prefix cache queries, in terms of number of queried tokens."
)
hits = counter("vllm:prefix_cache_hits", "Prefix cache hits, in terms of number of cached tokens.")
ext_queries = counter(
    "vllm:external_prefix_cache_queries",
    "External prefix cache queries from KV connector cross-instance cache sharing, in terms of "
    "number of queried tokens.",
)
ext_hits = counter(
    "vllm:external_prefix_cache_hits",
    "External prefix cache hits from KV connector cross-instance cache sharing, in terms of "
    "number of cached tokens.",
)
mm_queries = counter(
    "vllm:mm_cache_queries", "Multi-modal cache queries, in terms of number of queried items."
)
mm_hits = counter(
    "vllm:mm_cache_hits", "Multi-modal cache hits, in terms of number of cached items."
)
preemptions = counter("vllm:num_preemptions", "Cumulative number of preemption from the engine.")
prompt_tokens = counter("vllm:prompt_tokens", "Number of prefill tokens processed.")
by_source = counter(
    "vllm:prompt_tokens_by_source", "Number of prompt tokens by source.", ["source"]
)
cached = counter("vllm:prompt_tokens_cached", "Number of cached prompt tokens (local + external).")
generation = counter("vllm:generation_tokens", "Number of generation tokens processed.")
success = counter(
    "vllm:request_success", "Count of successfully processed requests.", ["finished_reason"]
)
req_prompt = hist("vllm:request_prompt_tokens", "Number of prefill tokens processed.", TOKENS)
req_gen = hist("vllm:request_generation_tokens", "Number of generation tokens processed.", TOKENS)
iteration = hist(
    "vllm:iteration_tokens_total", "Histogram of number of tokens per engine_step.", ITERATION
)
max_gen = hist(
    "vllm:request_max_num_generation_tokens",
    "Histogram of maximum number of requested generation tokens.",
    TOKENS,
)
params_n = hist("vllm:request_params_n", "Histogram of the n request parameter.", PARAMS_N)
params_max = hist(
    "vllm:request_params_max_tokens", "Histogram of the max_tokens request parameter.", TOKENS
)
ttft = hist(
    "vllm:time_to_first_token_seconds", "Histogram of time to first token in seconds.", TTFT
)
itl = hist("vllm:inter_token_latency_seconds", "Histogram of inter-token latency in seconds.", ITL)
tpot = hist(
    "vllm:request_time_per_output_token_seconds",
    "Histogram of time_per_output_token_seconds per request.",
    ITL,
)
e2e = hist(
    "vllm:e2e_request_latency_seconds",
    "Histogram of e2e request latency in seconds.",
    REQUEST_LATENCY,
)
queue = hist(
    "vllm:request_queue_time_seconds",
    "Histogram of time spent in WAITING phase for request.",
    REQUEST_LATENCY,
)
inference = hist(
    "vllm:request_inference_time_seconds",
    "Histogram of time spent in RUNNING phase for request.",
    REQUEST_LATENCY,
)
prefill = hist(
    "vllm:request_prefill_time_seconds",
    "Histogram of time spent in PREFILL phase for request.",
    REQUEST_LATENCY,
)
decode = hist(
    "vllm:request_decode_time_seconds",
    "Histogram of time spent in DECODE phase for request.",
    REQUEST_LATENCY,
)
req_preempt = hist(
    "vllm:request_num_preemptions",
    "Histogram of the number of times a request was preempted.",
    PREEMPTIONS,
)
kv_computed = hist(
    "vllm:request_prefill_kv_computed_tokens",
    "Histogram of new KV tokens computed during prefill (excluding cached tokens).",
    TOKENS,
)
CACHE_CONFIG = {
    "block_size": "16",
    "cache_dtype": "auto",
    "calculate_kv_scales": "False",
    "enable_prefix_caching": "True",
    "engine": "0",
    "gpu_memory_utilization": "0.92",
    "is_attention_free": "False",
    "kv_cache_memory_bytes": "None",
    "kv_offloading_backend": "native",
    "kv_offloading_size": "None",
    "kv_sharing_fast_prefill": "False",
    "mamba_cache_dtype": "auto",
    "num_cpu_blocks": "None",
    "num_gpu_blocks": "18240",
    "num_gpu_blocks_override": "None",
    "prefix_caching_hash_algo": "sha256",
    "sliding_window": "None",
}
info = Gauge(
    "vllm:cache_config_info",
    "Information of the LLMEngine CacheConfig",
    list(CACHE_CONFIG),
    registry=r,
    multiprocess_mode="mostrecent",
)

running.labels(**VALUES).set(2)
waiting.labels(**VALUES).set(1)
by_reason.labels(**VALUES, reason="capacity").set(1)
by_reason.labels(**VALUES, reason="deferred").set(0)
sleep.labels(**VALUES, sleep_state="awake").set(1)
sleep.labels(**VALUES, sleep_state="weights_offloaded").set(0)
sleep.labels(**VALUES, sleep_state="discard_all").set(0)
kv_usage.labels(**VALUES).set(0.0731)
for metric in (ext_queries, ext_hits, mm_queries, mm_hits, preemptions):
    metric.labels(**VALUES)

requests = [  # prompt, cached, generated, ttft, itl, queue, prefill, decode
    (4128, 0, 128, 0.512, 0.021, 0.004, 0.508, 2.69),
    (4160, 4096, 64, 0.031, 0.019, 0.002, 0.029, 1.21),
    (950, 944, 200, 0.012, 0.024, 0.001, 0.011, 4.83),
]
for prompt, hit, gen, first, step, wait, pre, dec in requests:
    queries.labels(**VALUES).inc(prompt)
    hits.labels(**VALUES).inc(hit)
    prompt_tokens.labels(**VALUES).inc(prompt)
    by_source.labels(**VALUES, source="local_compute").inc(prompt - hit)
    by_source.labels(**VALUES, source="local_cache_hit").inc(hit)
    by_source.labels(**VALUES, source="external_kv_transfer").inc(0)
    cached.labels(**VALUES).inc(hit)
    generation.labels(**VALUES).inc(gen)
    iteration.labels(**VALUES).observe(prompt - hit)
    req_prompt.labels(**VALUES).observe(prompt)
    req_gen.labels(**VALUES).observe(gen)
    max_gen.labels(**VALUES).observe(gen)
    params_n.labels(**VALUES).observe(1)
    params_max.labels(**VALUES).observe(gen)
    ttft.labels(**VALUES).observe(first)
    for _ in range(gen - 1):
        itl.labels(**VALUES).observe(step)
    tpot.labels(**VALUES).observe(step)
    e2e.labels(**VALUES).observe(wait + pre + dec)
    queue.labels(**VALUES).observe(wait)
    inference.labels(**VALUES).observe(pre + dec)
    prefill.labels(**VALUES).observe(pre)
    decode.labels(**VALUES).observe(dec)
    req_preempt.labels(**VALUES).observe(0)
    kv_computed.labels(**VALUES).observe(prompt - hit)
for reason in ("stop", "length", "abort", "error", "repetition"):
    success.labels(**VALUES, finished_reason=reason).inc(3 if reason == "length" else 0)
info.labels(**CACHE_CONFIG).set(1)

print(generate_latest(r).decode(), end="")
