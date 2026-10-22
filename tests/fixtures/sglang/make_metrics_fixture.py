"""Rebuild metrics.txt: an SGLang v0.5.21 /metrics page for a server with two
data parallel ranks and priority scheduling, rendered the way SGLang renders
it, by prometheus_client in multiprocess mode.

    uv run --with prometheus_client==0.26.0 python make_metrics_fixture.py > metrics.txt

Names, help text, label names and multiprocess modes are copied from
python/sglang/srt/observability/metrics_collector.py at v0.5.21; scheduler
labels come from SchedulerMetricsCollector.init_new and tokenizer labels from
TokenizerManager. Totals sit under priority="" and per priority breakdowns
under priority="<n>", as _log_gauge_queue_count writes them.
"""

import os
import sys
import tempfile

os.environ["PROMETHEUS_MULTIPROC_DIR"] = tempfile.mkdtemp()

from prometheus_client import (  # noqa: E402
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from prometheus_client.multiprocess import MultiProcessCollector  # noqa: E402

MODEL = "Qwen/Qwen3-8B"
SCHEDULER = ["model_name", "engine_type", "tp_rank", "pp_rank", "moe_ep_rank", "priority", "dp_rank"]
TOKENIZER = ["model_name", "engine_type", "priority"]
TTFT = [0.1, 0.2, 0.4, 0.6, 0.8, 1, 2, 4, 6, 8, 10, 20, 40, 60, 80, 100, 200, 400]


def gauge(name, doc, mode="mostrecent"):
    return Gauge(name, doc, SCHEDULER, multiprocess_mode=mode)


running = gauge("sglang:num_running_reqs", "The number of running requests.")
queue = gauge("sglang:num_queue_reqs", "The number of requests in the waiting queue.")
grammar = gauge(
    "sglang:num_grammar_queue_reqs", "The number of requests in the grammar waiting queue."
)
throughput = gauge("sglang:gen_throughput", "The generation throughput (token/s).")
hit_rate = gauge("sglang:cache_hit_rate", "The prefix cache hit rate.")
usage = gauge("sglang:token_usage", "The token usage.")
used = gauge("sglang:num_used_tokens", "The number of used tokens.")
available = gauge(
    "sglang:kv_available_tokens", "Number of free token slots in the KV cache pool."
)
evictable = gauge(
    "sglang:kv_evictable_tokens",
    "Number of evictable (radix-cached) token slots in the KV cache pool.",
)
retracted_now = Gauge(
    "sglang:num_retracted_reqs", "The number of retracted requests.", SCHEDULER
)
retracted = Counter(
    "sglang:num_retracted_requests_total", "Total number of retracted requests.", SCHEDULER
)
capacity = gauge(
    "sglang:max_total_num_tokens", "Maximum total number of tokens in the KV cache pool."
)
prompt = Counter(
    "sglang:prompt_tokens_total",
    "Number of prefill tokens processed.",
    [*TOKENIZER, "is_streaming"],
)
generated = Counter(
    "sglang:generation_tokens_total",
    "Number of generation tokens processed.",
    [*TOKENIZER, "is_streaming"],
)
cached = Counter(
    "sglang:cached_tokens_total",
    "Number of cached prompt tokens by source (device/host/storage).",
    [*TOKENIZER, "cache_source"],
)
requests = Counter(
    "sglang:num_requests_total", "Number of requests processed.", [*TOKENIZER, "is_streaming"]
)
ttft = Histogram(
    "sglang:time_to_first_token_seconds",
    "Histogram of time to first token in seconds.",
    [*TOKENIZER, "is_streaming"],
    buckets=TTFT,
)

RANKS = {
    "0": {"running": 3, "queue": 2, "usage": 0.25, "retracted": 1, "by_priority": (1, 2)},
    "1": {"running": 5, "queue": 0, "usage": 0.75, "retracted": 2, "by_priority": (5, 0)},
}

for dp, v in RANKS.items():
    base = {
        "model_name": MODEL,
        "engine_type": "unified",
        "tp_rank": "0",
        "pp_rank": "0",
        "moe_ep_rank": "0",
        "dp_rank": dp,
    }
    total = {**base, "priority": ""}
    running.labels(**total).set(v["running"])
    queue.labels(**total).set(v["queue"])
    for priority, count in enumerate(v["by_priority"]):
        running.labels(**{**base, "priority": str(priority)}).set(count)
        queue.labels(**{**base, "priority": str(priority)}).set(0)
    grammar.labels(**total).set(0)
    throughput.labels(**total).set(412.5)
    hit_rate.labels(**total).set(0.42)
    usage.labels(**total).set(v["usage"])
    used.labels(**total).set(int(v["usage"] * 200_000))
    available.labels(**total).set(int((1 - v["usage"]) * 150_000))
    evictable.labels(**total).set(50_000)
    retracted_now.labels(**total).set(0)
    retracted.labels(**total).inc(v["retracted"])
    capacity.labels(**total).set(200_000)

tok = {"model_name": MODEL, "engine_type": "unified", "priority": ""}
prompt.labels(**tok, is_streaming="true").inc(1000)
prompt.labels(**tok, is_streaming="false").inc(500)
generated.labels(**tok, is_streaming="true").inc(300)
generated.labels(**tok, is_streaming="false").inc(100)
cached.labels(**tok, cache_source="device").inc(600)
cached.labels(**tok, cache_source="host").inc(40)
requests.labels(**tok, is_streaming="true").inc(3)
requests.labels(**tok, is_streaming="false").inc(1)
for seconds in (0.15, 0.3, 0.9, 2.5):
    ttft.labels(**tok, is_streaming="true").observe(seconds)

registry = CollectorRegistry()
MultiProcessCollector(registry)
sys.stdout.write(generate_latest(registry).decode())
