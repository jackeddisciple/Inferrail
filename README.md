# Inferrail

[![CI](https://github.com/jackeddisciple/Inferrail/actions/workflows/ci.yml/badge.svg)](https://github.com/jackeddisciple/Inferrail/actions/workflows/ci.yml)
![Rust 1.98](https://img.shields.io/badge/rust-1.98-orange)
![License: MIT](https://img.shields.io/badge/license-MIT-blue)

Inferrail is a KV cache aware router and control plane for self hosted LLM
inference. It sits between OpenAI compatible clients and a fleet of vLLM or
SGLang engines, and sends every request to the replica that already holds
most of its prompt in cache, without letting any replica overload.

The goal is lower tail latency and fewer GPUs for the same traffic, proven
with benchmarks anyone can rerun.

> **Status: v0.1.0, foundations.** This release ships the workspace, the
> shared wire types and configuration, and `inferrail-sim`, a mock vLLM
> engine that the rest of the project is built and benchmarked against. The
> gateway, index and routing policies are scaffolded crates and not usable
> yet. See [Roadmap](#roadmap).

## Why this exists

Every LLM request has two phases. **Prefill** reads the whole prompt and
builds a key value cache; **decode** then produces tokens one at a time from
that cache. Engines like vLLM keep KV blocks in GPU memory after a request
finishes, so a later request with the same prefix skips most of its prefill.

That reuse only happens when the second request lands on the same replica.
Round robin and least connections ignore this: with N replicas a repeated
prefix finds a warm cache about 1/N of the time, and every miss pays for
prefill again in GPU time and time to first token (TTFT).

Prefix reuse is the normal case, not the exception:

* shared system prompts of thousands of tokens sent by every user,
* multi turn chat, where turn ten resends turns one to nine,
* agent loops that resend a growing context on every tool call,
* RAG over popular documents, and few shot evaluation batches.

The naive fixes fail in opposite directions. Hashing on the prefix piles a
hot system prompt onto one replica and melts it; pure load balancing throws
the cache away. Inferrail treats routing as a cost estimate instead: for each
replica, how long until this request's first token, given what is cached
there and what is queued there?

## What is in v0.1.0

| Piece | What it does |
| --- | --- |
| `inferrail-core` | Lossless OpenAI wire types for chat, completions, chunks, usage, models and errors; an incremental SSE decoder; validated YAML configuration |
| `inferrail-sim` | A CPU mock of vLLM v0.31.0 with the same HTTP API, streaming bytes and Prometheus metrics, a calibrated latency model and a real prefix cache |
| Other crates | `tokenize`, `index`, `events`, `policy`, `load`, `control`, `gateway`, `calibrate` compile as stubs with the boundaries the router will use |
| `bench/` | Python 3.12 package managed by uv; today it holds the OpenAI SDK compatibility and load suites |
| CI | fmt, clippy with warnings denied, nextest, doc tests, pytest, and the official OpenAI SDK against the simulator on every push |

## Quick start

Requirements: Rust (the pinned 1.98.1 toolchain installs itself through
rustup), [uv](https://docs.astral.sh/uv/) for the Python side, and optionally
[`cargo-nextest`](https://nexte.st).

```sh
cargo run --release -p inferrail-sim -- --port 8000 --enable-prompt-tokens-details
```

Talk to it like any vLLM server:

```sh
curl -s localhost:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model": "inferrail/sim-8b",
       "messages": [{"role": "user", "content": "Hello"}],
       "max_tokens": 16}'
```

or with the official SDK, changing nothing but the base URL:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
stream = client.chat.completions.create(
    model="inferrail/sim-8b",
    messages=[{"role": "user", "content": "Hello"}],
    max_tokens=16,
    stream=True,
    stream_options={"include_usage": True},
)
for chunk in stream:
    print(chunk)
```

Send the same long system prompt twice and the second response reports the
reused tokens in `usage.prompt_tokens_details.cached_tokens`, with a much
shorter time to first token.

## The simulator

Real GPUs are rented, not owned, so most development and every CI run happen
against `inferrail-sim`. It has to be faithful enough that a router tuned on
it behaves the same on vLLM.

**Wire format.** Chat and completion responses, stream chunks, the usage
chunk, `data: [DONE]`, model cards and error bodies match vLLM v0.31.0 byte
for byte, including which fields are omitted and which are explicit `null`.
Parity tests compare the simulator against fixtures derived from that
version's source.

**Endpoints.** `POST /v1/chat/completions`, `POST /v1/completions`,
`GET /v1/models`, `GET /health`, `GET /metrics`, `GET /version` and
`POST /reset_prefix_cache`. An `x-request-id` header becomes the response id,
as in vLLM.

**Latency model.** One FIFO prefill queue on a virtual timeline, so
concurrent prompts wait for each other exactly as on a single GPU. Prefill
of `u` uncached tokens attending to `c` cached ones costs

```
P(u, c) = a*u + b*u*(c + u/2) + d
```

the same shape as the router's future cost model, so calibration can recover
the parameters. Decode emits tokens at a fixed time per output token, and
deadlines are planned ahead so timer overshoot never accumulates. The
defaults approximate an 8B model on one 24 GB GPU.

**Prefix cache.** Prompts are cut into full blocks, hashed as a chain so each
hash names its whole prefix, and kept in an O(1) LRU sized in blocks. Long
prefixes are evicted from the tail first, as vLLM frees them, and a fully
cached prompt still recomputes its last block.

**Tokenizer.** A deterministic stand in whose token boundaries depend only on
the text to their left, so shared text means shared tokens. Generated text
retokenizes to the same ids, so multi turn reuse works as it does on a real
engine.

**Metrics.** Every default vLLM family the simulator can model: running and
waiting requests, KV usage, prefix cache queries and hits, token counters,
finish reasons, TTFT, inter token latency, end to end, queue, prefill and
decode time histograms, with vLLM's names, help text, labels, buckets and
`prometheus_client` text format.

**Cancellation.** A client that disconnects, whether queued, prefilling or
streaming, releases its sequence slot and KV blocks immediately and is
counted as an abort.

Useful flags (all names follow vLLM where one exists):

| Flag | Default | Meaning |
| --- | --- | --- |
| `--served-model-name` | model path | names clients may request |
| `--block-size` | 16 | tokens per KV block |
| `--num-gpu-blocks-override` | 4096 | KV cache capacity in blocks |
| `--max-num-seqs` | 256 | running sequence cap |
| `--max-model-len` | 32768 | context length |
| `--enable-prompt-tokens-details` | off | report cached tokens in usage |
| `--prefill-token-us`, `--prefill-pair-ns`, `--prefill-fixed-ms` | 100, 3.3, 5 | `a`, `b` and `d` above |
| `--decode-token-ms` | 20 | time per output token |
| `--zero-latency` | off | answer as fast as possible, for overhead benchmarks |

Not modeled yet: decode interference inside a continuous batch, preemption,
KV events, caching of generated blocks, `n > 1`, logprobs and stop strings.

## Configuration

The gateway reads one YAML file. Only `models` and `replicas` are required;
everything else has a safe default, unknown keys are rejected, and every
problem is reported at once with its path.

```yaml
models:
  - name: meta-llama/Llama-3.1-8B-Instruct
    aliases: [llama-8b]
replicas:
  - id: gpu-0
    url: http://10.0.0.10:8000
    models: [meta-llama/Llama-3.1-8B-Instruct]
routing:
  default_policy: cost_model
  epsilon: 0.25
  kappa: 0.92
```

See [`deploy/inferrail.example.yaml`](deploy/inferrail.example.yaml) for every
option.

## Testing

`make check` runs formatting, clippy with warnings denied, every Rust test
and the Python tests. Beyond unit tests the suite includes:

* **Property tests** for SSE decoding under any chunking, prefix stability of
  the tokenizer, block hash chaining, the LRU against a naive reference
  model, latency monotonicity and config round trips.
* **Fixture tests**: OpenAI payloads must round trip canonically and vLLM
  payloads byte for byte.
* **Parity tests** holding the simulator's streams, responses, errors and
  metrics to the vLLM fixtures.
* **Behaviour tests** over HTTP: cache hits cut TTFT, later turns reuse
  earlier ones, prefills queue, the sequence cap holds, and a disconnect
  frees its slot in under 100 ms.
* **SDK tests** (`make e2e`): the official OpenAI Python SDK, sync and async,
  against the simulator binary.
* **Stress tests** (`make stress`), on an Apple silicon laptop:

| Scenario | Result |
| --- | --- |
| 2,000 concurrent SSE streams | all complete in 0.55 s, zero errors |
| Open loop at 2,000 req/s for 10,000 requests | zero errors, p99 0.23 ms |
| 1,000 streams with 740 random disconnects | every abort counted, nothing leaked |
| 3,000 malformed and fuzzed bodies | zero server errors |
| 64 prompts near the 32k context limit | served, cache shared across them |
| 35,000 request soak | resident memory grew under 20 MB, within its 64 MB budget |
| 1,000 concurrent streams through the async SDK | all complete with exact usage |

## Repository layout

```
crates/
  inferrail-core/        wire types, SSE codec, configuration
  inferrail-sim/         mock vLLM engine (binary and library)
  inferrail-tokenize/    chat templates, tokenization, block hashing
  inferrail-index/       KV block index
  inferrail-events/      engine KV event subscribers
  inferrail-policy/      routing policies and the TTFT cost model
  inferrail-load/        engine metrics scraping, load snapshots
  inferrail-control/     discovery, health, admission, admin API
  inferrail-gateway/     the router binary
  inferrail-calibrate/   engine profiler that fits the cost model
bench/                   Python tooling and SDK suites
tests/fixtures/          OpenAI and vLLM reference payloads
deploy/                  example configuration
docs/adr/                architecture decision records
```

## Roadmap

The aim is a router that serves the same traffic at the same latency target
with fewer GPUs than a cache blind balancer, while adding under 1 ms of its
own latency. Every number below is a target to prove or disprove with
committed, rerunnable benchmarks, not a claim.

* **Gateway.** OpenAI compatible proxy with zero copy streaming, pooled
  upstream connections, retries only before the first byte, client
  cancellation propagated upstream, health checks and outlier ejection.
* **Baselines.** Round robin, random, least request and power of two choices
  behind one routing policy trait, so every comparison is fair and runs in
  one binary.
* **Tokens and hashes.** Real chat templates and tokenizers with byte level
  parity against Python `transformers`, and chained XXH3 128 bit block hashes
  salted by model and LoRA adapter.
* **KV index.** An approximate mode built from the router's own decisions,
  and an exact mode fed by engine KV events over ZMQ, with gap detection and
  safe fallback.
* **Cost model.** Estimated TTFT per replica as queue wait plus prefill of
  uncached tokens plus tier load time, with bounded load, KV pressure guards,
  hot prefix replication and optional session affinity.
* **Production controls.** SLO aware admission, token based rate limits,
  hot reload, shadow mode, an explain endpoint, Prometheus metrics,
  OpenTelemetry traces and Grafana dashboards.
* **Advanced regimes.** Disaggregated prefill and decode pools, tiered KV
  cache across GPU, CPU and disk, and Kubernetes packaging.
* **Proof.** A benchmark report against cache blind baselines on synthetic
  workloads and replayed production traces, with targets such as p99 router
  overhead under 1 ms, 80% or better prefix hit rate on shared prompts, 40%
  lower p99 TTFT than round robin, and at least one GPU saved per four.

## License

MIT. See [LICENSE](LICENSE).
