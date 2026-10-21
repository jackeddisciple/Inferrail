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

> **Status: v0.1.3, gateway and baselines.** The gateway is usable: it
> proxies chat and completions to a fleet of engines, streams responses
> untouched, retries before the first byte, ejects failing replicas,
> discovers replicas through DNS and routes with four baseline policies.
> The KV index and the cost model come next, so the cache aware policies
> currently route by their documented fallback, power of two choices. See
> [Roadmap](#roadmap).

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

## What is in v0.1.3

| Piece | What it does |
| --- | --- |
| `inferrail-gateway` | OpenAI compatible proxy on hyper: chat and completions, streaming pass through, model list, request IDs, retries before the first byte, health checks, outlier ejection, breakers, DNS discovery, graceful shutdown |
| `inferrail-policy` | The `RoutingPolicy` trait and the round robin, random, least request and power of two choices baselines, chosen per model |
| `inferrail-control` | Lock free replica registry, health state with exponential ejection backoff, the active health checker and DNS discovery |
| `inferrail-core` | Lossless OpenAI wire types, an incremental SSE decoder and validated YAML configuration |
| `inferrail-sim` | A CPU mock of vLLM v0.31.0 with the same HTTP API, streaming bytes and Prometheus metrics, a calibrated latency model and a real prefix cache |
| Other crates | `tokenize`, `index`, `events`, `load` and `calibrate` compile as stubs with the boundaries the router will use |
| `bench/` | Python 3.12 package managed by uv; today it holds the OpenAI SDK suites, run against the simulator and through the gateway |
| CI | fmt, clippy with warnings denied, nextest, doc tests, ruff, pytest, `cargo audit`, `cargo machete`, and the official OpenAI SDK against the simulator and the gateway on every push |

## Quick start

Requirements: Rust (the pinned 1.98.1 toolchain installs itself through
rustup), [uv](https://docs.astral.sh/uv/) for the Python side, and optionally
[`cargo-nextest`](https://nexte.st).

Start three simulated engines and a gateway in front of them:

```sh
cargo build --release -p inferrail-sim -p inferrail-gateway
for port in 8001 8002 8003; do
  ./target/release/inferrail-sim --port $port --enable-prompt-tokens-details &
done
./target/release/inferrail-gateway --config deploy/local.yaml
```

Talk to the gateway like any vLLM server. `sim` is an alias the gateway
resolves to `inferrail/sim-8b`:

```sh
curl -si localhost:8080/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model": "sim",
       "messages": [{"role": "user", "content": "Hello"}],
       "max_tokens": 16}'
```

The response headers name the replica that served it (`x-inferrail-replica`)
and the request ID the engine used (`x-request-id`). The official SDK works
by changing nothing but the base URL:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="unused")
stream = client.chat.completions.create(
    model="sim",
    messages=[{"role": "user", "content": "Hello"}],
    max_tokens=16,
    stream=True,
    stream_options={"include_usage": True},
)
for chunk in stream:
    print(chunk)
```

Kill one of the simulators while traffic is flowing: only responses it was
already sending can break. The gateway retries every request that had not
started on another replica, ejects the dead one within milliseconds and
readmits it once it is back.
`inferrail-gateway --check --config <file>` validates a config and exits.

## The gateway

**Routes.** `POST /v1/chat/completions` and `POST /v1/completions` are
proxied; `/v1/models` and `/health` are answered locally to GET and HEAD.
Other paths get a 404 and wrong methods a 405 with `Allow`, in the OpenAI
error shape vLLM uses. A rejected request that holds its body back behind
`Expect: 100-continue` is answered at once, without being asked to upload. HTTP/1.1 and h2c are accepted from clients; engines are reached
over pooled HTTP/1.1 keep alive connections, or h2c with `upstream.http2`.

**Pass through.** The request body is read once, within
`server.request_timeout` (a body that stops arriving gets a 408), and
forwarded byte for byte; only the `model` field is parsed. When a client asks for a gateway
alias, or omits the model while the gateway serves exactly one, only those
bytes are rewritten so the engine sees a name it serves. Responses stream
back frame by frame as they arrive, with headers intact except hop by hop
ones, and the gateway holds no per token state. Every response carries
`x-request-id` (the client's own when well formed, which vLLM also uses as
the response id), and proxied ones name their replica in
`x-inferrail-replica`. Request bodies are never logged unless
`logging.log_prompts` is on.

**Models.** `GET /v1/models` lists the configured models and aliases in
config order, with the metadata the engines published (context length,
root, permissions). The health checker refreshes those cards in the
background, so listing never calls an engine.

**Retries.** A request moves to another replica only before the first
response byte reaches the client: when the connection fails, the replica
closes before answering, it answers 502, 503 or 504, or it fails its health
checks while the request is still waiting for an answer. Untried replicas come
first, attempts stop at `retry.max_attempts` with exponential backoff, and
everything shares the `server.request_timeout` deadline. A stream that
breaks after it started is never replayed. The reasoning is in
[ADR 003](docs/adr/003-gateway-proxy-and-failure-semantics.md).

**Health and ejection.** Every replica is probed at `/health` each
`health.interval` and leaves rotation after `unhealthy_threshold` failed
probes. Independently, `unhealthy_threshold` consecutive failed requests
eject it for `ejection_backoff`, doubling on every repeat up to
`max_ejection_backoff`. Failed requests never eject more than
`max_ejection_percent` (half by default) of the replicas of a model that
pass their health checks, so requests that make every engine fail cannot
empty the pool. Under load a
crashed replica is out of rotation in milliseconds; it comes back once
probes pass and its window ends.

**Breakers.** `max_concurrency` caps requests in flight to a replica and
`max_pending` caps those still waiting for a response head. Caps never
overshoot, even under races. A full replica is skipped and a full fleet gets
an immediate 503 with `Retry-After`, so a replica that hangs stops absorbing
traffic instead of making every client wait.

**Discovery.** Replicas come from the config, from DNS groups, or both. Each
address a group name resolves to becomes a replica named
`<group>-<ip>-<port>`. Groups are resolved before the first request and then
every `refresh`; members that stay keep their health and counters, members
that leave finish their streams, and a failed lookup keeps the last known
set.

**Cancellation and shutdown.** A client that disconnects, whether streaming,
queued in the engine or waiting for a full response, frees the engine's slot
within 100 ms. Idle client connections are closed after
`server.header_read_timeout`. On SIGTERM the gateway stops accepting, lets
running streams finish for up to `server.shutdown_timeout`, then closes what
is left; a request a client sends on an idle connection at that very moment
can be lost, so put the gateway behind something that retries, or drain it
from the load balancer first.

## Routing policies

Every policy implements one `RoutingPolicy` trait and is chosen per model in
the config, so baselines and the main policy run in the same binary on the
same candidates. The gateway only ever offers a policy replicas that serve
the model, are healthy and have room. See
[ADR 004](docs/adr/004-routing-policies.md).

| Policy | Rule | Status |
| --- | --- | --- |
| `round_robin` | next replica in turn | available |
| `random` | uniform choice | available |
| `least_request` | fewest requests in flight, ties broken at random | available |
| `p2c` | two random replicas, the less loaded wins | available |
| `prefix_hash` | consistent hash on the first blocks | needs the KV index; routes as `p2c` meanwhile |
| `kv_aware` | weighted cache hit and load score | needs the KV index; routes as `p2c` meanwhile |
| `cost_model` | lowest estimated time to first token | needs the KV index; routes as `p2c` meanwhile |

Randomized policies are seeded from `routing.seed`, so a fixed seed replays
the same choices. A choice takes 2 to 106 ns on an Apple silicon laptop, from
round robin to least request over 256 replicas.

## The simulator

Real GPUs are rented, not owned, so most development and every CI run happen
against `inferrail-sim`. It has to be faithful enough that a router tuned on
it behaves the same on vLLM.

**Wire format.** Chat and completion responses, stream chunks, the usage
chunk, `data: [DONE]`, model cards and error bodies match vLLM v0.31.0 byte
for byte, including which fields are omitted and which are explicit `null`.
Request validation follows vLLM as well; for example a non empty
`stream_options` without `stream: true` is a 400. Parity tests compare the
simulator against fixtures derived from that version's source.

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
KV events and caching of generated blocks. Requests that need `n > 1`,
logprobs, `echo` or `best_of` get a 400 naming the parameter instead of an
answer that silently differs from vLLM; stop strings are ignored.

## Configuration

The gateway reads one YAML file. Only `models` and at least one replica or
DNS group are required; everything else has a safe default, unknown keys are
rejected, and every problem is reported at once with its path.

```yaml
models:
  - name: meta-llama/Llama-3.1-8B-Instruct
    aliases: [llama-8b]
    policy: least_request
replicas:
  - id: gpu-0
    url: http://10.0.0.10:8000
    models: [meta-llama/Llama-3.1-8B-Instruct]
    max_concurrency: 256
    max_pending: 32
discovery:
  dns:
    - name: llama-pool
      url: http://vllm.inference.svc.cluster.local:8000
      models: [meta-llama/Llama-3.1-8B-Instruct]
      refresh: 2s
routing:
  default_policy: cost_model
  seed: 42
health:
  interval: 1s
  timeout: 500ms
  unhealthy_threshold: 3
  ejection_backoff: 5s
retry:
  max_attempts: 3
  backoff: 25ms
```

See [`deploy/inferrail.example.yaml`](deploy/inferrail.example.yaml) for every
option with its default, and [`deploy/local.yaml`](deploy/local.yaml) for the
quick start fleet.

## Testing

`make check` runs formatting, clippy with warnings denied, ruff, every Rust
test and the Python tests. `make audit` checks dependencies against the
RustSec advisory database and for unused crates. Beyond unit tests the suite
includes:

* **Property tests** for SSE decoding under any chunking, tokenizer prefix
  stability, block hash chaining, the prefix cache LRU, latency
  monotonicity, config round trips, uniform policy choices under fixed seeds,
  least request always picking a minimum, power of two choices never picking
  a unique maximum, fleet membership and replica health against naive
  reference models, and the request splice changing nothing but the model.
* **Gateway invariants**: no policy ever routes to an unhealthy replica or
  one that does not serve the model; bytes the engine sends reach the client
  unchanged under any chunking; request bodies reach the engine byte for
  byte; no request, however malformed, gets a gateway failure; a request
  succeeds whenever a healthy replica is within its attempts.
* **Fuzzing**: random wire payloads must round trip, random engine schedules
  must keep FIFO admission and clean accounting, random HTTP requests must
  get responses that obey the vLLM contract, and hostile YAML fails fast.
* **Integration tests** against three in process simulators that can be
  killed for real or made to refuse, close, hang, delay or fail on demand:
  routing, retries, ejection and readmission, DNS joins and leaves, breakers,
  disconnects, graceful shutdown, header timeouts and log hygiene.
* **SDK tests** (`make e2e`): the official OpenAI Python SDK, sync and async,
  against the simulator binary and through the gateway binary in front of
  three simulators.
* **Stress tests** (`make stress`), on an Apple silicon laptop with client,
  gateway and simulators sharing one machine:

| Scenario | Result |
| --- | --- |
| 2,000 concurrent streams through the gateway over 3 simulators | all complete in 0.3 to 0.7 s, zero errors, even spread |
| Open loop at 2,000 and 5,000 req/s, 10,000 requests each | the gateway adds 0.04 to 0.12 ms at p50 and under 0.6 ms at p99 |
| A simulator killed under 1,000 req/s, then restarted | zero errors for requests sent after the kill, ejected within 10 ms, back in rotation within 60 ms |
| A replica that hangs, with requests waiting on it | they move to another replica as soon as probes mark it down |
| Requests that make every engine fail | at most half the replicas are ejected; the pool never empties |
| A replica flapping between healthy and 503 every 100 ms at 1,000 req/s | zero client errors |
| 1,000 streams with 741 random disconnects | every disconnect aborted in the engine, nothing leaked |
| 3,000 hostile requests | zero gateway failures |
| 35,000 request soak through the gateway | resident memory grew under 10 MB |
| 640 MiB streamed to 20 clients that stop reading | about 20 MB held in total |
| An 8 MiB request body sent in one byte chunks | about 11 MB held while it arrives |
| 1,000 concurrent streams through the async SDK, direct and via the gateway | all complete with exact usage |
| 2,000 concurrent SSE streams straight to one simulator | all complete in 0.55 s, zero errors |
| 35,000 request simulator soak | resident memory grew under 20 MB |

The suite runs clean ten times in a row with every core saturated.

## Repository layout

```
crates/
  inferrail-core/        wire types, SSE codec, configuration
  inferrail-gateway/     the router: proxy, retries, health, discovery wiring
  inferrail-policy/      routing policy trait and baseline policies
  inferrail-control/     replica registry, health, ejection, DNS discovery
  inferrail-sim/         mock vLLM engine (binary and library)
  inferrail-tokenize/    chat templates, tokenization, block hashing
  inferrail-index/       KV block index
  inferrail-events/      engine KV event subscribers
  inferrail-load/        engine metrics scraping, load snapshots
  inferrail-calibrate/   engine profiler that fits the cost model
bench/                   Python tooling and the OpenAI SDK suites
tests/fixtures/          OpenAI and vLLM reference payloads
deploy/                  example and local configurations
docs/adr/                architecture decision records
```

## Roadmap

The aim is a router that serves the same traffic at the same latency target
with fewer GPUs than a cache blind balancer, while adding under 1 ms of its
own latency. Every number below is a target to prove or disprove with
committed, rerunnable benchmarks, not a claim.

* **Gateway and baselines.** Shipped in v0.1.2: OpenAI compatible proxy,
  zero copy streaming, retries before the first byte, health checks, outlier
  ejection, breakers, DNS discovery, cancellation, and four baseline
  policies behind one trait.
* **Load signals and observability.** Engine metrics scraping merged with
  the gateway's own counts, Prometheus metrics, OpenTelemetry traces with
  routing attributes, Grafana dashboards, and a benchmark harness that
  compares the baselines on the simulator.
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
  hot reload, shadow mode, an explain endpoint, API keys and TLS.
* **Advanced regimes.** Disaggregated prefill and decode pools, tiered KV
  cache across GPU, CPU and disk, and Kubernetes packaging.
* **Proof.** A benchmark report against cache blind baselines on synthetic
  workloads and replayed production traces, with targets such as p99 router
  overhead under 1 ms, 80% or better prefix hit rate on shared prompts, 40%
  lower p99 TTFT than round robin, and at least one GPU saved per four.

## License

MIT. See [LICENSE](LICENSE).
