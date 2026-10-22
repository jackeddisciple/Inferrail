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

> **Status: v0.1.5, load signals, observability and the benchmark
> harness, hardened.** The gateway proxies chat and completions to a fleet of
> engines, streams responses untouched, retries before the first byte,
> ejects failing replicas, discovers replicas through DNS and routes with
> four baseline policies. It now reads every engine's load ten times a
> second, exports Prometheus metrics, OpenTelemetry traces and a decision
> log, and ships with a dashboard and a harness that compares policies on
> the simulator. The KV index and the cost model come next, so the cache
> aware policies still route by their documented fallback, power of two
> choices. See [Roadmap](#roadmap).

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

## What is in v0.1.5

| Piece | What it does |
| --- | --- |
| `inferrail-gateway` | OpenAI compatible proxy on hyper: chat and completions, streaming pass through, model list, request IDs, retries before the first byte, health checks, outlier ejection, breakers, DNS discovery, graceful shutdown, Prometheus metrics on an admin port, OpenTelemetry traces and a sampled decision log |
| `inferrail-load` | Reads vLLM and SGLang `/metrics` pages, scrapes every replica ten times a second and publishes load snapshots that are never used once stale |
| `inferrail-policy` | The `RoutingPolicy` trait and the round robin, random, least request and power of two choices baselines, chosen per model |
| `inferrail-control` | Lock free replica registry with request and prompt token counters, health state with exponential ejection backoff, the active health checker and DNS discovery |
| `inferrail-core` | Lossless OpenAI wire types, an incremental SSE decoder and validated YAML configuration |
| `inferrail-sim` | A CPU mock of vLLM v0.31.0 with the same HTTP API, streaming bytes and Prometheus metrics, a calibrated latency model and a real prefix cache |
| Other crates | `tokenize`, `index`, `events` and `calibrate` compile as stubs with the boundaries the router will use |
| `bench/` | The benchmark harness (`inferrail-bench`): open loop workloads W1, W2 and W6, TTFT, TPOT and ITL measurement, JSON results, charts and policy comparisons on simulators, plus the OpenAI SDK suites |
| `deploy/compose`, `dashboards/` | Gateway, simulators, Prometheus, Grafana and Jaeger in one `make stack`, with the first dashboard |
| CI | fmt, clippy with warnings denied, nextest, doc tests, ruff, pytest, `cargo audit`, `cargo machete`, and the official OpenAI SDK and the harness against the simulator and the gateway on every push |

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

The gateway's metrics are on the admin port:

```sh
curl -s localhost:9090/metrics | grep inferrail_replica_engine
```

To see everything on a dashboard, bring up the full stack with Docker
(gateway, three simulators, Prometheus, Grafana and Jaeger):

```sh
make stack        # Grafana on :3000, gateway on :8080, Jaeger on :16686
make stack-down
```

To measure the gateway like a benchmark would, drive it open loop:

```sh
cd bench
uv run inferrail-bench run --base-url http://localhost:8080 --model sim \
  --workload w1 --rate 20 --requests 600 --out results/try/w1.json
uv run inferrail-bench plot results/try/w1.json --out report/try
```

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

## Load signals

The gateway knows two things about each replica's load. Its own counters
are always current: requests in flight, requests still waiting for a
response head, and estimated prompt tokens in flight and still waiting for
their first byte (four bytes of request body per token until requests are
tokenized at the edge). The engine's own view comes from its Prometheus
page: running and waiting requests, KV cache usage, prefix cache counters,
preemptions and KV capacity.

**Engines.** vLLM v0.31.0 and SGLang v0.5.21 are read natively; the engine
is recognized by its metric prefix, data parallel ranks are summed, and
SGLang's per priority breakdowns are skipped. A page that lacks a required
family is reported as unreadable rather than taken as an idle engine, so a
renamed metric after an engine upgrade shows up in the scrape counters
instead of quietly skewing routing.

**Freshness.** Every replica is scraped in parallel every
`load.scrape_interval` (100 ms), each scrape bounded by
`load.scrape_timeout` and a 4 MiB page limit. A report older than
`load.max_age` (250 ms) is treated as missing: routing degrades to the
gateway's own counts instead of trusting stale numbers. A dead, hung or
garbage spewing engine is logged once when it stops answering and once when
it recovers, and never delays the other scrapes. Under 1,000 requests per
second the oldest report any reader saw was 102 ms; with 64 replicas
scraped at once it was 112 ms. See
[ADR 005](docs/adr/005-load-collector.md).

## Observability

**Metrics.** `GET /metrics` on `server.admin_listen` (127.0.0.1:9090 by
default) serves OpenMetrics, apart from the client port:

| Metric | Labels | Meaning |
| --- | --- | --- |
| `inferrail_requests_total` | model, replica, status | inference requests by outcome: a status class, `aborted` when the engine broke a started response, `cancelled` when the client left |
| `inferrail_ttft_seconds` | model, policy, replica | arrival to first byte of a streamed response |
| `inferrail_request_duration_seconds` | model, policy, replica | arrival to the end of the response |
| `inferrail_routing_decisions_total` | model, policy, replica | the policy's choices, one per attempt |
| `inferrail_routing_decision_seconds` | policy | time to choose and take a lease |
| `inferrail_upstream_retries_total`, `inferrail_client_disconnects_total` | model | retries before the first byte, abandoned responses |
| `inferrail_request_errors_total` | model, reason | errors the gateway answered itself |
| `inferrail_ejections_total`, `inferrail_load_scrapes_total` | replica (and result) | passive ejections, engine scrapes |
| `inferrail_replica_*` | replica | health, ejection, in flight requests and prompt tokens, the engine's running, waiting, KV usage and prefix cache counters, and the age of its report |

Replica gauges are read from the fleet when Prometheus scrapes, series of
replicas that leave are removed, and recording one streamed request costs
about 160 ns.

**Traces.** With `observability.otlp_endpoint` set, every request becomes a
server span, and every attempt a `route` span (policy, candidates offered,
winner, its in flight count and engine load, decision time) and an
`upstream` client span, exported over OTLP HTTP and sampled by
`trace_sample_ratio`. A bare collector address such as
`http://collector:4318` gets the standard `/v1/traces` path. A client's
W3C `traceparent` is continued, and the engine receives the gateway's span
as its parent. A warning is logged when export starts failing and a note
when it recovers. Without an endpoint the spans are never created.

**Decision log.** With `observability.decision_log.sample_rate` above zero,
sampled requests write one JSON line per routing decision to
`decision_log.path`: the policy, the attempt, the winner and every
candidate's state and load. Requests are sampled by id, so all attempts of
a request are kept together. A writer thread drains a bounded queue; if
the disk falls behind, records are dropped and counted, never waited for.
Prompts are never logged. See [ADR 006](docs/adr/006-observability.md).

**Dashboard.** `make stack` builds the gateway and simulator image and
brings up three simulators, the gateway, Prometheus, Grafana and Jaeger,
published on localhost only.
The provisioned dashboard ([`dashboards/inferrail.json`](dashboards/inferrail.json))
shows TTFT per replica and per policy from both the gateway and the
engines, in flight requests, engine queues and KV usage per replica, each
policy's traffic share, prefix cache hit rates, decision time, retries,
ejections and errors.

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

## Benchmark harness

`bench/` holds `inferrail-bench`, which measures policies the way the
benchmark plan requires (see [ADR 007](docs/adr/007-benchmark-harness.md)):

* **Open loop.** Requests arrive on a seeded Poisson schedule and each is
  sent at its time whatever the earlier ones are doing. TTFT and end to end
  latency count from the scheduled time, and the gap between scheduled and
  actual send time is recorded, so a server that falls behind cannot hide
  its queueing.
* **Workloads.** W1: a handful of 2k to 8k token system prompts shared by
  every user, with 50 to 500 token questions. W2: multi turn chats whose
  history is resent each turn, each user thinking 10 to 20 s before the
  next question, generated or replayed from a ShareGPT file. W6: prompts that share nothing, the control
  where cache affinity must not hurt.
* **Measurements.** TTFT, time per output token, inter token latency, end
  to end latency, throughput, goodput against the default SLO (TTFT at most
  2 s, TPOT at most 100 ms), the prefix hit rate from usage and from the
  engines' own counters (read from vLLM or SGLang over the same requests,
  after the warmup), errors, send lag and requests per replica. Time per
  output token counts tokens from usage, not stream chunks.
* **Fair comparisons.** `compare` gives every policy the same requests and
  arrival times and a cold gateway and simulators per run, discards a
  warmup, and repeats with new seeds.

```sh
make baselines   # every baseline on W1, W2 and W6, 3 repetitions
make report      # charts and report.md from bench/results
```

The committed results in [`bench/results/baselines`](bench/results/baselines)
compare the four baselines on three simulators with the default latency
model of an 8B model on one 24 GB GPU. Each workload runs at about 70% of
what the fleet sustains: W1 with 32 system prompts (more than one
replica's KV cache holds) at 6 req/s, W2 with users who think 10 to 20 s
between turns at 25 req/s, and W6 at 9 req/s. Medians over three
repetitions:

| Policy | W1 TTFT p50 / p99 | W1 hit rate | W2 TTFT p50 / p99 | W2 hit rate | W6 TTFT p50 / p99 |
| --- | --- | --- | --- | --- | --- |
| `round_robin` | 510 / 1,843 ms | 0.39 | 104 / 906 ms | 0.03 | 313 / 846 ms |
| `random` | 712 / 3,392 ms | 0.39 | 134 / 1,589 ms | 0.03 | 397 / 1,656 ms |
| `least_request` | 670 / 1,899 ms | 0.42 | 127 / 1,146 ms | 0.03 | 369 / 1,034 ms |
| `p2c` | 708 / 2,256 ms | 0.40 | 131 / 1,301 ms | 0.03 | 352 / 1,105 ms |

![TTFT by policy on W1](bench/report/baselines/w1_ttft.png)

Two things stand out. Under cache pressure every cache blind policy finds
a warm prefix for only about 40% of W1's prompt tokens, close to the one
in three a replica's share of the traffic would give, and p99 TTFT
reaches 1.8 to 3.4 s, at or past the 2 s SLO. Multi turn chat is worse:
by the time a user asks again, the conversation's history has been evicted
or the turn lands on another replica, so only 3% of it is reused. That gap
is what the cache aware policies are built to close. Even among cache
blind policies the load signal matters: least request and power of two
choices balance requests in flight, which here are mostly decoding and say
little about the prefill queue, so they trail plain round robin, and
random's uneven spread is worst everywhere. No run lost a request, the
engines' own hit rates match the ones measured from usage, and the client
sent every request within about 2 ms of its schedule at p99. The full
report, with distributions and hit rates for every workload, is in
[`bench/report/baselines/report.md`](bench/report/baselines/report.md).

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
load:
  scrape_interval: 100ms
  max_age: 250ms
observability:
  otlp_endpoint: http://localhost:4318/v1/traces
  trace_sample_ratio: 0.1
  decision_log:
    path: /var/log/inferrail/decisions.jsonl
    sample_rate: 0.01
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
  a unique maximum, fleet membership, replica health and lease token
  counters against naive reference models, the request splice changing
  nothing but the model, metrics pages that never crash the parser, and
  rendered pages that always parse back to the reference totals.
* **Gateway invariants**: no policy ever routes to an unhealthy replica or
  one that does not serve the model; bytes the engine sends reach the client
  unchanged under any chunking; request bodies reach the engine byte for
  byte; no request, however malformed, gets a gateway failure; a request
  succeeds whenever a healthy replica is within its attempts; every request
  is counted in exactly one metrics outcome; metrics of departed replicas
  never outlive them, even when requests race their removal.
* **Fuzzing**: random wire payloads must round trip, random engine schedules
  must keep FIFO admission and clean accounting, random HTTP requests must
  get responses that obey the vLLM contract, and hostile YAML fails fast.
* **Integration tests** against three in process simulators that can be
  killed for real or made to refuse, close, hang, delay or fail on demand:
  routing, retries, ejection and readmission, DNS joins and leaves, breakers,
  disconnects, graceful shutdown, header timeouts and log hygiene; load
  snapshots that stay under 250 ms old, engines that die or serve hostile
  metrics pages; every metric and its labels (a snapshot test), span
  attributes and trace propagation through an in memory exporter, real OTLP
  export from the binary, and the decision log.
* **Consistency tests**: every dashboard query names a metric the gateway or
  vLLM exports, and the compose stack and its configs are valid.
* **Harness tests**: workload shapes and determinism, Poisson statistics,
  TTFT, TPOT and ITL against servers with known timing, coordinated
  omission against a server that falls behind, a blocked client loop
  showing up as lag, results, reports and a full comparison on real
  binaries.
* **SDK tests** (`make e2e`): the official OpenAI Python SDK, sync and async,
  against the simulator binary and through the gateway binary in front of
  three simulators.
* **Stress tests** (`make stress`), on an Apple silicon laptop with client,
  gateway and simulators sharing one machine:

| Scenario | Result |
| --- | --- |
| 2,000 concurrent streams through the gateway over 3 simulators | all complete in 0.3 to 0.7 s, zero errors, even spread |
| Open loop at 2,000 and 5,000 req/s, 10,000 requests each, with metrics and load scraping on | the gateway adds about 0.1 ms at p50 and under 0.25 ms at p99 |
| The same at 5,000 req/s with the decision log at full sampling and traces exported to an unreachable collector | the gateway adds 0.15 ms at p50 and 0.34 ms at p99; 15,000 decisions logged, none dropped |
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
| Engine load reports under 1,000 req/s through 3 simulators | every reader saw a report under 102 ms old (p99 100 ms); never stale |
| 64 replicas scraped every 100 ms under 2,000 req/s | oldest report 112 ms, zero errors |
| `/metrics` scraped 100 times a second during 2,000 concurrent streams | every stream completes; counters and histograms consistent on every scrape |
| 20 replicas serving 50 MB or endless metrics pages for 5 s | the healthy replica's report never older than 104 ms; each scrape capped at 4 MiB and 100 ms |
| 8,000 replicas churned through discovery | the metrics page never exceeds 40 KB and renders in 0.05 ms |
| The harness driving zero latency simulators at 300 req/s | every request sent within 2.2 ms of its schedule at p99 |

The suite runs clean ten times in a row with every core saturated.

## Repository layout

```
crates/
  inferrail-core/        wire types, SSE codec, configuration
  inferrail-gateway/     the router: proxy, retries, health, metrics, traces
  inferrail-policy/      routing policy trait and baseline policies
  inferrail-control/     replica registry, health, ejection, DNS discovery
  inferrail-load/        engine metrics parsing, scraping, load snapshots
  inferrail-sim/         mock vLLM engine (binary and library)
  inferrail-tokenize/    chat templates, tokenization, block hashing
  inferrail-index/       KV block index
  inferrail-events/      engine KV event subscribers
  inferrail-calibrate/   engine profiler that fits the cost model
bench/                   benchmark harness and the OpenAI SDK suites
  results/               raw results of committed runs
  report/                charts and reports generated from them
dashboards/              Grafana dashboard JSON
deploy/                  example, local and compose configurations
tests/fixtures/          OpenAI, vLLM and SGLang reference payloads
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
* **Load signals and observability.** Shipped in v0.1.4: vLLM and SGLang
  metrics scraped ten times a second and merged with the gateway's own
  counts, Prometheus metrics, OpenTelemetry traces with routing attributes,
  a sampled decision log, a Grafana dashboard, and a benchmark harness that
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
