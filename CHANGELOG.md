# Changelog

Notable changes to Inferrail. Versions follow
[Semantic Versioning](https://semver.org).

## [0.1.4] Load Signals, Observability and the Benchmark Harness (22 October 2026)

The router can now see how busy every engine is, shows what it is doing on
a dashboard and in traces, and comes with a harness that compares policies
on the simulator.

### Added

#### `inferrail-load`

* Parsers for the `/metrics` pages of vLLM v0.31.0 and SGLang v0.5.21:
  running and waiting requests, KV cache usage, prefix cache counters,
  preemptions and KV capacity, summed over data parallel ranks. The engine
  is recognized by its metric prefix, and a page missing a required family
  is an error rather than an idle engine. A full vLLM page parses in about
  23 µs.
* A collector that scrapes every replica in parallel every
  `load.scrape_interval` (100 ms), each scrape bounded by a timeout and a
  4 MiB page limit.
* Load snapshots published per replica through `arc-swap`. A report older
  than `load.max_age` (250 ms) is treated as missing, so routing falls back
  to the gateway's own counts instead of trusting stale data.

#### `inferrail-gateway`

* Prometheus metrics in OpenMetrics format on `server.admin_listen` at
  `/metrics`, which is now bound: requests by model, replica and outcome,
  TTFT and duration histograms by model, policy and replica, routing
  decisions and decision time, retries, client disconnects, ejections,
  scrape results, and per replica gauges for health, in flight requests and
  prompt tokens, and the engine's own running, waiting, KV usage, prefix
  cache counters and report age. Series of replicas that leave the fleet
  are removed.
* OpenTelemetry tracing over OTLP HTTP when `observability.otlp_endpoint` is
  set: a server span per request, and per attempt a `route` span with the
  policy, candidates, winner, its load and the decision time, and an
  `upstream` client span. The client's W3C trace context is continued and
  passed on to the engine. Without an endpoint the spans cost nothing and
  trace headers pass through untouched.
* A sampled decision log: JSON lines with the policy, attempt, winner and
  every candidate's load, sampled by request id and written by a separate
  thread through a bounded channel, never holding up a request and never
  containing prompt text.
* Per replica counts of estimated prompt tokens in flight and still waiting
  for their first response byte.

#### Configuration

* `load` section: `enabled`, `scrape_interval`, `scrape_timeout` and
  `max_age`, validated so one slow scrape cannot make a replica look stale.
* `observability` section: `otlp_endpoint`, `service_name`,
  `trace_sample_ratio` and `decision_log` with `path` and `sample_rate`.

#### Benchmark harness (`bench/`)

* `inferrail-bench run`, `compare` and `plot`. Open loop Poisson arrivals,
  with latency measured from each request's scheduled time and send lag
  recorded, so a slow server cannot hide its queueing.
* Workloads W1 (shared system prompts), W2 (interleaved multi turn chat,
  optionally replayed from ShareGPT) and W6 (unique prompts).
* A streaming client that records TTFT, time per output token, inter token
  latency, usage with cached tokens and the replica that served each
  request.
* Results as JSON with the environment (commit, binary versions, config
  hash, machine), a summary with percentiles, goodput against the TTFT and
  TPOT SLO and both the usage and engine reported hit rates, and every
  request in columns.
* `compare` runs every policy on the same schedules with a cold gateway and
  simulators per run; `make baselines` reproduces the committed comparison
  of the four baselines in `bench/results/baselines`, and `make report`
  regenerates `bench/report/baselines`.

#### Deployment

* `deploy/compose`: gateway, three simulators, Prometheus, Grafana and
  Jaeger with `make stack`, and the first dashboard in
  `dashboards/inferrail.json` showing TTFT, load and routing per replica.

#### Tests

* Parser tests on the vLLM and SGLang fixtures, properties that arbitrary
  input never panics and that rendered pages parse to reference totals, a
  reference model for lease token counters, and collector tests on a paused
  clock.
* Integration tests for snapshot freshness, killed and hostile engines,
  metrics, a snapshot of every metric name and label set, span attributes
  and trace propagation, OTLP export from the binary, and the decision log.
* A test that every dashboard query names a metric the gateway or vLLM
  exports, and harness tests for workloads, Poisson statistics, timing,
  coordinated omission and reports.
* Stress tests: snapshot ages under load, 64 replicas scraped at once,
  metrics scraped during 2,000 streams, hostile metrics pages, discovery
  churn, overhead with every observability feature on, and the harness at
  300 requests per second.
* ADR 005 (load collector), ADR 006 (observability) and ADR 007 (benchmark
  methodology).

### Changed

* tokio moves to 1.54.1, since 1.54.0 was yanked.
* The gateway binary installs its logging through the new telemetry setup;
  routing spans never appear in the log output.
* The e2e fixtures share process management with the harness.

## [0.1.3] Bug fixes and Improvements (21 October 2026)

Fixes from a second sweep over the gateway, policies and control plane.

### Fixed

* Requests that made engines fail could passively eject every replica of a
  model, turning a bad request pattern into an outage. Passive ejection now
  takes out at most `health.max_ejection_percent` of the replicas passing
  health checks (half by default) and never a sole replica.
* Requests already waiting on a replica that then failed its health checks
  waited for the whole request timeout and failed, although nothing had
  reached the client. They now move to another replica at once.
* Request bodies had no read deadline, so a stalled upload held its
  connection forever. Reading now shares `server.request_timeout` and
  answers 408.
* A body sent in many tiny chunks cost about forty times its size in memory
  (307 MiB for an 8 MiB body in one byte chunks). It now costs about its
  size.
* Clients sending `Expect: 100-continue` were invited to upload bodies the
  gateway was about to reject with a 404, 405 or 413. Those answers now
  come at once.
* HEAD on `/health` and `/v1/models` returned 405; it now answers as GET
  does.
* Bodies starting with a byte order mark, which Python engines accept, got
  a 400. The mark is now skipped when reading the model and forwarded as
  sent.
* DNS groups were resolved one after another at startup, so a slow DNS
  server delayed startup by every group's timeout in turn. They now resolve
  together.
* `--check` accepted configs that startup refuses, such as https replicas.
* Health and stress tests measured ejection windows on a clock other than
  the gateway's.

### Added

* `health.max_ejection_percent`, the cap on passive ejection.
* Tests for each fix, and a stress test for bodies sent in tiny chunks.

## [0.1.2] OpenAI Compatible Gateway, Baseline Routing Policies and Failure Handling (21 October 2026)

The gateway becomes usable: it routes chat and completions across a fleet
of engines with a choice of baseline policies, streams responses untouched,
and keeps clients whole when replicas crash, hang, flap or disappear.

### Added

#### `inferrail-gateway`

* OpenAI compatible proxy on hyper for `POST /v1/chat/completions` and
  `POST /v1/completions`, plus `GET /v1/models` and `GET /health`, over
  HTTP/1.1 and h2c. Other routes and methods get OpenAI shaped 404 and 405
  answers.
* Request bodies forwarded byte for byte. Only the `model` field is read,
  and only its bytes are rewritten when a client uses a gateway alias or
  omits the model while one model is served.
* Streaming pass through: engine frames reach the client as they arrive,
  with no per token state and memory bounded by backpressure.
* Request IDs: a well formed client `x-request-id` is kept, otherwise one is
  minted; it reaches the engine and comes back on every response, along
  with `x-inferrail-replica` on proxied ones.
* Retries on another replica only before the first response byte, for
  connect failures, early closes and 502, 503 or 504 answers, bounded by
  `retry.max_attempts`, exponential backoff and the request deadline.
* Passive outlier ejection after consecutive failures, with ejection windows
  that double up to a cap and reset after a clean stretch.
* Model list served from memory, built from the cards each engine publishes
  and refreshed alongside health probes, aliases included.
* Graceful shutdown bounded by `server.shutdown_timeout`, a header read
  timeout against slow clients, and request bodies drained before local
  error answers so uploading clients get the error rather than a reset.
* `--check` validates a config and exits.

#### `inferrail-policy`

* The `RoutingPolicy` trait and `RouteContext`, with `round_robin`,
  `random`, `least_request` and `p2c`, chosen per model. Randomness is a lock
  free SplitMix64 seeded per model from `routing.seed`.
* `prefix_hash`, `kv_aware` and `cost_model` route as `p2c`, their
  documented fallback, until the KV index exists, with a startup warning.

#### `inferrail-control`

* Replica registry swapped atomically for lock free reads, with in flight
  and pending counters whose caps hold under any race.
* Health state combining active probes with passive ejection, and the
  active checker that probes every replica in parallel each interval.
* DNS discovery through a pluggable resolver: groups are resolved before the
  first request and on every refresh, survivors keep their state, leavers
  finish their streams and failed lookups keep the last known members.

#### Configuration

* `upstream` section for the connection pool: connect timeout, idle timeout,
  idle connections per host and h2c.
* `discovery.dns` groups, `replicas[].max_pending`, and
  `server.header_read_timeout` and `server.shutdown_timeout`.
* `deploy/local.yaml` for three local simulators behind one gateway.

#### Tests

* Property tests for every policy, fleet membership and replica health
  against naive reference models, the body splice, stream byte equality
  under any chunking, request body equality, gateway HTTP fuzzing, and the
  rule that no policy ever reaches an unhealthy replica or one that does not
  serve the model.
* Integration tests against three in process simulators that can be killed
  for real or made to refuse, close, hang, delay or fail on demand.
* Stress suite for the gateway: 2,000 concurrent streams, open loop overhead
  against a direct simulator at 2,000 and 5,000 req/s, a simulator killed
  and restarted under load, a flapping replica, a disconnect storm, hostile
  requests, a soak and slow readers.
* The OpenAI SDK suites now also run through the gateway binary in front of
  three simulators.
* ADR 003 (gateway proxy and failure semantics) and ADR 004 (routing
  policies).

### Changed

* `replicas` may be empty when DNS groups serve every model, and the "no
  replica serves" check counts DNS groups.
* The SDK suites moved from `bench/tests/sim` to `bench/tests/e2e`; `make e2e`,
  `make stress` and the CI e2e job build and drive the gateway as well.

## [0.1.1] Reliability, Security and vLLM Parity Hardening (21 October 2026)

A hardening pass over everything in 0.1.0, driven by a new round of
property, fuzz, mutation and load testing.

### Fixed

* SSE decoder: after an invalid UTF-8 line it kept already decoded lines in
  its buffer, so a caller that kept pushing saw events twice, and an
  oversized line kept growing the buffer. Errors are now terminal. A
  leading byte order mark is stripped, as the event stream spec requires.
* Wire types: some JSON floats changed in their last digit after a parse
  and serialize cycle. Floats now round trip exactly.
* Simulator: a streamed request held its sequence slot and KV blocks until
  the client read `[DONE]`. It now finishes at its last token, as in vLLM.
* Simulator: decoding in zero latency mode never yielded to the runtime and
  could starve other requests on the same worker thread.
* Simulator: `stream_options` without `stream: true` was accepted. It now
  gets vLLM's 400 with `param` set to `stream_options`.
* Simulator: `logprobs`, `prompt_logprobs`, `echo` and `best_of` were
  silently ignored. They now get a 400 that names the parameter.
* Simulator: a wrong HTTP method got an empty 405; it now gets vLLM's JSON
  body.
* Simulator: histogram `_count` is derived from the buckets, so a scrape
  can never see it below a cumulative bucket.
* Config: replica URLs with embedded credentials and replicas that list a
  model twice were accepted; both are now rejected. An alias equal to its
  own model name gets a clear message.

### Changed

* Config YAML is parsed by `serde-saphyr`, a maintained pure Rust parser,
  instead of `serde_yaml_ng`, which pulled in the unmaintained
  `unsafe-libyaml` (`RUSTSEC-2024-0449`). Deeply nested input now fails in
  milliseconds instead of about 30 seconds, and errors report line and
  column.
* `Sim::new` validates its config and returns an error instead of failing
  later inside a request.
* The unused `bytes` dependency is gone and `futures` is only used by tests.

### Added

* Property and fuzz suites: wire types must round trip whatever they
  accept; random engine schedules with cancellations must keep FIFO
  admission, the sequence cap and clean accounting; random chat requests
  must get responses that obey the vLLM contract; streamed and full
  responses must carry the same text; hostile YAML must fail fast.
* `make audit` and a CI job that runs `cargo audit` and `cargo machete`, and
  ruff lint and format checks for the Python code.

## [0.1.0] Initial Release: Core Platform and vLLM Compatible Simulator (21 October 2026)

The first release: project foundations and a faithful vLLM simulator.

### Added

* Cargo workspace with ten crates: `inferrail-core`, `tokenize`, `index`,
  `events`, `policy`, `load` and `control` as libraries, and `gateway`, `sim`
  and `calibrate` as binaries. Toolchain pinned to Rust 1.98.1, edition 2024,
  with workspace lints.
* `bench/` Python 3.12 package managed by uv, with pytest.
* `make check`, `test`, `bench`, `e2e`, `stress` and `report` targets.
* GitHub Actions CI: fmt, clippy with warnings denied, nextest, doc tests,
  pytest, and the official OpenAI SDK against the simulator.
* ADR 001 (Rust for the data plane, Python for benchmarks) and ADR 002
  (lossless wire types).

#### `inferrail-core`

* OpenAI wire types for chat and completion requests, responses, stream
  chunks, usage with cached token details, model lists and errors, including
  vLLM extensions.
* `Nullable<T>` that keeps absent, `null` and set fields apart, plus
  passthrough of unknown fields and exact JSON numbers, so payloads round
  trip without loss.
* Incremental SSE decoder that accepts the stream in any chunking.
* Validated `ReplicaId` and `ModelId` types.
* YAML configuration for server, models, replicas, routing, health, retry and
  logging, with safe defaults, rejection of unknown keys, human readable
  durations, and cross field validation that reports every problem with its
  path. Example config in `deploy/inferrail.example.yaml`.

#### `inferrail-sim`

* Mock vLLM v0.31.0 engine serving `/v1/chat/completions`,
  `/v1/completions`, `/v1/models`, `/health`, `/metrics`, `/version` and
  `/reset_prefix_cache`, with vLLM status codes and error bodies.
* Streaming and full responses that match vLLM byte for byte, including the
  role chunk, finish chunk, optional usage chunk and `data: [DONE]`.
* Latency model `P(u, c) = a*u + b*u*(c + u/2) + d` with a drift free FIFO
  prefill queue, a running sequence cap and fixed time per output token.
* Prefix cache of chained block hashes in an O(1) LRU that evicts long
  prefixes from the tail.
* Deterministic prefix stable tokenizer and Llama 3 style chat template;
  generated text retokenizes identically.
* Prometheus metrics with vLLM names, help, labels, buckets and text format.
* Immediate release of slots and KV blocks on client disconnect, counted as
  aborts.
* CLI with vLLM flag names and a zero latency mode.

#### Tests

* Unit, property, fixture and parity tests across core and the simulator.
* Behaviour tests for cache reuse, multi turn reuse, queueing, the sequence
  cap, cancellation within 100 ms and metric accounting.
* Stress suite: 2,000 concurrent streams, 10,000 open loop requests,
  disconnect storms, malformed bodies, long prompts and a memory soak.
* OpenAI Python SDK suite, sync and async, against the simulator binary.

[0.1.4]: https://github.com/jackeddisciple/Inferrail/tree/main
[0.1.3]: https://github.com/jackeddisciple/Inferrail/tree/c6fe3ff568d5f9033b86294b7281f49ed2bcb7cc
[0.1.2]: https://github.com/jackeddisciple/Inferrail/tree/d7b4d58b108f339496938df7649bd66592b2b8d9
[0.1.1]: https://github.com/jackeddisciple/Inferrail/tree/664e148a663636837ea98cb32ddc8676dd8b7093
[0.1.0]: https://github.com/jackeddisciple/Inferrail/tree/1bb0c4501423ae6499844e25290da8a7b04793aa
