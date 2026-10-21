# Changelog

Notable changes to Inferrail. Versions follow
[Semantic Versioning](https://semver.org).

## [0.1.0] (21 October 2026)

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

[0.1.0]: https://github.com/jackeddisciple/Inferrail/tree/main
