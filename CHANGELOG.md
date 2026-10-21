# Changelog

Notable changes to Inferrail. Versions follow
[Semantic Versioning](https://semver.org).

## [0.1.1] (21 October 2026)

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

[0.1.1]: https://github.com/jackeddisciple/Inferrail/tree/main
[0.1.0]: https://github.com/jackeddisciple/Inferrail/tree/1bb0c4501423ae6499844e25290da8a7b04793aa
