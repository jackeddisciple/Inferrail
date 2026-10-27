# Fixtures

* `openai/`: request bodies in the form the official OpenAI SDKs send, and
  response, chunk, model list and error bodies in the form the OpenAI API
  returns. Used for canonical round trip tests.
* `vllm/`: what vLLM v0.31.0 puts on the wire, one compact JSON document per
  file or per line. They are derived from that version's pydantic models and
  serializer settings (`model_dump` for full responses, `exclude_unset` for
  chunks, `exclude_unset` plus `exclude_none` for the usage chunk). Used for
  byte exact round trip tests and as the reference the simulator is held to.
  * `chat_stream.sse` and `completion_stream.sse` are the same chunks framed
    exactly as vLLM streams them, ending in `data: [DONE]`.
  * `metrics.txt` is a `/metrics` page with every default vLLM family,
    rendered by `prometheus_client` 0.26.0 from the metric definitions in
    vLLM's `v1/metrics/loggers.py`. `make_metrics_fixture.py` rebuilds it.
  * `kv_events/batches.jsonl` holds KV event batches as the engine publishes
    them over ZeroMQ, one message per line with its topic, sequence number
    and msgpack payload in hex, encoded by msgspec with the structs of
    vLLM's `distributed/kv_events.py`. `kv_events/make_fixture.py` rebuilds
    it.

* `sglang/`: what SGLang v0.5.21 serves on `/metrics` for a server with two
  data parallel ranks and priority scheduling, rendered by `prometheus_client`
  in multiprocess mode from the definitions in that version's
  `srt/observability/metrics_collector.py`. `make_metrics_fixture.py`
  rebuilds it; the `pid` label changes from run to run.

* `tokenizers/`: the tokenizer files of three model families at pinned
  Hugging Face revisions (`source.json` names each repository, revision,
  license and file checksum; `tokenizer.json` is gzipped), and
  `goldens.jsonl.gz`: 240 request bodies per family with the token ids vLLM
  v0.31.0 computes for them, or the reason it rejects them. The ids come
  from `make_goldens.py`, which runs vLLM's request handling, ported, through
  transformers 5.17.0 and pydantic; `make goldens` rebuilds everything. The
  families are Qwen3 8B, Granite 3.3 8B Instruct and Hermes 2 Pro Mistral 7B,
  all Apache 2.0. `INFERRAIL_GOLDEN_OUT`, `INFERRAIL_GOLDEN_COUNT` and
  `INFERRAIL_GOLDEN_SEED` write a larger set elsewhere for stress runs, which
  the parity test reads from `INFERRAIL_GOLDENS`.

When a real engine run is available, replace the derived vLLM files with
live recordings from the pinned version and keep the tests unchanged.
