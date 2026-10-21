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

When a real engine run is available, replace the derived vLLM files with
live recordings from the pinned version and keep the tests unchanged.
