# Fixtures

* `openai/`: request bodies in the form the official OpenAI SDKs send, and
  response, chunk, model list and error bodies in the form the OpenAI API
  returns. Used for canonical round trip tests.
* `vllm/`: response, chunk, model list and error bodies exactly as vLLM
  v0.31.0 writes them on the wire, one compact JSON document per file or per
  line. They are derived from that version's pydantic models and serializer
  settings (`model_dump` for full responses, `exclude_unset` for chunks,
  `exclude_unset` plus `exclude_none` for the usage chunk). Used for byte
  exact round trip tests and as the reference for the simulator.

When a real engine run is available, replace the derived vLLM files with
live recordings from the pinned version and keep the tests unchanged.
