# ADR 002: Lossless OpenAI wire types

Status: accepted

## Context

The gateway forwards request bodies to engines unchanged, but it still has
to read them (model, messages, stream flags, token limits) and it sometimes
has to write them (the simulator builds every response it sends, and later
features rewrite stream options). Two failure modes are common in proxies
that parse OpenAI payloads into strict structs:

* Unknown fields are dropped. vLLM accepts dozens of extensions (`top_k`,
  `min_p`, `ignore_eos`, `chat_template_kwargs`, `kv_transfer_params`) and
  OpenAI adds parameters every few months.
* Field states collapse. vLLM serializes stream chunks with pydantic's
  `exclude_unset`, so a field set to `None` is written as `null` while an
  unset one is omitted. `Option<T>` cannot tell the two apart.

Numbers drift too: reading `"temperature": 1` into an `f64` writes back
`1.0`.

## Decision

* Every optional field is a `Nullable<T>` with three states: `Absent`,
  `Null` and `Value(T)`.
* Every struct carries `#[serde(flatten)] extra` for fields it does not
  model, so unknown fields survive.
* Sampling parameters the router never interprets keep their exact JSON
  number (`serde_json::Number`).
* Struct fields are declared in vLLM's pydantic order, so types the
  simulator renders come out in the same key order as vLLM.
* Fixtures live in `tests/fixtures/`. OpenAI fixtures must round trip
  canonically: parse, serialize, and compare compact JSON with sorted keys.
  Clients send keys in any order, so key order is the one thing a typed
  struct cannot promise for them. vLLM fixtures must round trip byte for
  byte, key order included.

## Consequences

* A proxy built on these types never strips a field a client or engine
  relies on, and the simulator's output is indistinguishable from vLLM's at
  the byte level.
* Fields are more verbose to construct (`Nullable::Value(x)`), which only
  matters in the simulator and tests.
* Values typed as opaque `serde_json::Value` objects are written with sorted
  keys. Fields that must keep vLLM order are typed explicitly instead.
