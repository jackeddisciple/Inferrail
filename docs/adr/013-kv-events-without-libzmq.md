# ADR 013: KV events in vLLM's format, over ZMTP spoken directly

Status: accepted

## Context

The exact index will consume the KV events vLLM publishes over ZeroMQ, and
the simulator has to publish the same stream so the index can be built and
tested without GPUs. The obvious route is the `zmq` crate, bindings to
libzmq. That brings a C library and its build into every crate that touches
events, into CI and into the Docker image, for a small part of ZeroMQ: one
PUB socket, one ROUTER socket for replays, and their clients, over TCP with
no security mechanism.

## Decision

* `inferrail-events` speaks ZMTP 3.0 with the NULL mechanism itself: the
  64 byte greeting, READY with the socket type, and framed messages. The
  greeting says 3.0, so libzmq peers send subscriptions as messages that
  start with 1 or 0; SUBSCRIBE and CANCEL commands from 3.1 peers and PING
  are understood too. Frames over 256 MiB are refused.
* The publisher is shaped like vLLM's `ZmqEventPublisher`: every batch goes
  out as `[topic, seq, payload]` with an 8 byte big endian sequence number
  from 0, the last `buffer_steps` batches are kept, and a ROUTER socket
  answers `[empty, start]` with every kept batch from `start` and then
  sequence -1 with an empty payload. A subscriber more than `hwm` batches
  behind loses the oldest ones, which its sequence numbers reveal, as with
  a ZeroMQ high water mark.
* Batches are encoded byte for byte as vLLM v0.31.0's msgspec structs
  encode them: the batch an array of `ts`, the events and the data parallel
  rank, each event a map tagged by `type` with its fields in declaration
  order and defaults left out. Block hashes are the int form vLLM publishes
  by default, the low 64 bits; the decoder also takes the byte form and
  skips fields it does not know, so newer engines still decode.
* The simulator publishes when given vLLM's flag,
  `--kv-events-config '{"enable_kv_cache_events": true, ...}'`, with the
  same keys and defaults. Each step's stores, evictions and clears go out
  as one batch when the step ends; `BlockStored` carries the token ids of
  the blocks, generated ones included, so a consumer rebuilds its own
  hashes and never trusts the engine's.

## Proofs

* A fixture of batches encoded by msgspec with the structs copied from
  vLLM v0.31.0 matches the encoder byte for byte; random batches round
  trip; random bytes never panic the decoder.
* Applying the simulator's stream to a multiset of hashes reproduces its
  cache exactly, under sequential and concurrent traffic with evictions,
  and every stored block's hash is rebuilt from its token ids and parent.
* Through pyzmq, a libzmq SUB socket receives every batch in order and
  vLLM's own structs decode it; libzmq REQ and DEALER sockets replay it.

## Consequences

* No C toolchain or libzmq anywhere in the build.
* Only TCP endpoints. An explicit host binds where vLLM would connect out,
  since the simulator never publishes to a proxy.
* The subscriber side is minimal today; the exact index builds on it.
