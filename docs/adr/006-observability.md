# ADR 006: Metrics on the admin port, routing spans off by default

Status: accepted

## Context

A router that claims to beat cache blind balancing has to show its work:
which replica each request went to, why, how loaded it was, and what the
client saw. Operators need that on a dashboard, a request's path through
retries in a trace, and enough history to replay decisions offline. All of
it sits next to the hot path, which must stay well under a millisecond and
must not log prompts.

## Decision

* **Metrics** use `prometheus-client` and are served as OpenMetrics on
  `server.admin_listen`, never on the client port, so clients cannot read
  fleet internals and scrapes never compete with traffic.
  * Request counters, TTFT and duration histograms carry `model`, `policy`
    and `replica`, so TTFT by policy and per replica come from one query.
    TTFT is the time from arrival to the first body byte of a streamed
    response, which for vLLM is the first token.
  * Outcomes are a status class, `aborted` when the engine broke a started
    response, or `cancelled` when the client left first.
  * Replica gauges (health, in flight, token estimates, the engine's own
    running, waiting and KV usage, and how old that report is) are computed
    from the fleet when Prometheus scrapes, so the hot path does not update
    them.
  * Label values are shared id strings, so recording allocates nothing per
    request after a series exists: about 160 ns for the decision, first byte
    and end of one streamed request on an Apple silicon laptop.
  * Series that name a replica are tracked and removed once it leaves the
    fleet, so DNS churn cannot grow the page without bound.
* **Traces** are `tracing` spans on their own target, `inferrail::span`: a
  server span per request that stays open until its body ends, and per
  attempt a `route` span (policy, candidates offered, winner, its in flight
  count and engine load, decision time) and a client `upstream` span
  (address, status, error type). The log output filters that target out, and
  only the OTLP layer turns it on, so without `observability.otlp_endpoint`
  the span callsites stay disabled and cost nothing.
* With an endpoint, spans go out through `opentelemetry-otlp` over HTTP
  protobuf in batches, sampled by `trace_sample_ratio` unless the client's
  trace says otherwise. The client's W3C `traceparent` becomes the parent and
  the engine receives the upstream span as its parent, so a trace continues
  into vLLM when it exports traces too. Without an endpoint, trace headers
  pass through untouched like any other header.
* **The decision log** is a sampled JSON lines file of routing decisions,
  sampled by request id so every attempt of a request is kept or dropped
  together. Records go through a bounded channel to a writer thread; when it
  is full they are dropped and counted, so a slow disk never slows a
  request. Records hold ids, policies and loads, never prompt text.

## Consequences

* One dashboard can answer what the router is doing: TTFT by policy and
  replica, replica load as both sides see it, the policy's traffic share and
  the engines' prefix hit rate.
* Turning tracing on costs a few microseconds per request for span creation
  and export; the sample ratio bounds it.
* The admin listener has no authentication yet; it binds to localhost by
  default and should stay on a private network until API keys and TLS land.
