# ADR 005: Engine load from scraped metrics, trusted only while fresh

Status: accepted

## Context

The cost model needs to know how busy each replica is: how many requests it
is running, how many are queued, how full its KV cache is. The gateway
already counts what it has sent to each replica, but that misses other
gateways and other clients, and it cannot see KV usage at all. Engines
publish all of this on their Prometheus `/metrics` page.

A load signal that is wrong is worse than none: a replica that drained a
second ago but still looks full gets no traffic, and one that filled up
looks idle and gets everything. So freshness has to be a guarantee, not a
hope.

## Decision

* `inferrail-load` reads `/metrics` pages for two pinned engine versions:
  vLLM v0.31.0 and SGLang v0.5.21. The engine is recognized by its metric
  prefix. Running and waiting requests and KV usage are required; prefix
  cache counters, preemptions and KV capacity are read when present. A page
  missing a required family is an error, so a renamed metric shows up as a
  failing scrape rather than as an idle engine.
* Data parallel ranks add up: vLLM's `engine` label and SGLang's `dp_rank`
  label split a server into independent schedulers, so running and waiting
  are summed and KV usage is averaged. SGLang copies reported by tensor or
  pipeline ranks of one scheduler count once, and its per priority
  breakdowns are skipped because the total is reported separately.
* The parser borrows from the page and only parses lines whose metric name
  it needs: about 23 µs for a full vLLM page on an Apple silicon laptop.
* Every replica is scraped once per `load.scrape_interval` (100 ms), all in
  parallel, each bounded by `load.scrape_timeout` (100 ms) and a 4 MiB page
  limit. A sample is stamped with the time its request was sent.
* Each replica holds its latest sample in an `arc-swap` cell. Readers never
  lock, and a reader asking for load gets the engine's report only while it
  is at most `load.max_age` (250 ms) old. Older reports are treated as
  missing, never as current. Validation requires `max_age` to cover an
  interval plus a timeout, so one slow scrape does not make a healthy
  replica look stale.
* `Replica::load` merges that report with the gateway's own counters,
  which are always current: requests in flight and waiting for a response
  head, and estimated prompt tokens in flight and still waiting for their
  first response byte. Until requests are tokenized at the edge, a prompt
  is estimated at four bytes of request body per token.
* Policies do not read load yet. The snapshot feeds metrics, traces and the
  decision log now, and the cost model will read it through
  `Replica::load`, so the hot path pays nothing until a policy needs it.

## Consequences

* Engines are scraped ten times a second each. A vLLM page is about 50 KB,
  so 64 replicas cost a few MB/s of loopback traffic and about 1.5 ms of
  gateway CPU per second. Raise the interval (and `max_age` with it) if
  that is too much for an engine's API server.
* When metrics are missing, late or unreadable, routing degrades to the
  gateway's own counts instead of failing, and the engine is logged once
  when it stops answering and once when it recovers.
* Supporting another engine version means a new adapter and a new fixture,
  not a change to the collector.
