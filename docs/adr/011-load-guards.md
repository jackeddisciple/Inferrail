# ADR 011: Load guards around the cost model

Status: accepted

## Context

The cost model's queue term spreads prefill, but it does not see every way
a replica can be overloaded: decode slots fill without any prefill queued,
a nearly full KV cache preempts running requests when a long prompt lands,
a hot prefix can sit on one replica while its demand grows, and a
conversation can drift away from the replica that holds its history when
the index cannot see it. Each needs a guard that holds whatever the
estimates say.

## Decision

* **Bounded load.** After consistent hashing with bounded loads: a
  candidate is skipped when its requests in flight, with this request
  placed, would exceed `ceil((1 + epsilon) * (in_flight + 1) / n)` over the
  `n` candidates. Some candidate is always below that, so the guard never
  leaves nothing to choose, and a property test checks that across random
  arrivals and departures no replica ever takes more. `epsilon` defaults
  to 1.0, no replica past twice its share.
* **KV pressure.** A candidate whose engine reports more than `kappa`
  (0.92) of its KV cache in use takes no prompt with more than
  `short_prompt_tokens` (256) tokens to compute there. When every candidate
  is under pressure the guard stands down, since the request must go
  somewhere and the queue term still prices it.
* **Hot prefix replication.** Every request adds its prefill work, priced
  where it is cheapest, to its root prefix (the first `hot_prefix.blocks`
  blocks) in a count min sketch: four rows of 1,024 counters, over a
  sliding window of `hot_prefix.window` (10 s) that blends the previous
  window by how much of the current one has passed. Lock free, 64 KiB per
  model. A prefix whose demand needs more than `hot_prefix.utilization`
  (0.5) of a replica gets `ceil(demand / utilization)` owners: the
  candidates holding most of it, then those its key hashes to. Owners are
  priced as if they held what the best of them holds, so the cost model
  shares the prefix among them by their queues; the first request an owner
  takes warms it. Other replicas stay eligible at their real cost, so a
  congested set of owners still spills.
* **Session affinity**, off by default. A request names its conversation
  with `x-session-id` or the `user` field. The conversation's last replica
  wins while the load guards allow it; otherwise the cost model chooses
  and the conversation follows the new replica. A table of 65,536 slots
  holds a session tag and a replica tag per word, so lookups and updates
  are single atomic operations and a session whose slot was taken over is
  routed afresh, never to another session's replica.
* Every decision reports what the guards set aside and what steered it, in
  `inferrail_routing_guards_total` and `inferrail_routing_steered_total`,
  the decision log and the route span.

## Why epsilon is 1.0, not 0.25

The design started from `epsilon = 0.25`. Measured on the simulated fleet,
it costs a third of W1's cache hits: with three to five requests in flight
per replica, a replica holding a popular set of prompts crosses 1.25 times
the average on ordinary noise, and every request it turns away recomputes a
prompt elsewhere.

| epsilon | W1 hit rate | W1 TTFT p99 | Zipf hit rate | Zipf TTFT p99 | Zipf busiest replica |
| --- | --- | --- | --- | --- | --- |
| 0.25 | 0.75 | 1,028 ms | 0.89 | 835 ms | 26% |
| 0.5 | 0.83 | 866 ms | 0.92 | 552 ms | 27% |
| 1.0 | 0.91 | 648 ms | 0.93 | 524 ms | 29% |
| 2.0 | 0.94 | 273 ms | 0.94 | 368 ms | 30% |
| `kv_aware` | 0.89 | 850 ms | 0.94 | 552 ms | 26% |

At 1.0 the cost model beats `kv_aware` on every workload tried and no
replica takes more than twice its share; the gain beyond it comes from
letting a replica run further ahead of the others, which a real GPU pays
for in time per output token. The knob is there for the real benchmarks to
settle.

## Consequences

* A hot system prompt that melts `prefix_hash` (62% of requests on one of
  four replicas, p99 TTFT past 20 s) leaves the cost model at 26% per
  replica and 0.23 s p99; Zipf traffic over 32 prompts puts at most 29% on
  one replica, where `prefix_hash` puts 39%.
* In every simulated run the queue term and bounded load had already
  spread a hot prefix by the time its demand crossed the threshold, so
  replication changed no decision there. It matters for long prefixes
  whose demand rises while one replica holds them: without it, a second
  replica only computes the prefix once the first has queued the full
  recompute time.
* Under 4,000 requests at 400 a second over four simulators, with Zipf
  prompts and 300 sessions, 97% of each session's turns stayed on its
  replica and the busiest replica took 26.6% of the traffic.
* The guards are part of `cost_model` only; the baselines stay plain, as
  baselines should.
