# ADR 010: Routing on estimated time to first token

Status: accepted

## Context

The KV index predicts how much of a prompt each replica holds. The router
now has to act on it. Two naive answers fail in opposite directions:
hashing a prefix to a fixed replica piles hot prompts onto one engine, and
balancing load alone throws the cache away. The question a request actually
asks is how long until its first token on each replica, given what is
cached there and what is queued there.

Two weighted heuristics already used by open routers belong in the same
binary as baselines, so the main policy can be judged against them on the
same candidates.

## Decision

* `prefix_hash` picks a replica by rendezvous hashing on the prompt's
  first `routing.prefix_hash.blocks` blocks (4 by default). The key is the
  hash of the last of those blocks, which names the whole prefix since
  block hashes chain. Removing a replica only moves the prefixes it owned.
  A prompt without a full block falls back to power of two choices.
* `kv_aware` scores `cache_weight * cached / prompt - load_weight *
  in_flight / busiest` (1 and 1 by default); ties go to fewer requests in
  flight, then to chance.
* `cost_model` estimates `W + P + X` for every candidate and picks the
  lowest, within the load guards of ADR 011:
  * `P(u, c) = a*u + b*u*(c + u/2) + d` for `u` uncached tokens attending
    to `c` cached ones, the shape the simulator implements, with `a`, `b`,
    `d` from `routing.cost_model` or a model's own `cost_model`.
  * `W` is the prefill work queued on the replica plus
    `decode_interference_secs` per running request. Every routed request
    adds its estimated prefill to its replica's virtual queue, which drains
    in real time, starts from the request's arrival when nothing else is
    waiting, and counts as empty once every request on the replica has had
    its first byte. This is the queue the gateway itself built, so it is
    exact for one gateway in front of a FIFO prefill engine and needs no
    engine metric.
  * `X` prices cached tokens in CPU or disk tiers by their load time and
    only counts a tier hit when loading beats recomputing. The approximate
    index knows the GPU tier only, so `X` is zero until tier events exist.
  * Ties go to fewer requests in flight, then to chance, so identical idle
    replicas share traffic.
* The trait grows: `choose` returns a `Choice` (the index, what steered it,
  what the guards set aside, the estimate), and `explain` returns how the
  policy sees every candidate without changing any state. Candidates carry
  the queue, running requests and KV usage; the context carries the
  prompt's block hashes, a session key and the clock.
* Every decision, whatever the policy, adds the cost model's estimate to the
  chosen replica's queue, so switching policy keeps the queue right and the
  estimate is comparable across policies in `inferrail_estimated_ttft_seconds`.
* `inferrail-calibrate` measures an idle engine: prompts of known cached
  and uncached length, one at a time, the cached part sent alone first.
  Each probe's TTFT comes from the engine's own histogram where it exports
  one, so network and HTTP time stay out of the fit; the client side
  difference is reported separately, since every replica pays it. A non
  negative least squares fit over all subsets of `a`, `b`, `d` gives the
  parameters, written as a config fragment.
* The simulator now records TTFT and prefill time at the moment its GPU
  model produces the first token, as vLLM stamps them in its engine core,
  rather than when the timer wakes the stream. Without this, timer
  rounding of up to a millisecond added a bias to `d`.
* `POST /admin/explain/v1/chat/completions` and `/v1/completions` on the
  admin listener answer with the policy, its rule, the prompt, every
  replica serving the model (offered or why not), each one's load, queue,
  prediction, exclusion and estimate, the winner and the term that set it
  apart from the runner up. Nothing is routed and no state changes.

## Consequences

* On a simulated fleet with the simulator's timing (`tests/fleet.rs`), the
  cache aware policies keep 85 to 91% of W1's prompt tokens in cache where
  round robin keeps 37%, and on in process simulators 92 to 95% against
  37%. On a mix of shared and unique prompts, `cost_model` has a p99 TTFT
  26% below `kv_aware`'s, because in flight counts say little about queued
  prefill.
* In the committed comparison of every policy on three simulators
  (`bench/results/routing`), `cost_model` has the lowest median TTFT on
  all five workloads and the lowest p99 on four: 31% below `kv_aware` on
  the mix and 37% below `least_request` on unique prompts, where cache
  affinity has nothing to offer. On W1 its p99 is 55% below round robin's
  and 5% above `prefix_hash`'s, which gets there by putting 46% of the
  requests on one of three replicas.
* Calibration against a simulator recovers `a`, `b` and `d` to nine
  significant digits; the tests require 5% on two parameter sets.
* A full decision for an 8,000 token prompt, the index lookup of every
  candidate included, takes 7 µs at p99 over 64 replicas and 20 µs over
  256 (A5, target 200 µs).
* The virtual queue only knows this gateway's traffic. Several gateways in
  front of the same engines each see part of the queue; the engine's own
  waiting count is the signal to fold in once gateways run in pairs.
* `decode_interference_secs` defaults to zero, which is right for the
  simulator and not for a real engine running chunked prefill. It is a
  config knob until the GPU calibration window measures it.
