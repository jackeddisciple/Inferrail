# ADR 004: One routing policy trait, baselines first

Status: accepted

## Context

Inferrail's claim is that a cache aware cost model beats cache blind load
balancing. That claim is only credible if every baseline runs in the same
binary, sees the same candidates and the same load signals, and differs from
the main policy only in its choice. Baselines bolted on as special cases in
the gateway would make every comparison suspect.

Policies also sit on the hot path. A choice is made for every request and
must not take locks, allocate per token or wait on anything.

## Decision

* Every policy implements one trait:

  ```rust
  pub trait RoutingPolicy: Send + Sync + Debug {
      fn kind(&self) -> PolicyKind;
      fn choose(&self, ctx: &RouteContext<'_>) -> Option<usize>;
  }
  ```

  `RouteContext` carries the model and the candidates with their in flight
  counts. It will grow cache hits and load snapshots as those signals land;
  existing policies ignore what they do not use.
* The gateway, not the policy, decides eligibility: a candidate serves the
  model, is routable by health and ejection, and has room under its caps.
  Policies only rank. This makes the safety invariant, never route to an
  unhealthy replica or one that does not serve the model, hold for every
  policy by construction, and a property test checks it for all of them.
* The baselines are `round_robin` (an atomic counter), `random`,
  `least_request` (the minimum in flight count, ties broken uniformly at
  random) and `p2c` (two distinct random candidates, the less loaded wins).
* Randomness comes from SplitMix64 on an atomic counter: lock free, and for
  a fixed `routing.seed` the sequence a sequential caller sees is exactly
  reproducible. Each model gets its own stream, seeded from the routing seed
  and the model name, so adding a model does not perturb the others.
* Each model gets its own policy instance from `models[].policy` or
  `routing.default_policy`.
* `prefix_hash`, `kv_aware` and `cost_model` need the KV index, which does not
  exist yet. Until it does, they route by power of two choices, which is the
  documented fallback of the cost model when no replica has a cache match,
  and the gateway logs a warning at startup. The default config therefore
  keeps working while the index is built.

## Consequences

* A benchmark comparing policies changes one config line and nothing else.
* Choosing costs nanoseconds: about 2 ns for round robin and random, 4 ns for
  power of two choices, and 4 to 106 ns for least request at 8 to 256
  candidates on an Apple silicon laptop.
* Until the index lands, the three cache aware policies behave identically
  to `p2c`. The startup warning and the README say so, so nobody mistakes
  their numbers for the real thing.
