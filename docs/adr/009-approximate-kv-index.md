# ADR 009: Chained block hashes and an approximate KV index

Status: accepted

## Context

To route a request to the replica that already holds most of its prompt,
the router needs to know, for every replica, the longest cached prefix of
that prompt. Engines will report their cache through KV events, but not
every deployment enables them, and the router must work without them. The
router does know every prompt it routed and where it went.

The lookup is on the hot path of every request and must cost microseconds
with dozens of replicas and thousands of blocks per prompt.

## Decision

* Prompts are cut into full blocks of the model's `block_size`; a trailing
  partial block is never hashed, because engines never cache one. Block 0
  hashes a salt followed by its tokens, every later block the previous
  block's hash followed by its tokens, with XXH3 128, so one hash names a
  whole prefix. The salt is a hash of the model, the LoRA adapter and
  vLLM's per request `cache_salt`, length prefixed so that no two salts
  share bytes. Engine hash values are never used.
* The index maps a block hash to the set of replicas holding it: 64 shards
  chosen by the hash's top bits, buckets by its low bits, and a replica set
  that stays inline up to 64 replicas. A lookup walks the prompt's chain
  from the root, keeps the replicas holding every block so far and records
  where each one drops out. A replica missing block `i` never counts as
  holding a later block.
* In approximate mode every routed prompt is recorded on the chosen replica
  as soon as the decision is made, in a shadow LRU sized to that replica's
  KV capacity: `kv_capacity_blocks` when configured, else what the engine
  reports in its metrics, else `index.default_capacity_blocks`. Chains are
  inserted tail first, so eviction trims long prefixes from the end as
  engines free them. Blocks entering and leaving the shadow update the
  shared map under the replica's lock, so the two always agree.
* An attempt the engine refuses, answers with an error, or fails before
  answering never computed its prompt, so the blocks its record added are
  taken back. An attempt that runs out of time keeps them, since the engine
  was most likely computing it.
* An engine can serve from cache every block but the one holding the
  prompt's last token, so predictions are capped there, as engines cap
  their hits.
* A replica that leaves the fleet is forgotten and its slot reused under a
  new generation, so a stale handle never writes into it. An engine whose
  prefix cache counters go backwards restarted, and its predictions are
  dropped.
* Policies receive each candidate's predicted cached tokens and the prompt
  length in `RouteContext`. Responses carry `x-inferrail-prompt-tokens` and
  `x-inferrail-cached-tokens`, so a benchmark can compare the prediction
  with the engine's own count request by request.

## Consequences

* Hashing costs about 5.5 ns per 16 token block. A lookup of 512 blocks
  costs 2.4 µs over 64 replicas and grows linearly with depth; it stays
  flat in the number of replicas up to 64.
* A million blocks across 64 replicas take about 148 bytes each, the shared
  map and the shadow caches together.
* With sequential traffic predictions equal what the simulators cached on
  every request, and 3,000 concurrent requests at 400 per second over three
  simulators matched exactly as well.
* Approximate mode drifts where engines evict differently, cache generated
  tokens, or receive traffic from other gateways or clients. Taking back a
  failed attempt's blocks can also drop ones another request on the same
  replica added meanwhile, so that race errs toward predicting too little.
  The exact mode, fed by engine KV events, replaces predictions with
  observations where events are available.
