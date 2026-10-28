# ADR 012: A continuous batching simulator over one KV pool

Status: accepted

## Context

Until now `inferrail-sim` ran prefills one at a time in a FIFO queue,
decoded every sequence at a fixed time per token whatever else ran, kept
its prefix cache apart from the blocks running requests held, and inserted
a prompt's blocks into the cache the moment its prefill started. Policies
tuned on that engine never saw the things that hurt on a real one: prefill
stretching the decode steps of everything else on the GPU, a KV cache
whose running requests and cached prefixes compete for the same blocks,
and preemption when they run out. Every routing result from now on, and
the calibration against real GPUs that follows, is read against this
simulator, so it has to schedule the way vLLM v0.31.0 does.

## Decision

* **Steps, not queues.** One scheduler task runs the engine as vLLM's V1
  scheduler does. Each step has a token budget (`--max-num-batched-tokens`,
  2048 as vLLM defaults on GPUs under 70 GiB): running requests first, in
  the order they started, each decoding one token or computing the next
  chunk of its prompt, then waiting requests in arrival order while
  sequences (`--max-num-seqs`) and budget last. A waiting request that does
  not fit blocks the ones behind it.
* **Step time.** A step costs its prefill chunks plus its decode batch:
  `sum(a*u + b*u*(c + u/2))` over the chunks, `d` once per prefill, and
  `decode_token_secs + decode_seq_secs * (n - 1)` when `n` sequences decode
  (20 ms and 0.1 ms by default). Chunks of one prompt add up to exactly the
  `P(u, c)` the cost model and `inferrail-calibrate` assume, so on an idle
  engine TTFT is still `P` to the nanosecond, while prefill and decode
  sharing a step now slow each other down.
* **One block pool.** Running requests hold blocks by reference, so a
  shared prefix counts once in KV usage; a freed block keeps its hash and
  serves later hits until the pool hands it out again. Free blocks leave in
  vLLM's order: unhashed ones from the head, cached ones least recently
  freed first, a request's tail before its head. Blocks are cached when
  their computation is scheduled, generated ones too, and several blocks
  may hold one hash when requests compute the same prefix at once.
* **Preemption.** A running request that needs a block when none is free
  preempts the newest running request, which frees its blocks, goes back
  to the front of the queue and recomputes later from what is still
  cached. `vllm:num_preemptions` and `vllm:request_num_preemptions` count
  it. `--no-enable-prefix-caching` turns caching off.
* **Aborts land between steps.** A waiting request leaves the queue the
  moment its client goes; a running one is freed at the next step boundary,
  as vLLM aborts between steps.
* **Startup checks follow vLLM.** The pool must hold one request of
  `max_model_len` tokens, and the token budget must cover `max_num_seqs`.
* **The approximate index follows.** The engine now holds a prompt's last,
  partial block while it runs, so the index keeps that many fewer cached
  blocks for the replica (`KvIndex::record_with`), which keeps predictions
  exact on a cache under pressure when requests arrive one at a time.

## Proofs

* A paused clock makes every schedule deterministic. Under random bursts,
  work added ahead of the newest request never brings its first token
  sooner (6,000 cases). The same does not hold one by one for older
  requests or for arrivals spread over time, on vLLM or here: an older
  request shares its last prefill step with chunks of later prompts, and
  one arriving mid step waits for that step to end, so added work can move
  its first token earlier by part of a step. Over Poisson arrivals median
  TTFT rises with every rate, stays within twice its low load value at half
  capacity, and grows more than tenfold past capacity.
* Random schedules of arrivals, shared prompts, preemption and
  cancellations on a 64 block pool keep every invariant: FIFO admission,
  the sequence cap, gauges back to zero, cached tokens block aligned. A
  pool that runs dry preempts and still finishes every request with every
  token in order.
* A model based property test checks the pool against a naive one.

## Consequences

* Results recorded before this change are not comparable with later ones:
  TPOT now rises with load and with prefill, as on a GPU.
* The approximate index drifts from the engine when generated blocks or
  concurrent prefills shape the cache; the exact index fed by KV events
  removes that, and the sweep reports how far predictions drift.
* `decode_interference_secs` in the cost model stays 0. Its term is per
  running sequence, while the simulator, like vLLM, adds a whole decode
  step to every prefill chunk that shares one; measuring it belongs with
  the GPU calibration.
