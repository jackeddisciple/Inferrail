# Changelog

Notable changes to Inferrail. Versions follow
[Semantic Versioning](https://semver.org).

## [0.1.10] Bug Fixes and Improvements (28 October 2026)

Fixes for the gaps the 0.1.9 pass left open: hot prefix credit while one
replica holds the prefix, what counts as a hot prefix steer, a race in the
prefill queue, and how the affinity table tells conversations apart.

### Fixed

* While a single replica held a hot root prefix, a cold owner was still
  credited with all that replica held, a conversation's history included,
  so a turn could move to a replica holding none of it. An owner is now
  credited only with a prefix hot in its own right: the sketch counts the
  demand of the prefixes owners hold beside the root's, and a held prefix
  counts once it needs more than `utilization` of a replica itself. First
  turns of new conversations still spread at once, since their system
  prompt carries the root's whole demand.
* The affinity table took a slot from the low 16 bits of a conversation
  key and a tag from the high 32 with one bit forced on, so keys apart only
  in bits 16 to 32 shared both and one conversation was sent to another's
  replica. Keys are now mixed first and 48 of their bits checked.
* Enqueueing on a replica's virtual prefill queue decided whether the
  request waited alone before its compare and swap, so a request queued in
  between could have its work overwritten. It now decides inside the
  update. No test reproduced the race; one queueing eight requests at once,
  2,000 times, guards it.

### Changed

* `inferrail_routing_steered_total{reason="hot_prefix"}`, the decision log
  and route spans count a hot prefix steer only when the winner was
  credited with a prefix it did not hold yet and would have lost without
  that credit. Every win by an owner counted before, so on one hot prompt
  every decision read as steered, where now only the one that brings in a
  new owner does.
* A decision on a prompt whose root a replica holds adds its work to one or
  two more keys of the sketch. With 16 threads deciding at once on 8
  prompts a decision takes 3.8 µs where it took 2.7 µs; a single decision
  (A5) is unchanged at 7.9 µs p99 over 64 replicas.

### Added

* Regression tests: conversations on a hot system prompt held by one
  replica and by two, steers counted only when the credit changed the
  pick, keys apart in bits the table used to ignore, and requests queued
  at once.

## [0.1.9] Bug Fixes and Improvements (27 October 2026)

Fixes from an adversarial pass over the routing brain. Every policy's
explanation was checked against the choice it then makes on random fleets
over time, hot prefix replication ran on simulated fleets of multi turn
chats over one shared system prompt, and `inferrail-calibrate` ran against
simulators that had already seen its prompts or cache nothing.

### Fixed

* Once a root prefix turned hot, with hot prefix replication on as it is
  by default, every owner was priced as holding what the best owner held,
  which for a chat is often one conversation's whole history. The
  conversation's next turn then went to an owner holding only the system
  prompt, or nothing, priced at 16 ms where it took 235 to 339 ms. Owners
  are now priced on what all the owners holding the whole root share, so a
  history counts only where it is, and a replica holding no more than a
  chat template's opening block does not cool the rest. Over 40 live
  conversations on one system prompt at 20 requests a second, replication
  had cut the hit rate from 0.92 to 0.84 and raised p99 TTFT from 0.12 s to
  0.53 s; it now matches running without it.
* `/admin/explain` priced hot prefix owners without the request it
  explained, while routing counts the request's own prefill toward its
  prefix's demand first, so near the threshold explain named a winner that
  routing then passed over, or no hot prefix steer where routing reported
  one. Both now count it.
* Every `inferrail-calibrate` run drew its prompts from seed 0, so a second
  run against the same engine found the first run's prompts in its prefix
  cache and fitted the cache: no per token cost, 6.7 ms of fixed time and R
  squared 0.46, written out as a usable fragment. Each run now draws its
  own seed, recorded in the fragment and the JSON, unless `--seed` names
  one.
* Calibration assumed every probe found exactly its planned prefix cached.
  Against an engine with prefix caching off, every such probe recomputed in
  full and the fit came out with no per token cost and 184 ms of fixed
  time. Each probe now reads the engine's own count of cached prompt tokens
  around it, and the fit uses what the engine computed; the probes in the
  JSON carry that count.

### Added

* `RouteContext::block_size`, the tokens per block of the prompt's hashes,
  filled by the gateway from the tokenized prompt.
* A regression test for every fix: a property test that whenever a policy
  explains a winner, routing picks it with the same steer and guards; a
  sketch property that a rate counted ahead equals the rate after the add;
  a fleet simulation of multi turn chats over a hot system prompt; and
  calibrations run twice against one simulator and against one that caches
  nothing.

## [0.1.8] The Routing Brain (26 October 2026)

The gateway now routes on what it predicts each replica has cached. The
main policy estimates every replica's time to first token from the prefill
queued there and the prefill its cache saves, inside load guards that keep
any replica from melting; two cache aware baselines run beside it; a new
binary fits the estimate to an engine; and an admin route shows why any
replica would win. On three simulators, `cost_model` keeps 91% of shared
system prompt tokens cached where round robin keeps 39%, and has the
lowest median TTFT on all five workloads measured: its p99 is 55% below
round robin's on shared prompts, 31% below `kv_aware`'s on a mix of shared
and unique prompts, and 37% below `least_request`'s on unique prompts.

### Added

#### `inferrail-policy`

* `cost_model`: the lowest estimated `TTFT = W + P + X` among the replicas
  the load guards allow. `P(u, c) = a*u + b*u*(c + u/2) + d` prices the
  prompt tokens a replica would compute against those it has cached, `W`
  the prefill already queued there plus `decode_interference_secs` per
  running request, and `X` cached tokens in CPU or disk tiers, counted only
  when loading beats recomputing. Ties go to fewer requests in flight, then
  to chance.
* Load guards for `cost_model`: bounded load skips a replica that would
  hold more than `ceil((1 + epsilon) * (in_flight + 1) / n)` requests; KV
  pressure keeps prompts with more than `short_prompt_tokens` to compute
  off replicas using more than `kappa` of their KV cache, standing down
  when every replica is; hot prefix replication tracks each root prefix's
  prefill demand in a lock free decaying count min sketch and gives one
  that needs more than `utilization` of a replica enough owners, priced as
  holding it; session affinity, off by default, keeps a conversation named
  by `x-session-id` or the `user` field on its last replica while the
  guards allow it.
* `prefix_hash`: rendezvous hashing on the prompt's first blocks, so
  removing a replica only moves the prefixes it owned; power of two choices
  for prompts without a full block.
* `kv_aware`: the weighted score of open routers, `cache_weight` times the
  share cached minus `load_weight` times requests in flight over the
  busiest candidate's.
* `Choice`, what steered a decision and what the guards set aside, and
  `explain`, how a policy sees every candidate without changing any state.
  Candidates carry queued prefill, running requests and KV usage; the
  context carries the prompt's block hashes, a session key and the clock.

#### `inferrail-gateway`

* A virtual prefill queue per replica: every decision, whatever the
  policy, adds the request's estimated prefill, which drains in real time
  and counts as empty once every request on the replica has had its first
  byte.
* `POST /admin/explain/v1/chat/completions` and `/v1/completions` on the
  admin port: the prompt, every replica serving the model and why any was
  not offered, its load, queued prefill, predicted cache, guard and
  estimate, the winner and the term that put it ahead. Nothing is routed.
  Bodies are read as on the client port, within `server.request_timeout`
  and `server.max_body_bytes`.
* The metrics `inferrail_estimated_ttft_seconds`,
  `inferrail_routing_guards_total`, `inferrail_routing_steered_total` and
  `inferrail_replica_queued_seconds`; the estimate, steering and guards in
  the decision log and on route spans, and each candidate's queued prefill
  in the decision log.

#### `inferrail-calibrate`

* Probes an idle engine with prompts of known cached and uncached length,
  one at a time with the cached part sent first, reads each probe's TTFT
  from the engine's own vLLM or SGLang histogram, fits `a`, `b` and `d` by
  least squares with no term negative, and writes a `cost_model` fragment,
  with the fit's error and the client side overhead reported apart.

#### Configuration

* `routing.short_prompt_tokens`, `session_affinity`, `prefix_hash.blocks`,
  `kv_aware` weights, `cost_model` and `hot_prefix` (`enabled`, `blocks`,
  `utilization`, `window`), and a per model `cost_model`, all validated.
* The local and compose fleets route with `cost_model`.

#### Benchmark harness (`bench/`)

* The `mix` workload of shared and unique prompts and the `hot` workload
  of Zipf weighted system prompts, the busiest replica's share in every
  summary and report, `x-session-id` on W2 requests, a `+sessions` policy
  option, and `make routing`, which compares every policy on five
  workloads into `bench/results/routing` and `bench/report/routing`.
* The committed comparison: 108 runs from one clean commit, zero failed
  requests, and the gateway's predicted cached tokens equal to the engine's
  on all but 2 of 64,263 requests.

#### Dashboard

* A cost model row: estimated next to measured TTFT by policy, queued
  prefill per replica, and how often the guards and steering act.

#### Tests

* Fleet simulations of every policy on the same seeded traffic, a property
  test that no replica ever takes more than its bounded share, the design's
  worked example, routing tests against in process simulators, an explain
  snapshot, calibration recovering two simulator parameter sets within 5%,
  the A5 decision benchmark and stress tests for shared policies, eight
  replicas and Zipf traffic with conversations.
* ADR 010 (routing on estimated time to first token) and ADR 011 (load
  guards).

### Changed

* `routing.epsilon` defaults to 1.0. At the 0.25 the design started from,
  bounded load turned away a third of W1's cache hits on ordinary noise in
  requests in flight; ADR 011 has the measurements.
* `RoutingPolicy::choose` returns a `Choice` and `build` takes the model's
  `Settings`.
* The simulator records TTFT and prefill time when its model produces the
  first token, as vLLM stamps them in its engine core, rather than when its
  timer wakes the stream, which added up to a millisecond.
* Request inspection reads the `user` field; a repeated `user` keeps its
  last value, as Python does.

## [0.1.7] Bug Fixes and Improvements (26 October 2026)

Fixes from an adversarial pass over tokenization at the edge and the KV
index. Edge case requests and single template expressions ran through both
transformers 5.17 and the gateway's tokenizer on eleven model families, the
eight beyond the golden fixtures included, and the gateway was driven with
requests engines refuse and replicas that fail.

### Fixed

* An attempt the engine refused, answered with an error, or failed before
  answering left its blocks in the KV index, so later requests sharing the
  prefix were predicted as cached on a replica that never computed it. The
  blocks a record added are now taken back; an attempt that runs out of
  time keeps them, since the engine was most likely computing it. With
  retried 503s and refused requests mixed in, 545 of 545 predictions are
  now exact where 543 were, and 16 concurrent clients saw no prediction
  above the engine's count where 10 of 871 were.
* Tool call arguments and every request field a template sees were parsed
  by serde, so a negative zero became a float, integers wider than 64 bits
  lost digits, and `1e400`, `NaN` or `Infinity` in arguments emptied them.
  Prompts with any of these tokenized differently from vLLM's. They are now
  read with the rules of Python's `json.loads`; integers wider than 128
  bits get no prediction.
* `repr` printed private use, unassigned and most format characters, such
  as the tag characters of flag emoji, as they were, where Python escapes
  them. Templates that print tools or arguments as Python values, SmolLM3's
  among them, now render what transformers renders, from CPython 3.12's
  Unicode tables.
* A template keyword read before it is assigned, as in
  `set enable_thinking = enable_thinking | default(true)`, never reached the
  template: minijinja counts a `set` target as assigned before reading its
  value, so the request's or the configured value was dropped and the
  default rendered. Keywords now follow a port of Jinja2's own analysis,
  which agrees with Jinja2 on all eleven chat templates tried, where four
  differed before.
* `title()` started words only after spaces and hyphens, so `1st` and
  `a_b` came out unlike Python's `1St` and `A_B`; `title()` and
  `capitalize()` upper cased ligatures and digraphs where Python title
  cases them, `ß` to `SS` instead of `Ss`; and a final sigma was never
  lower cased to `ς`. The methods and the `title` and `capitalize` filters
  now follow CPython 3.12 and Jinja2.
* Wrapping a text part for a template that takes OpenAI style content
  dropped its `detail` field, which vLLM keeps with the other extra fields.
* The decision log gave every candidate a predicted 0 cached tokens for
  prompts that were never tokenized, and with `index.enabled: false` the
  `x-inferrail-cached-tokens` header, the route span and
  `inferrail_predicted_cached_tokens` still reported a prediction of 0.
  They now report nothing when nothing was predicted.
* Models configured alike share one tokenizer, and each reported that
  tokenizer's totals in `inferrail_tokenize_cache_lookups`, so a sum over
  models counted every lookup once per model. Each request now adds its own
  hits and misses to its own model.

### Changed

* A chat body is read once, by the reader with Python's JSON rules, and
  strings are scanned eight bytes at a time, so preparing a request costs
  what it did in 0.1.6: a warm 8,000 token chat still takes 0.1 ms. A body
  that repeats a top level field now keeps the last value, as Python does,
  where it was refused before.

### Added

* A regression test for every fix, records taken back in the KV index's
  reference model, and expectations produced by Jinja2 and CPython for the
  variable analysis, the JSON reader, `repr`, `title` and `capitalize`.

## [0.1.6] Tokens, Block Hashes and the Approximate KV Index (26 October 2026)

The gateway now sees prompts as the engines see them: it tokenizes every
request with token ids identical to vLLM's, hashes the tokens into chained
blocks, and predicts for every replica how much of the prompt it already
holds. The policies that act on those predictions come next.

### Added

#### `inferrail-tokenize`

* Hugging Face tokenizers loaded the way transformers 5.17 loads them,
  including the pipelines `LlamaTokenizer`, `Qwen2Tokenizer` and
  `GPT2Tokenizer` rebuild instead of using `tokenizer.json` as written, the
  added tokens of `tokenizer_config.json` and the special tokens templates
  see. Other classes load as written and are logged as unverified.
* Chat templates on minijinja set up as transformers sets up Jinja2:
  `trim_blocks` and `lstrip_blocks`, loop controls, Python string and dict
  methods, `tojson` as `json.dumps`, `raise_exception`, `strftime_now`,
  Python formatting of printed values, named templates such as `tool_use`,
  `chat_template.jinja` files and a template override.
* Chat requests prepared as vLLM v0.31.0 prepares them: messages rebuilt
  from the fields vLLM reads, text parts joined or wrapped by the content
  format vLLM detects from the template's syntax tree, tool call arguments
  parsed into objects, tools dumped from vLLM's schema, developer messages,
  `continue_final_message`, `reasoning_effort`, `documents`, and template
  keywords merged and filtered as vLLM does. Completions take text or token
  ids, and every request's `cache_salt` is kept.
* A cache of repeated prompt pieces per model, exact because tokenizers
  handle the text between added tokens on its own, and disabled for
  pipelines where that does not hold.
* Chained XXH3 128 bit block hashes over full blocks, salted by model, LoRA
  adapter and `cache_salt`, at about 5.5 ns per block.
* Tokenizers found in a model directory, the local Hugging Face cache or
  `builtin:sim`, the simulator's tokenizer, which moved here from
  `inferrail-sim`.

#### `inferrail-index`

* The KV index in approximate mode: a 64 way sharded map from block hash to
  replicas, a shadow LRU per replica sized to its KV capacity, and longest
  prefix lookup at about 4.8 ns per block, flat in the number of replicas up
  to 64. Forgotten replicas' slots are reused under a new generation, so a
  stale handle never writes into them.

#### `inferrail-gateway`

* Prompts tokenized and hashed at the edge, short bodies on the request's
  task and long ones on a bounded worker pool. A request that finds the pool
  full or waits longer than `tokenizer.max_wait` is routed without a
  prediction instead of waiting.
* Every routing candidate offered with the prompt tokens the index predicts
  it has cached, and every routed prompt recorded on the chosen replica,
  with capacity from `kv_capacity_blocks`, the engine's metrics or
  `index.default_capacity_blocks`.
* `x-inferrail-prompt-tokens` and `x-inferrail-cached-tokens` on proxied
  responses, prompt and cached tokens on route spans and in the decision
  log, and the metrics `inferrail_tokenized_requests`,
  `inferrail_tokenize_seconds`, `inferrail_prompt_tokens`,
  `inferrail_predicted_cached_tokens`, `inferrail_tokenize_cache_lookups`,
  `inferrail_index_entries`, `inferrail_index_bytes`,
  `inferrail_index_resets` and the per replica index blocks and capacity.
* Replicas that leave the fleet are forgotten by the index, and an engine
  whose prefix cache counters go backwards is treated as restarted with an
  empty cache.
* Tokenizers that cannot be loaded leave their model routing without
  predictions; startup logs it and the config check prints it.

#### `inferrail-policy`

* `Candidate::cached_tokens` and `RouteContext::prompt_tokens`, ignored by
  the baselines.

#### Configuration

* `models[].chat_template`, `chat_template_content_format` and
  `chat_template_kwargs`, mirroring the vLLM flags of the same names, and
  validated against the tokenizer they need.
* `tokenizer` section: `workers`, `queue`, `max_wait` and `cache_tokens`.
* `index` section: `enabled` and `default_capacity_blocks`.
* The local and compose configs tokenize with `builtin:sim`.

#### Benchmark harness (`bench/`)

* Predicted prompt and cached tokens recorded per request and compared with
  the engines' usage: exact, over and under predictions, mean error and the
  predicted hit rate, in every summary and report. Harness gateways
  tokenize with `builtin:sim`.

#### Dashboard

* A row for prompts and the index: predicted next to engine hit rate,
  tokenization outcomes and time, and the index's blocks per replica.

#### Tests

* Golden fixtures for Qwen3, Granite 3.3 and Hermes 2 Pro: tokenizer files at
  pinned revisions and 720 requests whose ids come from vLLM's request
  handling, ported, through transformers 5.17 and pydantic, rebuilt with
  `make goldens`. Every request vLLM accepts matches, cold and warm, with
  and without the cache.
* Property tests for block hash chaining, the index against a naive
  reference model, the shadow LRU, replica sets and tokenizers fed
  arbitrary bodies; known answer hashes checked against an independent
  implementation; and gateway tests that predictions equal what simulators
  cached.
* A1, A2 and A3 benchmarks, and stress tests for tokenization, the index and
  the gateway with predictions, run by `make stress`.
* ADR 008 (tokenization parity) and ADR 009 (block hashes and the
  approximate index).

### Changed

* Lease prompt tokens are counted for tokenized models instead of estimated
  from the body size.
* `Gateway::check` returns the warnings of tokenizers that cannot be loaded.

## [0.1.5] Bug Fixes and Improvements (22 October 2026)

Fixes from an adversarial pass over the load collector, observability and
the benchmark harness, and how they meet the gateway and simulator.

### Fixed

* Label values on the gateway metrics page were written unescaped, so a
  model name containing a quote or a backslash, which model names may,
  made the whole page unparseable for Prometheus.
* A request recording metrics for a replica as it left the fleet could
  create a series just after pruning had forgotten that replica, and the
  series then stayed forever: a test of 3,000 discovery changes under
  traffic left 42,000 stale series and a 4 MB page. Creation now happens
  under the pruning lock.
* Series of departed replicas were only removed while the page was
  rendered, so a gateway nobody scraped grew with every discovery change.
  They are now pruned every ten seconds.
* The decision log read each candidate's state after the winner's lease
  was taken, so a replica at its last free slot was logged as chosen but
  not eligible, with the request itself counted in flight. Candidates are
  now logged as the policy saw them.
* `--check` built a whole gateway, creating the decision log file and
  starting its writer. It now runs the same checks, including whether the
  log could be opened, without touching the file system.
* An `otlp_endpoint` without a path, the usual way to name a collector,
  sent spans to the root, where collectors answer 404, and every span was
  lost. Such endpoints now get the standard `/v1/traces` path.
* Failed trace exports were silent. A warning is now logged when export
  starts failing and a note when it recovers.
* The compose stack published Grafana, which allows anonymous admin access,
  along with Prometheus, Jaeger and the gateway, on every network
  interface. They are now bound to localhost.
* Benchmark harness:
  * an engine that stopped answering its metrics aborted the run after it
    had finished, and the whole comparison with it; such engines are now
    left out of the engine hit rate and listed as missing,
  * an engine that restarted during a run made its counter difference
    negative and dragged the hit rate down; it is now left out,
  * SGLang engines always reported an engine hit rate of zero; SGLang's
    prompt and cached token counters are now read,
  * time per output token was divided by the number of stream chunks,
    overstating it whenever an engine sends several tokens per chunk; it
    now uses the token count from usage,
  * the engine hit rate included the warmup that the usage hit rate leaves
    out, so the two disagreed; engines are now read when the warmup ends,
  * W2 sent 97% of follow up turns before the previous reply had ended, so
    it did not model conversations at all; each conversation now waits a
    think time of 10 to 20 s between turns.

### Changed

* The committed baseline comparison was recorded again with the corrected
  harness. W2 changes most: with real gaps between turns, cache blind
  routing finds almost none of a conversation's history.
* W2 takes `think_time` instead of a fixed number of interleaved
  conversations, and every workload now receives the arrival schedule.

### Added

* A test that under a random mix of outcomes every request is counted in
  exactly one metrics outcome and every gauge returns to zero, and a
  regression test for each fix.

## [0.1.4] Load Signals, Observability and the Benchmark Harness (22 October 2026)

The router can now see how busy every engine is, shows what it is doing on
a dashboard and in traces, and comes with a harness that compares policies
on the simulator.

### Added

#### `inferrail-load`

* Parsers for the `/metrics` pages of vLLM v0.31.0 and SGLang v0.5.21:
  running and waiting requests, KV cache usage, prefix cache counters,
  preemptions and KV capacity, summed over data parallel ranks. The engine
  is recognized by its metric prefix, and a page missing a required family
  is an error rather than an idle engine. A full vLLM page parses in about
  23 µs.
* A collector that scrapes every replica in parallel every
  `load.scrape_interval` (100 ms), each scrape bounded by a timeout and a
  4 MiB page limit.
* Load snapshots published per replica through `arc-swap`. A report older
  than `load.max_age` (250 ms) is treated as missing, so routing falls back
  to the gateway's own counts instead of trusting stale data.

#### `inferrail-gateway`

* Prometheus metrics in OpenMetrics format on `server.admin_listen` at
  `/metrics`, which is now bound: requests by model, replica and outcome,
  TTFT and duration histograms by model, policy and replica, routing
  decisions and decision time, retries, client disconnects, ejections,
  scrape results, and per replica gauges for health, in flight requests and
  prompt tokens, and the engine's own running, waiting, KV usage, prefix
  cache counters and report age. Series of replicas that leave the fleet
  are removed.
* OpenTelemetry tracing over OTLP HTTP when `observability.otlp_endpoint` is
  set: a server span per request, and per attempt a `route` span with the
  policy, candidates, winner, its load and the decision time, and an
  `upstream` client span. The client's W3C trace context is continued and
  passed on to the engine. Without an endpoint the spans cost nothing and
  trace headers pass through untouched.
* A sampled decision log: JSON lines with the policy, attempt, winner and
  every candidate's load, sampled by request id and written by a separate
  thread through a bounded channel, never holding up a request and never
  containing prompt text.
* Per replica counts of estimated prompt tokens in flight and still waiting
  for their first response byte.

#### Configuration

* `load` section: `enabled`, `scrape_interval`, `scrape_timeout` and
  `max_age`, validated so one slow scrape cannot make a replica look stale.
* `observability` section: `otlp_endpoint`, `service_name`,
  `trace_sample_ratio` and `decision_log` with `path` and `sample_rate`.

#### Benchmark harness (`bench/`)

* `inferrail-bench run`, `compare` and `plot`. Open loop Poisson arrivals,
  with latency measured from each request's scheduled time and send lag
  recorded, so a slow server cannot hide its queueing.
* Workloads W1 (shared system prompts), W2 (interleaved multi turn chat,
  optionally replayed from ShareGPT) and W6 (unique prompts).
* A streaming client that records TTFT, time per output token, inter token
  latency, usage with cached tokens and the replica that served each
  request.
* Results as JSON with the environment (commit, binary versions, config
  hash, machine), a summary with percentiles, goodput against the TTFT and
  TPOT SLO and both the usage and engine reported hit rates, and every
  request in columns.
* `compare` runs every policy on the same schedules with a cold gateway and
  simulators per run; `make baselines` reproduces the committed comparison
  of the four baselines in `bench/results/baselines`, and `make report`
  regenerates `bench/report/baselines`.

#### Deployment

* `deploy/compose`: gateway, three simulators, Prometheus, Grafana and
  Jaeger with `make stack`, and the first dashboard in
  `dashboards/inferrail.json` showing TTFT, load and routing per replica.

#### Tests

* Parser tests on the vLLM and SGLang fixtures, properties that arbitrary
  input never panics and that rendered pages parse to reference totals, a
  reference model for lease token counters, and collector tests on a paused
  clock.
* Integration tests for snapshot freshness, killed and hostile engines,
  metrics, a snapshot of every metric name and label set, span attributes
  and trace propagation, OTLP export from the binary, and the decision log.
* A test that every dashboard query names a metric the gateway or vLLM
  exports, and harness tests for workloads, Poisson statistics, timing,
  coordinated omission and reports.
* Stress tests: snapshot ages under load, 64 replicas scraped at once,
  metrics scraped during 2,000 streams, hostile metrics pages, discovery
  churn, overhead with every observability feature on, and the harness at
  300 requests per second.
* ADR 005 (load collector), ADR 006 (observability) and ADR 007 (benchmark
  methodology).

### Changed

* tokio moves to 1.54.1, since 1.54.0 was yanked.
* The gateway binary installs its logging through the new telemetry setup;
  routing spans never appear in the log output.
* The e2e fixtures share process management with the harness.

## [0.1.3] Bug fixes and Improvements (21 October 2026)

Fixes from a second sweep over the gateway, policies and control plane.

### Fixed

* Requests that made engines fail could passively eject every replica of a
  model, turning a bad request pattern into an outage. Passive ejection now
  takes out at most `health.max_ejection_percent` of the replicas passing
  health checks (half by default) and never a sole replica.
* Requests already waiting on a replica that then failed its health checks
  waited for the whole request timeout and failed, although nothing had
  reached the client. They now move to another replica at once.
* Request bodies had no read deadline, so a stalled upload held its
  connection forever. Reading now shares `server.request_timeout` and
  answers 408.
* A body sent in many tiny chunks cost about forty times its size in memory
  (307 MiB for an 8 MiB body in one byte chunks). It now costs about its
  size.
* Clients sending `Expect: 100-continue` were invited to upload bodies the
  gateway was about to reject with a 404, 405 or 413. Those answers now
  come at once.
* HEAD on `/health` and `/v1/models` returned 405; it now answers as GET
  does.
* Bodies starting with a byte order mark, which Python engines accept, got
  a 400. The mark is now skipped when reading the model and forwarded as
  sent.
* DNS groups were resolved one after another at startup, so a slow DNS
  server delayed startup by every group's timeout in turn. They now resolve
  together.
* `--check` accepted configs that startup refuses, such as https replicas.
* Health and stress tests measured ejection windows on a clock other than
  the gateway's.

### Added

* `health.max_ejection_percent`, the cap on passive ejection.
* Tests for each fix, and a stress test for bodies sent in tiny chunks.

## [0.1.2] OpenAI Compatible Gateway, Baseline Routing Policies and Failure Handling (21 October 2026)

The gateway becomes usable: it routes chat and completions across a fleet
of engines with a choice of baseline policies, streams responses untouched,
and keeps clients whole when replicas crash, hang, flap or disappear.

### Added

#### `inferrail-gateway`

* OpenAI compatible proxy on hyper for `POST /v1/chat/completions` and
  `POST /v1/completions`, plus `GET /v1/models` and `GET /health`, over
  HTTP/1.1 and h2c. Other routes and methods get OpenAI shaped 404 and 405
  answers.
* Request bodies forwarded byte for byte. Only the `model` field is read,
  and only its bytes are rewritten when a client uses a gateway alias or
  omits the model while one model is served.
* Streaming pass through: engine frames reach the client as they arrive,
  with no per token state and memory bounded by backpressure.
* Request IDs: a well formed client `x-request-id` is kept, otherwise one is
  minted; it reaches the engine and comes back on every response, along
  with `x-inferrail-replica` on proxied ones.
* Retries on another replica only before the first response byte, for
  connect failures, early closes and 502, 503 or 504 answers, bounded by
  `retry.max_attempts`, exponential backoff and the request deadline.
* Passive outlier ejection after consecutive failures, with ejection windows
  that double up to a cap and reset after a clean stretch.
* Model list served from memory, built from the cards each engine publishes
  and refreshed alongside health probes, aliases included.
* Graceful shutdown bounded by `server.shutdown_timeout`, a header read
  timeout against slow clients, and request bodies drained before local
  error answers so uploading clients get the error rather than a reset.
* `--check` validates a config and exits.

#### `inferrail-policy`

* The `RoutingPolicy` trait and `RouteContext`, with `round_robin`,
  `random`, `least_request` and `p2c`, chosen per model. Randomness is a lock
  free SplitMix64 seeded per model from `routing.seed`.
* `prefix_hash`, `kv_aware` and `cost_model` route as `p2c`, their
  documented fallback, until the KV index exists, with a startup warning.

#### `inferrail-control`

* Replica registry swapped atomically for lock free reads, with in flight
  and pending counters whose caps hold under any race.
* Health state combining active probes with passive ejection, and the
  active checker that probes every replica in parallel each interval.
* DNS discovery through a pluggable resolver: groups are resolved before the
  first request and on every refresh, survivors keep their state, leavers
  finish their streams and failed lookups keep the last known members.

#### Configuration

* `upstream` section for the connection pool: connect timeout, idle timeout,
  idle connections per host and h2c.
* `discovery.dns` groups, `replicas[].max_pending`, and
  `server.header_read_timeout` and `server.shutdown_timeout`.
* `deploy/local.yaml` for three local simulators behind one gateway.

#### Tests

* Property tests for every policy, fleet membership and replica health
  against naive reference models, the body splice, stream byte equality
  under any chunking, request body equality, gateway HTTP fuzzing, and the
  rule that no policy ever reaches an unhealthy replica or one that does not
  serve the model.
* Integration tests against three in process simulators that can be killed
  for real or made to refuse, close, hang, delay or fail on demand.
* Stress suite for the gateway: 2,000 concurrent streams, open loop overhead
  against a direct simulator at 2,000 and 5,000 req/s, a simulator killed
  and restarted under load, a flapping replica, a disconnect storm, hostile
  requests, a soak and slow readers.
* The OpenAI SDK suites now also run through the gateway binary in front of
  three simulators.
* ADR 003 (gateway proxy and failure semantics) and ADR 004 (routing
  policies).

### Changed

* `replicas` may be empty when DNS groups serve every model, and the "no
  replica serves" check counts DNS groups.
* The SDK suites moved from `bench/tests/sim` to `bench/tests/e2e`; `make e2e`,
  `make stress` and the CI e2e job build and drive the gateway as well.

## [0.1.1] Reliability, Security and vLLM Parity Hardening (21 October 2026)

A hardening pass over everything in 0.1.0, driven by a new round of
property, fuzz, mutation and load testing.

### Fixed

* SSE decoder: after an invalid UTF-8 line it kept already decoded lines in
  its buffer, so a caller that kept pushing saw events twice, and an
  oversized line kept growing the buffer. Errors are now terminal. A
  leading byte order mark is stripped, as the event stream spec requires.
* Wire types: some JSON floats changed in their last digit after a parse
  and serialize cycle. Floats now round trip exactly.
* Simulator: a streamed request held its sequence slot and KV blocks until
  the client read `[DONE]`. It now finishes at its last token, as in vLLM.
* Simulator: decoding in zero latency mode never yielded to the runtime and
  could starve other requests on the same worker thread.
* Simulator: `stream_options` without `stream: true` was accepted. It now
  gets vLLM's 400 with `param` set to `stream_options`.
* Simulator: `logprobs`, `prompt_logprobs`, `echo` and `best_of` were
  silently ignored. They now get a 400 that names the parameter.
* Simulator: a wrong HTTP method got an empty 405; it now gets vLLM's JSON
  body.
* Simulator: histogram `_count` is derived from the buckets, so a scrape
  can never see it below a cumulative bucket.
* Config: replica URLs with embedded credentials and replicas that list a
  model twice were accepted; both are now rejected. An alias equal to its
  own model name gets a clear message.

### Changed

* Config YAML is parsed by `serde-saphyr`, a maintained pure Rust parser,
  instead of `serde_yaml_ng`, which pulled in the unmaintained
  `unsafe-libyaml` (`RUSTSEC-2024-0449`). Deeply nested input now fails in
  milliseconds instead of about 30 seconds, and errors report line and
  column.
* `Sim::new` validates its config and returns an error instead of failing
  later inside a request.
* The unused `bytes` dependency is gone and `futures` is only used by tests.

### Added

* Property and fuzz suites: wire types must round trip whatever they
  accept; random engine schedules with cancellations must keep FIFO
  admission, the sequence cap and clean accounting; random chat requests
  must get responses that obey the vLLM contract; streamed and full
  responses must carry the same text; hostile YAML must fail fast.
* `make audit` and a CI job that runs `cargo audit` and `cargo machete`, and
  ruff lint and format checks for the Python code.

## [0.1.0] Initial Release: Core Platform and vLLM Compatible Simulator (21 October 2026)

The first release: project foundations and a faithful vLLM simulator.

### Added

* Cargo workspace with ten crates: `inferrail-core`, `tokenize`, `index`,
  `events`, `policy`, `load` and `control` as libraries, and `gateway`, `sim`
  and `calibrate` as binaries. Toolchain pinned to Rust 1.98.1, edition 2024,
  with workspace lints.
* `bench/` Python 3.12 package managed by uv, with pytest.
* `make check`, `test`, `bench`, `e2e`, `stress` and `report` targets.
* GitHub Actions CI: fmt, clippy with warnings denied, nextest, doc tests,
  pytest, and the official OpenAI SDK against the simulator.
* ADR 001 (Rust for the data plane, Python for benchmarks) and ADR 002
  (lossless wire types).

#### `inferrail-core`

* OpenAI wire types for chat and completion requests, responses, stream
  chunks, usage with cached token details, model lists and errors, including
  vLLM extensions.
* `Nullable<T>` that keeps absent, `null` and set fields apart, plus
  passthrough of unknown fields and exact JSON numbers, so payloads round
  trip without loss.
* Incremental SSE decoder that accepts the stream in any chunking.
* Validated `ReplicaId` and `ModelId` types.
* YAML configuration for server, models, replicas, routing, health, retry and
  logging, with safe defaults, rejection of unknown keys, human readable
  durations, and cross field validation that reports every problem with its
  path. Example config in `deploy/inferrail.example.yaml`.

#### `inferrail-sim`

* Mock vLLM v0.31.0 engine serving `/v1/chat/completions`,
  `/v1/completions`, `/v1/models`, `/health`, `/metrics`, `/version` and
  `/reset_prefix_cache`, with vLLM status codes and error bodies.
* Streaming and full responses that match vLLM byte for byte, including the
  role chunk, finish chunk, optional usage chunk and `data: [DONE]`.
* Latency model `P(u, c) = a*u + b*u*(c + u/2) + d` with a drift free FIFO
  prefill queue, a running sequence cap and fixed time per output token.
* Prefix cache of chained block hashes in an O(1) LRU that evicts long
  prefixes from the tail.
* Deterministic prefix stable tokenizer and Llama 3 style chat template;
  generated text retokenizes identically.
* Prometheus metrics with vLLM names, help, labels, buckets and text format.
* Immediate release of slots and KV blocks on client disconnect, counted as
  aborts.
* CLI with vLLM flag names and a zero latency mode.

#### Tests

* Unit, property, fixture and parity tests across core and the simulator.
* Behaviour tests for cache reuse, multi turn reuse, queueing, the sequence
  cap, cancellation within 100 ms and metric accounting.
* Stress suite: 2,000 concurrent streams, 10,000 open loop requests,
  disconnect storms, malformed bodies, long prompts and a memory soak.
* OpenAI Python SDK suite, sync and async, against the simulator binary.

[0.1.5]: https://github.com/jackeddisciple/Inferrail/tree/main
[0.1.4]: https://github.com/jackeddisciple/Inferrail/tree/1c79992445b589bad482d046cbbc94fe517c2e25
[0.1.3]: https://github.com/jackeddisciple/Inferrail/tree/c6fe3ff568d5f9033b86294b7281f49ed2bcb7cc
[0.1.2]: https://github.com/jackeddisciple/Inferrail/tree/d7b4d58b108f339496938df7649bd66592b2b8d9
[0.1.1]: https://github.com/jackeddisciple/Inferrail/tree/664e148a663636837ea98cb32ddc8676dd8b7093
[0.1.0]: https://github.com/jackeddisciple/Inferrail/tree/1bb0c4501423ae6499844e25290da8a7b04793aa
