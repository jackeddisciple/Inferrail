# ADR 015: Live config, the admin API, A/B splits and shadow policies

Status: accepted

## Context

Tuning a router means changing it while it runs: a different policy, a new
epsilon, a replica added or drained. A restart drops every open stream and
throws away what policies have learned. Trusting a new policy needs
evidence first, either a slice of real traffic or a policy that watches
every request without routing any.

## Decision

* **What reloads.** `routing`, each model's `policy`, `split`, `shadow`,
  `cost_model` and `aliases`, the static `replicas`, `retry` and
  `logging.log_prompts` change in place. Listeners, the upstream client,
  discovery, health checks, load scraping, telemetry, tokenizers, the index
  and the model list are built at startup; a reload that changes them is
  refused whole, naming the sections, and the running config stays.
* **One snapshot per request.** The config, model names and every model's
  policies live behind one atomic pointer. A request reads it once, so its
  retries use the policy that routed it and a reload never changes a
  decision halfway. A policy whose kind and settings are unchanged is
  carried into the new snapshot with its state: a round robin rotation, a
  hot prefix sketch, a session table. Unchanged static replicas keep their
  health and counters; requests already on a removed one finish there.
* **Three ways in.** The config file is reread every
  `server.reload_interval` (2 s, off at 0) and applied when it differs from
  the config in effect; SIGHUP rereads it at once; `POST /admin/reload`
  does the same over HTTP. `PATCH /admin/config` applies a JSON merge patch
  to the config in effect, leaving the file alone, so the next edit of the
  file replaces the patch. Every attempt counts in
  `inferrail_config_reloads_total{result}` and the generation in effect in
  `inferrail_config_generation`.
* **Admin views.** `GET /admin/config`, `/admin/index` (entries, bytes per
  entry, each replica's predicted blocks against its capacity),
  `/admin/replicas` (health, ejection, requests in flight, queued prefill,
  the engine's last report and its age) and `/admin/shadow`.
* **A/B splits.** `routing.split` or a model's `split` gives each policy a
  percent of the traffic. A request's arm comes from its conversation key,
  or its request id when it names none, so a conversation stays with one
  policy and a retry keeps its arm. Each arm has its own policy and seed;
  metrics already carry the policy label, responses name it in
  `x-inferrail-policy`, and `inferrail_model_policy_percent` reports the
  shares. `/admin/explain?policy=` explains any arm or shadow.
* **Shadow policies.** `routing.shadow` or a model's `shadow` lists
  policies that choose for every routed request without routing it. The
  route step hands over the candidates exactly as the active policy saw
  them, and one worker thread evaluates the shadows in that order, off the
  request path; when it falls behind, requests are dropped from the shadow
  and counted, never delayed. Each shadow pick is priced by the model's
  cost model next to the active pick, giving agreement, the share priced
  faster or slower, and the mean estimated gain, in metrics and on
  `/admin/shadow`. That is the cost model's view, which favors
  `cost_model` by construction, so a shadow's gain is a reason to try it in
  a split, not proof.
* **Replay.** With `observability.shadow_log`, every config generation and
  every shadow decision, with what the shadows saw, is written as JSON
  lines. `inferrail-gateway --replay-shadow` rebuilds each shadow from the
  logged configs with the same reuse rules, feeds it the logged requests in
  order and checks every pick, so the log is evidence anyone can rerun.

## Proofs

* 400 requests, half streamed, through 30 reloads that switch policy,
  epsilon and the replica set every 100 ms: every one completes.
* A round robin rotation continues across a reload that leaves it alone
  and restarts after one that changes its seed.
* 300 requests over two config generations, every policy in shadow: the
  offline replay matches all 1,800 picks, and a tampered line is caught.
* A 75/25 split routes within statistical bounds of its shares, and 40
  conversations each keep to one arm over five turns.

## Consequences

* A patch made over the API is lost on the next edit of the file; the file
  stays the source of truth.
* Admin routes are unauthenticated and bound to localhost by default until
  API keys and TLS arrive.
* SLOs, which the admission controller will use, join the reloadable
  sections when it lands.
