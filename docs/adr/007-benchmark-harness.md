# ADR 007: Open loop benchmarks on cold fleets

Status: accepted

## Context

Every claim Inferrail makes is a latency or throughput comparison between
policies, so the harness that measures them decides whether those claims
can be trusted. The classic failure is coordinated omission: a client that
sends the next request only when the previous one returns slows down
exactly when the server queues, and the queueing never shows up in its
numbers. The other is unequal conditions: one policy measured on a warm
cache and another on a cold one, or with a different arrival pattern.

## Decision

* **Open loop arrivals.** Requests arrive on a seeded Poisson schedule and
  each is sent at its scheduled time as its own task, whether or not earlier
  ones have answered. TTFT and end to end latency are measured from the
  scheduled time, not the send time, so any delay in the client counts
  against the result instead of hiding. The difference between scheduled and
  actual send time is recorded for every request and reported as send lag;
  a test drives a server that falls far behind and requires the lag to stay
  under 20 ms at p99.
* **Same everything except the policy.** For each repetition, every policy
  gets the same workload, the same seed and therefore the same requests and
  arrival times, and a fresh gateway in front of freshly started simulators,
  so no policy inherits another's cache. Repetitions use different seeds.
* **Warmup is discarded.** Requests scheduled in the first `--warmup`
  seconds are sent but left out of the summary.
* **Rates are per second of arrivals.** Throughput and goodput divide by the
  span of scheduled times, so a run where every request meets the SLO
  reports a goodput equal to its offered rate. Goodput counts requests with
  TTFT at most 2 s and time per output token at most 100 ms.
* **Two hit rates.** The usage hit rate comes from cached tokens in each
  response; the engine hit rate comes from the difference in the engines'
  own prefix cache counters before and after the run. They must agree on the
  simulator; on real engines a gap points at a measurement problem.
* **Results are data.** Each run writes one JSON file with its parameters,
  the environment (commit, dirty flag, binary versions, config hash,
  machine), the summary and every request in columns. Charts and the report
  are generated from those files only, so committed results can be
  plotted and analysed again without rerunning anything.
* Workload text is built from three and four letter words, each one token
  for the simulator's tokenizer and close to one for BPE, so prompt lengths
  are known before sending.

## Consequences

* The harness is Python with asyncio and one HTTP client; it keeps send lag
  under a few milliseconds at hundreds of requests per second, which is
  enough for simulator and single node GPU runs. Router overhead at
  thousands of requests per second is measured by the Rust stress suite
  instead.
* Runs take as long as their schedule plus the time to drain, and every run
  pays for starting a cold fleet; that cost buys comparable numbers.
