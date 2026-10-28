# ADR 014: Benchmark sweeps and the report

Status: accepted

## Context

`compare` ran every policy on a handful of workloads at one rate each. The
benchmark plan asks for more: every workload, rates stepped through each
policy's knee, bursty arrivals, three repetitions, and one command that
turns all of it into a report. A sweep of that size runs for hours, so it
must survive being stopped, and its numbers must reproduce when it is run
again.

## Decision

* **One sweep file.** TOML, read with the standard library: shared
  settings (policies, repetitions, duration, warmup, simulators, simulator
  flags, seed) and a list of points, each a workload with its rates and
  any of the shared settings or workload parameters overridden. A point is
  named, so the same workload can appear twice, as a fixed point and as a
  knee. `bench/sweeps/sim.toml` is the committed sweep.
* **Fair runs.** Every run gets a fresh gateway and fresh simulators, and
  every policy at a rate and repetition gets the same requests and arrival
  times. Policies are innermost in the run order, so whatever drifts over a
  long sweep falls on all policies of a comparison alike.
* **Resume by specification.** Each result file records its run's
  specification and a key hashed from it. A run whose file holds the same
  key is skipped; one whose specification changed, or whose file is
  unreadable, runs again. `--jobs` runs several fleets at once; send lag
  in every result shows whether the client kept up.
* **New workloads.** W3 agents resend a context that grows by a tool call
  and its result each step, over shared tool definitions, stopping before
  `max_context`. W4 quotes one to four documents drawn by Zipf weight from a
  corpus of 2,000, after a short shared instruction. W5 asks short
  questions after a shared block of worked examples. W8 is the shared
  prompt workload arriving in Gamma bursts over a diurnal ramp; any
  workload can take Gamma gaps of a chosen CV or the ramp, which follows
  `rate * (1 - amplitude * cos(2 pi t / T))` by inverting the cumulative
  rate, so the mean rate is kept. Poisson schedules are bit for bit what
  they were for every seed.
* **The report reads only result files.** A headline row per point at its
  main rate (the middle of a rate sweep), the milestone verdict (cost_model
  ahead of least_request on W1 to W3 in p99 TTFT without losing goodput),
  the benchmark targets the simulator can speak to, knee and goodput charts
  for points run at several rates, a table per rate and the environment
  every run recorded. `make sweep` runs the matrix and builds the report;
  `make report` rebuilds it from committed results.

## Reproducibility

The plan asks that the same seed reproduce within 3%. Run twice on fresh
fleets, a point's mean TTFT, mean and median end to end latency, mean TPOT,
hit rate and goodput agree within 1%; the end to end test holds them to 3%.
The TTFT median and p90 are not held to it: with most prompts cached they
sit where cache hits meet misses, so one request routed differently by a
millisecond of timer jitter moves them by 4 to 6% on a hundred requests.
Reports give medians and ranges over three repetitions for that reason.

## Consequences

* The full simulator sweep is 372 runs, about 3.5 hours on two jobs.
* Rates were chosen from a calibration run on the continuous batching
  simulator, near 70% of what a cache blind fleet sustains, so they differ
  from the earlier comparisons. Calibration runs lasted 30 s, too short for
  W2: its conversations grow with every turn, and over 60 s its 25 req/s
  saturated every policy. W2 runs at 15 req/s as its main rate and keeps 25
  as an overload point; calibration must run as long as the sweep.
