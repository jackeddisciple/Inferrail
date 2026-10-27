"""A sweep point run twice with the same seed on fresh fleets reproduces its
numbers within 3%. With most prompts cached, the TTFT median and tail sit
where hits meet misses, so one request routed differently by a millisecond
of jitter moves them; the run is held to means, the end to end median and
what timing cannot change."""

import pytest

from inferrail_bench import results, sweep

pytestmark = pytest.mark.e2e

POINT = """
policies = ["least_request", "cost_model"]
reps = 1
duration = 30
warmup = 3
sims = 2
sim_flags = ["--enable-prompt-tokens-details", "--decode-token-ms", "10"]

[[point]]
workload = "w1"
rates = [4]
params = { prompts = 8, system_tokens = [2000, 4000], output_tokens = [16, 48] }
"""

KEYS = [
    "ttft_s.mean",
    "e2e_s.mean",
    "e2e_s.p50",
    "tpot_s.mean",
    "hit_rate",
    "goodput_rps",
]


def value(doc: dict, path: str) -> float:
    out = doc["summary"]
    for key in path.split("."):
        out = out[key]
    return float(out)


def test_the_same_seed_reproduces_within_three_percent(tmp_path):
    plan = sweep.parse(POINT)
    execute = sweep.execute("debug")
    runs = []
    for attempt in ("first", "second"):
        written = sweep.run(plan, tmp_path / attempt, execute, log=lambda _: None)
        runs.append({results.read(p)["run"]["policy"]: results.read(p) for p in written})
    for policy in ("least_request", "cost_model"):
        first, second = runs[0][policy], runs[1][policy]
        assert first["summary"]["errors"] == second["summary"]["errors"] == 0
        for key in KEYS:
            a, b = value(first, key), value(second, key)
            assert abs(a - b) <= 0.03 * max(abs(a), abs(b), 1e-9), f"{policy} {key}: {a} vs {b}"
