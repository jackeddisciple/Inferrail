"""The benchmark harness end to end: cold simulators, a gateway per policy,
the same schedule for each, results and a report."""

import pytest

from inferrail_bench import cli, results

pytestmark = pytest.mark.e2e

FAST = [
    "--sim-flag=--enable-prompt-tokens-details",
    "--sim-flag=--prefill-token-us=20",
    "--sim-flag=--decode-token-ms=2",
]


def test_compare_runs_every_policy_on_the_same_schedule(tmp_path):
    out = tmp_path / "results"
    code = cli.main(
        [
            "compare",
            "--debug",
            "--workloads",
            "w1,w6",
            "--policies",
            "round_robin,least_request",
            "--reps",
            "1",
            "--requests",
            "30",
            "--rate",
            "40",
            "--warmup",
            "0",
            "--sims",
            "2",
            "--param",
            "w1.prompts=2",
            *FAST,
            "--out",
            str(out),
        ]
    )
    assert code == 0
    files = sorted(out.glob("*/*.json.gz"))
    assert [f"{p.parent.name}/{p.name}" for p in files] == [
        "w1/least_request-r0.json.gz",
        "w1/round_robin-r0.json.gz",
        "w6/least_request-r0.json.gz",
        "w6/round_robin-r0.json.gz",
    ]
    docs = [results.read(p) for p in files]
    for doc in docs:
        assert doc["run"]["params"] == ({"prompts": 2} if doc["run"]["workload"] == "w1" else {})
        s = doc["summary"]
        assert s["ok"] == 30, s["error_kinds"]
        assert set(s["replicas"]) == {"r0", "r1"}
        assert doc["env"]["gateway"].startswith("inferrail-gateway")
        assert len(doc["env"]["config_sha256"]) == 64
        assert "engine" in s
    by_workload = {}
    for doc in docs:
        by_workload.setdefault(doc["run"]["workload"], []).append(doc["requests"]["scheduled"])
    for schedules in by_workload.values():
        assert schedules[0] == schedules[1]
    w1 = [d for d in docs if d["run"]["workload"] == "w1"]
    assert all(d["summary"]["hit_rate"] > 0.3 for d in w1)
    for doc in docs:
        predicted = doc["summary"]["prediction"]
        assert predicted["requests"] == 30
        assert predicted["prompt_agreement"] == 1.0
        assert predicted["cached_agreement"] >= 0.9, predicted
        assert len(doc["requests"]["predicted_cached_tokens"]) == 30
    assert cli.main(["plot", *map(str, files), "--out", str(tmp_path / "report")]) == 0
    text = (tmp_path / "report" / "report.md").read_text()
    assert "| round_robin |" in text
    assert "| least_request |" in text
