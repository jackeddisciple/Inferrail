import math

import pytest

from inferrail_bench import cli, report, results
from inferrail_bench.client import Record
from inferrail_bench.summary import summarize

POLICIES = {"round_robin": 4.0, "least_request": 2.0, "kv_aware": 1.5, "cost_model": 1.0}


def doc(point, workload, rate, policy, rep, slow):
    records = []
    for i in range(40):
        r = Record(index=i, scheduled=float(i) / rate)
        r.sent = r.scheduled
        r.status = 200
        ttft = slow * 0.1 * rate * (1 + i % 10 / 10)
        r.first_token = r.scheduled + ttft
        r.token_times = [r.first_token + k * 0.02 for k in range(5)]
        r.end = r.token_times[-1]
        r.prompt_tokens, r.completion_tokens = 1000, 5
        r.cached_tokens = 900 if policy == "cost_model" else 300
        r.replica = f"r{i % 3}"
        records.append(r)
    run = {
        "point": point,
        "workload": workload,
        "policy": policy,
        "rep": rep,
        "rate": rate,
        "arrivals": {"kind": "gamma", "cv": 2.0, "ramp": "diurnal", "amplitude": 0.6}
        if workload == "w8"
        else {"kind": "poisson", "cv": 1.0, "ramp": None, "amplitude": 0.6},
        "requests": 40,
        "seed": rep,
        "warmup_s": 0,
        "sim_flags": ["--enable-prompt-tokens-details"],
    }
    env = results.environment(gateway="inferrail-gateway 0.1.11", sim="inferrail-sim 0.1.11")
    return results.document(run, env, summarize(records), records)


def corpus(cost_model_slow=1.0):
    docs = []
    for point, workload, rates in [
        ("w1", "w1", [6]),
        ("w2", "w2", [20]),
        ("w3", "w3", [1, 2, 3]),
        ("w6", "w6", [9]),
        ("w8-cv4", "w8", [5]),
    ]:
        for rate in rates:
            for policy, slow in POLICIES.items():
                if policy == "cost_model":
                    slow = cost_model_slow
                for rep in range(2):
                    docs.append(doc(point, workload, rate, policy, rep, slow * (1 + rep * 0.05)))
    return docs


def test_the_report_names_the_milestone_targets_and_every_point(tmp_path):
    path = report.build(corpus(), tmp_path)
    text = path.read_text()
    assert "**cost_model beats least_request on W1 to W3: met.**" in text
    assert "* W3 (w3): p99 TTFT" in text
    assert "target 80%+: 90%" in text
    for name in ["w1", "w2", "w3", "w6", "w8-cv4"]:
        assert f"## {name}: " in text
    assert "Gamma CV 2 with a diurnal ramp arrivals" in text
    assert "### 2 req/s, 40 requests" in text
    for chart in ["w3_knee_p99.png", "w3_knee_p50.png", "w3_goodput.png", "w1_6_ttft.png"]:
        assert (tmp_path / chart).stat().st_size > 1000
        assert f"]({chart})" in text
    assert not (tmp_path / "w1_knee_p99.png").exists()
    assert "| w3 | w3 | 2 | " in text
    assert "* 56 runs, 2,240 measured requests, 0 failed." in text


def test_a_slower_main_policy_misses_the_milestone(tmp_path):
    text = report.build(corpus(cost_model_slow=3.0), tmp_path).read_text()
    assert "**cost_model beats least_request on W1 to W3: missed.**" in text
    assert "missed." in text.split("## Headline")[0]


def test_the_report_rebuilds_byte_for_byte_from_result_files(tmp_path):
    for i, d in enumerate(corpus()):
        results.write(tmp_path / "results" / d["run"]["point"] / f"{i}.json.gz", d)
    assert cli.main(["report", str(tmp_path / "results"), "--out", str(tmp_path / "a")]) == 0
    assert cli.main(["report", str(tmp_path / "results"), "--out", str(tmp_path / "b")]) == 0
    assert (tmp_path / "a" / "report.md").read_text() == (tmp_path / "b" / "report.md").read_text()
    with pytest.raises(SystemExit, match="no result files"):
        cli.main(["report", str(tmp_path / "empty"), "--out", str(tmp_path / "c")])


def test_main_rate_is_the_middle_of_a_sweep():
    assert report.main_rate({1.0: {}, 2.0: {}, 3.0: {}, 4.0: {}}) == 2.0
    assert report.main_rate({5.0: {}}) == 5.0
    assert report._ratio(1.0, 2.0) == "-50%"
    assert report._ratio(None, 2.0) == "n/a"
    assert math.isclose(report.median([], "x") or 0.0, 0.0)
