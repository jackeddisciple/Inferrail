import threading
from pathlib import Path

import pytest

from inferrail_bench import cli, results, sweep
from inferrail_bench.client import Record
from inferrail_bench.summary import summarize

REPO_ROOT = Path(__file__).resolve().parents[2]

SMALL = """
name = "small"
policies = ["round_robin", "cost_model"]
reps = 2
duration = 10
sims = 2

[[point]]
workload = "w1"
rates = [2, 4]
params = { prompts = 4 }

[[point]]
name = "w8-cv4"
workload = "w8"
rates = [3]
reps = 1
policies = ["least_request"]
arrivals = { cv = 4.0 }
"""


def fake_execute(calls):
    lock = threading.Lock()

    def one(run, work):
        with lock:
            calls.append((run.point.name, run.rate, run.policy, run.rep))
        record = Record(index=0, scheduled=0.0)
        record.sent, record.first_token, record.end, record.status = 0.0, 0.1, 0.2, 200
        record.token_times = [0.1, 0.2]
        record.prompt_tokens = record.completion_tokens = 2
        return results.document(run.info(), results.environment(), summarize([record]), [record])

    return one


def test_points_inherit_the_sweep_and_override_it():
    plan = sweep.parse(SMALL)
    assert plan.name == "small"
    w1, w8 = plan.points
    assert (w1.name, w1.rates, w1.reps, w1.sims) == ("w1", (2.0, 4.0), 2, 2)
    assert w1.policies == ("round_robin", "cost_model")
    assert w1.warmup == sweep.DEFAULTS["warmup"]
    assert w1.params == {"prompts": 4}
    assert (w8.name, w8.reps, w8.policies) == ("w8-cv4", 1, ("least_request",))
    run = next(r for r in plan.runs() if r.point is w8)
    assert run.info()["arrivals"] == {
        "kind": "gamma",
        "cv": 4.0,
        "ramp": "diurnal",
        "amplitude": 0.6,
    }


def test_runs_put_policies_innermost_and_number_seeds_by_repetition():
    runs = sweep.parse(SMALL).runs()
    assert len(runs) == 2 * 2 * 2 + 1
    assert [(r.rate, r.rep, r.policy) for r in runs[:4]] == [
        (2.0, 0, "round_robin"),
        (2.0, 0, "cost_model"),
        (2.0, 1, "round_robin"),
        (2.0, 1, "cost_model"),
    ]
    assert [r.seed for r in runs[:4]] == [0, 0, 1, 1]
    assert runs[0].requests == 20
    assert runs[0].path(Path("out")) == Path("out/w1/rate2/round_robin-r0.json.gz")
    assert [r.point.name for r in sweep.parse(SMALL).runs(only=["w8-cv4"])] == ["w8-cv4"]


def test_the_key_names_everything_that_shapes_a_run():
    plan = sweep.parse(SMALL)
    run = plan.runs()[0]
    assert run.info()["key"] == plan.runs()[0].info()["key"]
    other = sweep.parse(SMALL.replace("prompts = 4", "prompts = 5")).runs()[0]
    assert other.info()["key"] != run.info()["key"]
    longer = sweep.parse(SMALL.replace("duration = 10", "duration = 11")).runs()[0]
    assert longer.info()["key"] != run.info()["key"]


@pytest.mark.parametrize(
    ("text", "error"),
    [
        ('workloads = ["w1"]\n[[point]]\nworkload = "w1"\nrates = [1]', "unknown sweep keys"),
        ('[[point]]\nworkload = "w1"\nrates = [1]\nrate = 2', "unknown point keys"),
        ('[[point]]\nworkload = "w99"\nrates = [1]', "unknown workload"),
        ('[[point]]\nworkload = "w1"\nrates = []', "positive rates"),
        ('[[point]]\nworkload = "w1"\nrates = [0]', "positive rates"),
        ('[[point]]\nworkload = "w1"\nrates = [1]\n[[point]]\nworkload = "w1"\nrates = [2]', "two"),
        ('[[point]]\nworkload = "w1"\nrates = [1]\narrivals = { kind = "x" }', "unknown arrivals"),
        ('name = "empty"', "at least one point"),
    ],
)
def test_bad_sweep_files_are_refused(text, error):
    with pytest.raises(ValueError, match=error):
        sweep.parse(text)


def test_a_sweep_resumes_and_reruns_what_changed(tmp_path):
    plan = sweep.parse(SMALL)
    calls: list = []
    lines: list[str] = []
    written = sweep.run(plan, tmp_path, fake_execute(calls), log=lines.append)
    assert len(calls) == len(written) == 9
    assert all(p.exists() for p in written)
    assert results.read(written[0])["run"]["key"] == plan.runs()[0].info()["key"]
    assert lines[0] == "9 runs, 0 already done, 9 to go"
    calls.clear()
    sweep.run(plan, tmp_path, fake_execute(calls), log=lines.append)
    assert calls == []
    changed = sweep.parse(SMALL.replace("cv = 4.0", "cv = 2.0"))
    sweep.run(changed, tmp_path, fake_execute(calls), log=lines.append)
    assert calls == [("w8-cv4", 3.0, "least_request", 0)]
    written[1].write_bytes(b"not gzip")
    calls.clear()
    sweep.run(plan, tmp_path, fake_execute(calls), only=["w1"], log=lines.append)
    assert calls == [("w1", 2.0, "cost_model", 0)]


def test_parallel_jobs_run_everything_once(tmp_path):
    calls: list = []
    sweep.run(sweep.parse(SMALL), tmp_path, fake_execute(calls), jobs=3, log=lambda _: None)
    assert sorted(calls) == sorted(
        (r.point.name, r.rate, r.policy, r.rep) for r in sweep.parse(SMALL).runs()
    )


def test_the_committed_sweep_file_is_valid_and_dry_runs(tmp_path, capsys):
    path = REPO_ROOT / "bench" / "sweeps" / "sim.toml"
    plan = sweep.load(path)
    names = {p.name for p in plan.points}
    workloads = {p.workload for p in plan.points}
    assert {"w1", "w2", "w3", "w4", "w5", "w6", "w8"} <= workloads
    assert len(names) == len(plan.points)
    assert any(len(p.rates) >= 4 for p in plan.points), "no rate sweep for a knee"
    for point in plan.points:
        assert "least_request" in point.policies
        assert "cost_model" in point.policies
    assert cli.main(["sweep", str(path), "--out", str(tmp_path), "--dry-run"]) == 0
    assert "runs, about" in capsys.readouterr().out
