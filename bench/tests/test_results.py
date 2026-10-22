import json
import math
from pathlib import Path

import pytest
from fakes import Background, sse

from inferrail_bench import cli, engine, plot, results
from inferrail_bench.client import Record
from inferrail_bench.summary import Slo, percentiles, summarize

REPO_ROOT = Path(__file__).resolve().parents[2]


def record(i, ttft, tokens=5, gap=0.01, status=200, lag=0.001, prompt=100, cached=50):
    r = Record(index=i, scheduled=float(i))
    r.sent = r.scheduled + lag
    r.status = status
    if status == 200:
        r.first_token = r.scheduled + ttft
        r.token_times = [r.first_token + k * gap for k in range(tokens)]
        r.end = r.token_times[-1] + 0.001
        r.prompt_tokens, r.cached_tokens, r.completion_tokens = prompt, cached, tokens
        r.replica = f"r{i % 2}"
    else:
        r.error = "boom"
        r.end = r.sent
    return r


def test_percentiles_ignore_missing_values():
    p = percentiles([float(v) for v in range(1, 101)] + [math.nan])
    assert p["p50"] == pytest.approx(50.5)
    assert p["p99"] == pytest.approx(99.01)
    assert p["max"] == 100.0
    assert percentiles([math.nan]) is None


def test_summary_counts_slo_goodput_hit_rate_and_errors():
    records = [record(i, ttft=0.1 * (i + 1)) for i in range(30)]
    records.append(record(30, ttft=0, status=503))
    records.append(record(31, ttft=0.5, gap=0.2))
    s = summarize(records, slo=Slo(ttft=2.0, tpot=0.1))
    assert (s["requests"], s["ok"], s["errors"]) == (32, 31, 1)
    assert s["error_kinds"] == {"503": 1}
    met = sum(1 for i in range(30) if 0.1 * (i + 1) <= 2.0)
    assert s["slo_attainment"] == pytest.approx(met / 32)
    assert s["goodput_rps"] == pytest.approx(met / s["duration_s"])
    assert s["hit_rate"] == pytest.approx(0.5)
    assert s["tpot_s"]["p50"] == pytest.approx(0.01)
    assert s["replicas"] == {"r0": 15, "r1": 16}
    assert s["lag_s"]["max"] == pytest.approx(0.001)


def test_warmup_leaves_out_the_first_seconds():
    records = [record(i, ttft=5.0 if i < 10 else 0.1) for i in range(40)]
    assert summarize(records)["ttft_s"]["max"] == pytest.approx(5.0)
    warm = summarize(records, warmup=10)
    assert warm["requests"] == 30
    assert warm["ttft_s"]["max"] == pytest.approx(0.1)


def test_an_engine_that_does_not_answer_is_left_out_not_fatal(tmp_path):
    import asyncio

    from fakes import serve

    from inferrail_bench.cli import measure

    async def main():
        server, url = await serve(sse(first=0.01, gap=0.0, tokens=2))
        async with server:
            return await measure(url, "m", "w6", 200, 5, seed=0, engines=["http://127.0.0.1:9"])

    records, summary = asyncio.run(main())
    assert all(r.ok for r in records)
    assert summary["engine"]["missing"] == ["http://127.0.0.1:9"]
    assert summary["engine"]["queries"] == 0
    before = {"a": {"queries": 1.0, "hits": 1.0}, "b": None}
    after = {"a": {"queries": 5.0, "hits": 3.0}, "b": {"queries": 9.0, "hits": 9.0}}
    assert engine.hit_rate(before, after) == {
        "queries": 4.0,
        "hits": 2.0,
        "hit_rate": 0.5,
        "missing": ["b"],
    }


def test_an_engine_that_restarted_is_left_out_of_the_hit_rate():
    before = {"a": {"queries": 100.0, "hits": 50.0}, "b": {"queries": 900.0, "hits": 800.0}}
    after = {"a": {"queries": 300.0, "hits": 150.0}, "b": {"queries": 40.0, "hits": 10.0}}
    rate = engine.hit_rate(before, after)
    assert rate == {"queries": 200.0, "hits": 100.0, "hit_rate": 0.5, "missing": ["b"]}


def test_engine_counters_come_from_vllm_and_sglang_pages():
    page = (REPO_ROOT / "tests/fixtures/vllm/metrics.txt").read_text()
    assert engine.prefix_counters(page) == {"queries": 9238.0, "hits": 5040.0}
    sglang = (REPO_ROOT / "tests/fixtures/sglang/metrics.txt").read_text()
    assert engine.prefix_counters(sglang) == {"queries": 1500.0, "hits": 640.0}
    before = {"a": {"queries": 100.0, "hits": 10.0}, "b": {"queries": 0.0, "hits": 0.0}}
    after = {"a": {"queries": 300.0, "hits": 110.0}, "b": {"queries": 200.0, "hits": 100.0}}
    assert engine.hit_rate(before, after) == {
        "queries": 400.0,
        "hits": 200.0,
        "hit_rate": 0.5,
        "missing": [],
    }


def test_documents_round_trip_with_environment(tmp_path):
    records = [record(i, ttft=0.2) for i in range(5)] + [record(5, ttft=0, status=500)]
    doc = results.document(
        {"workload": "w1", "policy": "p2c"},
        results.environment(seed=3),
        summarize(records),
        records,
    )
    path = results.write(tmp_path / "x" / "run.json", doc)
    again = results.read(path)
    assert again == json.loads(json.dumps(doc))
    assert again["requests"]["ttft"][-1] is None
    assert again["requests"]["status"][-1] == 500
    env = again["env"]
    assert env["seed"] == 3
    assert env["commit"] is None or len(env["commit"]) == 40
    assert env["cpus"] >= 1
    packed = results.write(tmp_path / "run.json.gz", doc)
    assert packed.read_bytes()[:2] == b"\x1f\x8b"
    assert results.read(packed) == again
    bad = tmp_path / "old.json"
    bad.write_text(json.dumps({"schema": 0}))
    with pytest.raises(ValueError, match="schema"):
        results.read(bad)


def fake_doc(workload, policy, rep, ttft):
    records = [record(i, ttft=ttft * (1 + i / 50)) for i in range(50)]
    summary = summarize(records)
    summary["engine"] = {"hit_rate": 0.4}
    run = {
        "workload": workload,
        "policy": policy,
        "rep": rep,
        "rate": 10,
        "requests": 50,
        "seed": 1,
    }
    return results.document(run, {}, summary, records)


def test_report_has_charts_and_a_row_per_policy(tmp_path):
    docs = [
        fake_doc(w, p, rep, t)
        for w in ("w1", "w6")
        for p, t in (("least_request", 0.2), ("round_robin", 0.4))
        for rep in range(2)
    ]
    report = plot.report(docs, tmp_path)
    text = report.read_text()
    for w in ("w1", "w6"):
        for chart in ("ttft_cdf", "ttft", "hit_rate"):
            png = tmp_path / f"{w}_{chart}.png"
            assert png.stat().st_size > 5000, png
    assert "## W1: Shared system prompt" in text
    rows = [line for line in text.splitlines() if line.startswith("| round_robin")]
    assert len(rows) == 2
    assert text.index("| round_robin") < text.index("| least_request")
    assert "0.40" in rows[0]


def test_params_parse_numbers_and_group_by_workload():
    assert cli.parse_params(["prompts=32", "rate=1.5", "name=x", "span=2,8"]) == {
        "prompts": 32,
        "rate": 1.5,
        "name": "x",
        "span": (2, 8),
    }
    assert cli._compare_params(["w1.prompts=4", "w1.user_tokens=7", "w2.sessions=3"]) == {
        "w1": {"prompts": 4, "user_tokens": 7},
        "w2": {"sessions": 3},
    }
    with pytest.raises(SystemExit):
        cli.parse_params(["oops"])


def test_cli_runs_a_workload_and_plots_it(tmp_path, capsys):
    server = Background(sse(first=0.01, gap=0.002, tokens=4))
    try:
        out = tmp_path / "w6.json"
        code = cli.main(
            [
                "run",
                "--base-url",
                server.url,
                "--model",
                "m",
                "--workload",
                "w6",
                "--rate",
                "200",
                "--requests",
                "40",
                "--seed",
                "2",
                "--label",
                "p2c",
                "--param",
                "prompt_tokens=50,60",
                "--out",
                str(out),
            ]
        )
    finally:
        server.close()
    assert code == 0
    assert "40/40 ok" in capsys.readouterr().out
    doc = results.read(out)
    assert doc["run"]["policy"] == "p2c"
    assert doc["run"]["params"] == {"prompt_tokens": [50, 60]}
    assert doc["summary"]["ok"] == 40
    assert doc["summary"]["hit_rate"] == pytest.approx(0.8)
    assert cli.main(["plot", str(out), "--out", str(tmp_path / "report")]) == 0
    assert (tmp_path / "report" / "w6_ttft_cdf.png").exists()
