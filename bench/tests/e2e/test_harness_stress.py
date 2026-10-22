"""The harness under load; run with `make stress`."""

import asyncio

import pytest

from inferrail_bench import fleet
from inferrail_bench.cli import measure

pytestmark = pytest.mark.stress

RATE = 300
REQUESTS = 3000


def test_open_loop_keeps_its_schedule_at_three_hundred_requests_a_second(tmp_path):
    flags = ["--zero-latency", "--enable-prompt-tokens-details"]
    with fleet.fleet(tmp_path, 3, "least_request", flags) as f:
        _, summary = asyncio.run(
            measure(f.gateway, fleet.MODEL, "w6", RATE, REQUESTS, seed=1, engines=f.sims)
        )
    lag = summary["lag_s"]
    ttft = summary["ttft_s"]
    print(
        f"\nharness at {RATE} req/s x {REQUESTS} (w6, zero latency sims via gateway): "
        f"send lag p50 {lag['p50'] * 1e3:.2f} ms p99 {lag['p99'] * 1e3:.2f} ms "
        f"max {lag['max'] * 1e3:.1f} ms, TTFT p50 {ttft['p50'] * 1e3:.1f} ms "
        f"p99 {ttft['p99'] * 1e3:.1f} ms, throughput {summary['throughput_rps']:.0f} req/s"
    )
    assert summary["errors"] == 0, summary["error_kinds"]
    assert lag["p99"] < 0.02
    assert summary["throughput_rps"] > RATE * 0.9
    assert summary["engine"]["queries"] > 0
    assert summary["engine"]["hit_rate"] < 0.05
