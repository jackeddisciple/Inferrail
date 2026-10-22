import asyncio
import math
import statistics
import time

import pytest
from fakes import serve, sse, status

from inferrail_bench import runner
from inferrail_bench.client import Record, stream_chat
from inferrail_bench.schedule import poisson
from inferrail_bench.workloads import Message, Request

REQUEST = Request(messages=(Message("user", "hi"),), max_tokens=8)


def run(coro):
    return asyncio.run(coro)


def test_ttft_itl_and_tpot_match_the_server_timing():
    async def main():
        server, url = await serve(sse(first=0.12, gap=0.02, tokens=10))
        async with server:
            return await runner.run(url, "m", [REQUEST] * 3, [0.0, 0.01, 0.02])

    records = run(main())
    for r in records:
        assert r.ok, r.error
        assert r.replica == "r0"
        assert len(r.token_times) == 10
        assert (r.prompt_tokens, r.completion_tokens, r.cached_tokens) == (40, 10, 32)
        assert 0.11 <= r.ttft <= 0.2, r.ttft
        assert abs(r.tpot - 0.02) < 0.008, r.tpot
        assert abs(statistics.median(r.itl) - 0.02) < 0.008
        assert r.e2e >= r.ttft + 9 * 0.018
        assert r.lag < 0.02


def test_errors_and_unreachable_servers_end_up_in_the_record():
    async def main():
        server, url = await serve(status(503, '{"error": {"message": "full"}}'))
        async with server, runner.client(url) as http:
            failed = await stream_chat(http, REQUEST.body("m"), Record(0, time.perf_counter()))
        async with runner.client("http://127.0.0.1:9") as http:
            refused = await stream_chat(http, REQUEST.body("m"), Record(1, time.perf_counter()))
        return failed, refused

    failed, refused = run(main())
    assert failed.status == 503
    assert not failed.ok
    assert "full" in failed.error
    assert refused.status == 0
    assert not refused.ok
    assert "ConnectError" in refused.error
    assert math.isnan(refused.ttft)


def test_open_loop_keeps_sending_on_schedule_while_the_server_queues():
    """Coordinated omission: a server that handles one request at a time,
    50 ms each, falls far behind 80 req/s. An open loop client must still
    send every request on time and charge the queueing to latency."""

    async def main():
        server, url = await serve(sse(first=0.05, gap=0.0, tokens=1), gate=asyncio.Lock())
        async with server:
            offsets = poisson(80, 40, seed=3)
            return offsets, await runner.run(url, "m", [REQUEST] * 40, offsets)

    offsets, records = run(main())
    assert all(r.ok for r in records)
    lags = sorted(r.lag for r in records)
    assert lags[int(len(lags) * 0.99) - 1] < 0.02, lags[-5:]
    span = offsets[-1] - offsets[0]
    sent = max(r.sent for r in records) - min(r.sent for r in records)
    assert abs(sent - span) < 0.05
    worst = max(r.ttft for r in records)
    assert worst > 40 * 0.05 - span - 0.3, worst
    from_send = max(r.first_token - r.sent for r in records)
    assert worst >= from_send


def test_a_blocked_event_loop_shows_up_as_lag():
    async def main():
        server, url = await serve(sse(first=0.0, gap=0.0, tokens=1))

        async def stall():
            await asyncio.sleep(0.2)
            time.sleep(0.25)

        async with server:
            offsets = [i * 0.02 for i in range(40)]
            staller = asyncio.create_task(stall())
            records = await runner.run(url, "m", [REQUEST] * 40, offsets)
            await staller
            return records

    records = run(main())
    assert max(r.lag for r in records) >= 0.15
    assert min(r.lag for r in records) < 0.01


def test_mismatched_schedules_are_refused():
    with pytest.raises(ValueError, match="offset"):
        run(runner.run("http://127.0.0.1:9", "m", [REQUEST], []))
