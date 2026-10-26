"""Open loop load: every request leaves at its scheduled time whether or not
earlier ones have answered, so a slow server cannot slow the arrivals down
and hide its own queueing (coordinated omission)."""

import asyncio
import time
from collections.abc import Sequence

import httpx2

from inferrail_bench.client import Record, stream_chat
from inferrail_bench.workloads import Request

LEAD = 0.05
SESSION_HEADER = "x-session-id"


def client(base_url: str, timeout: float = 600.0) -> httpx2.AsyncClient:
    limits = httpx2.Limits(max_connections=None, max_keepalive_connections=1024)
    return httpx2.AsyncClient(base_url=base_url, timeout=timeout, limits=limits)


async def run(
    base_url: str,
    model: str,
    requests: Sequence[Request],
    offsets: Sequence[float],
    timeout: float = 600.0,
) -> list[Record]:
    """Sends `requests[i]` at `offsets[i]` seconds after the start and returns
    one record per request, in order."""
    if len(requests) != len(offsets):
        raise ValueError("one offset per request")
    async with client(base_url, timeout) as http:
        start = time.perf_counter() + LEAD
        tasks = []
        for index, (request, offset) in enumerate(zip(requests, offsets, strict=True)):
            due = start + offset
            delay = due - time.perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)
            record = Record(index=index, scheduled=due)
            headers = {SESSION_HEADER: request.session} if request.session else None
            tasks.append(
                asyncio.create_task(stream_chat(http, request.body(model), record, headers=headers))
            )
        return list(await asyncio.gather(*tasks))
