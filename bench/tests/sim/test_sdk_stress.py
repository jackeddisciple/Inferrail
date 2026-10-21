"""Heavy concurrency through the async OpenAI SDK; run with `make stress`."""

import asyncio
import time

import httpx2
import pytest
from openai import AsyncOpenAI, DefaultAsyncHttpx2Client

from conftest import MODEL

pytestmark = pytest.mark.stress

STREAMS = 1000
TOKENS = 32


def test_thousand_concurrent_sdk_streams(sim_base_url):
    async def run():
        limits = httpx2.Limits(max_connections=STREAMS, max_keepalive_connections=STREAMS)
        http = DefaultAsyncHttpx2Client(limits=limits, timeout=60)
        async with AsyncOpenAI(
            base_url=f"{sim_base_url}/v1", api_key="unused", http_client=http
        ) as client:

            async def one(i):
                started = time.perf_counter()
                stream = await client.chat.completions.create(
                    model=MODEL,
                    messages=[{"role": "user", "content": f"stress {i}"}],
                    max_tokens=TOKENS,
                    stream=True,
                    stream_options={"include_usage": True},
                )
                first = None
                chunks = 0
                usage = None
                async for chunk in stream:
                    if first is None:
                        first = time.perf_counter() - started
                    chunks += 1
                    usage = chunk.usage or usage
                return first, chunks, usage.completion_tokens

            began = time.perf_counter()
            results = await asyncio.gather(*(one(i) for i in range(STREAMS)))
            return time.perf_counter() - began, results

    wall, results = asyncio.run(run())
    ttfts = sorted(r[0] for r in results)
    p50 = ttfts[len(ttfts) // 2] * 1000
    p99 = ttfts[int(len(ttfts) * 0.99)] * 1000
    print(f"\nsdk stress: {STREAMS} streams in {wall:.2f}s, ttft p50={p50:.1f}ms p99={p99:.1f}ms")
    assert all(chunks == TOKENS + 2 for _, chunks, _ in results)
    assert all(tokens == TOKENS for _, _, tokens in results)
