"""Streams one chat completion and records when every piece of it arrived.

All times are `time.perf_counter()` readings, so they compare directly with
the schedule the runner keeps."""

import json
import math
import time
from dataclasses import dataclass, field

import httpx2

REPLICA_HEADER = "x-inferrail-replica"


@dataclass
class Record:
    index: int
    scheduled: float
    sent: float = math.nan
    first_token: float = math.nan
    end: float = math.nan
    token_times: list[float] = field(default_factory=list)
    status: int = 0
    error: str | None = None
    replica: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0

    @property
    def ok(self) -> bool:
        return self.status == 200 and self.error is None and not math.isnan(self.end)

    @property
    def lag(self) -> float:
        """How late the request left compared with its schedule."""
        return self.sent - self.scheduled

    @property
    def ttft(self) -> float:
        """From when the request was due, so client delays count against it."""
        return self.first_token - self.scheduled

    @property
    def e2e(self) -> float:
        return self.end - self.scheduled

    @property
    def tpot(self) -> float:
        """Mean time per output token after the first. Engines may send
        several tokens in one chunk, so the count comes from usage when the
        engine reports it, and from the chunks only otherwise."""
        tokens = self.completion_tokens or len(self.token_times)
        if tokens < 2 or not self.token_times:
            return math.nan
        return (self.token_times[-1] - self.token_times[0]) / (tokens - 1)

    @property
    def itl(self) -> list[float]:
        return [b - a for a, b in zip(self.token_times, self.token_times[1:], strict=False)]


def _apply(record: Record, chunk: dict, now: float) -> None:
    for choice in chunk.get("choices") or []:
        delta = choice.get("delta") or {}
        if delta.get("content") or choice.get("text"):
            if not record.token_times:
                record.first_token = now
            record.token_times.append(now)
    usage = chunk.get("usage")
    if usage:
        record.prompt_tokens = usage.get("prompt_tokens") or 0
        record.completion_tokens = usage.get("completion_tokens") or 0
        details = usage.get("prompt_tokens_details") or {}
        record.cached_tokens = details.get("cached_tokens") or 0


async def stream_chat(
    client: httpx2.AsyncClient, body: dict, record: Record, path: str = "/v1/chat/completions"
) -> Record:
    """Sends `body` and fills `record` as the stream arrives. Never raises:
    failures end up in `record.error`."""
    record.sent = time.perf_counter()
    try:
        async with client.stream("POST", path, json=body) as response:
            record.status = response.status_code
            record.replica = response.headers.get(REPLICA_HEADER)
            if response.status_code != 200:
                payload = await response.aread()
                record.error = payload.decode(errors="replace")[:500]
                record.end = time.perf_counter()
                return record
            done = False
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    done = True
                    break
                chunk = json.loads(data)
                if "error" in chunk:
                    record.error = json.dumps(chunk["error"])[:500]
                    break
                _apply(record, chunk, time.perf_counter())
            record.end = time.perf_counter()
            if not done and record.error is None:
                record.error = "stream ended without [DONE]"
    except (httpx2.HTTPError, json.JSONDecodeError, OSError) as err:
        record.error = f"{type(err).__name__}: {err}"[:500]
        record.end = time.perf_counter()
    return record
