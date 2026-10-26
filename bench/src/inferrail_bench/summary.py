"""Turns per request records into the numbers a run is judged by."""

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np

from inferrail_bench.client import Record


@dataclass(frozen=True)
class Slo:
    """The default service level: TTFT p99 at most 2 s, TPOT at most 100 ms."""

    ttft: float = 2.0
    tpot: float = 0.1

    def met(self, record: Record) -> bool:
        tpot = record.tpot
        return record.ok and record.ttft <= self.ttft and (math.isnan(tpot) or tpot <= self.tpot)


def percentiles(values: Sequence[float], qs: Sequence[int] = (50, 90, 99)) -> dict | None:
    data = np.asarray([v for v in values if not math.isnan(v)], dtype=float)
    if data.size == 0:
        return None
    out = {f"p{q}": float(np.percentile(data, q)) for q in qs}
    out["mean"] = float(data.mean())
    out["max"] = float(data.max())
    return out


def _error_kind(record: Record) -> str:
    if record.status and record.status != 200:
        return str(record.status)
    return (record.error or "unknown").split(":")[0]


def prediction(records: Sequence[Record]) -> dict | None:
    """How well the gateway's KV index predicted what the engine served
    from cache, over successful requests that carry a prediction: exact
    agreement, the share over and under the engine's count, and the mean
    error in tokens. Prompt agreement checks the edge tokenizer itself."""
    rows = [r for r in records if r.ok and r.predicted_cached_tokens is not None]
    if not rows:
        return None
    n = len(rows)
    diffs = [r.predicted_cached_tokens - r.cached_tokens for r in rows]
    prompt = sum(r.prompt_tokens for r in rows)
    return {
        "requests": n,
        "prompt_agreement": sum(r.predicted_prompt_tokens == r.prompt_tokens for r in rows) / n,
        "cached_agreement": sum(d == 0 for d in diffs) / n,
        "over_predicted": sum(d > 0 for d in diffs) / n,
        "under_predicted": sum(d < 0 for d in diffs) / n,
        "mean_abs_error_tokens": sum(abs(d) for d in diffs) / n,
        "predicted_hit_rate": sum(r.predicted_cached_tokens for r in rows) / prompt
        if prompt
        else 0.0,
    }


def summarize(records: Sequence[Record], warmup: float = 0.0, slo: Slo | None = None) -> dict:
    """Statistics over the requests scheduled at least `warmup` seconds after
    the first one. Latencies count from each request's scheduled time; rates
    are per second of the arrival window, so a run where every request meets
    the SLO has a goodput equal to its offered rate."""
    slo = slo or Slo()
    if not records:
        return {"requests": 0}
    start = min(r.scheduled for r in records)
    measured = [r for r in records if r.scheduled - start >= warmup]
    ok = [r for r in measured if r.ok]
    scheduled = [r.scheduled for r in measured]
    window = max(scheduled, default=start) - min(scheduled, default=start)

    def rate(count: int) -> float:
        return count / window if window > 0 else 0.0

    prompt = sum(r.prompt_tokens for r in ok)
    cached = sum(r.cached_tokens for r in ok)
    met = sum(slo.met(r) for r in measured)
    return {
        "requests": len(measured),
        "ok": len(ok),
        "errors": len(measured) - len(ok),
        "error_kinds": dict(Counter(_error_kind(r) for r in measured if not r.ok)),
        "duration_s": window,
        "throughput_rps": rate(len(ok)),
        "output_tokens_per_s": rate(sum(r.completion_tokens for r in ok)),
        "ttft_s": percentiles([r.ttft for r in ok]),
        "tpot_s": percentiles([r.tpot for r in ok]),
        "itl_s": percentiles([gap for r in ok for gap in r.itl]),
        "e2e_s": percentiles([r.e2e for r in ok]),
        "lag_s": percentiles([r.lag for r in measured]),
        "slo": asdict(slo),
        "slo_attainment": met / len(measured) if measured else 0.0,
        "goodput_rps": rate(met),
        "prompt_tokens": prompt,
        "cached_tokens": cached,
        "hit_rate": cached / prompt if prompt else 0.0,
        "prediction": prediction(ok),
        "replicas": dict(sorted(Counter(r.replica or "" for r in ok).items())),
    }
