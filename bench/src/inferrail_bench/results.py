"""Benchmark results as JSON: what was run, where, the summary, and every
request in columns so analysis can be redone from the file alone."""

import gzip
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from inferrail_bench import __version__
from inferrail_bench.client import Record

SCHEMA = 1
REPO_ROOT = Path(__file__).resolve().parents[3]


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True, timeout=10
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip()


def config_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def environment(**extra) -> dict:
    status = _git("status", "--porcelain", "--untracked-files=no")
    return {
        "commit": _git("rev-parse", "HEAD"),
        "dirty": bool(status) if status is not None else None,
        "bench_version": __version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpus": os.cpu_count(),
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **extra,
    }


def _clean(value: float, digits: int = 6) -> float | None:
    return None if math.isnan(value) else round(value, digits)


def columns(records: Sequence[Record]) -> dict:
    start = min((r.scheduled for r in records), default=0.0)
    return {
        "scheduled": [round(r.scheduled - start, 6) for r in records],
        "lag": [_clean(r.lag) for r in records],
        "ttft": [_clean(r.ttft) for r in records],
        "tpot": [_clean(r.tpot) for r in records],
        "itl_max": [_clean(max(r.itl, default=math.nan)) for r in records],
        "e2e": [_clean(r.e2e) for r in records],
        "status": [r.status for r in records],
        "replica": [r.replica for r in records],
        "prompt_tokens": [r.prompt_tokens for r in records],
        "cached_tokens": [r.cached_tokens for r in records],
        "completion_tokens": [r.completion_tokens for r in records],
        "predicted_prompt_tokens": [r.predicted_prompt_tokens for r in records],
        "predicted_cached_tokens": [r.predicted_cached_tokens for r in records],
        "error": [r.error for r in records],
    }


def document(run: dict, env: dict, summary: dict, records: Sequence[Record]) -> dict:
    return {
        "schema": SCHEMA,
        "run": run,
        "env": env,
        "summary": summary,
        "requests": columns(records),
    }


def write(path: str | Path, doc: dict) -> Path:
    """Writes compact JSON, gzipped when the name ends in `.gz`."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(doc, separators=(",", ":"), allow_nan=False) + "\n").encode()
    if path.suffix == ".gz":
        data = gzip.compress(data, mtime=0)
    path.write_bytes(data)
    return path


def read(path: str | Path) -> dict:
    data = Path(path).read_bytes()
    if Path(path).suffix == ".gz":
        data = gzip.decompress(data)
    doc = json.loads(data)
    if doc.get("schema") != SCHEMA:
        raise ValueError(f"{path}: unsupported schema {doc.get('schema')}")
    return doc
