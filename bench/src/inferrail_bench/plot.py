"""Charts and a markdown report comparing policies, built only from result
files so any report can be regenerated from committed data."""

import statistics
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from inferrail_bench.workloads import WORKLOADS

POLICY_ORDER = [
    "round_robin",
    "random",
    "least_request",
    "p2c",
    "prefix_hash",
    "kv_aware",
    "cost_model",
]


def _policies(docs: Iterable[dict]) -> list[str]:
    found = {d["run"]["policy"] for d in docs}
    return [p for p in POLICY_ORDER if p in found] + sorted(found - set(POLICY_ORDER))


def _get(summary: dict, path: str) -> float | None:
    value = summary
    for key in path.split("."):
        if not isinstance(value, dict) or value.get(key) is None:
            return None
        value = value[key]
    return float(value)


def spread(docs: list[dict], path: str) -> tuple[float, float, float] | None:
    """Median, min and max of one summary value across repetitions."""
    values = [v for d in docs if (v := _get(d["summary"], path)) is not None]
    if not values:
        return None
    return statistics.median(values), min(values), max(values)


def group(docs: Iterable[dict]) -> dict[str, dict[str, list[dict]]]:
    out: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for doc in docs:
        out[doc["run"]["workload"]][doc["run"]["policy"]].append(doc)
    return out


def _ttft_cdf(by_policy: dict[str, list[dict]], order: list[str], path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    for policy in order:
        values = sorted(
            v * 1e3 for d in by_policy[policy] for v in d["requests"]["ttft"] if v is not None
        )
        if values:
            ax.plot(values, np.linspace(0, 1, len(values)), label=policy)
    ax.set_xscale("log")
    ax.set_xlabel("TTFT (ms, from scheduled send)")
    ax.set_ylabel("share of requests")
    ax.set_title(f"{title}: TTFT distribution")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _bars(
    by_policy: dict[str, list[dict]],
    order: list[str],
    metrics: list[tuple[str, str, float]],
    path: Path,
    title: str,
    ylabel: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    width = 0.8 / len(metrics)
    x = np.arange(len(order))
    for i, (label, key, scale) in enumerate(metrics):
        points = [spread(by_policy[p], key) for p in order]
        heights = [pt[0] * scale if pt else 0 for pt in points]
        low = [(pt[0] - pt[1]) * scale if pt else 0 for pt in points]
        high = [(pt[2] - pt[0]) * scale if pt else 0 for pt in points]
        ax.bar(x + i * width, heights, width, yerr=[low, high], capsize=3, label=label)
    ax.set_xticks(x + width * (len(metrics) - 1) / 2, order)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    if len(metrics) > 1:
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _cell(point: tuple[float, float, float] | None, scale: float, fmt: str) -> str:
    if point is None:
        return "n/a"
    mid, low, high = (format(v * scale, fmt) for v in point)
    return mid if low == high else f"{mid} [{low}, {high}]"


COLUMNS = [
    ("TTFT p50 ms", "ttft_s.p50", 1e3, ".0f"),
    ("TTFT p99 ms", "ttft_s.p99", 1e3, ".0f"),
    ("TPOT p50 ms", "tpot_s.p50", 1e3, ".1f"),
    ("ITL p99 ms", "itl_s.p99", 1e3, ".1f"),
    ("goodput req/s", "goodput_rps", 1, ".2f"),
    ("hit rate", "hit_rate", 1, ".2f"),
    ("engine hit rate", "engine.hit_rate", 1, ".2f"),
    ("errors", "errors", 1, ".0f"),
    ("send lag p99 ms", "lag_s.p99", 1e3, ".1f"),
]


def report(docs: list[dict], out: str | Path) -> Path:
    """Writes one CDF and two bar charts per workload and `report.md`."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    lines = ["# Policy comparison", ""]
    lines += [
        "Medians over repetitions, with the range in brackets when they differ.",
        "Latencies count from each request's scheduled send time.",
        "",
    ]
    for workload, by_policy in sorted(group(docs).items()):
        order = _policies(d for runs in by_policy.values() for d in runs)
        title = WORKLOADS[workload].title if workload in WORKLOADS else workload
        first = next(iter(by_policy.values()))[0]["run"]
        lines += [f"## {workload.upper()}: {title}", ""]
        lines += [
            f"{first.get('requests')} requests at {first.get('rate')} req/s, "
            f"{len(next(iter(by_policy.values())))} repetitions, seed {first.get('seed')}.",
            "",
        ]
        _ttft_cdf(by_policy, order, out / f"{workload}_ttft_cdf.png", title)
        _bars(
            by_policy,
            order,
            [("p50", "ttft_s.p50", 1e3), ("p99", "ttft_s.p99", 1e3)],
            out / f"{workload}_ttft.png",
            f"{title}: TTFT by policy",
            "ms",
        )
        _bars(
            by_policy,
            order,
            [("usage", "hit_rate", 1.0), ("engine", "engine.hit_rate", 1.0)],
            out / f"{workload}_hit_rate.png",
            f"{title}: prefix cache hit rate",
            "share of prompt tokens",
        )
        lines.append("| policy | " + " | ".join(c[0] for c in COLUMNS) + " |")
        lines.append("|---" * (len(COLUMNS) + 1) + "|")
        for policy in order:
            cells = [
                _cell(spread(by_policy[policy], key), scale, fmt) for _, key, scale, fmt in COLUMNS
            ]
            lines.append(f"| {policy} | " + " | ".join(cells) + " |")
        lines += [
            "",
            f"![TTFT distribution]({workload}_ttft_cdf.png)",
            f"![TTFT by policy]({workload}_ttft.png)",
            f"![Hit rate]({workload}_hit_rate.png)",
            "",
        ]
    path = out / "report.md"
    path.write_text("\n".join(lines))
    return path
