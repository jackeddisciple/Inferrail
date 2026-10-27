"""The sweep report, built from result files alone: a headline table of
every point against the benchmark targets, the milestone verdicts, knee
charts for points run at several rates, and a table and charts for each
point, so the whole report can be regenerated from committed data."""

import statistics
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from inferrail_bench import plot
from inferrail_bench.workloads import WORKLOADS

MILESTONE = ("w1", "w2", "w3")
BASELINE = "least_request"
FLOOR = "round_robin"
MAIN = "cost_model"


def _point(doc: dict) -> str:
    return doc["run"].get("point") or doc["run"]["workload"]


def by_point(docs: Iterable[dict]) -> dict[str, dict[float, dict[str, list[dict]]]]:
    out: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for doc in docs:
        run = doc["run"]
        out[_point(doc)][float(run["rate"])][run["policy"]].append(doc)
    return out


def median(docs: list[dict], path: str) -> float | None:
    point = plot.spread(docs, path)
    return point[0] if point else None


def main_rate(rates: dict[float, dict]) -> float:
    """A single rate point's rate, or the middle one of a rate sweep."""
    ordered = sorted(rates)
    return ordered[(len(ordered) - 1) // 2]


def _title(workload: str) -> str:
    return WORKLOADS[workload].title if workload in WORKLOADS else workload


def _ratio(a: float | None, b: float | None) -> str:
    if a is None or b is None or b == 0:
        return "n/a"
    return f"{(a / b - 1) * 100:+.0f}%"


def _knee(rates: dict[float, dict[str, list[dict]]], path: Path, title: str, key: str) -> None:
    order = plot._policies(d for by in rates.values() for runs in by.values() for d in runs)
    fig, ax = plt.subplots(figsize=(7, 4))
    for policy in order:
        xs, mids, lows, highs = [], [], [], []
        for rate in sorted(rates):
            point = plot.spread(rates[rate].get(policy, []), key)
            if point:
                xs.append(rate)
                mids.append(point[0] * 1e3)
                lows.append(point[1] * 1e3)
                highs.append(point[2] * 1e3)
        if xs:
            line = ax.plot(xs, mids, marker="o", label=policy)[0]
            ax.fill_between(xs, lows, highs, color=line.get_color(), alpha=0.15)
    ax.set_yscale("log")
    ax.set_xlabel("offered load (req/s)")
    ax.set_ylabel(f"TTFT {key.split('.')[-1]} (ms)")
    ax.set_title(title)
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _goodput(rates: dict[float, dict[str, list[dict]]], path: Path, title: str) -> None:
    order = plot._policies(d for by in rates.values() for runs in by.values() for d in runs)
    fig, ax = plt.subplots(figsize=(7, 4))
    for policy in order:
        xs = [r for r in sorted(rates) if rates[r].get(policy)]
        ys = [median(rates[r][policy], "goodput_rps") or 0 for r in xs]
        ax.plot(xs, ys, marker="o", label=policy)
    top = max(rates)
    ax.plot([0, top], [0, top], color="grey", linestyle="--", linewidth=1, label="offered")
    ax.set_xlabel("offered load (req/s)")
    ax.set_ylabel("goodput (req/s within SLO)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def headline(points: dict) -> list[str]:
    """One row per point at its main rate: the main policy's p99 TTFT and
    goodput next to the baselines'."""
    lines = [
        "| point | workload | rate | p99 TTFT ms, cost_model | vs least_request | vs round_robin "
        "| goodput vs least_request | hit rate | best p99 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, rates in sorted(points.items()):
        rate = main_rate(rates)
        by_policy = rates[rate]
        if MAIN not in by_policy:
            continue
        workload = by_policy[MAIN][0]["run"]["workload"]
        p99 = {p: median(runs, "ttft_s.p99") for p, runs in by_policy.items()}
        goodput = {p: median(runs, "goodput_rps") for p, runs in by_policy.items()}
        best = min((v, p) for p, v in p99.items() if v is not None)[1]
        hit = median(by_policy[MAIN], "hit_rate")
        lines.append(
            f"| {name} | {workload} | {rate:g} | {p99[MAIN] * 1e3:.0f} "
            f"| {_ratio(p99[MAIN], p99.get(BASELINE))} | {_ratio(p99[MAIN], p99.get(FLOOR))} "
            f"| {_ratio(goodput[MAIN], goodput.get(BASELINE))} "
            f"| {hit:.2f} | {best} |"
        )
    return lines


def verdicts(points: dict) -> list[str]:
    """The milestone, cost_model ahead of least_request on W1 to W3, and
    the benchmark targets the simulator can speak to."""
    lines = []

    def first(workload: str):
        for name in sorted(points):
            rates = points[name]
            rate = main_rate(rates)
            runs = rates[rate].get(MAIN, [])
            if runs and runs[0]["run"]["workload"] == workload and BASELINE in rates[rate]:
                return name, rates[rate]
        return None

    beats = []
    for workload in MILESTONE:
        found = first(workload)
        if not found:
            beats.append(None)
            lines.append(f"* {workload.upper()}: not run.")
            continue
        name, by_policy = found
        ours, theirs = (median(by_policy[p], "ttft_s.p99") for p in (MAIN, BASELINE))
        good_ours, good_theirs = (median(by_policy[p], "goodput_rps") for p in (MAIN, BASELINE))
        won = ours < theirs and good_ours >= good_theirs * 0.99
        beats.append(won)
        lines.append(
            f"* {workload.upper()} ({name}): p99 TTFT {ours * 1e3:.0f} ms against "
            f"{theirs * 1e3:.0f} ms ({_ratio(ours, theirs)}), goodput {good_ours:.2f} against "
            f"{good_theirs:.2f} req/s: {'met' if won else 'missed'}."
        )
    verdict = "met" if all(beats) else "missed"
    lines.insert(0, f"**cost_model beats least_request on W1 to W3: {verdict}.**")
    lines.append("")
    targets = []
    w1 = first("w1")
    if w1:
        hit = median(w1[1][MAIN], "engine.hit_rate") or median(w1[1][MAIN], "hit_rate")
        floor = median(w1[1].get(FLOOR, []), "ttft_s.p99")
        ours = median(w1[1][MAIN], "ttft_s.p99")
        targets.append(f"* Engine prefix hit rate on W1, target 80%+: {hit * 100:.0f}%.")
        if floor:
            cut = (1 - ours / floor) * 100
            targets.append(f"* p99 TTFT on W1 against round robin, target 40%+ lower: {cut:.0f}%.")
    w6 = first("w6")
    if w6 and BASELINE in w6[1]:
        ours, theirs = (median(w6[1][p], "ttft_s.p99") for p in (MAIN, BASELINE))
        targets.append(
            f"* p99 TTFT on unique prompts (W6) against least_request, target within 5%: "
            f"{_ratio(ours, theirs)}."
        )
    gains = []
    for rates in points.values():
        rate = main_rate(rates)
        if MAIN in rates[rate] and BASELINE in rates[rate]:
            ours, theirs = (median(rates[rate][p], "goodput_rps") for p in (MAIN, BASELINE))
            if theirs:
                gains.append((ours / theirs - 1) * 100)
    if gains:
        targets.append(
            "* Goodput against least_request, target 30%+ higher: "
            f"{statistics.median(gains):+.0f}% median over points, best {max(gains):+.0f}%."
        )
    return lines + targets


def environment(docs: list[dict]) -> list[str]:
    commits = sorted({d["env"].get("commit") or "unknown" for d in docs})
    dirty = sum(bool(d["env"].get("dirty")) for d in docs)
    versions = sorted({(d["env"].get("gateway"), d["env"].get("sim")) for d in docs})
    flags = sorted({" ".join(d["run"].get("sim_flags", [])) for d in docs})
    failed = sum(d["summary"].get("errors", 0) for d in docs)
    requests = sum(d["summary"].get("requests", 0) for d in docs)
    return [
        f"* {len(docs)} runs, {requests:,} measured requests, {failed} failed.",
        f"* Commits: {', '.join(c[:12] for c in commits)}; {dirty} runs from a dirty tree.",
        *(f"* Binaries: {g}; {s}." for g, s in versions),
        *(f"* Simulator flags: `{f or 'defaults'}`." for f in flags),
        f"* Platform: {docs[0]['env'].get('platform')}, {docs[0]['env'].get('cpus')} cpus.",
    ]


def build(docs: list[dict], out: str | Path, title: str = "Simulator sweep") -> Path:
    """Writes `report.md` and every chart under `out`."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    points = by_point(docs)
    lines = [f"# {title}", ""]
    lines += [
        "Every policy ran on the same requests and arrival times on a fresh gateway and "
        "fresh simulators. Medians over repetitions, with the range in brackets when they "
        "differ; latencies count from each request's scheduled send time.",
        "",
        "## Milestone and targets",
        "",
        *verdicts(points),
        "",
        "## Headline",
        "",
        "Each point at its main rate, the middle of a rate sweep.",
        "",
        *headline(points),
        "",
    ]
    for name, rates in sorted(points.items()):
        sample = next(iter(next(iter(rates.values())).values()))[0]["run"]
        workload = sample["workload"]
        arrivals = sample.get("arrivals", {})
        shape = arrivals.get("kind", "poisson")
        if arrivals.get("kind") == "gamma":
            shape += f" CV {arrivals['cv']:g}"
        if arrivals.get("ramp"):
            shape += f" with a {arrivals['ramp']} ramp"
        lines += [f"## {name}: {_title(workload)}", ""]
        reps = len(next(iter(next(iter(rates.values())).values())))
        lines += [
            f"{shape[0].upper()}{shape[1:]} arrivals, {reps} repetitions per rate, "
            f"seed {sample['seed']}.",
            "",
        ]
        if len(rates) > 1:
            _knee(rates, out / f"{name}_knee_p99.png", f"{name}: p99 TTFT by load", "ttft_s.p99")
            _knee(rates, out / f"{name}_knee_p50.png", f"{name}: median TTFT by load", "ttft_s.p50")
            _goodput(rates, out / f"{name}_goodput.png", f"{name}: goodput by load")
            lines += [
                f"![p99 TTFT by load]({name}_knee_p99.png)",
                f"![Median TTFT by load]({name}_knee_p50.png)",
                f"![Goodput by load]({name}_goodput.png)",
                "",
            ]
        for rate in sorted(rates):
            by_policy = rates[rate]
            order = plot._policies(d for runs in by_policy.values() for d in runs)
            count = next(iter(by_policy.values()))[0]["run"]["requests"]
            lines += [f"### {rate:g} req/s, {count} requests", ""]
            lines.append("| policy | " + " | ".join(c[0] for c in plot.COLUMNS) + " |")
            lines.append("|---" * (len(plot.COLUMNS) + 1) + "|")
            for policy in order:
                cells = [
                    plot._cell(plot.spread(by_policy[policy], key), scale, fmt)
                    for _, key, scale, fmt in plot.COLUMNS
                ]
                lines.append(f"| {policy} | " + " | ".join(cells) + " |")
            lines.append("")
        rate = main_rate(rates)
        by_policy = rates[rate]
        order = plot._policies(d for runs in by_policy.values() for d in runs)
        label = f"{name}_{rate:g}"
        plot._ttft_cdf(by_policy, order, out / f"{label}_ttft_cdf.png", f"{name} at {rate:g}/s")
        plot._bars(
            by_policy,
            order,
            [("p50", "ttft_s.p50", 1e3), ("p99", "ttft_s.p99", 1e3)],
            out / f"{label}_ttft.png",
            f"{name} at {rate:g}/s: TTFT by policy",
            "ms",
        )
        lines += [
            f"![TTFT distribution]({label}_ttft_cdf.png)",
            f"![TTFT by policy]({label}_ttft.png)",
            "",
        ]
    lines += ["## Environment", "", *environment(docs), ""]
    path = out / "report.md"
    path.write_text("\n".join(lines))
    return path
