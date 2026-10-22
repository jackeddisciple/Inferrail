"""inferrail-bench: run workloads open loop and compare policies."""

import argparse
import asyncio
import sys
from collections.abc import Sequence
from pathlib import Path

from inferrail_bench import engine, plot, results, runner, schedule
from inferrail_bench.summary import summarize
from inferrail_bench.workloads import WORKLOADS, generate
from inferrail_bench.workloads.multi_turn import load_sharegpt


def workload_params(name: str, sharegpt: str | None) -> dict:
    if sharegpt and name == "w2":
        return {"conversations": tuple(tuple(c) for c in load_sharegpt(sharegpt))}
    return {}


async def measure(
    base_url: str,
    model: str,
    workload: str,
    rate: float,
    count: int,
    seed: int,
    warmup: float = 0.0,
    engines: Sequence[str] = (),
    params: dict | None = None,
) -> tuple[list, dict]:
    """One open loop run: the same workload and schedule for the same seed."""
    requests = generate(workload, count, seed, **(params or {}))
    offsets = schedule.poisson(rate, count, seed)
    before = await engine.scrape(list(engines)) if engines else None
    records = await runner.run(base_url, model, requests, offsets)
    summary = summarize(records, warmup=warmup)
    if before is not None:
        summary["engine"] = engine.hit_rate(before, await engine.scrape(list(engines)))
    return records, summary


def describe(summary: dict) -> str:
    ttft = summary.get("ttft_s") or {}
    return (
        f"{summary.get('ok', 0)}/{summary.get('requests', 0)} ok, "
        f"TTFT p50 {ttft.get('p50', float('nan')) * 1e3:.0f} ms "
        f"p99 {ttft.get('p99', float('nan')) * 1e3:.0f} ms, "
        f"hit rate {summary.get('hit_rate', 0):.2f}, "
        f"goodput {summary.get('goodput_rps', 0):.2f} req/s"
    )


def cmd_run(args: argparse.Namespace) -> int:
    params = workload_params(args.workload, args.sharegpt)
    records, summary = asyncio.run(
        measure(
            args.base_url,
            args.model,
            args.workload,
            args.rate,
            args.requests,
            args.seed,
            args.warmup,
            args.engine,
            params,
        )
    )
    run = {
        "workload": args.workload,
        "policy": args.label,
        "rate": args.rate,
        "requests": args.requests,
        "seed": args.seed,
        "warmup_s": args.warmup,
        "target": args.base_url,
    }
    path = results.write(args.out, results.document(run, results.environment(), summary, records))
    print(f"{path}: {describe(summary)}")
    return 0 if summary.get("ok") else 1


def cmd_plot(args: argparse.Namespace) -> int:
    docs = [results.read(p) for p in args.results]
    print(plot.report(docs, args.out))
    return 0


def parser() -> argparse.ArgumentParser:
    top = argparse.ArgumentParser(prog="inferrail-bench", description=__doc__)
    sub = top.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run one workload against a running endpoint")
    run.add_argument(
        "--base-url", required=True, help="gateway or engine, e.g. http://127.0.0.1:8080"
    )
    run.add_argument("--model", required=True)
    run.add_argument("--workload", choices=sorted(WORKLOADS), required=True)
    run.add_argument("--rate", type=float, required=True, help="mean arrivals per second")
    run.add_argument("--requests", type=int, default=2000)
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--warmup", type=float, default=0.0, help="seconds left out of the summary")
    run.add_argument("--engine", action="append", default=[], help="engine URL for hit rates")
    run.add_argument("--label", default="", help="policy name to file the result under")
    run.add_argument("--sharegpt", help="ShareGPT JSON to replay as w2 conversations")
    run.add_argument("--out", type=Path, required=True)
    run.set_defaults(handler=cmd_run)

    report = sub.add_parser("plot", help="charts and report.md from result files")
    report.add_argument("results", nargs="+", type=Path)
    report.add_argument("--out", type=Path, required=True)
    report.set_defaults(handler=cmd_plot)
    return top


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    sys.exit(main())
