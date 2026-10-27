"""inferrail-bench: run workloads open loop and compare policies."""

import argparse
import asyncio
import sys
from collections.abc import Sequence
from pathlib import Path

from inferrail_bench import engine, plot, results, runner
from inferrail_bench.schedule import Arrivals
from inferrail_bench.summary import summarize
from inferrail_bench.workloads import WORKLOADS, generate
from inferrail_bench.workloads.multi_turn import load_sharegpt


def _value(text: str) -> int | float | str | tuple:
    if "," in text:
        return tuple(_value(part.strip()) for part in text.split(","))
    for kind in (int, float):
        try:
            return kind(text)
        except ValueError:
            pass
    return text


def parse_params(items: Sequence[str]) -> dict:
    """`key=value` pairs; numbers become numbers and `a,b` a range."""
    params = {}
    for item in items:
        key, sep, value = item.partition("=")
        if not sep:
            raise SystemExit(f"--param {item!r} is not key=value")
        params[key.strip()] = _value(value.strip())
    return params


def arrivals_for(workload: str, overrides: dict | None = None) -> Arrivals:
    """The workload's own arrival process with `overrides` on top."""
    return Arrivals.from_dict({**WORKLOADS[workload].arrivals, **(overrides or {})})


def arrival_overrides(args: argparse.Namespace) -> dict:
    out: dict = {}
    if args.arrivals:
        out["kind"] = args.arrivals
    if args.cv is not None:
        out["cv"] = args.cv
    if args.ramp:
        out["ramp"] = None if args.ramp == "none" else args.ramp
    return out


def workload_params(name: str, sharegpt: str | None, items: Sequence[str] = ()) -> dict:
    params: dict = parse_params(items)
    if sharegpt and name == "w2":
        params["conversations"] = tuple(tuple(c) for c in load_sharegpt(sharegpt))
    return params


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
    arrivals: Arrivals | None = None,
) -> tuple[list, dict]:
    """One open loop run: the same workload and schedule for the same seed.
    The engines' counters are read when the warmup ends and after the run,
    so the engine hit rate covers the same requests as the summary."""
    offsets = (arrivals or arrivals_for(workload)).offsets(rate, count, seed)
    requests = generate(workload, count, seed, offsets, **(params or {}))
    urls = list(engines)

    async def at_warmup() -> dict:
        await asyncio.sleep(runner.LEAD + warmup)
        return await engine.scrape(urls)

    before = asyncio.create_task(at_warmup()) if urls else None
    records = await runner.run(base_url, model, requests, offsets)
    summary = summarize(records, warmup=warmup)
    if before is not None:
        after = await engine.scrape(urls)
        if before.done():
            summary["engine"] = engine.hit_rate(before.result(), after)
        else:
            before.cancel()
            summary["engine"] = engine.hit_rate(dict.fromkeys(urls), after)
    return records, summary


def describe(summary: dict) -> str:
    ttft = summary.get("ttft_s") or {}
    text = (
        f"{summary.get('ok', 0)}/{summary.get('requests', 0)} ok, "
        f"TTFT p50 {ttft.get('p50', float('nan')) * 1e3:.0f} ms "
        f"p99 {ttft.get('p99', float('nan')) * 1e3:.0f} ms, "
        f"hit rate {summary.get('hit_rate', 0):.2f}, "
        f"goodput {summary.get('goodput_rps', 0):.2f} req/s"
    )
    if predicted := summary.get("prediction"):
        text += f", predictions exact {predicted['cached_agreement']:.2f}"
    return text


def cmd_run(args: argparse.Namespace) -> int:
    params = workload_params(args.workload, args.sharegpt, args.param)
    arrivals = arrivals_for(args.workload, arrival_overrides(args))
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
            arrivals,
        )
    )
    run = {
        "workload": args.workload,
        "policy": args.label,
        "rate": args.rate,
        "arrivals": arrivals.to_dict(),
        "requests": args.requests,
        "seed": args.seed,
        "warmup_s": args.warmup,
        "params": parse_params(args.param),
        "target": args.base_url,
    }
    path = results.write(args.out, results.document(run, results.environment(), summary, records))
    print(f"{path}: {describe(summary)}")
    return 0 if summary.get("ok") else 1


def cmd_plot(args: argparse.Namespace) -> int:
    docs = [results.read(p) for p in args.results]
    print(plot.report(docs, args.out))
    return 0


def _rates(text: str, workloads: list[str]) -> dict[str, float]:
    if "=" not in text:
        return dict.fromkeys(workloads, float(text))
    rates = {}
    for part in text.split(","):
        name, value = part.split("=", 1)
        rates[name.strip()] = float(value)
    missing = set(workloads) - set(rates)
    if missing:
        raise SystemExit(f"--rate has no rate for {', '.join(sorted(missing))}")
    return rates


def _compare_params(items: Sequence[str]) -> dict[str, dict]:
    """`workload.key=value` pairs, grouped by workload."""
    grouped: dict[str, dict] = {}
    for item in items:
        name, dot, rest = item.partition(".")
        if not dot:
            raise SystemExit(f"--param {item!r} is not workload.key=value")
        grouped.setdefault(name, {}).update(parse_params([rest]))
    return grouped


def cmd_compare(args: argparse.Namespace) -> int:
    from inferrail_bench import compare

    workloads = args.workloads.split(",")
    plan = compare.Plan(
        workloads=tuple(workloads),
        policies=tuple(args.policies.split(",")),
        reps=args.reps,
        requests=args.requests,
        duration=args.duration,
        rates=_rates(args.rate, workloads) if args.rate else dict(compare.RATES),
        seed=args.seed,
        warmup=args.warmup,
        sims=args.sims,
        sim_flags=tuple(args.sim_flag) if args.sim_flag else compare.SIM_FLAGS,
        params=_compare_params(args.param) if args.param else compare.Plan().params,
        profile="debug" if args.debug else "release",
        arrivals=arrival_overrides(args),
    )
    written = compare.run(plan, args.out)
    docs = [results.read(p) for p in written]
    failed = sum(d["summary"].get("errors", 0) for d in docs)
    print(f"{len(written)} runs written to {args.out}; {failed} failed requests")
    return 0


def add_arrivals(command: argparse.ArgumentParser) -> None:
    command.add_argument(
        "--arrivals", choices=["poisson", "gamma"], help="arrival process; the workload's default"
    )
    command.add_argument("--cv", type=float, help="coefficient of variation of Gamma gaps")
    command.add_argument("--ramp", choices=["diurnal", "none"], help="vary the rate over the run")


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
    run.add_argument("--param", action="append", default=[], help="workload parameter, k=v")
    add_arrivals(run)
    run.add_argument("--out", type=Path, required=True)
    run.set_defaults(handler=cmd_run)

    cmp = sub.add_parser("compare", help="every policy on fresh simulators, same schedules")
    cmp.add_argument("--workloads", default="w1,w2,w6")
    cmp.add_argument("--policies", default="round_robin,random,least_request,p2c")
    cmp.add_argument("--reps", type=int, default=3)
    cmp.add_argument("--requests", type=int, help="per run; default rate times --duration")
    cmp.add_argument("--duration", type=float, default=60.0, help="seconds of arrivals per run")
    cmp.add_argument("--rate", help="one rate, or per workload as w1=12,w6=7")
    cmp.add_argument("--seed", type=int, default=0)
    cmp.add_argument("--warmup", type=float, default=10.0)
    cmp.add_argument("--sims", type=int, default=3)
    cmp.add_argument("--sim-flag", action="append", help="simulator flag, as --sim-flag=--x=y")
    cmp.add_argument(
        "--param", action="append", help="workload parameter, as w1.prompts=32; replaces defaults"
    )
    cmp.add_argument("--debug", action="store_true", help="use debug builds")
    add_arrivals(cmp)
    cmp.add_argument("--out", type=Path, required=True)
    cmp.set_defaults(handler=cmd_compare)

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
