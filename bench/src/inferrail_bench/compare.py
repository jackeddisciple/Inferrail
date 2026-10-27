"""Policies against the same workloads, schedules and cold simulators: the
comparisons behind `make baselines` and `make routing`."""

import asyncio
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from inferrail_bench import fleet, results
from inferrail_bench.cli import arrivals_for, describe, measure

BASELINES = ("round_robin", "random", "least_request", "p2c")
POLICIES = (*BASELINES, "prefix_hash", "kv_aware", "cost_model")
SIM_FLAGS = ("--enable-prompt-tokens-details",)
RATES = {
    "w1": 6.0,
    "w2": 25.0,
    "w3": 2.0,
    "w4": 8.0,
    "w5": 10.0,
    "w6": 9.0,
    "mix": 7.0,
    "hot": 10.0,
}
PARAMS = {"w1": {"prompts": 32}}


@dataclass
class Plan:
    workloads: tuple[str, ...] = ("w1", "w2", "w6")
    policies: tuple[str, ...] = BASELINES
    reps: int = 3
    requests: int | None = None
    duration: float = 60.0
    rates: dict[str, float] = field(default_factory=lambda: dict(RATES))
    params: dict[str, dict] = field(default_factory=lambda: {k: dict(v) for k, v in PARAMS.items()})
    seed: int = 0
    warmup: float = 10.0
    sims: int = 3
    sim_flags: tuple[str, ...] = SIM_FLAGS
    profile: str = "release"
    arrivals: dict = field(default_factory=dict)


def run(plan: Plan, out: Path, log=print) -> list[Path]:
    """Runs every workload, repetition and policy, each on a fresh fleet, and
    writes one result file per run under `out/<workload>/`."""
    versions = {
        "gateway": fleet.version("inferrail-gateway", plan.profile),
        "sim": fleet.version("inferrail-sim", plan.profile),
    }
    written = []
    with tempfile.TemporaryDirectory(prefix="inferrail-bench-") as tmp:
        for workload in plan.workloads:
            rate = plan.rates[workload]
            count = plan.requests or round(rate * plan.duration)
            arrivals = arrivals_for(workload, plan.arrivals)
            for rep in range(plan.reps):
                seed = plan.seed + rep
                for policy in plan.policies:
                    work = Path(tmp) / f"{workload}-{policy}-{rep}"
                    with fleet.fleet(
                        work, plan.sims, policy, list(plan.sim_flags), plan.profile
                    ) as f:
                        records, summary = asyncio.run(
                            measure(
                                f.gateway,
                                fleet.MODEL,
                                workload,
                                rate,
                                count,
                                seed,
                                plan.warmup,
                                f.sims,
                                plan.params.get(workload),
                                arrivals,
                            )
                        )
                    run_info = {
                        "workload": workload,
                        "policy": policy,
                        "rep": rep,
                        "rate": rate,
                        "arrivals": arrivals.to_dict(),
                        "requests": count,
                        "seed": seed,
                        "warmup_s": plan.warmup,
                        "params": plan.params.get(workload, {}),
                        "sims": plan.sims,
                        "sim_flags": list(plan.sim_flags),
                    }
                    env = results.environment(
                        config_sha256=results.config_hash(f.config), **versions
                    )
                    doc = results.document(run_info, env, summary, records)
                    path = results.write(out / workload / f"{policy}-r{rep}.json.gz", doc)
                    written.append(path)
                    log(f"{workload} {policy} rep {rep}: {describe(summary)}")
    return written
