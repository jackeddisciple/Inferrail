"""Benchmark matrices from one sweep file: every point is a workload run at
each of its rates, repetitions and policies on a fresh fleet, with the same
requests and arrival times for every policy. Runs that already have a
result for the same specification are skipped, so a sweep that stopped
picks up where it left off."""

import hashlib
import json
import tempfile
import time
import tomllib
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from inferrail_bench import compare, results
from inferrail_bench.cli import arrivals_for, describe
from inferrail_bench.workloads import WORKLOADS

DEFAULTS = {
    "policies": list(compare.POLICIES),
    "reps": 3,
    "duration": 60.0,
    "warmup": 10.0,
    "sims": 3,
    "sim_flags": list(compare.SIM_FLAGS),
    "seed": 0,
}
POINT_KEYS = {"name", "workload", "rates", "params", "arrivals", *DEFAULTS}


@dataclass(frozen=True)
class Point:
    name: str
    workload: str
    rates: tuple[float, ...]
    policies: tuple[str, ...]
    reps: int
    duration: float
    warmup: float
    sims: int
    sim_flags: tuple[str, ...]
    seed: int
    params: dict = field(default_factory=dict)
    arrivals: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Run:
    point: Point
    rate: float
    policy: str
    rep: int

    @property
    def seed(self) -> int:
        return self.point.seed + self.rep

    @property
    def requests(self) -> int:
        return max(1, round(self.rate * self.point.duration))

    def info(self) -> dict:
        """What the result file records about the run; equal information
        means an equal run."""
        arrivals = arrivals_for(self.point.workload, self.point.arrivals)
        info = {
            "point": self.point.name,
            "workload": self.point.workload,
            "policy": self.policy,
            "rep": self.rep,
            "rate": self.rate,
            "arrivals": arrivals.to_dict(),
            "requests": self.requests,
            "seed": self.seed,
            "warmup_s": self.point.warmup,
            "params": self.point.params,
            "sims": self.point.sims,
            "sim_flags": list(self.point.sim_flags),
        }
        info["key"] = hashlib.sha256(json.dumps(info, sort_keys=True).encode()).hexdigest()[:16]
        return info

    def path(self, out: Path) -> Path:
        return out / self.point.name / f"rate{self.rate:g}" / f"{self.policy}-r{self.rep}.json.gz"


@dataclass(frozen=True)
class Sweep:
    name: str
    points: tuple[Point, ...]

    def runs(self, only: Sequence[str] = ()) -> list[Run]:
        """Every run, policies innermost, so drift over a long sweep falls on
        all policies of a comparison alike."""
        out = []
        for point in self.points:
            if only and point.name not in only:
                continue
            for rate in point.rates:
                for rep in range(point.reps):
                    out.extend(Run(point, rate, policy, rep) for policy in point.policies)
        return out


def parse(text: str) -> Sweep:
    data = tomllib.loads(text)
    unknown = set(data) - {"name", "point", *DEFAULTS}
    if unknown:
        raise ValueError(f"unknown sweep keys: {', '.join(sorted(unknown))}")
    shared = {**DEFAULTS, **{k: v for k, v in data.items() if k in DEFAULTS}}
    points = []
    names = set()
    for raw in data.get("point", []):
        unknown = set(raw) - POINT_KEYS
        if unknown:
            raise ValueError(f"unknown point keys: {', '.join(sorted(unknown))}")
        spec = {**shared, **raw}
        workload = spec["workload"]
        if workload not in WORKLOADS:
            raise ValueError(f"unknown workload {workload!r}")
        name = spec.get("name", workload)
        if name in names:
            raise ValueError(f"two points are named {name!r}")
        names.add(name)
        rates = tuple(float(r) for r in spec["rates"])
        if not rates or min(rates) <= 0:
            raise ValueError(f"point {name!r} needs positive rates")
        point = Point(
            name=name,
            workload=workload,
            rates=rates,
            policies=tuple(spec["policies"]),
            reps=int(spec["reps"]),
            duration=float(spec["duration"]),
            warmup=float(spec["warmup"]),
            sims=int(spec["sims"]),
            sim_flags=tuple(spec["sim_flags"]),
            seed=int(spec["seed"]),
            params=dict(spec.get("params", {})),
            arrivals=dict(spec.get("arrivals", {})),
        )
        arrivals_for(workload, point.arrivals)
        points.append(point)
    if not points:
        raise ValueError("a sweep needs at least one point")
    return Sweep(name=data.get("name", "sweep"), points=tuple(points))


def load(path: str | Path) -> Sweep:
    return parse(Path(path).read_text())


def done(run: Run, out: Path) -> bool:
    path = run.path(out)
    if not path.exists():
        return False
    try:
        return results.read(path)["run"].get("key") == run.info()["key"]
    except (OSError, ValueError, KeyError):
        return False


Execute = Callable[[Run, Path], dict]


def execute(profile: str) -> Execute:
    """Runs one point of the matrix on a fresh fleet of `profile` builds."""
    built = compare.versions(profile)

    def one(run: Run, work: Path) -> dict:
        return compare.one_run(
            work,
            run.info(),
            sims=run.point.sims,
            sim_flags=run.point.sim_flags,
            profile=profile,
            built=built,
            arrivals=arrivals_for(run.point.workload, run.point.arrivals),
        )

    return one


def run(
    sweep: Sweep,
    out: Path,
    execute_one: Execute,
    jobs: int = 1,
    only: Sequence[str] = (),
    log: Callable[[str], None] = lambda line: print(line, flush=True),
) -> list[Path]:
    """Runs every run without a current result and returns every result's
    path, old and new."""
    runs = sweep.runs(only)
    todo = [r for r in runs if not done(r, out)]
    log(f"{len(runs)} runs, {len(runs) - len(todo)} already done, {len(todo)} to go")
    started = time.monotonic()

    def one(index: int, item: Run) -> None:
        with tempfile.TemporaryDirectory(prefix="inferrail-sweep-") as tmp:
            doc = execute_one(item, Path(tmp))
        results.write(item.path(out), doc)
        spent = time.monotonic() - started
        log(
            f"[{index + 1}/{len(todo)} {spent / 60:.0f} min] {item.point.name} "
            f"rate {item.rate:g} {item.policy} rep {item.rep}: {describe(doc['summary'])}"
        )

    if jobs <= 1:
        for index, item in enumerate(todo):
            one(index, item)
    else:
        with ThreadPoolExecutor(max_workers=jobs) as pool:
            for future in [pool.submit(one, i, r) for i, r in enumerate(todo)]:
                future.result()
    return [r.path(out) for r in runs]


def estimate(sweep: Sweep, only: Sequence[str] = (), overhead: float = 8.0) -> float:
    """Seconds the sweep takes on one job: arrivals plus fleet start up."""
    return sum(r.point.duration + overhead for r in sweep.runs(only))
