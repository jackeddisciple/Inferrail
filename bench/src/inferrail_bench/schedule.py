"""Arrival schedules for open loop load: when each request must be sent,
regardless of how earlier ones are doing."""

import math
import random
from dataclasses import asdict, dataclass

KINDS = ("poisson", "gamma")
RAMPS = (None, "diurnal")


def poisson(rate: float, count: int, seed: int) -> list[float]:
    """Offsets in seconds of `count` Poisson arrivals at `rate` per second."""
    return Arrivals().offsets(rate, count, seed)


@dataclass(frozen=True)
class Arrivals:
    """A renewal process with mean gap `1 / rate`: exponential gaps (Poisson)
    or Gamma gaps whose coefficient of variation is `cv`, so CV 2 and 4 come
    in bursts with long lulls between. With the diurnal ramp the rate
    follows one compressed day over the run, `rate * (1 - amplitude *
    cos(2 pi t / T))`, lowest at the start and end and highest midway."""

    kind: str = "poisson"
    cv: float = 1.0
    ramp: str | None = None
    amplitude: float = 0.6

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"unknown arrivals {self.kind!r}; expected one of {KINDS}")
        if self.ramp not in RAMPS:
            raise ValueError(f"unknown ramp {self.ramp!r}; expected diurnal or none")
        if not self.cv > 0:
            raise ValueError("cv must be positive")
        if not 0 <= self.amplitude < 1:
            raise ValueError("amplitude must be at least 0 and below 1")
        if self.kind == "poisson":
            object.__setattr__(self, "cv", 1.0)

    @classmethod
    def from_dict(cls, spec: dict | None) -> "Arrivals":
        return cls(**(spec or {}))

    def to_dict(self) -> dict:
        return asdict(self)

    def label(self) -> str:
        base = "poisson" if self.kind == "poisson" else f"gamma_cv{self.cv:g}"
        return f"{base}_diurnal" if self.ramp else base

    def gaps(self, count: int, rng: random.Random) -> list[float]:
        """Gaps with mean 1."""
        if self.kind == "poisson":
            return [rng.expovariate(1.0) for _ in range(count)]
        shape = 1 / self.cv**2
        return [rng.gammavariate(shape, 1 / shape) for _ in range(count)]

    def offsets(self, rate: float, count: int, seed: int) -> list[float]:
        """Offsets in seconds of `count` arrivals at a mean of `rate` per second."""
        if rate <= 0:
            raise ValueError("rate must be positive")
        plain = self.kind == "poisson" and not self.ramp
        rng = random.Random(seed if plain else f"{self}-{seed}")
        gaps = self.gaps(count, rng)
        at = 0.0
        out = []
        if not self.ramp:
            for gap in gaps:
                at += gap / rate
                out.append(at)
            return out
        span = count / rate
        for gap in gaps:
            at += gap
            out.append(_invert(at, rate, span, self.amplitude))
        return out


def _cumulative(t: float, rate: float, span: float, amplitude: float) -> float:
    """Expected arrivals by `t` under the diurnal rate."""
    w = 2 * math.pi / span
    return rate * (t - amplitude * math.sin(w * t) / w)


def _invert(target: float, rate: float, span: float, amplitude: float) -> float:
    """The time by which `target` arrivals are expected, by bisection; the
    cumulative rate only rises, so the answer is unique."""
    low, high = 0.0, target / (rate * (1 - amplitude)) + span
    for _ in range(200):
        mid = (low + high) / 2
        if _cumulative(mid, rate, span, amplitude) < target:
            low = mid
        else:
            high = mid
        if high - low < 1e-9:
            break
    return (low + high) / 2
