"""Arrival schedules for open loop load: when each request must be sent,
regardless of how earlier ones are doing."""

import random


def poisson(rate: float, count: int, seed: int) -> list[float]:
    """Offsets in seconds of `count` Poisson arrivals at `rate` per second."""
    if rate <= 0:
        raise ValueError("rate must be positive")
    rng = random.Random(seed)
    at = 0.0
    offsets = []
    for _ in range(count):
        at += rng.expovariate(rate)
        offsets.append(at)
    return offsets
