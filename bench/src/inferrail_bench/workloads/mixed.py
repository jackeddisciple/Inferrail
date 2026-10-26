"""Shared system prompts and unique prompts arriving together: each request
is a W1 request with probability `shared`, a W6 one otherwise. A policy has
to weigh the cache one kind finds against the prefill the other queues."""

import dataclasses
import random
from collections.abc import Sequence
from dataclasses import dataclass

from inferrail_bench.workloads.shared_prompt import SharedPrompt
from inferrail_bench.workloads.types import Request
from inferrail_bench.workloads.unique import Unique


@dataclass(frozen=True)
class Mixed:
    shared: float = 0.5
    prompts: int = 32

    def requests(
        self, count: int, seed: int, offsets: Sequence[float] | None = None
    ) -> list[Request]:
        rng = random.Random(f"mix-{seed}")
        picks = [rng.random() < self.shared for _ in range(count)]
        shared = iter(SharedPrompt(prompts=self.prompts).requests(sum(picks), seed))
        unique = iter(Unique().requests(count - sum(picks), seed))
        out = []
        for pick in picks:
            request = next(shared) if pick else next(unique)
            kind = "shared" if pick else "unique"
            out.append(dataclasses.replace(request, tags={**request.tags, "kind": kind}))
        return out
