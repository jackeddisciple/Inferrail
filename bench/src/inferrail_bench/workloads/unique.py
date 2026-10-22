"""W6: prompts that share nothing, the control where cache affinity cannot
help and must not hurt."""

import random
from dataclasses import dataclass

from inferrail_bench.text import words
from inferrail_bench.workloads.types import Message, Request


@dataclass(frozen=True)
class Unique:
    prompt_tokens: tuple[int, int] = (100, 4000)
    output_tokens: tuple[int, int] = (32, 128)

    def requests(self, count: int, seed: int) -> list[Request]:
        rng = random.Random(f"w6-{seed}")
        out = []
        for i in range(count):
            tag = f"{rng.getrandbits(64):016x}"
            length = rng.randint(*self.prompt_tokens)
            out.append(
                Request(
                    messages=(Message("user", f"{tag} {i} " + words(rng, length)),),
                    max_tokens=rng.randint(*self.output_tokens),
                )
            )
        return out
