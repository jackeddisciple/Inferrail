"""W1: a few long system prompts shared by every user, short user turns.

The product's best case: almost all of each prompt is a prefix some other
request already sent. With `skew` above zero the system prompts are drawn
with Zipf weights instead of uniformly, so a few of them are hot."""

import random
from collections.abc import Sequence
from dataclasses import dataclass

from inferrail_bench.text import words
from inferrail_bench.workloads.types import Message, Request


@dataclass(frozen=True)
class SharedPrompt:
    prompts: int = 8
    system_tokens: tuple[int, int] = (2000, 8000)
    user_tokens: tuple[int, int] = (50, 500)
    output_tokens: tuple[int, int] = (32, 128)
    skew: float = 0.0

    def system_prompts(self, seed: int) -> list[str]:
        rng = random.Random(f"w1-system-{seed}")
        return [
            f"System prompt {i}: " + words(rng, rng.randint(*self.system_tokens))
            for i in range(self.prompts)
        ]

    def requests(
        self, count: int, seed: int, offsets: Sequence[float] | None = None
    ) -> list[Request]:
        systems = self.system_prompts(seed)
        rng = random.Random(f"w1-{seed}")
        weights = [1 / (rank**self.skew) for rank in range(1, len(systems) + 1)]
        out = []
        for _ in range(count):
            if self.skew:
                which = rng.choices(range(len(systems)), weights=weights)[0]
            else:
                which = rng.randrange(len(systems))
            out.append(
                Request(
                    messages=(
                        Message("system", systems[which]),
                        Message("user", words(rng, rng.randint(*self.user_tokens))),
                    ),
                    max_tokens=rng.randint(*self.output_tokens),
                    tags={"prompt": which},
                )
            )
        return out
