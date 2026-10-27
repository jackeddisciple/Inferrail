"""W5: few shot evaluation. Every request of a task carries the same long
block of worked examples and asks one new question with a short answer, so
a batch reuses a 3k token prefix at a high rate."""

import random
from collections.abc import Sequence
from dataclasses import dataclass

from inferrail_bench.text import words
from inferrail_bench.workloads.types import Message, Request


@dataclass(frozen=True)
class FewShot:
    tasks: int = 4
    shot_tokens: int = 3000
    shots: int = 8
    question_tokens: tuple[int, int] = (50, 300)
    output_tokens: tuple[int, int] = (1, 16)

    def examples(self, seed: int) -> list[str]:
        rng = random.Random(f"w5-shots-{seed}")
        per_shot = max(1, self.shot_tokens // self.shots)
        return [
            f"Task {task} examples:\n"
            + "\n".join(
                f"Q{i}: {words(rng, per_shot // 2)} A{i}: {words(rng, per_shot - per_shot // 2)}"
                for i in range(self.shots)
            )
            for task in range(self.tasks)
        ]

    def requests(
        self, count: int, seed: int, offsets: Sequence[float] | None = None
    ) -> list[Request]:
        examples = self.examples(seed)
        rng = random.Random(f"w5-{seed}")
        out = []
        for _ in range(count):
            task = rng.randrange(len(examples))
            question = words(rng, rng.randint(*self.question_tokens))
            out.append(
                Request(
                    messages=(Message("user", f"{examples[task]}\nQuestion: {question}\nA:"),),
                    max_tokens=rng.randint(*self.output_tokens),
                    tags={"task": task},
                )
            )
        return out
