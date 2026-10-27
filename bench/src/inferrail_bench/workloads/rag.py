"""W4: retrieval augmented generation over a corpus whose popular documents
are drawn far more often: each prompt quotes one to four documents drawn by
Zipf weight, in the order retrieved, after a short shared instruction. Only
the instruction and a hot first document are shared, so reuse is partial
and skewed toward a few prefixes."""

import random
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import accumulate

from inferrail_bench.text import words
from inferrail_bench.workloads.types import Message, Request

INSTRUCTION = "Answer the question using only the documents below."


@dataclass(frozen=True)
class Rag:
    documents: int = 2000
    doc_tokens: tuple[int, int] = (200, 1000)
    per_prompt: tuple[int, int] = (1, 4)
    skew: float = 1.1
    question_tokens: tuple[int, int] = (20, 120)
    output_tokens: tuple[int, int] = (32, 192)

    def document(self, seed: int, doc: int) -> str:
        rng = random.Random(f"w4-doc-{seed}-{doc}")
        return f"Document {doc}: " + words(rng, rng.randint(*self.doc_tokens))

    def requests(
        self, count: int, seed: int, offsets: Sequence[float] | None = None
    ) -> list[Request]:
        rng = random.Random(f"w4-{seed}")
        weights = list(accumulate(1 / rank**self.skew for rank in range(1, self.documents + 1)))
        cache: dict[int, str] = {}
        out = []
        for _ in range(count):
            wanted = min(rng.randint(*self.per_prompt), self.documents)
            docs: list[int] = []
            while len(docs) < wanted:
                doc = rng.choices(range(self.documents), cum_weights=weights)[0]
                if doc not in docs:
                    docs.append(doc)
            quoted = "\n\n".join(cache.setdefault(d, self.document(seed, d)) for d in docs)
            question = words(rng, rng.randint(*self.question_tokens))
            out.append(
                Request(
                    messages=(
                        Message("system", INSTRUCTION),
                        Message("user", f"{quoted}\n\nQuestion: {question}"),
                    ),
                    max_tokens=rng.randint(*self.output_tokens),
                    tags={"documents": docs},
                )
            )
        return out
