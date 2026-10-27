"""Workload generators. Each yields chat requests in arrival order; the
schedule decides when they are sent."""

from collections.abc import Sequence
from dataclasses import dataclass, field

from inferrail_bench.workloads.agent_loop import AgentLoop
from inferrail_bench.workloads.few_shot import FewShot
from inferrail_bench.workloads.mixed import Mixed
from inferrail_bench.workloads.multi_turn import MultiTurn
from inferrail_bench.workloads.rag import Rag
from inferrail_bench.workloads.shared_prompt import SharedPrompt
from inferrail_bench.workloads.types import Message, Request
from inferrail_bench.workloads.unique import Unique


@dataclass(frozen=True)
class Spec:
    name: str
    title: str
    build: type
    defaults: dict = field(default_factory=dict)


WORKLOADS: dict[str, Spec] = {
    "w1": Spec("w1", "Shared system prompt", SharedPrompt),
    "w2": Spec("w2", "Multi turn chat", MultiTurn),
    "w3": Spec("w3", "Agent loop", AgentLoop),
    "w4": Spec("w4", "RAG over popular documents", Rag),
    "w5": Spec("w5", "Few shot evaluation", FewShot),
    "w6": Spec("w6", "Unique prompts", Unique),
    "mix": Spec("mix", "Shared and unique prompts", Mixed),
    "hot": Spec("hot", "Hot system prompts", SharedPrompt, {"prompts": 32, "skew": 1.1}),
}


def generate(
    name: str, count: int, seed: int, offsets: Sequence[float] | None = None, **params
) -> list[Request]:
    """`count` requests of workload `name` arriving at `offsets`, the same for
    the same seed and schedule."""
    spec = WORKLOADS[name]
    return spec.build(**{**spec.defaults, **params}).requests(count, seed, offsets)


__all__ = [
    "WORKLOADS",
    "AgentLoop",
    "FewShot",
    "Message",
    "Mixed",
    "MultiTurn",
    "Rag",
    "Request",
    "SharedPrompt",
    "Unique",
    "generate",
]
