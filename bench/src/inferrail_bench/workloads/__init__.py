"""Workload generators. Each yields chat requests in arrival order; the
schedule decides when they are sent."""

from dataclasses import dataclass, field

from inferrail_bench.workloads.multi_turn import MultiTurn
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
    "w6": Spec("w6", "Unique prompts", Unique),
}


def generate(name: str, count: int, seed: int, **params) -> list[Request]:
    """`count` requests of workload `name`, the same for the same seed."""
    spec = WORKLOADS[name]
    return spec.build(**{**spec.defaults, **params}).requests(count, seed)


__all__ = ["WORKLOADS", "Message", "MultiTurn", "Request", "SharedPrompt", "Unique", "generate"]
