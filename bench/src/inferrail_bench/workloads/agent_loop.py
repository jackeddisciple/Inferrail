"""W3: agent loops. Each agent works on its own task over a shared system
prompt of tool definitions, and every step resends its whole context with
the last tool call and its result appended, so prefixes are long and grow
by hundreds of tokens a step. An arrival continues an agent whose tool has
returned, at least `tool_time` after its previous step, or starts a new one;
an agent stops after its steps or before its context would pass
`max_context`."""

import heapq
import random
from collections.abc import Sequence
from dataclasses import dataclass, field

from inferrail_bench.text import words
from inferrail_bench.workloads.types import Message, Request


@dataclass
class _Agent:
    id: str
    system: int
    messages: list[Message]
    steps: int
    tokens: int
    step: int = 0
    pending: list[Message] = field(default_factory=list)


@dataclass(frozen=True)
class AgentLoop:
    systems: int = 2
    system_tokens: tuple[int, int] = (2500, 3500)
    task_tokens: tuple[int, int] = (500, 1500)
    steps: tuple[int, int] = (10, 60)
    step_tokens: tuple[int, int] = (300, 1500)
    output_tokens: tuple[int, int] = (30, 120)
    tool_time: tuple[float, float] = (1.0, 4.0)
    max_context: int = 28_000

    def system_prompts(self, seed: int) -> list[str]:
        rng = random.Random(f"w3-system-{seed}")
        return [
            f"Agent tools {i}: " + words(rng, rng.randint(*self.system_tokens))
            for i in range(self.systems)
        ]

    def _agent(self, rng: random.Random, systems: list[str], number: int, seed: int) -> _Agent:
        which = rng.randrange(len(systems))
        task = words(rng, rng.randint(*self.task_tokens))
        messages = [
            Message("system", systems[which]),
            Message("user", f"Task a{seed}-{number}: {task}"),
        ]
        tokens = len(systems[which].split()) + len(task.split())
        return _Agent(f"a{seed}-{number}", which, messages, rng.randint(*self.steps), tokens)

    def requests(
        self, count: int, seed: int, offsets: Sequence[float] | None = None
    ) -> list[Request]:
        """`count` steps for arrivals at `offsets` (one per second if none)."""
        offsets = offsets if offsets is not None else [float(i) for i in range(count)]
        rng = random.Random(f"w3-{seed}")
        systems = self.system_prompts(seed)
        started = 0
        working: list[tuple[float, int, _Agent]] = []
        ready: list[_Agent] = []
        out = []
        for at in offsets[:count]:
            while working and working[0][0] <= at:
                ready.append(heapq.heappop(working)[2])
            if ready:
                agent = ready.pop(rng.randrange(len(ready)))
            else:
                agent = self._agent(rng, systems, started, seed)
                started += 1
            agent.messages.extend(agent.pending)
            output = rng.randint(*self.output_tokens)
            out.append(
                Request(
                    messages=tuple(agent.messages),
                    max_tokens=output,
                    session=agent.id,
                    turn=agent.step,
                    tags={"system": agent.system, "context": agent.tokens},
                )
            )
            call = words(rng, output)
            result = words(rng, rng.randint(*self.step_tokens))
            agent.pending = [
                Message("assistant", call),
                Message("user", f"Tool result {agent.step}: {result}"),
            ]
            agent.tokens += output + len(result.split()) + 3
            agent.step += 1
            if agent.step < agent.steps and agent.tokens + self.step_tokens[1] < self.max_context:
                due = at + rng.uniform(*self.tool_time)
                heapq.heappush(working, (due, len(out), agent))
        return out
