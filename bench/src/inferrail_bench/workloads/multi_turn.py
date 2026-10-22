"""W2: multi turn chat. Every turn resends the whole conversation so far, so
reuse grows within a conversation. Each arrival continues a conversation
whose user has finished thinking, at least `think_time` after its previous
turn, so a reply can arrive before the next question is sent, or starts a
new conversation when none is ready; arrivals stay Poisson and every
conversation keeps its order."""

import heapq
import json
import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from inferrail_bench.text import tokens, words
from inferrail_bench.workloads.types import Message, Request

Turn = tuple[str, str]


def load_sharegpt(path: str | Path, min_turns: int = 2) -> list[list[Turn]]:
    """Conversations from a ShareGPT style JSON file: a list of objects whose
    `conversations` alternate `human` and `gpt` messages."""
    conversations = []
    for item in json.loads(Path(path).read_text()):
        turns: list[Turn] = []
        messages = item.get("conversations", []) if isinstance(item, dict) else []
        for user, reply in zip(messages[::2], messages[1::2], strict=False):
            if not (isinstance(user, dict) and isinstance(reply, dict)):
                break
            if (user.get("from"), reply.get("from")) != ("human", "gpt"):
                break
            if not (isinstance(user.get("value"), str) and isinstance(reply.get("value"), str)):
                break
            turns.append((user["value"], reply["value"]))
        if len(turns) >= min_turns:
            conversations.append(turns)
    return conversations


@dataclass
class _Session:
    id: str
    turns: list[Turn]
    position: int = 0


@dataclass(frozen=True)
class MultiTurn:
    turns: tuple[int, int] = (3, 10)
    user_tokens: tuple[int, int] = (50, 300)
    assistant_tokens: tuple[int, int] = (50, 300)
    think_time: tuple[float, float] = (10.0, 20.0)
    conversations: tuple[tuple[Turn, ...], ...] | None = None

    def _session(self, rng: random.Random, number: int, seed: int) -> _Session:
        sid = f"s{seed}-{number}"
        if self.conversations:
            turns = list(self.conversations[number % len(self.conversations)])
        else:
            turns = [
                (
                    words(rng, rng.randint(*self.user_tokens)),
                    words(rng, rng.randint(*self.assistant_tokens)),
                )
                for _ in range(rng.randint(*self.turns))
            ]
        first, reply = turns[0]
        turns[0] = (f"Conversation {sid}: {first}", reply)
        return _Session(sid, turns)

    def requests(
        self, count: int, seed: int, offsets: Sequence[float] | None = None
    ) -> list[Request]:
        """`count` turns for arrivals at `offsets` (one per second if none)."""
        offsets = offsets if offsets is not None else [float(i) for i in range(count)]
        rng = random.Random(f"w2-{seed}")
        started = 0
        thinking: list[tuple[float, int, _Session]] = []
        ready: list[_Session] = []
        out = []
        for at in offsets[:count]:
            while thinking and thinking[0][0] <= at:
                ready.append(heapq.heappop(thinking)[2])
            if ready:
                session = ready.pop(rng.randrange(len(ready)))
            else:
                session = self._session(rng, started, seed)
                started += 1
            history: list[Message] = []
            for user, reply in session.turns[: session.position]:
                history += [Message("user", user), Message("assistant", reply)]
            user, reply = session.turns[session.position]
            history.append(Message("user", user))
            out.append(
                Request(
                    messages=tuple(history),
                    max_tokens=max(1, tokens(reply)),
                    session=session.id,
                    turn=session.position,
                )
            )
            session.position += 1
            if session.position < len(session.turns):
                due = at + rng.uniform(*self.think_time)
                heapq.heappush(thinking, (due, len(out), session))
        return out
