import dataclasses
import itertools
import json
import math
import random
from collections import Counter

import pytest

from inferrail_bench import schedule, text
from inferrail_bench.workloads import WORKLOADS, Mixed, MultiTurn, SharedPrompt, Unique, generate
from inferrail_bench.workloads.multi_turn import load_sharegpt


def prompt(request) -> str:
    return "\n".join(f"{m.role}: {m.content}" for m in request.messages)


@pytest.mark.parametrize("name", sorted(WORKLOADS))
def test_same_seed_same_requests_and_other_seeds_differ(name):
    first = generate(name, 60, seed=7)
    assert first == generate(name, 60, seed=7)
    assert first != generate(name, 60, seed=8)
    assert len(first) == 60
    for request in first:
        body = request.body("m")
        assert body["stream"] is True
        assert body["stream_options"] == {"include_usage": True}
        assert body["max_tokens"] >= 1
        assert all(isinstance(m["content"], str) and m["content"] for m in body["messages"])


def test_words_are_one_token_each():
    rng = random.Random(1)
    sample = text.words(rng, 500)
    assert text.tokens(sample) == 500
    assert all(3 <= len(w) <= 4 and w.isalpha() and w.islower() for w in sample.split())
    assert len(text.VOCABULARY) > 1000


def test_w1_requests_share_one_of_a_few_long_system_prompts():
    workload = SharedPrompt(prompts=4)
    requests = workload.requests(400, seed=3)
    systems = {r.messages[0].content for r in requests}
    assert len(systems) == 4
    assert systems <= set(workload.system_prompts(3))
    for system in systems:
        assert 2000 <= text.tokens(system) - 3 <= 8000
    for request in requests:
        assert [m.role for m in request.messages] == ["system", "user"]
        assert 50 <= text.tokens(request.messages[1].content) <= 500
        assert 32 <= request.max_tokens <= 128
    counts = [sum(r.tags["prompt"] == i for r in requests) for i in range(4)]
    assert min(counts) > 60


def test_hot_system_prompts_are_drawn_by_zipf_weight():
    requests = generate("hot", 4000, seed=2)
    counts = Counter(r.tags["prompt"] for r in requests)
    weights = [1 / rank**1.1 for rank in range(1, 33)]
    assert abs(counts[0] / 4000 - weights[0] / sum(weights)) < 0.03
    assert counts[0] > counts[3] > counts[31]
    assert len({r.messages[0].content for r in requests}) <= 32
    assert SharedPrompt().requests(50, seed=1) == SharedPrompt(skew=0.0).requests(50, seed=1)


def test_mix_interleaves_shared_and_unique_prompts():
    requests = Mixed().requests(1000, seed=4)
    kinds = Counter(r.tags["kind"] for r in requests)
    assert 420 < kinds["shared"] < 580
    shared = [r for r in requests if r.tags["kind"] == "shared"]
    unique = [r for r in requests if r.tags["kind"] == "unique"]
    assert all([m.role for m in r.messages] == ["system", "user"] for r in shared)
    assert len({r.messages[0].content for r in shared}) <= 32
    assert len({r.messages[0].content[:16] for r in unique}) == len(unique)
    assert Mixed(shared=1.0).requests(20, seed=4) == [
        dataclasses.replace(r, tags={**r.tags, "kind": "shared"})
        for r in SharedPrompt(prompts=32).requests(20, seed=4)
    ]


def test_w2_turns_extend_their_own_history_in_order():
    offsets = schedule.poisson(25, 600, seed=11)
    requests = MultiTurn(turns=(2, 6)).requests(600, seed=11, offsets=offsets)
    by_session: dict[str, list] = {}
    for r in requests:
        by_session.setdefault(r.session, []).append(r)
    assert len(by_session) > 50
    for turns in by_session.values():
        assert [t.turn for t in turns] == list(range(len(turns)))
        for earlier, later in itertools.pairwise(turns):
            assert later.messages[: len(earlier.messages)] == earlier.messages
            assert len(later.messages) == len(earlier.messages) + 2
            assert later.messages[len(earlier.messages)].role == "assistant"
            assert text.tokens(later.messages[len(earlier.messages)].content) == earlier.max_tokens
    openings = {turns[0].messages[0].content.split(":")[0] for turns in by_session.values()}
    assert len(openings) == len(by_session)
    interleaved = [r.session for r in requests[:20]]
    assert len(set(interleaved)) > 3


def test_w2_waits_a_think_time_between_a_conversations_turns():
    offsets = schedule.poisson(25, 1500, seed=0)
    requests = MultiTurn().requests(1500, seed=0, offsets=offsets)
    last: dict[str, float] = {}
    gaps = []
    for request, at in zip(requests, offsets, strict=True):
        if request.session in last:
            gaps.append(at - last[request.session])
        last[request.session] = at
    assert gaps, "no conversation got a second turn"
    assert min(gaps) >= 10.0, min(gaps)
    assert sum(r.turn > 0 for r in requests) > 600
    assert requests == MultiTurn().requests(1500, seed=0, offsets=offsets)


def test_w2_replays_sharegpt_conversations(tmp_path):
    data = [
        {
            "conversations": [
                {"from": "human", "value": "hi"},
                {"from": "gpt", "value": "hello there"},
                {"from": "human", "value": "how are you"},
                {"from": "gpt", "value": "fine thanks"},
            ]
        },
        {"conversations": [{"from": "human", "value": "only one"}]},
        {
            "conversations": [
                {"from": "gpt", "value": "wrong order"},
                {"from": "human", "value": "x"},
            ]
        },
        "not an object",
        {"conversations": [{"from": "human", "value": 3}, {"from": "gpt", "value": "x"}]},
    ]
    path = tmp_path / "sharegpt.json"
    path.write_text(json.dumps(data))
    conversations = load_sharegpt(path)
    assert conversations == [[("hi", "hello there"), ("how are you", "fine thanks")]]
    workload = MultiTurn(conversations=tuple(tuple(c) for c in conversations), think_time=(0, 0))
    requests = workload.requests(4, seed=0)
    assert requests[0].messages[0].content.endswith(": hi")
    assert [m.content for m in requests[1].messages[1:]] == ["hello there", "how are you"]
    assert requests[1].max_tokens == 2


def test_w6_prompts_share_no_prefix():
    requests = Unique().requests(500, seed=5)
    heads = {r.messages[0].content[:16] for r in requests}
    assert len(heads) == 500
    for r in requests:
        assert 100 <= text.tokens(r.messages[0].content) - 2 <= 4000


def test_poisson_offsets_have_the_right_rate_and_shape():
    rate = 50.0
    offsets = schedule.poisson(rate, 20_000, seed=1)
    assert offsets == sorted(offsets)
    assert offsets == schedule.poisson(rate, 20_000, seed=1)
    gaps = [b - a for a, b in zip([0.0, *offsets], offsets, strict=False)]
    mean = sum(gaps) / len(gaps)
    assert abs(mean - 1 / rate) < 0.02 / rate
    gaps.sort()
    n = len(gaps)
    statistic = max(
        max((i + 1) / n - (1 - math.exp(-rate * g)), (1 - math.exp(-rate * g)) - i / n)
        for i, g in enumerate(gaps)
    )
    assert statistic < 1.36 / math.sqrt(n)
    with pytest.raises(ValueError, match="positive"):
        schedule.poisson(0, 1, seed=1)
