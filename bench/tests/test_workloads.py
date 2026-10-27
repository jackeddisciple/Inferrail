import dataclasses
import itertools
import json
import math
import random
from collections import Counter

import pytest

from inferrail_bench import schedule, text
from inferrail_bench.workloads import (
    WORKLOADS,
    AgentLoop,
    FewShot,
    Mixed,
    MultiTurn,
    Rag,
    SharedPrompt,
    Unique,
    generate,
)
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


def test_w3_agents_resend_a_growing_context_after_each_tool_call():
    workload = AgentLoop()
    offsets = [i * 0.2 for i in range(600)]
    requests = workload.requests(600, seed=5, offsets=offsets)
    by_agent: dict[str, list] = {}
    for at, request in zip(offsets, requests, strict=True):
        by_agent.setdefault(request.session, []).append((at, request))
    assert 5 < len(by_agent) < 120
    systems = set(workload.system_prompts(5))
    for steps in by_agent.values():
        assert [r.turn for _, r in steps] == list(range(len(steps)))
        assert steps[0][1].messages[0].content in systems
        for (before, a), (after, b) in itertools.pairwise(steps):
            assert b.messages[: len(a.messages)] == a.messages
            assert len(b.messages) == len(a.messages) + 2
            assert b.messages[len(a.messages)].role == "assistant"
            assert text.tokens(b.messages[-1].content) - 3 >= 300
            assert after - before >= 1.0
        for _, request in steps:
            context = sum(text.tokens(m.content) for m in request.messages)
            assert context < workload.max_context
            assert 30 <= request.max_tokens <= 120
    longest = max(len(steps) for steps in by_agent.values())
    assert longest >= 10
    first = [r for r in requests if r.turn == 0]
    assert all(3_000 <= sum(text.tokens(m.content) for m in r.messages) <= 5_100 for r in first)


def test_w3_agents_stop_before_their_context_runs_out():
    workload = AgentLoop(steps=(200, 200), max_context=8_000)
    requests = workload.requests(400, seed=1, offsets=[i * 2.0 for i in range(400)])
    for request in requests:
        assert sum(text.tokens(m.content) for m in request.messages) < 8_000
    assert len({r.session for r in requests}) > 10


def test_w4_prompts_quote_zipf_popular_documents():
    workload = Rag()
    requests = workload.requests(3000, seed=6)
    first = Counter(r.tags["documents"][0] for r in requests)
    every = Counter(d for r in requests for d in r.tags["documents"])
    assert first.most_common(1)[0][0] == 0
    assert every[0] > every[10] > every[500]
    weights = [1 / rank**1.1 for rank in range(1, 2001)]
    assert abs(first[0] / 3000 - weights[0] / sum(weights)) < 0.03
    for request in requests:
        docs = request.tags["documents"]
        assert 1 <= len(docs) <= 4
        assert len(set(docs)) == len(docs)
        system, user = request.messages
        assert system.content.startswith("Answer the question")
        assert user.content.startswith(f"Document {docs[0]}: ")
        assert user.content.startswith(workload.document(6, docs[0]))
    assert workload.document(6, 3) == workload.document(6, 3)
    assert workload.document(6, 3) != workload.document(7, 3)


def test_w5_batches_share_a_long_block_of_examples():
    workload = FewShot()
    requests = workload.requests(800, seed=2)
    blocks = workload.examples(2)
    tasks = Counter(r.tags["task"] for r in requests)
    assert set(tasks) == set(range(4))
    assert min(tasks.values()) > 150
    for request in requests:
        (message,) = request.messages
        assert message.content.startswith(blocks[request.tags["task"]])
        assert 1 <= request.max_tokens <= 16
    for block in blocks:
        assert 2_900 <= text.tokens(block) <= 3_200
    assert len({r.messages[0].content for r in requests}) == len(requests)


def gaps_of(offsets):
    return [b - a for a, b in zip([0.0, *offsets], offsets, strict=False)]


@pytest.mark.parametrize("cv", [2.0, 4.0])
def test_gamma_arrivals_keep_the_rate_and_burst_at_their_cv(cv):
    arrivals = schedule.Arrivals("gamma", cv)
    offsets = arrivals.offsets(10.0, 40_000, seed=3)
    assert offsets == arrivals.offsets(10.0, 40_000, seed=3)
    assert offsets != arrivals.offsets(10.0, 40_000, seed=4)
    gaps = gaps_of(offsets)
    mean = sum(gaps) / len(gaps)
    spread = math.sqrt(sum((g - mean) ** 2 for g in gaps) / len(gaps))
    assert abs(mean - 0.1) < 0.004
    assert abs(spread / mean - cv) < 0.15 * cv
    assert all(b >= a for a, b in itertools.pairwise(offsets))
    assert arrivals.label() == f"gamma_cv{cv:g}"


def test_the_diurnal_ramp_peaks_midway_and_keeps_the_mean_rate():
    arrivals = schedule.Arrivals(ramp="diurnal")
    offsets = arrivals.offsets(10.0, 30_000, seed=1)
    span = offsets[-1]
    assert abs(span - 3_000) < 60

    def count(lo, hi):
        return sum(lo * span <= t < hi * span for t in offsets)

    edges = count(0.0, 0.1) + count(0.9, 1.0)
    middle = count(0.4, 0.6)
    assert middle > 3 * edges
    swing = (math.sin(2 * math.pi * 0.6) - math.sin(2 * math.pi * 0.4)) / (2 * math.pi * 0.2)
    want = 10.0 * (1 - 0.6 * swing) * 0.2 * span
    assert abs(middle - want) < 0.03 * want
    bursty = schedule.Arrivals("gamma", 4.0, "diurnal").offsets(10.0, 30_000, seed=1)
    assert abs(bursty[-1] - 3_000) < 300
    assert schedule.Arrivals("gamma", 4.0, "diurnal").label() == "gamma_cv4_diurnal"


def test_poisson_schedules_are_the_same_as_before_arrivals_were_configurable():
    assert schedule.poisson(5.0, 100, 7) == schedule.Arrivals().offsets(5.0, 100, 7)
    rng = random.Random(7)
    at, old = 0.0, []
    for _ in range(100):
        at += rng.expovariate(5.0)
        old.append(at)
    assert all(abs(a - b) < 1e-9 for a, b in zip(old, schedule.poisson(5.0, 100, 7), strict=True))


@pytest.mark.parametrize(
    "spec",
    [{"kind": "weibull"}, {"cv": 0}, {"ramp": "weekly"}, {"amplitude": 1.0}],
)
def test_bad_arrival_specs_are_refused(spec):
    with pytest.raises(ValueError, match=r"\w"):
        schedule.Arrivals.from_dict(spec)


def test_w8_is_shared_prompts_arriving_in_bursts_over_a_ramp():
    from inferrail_bench.cli import arrival_overrides, arrivals_for, parser

    assert arrivals_for("w8") == schedule.Arrivals("gamma", 2.0, "diurnal")
    assert arrivals_for("w1") == schedule.Arrivals()
    run = ["run", "--base-url", "u", "--model", "m", "--workload", "w8", "--rate", "1"]
    args = parser().parse_args([*run, "--out", "o", "--cv", "4"])
    assert arrivals_for("w8", arrival_overrides(args)).cv == 4.0
    args = parser().parse_args(["compare", "--out", "o", "--arrivals", "poisson", "--ramp", "none"])
    assert arrivals_for("w8", arrival_overrides(args)) == schedule.Arrivals()
    requests = generate("w8", 50, seed=1)
    assert len({r.messages[0].content for r in requests}) <= 32
