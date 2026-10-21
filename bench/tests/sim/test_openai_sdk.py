"""The official OpenAI SDK must work against the simulator with only a base URL change."""

import asyncio

import openai
import pytest
from openai import AsyncOpenAI

from conftest import MODEL

pytestmark = pytest.mark.e2e

LONG_SYSTEM = " ".join(f"rule{i} applies to every request" for i in range(300))


def test_chat_completion(client):
    reply = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "Say hello."}],
        max_tokens=6,
    )
    assert reply.id.startswith("chatcmpl-")
    assert reply.object == "chat.completion"
    assert reply.model == MODEL
    choice = reply.choices[0]
    assert choice.message.role == "assistant"
    assert choice.message.content
    assert choice.finish_reason == "length"
    assert reply.usage.completion_tokens == 6
    assert reply.usage.total_tokens == reply.usage.prompt_tokens + 6
    assert reply.usage.prompt_tokens_details.cached_tokens == 0


def test_streaming_chat_with_usage(client):
    stream = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "Count to five."}],
        max_tokens=5,
        stream=True,
        stream_options={"include_usage": True},
    )
    chunks = list(stream)
    assert chunks[0].choices[0].delta.role == "assistant"
    text = "".join(c.choices[0].delta.content or "" for c in chunks if c.choices)
    assert text
    finishing = [c for c in chunks if c.choices and c.choices[0].finish_reason]
    assert [c.choices[0].finish_reason for c in finishing] == ["length"]
    assert chunks[-1].choices == []
    assert chunks[-1].usage.completion_tokens == 5
    assert len({c.id for c in chunks}) == 1


def test_prefix_cache_is_visible_in_usage(client):
    def ask(question):
        return client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": LONG_SYSTEM},
                {"role": "user", "content": question},
            ],
            max_tokens=2,
        )

    first = ask("first question")
    second = ask("second question")
    assert second.usage.prompt_tokens_details.cached_tokens > 0.9 * first.usage.prompt_tokens


def test_legacy_completions_plain_and_streaming(client):
    reply = client.completions.create(model=MODEL, prompt="Once upon a time", max_tokens=4)
    assert reply.object == "text_completion"
    assert reply.choices[0].text
    assert reply.usage.completion_tokens == 4

    chunks = list(
        client.completions.create(
            model=MODEL,
            prompt=[128000, 791, 6864],
            max_tokens=3,
            stream=True,
            stream_options={"include_usage": True},
        )
    )
    assert len(chunks) == 4
    assert chunks[2].choices[0].finish_reason == "length"
    assert chunks[3].usage.prompt_tokens == 3


def test_models_list(client):
    ids = [model.id for model in client.models.list()]
    assert ids == [MODEL, "sim-alias"]


def test_vllm_extensions_pass_through_extra_body(client):
    reply = client.chat.completions.create(
        model="sim-alias",
        messages=[{"role": "user", "content": "x"}],
        max_tokens=3,
        seed=7,
        extra_body={"top_k": 20, "min_p": 0.1, "ignore_eos": True},
    )
    assert reply.model == "sim-alias"
    again = client.chat.completions.create(
        model="sim-alias",
        messages=[{"role": "user", "content": "x"}],
        max_tokens=3,
        seed=7,
    )
    assert reply.choices[0].message.content == again.choices[0].message.content


def test_errors_map_to_sdk_exceptions(client):
    with pytest.raises(openai.NotFoundError) as missing:
        client.chat.completions.create(model="gpt-4", messages=[], max_tokens=1)
    assert "does not exist" in str(missing.value)
    with pytest.raises(openai.BadRequestError):
        client.chat.completions.create(model=MODEL, messages=[], n=2)
    with pytest.raises(openai.BadRequestError):
        client.chat.completions.create(model=MODEL, messages=[], max_tokens=10**6)


def test_raw_response_headers(client):
    raw = client.chat.completions.with_raw_response.create(
        model=MODEL, messages=[{"role": "user", "content": "x"}], max_tokens=1
    )
    assert raw.status_code == 200
    assert raw.headers["content-type"] == "application/json"
    assert raw.parse().usage.completion_tokens == 1


def test_async_client_runs_many_streams_concurrently(sim_base_url):
    async def run():
        async with AsyncOpenAI(base_url=f"{sim_base_url}/v1", api_key="unused") as client:

            async def one(i):
                stream = await client.chat.completions.create(
                    model=MODEL,
                    messages=[{"role": "user", "content": f"request {i}"}],
                    max_tokens=8,
                    stream=True,
                )
                return [chunk async for chunk in stream]

            return await asyncio.gather(*(one(i) for i in range(100)))

    results = asyncio.run(run())
    assert all(len(chunks) == 9 for chunks in results)
