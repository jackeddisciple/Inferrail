"""Gateway behaviour visible to an OpenAI SDK client."""

import openai
import pytest

from conftest import MODEL

pytestmark = pytest.mark.e2e


@pytest.fixture
def gateway(gateway_base_url):
    from openai import OpenAI

    with OpenAI(base_url=f"{gateway_base_url}/v1", api_key="unused", max_retries=0) as client:
        yield client


def test_request_ids_and_replicas_are_reported(gateway):
    raw = gateway.chat.completions.with_raw_response.create(
        model=MODEL,
        messages=[{"role": "user", "content": "x"}],
        max_tokens=1,
        extra_headers={"x-request-id": "sdk-trace-1"},
    )
    assert raw.headers["x-request-id"] == "sdk-trace-1"
    assert raw.headers["x-inferrail-replica"] in {"r0", "r1", "r2"}
    assert raw.parse().id == "chatcmpl-sdk-trace-1"


def test_traffic_spreads_across_replicas(gateway):
    replicas = set()
    for _ in range(30):
        raw = gateway.completions.with_raw_response.create(model=MODEL, prompt="x", max_tokens=1)
        replicas.add(raw.headers["x-inferrail-replica"])
    assert replicas == {"r0", "r1", "r2"}


def test_gateway_errors_map_to_sdk_exceptions(gateway):
    with pytest.raises(openai.NotFoundError):
        gateway.chat.completions.create(model="missing", messages=[], max_tokens=1)
    with pytest.raises(openai.BadRequestError):
        gateway.chat.completions.create(model=MODEL, messages=[], max_tokens=1, n=2)
