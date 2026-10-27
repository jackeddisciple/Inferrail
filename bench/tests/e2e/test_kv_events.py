"""The simulator's KV events read by libzmq through pyzmq and decoded with
vLLM v0.31.0's own msgspec structs, as a consumer of a real engine would."""

import importlib.util
import json
import time

import httpx2
import msgspec
import pytest
import zmq

from conftest import REPO_ROOT, SIM_FLAGS, _binary, _free_port, _start, _stop

pytestmark = pytest.mark.e2e

FIXTURE = REPO_ROOT / "tests" / "fixtures" / "vllm" / "kv_events" / "make_fixture.py"
END_SEQ = (-1).to_bytes(8, "big", signed=True)


def vllm_structs():
    spec = importlib.util.spec_from_file_location("vllm_kv_events", FIXTURE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def publishing_sim(tmp_path):
    port, events, replay = _free_port(), _free_port(), _free_port()
    config = {
        "enable_kv_cache_events": True,
        "publisher": "zmq",
        "endpoint": f"tcp://127.0.0.1:{events}",
        "replay_endpoint": f"tcp://127.0.0.1:{replay}",
        "topic": "kv-events",
    }
    args = [str(_binary("inferrail-sim")), "--host", "127.0.0.1", "--port", str(port)]
    args += ["--served-model-name", "sim-model", *SIM_FLAGS]
    args += ["--num-gpu-blocks-override", "128", "--max-model-len", "2048"]
    args += ["--kv-events-config", json.dumps(config)]
    url = f"http://127.0.0.1:{port}"
    process = _start(args, tmp_path / "sim.log", f"{url}/health")
    yield url, f"tcp://127.0.0.1:{events}", f"tcp://127.0.0.1:{replay}"
    _stop(process)


def test_libzmq_subscribers_decode_vllm_batches(publishing_sim):
    url, events, replay_endpoint = publishing_sim
    structs = vllm_structs()
    decoder = msgspec.msgpack.Decoder(type=structs.KVEventBatch)
    context = zmq.Context()
    sub = context.socket(zmq.SUB)
    sub.setsockopt(zmq.SUBSCRIBE, b"kv")
    sub.setsockopt(zmq.RCVTIMEO, 500)
    sub.connect(events)
    try:
        time.sleep(0.3)
        prompt = list(range(1000, 1300))
        for round in range(25):
            body = {
                "model": "sim-model",
                "prompt": prompt + [5000 + round] * 40,
                "max_tokens": 60,
            }
            reply = httpx2.post(f"{url}/v1/completions", json=body, timeout=30)
            assert reply.status_code == 200, reply.text
        live = []
        while True:
            try:
                live.append(sub.recv_multipart())
            except zmq.Again:
                break
        assert live, "no events arrived"
        held: dict[int, int] = {}
        kinds = set()
        for i, (topic, seq, payload) in enumerate(live):
            assert topic == b"kv-events"
            assert int.from_bytes(seq, "big") == i
            batch = decoder.decode(payload)
            assert batch.data_parallel_rank == 0
            for event in batch.events:
                kinds.add(type(event).__name__)
                if isinstance(event, structs.BlockStored):
                    assert len(event.token_ids) == 16 * len(event.block_hashes)
                    assert (event.medium, event.group_idx, event.block_size) == ("GPU", 0, 16)
                    for h in event.block_hashes:
                        held[h] = held.get(h, 0) + 1
                elif isinstance(event, structs.BlockRemoved):
                    for h in event.block_hashes:
                        held[h] -= 1
        assert kinds == {"BlockStored", "BlockRemoved"}
        first = decoder.decode(live[0][2]).events[0]
        assert first.parent_block_hash is None
        assert first.token_ids[:16] == prompt[:16]
        assert all(count >= 0 for count in held.values())

        req = context.socket(zmq.REQ)
        req.setsockopt(zmq.RCVTIMEO, 2000)
        req.connect(replay_endpoint)
        req.send((len(live) - 1).to_bytes(8, "big"))
        topic, seq, payload = req.recv_multipart()
        assert (topic, seq, payload) == tuple(live[-1])
        req.close(linger=0)

        dealer = context.socket(zmq.DEALER)
        dealer.setsockopt(zmq.RCVTIMEO, 2000)
        dealer.connect(replay_endpoint)
        dealer.send_multipart([b"", (0).to_bytes(8, "big")])
        replayed = []
        while True:
            parts = dealer.recv_multipart()
            assert parts[0] == b""
            if parts[2] == END_SEQ:
                break
            replayed.append(parts[1:])
        dealer.close(linger=0)
        assert replayed == live
    finally:
        sub.close(linger=0)
        context.term()
