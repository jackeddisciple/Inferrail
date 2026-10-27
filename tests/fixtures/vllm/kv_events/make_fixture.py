"""Rebuild batches.jsonl: KV event batches as a vLLM v0.31.0 engine publishes
them over ZMQ, one message per line with its topic, sequence number and the
msgpack payload in hex.

    uv run --with msgspec==0.22.0 python make_fixture.py > batches.jsonl

The structs are copied from vllm/distributed/kv_events.py at v0.31.0, and the
fields filled as vllm/v1/core/block_pool.py fills them: GPU medium, group 0,
block hashes as the low 64 bits of the engine's hash, and the data parallel
rank the publisher stamps on every batch.
"""

import json
from typing import Any

import msgspec

ExternalBlockHash = bytes | int


class EventBatch(msgspec.Struct, array_like=True, omit_defaults=True, gc=False):
    ts: float
    events: list[Any]
    data_parallel_rank: int | None = None


class KVCacheEvent(msgspec.Struct, omit_defaults=True, gc=False, tag=True):
    """Base class for all KV cache-related events."""


class BlockStored(KVCacheEvent):
    block_hashes: list[ExternalBlockHash]
    parent_block_hash: ExternalBlockHash | None
    token_ids: list[int]
    block_size: int
    lora_id: int | None
    medium: str | None
    lora_name: str | None
    extra_keys: list[tuple[Any, ...] | None] | None = None
    group_idx: int | None = None
    kv_cache_spec_kind: str | None = None
    kv_cache_spec_sliding_window: int | None = None
    locality: str | None = None
    ownership: str | None = None
    session_id: str | None = None


class BlockRemoved(KVCacheEvent):
    block_hashes: list[ExternalBlockHash]
    medium: str | None
    group_idx: int | None = None
    locality: str | None = None
    ownership: str | None = None


class AllBlocksCleared(KVCacheEvent):
    pass


class KVEventBatch(EventBatch):
    events: list[BlockStored | BlockRemoved | AllBlocksCleared]


def stored(hashes, parent, tokens):
    return BlockStored(
        block_hashes=hashes,
        parent_block_hash=parent,
        token_ids=tokens,
        block_size=16,
        lora_id=None,
        medium="GPU",
        lora_name=None,
        group_idx=0,
    )


def removed(block_hash):
    return BlockRemoved(block_hashes=[block_hash], medium="GPU", group_idx=0)


BATCHES = [
    KVEventBatch(
        ts=1793000000.25,
        events=[stored([0x1234, 0xFEDC_BA98_7654_3210], None, list(range(32)))],
        data_parallel_rank=0,
    ),
    KVEventBatch(
        ts=1793000000.5,
        events=[
            stored([7], 0xFEDC_BA98_7654_3210, [128_000 + i for i in range(16)]),
            removed(0x1234),
            removed(2**64 - 1),
        ],
        data_parallel_rank=0,
    ),
    KVEventBatch(ts=1793000001.0, events=[AllBlocksCleared()], data_parallel_rank=0),
]


def main() -> None:
    encoder = msgspec.msgpack.Encoder()
    decoder = msgspec.msgpack.Decoder(type=KVEventBatch)
    for seq, batch in enumerate(BATCHES):
        payload = encoder.encode(batch)
        assert decoder.decode(payload) == batch
        print(json.dumps({"topic": "kv-events", "seq": seq, "payload": payload.hex()}))


if __name__ == "__main__":
    main()
