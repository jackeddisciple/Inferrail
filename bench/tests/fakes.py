"""Tiny HTTP/1.1 servers with exactly known timing, for client tests."""

import asyncio
import json
from collections.abc import Awaitable, Callable

Handler = Callable[[asyncio.StreamWriter], Awaitable[None]]


async def _read_request(reader: asyncio.StreamReader) -> str:
    head = await reader.readuntil(b"\r\n\r\n")
    length = 0
    for line in head.split(b"\r\n"):
        if line.lower().startswith(b"content-length:"):
            length = int(line.split(b":")[1])
    await reader.readexactly(length)
    return head.split(b" ")[1].decode()


async def serve(
    handler: Handler, gate: asyncio.Lock | None = None, paths: dict[str, Handler] | None = None
):
    """Answers every request with `handler`, or with `paths[path]` when the
    path is listed."""

    async def on_connection(reader, writer):
        try:
            while True:
                try:
                    path = await _read_request(reader)
                except (asyncio.IncompleteReadError, ConnectionError):
                    return
                chosen = (paths or {}).get(path, handler)
                if gate is None:
                    await chosen(writer)
                else:
                    async with gate:
                        await chosen(writer)
        finally:
            writer.close()

    server = await asyncio.start_server(on_connection, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    return server, f"http://127.0.0.1:{port}"


def _chunk(data: bytes) -> bytes:
    return f"{len(data):x}\r\n".encode() + data + b"\r\n"


def _event(payload: dict | str) -> bytes:
    text = payload if isinstance(payload, str) else json.dumps(payload)
    return _chunk(f"data: {text}\n\n".encode())


def sse(
    first: float,
    gap: float,
    tokens: int,
    replica: str = "r0",
    per_chunk: int = 1,
    predicted: tuple[int, int] | None = None,
) -> Handler:
    """Streams `tokens` content chunks of `per_chunk` tokens each, the first
    after `first` seconds and the rest `gap` apart, then usage and [DONE].
    `predicted` adds the gateway's prompt and cached token headers."""

    headers = ""
    if predicted is not None:
        headers = (
            f"x-inferrail-prompt-tokens: {predicted[0]}\r\n"
            f"x-inferrail-cached-tokens: {predicted[1]}\r\n"
        )

    async def handler(writer: asyncio.StreamWriter) -> None:
        writer.write(
            b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\n"
            + f"x-inferrail-replica: {replica}\r\n{headers}".encode()
            + b"transfer-encoding: chunked\r\n\r\n"
        )
        await writer.drain()
        await asyncio.sleep(first)
        writer.write(
            _event({"choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}}]})
        )
        for i in range(tokens):
            if i:
                await asyncio.sleep(gap)
            text = "".join(f" t{i}.{k}" for k in range(per_chunk))
            writer.write(_event({"choices": [{"index": 0, "delta": {"content": text}}]}))
            await writer.drain()
        usage = {
            "prompt_tokens": 40,
            "completion_tokens": tokens * per_chunk,
            "prompt_tokens_details": {"cached_tokens": 32},
        }
        writer.write(_event({"choices": [], "usage": usage}))
        writer.write(_event("[DONE]"))
        writer.write(b"0\r\n\r\n")
        await writer.drain()

    return handler


def status(code: int, body: str) -> Handler:
    async def handler(writer: asyncio.StreamWriter) -> None:
        data = body.encode()
        writer.write(
            f"HTTP/1.1 {code} Error\r\ncontent-type: application/json\r\n"
            f"content-length: {len(data)}\r\n\r\n".encode()
            + data
        )
        await writer.drain()

    return handler


class Background:
    """A fake server on its own event loop thread, for code that runs its
    own loop, such as the CLI."""

    def __init__(self, handler: Handler):
        import threading

        self._loop = asyncio.new_event_loop()
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._run, args=(handler,), daemon=True)
        self._thread.start()
        self._ready.wait(5)

    def _run(self, handler: Handler) -> None:
        asyncio.set_event_loop(self._loop)
        self.server, self.url = self._loop.run_until_complete(serve(handler))
        self._ready.set()
        self._loop.run_forever()

    def close(self) -> None:
        self._loop.call_soon_threadsafe(self.server.close)
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(5)
