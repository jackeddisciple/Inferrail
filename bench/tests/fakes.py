"""Tiny HTTP/1.1 servers with exactly known timing, for client tests."""

import asyncio
import json
from collections.abc import Awaitable, Callable

Handler = Callable[[asyncio.StreamWriter], Awaitable[None]]


async def _read_request(reader: asyncio.StreamReader) -> bytes:
    head = await reader.readuntil(b"\r\n\r\n")
    length = 0
    for line in head.split(b"\r\n"):
        if line.lower().startswith(b"content-length:"):
            length = int(line.split(b":")[1])
    return await reader.readexactly(length)


async def serve(handler: Handler, gate: asyncio.Lock | None = None):
    async def on_connection(reader, writer):
        try:
            while True:
                try:
                    await _read_request(reader)
                except (asyncio.IncompleteReadError, ConnectionError):
                    return
                if gate is None:
                    await handler(writer)
                else:
                    async with gate:
                        await handler(writer)
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


def sse(first: float, gap: float, tokens: int, replica: str = "r0") -> Handler:
    """Streams `tokens` content chunks, the first after `first` seconds and
    the rest `gap` apart, then usage and [DONE]."""

    async def handler(writer: asyncio.StreamWriter) -> None:
        writer.write(
            b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\n"
            + f"x-inferrail-replica: {replica}\r\n".encode()
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
            writer.write(_event({"choices": [{"index": 0, "delta": {"content": f" t{i}"}}]}))
            await writer.drain()
        usage = {
            "prompt_tokens": 40,
            "completion_tokens": tokens,
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
