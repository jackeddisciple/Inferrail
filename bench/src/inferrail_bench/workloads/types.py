from dataclasses import dataclass, field


@dataclass(frozen=True)
class Message:
    role: str
    content: str

    def to_json(self) -> dict:
        return {"role": self.role, "content": self.content}


@dataclass(frozen=True)
class Request:
    messages: tuple[Message, ...]
    max_tokens: int
    session: str | None = None
    turn: int = 0
    tags: dict = field(default_factory=dict)

    def body(self, model: str, stream: bool = True) -> dict:
        body = {
            "model": model,
            "messages": [m.to_json() for m in self.messages],
            "max_tokens": self.max_tokens,
        }
        if stream:
            body["stream"] = True
            body["stream_options"] = {"include_usage": True}
        return body
