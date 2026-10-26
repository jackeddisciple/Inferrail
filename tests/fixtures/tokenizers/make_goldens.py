# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "huggingface-hub==1.33.0",
#     "jinja2==3.1.6",
#     "openai==3.37.0",
#     "pydantic==2.14.1",
#     "tokenizers==0.23.3",
#     "transformers==5.17.0",
# ]
# ///
"""Rebuild the tokenizer fixtures and the golden token ids they are judged by.

    uv run make_goldens.py

Each family directory gets the tokenizer files of one pinned Hugging Face
revision (tokenizer.json gzipped) and goldens.jsonl.gz: request bodies with
the token ids vLLM v0.31.0 would compute for them. The request handling is
ported from that version's vllm/entrypoints/chat_utils.py,
vllm/renderers/hf.py and the chat completion protocol, and rendering and
tokenization are transformers 5.17.0, the newest release that vLLM version
accepts. strftime_now is pinned to FIXED_NOW so the goldens are stable.
"""

import datetime
import gzip
import hashlib
import inspect
import json
import random
import shutil
import sys
import tempfile
from collections import deque
from pathlib import Path
from typing import Any, Literal

import jinja2
import jinja2.meta
import jinja2.nodes
import jinja2.sandbox
import transformers.utils.chat_template_utils as hf_chat
from huggingface_hub import hf_hub_download
from openai.types.chat import (
    ChatCompletionContentPartRefusalParam,
    ChatCompletionFunctionToolParam,
    ChatCompletionMessageToolCallParam,
)
from openai.types.chat import ChatCompletionContentPartParam as OpenAIContentPartParam
from openai.types.chat import ChatCompletionMessageParam as OpenAIMessageParam
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_serializer
from transformers import AutoTokenizer
from typing_extensions import Required, TypedDict

HERE = Path(__file__).resolve().parent
FIXED_NOW = datetime.datetime(2026, 1, 15, 9, 30, 0)
CHAT_PER_FAMILY = 200
COMPLETIONS_PER_FAMILY = 40

FAMILIES = [
    {
        "name": "qwen3",
        "repo": "Qwen/Qwen3-8B",
        "revision": "b968826d9c46dd6066d109eabc6255188de91218",
        "license": "apache-2.0",
        "files": ["tokenizer.json", "tokenizer_config.json"],
    },
    {
        "name": "granite3",
        "repo": "ibm-granite/granite-3.3-8b-instruct",
        "revision": "51dd4bc2ade4059a6bd87649d68aa11e4fb2529b",
        "license": "apache-2.0",
        "files": ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"],
    },
    {
        "name": "hermes2pro",
        "repo": "NousResearch/Hermes-2-Pro-Mistral-7B",
        "revision": "24dbda51d986efa05201eec7a093ac91fc5519cf",
        "license": "apache-2.0",
        "files": ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"],
    },
]


class FixedDatetime(datetime.datetime):
    @classmethod
    def now(cls, tz=None):
        return FIXED_NOW


hf_chat.datetime = FixedDatetime


class ThinkPartParam(TypedDict, total=False):
    thinking: Required[str]
    closed: bool
    type: Required[Literal["thinking"]]


ContentPartParam = OpenAIContentPartParam | ChatCompletionContentPartRefusalParam | str | ThinkPartParam


class CustomMessageParam(TypedDict, total=False):
    role: Required[str]
    content: str | list[ContentPartParam]
    name: str
    tool_call_id: str | None
    tool_calls: list[ChatCompletionMessageToolCallParam] | None
    reasoning: str | None
    tools: list[ChatCompletionFunctionToolParam] | None
    task: str | None


class FunctionDefinition(BaseModel):
    model_config = ConfigDict(extra="allow")
    name: str
    description: str | None = None
    parameters: dict[str, Any] | None = None
    strict: bool | None = None
    defer_loading: bool | None = None

    @model_serializer(mode="wrap")
    def _serialize(self, handler):
        data = handler(self)
        data = {k: v for k, v in data.items() if k in type(self).model_fields}
        if self.strict is None:
            data.pop("strict", None)
        if self.defer_loading is None:
            data.pop("defer_loading", None)
        return data


class ToolParam(BaseModel):
    model_config = ConfigDict(extra="allow")
    type: Literal["function"] = "function"
    function: FunctionDefinition
    defer_loading: bool | None = None

    @model_serializer(mode="wrap")
    def _serialize(self, handler):
        data = handler(self)
        data = {k: v for k, v in data.items() if k in type(self).model_fields}
        if self.defer_loading is None:
            data.pop("defer_loading", None)
        return data


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    messages: list[OpenAIMessageParam | CustomMessageParam]
    tools: list[ToolParam] | None = None
    add_generation_prompt: bool = True
    continue_final_message: bool = False
    add_special_tokens: bool = False
    documents: list[dict[str, str]] | None = None
    chat_template_kwargs: dict[str, Any] | None = None
    reasoning_effort: (
        Literal["none", "minimal", "low", "medium", "high", "xhigh", "max"] | None
    ) = Field(default=None)


class CompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    prompt: str
    add_special_tokens: bool = True


class Rejected(Exception):
    pass


TEXT_PARTS = {"text": "text", "input_text": "text", "output_text": "text"}


def parse_parts(parts, fmt: str) -> Any:
    wrap = fmt == "openai"
    content = []
    for part in parts:
        if isinstance(part, str):
            content.append({"type": "text", "text": part} if wrap else part)
            continue
        kind = part.get("type")
        if kind in TEXT_PARTS or kind in ("thinking", "refusal"):
            key = TEXT_PARTS.get(kind, kind)
            text = part.get(key)
            if text is None:
                continue
            if wrap:
                out = {"type": "text", "text": text}
                out.update({k: v for k, v in part.items() if k not in KNOWN_PART_FIELDS})
                content.append(out)
            else:
                content.append(text)
            continue
        raise Rejected(f"unsupported part {kind!r}")
    if wrap:
        return content
    return "\n".join(content)


KNOWN_PART_FIELDS = {
    "text",
    "type",
    "image_url",
    "input_audio",
    "file",
    "refusal",
    "audio_url",
    "video_url",
    "image_pil",
    "image_embeds",
    "audio_embeds",
    "video_embeds",
    "data",
    "uuid",
    "thinking",
    "closed",
    "name",
    "prompt_cache_breakpoint",
}


def parse_message(message: dict, fmt: str) -> dict:
    role = message["role"]
    content = message.get("content")
    reasoning = message.get("reasoning")
    if content is None:
        content = []
    elif isinstance(content, str):
        content = [{"type": "text", "text": content}]
    result = {"role": role, "content": parse_parts(content, fmt)}
    if role == "assistant":
        if "tool_calls" in message and message["tool_calls"] is not None:
            result["tool_calls"] = list(message["tool_calls"])
        if reasoning is not None:
            result["reasoning"] = reasoning
            result["reasoning_content"] = reasoning
    elif role == "tool":
        if "tool_call_id" in message:
            result["tool_call_id"] = message["tool_call_id"]
        if isinstance(result["content"], list):
            items = result["content"]
            if any(isinstance(i, dict) and i.get("type") != "text" for i in items):
                pass
            else:
                texts = [
                    i.get("text", "") for i in items if isinstance(i, dict) and i.get("type") == "text"
                ]
                result["content"] = "\n".join(texts) if texts else ""
    if "name" in message and isinstance(message["name"], str):
        result["name"] = message["name"]
    if "task" in message and isinstance(message["task"], str):
        result["task"] = message["task"]
    if role == "developer":
        result["tools"] = message.get("tools", None)
    return result


def postprocess(messages: list[dict]) -> None:
    for message in messages:
        if message["role"] != "assistant" or "tool_calls" not in message:
            continue
        calls = message.get("tool_calls")
        if not isinstance(calls, list):
            continue
        if len(calls) == 0:
            message.pop("tool_calls", None)
            continue
        for item in calls:
            if not isinstance(item, dict):
                raise Rejected("tool call is not an object")
            function = item.get("function")
            if item.get("type", "function") != "function" or not isinstance(function, dict):
                raise Rejected("tool call is not a function")
            if content := function.get("arguments"):
                if isinstance(content, dict):
                    parsed = content
                else:
                    if isinstance(content, str):
                        try:
                            parsed = json.loads(content)
                        except json.JSONDecodeError:
                            parsed = None
                    else:
                        parsed = content
                    if not isinstance(parsed, dict):
                        parsed = {}
                function["arguments"] = parsed
            else:
                function["arguments"] = {}


def is_var(node, name):
    return isinstance(node, jinja2.nodes.Name) and node.ctx == "load" and node.name == name


def is_attr(node, name, key):
    if isinstance(node, jinja2.nodes.Getitem):
        return (
            is_var(node.node, name)
            and isinstance(node.arg, jinja2.nodes.Const)
            and node.arg.value == key
        )
    if isinstance(node, jinja2.nodes.Getattr):
        return is_var(node.node, name) and node.attr == key
    return False


def is_var_or_elems(node, name, key=None):
    if isinstance(node, jinja2.nodes.Filter):
        return node.node is not None and is_var_or_elems(node.node, name, key)
    if isinstance(node, jinja2.nodes.Test):
        return is_var_or_elems(node.node, name, key)
    if isinstance(node, jinja2.nodes.Getitem) and isinstance(node.arg, jinja2.nodes.Slice):
        return is_var_or_elems(node.node, name, key)
    return is_attr(node, name, key) if key else is_var(node, name)


def assigned_from(root, name):
    yield root, name
    related = deque([name])
    while related:
        current = related.popleft()
        for assign in root.find_all(jinja2.nodes.Assign):
            if is_var_or_elems(assign.node, current):
                yield assign, assign.target.name
                if assign.target.name != current:
                    related.append(assign.target.name)


def message_loops(root):
    names = [n for _, n in assigned_from(root, "messages")]
    for loop in root.find_all(jinja2.nodes.For):
        for name in names:
            if is_var_or_elems(loop.iter, name):
                yield loop, loop.target.name
                break


def content_loops(root):
    names = [n for _, n in message_loops(root)]
    by_loop: dict[int, set[str]] = {}
    in_macros: set[int] = set()
    for macro in root.find_all(jinja2.nodes.Macro):
        params = {a.name for a in macro.args}
        content_params: set[str] = set()
        for call in root.find_all(jinja2.nodes.Call):
            if isinstance(call.node, jinja2.nodes.Name) and call.node.name == macro.name:
                for i, arg in enumerate(call.args):
                    if i < len(macro.args) and any(
                        is_var_or_elems(arg, n, "content") for n in names
                    ):
                        content_params.add(macro.args[i].name)
                for kwarg in call.kwargs:
                    if kwarg.key in params and any(
                        is_var_or_elems(kwarg.value, n, "content") for n in names
                    ):
                        content_params.add(kwarg.key)
        for loop in macro.find_all(jinja2.nodes.For):
            in_macros.add(id(loop))
            if content_params:
                by_loop[id(loop)] = content_params
    for loop in root.find_all(jinja2.nodes.For):
        if any(is_var_or_elems(loop.iter, n, "content") for n in names):
            yield loop
            continue
        params = by_loop.get(id(loop))
        if isinstance(loop.iter, jinja2.nodes.Name) and params and loop.iter.name in params:
            yield loop
            continue
        if (
            id(loop) not in in_macros
            and isinstance(loop.iter, jinja2.nodes.Name)
            and loop.iter.name == "content"
        ):
            yield loop


def environment():
    class AssistantTracker(jinja2.ext.Extension):
        tags = {"generation"}

        def parse(self, parser):
            lineno = next(parser.stream).lineno
            body = parser.parse_statements(("name:endgeneration",), drop_needle=True)
            call = self.call_method("_generation_support")
            return jinja2.nodes.CallBlock(call, [], [], body).set_lineno(lineno)

    return jinja2.sandbox.ImmutableSandboxedEnvironment(
        trim_blocks=True,
        lstrip_blocks=True,
        extensions=[AssistantTracker, jinja2.ext.loopcontrols],
    )


def content_format(template: str) -> str:
    try:
        root = environment().parse(template)
    except jinja2.TemplateError:
        return "string"
    return "openai" if next(content_loops(root), None) is not None else "string"


def template_variables(template: str) -> set[str]:
    return jinja2.meta.find_undeclared_variables(environment().parse(template))


def merge(defaults: dict | None, overrides: dict | None) -> dict:
    defaults = defaults or {}
    overrides = overrides or {}
    return defaults | {k: v for k, v in overrides.items() if v is not None and v != "auto"}


def base_params() -> set[str]:
    from transformers.tokenization_python import PythonBackend

    signature = inspect.signature(PythonBackend.apply_chat_template)
    return {
        p.name
        for p in signature.parameters.values()
        if p.kind not in (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL)
    }


def chat_ids(tokenizer, body: dict) -> tuple[str, list[int]]:
    if body.get("continue_final_message") and body.get("add_generation_prompt"):
        raise Rejected("continue_final_message and add_generation_prompt")
    try:
        request = ChatRequest.model_validate(body)
    except ValidationError as err:
        raise Rejected(str(err)) from err
    tools = None if request.tools is None else [t.model_dump() for t in request.tools]
    template = tokenizer.get_chat_template(None, tools=tools)
    fmt = content_format(template)
    try:
        conversation = [parse_message(m, fmt) for m in request.messages]
    except ValidationError as err:
        raise Rejected(str(err)) from err
    postprocess(conversation)
    developer = '"developer"' in template or "'developer'" in template
    if any(m["role"] == "developer" for m in conversation) and not developer:
        conversation = consolidate(
            [
                {**{k: v for k, v in m.items() if k != "tools"}, "role": "system"}
                if m["role"] == "developer"
                else m
                for m in conversation
            ]
        )
    extra = {
        "add_generation_prompt": request.add_generation_prompt,
        "continue_final_message": request.continue_final_message,
        "documents": request.documents,
        "reasoning_effort": request.reasoning_effort,
    }
    user = request.chat_template_kwargs or {}
    if request.reasoning_effort is not None and "enable_thinking" not in user:
        extra["enable_thinking"] = request.reasoning_effort != "none"
    kwargs = merge(merge({}, {"tools": tools, "tokenize": False}), merge(user, extra))
    kwargs = merge(kwargs, {"chat_template": None, "return_dict": False})
    tools_arg = kwargs.pop("tools", None)
    kwargs.pop("tokenize", None)
    kwargs.pop("chat_template", None)
    accept = template_variables(template) | base_params()
    resolved = {k: v for k, v in kwargs.items() if k in accept - {"chat_template", "tokenize"}}
    try:
        text = tokenizer.apply_chat_template(
            conversation=conversation,
            tools=tools_arg,
            chat_template=template,
            tokenize=False,
            **resolved,
        )
    except Exception as err:
        raise Rejected(f"template: {err}") from err
    ids = tokenizer(text, add_special_tokens=request.add_special_tokens)["input_ids"]
    return text, ids


def consolidate(conversation: list[dict]) -> list[dict]:
    contents, rest, needed = [], [], False
    for i, message in enumerate(conversation):
        if message["role"] == "system":
            if i > 0 or contents:
                needed = True
            content = message.get("content", "")
            if isinstance(content, list):
                content = "\n".join(
                    p["text"] if isinstance(p, dict) and "text" in p else p
                    for p in content
                    if (isinstance(p, dict) and "text" in p) or isinstance(p, str)
                )
            if content:
                contents.append(content)
        else:
            rest.append(message)
    if not needed:
        return conversation
    return [{"role": "system", "content": "\n\n".join(contents)}, *rest]


WORDS = (
    "the cache router prefix token replica latency block engine queue decode prefill memory "
    "warm request stream model batch first load hit miss fleet signal budget tail steady quick "
    "bright river stone answer question explain why how when where gateway kernel tensor "
    "attention layer weight shard rank vector embedding scheduler eviction capacity pressure"
).split()
UNICODE = [
    "naïve café",
    "日本語のテキスト",
    "中文分词测试",
    "Ελληνικά",
    "русский текст",
    "עברית",
    "العربية",
    "emoji 😀🚀👩‍💻",
    "é combining",
    "zero​width",
    "tab\tseparated",
    "CRLF\r\nline",
    "  double  spaces  ",
    " nbsp",
    "ﬁ ligature",
    "Ⅻ roman",
    "½ fraction",
    "ß straße",
]
CODE = [
    "def add(a, b):\n    return a + b\n",
    "fn main() {\n    println!(\"hi\");\n}\n",
    '{"key": [1, 2.5, null, true], "nested": {"x": "y"}}',
    "SELECT * FROM t WHERE id = 42;",
    "```python\nprint('fenced')\n```",
    "<div class=\"a\">&amp; html</div>",
]
SPECIALS = {
    "qwen3": ["<|im_start|>", "<|im_end|>", "<|endoftext|>", "<tool_call>", "<think>"],
    "granite3": ["<|start_of_role|>", "<|end_of_role|>", "<|end_of_text|>", "<|tool_call|>"],
    "hermes2pro": ["<|im_start|>", "<|im_end|>", "<s>", "</s>", "<unk>", "<tool_call>"],
}


def sentence(rng: random.Random, low: int = 3, high: int = 30) -> str:
    words = [rng.choice(WORDS) for _ in range(rng.randint(low, high))]
    text = " ".join(words)
    roll = rng.random()
    if roll < 0.08:
        text = rng.choice(UNICODE) + " " + text
    elif roll < 0.14:
        text = text + "\n" + rng.choice(CODE)
    elif roll < 0.18:
        text = text.capitalize() + "?"
    elif roll < 0.21:
        text = " " + text
    elif roll < 0.23:
        text = text + "\n\n"
    return text


def paragraph(rng: random.Random, family: str) -> str:
    parts = [sentence(rng) for _ in range(rng.randint(1, 4))]
    if rng.random() < 0.05:
        parts.insert(rng.randint(0, len(parts)), rng.choice(SPECIALS[family]))
    return rng.choice([" ", "\n", ". ", "\n\n"]).join(parts)


def schema(rng: random.Random, depth: int = 0) -> dict:
    kind = rng.choice(["string", "integer", "number", "boolean", "array", "object", "enum"])
    if depth > 1 and kind in ("array", "object"):
        kind = "string"
    if kind == "enum":
        return {"type": "string", "enum": rng.sample(WORDS, rng.randint(2, 4))}
    out: dict = {"type": kind}
    if rng.random() < 0.6:
        out["description"] = sentence(rng, 2, 8)
    if kind == "array":
        out["items"] = schema(rng, depth + 1)
    if kind == "object":
        props = {f"{rng.choice(WORDS)}_{i}": schema(rng, depth + 1) for i in range(rng.randint(1, 3))}
        out["properties"] = props
        out["required"] = list(props)[: rng.randint(0, len(props))]
    if kind == "number" and rng.random() < 0.5:
        out["minimum"] = rng.choice([0, 0.5, -1.25, 1e-07, 3.0, 1e20])
    return out


def tool(rng: random.Random, index: int) -> dict:
    name = f"{rng.choice(WORDS)}_{rng.choice(WORDS)}_{index}"
    props = {f"{rng.choice(WORDS)}{i}": schema(rng) for i in range(rng.randint(1, 4))}
    function: dict = {"name": name}
    if rng.random() < 0.9:
        function["description"] = sentence(rng, 4, 16)
    if rng.random() < 0.95:
        function["parameters"] = {
            "type": "object",
            "properties": props,
            "required": list(props)[: rng.randint(0, len(props))],
        }
    if rng.random() < 0.1:
        function["strict"] = rng.choice([True, False])
    out = {"type": "function", "function": function}
    if rng.random() < 0.05:
        out["x_extra"] = "dropped"
    return out


def call_arguments(rng: random.Random) -> Any:
    roll = rng.random()
    if roll < 0.75:
        args = {rng.choice(WORDS): rng.choice([sentence(rng, 1, 4), rng.randint(-5, 500), 2.5, True, None, [1, "a"]]) for _ in range(rng.randint(0, 3))}
        return json.dumps(args, ensure_ascii=rng.random() < 0.5)
    if roll < 0.82:
        return "{not json"
    if roll < 0.87:
        return "[1, 2]"
    if roll < 0.92:
        return ""
    return {"already": "a dict", "n": 1}


def tool_call_id(rng: random.Random) -> str:
    return "".join(rng.choice("abcdefghijklmnopqrstuvwxyz0123456789") for _ in range(9))


def assistant_with_calls(rng: random.Random, tools: list[dict]) -> tuple[dict, list[str]]:
    calls, ids = [], []
    for _ in range(rng.randint(1, 3)):
        name = rng.choice(tools)["function"]["name"] if tools else "lookup"
        ident = tool_call_id(rng)
        ids.append(ident)
        call = {"id": ident, "type": "function", "function": {"name": name, "arguments": call_arguments(rng)}}
        if rng.random() < 0.3:
            call = {"function": call["function"], "type": "function", "id": ident, "index": 0}
        calls.append(call)
    content = None if rng.random() < 0.6 else sentence(rng, 3, 12)
    message = {"role": "assistant", "content": content, "tool_calls": calls}
    return message, ids


def maybe_parts(rng: random.Random, text: str) -> Any:
    roll = rng.random()
    if roll < 0.08:
        return [{"type": "text", "text": text}]
    if roll < 0.12:
        half = len(text) // 2
        return [{"type": "text", "text": text[:half]}, {"type": "text", "text": text[half:]}]
    if roll < 0.14:
        return [text, {"type": "input_text", "text": "more"}]
    return text


def chat_body(rng: random.Random, family: str, model: str, index: int) -> dict:
    kind = rng.choices(
        ["single", "system", "multi", "tools", "tool_loop", "parts", "no_gen", "continue",
         "kwargs", "documents", "developer", "reasoning", "invalid", "empty_tools", "long"],
        weights=[14, 10, 24, 8, 14, 5, 4, 4, 5, 3, 2, 3, 2, 1, 1],
    )[0]
    body: dict = {"model": model}
    messages: list[dict] = []
    if kind in ("system", "multi", "tools", "tool_loop", "documents") and rng.random() < 0.7:
        messages.append({"role": "system", "content": maybe_parts(rng, paragraph(rng, family))})
    if kind in ("single", "system", "parts", "kwargs", "documents", "empty_tools"):
        messages.append({"role": "user", "content": maybe_parts(rng, paragraph(rng, family))})
    elif kind in ("multi", "no_gen", "continue", "reasoning", "long", "developer"):
        if kind == "developer":
            messages.append({"role": "developer", "content": sentence(rng)})
        turns = rng.randint(1, 6) if kind != "long" else 12
        for _ in range(turns):
            messages.append({"role": "user", "content": maybe_parts(rng, paragraph(rng, family))})
            answer = paragraph(rng, family)
            if kind == "reasoning" and rng.random() < 0.6:
                if rng.random() < 0.5:
                    answer = f"<think>\n{sentence(rng)}\n</think>\n\n{answer}"
                    messages.append({"role": "assistant", "content": answer})
                else:
                    messages.append({"role": "assistant", "content": answer, "reasoning": sentence(rng)})
            else:
                messages.append({"role": "assistant", "content": maybe_parts(rng, answer)})
        if kind in ("multi", "reasoning", "long", "developer") or rng.random() < 0.5:
            messages.append({"role": "user", "content": paragraph(rng, family)})
    elif kind in ("tools", "tool_loop"):
        tools = [tool(rng, i) for i in range(rng.randint(1, 4))]
        body["tools"] = tools
        messages.append({"role": "user", "content": paragraph(rng, family)})
        if kind == "tool_loop":
            for _ in range(rng.randint(1, 2)):
                message, ids = assistant_with_calls(rng, tools)
                messages.append(message)
                for ident in ids:
                    result = rng.choice(
                        [sentence(rng, 2, 10), json.dumps({"ok": True, "v": rng.randint(0, 99)}), [{"type": "text", "text": sentence(rng, 2, 6)}]]
                    )
                    messages.append({"role": "tool", "tool_call_id": ident, "content": result})
            if rng.random() < 0.5:
                messages.append({"role": "assistant", "content": sentence(rng)})
                messages.append({"role": "user", "content": sentence(rng)})
    elif kind == "invalid":
        messages.append({"role": "user", "content": sentence(rng)})
        choice = rng.random()
        if choice < 0.35:
            messages.append({"role": "assistant", "content": None, "tool_calls": ["not an object"]})
        elif choice < 0.7:
            messages.append({"role": "user", "content": [{"type": "image_url", "image_url": {"url": "https://example.com/a.png"}}]})
        else:
            body["continue_final_message"] = True
            body["add_generation_prompt"] = True
            messages.append({"role": "assistant", "content": "partial"})
    if kind == "empty_tools":
        body["tools"] = []
    if kind == "no_gen":
        body["add_generation_prompt"] = False
    if kind == "continue":
        messages.append({"role": "assistant", "content": rng.choice(["The answer is", "Sure, ", sentence(rng, 2, 6), "  spaced  "])})
        body["add_generation_prompt"] = False
        body["continue_final_message"] = True
    if kind == "kwargs":
        options = {
            "qwen3": [{"enable_thinking": False}, {"enable_thinking": True}, {"unused_flag": 1}],
            "granite3": [{"thinking": True}, {"controls": {"citations": True}}, {"unused_flag": 1}],
            "hermes2pro": [{"unused_flag": 1}, {"bos_token": "<s>"}],
        }[family]
        body["chat_template_kwargs"] = rng.choice(options)
        if rng.random() < 0.4:
            body["reasoning_effort"] = rng.choice(["none", "low", "high"])
    if kind == "documents":
        body["documents"] = [
            {"doc_id": str(i), "title": rng.choice(WORDS), "text": paragraph(rng, family)}
            for i in range(rng.randint(1, 3))
        ]
    if rng.random() < 0.05:
        body["add_special_tokens"] = True
    if rng.random() < 0.1:
        body["stream"] = True
        body["max_tokens"] = rng.randint(1, 256)
    body["messages"] = messages
    return body


def completion_body(rng: random.Random, family: str, model: str) -> dict:
    prompt = paragraph(rng, family)
    if rng.random() < 0.2:
        prompt = " " + prompt
    body: dict = {"model": model, "prompt": prompt}
    if rng.random() < 0.15:
        body["add_special_tokens"] = False
    return body


def completion_ids(tokenizer, body: dict) -> list[int]:
    request = CompletionRequest.model_validate(body)
    return tokenizer(request.prompt, add_special_tokens=request.add_special_tokens)["input_ids"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch(family: dict, out: Path, scratch: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    files = {}
    for name in family["files"]:
        path = Path(hf_hub_download(family["repo"], name, revision=family["revision"]))
        shutil.copyfile(path, scratch / name)
        files[name] = sha256(path)
        if name == "tokenizer.json":
            data = gzip.compress(path.read_bytes(), compresslevel=9, mtime=0)
            (out / "tokenizer.json.gz").write_bytes(data)
        else:
            shutil.copyfile(path, out / name)
    source = {k: family[k] for k in ("repo", "revision", "license")} | {"sha256": files}
    (out / "source.json").write_text(json.dumps(source, indent=2) + "\n")
    return source


def build(family: dict) -> None:
    out = HERE / family["name"]
    with tempfile.TemporaryDirectory() as tmp:
        scratch = Path(tmp)
        fetch(family, out, scratch)
        tokenizer = AutoTokenizer.from_pretrained(str(scratch))
        rng = random.Random(f"inferrail goldens {family['name']}")
        lines, rejected = [], 0
        for i in range(CHAT_PER_FAMILY):
            body = chat_body(rng, family["name"], family["repo"], i)
            try:
                text, ids = chat_ids(tokenizer, body)
                lines.append({"id": i, "endpoint": "chat", "body": body, "text": text, "ids": ids})
            except Rejected as err:
                rejected += 1
                lines.append({"id": i, "endpoint": "chat", "body": body, "error": str(err)[:200]})
        for j in range(COMPLETIONS_PER_FAMILY):
            body = completion_body(rng, family["name"], family["repo"])
            ids = completion_ids(tokenizer, body)
            lines.append({"id": CHAT_PER_FAMILY + j, "endpoint": "completion", "body": body, "ids": ids})
    data = "".join(json.dumps(line, ensure_ascii=False) + "\n" for line in lines).encode()
    (out / "goldens.jsonl.gz").write_bytes(gzip.compress(data, compresslevel=9, mtime=0))
    tokens = sum(len(line.get("ids", [])) for line in lines)
    print(f"{family['name']}: {len(lines)} requests, {rejected} rejected, {tokens} tokens", file=sys.stderr)


def main() -> None:
    for family in FAMILIES:
        build(family)


if __name__ == "__main__":
    main()
