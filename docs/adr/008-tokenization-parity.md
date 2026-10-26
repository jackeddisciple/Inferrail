# ADR 008: Tokenize at the edge, exactly as the engine does

Status: accepted

## Context

A cache aware router predicts which replica holds a prompt's prefix by
hashing the prompt's tokens. If the router's token ids differ from the
engine's by a single token, every later block hash differs and no
prediction ever matches. Tokenization therefore has to be exact, not
approximate.

What an engine tokenizes is not the request body. vLLM v0.31.0 validates
the messages with pydantic, rebuilds each message from the fields it reads,
joins or wraps text parts depending on the template, parses tool call
arguments into objects, dumps tools from its own schema, merges template
keywords from the request and the server, renders the chat template with
transformers and tokenizes the result. transformers 5 itself does not use
`tokenizer.json` as written for every model: classes such as
`LlamaTokenizer` rebuild the pipeline from the vocabulary, which changes how
leading spaces tokenize.

## Decision

* `inferrail-tokenize` reproduces that chain in Rust, pinned to vLLM
  v0.31.0 and transformers 5.17.0, the newest transformers that vLLM
  version accepts:
  * chat templates run on minijinja with transformers' environment:
    `trim_blocks`, `lstrip_blocks`, loop controls, Python string and dict
    methods, `tojson` as `json.dumps`, `raise_exception`, `strftime_now`,
    and values printed the way Python prints them;
  * requests are prepared as vLLM prepares them, including the content
    format detection vLLM runs on the template's syntax tree, developer
    role handling, `continue_final_message`, `reasoning_effort` and the
    filtering of template keywords;
  * what a template sees of the request, tool call arguments included, is
    read with the rules of Python's `json.loads`: integers keep every
    digit, `-0` is the integer zero, floats out of range are infinities,
    and `NaN` and `Infinity` are accepted;
  * tokenizer files are loaded as transformers loads them: generic classes
    as written, and `LlamaTokenizer`, `Qwen2Tokenizer` and `GPT2Tokenizer`
    rebuilt the way their constructors rebuild them. Other classes load as
    written and are logged as unverified.
* Parity is a test, not a claim. `tests/fixtures/tokenizers` holds the
  tokenizer files of three families with Apache 2.0 licenses (Qwen3,
  Granite 3.3 and Hermes 2 Pro on Mistral) at pinned revisions, and
  `make_goldens.py` computes the expected ids by running vLLM's request
  handling, ported, through the real transformers and pydantic. Every
  request vLLM accepts must produce identical ids; requests it rejects
  carry no expectation, since nothing is ever cached for them.
* Mistral repositories are tokenized by `mistral_common` in vLLM's default
  mode, not by the Hugging Face template, so they are not a parity family.
* Tokenizers come from a model directory, a repository already in the local
  Hugging Face cache, or `builtin:sim`, the simulator's own tokenizer.
  Nothing is downloaded at startup.
* A tokenizer that cannot be loaded does not stop the gateway. Its model is
  routed without cache predictions, startup logs a warning and the config
  check prints it. A request the tokenizer cannot follow (images, a chat
  template in the request, a batch of prompts) is routed the same way.
* Repeated prompt pieces are tokenized once. Tokenizers split text at added
  tokens before anything else and handle each piece on its own, so a piece
  seen before always yields the same ids. Pipelines where a piece's ids
  depend on its position are never cached.
* Short bodies are tokenized on the request's task; longer ones on a bounded
  pool of threads. A request that finds the pool full, or waits longer than
  `tokenizer.max_wait`, is routed without a prediction rather than delayed.

## Consequences

* 648 golden requests, about 194,000 tokens, match transformers exactly,
  cold and warm, with and without the cache, and a stress run of 9,621 more
  matches too.
* Cold tokenization is bounded by the Hugging Face pre tokenizer at about
  1.7 million tokens per second per core: 4.5 ms for an 8,000 token chat.
  With a shared system prompt or an earlier turn already seen, the same
  request takes 0.1 ms.
* Requests nested deeper than 128 JSON levels are refused by the JSON
  parser, which Python would accept; they get no prediction. Neither do
  requests holding integers wider than 128 bits or lone surrogates.
* Supporting another tokenizer class means porting its constructor and
  adding a family to the goldens.
