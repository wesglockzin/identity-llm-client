# Identity LLM Client

A small, dependency-free Python client for **local** LLM inference through
[Ollama](https://ollama.com)'s OpenAI-compatible endpoints. Written for
identity tooling where SAML configs, authentication logs and federation
metadata must not leave the host.

## Why it exists

Identity data can't go to a cloud model. Local inference solves that, but
every tool growing its own Ollama integration is the wrong shape. This module
is the one place the calls go through: one request helper, one streaming
parser, one model map, one error path.

It's used by the AI analysis panel in
[adfs-to-okta-migration](https://github.com/wesglockzin/adfs-to-okta-migration).

## What's in it

| Function | What it does |
|---|---|
| `ask(prompt, context=None, model="reason", system=...)` | One-shot chat completion; returns the response text |
| `ask_stream(prompt, context=None, model="reason", system=...)` | Same call, yields text chunks as they arrive (for Flask SSE endpoints) |
| `embed(text)` | Returns an embedding vector from the `LLM_MODEL_EMBED` model |
| `health()` | Checks Ollama is up and which of the expected models are present |

`model` is a role or a literal model name: `"reason"` is for deep analysis and
`"fast"` is for quick queries and config parsing. Each role maps to the model
named in its `LLM_MODEL_*` setting below. Any other string is sent as the
model name unchanged.

`system` defaults to an IAM-analyst persona (ADFS claim rules, Okta
policies, SAML/OIDC mechanics). Pass your own `system=` string to replace it,
or `system=""` to send no system message.

**Type:** library. Install with `pip install -e .`; there is no command to run.

## Settings

All settings are environment variables, read on each call. The server URL and
the model names have no default and must be set; the rest are optional.
They are also listed in `.env.example` (the library does not load `.env` itself).
For a local Ollama the URL is `http://localhost:11434`. Any model you have
pulled works; the examples used so far are `qwen2.5:72b` (about 47 GB of memory),
`qwen2.5-coder:32b` and `nomic-embed-text` (embeddings).

| Variable | Default | Meaning |
|---|---|---|
| `OLLAMA_BASE_URL` | none, required | Server base URL |
| `LLM_MODEL_REASON` | none, required | Model for role `reason` |
| `LLM_MODEL_FAST` | none, required | Model for role `fast` |
| `LLM_MODEL_EMBED` | none, required | Model for `embed()` |
| `LLM_TIMEOUT` | `300` | Seconds for chat and embedding requests |
| `LLM_HEALTH_TIMEOUT` | `5` | Seconds for `health()` |
| `LLM_API_KEY` | unset | Sent as `Authorization: Bearer ...` when set |
| `LLM_SYSTEM_PROMPT` | built-in IAM prompt | Replaces the default system prompt |

## Usage

```python
from llm.client import ask, ask_stream, embed, health

print(health())                                   # {"status": "ok", "missing": []}

answer = ask("What does this claim rule do?", context=rule_text)

for chunk in ask_stream("Summarize the anomalies", context=scan_json):
    print(chunk, end="", flush=True)

vector = embed("Okta routing rule for group finance-users")
```

The module lives at `llm/client.py`. Install it with `pip install -e .`
(add `.[dev]` for the tests and linter); it has no third-party dependencies.
Run the tests with `pytest`; they use a local fake server, not Ollama.

## Requirements

- Python 3.9+ (uses only the standard library)
- Ollama reachable at `OLLAMA_BASE_URL` with the models you named:

```bash
ollama pull <model>     # once per LLM_MODEL_* value
ollama serve
```

## Behaviour worth knowing

- **Errors are raised as `RuntimeError`.** An HTTP error from Ollama (404
  unknown model, 400 bad request) reports the status and response body; a
  connection failure says "unreachable", and a timeout or socket error while
  reading is also a `RuntimeError`. Malformed or unexpected responses are
  `RuntimeError` too. A missing required setting or a non-numeric
  `LLM_TIMEOUT` raises `ValueError` from `ask`, `ask_stream` and `embed`.
- **Error text is for developers.** Messages include the server's response
  body. Do not show them to end users unfiltered.
- **`health()` never raises**, including for a missing or invalid setting
  such as `LLM_HEALTH_TIMEOUT=soon`; it returns
  `{"status": "error", "detail": ...}` and logs a warning.
- **Streaming** parses Ollama's `data: {...}` / `data: [DONE]` SSE lines and
  yields only the content deltas.

## Known limitations

- No structured-output (`format=json` / schema) path yet.
- `health()` uses Ollama's `/api/tags`, so it works only against Ollama.
- Plain `http://` is allowed; use an `https://` base URL for remote servers.
  A warning is logged when `LLM_API_KEY` is sent over `http://`.
- `context` is placed in the prompt unchanged, so untrusted text can carry
  prompt injection.
- Large models need a lot of memory; pick `LLM_MODEL_*` values that fit your hardware.

## License

MIT.

## Author

Wes Glockzin
