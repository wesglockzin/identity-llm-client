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
| `embed(text)` | Returns an embedding vector (`nomic-embed-text`, 768 floats) |
| `health()` | Checks Ollama is up and which of the expected models are present |

`model` is a role or a literal model name: `"reason"` → `qwen2.5:72b` for deep
analysis, `"fast"` → `qwen2.5-coder:32b` for quick queries and config parsing.
Any other string is sent as the model name unchanged. Change the mapping with
the `LLM_MODEL_*` settings below.

`system` defaults to an IAM-analyst persona (ADFS claim rules, Okta
policies, SAML/OIDC mechanics). Pass your own `system=` string to replace it,
or `system=""` to send no system message.

**Type:** library. Install with `pip install -e .`; there is no command to run.

## Settings

All settings are environment variables, read on each call. All are optional.
They are also listed in `.env.example` (the library does not load `.env` itself).

| Variable | Default | Meaning |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Server base URL |
| `LLM_MODEL_REASON` | `qwen2.5:72b` | Model for role `reason` |
| `LLM_MODEL_FAST` | `qwen2.5-coder:32b` | Model for role `fast` |
| `LLM_MODEL_EMBED` | `nomic-embed-text` | Model for `embed()` |
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
- Ollama reachable at `OLLAMA_BASE_URL` with the models you map:

```bash
ollama pull qwen2.5:72b
ollama pull qwen2.5-coder:32b
ollama pull nomic-embed-text
ollama serve
```

The 72B model at Q4 needs roughly 47 GB of memory to stay resident.

## Behaviour worth knowing

- **Errors are raised as `RuntimeError`.** An HTTP error from Ollama (404
  unknown model, 400 bad request) reports the status and response body; a
  connection failure says "unreachable". Malformed or unexpected responses
  are also `RuntimeError`. A non-numeric timeout setting raises `ValueError`.
- **`health()` never raises**; it returns `{"status": "error", "detail": ...}`.
- **Streaming** parses Ollama's `data: {...}` / `data: [DONE]` SSE lines and
  yields only the content deltas.

## Known limitations

- No structured-output (`format=json` / schema) path yet.
- `health()` uses Ollama's `/api/tags`, so it works only against Ollama.
- Plain `http://` by default; use an `https://` base URL for remote servers.
- `context` is placed in the prompt unchanged, so untrusted text can carry
  prompt injection.
- The default models are large (the 72B model needs about 47 GB); set
  `LLM_MODEL_*` for smaller hardware.

## License

MIT.

## Author

Wes Glockzin
