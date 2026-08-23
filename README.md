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

`model` is a role, not a model name: `"reason"` → `qwen2.5:72b` for deep
analysis, `"fast"` → `qwen2.5-coder:32b` for quick queries and config parsing.
Edit `MODELS` in `llm/client.py` to change the mapping.

`system` defaults to an IAM-analyst persona (ADFS claim rules, Okta
policies, SAML/OIDC mechanics). Pass your own `system=` string to replace it,
or `system=""` for none.

## Usage

```python
from llm.client import ask, ask_stream, embed, health

print(health())                                   # {"status": "ok", "missing": []}

answer = ask("What does this claim rule do?", context=rule_text)

for chunk in ask_stream("Summarize the anomalies", context=scan_json):
    print(chunk, end="", flush=True)

vector = embed("Okta routing rule for group finance-users")
```

The module lives at `llm/client.py`; import it from the repository root (or
copy the file into your tool — it has no third-party dependencies).

## Requirements

- Python 3.9+ (uses only the standard library)
- Ollama running on `localhost:11434` with the models you map:

```bash
ollama pull qwen2.5:72b
ollama pull qwen2.5-coder:32b
ollama pull nomic-embed-text
ollama serve
```

The 72B model at Q4 needs roughly 47 GB of memory to stay resident.

## Behaviour worth knowing

- **Timeouts are fixed:** 300 s for completions, 5 s for `health()`.
- **Errors are raised as `RuntimeError`** with the Ollama base URL in the
  message. Today an HTTP error from Ollama (404 unknown model, 400 bad
  request) is reported with the same "unreachable" wording as a connection
  failure — see *Known limitations*.
- **Streaming** parses Ollama's `data: {...}` / `data: [DONE]` SSE lines and
  yields only the content deltas.

## Known limitations

- `OLLAMA_BASE` is a constant; an environment-variable override is the next
  change.
- `HTTPError` is caught as `URLError`, so server-side errors are mislabeled
  as "unreachable" and Ollama's error body is discarded.
- No structured-output (`format=json` / schema) path yet.
- Not packaged for `pip install`; no tests yet.

## License

MIT.

## Author

Wes Glockzin
