"""
LLM client for Ollama's OpenAI-compatible endpoints (standard library only).

Settings are read from environment variables on every call; see README.md and
.env.example.

Usage:
    from llm.client import ask, ask_stream, embed

    # Single response
    result = ask("Explain this SAML assertion", context=xml_string)

    # Streaming response (for Flask SSE endpoints)
    for chunk in ask_stream("Analyze this claim rule", context=rule_text):
        yield chunk

    # Embedding (for RAG / semantic search)
    vector = embed("Okta routing rule for group finance-users")
"""

import json
import logging
import os
import urllib.error
import urllib.request
from typing import Generator, Optional

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_TIMEOUT = 300.0
DEFAULT_HEALTH_TIMEOUT = 5.0

DEFAULT_MODELS = {
    "reason": "qwen2.5:72b",
    "fast": "qwen2.5-coder:32b",
    "embed": "nomic-embed-text",
}

# Environment variable that overrides each role's model name.
MODEL_ENV_VARS = {
    "reason": "LLM_MODEL_REASON",
    "fast": "LLM_MODEL_FAST",
    "embed": "LLM_MODEL_EMBED",
}

SYSTEM_PROMPT = """You are an expert Identity and Access Management analyst with deep knowledge of:
- Active Directory (AD) and ADFS claim rules, relying party trusts, federation metadata
- Okta: SAML/OIDC app integrations, routing rules, sign-on policies, attribute mappings
- Duo MFA: policies, enrollment, bypass, auth logs
- SAML 2.0 and OIDC protocol mechanics
- Security posture analysis for SSO/MFA configurations

Be precise and concise. When analyzing configuration data, flag anomalies, misconfigurations,
and security concerns directly. Use IAM terminology accurately."""


def base_url() -> str:
    """Ollama base URL from OLLAMA_BASE_URL (no trailing slash)."""
    return os.environ.get("OLLAMA_BASE_URL", DEFAULT_BASE_URL).rstrip("/")


def models() -> dict:
    """Role -> model name map, with LLM_MODEL_* overrides applied."""
    return {
        role: os.environ.get(MODEL_ENV_VARS[role], default)
        for role, default in DEFAULT_MODELS.items()
    }


def default_system_prompt() -> str:
    """System prompt from LLM_SYSTEM_PROMPT, else the built-in IAM persona."""
    return os.environ.get("LLM_SYSTEM_PROMPT", SYSTEM_PROMPT)


def _float_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError as e:
        raise ValueError(f"{name} must be a number, got {raw!r}") from e


def _headers() -> dict:
    headers = {"Content-Type": "application/json"}
    key = os.environ.get("LLM_API_KEY")
    if key:
        headers["Authorization"] = f"Bearer {key}"
    return headers


# ── Core request helpers ──────────────────────────────────────────────────────


def _request(path: str, payload: dict) -> urllib.request.Request:
    return urllib.request.Request(
        f"{base_url()}{path}",
        data=json.dumps(payload).encode(),
        headers=_headers(),
        method="POST",
    )


def _http_error(e: urllib.error.HTTPError) -> RuntimeError:
    """Report an HTTP error with its status and response body."""
    try:
        body = e.read().decode(errors="replace")
    except OSError:
        body = ""
    return RuntimeError(f"Ollama returned HTTP {e.code} from {base_url()}: {body}")


def _unreachable(e: urllib.error.URLError) -> RuntimeError:
    return RuntimeError(
        f"Ollama unreachable at {base_url()} — is the service running? ({e})"
    )


def _post(path: str, payload: dict) -> dict:
    """Raw HTTP POST to Ollama — no external dependencies required."""
    req = _request(path, payload)
    timeout = _float_env("LLM_TIMEOUT", DEFAULT_TIMEOUT)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as e:
        raise _http_error(e) from e
    except urllib.error.URLError as e:
        raise _unreachable(e) from e
    try:
        return json.loads(raw)
    except ValueError as e:
        raise RuntimeError(f"Ollama returned invalid JSON: {raw[:200]!r}") from e


def _post_stream(path: str, payload: dict) -> Generator[str, None, None]:
    """Streaming POST — yields content chunks as they arrive."""
    req = _request(path, payload)
    timeout = _float_env("LLM_TIMEOUT", DEFAULT_TIMEOUT)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw_line in resp:
                line = raw_line.decode().strip()
                if not line or line == "data: [DONE]":
                    continue
                if line.startswith("data: "):
                    line = line[6:]
                try:
                    chunk = json.loads(line)
                except ValueError as e:
                    raise RuntimeError(
                        f"Ollama returned a malformed stream line: {line[:200]!r}"
                    ) from e
                choices = chunk.get("choices") or [{}]
                delta = choices[0].get("delta", {}).get("content", "")
                if delta:
                    yield delta
    except urllib.error.HTTPError as e:
        raise _http_error(e) from e
    except urllib.error.URLError as e:
        raise _unreachable(e) from e


def _chat_payload(
    prompt: str,
    context: Optional[str],
    model: str,
    system: Optional[str],
    stream: bool,
) -> dict:
    full_prompt = f"{prompt}\n\n{context}".strip() if context else prompt
    if system is None:
        system = default_system_prompt()
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": full_prompt})
    return {
        "model": models().get(model, model),
        "messages": messages,
        "stream": stream,
    }


# ── Public API ────────────────────────────────────────────────────────────────


def ask(
    prompt: str,
    context: Optional[str] = None,
    model: str = "reason",
    system: Optional[str] = None,
) -> str:
    """
    Send a prompt to the local LLM and return the full response.

    Args:
        prompt:  The question or instruction.
        context: Raw data to analyze (SAML XML, JSON, log lines, etc.).
                 Appended to the prompt automatically.
        model:   A role ("reason", "fast") or a literal model name.
        system:  System prompt. None uses the default; "" sends no system
                 message.

    Returns:
        The model's response as a string.
    """
    payload = _chat_payload(prompt, context, model, system, stream=False)
    result = _post("/v1/chat/completions", payload)
    try:
        return result["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as e:
        raise RuntimeError(f"Unexpected chat response shape: {result!r}") from e


def ask_stream(
    prompt: str,
    context: Optional[str] = None,
    model: str = "reason",
    system: Optional[str] = None,
) -> Generator[str, None, None]:
    """
    Streaming version of ask(). Yields content chunks as they arrive.
    Use this for Flask SSE endpoints so the UI updates in real time.

    Example (Flask):
        @app.route("/analyze")
        def analyze():
            def generate():
                for chunk in ask_stream("Analyze this SAML assertion", context=xml):
                    yield f"data: {chunk}\\n\\n"
            return Response(generate(), mimetype="text/event-stream")
    """
    payload = _chat_payload(prompt, context, model, system, stream=True)
    yield from _post_stream("/v1/chat/completions", payload)


def embed(text: str) -> list:
    """
    Generate an embedding vector for the given text.
    Used for semantic search / RAG over identity configs and logs.

    Args:
        text: The text to embed (config snippet, log line, policy description, etc.)

    Returns:
        A list of floats representing the embedding vector.
    """
    payload = {"model": models()["embed"], "input": text}
    result = _post("/v1/embeddings", payload)
    try:
        return result["data"][0]["embedding"]
    except (KeyError, IndexError, TypeError) as e:
        raise RuntimeError(f"Unexpected embeddings response shape: {result!r}") from e


def health() -> dict:
    """Check Ollama service and confirm expected models are available."""
    expected = models()
    try:
        req = urllib.request.Request(f"{base_url()}/api/tags", headers=_headers())
        timeout = _float_env("LLM_HEALTH_TIMEOUT", DEFAULT_HEALTH_TIMEOUT)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
        available = {m["name"] for m in data.get("models", [])}
    except urllib.error.HTTPError as e:
        logger.warning("health check got HTTP %s", e.code)
        return {"status": "error", "detail": f"HTTP {e.code}: {e.reason}"}
    except (urllib.error.URLError, OSError, ValueError, KeyError, TypeError) as e:
        logger.warning("health check failed: %s", e)
        return {"status": "error", "detail": str(e)}
    return {
        "status": "ok",
        "available_models": sorted(available),
        "expected": dict(expected),
        "missing": [v for v in expected.values() if not any(v in a for a in available)],
    }
