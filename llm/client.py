"""
Identity Intelligence Platform — LLM Client
All Ollama calls route through here. Never call Ollama directly from app code.

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
import urllib.request
import urllib.error
from typing import Generator

# ── Config ────────────────────────────────────────────────────────────────────

OLLAMA_BASE = "http://localhost:11434"

MODELS = {
    "reason": "qwen2.5:72b",        # Deep analysis, policy reasoning, complex queries
    "fast":   "qwen2.5-coder:32b",  # Quick queries, code tasks, config parsing
    "embed":  "nomic-embed-text",   # Embeddings for RAG / semantic search
}

SYSTEM_PROMPT = """You are an expert Identity and Access Management analyst with deep knowledge of:
- Active Directory (AD) and ADFS claim rules, relying party trusts, federation metadata
- Okta: SAML/OIDC app integrations, routing rules, sign-on policies, attribute mappings
- Duo MFA: policies, enrollment, bypass, auth logs
- SAML 2.0 and OIDC protocol mechanics
- Security posture analysis for SSO/MFA configurations

Be precise and concise. When analyzing configuration data, flag anomalies, misconfigurations,
and security concerns directly. Use IAM terminology accurately."""


# ── Core request helper ───────────────────────────────────────────────────────

def _post(path: str, payload: dict) -> dict:
    """Raw HTTP POST to Ollama — no external dependencies required."""
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{OLLAMA_BASE}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            return json.loads(resp.read())
    except urllib.error.URLError as e:
        raise RuntimeError(f"Ollama unreachable at {OLLAMA_BASE} — is the service running? ({e})") from e


def _post_stream(path: str, payload: dict) -> Generator[str, None, None]:
    """Streaming POST — yields content chunks as they arrive."""
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{OLLAMA_BASE}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            for raw_line in resp:
                line = raw_line.decode().strip()
                if not line or line == "data: [DONE]":
                    continue
                if line.startswith("data: "):
                    line = line[6:]
                chunk = json.loads(line)
                delta = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                if delta:
                    yield delta
    except urllib.error.URLError as e:
        raise RuntimeError(f"Ollama unreachable at {OLLAMA_BASE} — is the service running? ({e})") from e


# ── Public API ────────────────────────────────────────────────────────────────

def ask(
    prompt: str,
    context: str = "",
    model: str = "reason",
    system: str = SYSTEM_PROMPT,
) -> str:
    """
    Send a prompt to the local LLM and return the full response.

    Args:
        prompt:  The question or instruction.
        context: Raw data to analyze (SAML XML, JSON, log lines, etc.).
                 Appended to the prompt automatically.
        model:   "reason" (72B, default) or "fast" (32B).
        system:  Override the system prompt if needed.

    Returns:
        The model's response as a string.
    """
    full_prompt = f"{prompt}\n\n{context}".strip() if context else prompt
    payload = {
        "model": MODELS.get(model, model),
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": full_prompt},
        ],
        "stream": False,
    }
    result = _post("/v1/chat/completions", payload)
    return result["choices"][0]["message"]["content"]


def ask_stream(
    prompt: str,
    context: str = "",
    model: str = "reason",
    system: str = SYSTEM_PROMPT,
) -> Generator[str, None, None]:
    """
    Streaming version of ask(). Yields content chunks as they arrive.
    Use this for Flask SSE endpoints so the UI updates in real time.

    Example (Flask):
        @app.route("/analyze")
        def analyze():
            def generate():
                for chunk in ask_stream("Analyze this SAML assertion", context=xml):
                    yield f"data: {chunk}\n\n"
            return Response(generate(), mimetype="text/event-stream")
    """
    full_prompt = f"{prompt}\n\n{context}".strip() if context else prompt
    payload = {
        "model": MODELS.get(model, model),
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": full_prompt},
        ],
        "stream": True,
    }
    yield from _post_stream("/v1/chat/completions", payload)


def embed(text: str) -> list[float]:
    """
    Generate an embedding vector for the given text.
    Used for semantic search / RAG over identity configs and logs.

    Args:
        text: The text to embed (config snippet, log line, policy description, etc.)

    Returns:
        A list of floats representing the embedding vector.
    """
    payload = {
        "model": MODELS["embed"],
        "input": text,
    }
    result = _post("/v1/embeddings", payload)
    return result["data"][0]["embedding"]


def health() -> dict:
    """Check Ollama service and confirm expected models are available."""
    try:
        req = urllib.request.Request(f"{OLLAMA_BASE}/api/tags")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
        available = {m["name"] for m in data.get("models", [])}
        return {
            "status": "ok",
            "available_models": sorted(available),
            "expected": {k: v for k, v in MODELS.items()},
            "missing": [v for v in MODELS.values() if not any(v in a for a in available)],
        }
    except Exception as e:
        return {"status": "error", "detail": str(e)}
