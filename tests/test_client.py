import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from llm import client


class FakeOllama:
    """Local fake server. `routes` maps path -> (status, body bytes)."""

    def __init__(self):
        self.routes = {}
        self.requests = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def _reply(self):
                length = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(length) if length else b""
                outer.requests.append(
                    (self.path, json.loads(body) if body else None, dict(self.headers))
                )
                status, payload = outer.routes.get(self.path, (404, b"no route"))
                self.send_response(status)
                self.end_headers()
                self.wfile.write(payload)

            do_GET = _reply
            do_POST = _reply

            def log_message(self, *args):
                pass

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_port}"

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def fake(monkeypatch):
    for name in (
        "LLM_MODEL_REASON",
        "LLM_MODEL_FAST",
        "LLM_MODEL_EMBED",
        "LLM_API_KEY",
        "LLM_SYSTEM_PROMPT",
        "LLM_TIMEOUT",
        "LLM_HEALTH_TIMEOUT",
    ):
        monkeypatch.delenv(name, raising=False)
    server = FakeOllama()
    monkeypatch.setenv("OLLAMA_BASE_URL", server.url)
    yield server
    server.close()


def chat_body(text):
    return json.dumps({"choices": [{"message": {"content": text}}]}).encode()


def test_ask_returns_content_and_builds_payload(fake):
    fake.routes["/v1/chat/completions"] = (200, chat_body("hi"))
    assert client.ask("question", context="data", model="fast") == "hi"
    _, payload, _ = fake.requests[0]
    assert payload["model"] == "qwen2.5-coder:32b"
    assert payload["stream"] is False
    assert payload["messages"][0] == {"role": "system", "content": client.SYSTEM_PROMPT}
    assert payload["messages"][1]["content"] == "question\n\ndata"


def test_context_none_and_literal_model_name(fake):
    fake.routes["/v1/chat/completions"] = (200, chat_body("ok"))
    client.ask("q", context=None, model="custom:1b")
    _, payload, _ = fake.requests[0]
    assert payload["model"] == "custom:1b"
    assert payload["messages"][1]["content"] == "q"


def test_empty_system_sends_no_system_message(fake):
    fake.routes["/v1/chat/completions"] = (200, chat_body("ok"))
    client.ask("q", system="")
    _, payload, _ = fake.requests[0]
    assert [m["role"] for m in payload["messages"]] == ["user"]


def test_env_overrides(fake, monkeypatch):
    monkeypatch.setenv("LLM_MODEL_REASON", "tiny:1b")
    monkeypatch.setenv("LLM_SYSTEM_PROMPT", "be brief")
    monkeypatch.setenv("LLM_API_KEY", "k")
    fake.routes["/v1/chat/completions"] = (200, chat_body("ok"))
    client.ask("q")
    _, payload, headers = fake.requests[0]
    assert payload["model"] == "tiny:1b"
    assert payload["messages"][0]["content"] == "be brief"
    assert headers["Authorization"] == "Bearer k"


def test_http_error_reports_status_and_body(fake):
    fake.routes["/v1/chat/completions"] = (404, b'{"error":"model not found"}')
    with pytest.raises(RuntimeError) as exc:
        client.ask("q")
    assert "404" in str(exc.value)
    assert "model not found" in str(exc.value)
    assert "unreachable" not in str(exc.value)


def test_unreachable_server(monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:1")
    with pytest.raises(RuntimeError, match="unreachable"):
        client.ask("q")


def test_bad_response_shapes_raise_runtime_error(fake):
    fake.routes["/v1/chat/completions"] = (200, b'{"choices": []}')
    with pytest.raises(RuntimeError, match="shape"):
        client.ask("q")
    fake.routes["/v1/chat/completions"] = (200, b"not json")
    with pytest.raises(RuntimeError, match="invalid JSON"):
        client.ask("q")


def test_invalid_timeout_setting(fake, monkeypatch):
    monkeypatch.setenv("LLM_TIMEOUT", "soon")
    with pytest.raises(ValueError, match="LLM_TIMEOUT"):
        client.ask("q")


def sse(*chunks):
    lines = []
    for c in chunks:
        lines.append("data: " + json.dumps({"choices": [{"delta": {"content": c}}]}))
        lines.append("")
    lines.append("data: [DONE]")
    return "\n".join(lines).encode()


def test_stream_yields_deltas_and_skips_empty(fake):
    body = sse("Hel", "", "lo") + b'\ndata: {"choices": []}\n'
    fake.routes["/v1/chat/completions"] = (200, body)
    assert list(client.ask_stream("q")) == ["Hel", "lo"]
    assert fake.requests[0][1]["stream"] is True


def test_stream_accepts_lines_without_data_prefix(fake):
    line = json.dumps({"choices": [{"delta": {"content": "x"}}]})
    fake.routes["/v1/chat/completions"] = (200, line.encode())
    assert list(client.ask_stream("q")) == ["x"]


def test_stream_malformed_line(fake):
    fake.routes["/v1/chat/completions"] = (200, b"data: {oops\n")
    with pytest.raises(RuntimeError, match="malformed"):
        list(client.ask_stream("q"))


def test_stream_http_error(fake):
    fake.routes["/v1/chat/completions"] = (400, b"bad request body")
    with pytest.raises(RuntimeError, match="400.*bad request body"):
        list(client.ask_stream("q"))


def test_embed(fake):
    fake.routes["/v1/embeddings"] = (200, b'{"data": [{"embedding": [0.5, 1.5]}]}')
    assert client.embed("text") == [0.5, 1.5]
    assert fake.requests[0][1]["model"] == "nomic-embed-text"
    fake.routes["/v1/embeddings"] = (200, b'{"data": []}')
    with pytest.raises(RuntimeError, match="shape"):
        client.embed("text")


def test_health_ok_and_missing(fake):
    tags = {"models": [{"name": "qwen2.5:72b"}, {"name": "nomic-embed-text:latest"}]}
    fake.routes["/api/tags"] = (200, json.dumps(tags).encode())
    result = client.health()
    assert result["status"] == "ok"
    assert result["missing"] == ["qwen2.5-coder:32b"]


def test_health_errors_do_not_raise(fake, monkeypatch):
    assert client.health()["status"] == "error"  # 404 from the fake
    fake.routes["/api/tags"] = (200, b"garbage")
    assert client.health()["status"] == "error"
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:1")
    assert client.health()["status"] == "error"
