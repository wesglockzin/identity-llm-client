import json
import logging
import threading
import urllib.error
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from llm import client

FAKE_KEY = "test-key-not-real"
MODELS = {"reason": "model-r", "fast": "model-f", "embed": "model-e"}


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
                route = outer.routes.get(self.path, (404, b"no route"))
                status, payload = route[0], route[1]
                self.send_response(status)
                for name, value in route[2:]:
                    self.send_header(name, value)
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
        "LLM_API_KEY",
        "LLM_SYSTEM_PROMPT",
        "LLM_TIMEOUT",
        "LLM_HEALTH_TIMEOUT",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(client, "_warned_plain_http", False)
    for role, model in MODELS.items():
        monkeypatch.setenv(client.MODEL_ENV_VARS[role], model)
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
    assert payload["model"] == "model-f"
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
    monkeypatch.setenv("LLM_API_KEY", FAKE_KEY)
    fake.routes["/v1/chat/completions"] = (200, chat_body("ok"))
    client.ask("q")
    _, payload, headers = fake.requests[0]
    assert payload["model"] == "tiny:1b"
    assert payload["messages"][0]["content"] == "be brief"
    assert headers["Authorization"] == f"Bearer {FAKE_KEY}"


def test_http_error_reports_status_and_body(fake):
    fake.routes["/v1/chat/completions"] = (404, b'{"error":"model not found"}')
    with pytest.raises(RuntimeError) as exc:
        client.ask("q")
    assert "404" in str(exc.value)
    assert "model not found" in str(exc.value)
    assert "unreachable" not in str(exc.value)


def test_unreachable_server(fake, monkeypatch):
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
    assert fake.requests[0][1]["model"] == "model-e"
    fake.routes["/v1/embeddings"] = (200, b'{"data": []}')
    with pytest.raises(RuntimeError, match="shape"):
        client.embed("text")


def test_health_ok_and_missing(fake):
    tags = {"models": [{"name": "model-r"}, {"name": "model-e:latest"}]}
    fake.routes["/api/tags"] = (200, json.dumps(tags).encode())
    result = client.health()
    assert result["status"] == "ok"
    assert result["missing"] == ["model-f"]


def test_health_errors_do_not_raise(fake, monkeypatch):
    assert client.health()["status"] == "error"  # 404 from the fake
    fake.routes["/api/tags"] = (200, b"garbage")
    assert client.health()["status"] == "error"
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:1")
    assert client.health()["status"] == "error"


def test_health_http_error_includes_body(fake):
    fake.routes["/api/tags"] = (404, b"tags-body-here")
    result = client.health()
    assert result["status"] == "error"
    assert "404" in result["detail"]
    assert "tags-body-here" in result["detail"]


def test_health_entry_without_name(fake):
    fake.routes["/api/tags"] = (200, b'{"models": [{"size": 1}, {"name": "model-r"}]}')
    result = client.health()
    assert result["status"] == "ok"
    assert "model-r" in result["available_models"]


@pytest.mark.parametrize(
    "body",
    [
        b'{"models": ["a"]}',
        b'["a"]',
        b'{"models": 5}',
        b'{"models": [{"name": 5}]}',
        b'{"models": [{"name": 5}, {"name": "x"}]}',
        b'{"models": [{"name": null}]}',
        b'{"name": null}',
        b"{}",
        b'{"models": {}}',
        b'{"models": ""}',
    ],
)
def test_health_malformed_list_does_not_raise(fake, body):
    fake.routes["/api/tags"] = (200, body)
    assert client.health()["status"] == "error"


def test_health_bad_timeout_does_not_raise(fake, monkeypatch):
    fake.routes["/api/tags"] = (200, b'{"models": []}')
    monkeypatch.setenv("LLM_HEALTH_TIMEOUT", "soon")
    result = client.health()
    assert result["status"] == "error"
    assert "LLM_HEALTH_TIMEOUT" in result["detail"]


def test_missing_base_url_setting(fake, monkeypatch):
    monkeypatch.delenv("OLLAMA_BASE_URL")
    with pytest.raises(ValueError, match="OLLAMA_BASE_URL"):
        client.ask("q")
    assert client.health()["status"] == "error"


def test_missing_model_setting(fake, monkeypatch):
    fake.routes["/v1/chat/completions"] = (200, chat_body("ok"))
    monkeypatch.delenv("LLM_MODEL_REASON")
    with pytest.raises(ValueError, match="LLM_MODEL_REASON"):
        client.ask("q")
    # A literal model name needs no role setting.
    assert client.ask("q", model="literal:1b") == "ok"
    assert client.health()["status"] == "error"


class BrokenBody:
    """Stands in for an HTTP response whose socket fails while being read."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        raise TimeoutError("timed out")

    def __iter__(self):
        raise ConnectionResetError("reset")


def test_unreadable_error_body_is_logged(caplog):
    class Unreadable(urllib.error.HTTPError):
        def read(self, *a, **k):
            raise ConnectionResetError("reset")

    err = Unreadable("http://x", 502, "bad", {}, None)
    with caplog.at_level(logging.WARNING, logger=client.logger.name):
        result = client._http_error(err)
    assert str(result).endswith("(body unreadable)")
    assert "502" in caplog.text
    assert "could not read body of HTTP" in caplog.text
    assert "ConnectionResetError" in caplog.text


def test_socket_errors_while_reading_are_wrapped(fake, monkeypatch):
    monkeypatch.setattr(client, "_open", lambda *a, **k: BrokenBody())
    with pytest.raises(RuntimeError, match="failed"):
        client.ask("q")
    with pytest.raises(RuntimeError, match="failed"):
        list(client.ask_stream("q"))


def test_api_key_over_plain_http_logs_warning(fake, monkeypatch, caplog):
    monkeypatch.setenv("LLM_API_KEY", FAKE_KEY)
    fake.routes["/v1/chat/completions"] = (200, chat_body("ok"))
    with caplog.at_level("WARNING", logger="llm.client"):
        client.ask("q")
        client.ask("q")
    assert caplog.text.count("plain http") == 1


def test_redirect_is_refused_and_key_not_forwarded(fake, monkeypatch):
    other = FakeOllama()
    try:
        monkeypatch.setenv("LLM_API_KEY", FAKE_KEY)
        other.routes["/v1/chat/completions"] = (200, chat_body("leaked"))
        target = other.url + "/v1/chat/completions"
        # 302 on a POST is one urllib would follow (as a GET) without the guard.
        fake.routes["/v1/chat/completions"] = (302, b"", ("Location", target))
        with pytest.raises(RuntimeError, match="302"):
            client.ask("q")
        with pytest.raises(RuntimeError, match="302"):
            list(client.ask_stream("q"))
        assert other.requests == []
    finally:
        other.close()


def test_health_redirect_is_refused(fake, monkeypatch):
    other = FakeOllama()
    try:
        monkeypatch.setenv("LLM_API_KEY", FAKE_KEY)
        other.routes["/api/tags"] = (200, b'{"models": []}')
        target = other.url + "/api/tags"
        fake.routes["/api/tags"] = (302, b"", ("Location", target))
        result = client.health()
        assert result["status"] == "error"
        assert "302" in result["detail"]
        assert other.requests == []
    finally:
        other.close()


def test_empty_system_prompt_setting_means_unset(fake, monkeypatch):
    monkeypatch.setenv("LLM_SYSTEM_PROMPT", "")
    fake.routes["/v1/chat/completions"] = (200, chat_body("ok"))
    client.ask("q")
    _, payload, _ = fake.requests[0]
    assert payload["messages"][0] == {"role": "system", "content": client.SYSTEM_PROMPT}


def test_no_warning_without_key_or_over_https(fake, monkeypatch, caplog):
    fake.routes["/v1/chat/completions"] = (200, chat_body("ok"))
    with caplog.at_level("WARNING", logger="llm.client"):
        client.ask("q")  # no key
        monkeypatch.setenv("LLM_API_KEY", FAKE_KEY)
        monkeypatch.setattr(client, "base_url", lambda: "https://example.invalid")
        client._headers()
    assert "plain http" not in caplog.text
