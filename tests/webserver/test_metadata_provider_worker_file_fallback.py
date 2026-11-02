"""The subprocess must not reopen an unreviewed provider environment file."""

import io
import json
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("environment", [{}, {"EASYICU_DISABLE_PROVIDER_ENV_FILE": "0"}, None])
def test_worker_refuses_file_fallback_before_provider_construction(tmp_path, monkeypatch, environment):
    from easyicu.webserver import metadata_provider_worker as worker, provider_adapter

    payload = json.dumps({"environment": environment}).encode()
    stdout = io.BytesIO()
    calls = []
    monkeypatch.setattr(worker.sys, "stdin", SimpleNamespace(buffer=io.BytesIO(payload)))
    monkeypatch.setattr(worker.sys, "stdout", SimpleNamespace(buffer=stdout))
    monkeypatch.setattr(worker, "verify_host_worker", lambda *_args: "expected")
    monkeypatch.setattr(worker, "identity", lambda: "expected")
    monkeypatch.setattr(worker, "complete_host_worker_cleanup", lambda *a, **kw: None)
    monkeypatch.setattr(provider_adapter, "build_research_agent_provider_client",
                        lambda *a, **kw: calls.append("provider"))

    def stop(_seconds):
        raise SystemExit

    monkeypatch.setattr(worker.time, "sleep", stop)
    with pytest.raises(SystemExit):
        worker._worker(tmp_path, "nonce", "expected")
    response = json.loads(stdout.getvalue())
    assert response["ok"] is False and response["type"] == "ValueError"
    assert calls == []


def test_real_guarded_worker_ignores_changed_legacy_file(tmp_path):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    import threading

    from easyicu.webserver.metadata_provider_worker import guarded_call

    seen = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            seen.append(self.headers.get("Authorization") == "Bearer synthetic-environment-key")
            self.rfile.read(int(self.headers["Content-Length"]))
            payload = json.dumps({"id": "fake", "object": "chat.completion", "created": 1,
                "model": "synthetic", "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"},
                "finish_reason": "stop"}], "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3}}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        env_file = tmp_path / "synthetic-provider.env"
        env_file.write_text("OPENAI_API_KEY=synthetic-file-key\n", encoding="utf-8")
        env_file.chmod(0o600)
        environment = {"OPENAI_API_KEY": "synthetic-environment-key",
            "OPENAI_BASE_URL": f"http://127.0.0.1:{server.server_port}/v1",
            "OPENAI_MODEL": "synthetic", "EASYICU_DISABLE_PROVIDER_ENV_FILE": "1",
            "EASYICU_LLM_ENV_FILE": str(env_file), "EASYICU_PROVIDER_DATA_HANDLING": "user_accepted_external"}
        env_file.write_text("OPENAI_API_KEY=changed-file-key\n", encoding="utf-8")
        env_file.chmod(0o000)
        result = guarded_call(tmp_path / "guard", {
            "provider": {"provider": "openai", "external": True}, "environment": environment,
            "provider_identity": {}, "timeout": 10,
            "messages": [{"role": "user", "content": "synthetic"}],
            "max_tokens": 8, "temperature": 0.1,
        }, timeout=10)
        assert result["ok"] is True and result["text"] == "ok"
        assert seen == [True]
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()
