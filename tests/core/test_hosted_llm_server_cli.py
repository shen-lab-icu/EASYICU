"""CLI entry point of the hosted LLM relay: --help and missing-key behaviour."""

from __future__ import annotations

import pytest

from easyicu import hosted_llm_server as srv


def test_help_works_without_openrouter_key(monkeypatch, capsys):
    monkeypatch.setattr(srv, "OPENROUTER_API_KEY", "")
    with pytest.raises(SystemExit) as exc:
        srv.main(["--help"])
    assert exc.value.code == 0
    assert "usage" in capsys.readouterr().out.lower()


def test_start_without_key_fails_with_one_line_and_never_serves(monkeypatch, capsys):
    monkeypatch.setattr(srv, "OPENROUTER_API_KEY", "")
    served = []
    monkeypatch.setattr(
        "uvicorn.run", lambda *a, **k: served.append((a, k)), raising=True
    )
    code = srv.main([])
    captured = capsys.readouterr()
    assert code != 0
    assert not served
    lines = [line for line in captured.err.splitlines() if line.strip()]
    assert len(lines) == 1
    assert "OPENROUTER_API_KEY" in lines[0]
    assert "Traceback" not in captured.err + captured.out


def test_start_with_key_reaches_server_without_printing_key(monkeypatch, capsys):
    monkeypatch.setattr(srv, "OPENROUTER_API_KEY", "sk-test-secret")
    monkeypatch.setattr(srv, "HOSTED_ALLOW_UNAUTHENTICATED_LOCAL", True)
    served = []
    monkeypatch.setattr("uvicorn.run", lambda *a, **k: served.append(k), raising=True)
    assert srv.main(["--host", "127.0.0.1", "--port", "9"]) == 0
    assert served
    out = capsys.readouterr()
    assert "sk-test-secret" not in out.out + out.err
