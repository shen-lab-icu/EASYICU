from __future__ import annotations

from pathlib import Path

from easyicu import webapp


def test_streamlit_command_skips_flags_not_supported_by_current_version(monkeypatch) -> None:
    monkeypatch.setattr(
        webapp,
        "_supported_streamlit_flags",
        lambda: {
            "--server.fileWatcherType",
            "--server.runOnSave",
            "--browser.gatherUsageStats",
            "--server.enableCORS",
            "--server.enableXsrfProtection",
            "--server.disconnectedSessionTTL",
        },
    )

    cmd = webapp._build_streamlit_run_cmd(
        Path("/tmp/app.py"),
        host="127.0.0.1",
        port=8501,
        debug=False,
    )

    assert "--server.websocketPingInterval" not in cmd
    assert "--server.disconnectedSessionTTL" in cmd
