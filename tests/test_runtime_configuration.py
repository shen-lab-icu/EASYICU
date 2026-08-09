"""Process-boundary contracts for explicit EasyICU runtime setup."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest


def test_import_does_not_initialize_or_clear_cache() -> None:
    env = os.environ.copy()
    env["EASYICU_AUTO_CLEAR_CACHE"] = "true"
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import easyicu; "
                "from easyicu.runtime import cache_manager; "
                "assert cache_manager._cache_manager is None"
            ),
        ],
        check=False,
        capture_output=True,
        env=env,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_configure_runtime_initializes_without_clearing_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from easyicu.runtime import cache_manager, project_config
    from easyicu.runtime.configure import configure_runtime

    class StubManager:
        def clear_all_cache(self):
            raise AssertionError("cache must not be cleared")

    monkeypatch.setattr(cache_manager, "get_cache_manager", StubManager)
    monkeypatch.setattr(project_config, "AUTO_CLEAR_CACHE", True)

    status = configure_runtime(
        stdio_encoding=False,
        initialize_cache=True,
        clear_cache=False,
    )

    assert status == {
        "stdio_configured": False,
        "cache_initialized": True,
        "cache_cleared": False,
    }


def test_configure_runtime_can_explicitly_clear_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from easyicu.runtime import cache_manager
    from easyicu.runtime.configure import configure_runtime

    clear_result = {"disk_cache": {}, "memory_cache": {}}

    class StubManager:
        def clear_all_cache(self):
            return clear_result

    monkeypatch.setattr(cache_manager, "get_cache_manager", StubManager)

    status = configure_runtime(
        stdio_encoding=False,
        initialize_cache=True,
        clear_cache=True,
    )

    assert status["cache_initialized"] is True
    assert status["cache_cleared"] is True
    assert status["cache_result"] is clear_result


def test_public_cache_clear_also_clears_global_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import easyicu.api as api
    from easyicu.runtime import cache_manager

    calls: list[str] = []

    class StubManager:
        def clear_all_cache(self):
            calls.append("cache")
            return {"ok": True}

    monkeypatch.setattr(cache_manager, "get_cache_manager", StubManager)
    monkeypatch.setattr(api, "clear_global_loader", lambda: calls.append("loader"))

    result = cache_manager.clear_easyicu_cache()

    assert result == {"ok": True}
    assert calls == ["cache", "loader"]
