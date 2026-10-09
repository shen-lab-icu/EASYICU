"""The core package keeps its own state in an isolated EASYICU_HOME too.

``easyicu.state_paths`` moves the WebApp's state with ``EASYICU_HOME``, but two
core modules still derived theirs from the real home: ``easyicu.config`` read
(and on import created) ``~/.easyicu/config``, and the cache manager cleared
``~/.easyicu_cache``. A review, test or desktop server under its own home took
on the real home's saved settings and could delete the real home's cache.
"""

from __future__ import annotations

from pathlib import Path
import re

from easyicu import config, state_paths
from easyicu.runtime.cache_manager import CacheManager


PACKAGE = Path(state_paths.__file__).resolve().parent


def _homes(tmp_path, monkeypatch):
    real, isolated = tmp_path / "real", tmp_path / "isolated"
    monkeypatch.setenv("HOME", str(real))
    monkeypatch.setenv("EASYICU_HOME", str(isolated))
    return real, isolated


def test_an_isolated_home_reads_and_deletes_only_its_own_settings(tmp_path, monkeypatch):
    """XDG_CONFIG_HOME comes from the shell profile and names the real home's."""

    real, isolated = _homes(tmp_path, monkeypatch)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(real / ".config"))
    real_files = [
        real / ".easyicu" / "config" / "easyicu.json",
        real / ".config" / "easyicu" / "easyicu.json",
    ]
    for path in real_files:
        path.parent.mkdir(parents=True)
        path.write_text('{"marker": "real"}', encoding="utf-8")
    own = isolated / ".easyicu" / "config" / "easyicu.json"
    own.parent.mkdir(parents=True)
    own.write_text('{"marker": "isolated"}', encoding="utf-8")

    assert config.get_config_dir() == own.parent
    assert config.load_config(merge=False) == {"marker": "isolated"}
    config.delete_config()

    assert not own.exists()
    assert [path.read_text(encoding="utf-8") for path in real_files] == ['{"marker": "real"}'] * 2


def test_without_an_isolated_home_settings_stay_where_they_were(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.delenv("EASYICU_HOME")
    assert config.get_config_dir() == tmp_path / ".easyicu" / "config"

    # A blank EASYICU_HOME is unset here too, so XDG_CONFIG_HOME still applies.
    monkeypatch.setenv("EASYICU_HOME", "   ")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    assert config.get_config_dir() == tmp_path / "xdg" / "easyicu"


def test_an_isolated_home_clears_only_its_own_disk_cache(tmp_path, monkeypatch):
    """get_cache_info lists the directories clear_disk_cache removes."""

    real, isolated = _homes(tmp_path, monkeypatch)
    for home in (real, isolated):
        (home / ".easyicu_cache").mkdir(parents=True)
    monkeypatch.setattr(CacheManager, "_instance", None)

    info = CacheManager().get_cache_info()
    listed = {Path(row["path"]) for row in info["disk_cache_dirs"]}

    assert isolated / ".easyicu_cache" in listed
    assert real / ".easyicu_cache" not in listed


# These look for what the user installed or keeps in their real home, never
# for EasyICU's own state, so EASYICU_HOME must not move them. The webserver
# has its own guard and allowlist (tests/webserver/test_webserver_state_paths.py).
ALLOWED_REAL_HOME = {
    "state_paths.py",  # the owner
    "base.py",  # raw database folders under ~/data
    "research_agent/reporting/pdf_render.py",  # TinyTeX
    "research_agent/execution/docker_locality.py",  # Docker Desktop's CLI
}


def test_no_core_module_derives_its_state_from_path_home() -> None:
    offenders: list[str] = []
    for path in sorted(PACKAGE.rglob("*.py")):
        relative = path.relative_to(PACKAGE).as_posix()
        if relative.startswith("webserver/") or relative in ALLOWED_REAL_HOME:
            continue
        source = path.read_text(encoding="utf-8")
        for match in re.finditer(r"Path\.home\(\)", source):
            line = source[: match.start()].count("\n") + 1
            offenders.append(f"{relative}:{line}")
    assert offenders == [], (
        "these modules derive EasyICU state from the real home; use "
        f"easyicu.state_paths instead: {offenders}"
    )
