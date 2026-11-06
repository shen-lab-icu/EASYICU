"""A failed reseal leaves the old state, or a journal that restores it (D2-RESEAL).

Codex delta2: the tool replaced the LOCK, profiles.py and client_trust.py one
by one and its rollback stopped at the first restore error. Every --apply
here runs on a copy of ``src``; the real tree is only read.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
FILES = (
    "src/easyicu/data/concept-dict.LOCK.json",
    "src/easyicu/research_agent/orchestration/profiles.py",
    "src/easyicu/research_agent/providers/client_trust.py",
)
APPLY = dict(consent="test consent", label="test reseal", locked_at="2026-11-07T12:00:00Z",
             source_commit="0" * 40)


def _tool():
    spec = importlib.util.spec_from_file_location("reseal_concept_dict", ROOT / "tools/reseal_concept_dict.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _digests(root):
    return {rel: hashlib.sha256((root / rel).read_bytes()).hexdigest() for rel in FILES}


@pytest.fixture
def root(tmp_path):
    shutil.copytree(ROOT / "src", tmp_path / "src",
                    ignore=shutil.ignore_patterns("__pycache__", "static", "*.pyc"))
    path = tmp_path / "src/easyicu/data/concept-dict.json"
    data = json.loads(path.read_text())
    data["_comment_reseal_test"] = "changed by the test"
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    return tmp_path


def _fail_on_nth_replace(monkeypatch, tool, n):
    real = tool._write_atomic
    seen = []

    def flaky(path, raw):
        if path.name != tool.RECOVERY_JOURNAL:
            seen.append(path.name)
            if len(seen) == n:
                raise OSError(28, "No space left on device (injected)")
        return real(path, raw)

    monkeypatch.setattr(tool, "_write_atomic", flaky)
    return seen


@pytest.mark.parametrize("n", [1, 2, 3])
def test_a_failure_mid_way_restores_the_full_old_state(root, monkeypatch, n):
    tool = _tool()
    before = _digests(root)
    _fail_on_nth_replace(monkeypatch, tool, n)
    with pytest.raises(tool.ResealError) as caught:
        tool.apply(root, **APPLY)
    assert caught.value.reason == "reseal_rolled_back"
    assert _digests(root) == before
    assert not (root / tool.RECOVERY_JOURNAL).exists() and not (root / tool.RESEAL_LOCK).exists()


def test_disagreeing_pins_after_writing_restore_every_file(root, monkeypatch):
    tool = _tool()
    before = _digests(root)
    monkeypatch.setattr(tool, "verify_pins", lambda _root: {"current_profiles": 0, "problems": ["injected"]})
    with pytest.raises(tool.ResealError, match="reseal_pins_disagree"):
        tool.apply(root, **APPLY)
    assert _digests(root) == before


def test_every_restore_failure_is_reported_and_the_journal_recovers(root, monkeypatch):
    tool = _tool()
    before = _digests(root)
    real = tool._write_atomic
    state = {"writes": 0}

    def flaky(path, raw):
        if path.name == tool.RECOVERY_JOURNAL:
            return real(path, raw)
        state["writes"] += 1
        if state["writes"] <= 3:
            return real(path, raw)  # all three files are replaced ...
        raise PermissionError(13, "read-only (injected)")  # ... and no restore succeeds

    monkeypatch.setattr(tool, "_write_atomic", flaky)
    monkeypatch.setattr(tool, "verify_pins", lambda _root: {"current_profiles": 0, "problems": ["injected"]})
    with pytest.raises(tool.ResealError) as caught:
        tool.apply(root, **APPLY)
    assert caught.value.reason == "reseal_rollback_incomplete"
    # Every file was attempted and every failure is listed, not just the first.
    assert sorted(f.split(":")[0] for f in caught.value.failures) == sorted(FILES)
    # The journal marks the unfinished reseal; the lifecycle lock was released.
    assert (root / tool.RECOVERY_JOURNAL).exists() and not (root / tool.RESEAL_LOCK).exists()
    with pytest.raises(tool.ResealError) as blocked:
        tool.apply(root, **APPLY)
    assert blocked.value.reason == "reseal_recovery_pending"

    monkeypatch.setattr(tool, "_write_atomic", real)
    assert sorted(tool.recover(root)["restored"]) == sorted(FILES)
    assert _digests(root) == before
    assert not (root / tool.RECOVERY_JOURNAL).exists() and not (root / tool.RESEAL_LOCK).exists()


def test_a_crash_between_files_is_recovered_from_the_journal(root, monkeypatch):
    tool = _tool()
    before = _digests(root)
    real = tool._write_atomic
    written = []

    class Crash(BaseException):
        pass

    def crash_after_two(path, raw):
        real(path, raw)
        if path.name != tool.RECOVERY_JOURNAL:
            written.append(path.name)
            if len(written) == 2:
                raise Crash()

    monkeypatch.setattr(tool, "_write_atomic", crash_after_two)
    # The process dies: no rollback runs.
    monkeypatch.setattr(tool, "_restore", lambda *_: (_ for _ in ()).throw(Crash()))
    with pytest.raises(Crash):
        tool.apply(root, **APPLY)
    assert _digests(root) != before  # half-written
    monkeypatch.undo()
    tool = _tool()
    assert tool.main(["--root", str(root), "--recover"]) == 0
    assert _digests(root) == before


def test_a_held_lock_refuses_a_second_reseal(root):
    # D4-RESEAL-RACE replaced the marker file with an OS lock; a lock file
    # alone no longer refuses (tests/governance/test_d4_reseal_lifecycle.py).
    tool = _tool()
    before = _digests(root)
    with tool._Lifecycle(root):
        with pytest.raises(tool.ResealError) as caught:
            tool.apply(root, **APPLY)
    assert caught.value.reason == "reseal_active"
    assert _digests(root) == before


def test_a_successful_reseal_leaves_no_lock_or_journal(root):
    tool = _tool()
    result = tool.apply(root, **APPLY)
    assert result["verify"]["problems"] == []
    assert not (root / tool.RECOVERY_JOURNAL).exists() and not (root / tool.RESEAL_LOCK).exists()
