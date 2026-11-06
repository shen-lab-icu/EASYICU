"""tools/reseal_concept_dict.py moves the LOCK and every current profile pin together.

The real tree is only read here. Every --apply runs on a copy of ``src``.
"""
import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _tool():
    spec = importlib.util.spec_from_file_location("reseal_concept_dict", ROOT / "tools/reseal_concept_dict.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve their module by name
    spec.loader.exec_module(module)
    return module


def _lock_matches_files(root: Path) -> bool:
    lock = json.loads((root / "src/easyicu/data/concept-dict.LOCK.json").read_text())
    return all(
        lock[field] == hashlib.sha256((root / path).read_bytes()).hexdigest()
        for field, path in (("concept_dict_sha256", "src/easyicu/data/concept-dict.json"),
                            ("sofa2_dict_sha256", "src/easyicu/data/sofa2-dict.json"))
    )


@pytest.fixture
def copy(tmp_path):
    shutil.copytree(
        ROOT / "src", tmp_path / "src",
        ignore=shutil.ignore_patterns("__pycache__", "static", "*.pyc"),
    )
    return tmp_path


def _pending_copy(copy: Path) -> Path:
    """A copy whose concept dictionary differs from its LOCK, as before any reseal."""
    path = copy / "src/easyicu/data/concept-dict.json"
    if _lock_matches_files(copy):
        data = json.loads(path.read_text())
        data["_comment_reseal_test"] = "changed by the test"
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    return copy


def test_dry_run_reports_a_pending_reseal_exactly_when_the_lock_is_stale(capsys):
    tool = _tool()
    assert tool.main(["--root", str(ROOT)]) == 0
    out = capsys.readouterr().out
    assert ("pending: concept_dict_sha256" in out) == (not _lock_matches_files(ROOT))
    assert "dry run: nothing written" in out


def test_apply_moves_the_lock_and_every_current_profile_together(copy):
    tool = _tool()
    root = _pending_copy(copy)
    before = tool.verify_pins(root)
    assert any("LOCK concept_dict_sha256" in p for p in before["problems"])
    result = tool.apply(root, consent="test consent", label="test reseal", locked_at="2026-11-07T12:00:00Z",
                        source_commit="0" * 40)
    assert result["verify"]["problems"] == []
    assert result["verify"]["current_profiles"] >= 13
    assert _lock_matches_files(root)
    lock = json.loads((root / "src/easyicu/data/concept-dict.LOCK.json").read_text())
    assert lock["finalized_foundation_changes"][-1]["git_commit"] == "0" * 40
    assert "test consent" in lock["finalized_foundation_changes"][-1]["change"]
    assert lock["total_concepts"] == len(json.loads((root / "src/easyicu/data/concept-dict.json").read_text()))
    # Nothing left to reseal: a second run refuses.
    with pytest.raises(tool.ResealError, match="already match"):
        tool.apply(root, consent="x", label="x", locked_at="2026-11-08T12:00:00Z", source_commit="0" * 40)


def test_a_single_moved_pin_is_caught(copy):
    tool = _tool()
    root = _pending_copy(copy)
    tool.apply(root, consent="test consent", label="test reseal", locked_at="2026-11-07T12:00:00Z",
               source_commit="0" * 40)
    lock_path = root / "src/easyicu/data/concept-dict.LOCK.json"
    lock = json.loads(lock_path.read_text())
    lock["concept_dict_sha256"] = "f" * 64
    lock_path.write_text(json.dumps(lock, indent=2, ensure_ascii=False) + "\n")
    problems = tool.verify_pins(root)["problems"]
    assert any("pins differ from LOCK" in p for p in problems)
    assert any("LOCK concept_dict_sha256" in p for p in problems)


def test_apply_refuses_without_consent_and_writes_nothing(copy):
    tool = _tool()
    root = _pending_copy(copy)
    paths = [root / p for p in (tool.LOCK, tool.PROFILES, tool.CLIENT_TRUST)]
    before = [p.read_bytes() for p in paths]
    with pytest.raises(tool.ResealError, match="consent"):
        tool.apply(root, consent="  ", label="x", locked_at="2026-11-07T12:00:00Z", source_commit="0" * 40)
    assert [p.read_bytes() for p in paths] == before
