from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL = REPO_ROOT / "tools" / "verify_git_task_scope.py"
SPEC = importlib.util.spec_from_file_location("verify_git_task_scope", TOOL)
assert SPEC and SPEC.loader
scope_guard = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = scope_guard
SPEC.loader.exec_module(scope_guard)


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _init_repo(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.name", "Scope Guard Test")
    _git(repo, "config", "user.email", "scope-guard@example.invalid")
    (repo / "base.txt").write_text("base\n", encoding="utf-8")
    _git(repo, "add", "base.txt")
    _git(repo, "commit", "-m", "base")
    return repo, _git(repo, "rev-parse", "HEAD")


def _add_linked_worktree(repo: Path, tmp_path: Path) -> Path:
    worktree = tmp_path / "task-worktree"
    _git(repo, "worktree", "add", "-b", "task-scope-test", str(worktree))
    return worktree


def test_exact_linked_worktree_scope_passes_and_binds_receipt(tmp_path: Path) -> None:
    repo, base_head = _init_repo(tmp_path)
    worktree = _add_linked_worktree(repo, tmp_path)
    intended = worktree / "intended.txt"
    intended.write_text("staged\n", encoding="utf-8")
    _git(worktree, "add", "intended.txt")
    intended.write_text("staged\nallowed unstaged\n", encoding="utf-8")
    (worktree / "allowed-note.txt").write_text("allowed untracked\n", encoding="utf-8")

    receipt = scope_guard.evaluate_task_scope(
        worktree,
        base_head=base_head,
        allowed_paths=("intended.txt", "allowed-note.txt"),
    )

    assert receipt["status"] == "pass"
    assert receipt["reason_codes"] == []
    assert receipt["linked_worktree"] is True
    assert receipt["staged_paths"] == ["intended.txt"]
    assert receipt["unstaged_paths"] == ["intended.txt"]
    assert receipt["untracked_paths"] == ["allowed-note.txt"]
    assert receipt["unexpected_paths"] == []
    assert len(receipt["scope_sha256"]) == 64


def test_primary_worktree_fails_closed_by_default(tmp_path: Path) -> None:
    repo, base_head = _init_repo(tmp_path)
    (repo / "intended.txt").write_text("staged\n", encoding="utf-8")
    _git(repo, "add", "intended.txt")

    receipt = scope_guard.evaluate_task_scope(
        repo,
        base_head=base_head,
        allowed_paths=("intended.txt",),
    )

    assert receipt["status"] == "fail"
    assert "task_scope_linked_worktree_required" in receipt["reason_codes"]


@pytest.mark.parametrize("change_kind", ["staged", "unstaged", "untracked"])
def test_unexpected_path_fails_in_every_worktree_state(
    tmp_path: Path,
    change_kind: str,
) -> None:
    repo, base_head = _init_repo(tmp_path)
    worktree = _add_linked_worktree(repo, tmp_path)
    (worktree / "intended.txt").write_text("staged\n", encoding="utf-8")
    _git(worktree, "add", "intended.txt")

    unexpected = worktree / "unexpected.txt"
    if change_kind == "staged":
        unexpected.write_text("staged\n", encoding="utf-8")
        _git(worktree, "add", "unexpected.txt")
    elif change_kind == "unstaged":
        unexpected.write_text("tracked\n", encoding="utf-8")
        _git(worktree, "add", "unexpected.txt")
        _git(worktree, "commit", "-m", "track unexpected fixture")
        base_head = _git(worktree, "rev-parse", "HEAD")
        unexpected.write_text("tracked\nunexpected unstaged\n", encoding="utf-8")
        (worktree / "intended.txt").write_text("staged again\n", encoding="utf-8")
        _git(worktree, "add", "intended.txt")
    else:
        unexpected.write_text("unexpected untracked\n", encoding="utf-8")

    receipt = scope_guard.evaluate_task_scope(
        worktree,
        base_head=base_head,
        allowed_paths=("intended.txt",),
    )

    assert receipt["status"] == "fail"
    assert "task_scope_unexpected_paths" in receipt["reason_codes"]
    assert receipt["unexpected_paths"] == ["unexpected.txt"]


def test_head_drift_and_empty_index_are_reported_together(tmp_path: Path) -> None:
    repo, base_head = _init_repo(tmp_path)
    worktree = _add_linked_worktree(repo, tmp_path)
    (worktree / "later.txt").write_text("later\n", encoding="utf-8")
    _git(worktree, "add", "later.txt")
    _git(worktree, "commit", "-m", "move head")

    receipt = scope_guard.evaluate_task_scope(
        worktree,
        base_head=base_head,
        allowed_paths=("later.txt",),
    )

    assert receipt["status"] == "fail"
    assert receipt["reason_codes"] == [
        "task_scope_head_mismatch",
        "task_scope_no_staged_changes",
    ]


def test_cli_failure_is_machine_readable(tmp_path: Path) -> None:
    repo, base_head = _init_repo(tmp_path)
    (repo / "intended.txt").write_text("staged\n", encoding="utf-8")
    _git(repo, "add", "intended.txt")

    result = subprocess.run(
        [
            sys.executable,
            str(TOOL),
            "--repo-root",
            str(repo),
            "--base-head",
            base_head,
            "--allow",
            "intended.txt",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    receipt = json.loads(result.stdout)
    assert receipt["status"] == "fail"
    assert receipt["reason_codes"] == ["task_scope_linked_worktree_required"]


def test_contribution_surfaces_require_scope_receipt_and_independent_review() -> None:
    contributing = (REPO_ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")
    gitignore = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    template = (REPO_ROOT / ".github" / "pull_request_template.md").read_text(
        encoding="utf-8"
    )

    assert "verify_git_task_scope.py" in contributing
    assert "linked worktree" in contributing
    assert "!.github/pull_request_template.md" in gitignore
    assert "scope_sha256" in template
    assert "independent reviewer" in template
    assert "exact PR head" in template
