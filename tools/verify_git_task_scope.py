#!/usr/bin/env python3
"""Verify that one commit comes from an isolated, explicitly bounded task.

The guard is intentionally independent of Git hooks. Run it after staging and
before committing so its JSON receipt binds the starting HEAD, worktree kind,
and every staged, unstaged, and untracked path visible to the task.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
from typing import Iterable


SCHEMA_VERSION = "easyicu-git-task-scope-v1"
FULL_SHA_RE = re.compile(r"^[0-9a-fA-F]{40}$")
UNMERGED_CODES = frozenset({"DD", "AU", "UD", "UA", "DU", "AA", "UU"})


class TaskScopeInvocationError(RuntimeError):
    """The guard could not establish a trustworthy repository state."""


def _git_bytes(repo_root: Path, *args: str) -> bytes:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), *args],
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = ""
        if isinstance(exc, subprocess.CalledProcessError) and exc.stderr:
            detail = os.fsdecode(exc.stderr).strip()
        suffix = f": {detail}" if detail else ""
        raise TaskScopeInvocationError(
            f"git {' '.join(args)} failed{suffix}"
        ) from exc
    return result.stdout


def _git_text(repo_root: Path, *args: str) -> str:
    return os.fsdecode(_git_bytes(repo_root, *args)).strip()


def _normalize_allowed_path(raw: str) -> str:
    candidate = raw.strip().replace("\\", "/")
    while candidate.startswith("./"):
        candidate = candidate[2:]
    path = PurePosixPath(candidate)
    if (
        not candidate
        or path.is_absolute()
        or candidate.endswith("/")
        or ".." in path.parts
        or path.as_posix() in {"", "."}
    ):
        raise TaskScopeInvocationError(
            f"allowed path must be one exact repository-relative file: {raw!r}"
        )
    return path.as_posix()


def _absolute_git_path(repo_root: Path, raw: str) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def _parse_status(
    payload: bytes,
) -> tuple[set[str], set[str], set[str], set[str]]:
    staged: set[str] = set()
    unstaged: set[str] = set()
    untracked: set[str] = set()
    unmerged: set[str] = set()
    entries = payload.split(b"\0")
    index = 0

    while index < len(entries) and entries[index]:
        entry = entries[index]
        if len(entry) < 4 or entry[2:3] != b" ":
            raise TaskScopeInvocationError("git status returned an invalid porcelain row")
        try:
            code = entry[:2].decode("ascii")
        except UnicodeDecodeError as exc:
            raise TaskScopeInvocationError(
                "git status returned a non-ASCII state code"
            ) from exc

        paths = [os.fsdecode(entry[3:])]
        if "R" in code or "C" in code:
            index += 1
            if index >= len(entries) or not entries[index]:
                raise TaskScopeInvocationError(
                    "git status omitted a rename/copy source path"
                )
            paths.append(os.fsdecode(entries[index]))

        if code == "??":
            untracked.update(paths)
        else:
            if code[0] not in {" ", "?"}:
                staged.update(paths)
            if code[1] not in {" ", "?"}:
                unstaged.update(paths)
            if code in UNMERGED_CODES:
                unmerged.update(paths)
        index += 1

    return staged, unstaged, untracked, unmerged


def _scope_digest(receipt: dict[str, object]) -> str:
    bound = {
        key: value
        for key, value in receipt.items()
        if key not in {"repository_root", "scope_sha256"}
    }
    encoded = json.dumps(
        bound,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8", errors="surrogatepass")
    return hashlib.sha256(encoded).hexdigest()


def evaluate_task_scope(
    repo_root: Path,
    *,
    base_head: str,
    allowed_paths: Iterable[str],
    require_linked_worktree: bool = True,
) -> dict[str, object]:
    """Return a deterministic pass/fail receipt for the current Git task scope."""

    if not FULL_SHA_RE.fullmatch(base_head):
        raise TaskScopeInvocationError("base HEAD must be one full 40-character SHA")
    requested_root = repo_root.resolve()
    discovered_root = Path(
        _git_text(requested_root, "rev-parse", "--show-toplevel")
    ).resolve()
    resolved_base = _git_text(
        discovered_root,
        "rev-parse",
        "--verify",
        f"{base_head}^{{commit}}",
    )
    head = _git_text(discovered_root, "rev-parse", "--verify", "HEAD^{commit}")

    normalized_allowed = tuple(
        sorted({_normalize_allowed_path(path) for path in allowed_paths})
    )
    if not normalized_allowed:
        raise TaskScopeInvocationError("at least one exact allowed path is required")

    git_dir = _absolute_git_path(
        discovered_root,
        _git_text(discovered_root, "rev-parse", "--git-dir"),
    )
    common_dir = _absolute_git_path(
        discovered_root,
        _git_text(discovered_root, "rev-parse", "--git-common-dir"),
    )
    linked_worktree = git_dir != common_dir

    status = _git_bytes(
        discovered_root,
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=all",
    )
    staged, unstaged, untracked, unmerged = _parse_status(status)
    changed = staged | unstaged | untracked | unmerged
    unexpected = changed - set(normalized_allowed)

    reason_codes: list[str] = []
    if head != resolved_base:
        reason_codes.append("task_scope_head_mismatch")
    if require_linked_worktree and not linked_worktree:
        reason_codes.append("task_scope_linked_worktree_required")
    if not staged:
        reason_codes.append("task_scope_no_staged_changes")
    if unmerged:
        reason_codes.append("task_scope_unmerged_paths")
    if unexpected:
        reason_codes.append("task_scope_unexpected_paths")

    receipt: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "status": "fail" if reason_codes else "pass",
        "reason_codes": reason_codes,
        "repository_root": str(discovered_root),
        "base_head": resolved_base,
        "head": head,
        "linked_worktree": linked_worktree,
        "linked_worktree_required": require_linked_worktree,
        "allowed_paths": list(normalized_allowed),
        "staged_paths": sorted(staged),
        "unstaged_paths": sorted(unstaged),
        "untracked_paths": sorted(untracked),
        "unmerged_paths": sorted(unmerged),
        "unexpected_paths": sorted(unexpected),
    }
    receipt["scope_sha256"] = _scope_digest(receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--base-head",
        required=True,
        help="Full 40-character HEAD recorded before the task began.",
    )
    parser.add_argument(
        "--allow",
        action="append",
        required=True,
        dest="allowed_paths",
        help="Exact repository-relative path owned by this task; repeat as needed.",
    )
    parser.add_argument(
        "--allow-primary-worktree",
        action="store_true",
        help="Explicit sole-user-clone exception; linked worktrees are the safe default.",
    )
    args = parser.parse_args()

    try:
        receipt = evaluate_task_scope(
            args.repo_root,
            base_head=args.base_head,
            allowed_paths=args.allowed_paths,
            require_linked_worktree=not args.allow_primary_worktree,
        )
    except TaskScopeInvocationError as exc:
        receipt = {
            "schema_version": SCHEMA_VERSION,
            "status": "fail",
            "reason_codes": ["task_scope_invocation_invalid"],
            "error": str(exc),
        }
        print(json.dumps(receipt, ensure_ascii=True, sort_keys=True))
        return 2

    print(json.dumps(receipt, ensure_ascii=True, sort_keys=True))
    return 0 if receipt["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
