# GPT review adjudication and Git task-scope guard

Date: 2026-08-15
Owner: repository contribution and commit-scope governance
Clean branch: `codex/git-task-scope-guard-20260815` from `origin/main@4ab68685a77bdc26ac5fa89226411782865d5685`

## Review adjudication

The new external review mixed source findings from an older snapshot with a
current pull-request process finding.

- At `fix/pi-workspace-review-20260809@84d572d`, the cited SOFA-2 raw-input,
  FiO2, GCS/motor/physiology, delirium-category, aggregate-key, P/F-ratio,
  documentation, and GitHub Action pinning defects already have focused
  fail-closed contracts. Those changes are not duplicated in this branch.
- The process criticism remains valid. The GitHub pull-request API reported PR
  #7 open at `84d572d`, with 270 commits, 879 changed files, no requested
  reviewer/team, and merge state `unstable` at the time of review.
- Historical commit `697f42b` explicitly records that two sessions shared one
  index and that staged content landed in the wrong commits. This is a
  reproducibility defect even when the resulting source later passes tests.

Therefore no further change is appended to PR #7. This branch addresses only
the missing local guard that allowed cross-session index contamination.

## Owner and public contract

`tools/verify_git_task_scope.py` is the single owner. It depends only on Python
stdlib and Git and exposes `evaluate_task_scope(...)` plus a CLI JSON receipt.
The receipt binds:

- the full starting and current commit SHA;
- whether the task uses a linked worktree;
- the exact allowed file paths;
- all staged, unstaged, untracked, and unmerged paths;
- stable reason codes and a canonical `scope_sha256`.

The default is fail closed. Stable failure codes are:

- `task_scope_head_mismatch`
- `task_scope_linked_worktree_required`
- `task_scope_no_staged_changes`
- `task_scope_unmerged_paths`
- `task_scope_unexpected_paths`
- `task_scope_invocation_invalid`

The sole-user primary-worktree exception is explicit (`--allow-primary-worktree`)
and is forbidden by `CONTRIBUTING.md` for concurrent-agent workspaces.

## Tests-first evidence

Before implementation, the new test module failed collection because
`tools/verify_git_task_scope.py` did not exist. After implementation:

```text
python -m pytest -q tests/test_verify_git_task_scope.py
8 passed

python -m pytest -q tests/test_verify_git_task_scope.py tests/test_repository_contract.py
27 passed

ruff check tools/verify_git_task_scope.py tests/test_verify_git_task_scope.py tests/test_repository_contract.py
All checks passed!

git diff --check
passed
```

The guard verified its own first commit with no unexpected worktree state:

```text
schema_version: easyicu-git-task-scope-v1
base_head: 4ab68685a77bdc26ac5fa89226411782865d5685
staged_paths: 5 exact allowed files
unstaged_paths: []
untracked_paths: []
unexpected_paths: []
scope_sha256: 5a399573e4ed517b97da224165ccecb8b3eeb395b6025fd3ea7e182a6dcf0c0f
```

## Verification boundary and remaining work

- No clinical scoring, data extraction, Research Agent, or Web behavior changed.
- This is focused local verification, not full-repository CI and not a real
  GitHub Actions run.
- PR #7 has not been closed, converted to draft, rewritten, or split by this
  task. The guard prevents recurrence; it cannot repair the existing history.
- A clean split of the clinical, Research Agent, Web/Pi, and governance changes
  from `origin/main` still requires explicit curation and independent reviewers.
- GitHub branch protection, required-review rules, CODEOWNERS, and exact-head
  required checks remain platform-admin work and must not be inferred from the
  repository template.
