# Contributing to EasyICU

Thanks for helping improve EasyICU. This repository supports both research-facing features and reusable open-source infrastructure, so we aim to keep changes reviewable and reproducible.

## Local setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -e ".[dev,webapp]"
```

## Before opening a pull request

1. Run the task-scope guard described below and retain its `scope_sha256` in
   the pull-request evidence.
2. Run `pytest -q` for the default FastAPI/core gate.
3. Update `README.md` / `README_zh.md` if user-facing behavior changed.
4. Keep AI-related changes explicitly advisory and human-confirmed in both code and docs.
5. Prefer small, reviewable patches over broad unrelated refactors.

## Concurrent work and commit scope

Concurrent agents or developer sessions must use one linked worktree per task.
Record the full starting HEAD before editing, stage only the task-owned files,
then run the repository guard immediately before every commit:

```bash
python tools/verify_git_task_scope.py \
  --base-head <40-character-starting-head> \
  --allow path/to/owned_file.py \
  --allow tests/test_owned_contract.py
```

The command fails closed when HEAD moved, the task is running in the shared
primary worktree, no changes are staged, an unmerged path exists, or any staged,
unstaged, or untracked path falls outside the exact allowlist. Paste the passing
JSON receipt or at least its `scope_sha256` into the pull request.

A developer working alone in a dedicated clone may pass
`--allow-primary-worktree`, but must explain that exception in the pull request.
This exception is not valid for a workspace shared by concurrent agents.

## Pull request guidance

- Explain the user-facing motivation for the change.
- Call out any database-specific assumptions or limitations.
- Request an independent reviewer before merge and require the repository's
  checks to pass on the exact pull-request HEAD.
- The maintained Web UI is the native FastAPI app. The legacy Streamlit package was removed from the active package boundary; recover it from git history only for archive forensics.
- Mention any follow-up work that remains intentionally out of scope.
