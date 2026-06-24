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

1. Run `pytest -q` for the default FastAPI/core gate.
2. Update `README.md` / `README_zh.md` if user-facing behavior changed.
3. Keep AI-related changes explicitly advisory and human-confirmed in both code and docs.
4. Prefer small, reviewable patches over broad unrelated refactors.

## Pull request guidance

- Explain the user-facing motivation for the change.
- Call out any database-specific assumptions or limitations.
- The legacy Streamlit UI is deprecated. If archive work must touch it, install `easyicu[webapp-legacy]`, run the affected tests with `pytest --run-legacy-streamlit ...`, and include screenshots only for that explicit legacy scope.
- Mention any follow-up work that remains intentionally out of scope.
