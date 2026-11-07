#!/usr/bin/env bash
# Entry point for the seal-tip bucket rebuild; see seal_rebuild.py for the contract and step order.
# Runs the orchestrator isolated (-I: no PYTHONPATH, no user site, no script-dir imports) with the
# server's EasyICU venv unless SEAL_PYTHON names another interpreter. Its exit code is passed through:
# 0 every selected DB passed, 1 any typed failure, 2 usage.
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
python=${SEAL_PYTHON:-/home/zhuhb/easyicu_dev9/venv311/bin/python}
exec "$python" -I "$here/seal_rebuild.py" "$@"
