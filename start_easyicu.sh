#!/usr/bin/env bash
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
else
  echo "Python 3.9+ was not found. Please install Python and try again."
  exit 1
fi

"$PYTHON_BIN" "$SCRIPT_DIR/launch_easyicu.py" start --force-reinstall "$@"
STATUS=$?

if [ "$STATUS" -ne 0 ]; then
  echo
  echo "EasyICU launcher failed. Press Enter to close."
  read -r _
fi

exit "$STATUS"
