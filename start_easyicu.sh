#!/usr/bin/env bash
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

is_python_39_plus() {
  "$1" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 9) else 1)' >/dev/null 2>&1
}

if command -v python3 >/dev/null 2>&1 && is_python_39_plus python3; then
  PYTHON_BIN="python3"
elif command -v python >/dev/null 2>&1 && is_python_39_plus python; then
  PYTHON_BIN="python"
else
  echo "Python 3.9+ was not found. Please install a compatible Python version and try again."
  exit 1
fi

"$PYTHON_BIN" "$SCRIPT_DIR/scripts/launch_easyicu.py" start --force-reinstall "$@"
STATUS=$?

if [ "$STATUS" -ne 0 ]; then
  echo
  echo "EasyICU launcher failed. Press Enter to close."
  read -r _
fi

exit "$STATUS"
