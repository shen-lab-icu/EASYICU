#!/usr/bin/env bash
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MIN_PYTHON_CODE=310

python_version_code() {
  "$1" -c 'import sys; print(sys.version_info.major * 100 + sys.version_info.minor)' 2>/dev/null
}

select_python_bin() {
  best_bin=""
  best_code=0
  for candidate in python3.13 python3.12 python3.11 python3.10 python3 python; do
    if ! command -v "$candidate" >/dev/null 2>&1; then
      continue
    fi
    version_code="$(python_version_code "$candidate" || true)"
    if [ -z "$version_code" ]; then
      continue
    fi
    if [ "$version_code" -ge "$MIN_PYTHON_CODE" ] && [ "$version_code" -gt "$best_code" ]; then
      best_bin="$candidate"
      best_code="$version_code"
    fi
  done
  printf '%s' "$best_bin"
}

PYTHON_BIN="$(select_python_bin)"

if [ -z "$PYTHON_BIN" ]; then
  echo "Python 3.10+ was not found. Please install a compatible Python version and try again."
  exit 1
fi

"$PYTHON_BIN" "$SCRIPT_DIR/scripts/launch_easyicu.py" start "$@"
STATUS=$?

if [ "$STATUS" -ne 0 ]; then
  echo
  echo "EasyICU launcher failed. Press Enter to close."
  read -r _
fi

exit "$STATUS"
