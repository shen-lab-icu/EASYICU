#!/usr/bin/env bash
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MIN_PYTHON_CODE=310

python_version_code() {
  "$1" -c 'import sys; print(sys.version_info.major * 100 + sys.version_info.minor)' 2>/dev/null
}

append_python_candidate() {
  candidate="$1"
  [ -n "$candidate" ] || return 0
  case ":$PYTHON_CANDIDATES:" in
    *":$candidate:"*) return 0 ;;
  esac
  PYTHON_CANDIDATES="${PYTHON_CANDIDATES:+$PYTHON_CANDIDATES:}$candidate"
}

append_conda_python_candidates() {
  if [ -n "${CONDA_PREFIX:-}" ]; then
    append_python_candidate "$CONDA_PREFIX/bin/python"
  fi

  if command -v conda >/dev/null 2>&1; then
    conda_base="$(conda info --base 2>/dev/null | tail -n 1 || true)"
    if [ -n "$conda_base" ]; then
      append_python_candidate "$conda_base/bin/python"
    fi
  fi

  for conda_root in \
    "$HOME/miniconda3" \
    "$HOME/anaconda3" \
    "$HOME/mambaforge" \
    "$HOME/miniforge3" \
    "/opt/miniconda3" \
    "/opt/anaconda3" \
    "/opt/homebrew/anaconda3" \
    "/opt/homebrew/miniconda3"
  do
    append_python_candidate "$conda_root/bin/python"
  done
}

select_python_bin() {
  best_bin=""
  best_code=0
  PYTHON_CANDIDATES=""
  for candidate in python3.13 python3.12 python3.11 python3.10 python3 python; do
    append_python_candidate "$candidate"
  done
  append_conda_python_candidates

  old_ifs="$IFS"
  IFS=":"
  for candidate in $PYTHON_CANDIDATES; do
    IFS="$old_ifs"
    if ! command -v "$candidate" >/dev/null 2>&1 && [ ! -x "$candidate" ]; then
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
    IFS=":"
  done
  IFS="$old_ifs"
  printf '%s' "$best_bin"
}

PYTHON_BIN="$(select_python_bin)"

if [ -z "$PYTHON_BIN" ]; then
  echo "Python 3.10+ was not found. Checked PATH plus common Conda/Anaconda locations."
  echo "If you use Conda, install Python 3.10+ in base or set CONDA_PREFIX before launching."
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
