#!/usr/bin/env bash
# Sealed entry point for the seal-tip bucket rebuild; see seal_rebuild.py for the contract and step order.
#
#   seal_rebuild.sh SEAL_SHA MODE [options]      (the seal SHA always comes first)
#
# The tool that runs is never "the file next to this script": seal_rebuild.py, footer_scan.py and
# bucket_compare.py are taken from the seal SHA's own git blobs (regular files only; a link is refused)
# in the mirror (SEAL_MIRROR, default the server mirror), written into a fresh read-only directory and
# executed from there with -I -S (no PYTHONPATH, no user site, no site hooks, no script-dir imports) by
# the server's EasyICU venv unless SEAL_PYTHON names another interpreter. The tool then re-checks its own
# bytes against the seal SHA (tool_not_sealed) and records them in every receipt. The extraction is
# removed afterwards; the exit code is passed through: 0 every selected DB passed, 1 any typed failure,
# 2 usage. Only `seal_rebuild.sh` (no arguments) and `--help` run without a seal SHA: usage, nothing else.
set -euo pipefail
python=${SEAL_PYTHON:-/home/zhuhb/easyicu_dev9/venv311/bin/python}
mirror=${SEAL_MIRROR:-/home/zhuhb/easyicu_dev9/codex_ws/_mirror.git}
if [ "$#" -eq 0 ] || { [ "$#" -eq 1 ] && { [ "$1" = "-h" ] || [ "$1" = "--help" ]; }; }; then
  echo "usage: seal_rebuild.sh SEAL_SHA build1|build2|compare|accept [options] (full SHA first)"
  echo "Use SEAL_SHA --help for the sealed tool's full help."
  if [ "$#" -eq 0 ]; then exit 2; fi
  exit 0
fi
case "$1" in
  -*) echo "usage: seal_rebuild.sh SEAL_SHA MODE [options] (the seal SHA comes first)" >&2; exit 2 ;;
esac
if [[ ! "$1" =~ ^[0-9a-f]{40}$ ]]; then
  echo "all FAILED seal_sha_unknown $1 (full SHA required)"
  exit 1
fi
gitd=$mirror
if [ -d "$mirror/.git" ]; then gitd=$mirror/.git; fi
if ! full=$(git --git-dir "$gitd" rev-parse --verify --quiet "$1^{commit}"); then
  echo "all FAILED seal_sha_unknown $1"
  exit 1
fi
launch=$(mktemp -d "${SEAL_LAUNCH_ROOT:-${TMPDIR:-/tmp}}/seal_launch.XXXXXX")
cleanup() { chmod -R u+w "${launch:?}"; rm -rf "${launch:?}"; }
trap cleanup EXIT
for name in seal_rebuild.py footer_scan.py bucket_compare.py; do
  entry=$(git --git-dir "$gitd" ls-tree "$full" -- "tools/seal_buckets/$name")
  mode=${entry%% *}; rest=${entry#* }; type=${rest%% *}; obj=${rest#* }; obj=${obj%%[[:space:]]*}
  if [ "${type:-}" != blob ] || { [ "${mode:-}" != 100644 ] && [ "${mode:-}" != 100755 ]; }; then
    echo "all FAILED tool_not_sealed tools/seal_buckets/$name"
    exit 1
  fi
  git --git-dir "$gitd" cat-file blob "$obj" >"$launch/$name"
  chmod 0444 "$launch/$name"
done
chmod 0555 "$launch"
rc=0
"$python" -I -S "$launch/seal_rebuild.py" "$@" || rc=$?
exit "$rc"
