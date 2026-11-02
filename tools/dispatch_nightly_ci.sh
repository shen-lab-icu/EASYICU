#!/bin/bash
# Dispatch the full CI checkpoint (ci.yml, reason "nightly") for a branch's
# current head, once per new commit. GitHub schedules only fire on the default
# branch, so a long-lived development branch is dispatched from a workstation
# (launchd/cron). A commit pushed between the lookup and the dispatch makes the
# checkpoint fail closed on its exact-SHA check; the next run picks it up.
#
# Usage: tools/dispatch_nightly_ci.sh [branch]   (needs an authenticated gh)
set -euo pipefail

REPO="${EASYICU_CI_REPO:-shen-lab-icu/EASYICU}"
BRANCH="${1:-codex/dev9-expansion-20260917}"
GH="${GH:-$(command -v gh || echo /opt/homebrew/bin/gh)}"

sha="$("$GH" api "repos/$REPO/commits/$BRANCH" --jq .sha)"
if [ "${#sha}" -ne 40 ]; then
  echo "could not resolve $BRANCH head" >&2
  exit 1
fi
if "$GH" run list -R "$REPO" --workflow ci.yml --branch "$BRANCH" --event workflow_dispatch \
    --limit 30 --json headSha --jq '.[].headSha' | grep -qx "$sha"; then
  echo "$(date '+%F %T') $sha already has a full checkpoint run"
  exit 0
fi
"$GH" workflow run ci.yml -R "$REPO" --ref "$BRANCH" \
  -f checkpoint_reason=nightly -f expected_head_sha="$sha"
echo "$(date '+%F %T') dispatched nightly checkpoint for $sha"
