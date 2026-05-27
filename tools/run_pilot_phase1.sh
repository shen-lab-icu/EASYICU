#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CASE_NAME="case_b_sofa2_sepsis"
BACKEND="openai"
MODEL="${EASYICU_PILOT_MODEL:-gpt-5.4}"
OUT_ROOT="${REPO_ROOT}/pilot_runs/pilot_phase1/${CASE_NAME}_$(date -u +%Y%m%dT%H%M%SZ)"
ENV_FILE="/tmp/easyicu_local_llm.env"
REQUEST_TIMEOUT="300"
MAX_TOTAL_STEPS=""
DRY_RUN=0
ALLOW_DIRTY=0

usage() {
  cat <<'EOF'
Usage: bash tools/run_pilot_phase1.sh [options]

Options:
  --case NAME              Case directory under benchmark/cases.
  --backend NAME           mock, openai, or openrouter. Default: openai.
  --model NAME             Model name. Default: $EASYICU_PILOT_MODEL or gpt-5.4.
  --out-root PATH          Output root for the pilot run. Default: pilot_runs/pilot_phase1/...
  --env-file PATH          Local LLM env file. Default: /tmp/easyicu_local_llm.env.
  --request-timeout SEC    Per-request timeout for the bench runner.
  --max-total-steps N      Optional ResearchAgentPipeline max_total_steps.
  --dry-run                Validate bootstrap and write a dry-run manifest only.
  --allow-dirty            Permit a dirty git working tree.
  -h, --help               Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --case) CASE_NAME="$2"; shift 2 ;;
    --backend) BACKEND="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --out-root) OUT_ROOT="$2"; shift 2 ;;
    --env-file) ENV_FILE="$2"; shift 2 ;;
    --request-timeout) REQUEST_TIMEOUT="$2"; shift 2 ;;
    --max-total-steps) MAX_TOTAL_STEPS="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --allow-dirty) ALLOW_DIRTY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ "${BACKEND}" == "mock" ]]; then
  MODEL="mock"
fi

if [[ "${ALLOW_DIRTY}" -ne 1 ]]; then
  if [[ -n "$(git -C "${REPO_ROOT}" status --porcelain)" ]]; then
    echo "Refusing to run archival pilot on a dirty worktree." >&2
    echo "Pass --allow-dirty for a disposable/non-archival pilot." >&2
    exit 3
  fi
fi

PYTHON_BIN="${PYTHON:-python}"

CASE_NAME="${CASE_NAME}" "${PYTHON_BIN}" - <<'PY'
import importlib
import os

from easyicu.research_agent.cohort_schema import PatternRegistry

case_name = os.environ["CASE_NAME"]
module = importlib.import_module(f"benchmark.cases.{case_name}.register_patterns")
register = getattr(module, "register_patterns")
registry = PatternRegistry()
register(registry)
print(f"registered_case_patterns={case_name}")
PY

CASE_CONFIG_LINES="$(REPO_ROOT="${REPO_ROOT}" CASE_NAME="${CASE_NAME}" "${PYTHON_BIN}" - <<'PY'
import os
from pathlib import Path

config_path = Path(os.environ["REPO_ROOT"]) / "benchmark" / "cases" / os.environ["CASE_NAME"] / "case_config.yaml"
bench_kind = None
bench_items = []
in_items = False
for line in config_path.read_text(encoding="utf-8").splitlines():
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        continue
    if stripped.startswith("bench_kind:"):
        bench_kind = stripped.split(":", 1)[1].strip().strip("'\"")
        in_items = False
        continue
    if stripped.startswith("bench_items:"):
        in_items = True
        continue
    if in_items:
        if line.startswith((" ", "\\t")) and stripped.startswith("- "):
            bench_items.append(stripped[2:].strip().strip("'\""))
            continue
        in_items = False
if not bench_kind:
    raise SystemExit(f"{config_path} is missing bench_kind")
if not bench_items:
    raise SystemExit(f"{config_path} is missing bench_items")
print(f"bench_kind\t{bench_kind}")
for item in bench_items:
    print(f"bench_item\t{item}")
PY
)"
BENCH_KIND=""
BENCH_ITEMS=()
while IFS=$'\t' read -r key value; do
  case "${key}" in
    bench_kind) BENCH_KIND="${value}" ;;
    bench_item) BENCH_ITEMS+=("${value}") ;;
  esac
done <<< "${CASE_CONFIG_LINES}"
if [[ -z "${BENCH_KIND}" || "${#BENCH_ITEMS[@]}" -eq 0 ]]; then
  echo "Failed to read bench_kind/bench_items from case_config.yaml" >&2
  exit 7
fi

if [[ "${BACKEND}" != "mock" ]]; then
  if [[ ! -f "${ENV_FILE}" ]]; then
    echo "Missing local LLM env file: ${ENV_FILE}" >&2
    exit 4
  fi
  PERM="$(stat -f "%Lp" "${ENV_FILE}" 2>/dev/null || stat -c "%a" "${ENV_FILE}")"
  if [[ "${PERM}" != "600" ]]; then
    echo "Refusing to source ${ENV_FILE}: expected chmod 600, got ${PERM}" >&2
    exit 5
  fi
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  BASE_URL="${OPENAI_BASE_URL:-http://127.0.0.1:8787/v1}"
  if ! curl -fsS -m 5 "${BASE_URL%/}/models" >/dev/null; then
    echo "Local OpenAI-compatible service is not reachable at ${BASE_URL%/}/models" >&2
    exit 6
  fi
fi

mkdir -p "${OUT_ROOT}"

CMD=(
  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/pilot_real_llm.py"
  --case "${CASE_NAME}"
  --bench-kind "${BENCH_KIND}"
  --bench-items "${BENCH_ITEMS[@]}"
  --arms aware
  --submission-profile
  --profile npj_dm/20260527
  --provider "${BACKEND}"
  --model "${MODEL}"
  --out-root "${OUT_ROOT}"
  --request-timeout "${REQUEST_TIMEOUT}"
)

if [[ -n "${MAX_TOTAL_STEPS}" ]]; then
  CMD+=(--max-total-steps "${MAX_TOTAL_STEPS}")
fi

if [[ "${DRY_RUN}" -eq 1 ]]; then
  OUT_ROOT="${OUT_ROOT}" CASE_NAME="${CASE_NAME}" BACKEND="${BACKEND}" MODEL="${MODEL}" \
  REQUEST_TIMEOUT="${REQUEST_TIMEOUT}" MAX_TOTAL_STEPS="${MAX_TOTAL_STEPS}" \
  COMMAND_PREVIEW="${CMD[*]}" "${PYTHON_BIN}" - <<'PY'
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from easyicu.research_agent.concept_dict_audit import compute_concept_dict_fingerprint

out = Path(os.environ["OUT_ROOT"])
payload = {
    "mode": "dry_run",
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "case": os.environ["CASE_NAME"],
    "backend": os.environ["BACKEND"],
    "model": os.environ["MODEL"],
    "request_timeout": os.environ["REQUEST_TIMEOUT"],
    "max_total_steps": os.environ.get("MAX_TOTAL_STEPS") or None,
    "command_preview": os.environ["COMMAND_PREVIEW"],
    "concept_dict_fingerprint": compute_concept_dict_fingerprint().to_dict(),
    "artifacts_expected": [
        "plan_locked.json",
        "cohort_locked.json",
        "robustness_panel.json",
        "manuscript.md",
        "manifest.json",
        "side_findings.md",
    ],
}
(out / "pilot_phase1_dry_run_manifest.json").write_text(
    json.dumps(payload, indent=2, ensure_ascii=False),
    encoding="utf-8",
)
print(f"DRY RUN complete: {out / 'pilot_phase1_dry_run_manifest.json'}")
PY
  exit 0
fi

"${CMD[@]}"

echo
echo "Pilot artifacts under ${OUT_ROOT}:"
find "${OUT_ROOT}" \( -name 'plan_locked.json' -o -name 'cohort_locked.json' \
  -o -name 'robustness_panel.json' -o -name 'manifest.json' \
  -o -name 'manuscript.md' -o -name 'side_findings.md' \) -print | sort
