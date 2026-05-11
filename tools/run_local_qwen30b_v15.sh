#!/usr/bin/env bash
# EasyICU v15 local qwen3-coder-30b launcher
# ==========================================
# Drives tools/run_v14_agent_experiments.py against the locally hosted
# qwen3-coder-30b endpoint that exposes an OpenAI-compatible API at
# http://127.0.0.1:11435/v1.
#
# v15 vs v14 manuscript-ready setup:
#   * enable_replanning = True              (was False)
#   * enable_memory     = True              (was False; intra-run RunMemory on)
#   * 2x2 factorial arms over (icu_context, user_preferences):
#       aware, aware_no_pref, naive_with_pref, naive
#   * VLM visual QA stays OFF because qwen3-coder-30b is text-only.
#   * Literature / HypothesisBlueprintAgent stays OFF by default in this script:
#     it gates the pipeline as "blocked" on descriptive / correlation / audit
#     tasks (t01, t02, t03, t09) because they lack a clinical hypothesis, which
#     dropped a v15 trial run from 100% to 33% clean_ok. Re-enable it via
#     EASYICU_V15_ENABLE_LITERATURE=1 if you specifically want the literature-
#     driven hypothesis-generation arm for ablation purposes.
#
# Usage:
#   tools/run_local_qwen30b_v15.sh smoke      # 1 task x 2 arms (aware vs naive), wiring check
#   tools/run_local_qwen30b_v15.sh smoke4     # 1 task x 4 arms, full factorial wiring
#   tools/run_local_qwen30b_v15.sh full       # 10 tasks x 4 arms, manuscript run
#   tools/run_local_qwen30b_v15.sh aggregate <out-root>  # re-aggregate only

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

MODE="${1:-smoke}"

# Local qwen3-coder-30b endpoint (OpenAI-compatible). API key is unused by
# the upstream server but the OpenAI client requires the env var to be set.
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://127.0.0.1:11435/v1}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-ollama}"

MODEL="${EASYICU_LOCAL_QWEN_MODEL:-qwen3-coder-30b}"
PROVIDER="openai"

# Pick the python interpreter that already has pandas + easyicu installed.
# Defaults to the user's anaconda Python; override with EASYICU_PY if needed.
PYTHON_BIN="${EASYICU_PY:-/opt/anaconda3/bin/python}"
if ! "${PYTHON_BIN}" -c "import pandas, easyicu" >/dev/null 2>&1; then
  echo "ERROR: ${PYTHON_BIN} missing pandas or easyicu; set EASYICU_PY=/path/to/python." >&2
  exit 3
fi

STAMP="$(date -u +%Y%m%d_%H%M)"
RUN_ROOT="${REPO_ROOT}/research_output/v15_experiments_${STAMP}"

ARMS_FULL=(aware aware_no_pref naive_with_pref naive)
ARMS_LEGACY=(aware naive)

case "${MODE}" in
  smoke)
    OUT="${RUN_ROOT}_smoke"
    ITEMS=(t04_lactate_mortality_association)
    ARMS=("${ARMS_LEGACY[@]}")
    REQUEST_TIMEOUT=180
    TASK_TIMEOUT=900
    ;;
  smoke4)
    OUT="${RUN_ROOT}_smoke4"
    ITEMS=(t04_lactate_mortality_association)
    ARMS=("${ARMS_FULL[@]}")
    REQUEST_TIMEOUT=180
    TASK_TIMEOUT=900
    ;;
  full)
    OUT="${RUN_ROOT}_full"
    ITEMS=()  # all 10 tasks
    ARMS=("${ARMS_FULL[@]}")
    REQUEST_TIMEOUT=300
    TASK_TIMEOUT=1800
    ;;
  aggregate)
    if [[ -z "${2:-}" ]]; then
      echo "usage: $0 aggregate <out-root>" >&2
      exit 2
    fi
    OUT="$2"
    "${PYTHON_BIN}" tools/run_v14_agent_experiments.py \
      --provider "${PROVIDER}" \
      --model "${MODEL}" \
      --arms "${ARMS_FULL[@]}" \
      --out-root "${OUT}" \
      --aggregate-only
    exit 0
    ;;
  *)
    echo "unknown mode: ${MODE}" >&2
    echo "usage: $0 {smoke|smoke4|full|aggregate <out-root>}" >&2
    exit 2
    ;;
esac

mkdir -p "${OUT}"

if [[ "${EASYICU_V15_ENABLE_LITERATURE:-0}" == "1" ]]; then
  LITERATURE_FLAG="--enable-literature"
else
  LITERATURE_FLAG="--no-literature"
fi

CMD=(
  "${PYTHON_BIN}" tools/run_v14_agent_experiments.py
  --provider "${PROVIDER}"
  --model "${MODEL}"
  --arms "${ARMS[@]}"
  --enable-replanning
  "${LITERATURE_FLAG}"
  --enable-memory
  --no-vlm-visual-qa
  --experiment-mode guardrails
  --request-timeout "${REQUEST_TIMEOUT}"
  --task-timeout "${TASK_TIMEOUT}"
  --max-retries 1
  --out-root "${OUT}"
)

if [[ ${#ITEMS[@]} -gt 0 ]]; then
  CMD+=(--items "${ITEMS[@]}")
fi

echo "[$(date -u +%FT%TZ)] mode=${MODE}"
echo "[$(date -u +%FT%TZ)] OPENAI_BASE_URL=${OPENAI_BASE_URL}"
echo "[$(date -u +%FT%TZ)] model=${MODEL} provider=${PROVIDER}"
echo "[$(date -u +%FT%TZ)] out=${OUT}"
echo "[$(date -u +%FT%TZ)] cmd: ${CMD[*]}"
exec "${CMD[@]}"
