#!/bin/bash
# bench7v3e = bench7v3d with each declared_absent entry also carrying "readers": 0 (an int, the grep basis)
# (2026-11-07). The sealed receipt further requires every entry to equal one of the approved entries in the
# sealed checkout's tools/formal_declared_absent.json. bench7v3d.sh and older are left as is (superseded).
# bench7v3d = bench7v3c accepting the optional top-level "declared_absent" allowlist of REQUIRED_LAYOUT.json
# (2026-11-07): [{"db", "table", "reason", "evidence"}], all non-empty strings, each (db, table) once
# (case-insensitive). The sealed receipt excuses a formal run's storage_unavailable records only through it.
# bench7v3c.sh, bench7v3b.sh and bench7v3.sh are left as is (superseded).
# bench7v3c = bench7v3b with the delta14v required-layout rules (2026-11-06): a third kind "csv" (a CSV the
# conversion manifest lists as the table's input; root_realpath optional like flat) and no top-level
# keys beyond schema/complete/databases. bench7v3b.sh and bench7v3.sh are left as is (superseded).
# bench7v3b = bench7v3 with the required-layout schema /2 (G4 amend, 2026-11-06): each table names its
# layout AND root. bench7v3.sh is left as is (it validates the superseded /1 schema; do not use it).
# bench7v3 = bench7v2 + the run's required layout (G4, 2026-11-06). bench7v2.sh is left as is.
# With BENCH_FORMAL_RECEIPT=1, BENCH_REQUIRED_LAYOUT must name a readable REQUIRED_LAYOUT.json:
#   {"schema": "easyicu.required_layout/2", "complete": true,
#    "databases": {DB: {TABLE: {"layout": "flat"|"csv"|"bucket", "root_realpath": "/abs"}, ...}, ...}}
#   one entry per spec database; root_realpath is required for a bucket (the installation it must be
#   served from) and optional for flat (default: the database's input root). The receipt compares
#   every read of every table with layout and root after resolving symlinks.
# It is copied to $R/REQUIRED_LAYOUT.json before the input snapshot (an existing different copy is
# refused) and its sha256 goes to status as `required_layout sha256=...`; a missing, incomplete or
# malformed list starts no job. The sealed receipt tool records the sha256 and refuses any table not
# served from its required layout (layout_deviation). Everything else is bench7v2 unchanged.
# bench7v2 = bench7 + input snapshots before and after the run (below). bench7.sh is left as is.
# bench7 = bench6 with the code dir a clean detached `git worktree` of SHA (lead-r13 provenance,
# 2026-11-06): the imported easyicu sits inside a Git checkout of exactly SHA, so every native
# manifest's runtime_provenance records easyicu_git_commit=SHA and easyicu_git_dirty=false.
# The worktrees belong to bench7_repo.git (a shallow bare repo owned by bench7, one commit fetched
# per SHA); the shared mirror is only fetched from. SHA must be the full 40-character commit. Everything else is bench6 unchanged.
# BENCH_FORMAL_RECEIPT=1 (formal runs): after every database exits, the sealed checkout's
# tools/formal_run_receipt.py writes RUN_RECEIPT.json in the run root (sealed SHA, parameters,
# per-database inputs, exits, peaks, native manifest commit, per-table storage_reads, and for SIC
# the cases.csv.gz -> cases.parquet binding with a scratch reconversion). Its exit goes to status
# as `receipt exit=N`; any nonzero exit means the run has no receipt.
# Inputs (delta9b ruling, 2026-11-06): before any job, the sealed checkout's easyicu.formal_inputs
# hashes every input file of every spec into $R/INPUT_SNAPSHOT.json (outside the inputs); after every
# job has exited it hashes the same set again into $R/INPUT_SNAPSHOT_POST.json. A failed pre-run
# snapshot starts no job; a difference after the run writes $R/RUN_INVALID, skips RUN_RECEIPT, and the
# run is not usable. Status lines: `inputs_pre exit=N`, `inputs_post exit=N`.
# Each snapshot step is admitted through memgate like a job (MemoryMax 4G, under memgate4.lock) and its
# script is $B/inputs_bench_local.py: memgate.py counts scopes whose command line contains the substring
# "bench_local.py" (and the core checks match the same pattern), so the step is counted under its own
# real script name; it stays on the first spec's cores and does not wait on the core-overlap check.
# Removing a code_<tag> dir later: `git --git-dir=$B/bench7_repo.git worktree remove code_<tag>`.
# bench6 = bench5 with the job timeout settable (BENCH_TIMEOUT, default 5h); lead-r13 2026-11-06 for the 2786 CareVue SOFA/sepsis reference.
# bench5 = bench4 + admission checked without holding the lock while waiting; bench4 = bench3 with the job subshell closing the admission lock (fd 9), so a running job no longer holds memgate4.lock.
# Local-profile benchmark: bench.sh TAG SHA MODE DB:DATA_DIR:CORES ...
# Each database runs in its own scope: MemoryMax=${BENCH_MEMMAX:-14G}, no swap, pinned cores.
# BENCH_THREADS=auto leaves EasyICU worker/engine-thread env unset (product defaults);
# otherwise engine threads = BENCH_THREADS and workers = BENCH_WORKERS (default = threads).
# Server rules (peer, 2026-11-06): cores only inside 188-255 (never 5-104); every library
# pool capped at the scope's core count; concurrent bench cores <= the segment width;
# wait while load1 > _codex/LOAD_GATE; bench_local.py re-checks its affinity and exits.
# bench2.sh (peer, 2026-11-06): segments 106-145 and 180-255 until 2026-11-07T14:00Z, then 188-255
# only; never 5-105 or 146-179. A spec must sit inside one segment; the window is re-checked at start.
# bench3.sh (peer, 2026-11-06 ~06:50Z): memgate.py admits a job only when the MemoryMax of all running
# bench scopes plus its own MemoryMax <= 260 GiB; under memgate.lock. Set BENCH_MEMMAX honestly per job.
set -u
START=$(date +%s)
TAG=$1; SHA=$2; MODE=$3; shift 3
[[ "$SHA" =~ ^[0-9a-f]{40}$ ]] || { echo "refusing $SHA: bench7 needs the full 40-character commit" >&2; exit 2; }
T=${BENCH_THREADS:-4}; WK=${BENCH_WORKERS:-$T}
EXPIRES=$(date -u -d 2026-11-07T14:00:00Z +%s)
segments() { if [ "$(date -u +%s)" -lt "$EXPIRES" ]; then echo "106-145 180-255"; else echo "188-255"; fi; }
seg_of() {  # the allowed segment holding [lo,hi], or empty
  local s; for s in $(segments); do [ "$1" -ge "${s%-*}" ] && [ "$2" -le "${s#*-}" ] && { echo "$s"; return; }; done; }
seg_width() { local s n=0; for s in $(segments); do n=$((n + ${s#*-} - ${s%-*} + 1)); done; echo $n; }
SEG_W=$(seg_width)
GATE_FILE=/home/zhuhb/workspace/claude-ws/_codex/LOAD_GATE
B=/home/zhuhb/easyicu_dev9/demo_sources/bench_local
C=$B/code_$TAG; R=$B/run_$TAG; PY=/home/zhuhb/easyicu_dev9/venv311/bin/python
G=/home/zhuhb/easyicu_dev9/codex_ws/_mirror.git
for spec in "$@"; do
  IFS=: read -r db dir cores <<< "$spec"
  lo=${cores%-*}; hi=${cores#*-}
  if ! [[ "$lo" =~ ^[0-9]+$ && "$hi" =~ ^[0-9]+$ ]] || [ "$lo" -gt "$hi" ] || [ -z "$(seg_of "$lo" "$hi")" ]; then
    echo "refusing $spec: cores must be a range inside one of: $(segments)" >&2; exit 2; fi
done
mkdir -p "$C" "$R" || exit 1
cd "$B" || exit 1
K=$B/bench7_repo.git
if [ ! -d "$C/src" ]; then
  ( flock 8
    [ -d "$K" ] || git init -q --bare "$K" || exit 1
    git --git-dir="$K" fetch -q --depth 1 "$G" "$SHA" || exit 1
    git --git-dir="$K" worktree add -q --detach "$C" "$SHA" || exit 1
  ) 8>"$B/bench7_repo.lock" || exit 1
fi
# The import path must be a clean checkout of exactly $SHA; refuse anything else.
[ "$(git -C "$C" rev-parse HEAD 2>/dev/null)" = "$SHA" ] || { echo "code_$TAG is not a checkout of $SHA" >&2; exit 1; }
[ -z "$(git -C "$C" status --porcelain --untracked-files=all)" ] || { echo "code_$TAG is not clean" >&2; exit 1; }
echo "tag=$TAG sha=$SHA mode=$MODE threads=$T workers=$WK" > "$R/status"
if [ "${BENCH_FORMAL_RECEIPT:-0}" = 1 ]; then
  RL=${BENCH_REQUIRED_LAYOUT:-}
  [ -n "$RL" ] && [ -f "$RL" ] && [ -r "$RL" ] || {
    echo "refusing: BENCH_FORMAL_RECEIPT=1 needs BENCH_REQUIRED_LAYOUT=<REQUIRED_LAYOUT.json>" >&2
    echo "required_layout_missing" > "$R/RUN_INVALID"; echo DONE >> "$R/status"; exit 2; }
  if [ -e "$R/REQUIRED_LAYOUT.json" ] && ! cmp -s "$RL" "$R/REQUIRED_LAYOUT.json"; then
    echo "refusing: $R/REQUIRED_LAYOUT.json exists and differs from $RL" >&2; exit 2; fi
  DBS=(); for spec in "$@"; do IFS=: read -r db _ _ <<< "$spec"; DBS+=("$db"); done
  env -i PATH=/usr/bin:/bin timeout 60 $PY -I - "$RL" "${DBS[@]}" <<'PYEOF' || {
import json, sys
p = json.load(open(sys.argv[1], encoding="utf-8"))
d = p.get("databases")
import os
def valid(v):
    if not isinstance(v, dict) or set(v) - {"layout", "root_realpath"}:
        return False
    r = v.get("root_realpath")
    if v.get("layout") == "bucket":
        return isinstance(r, str) and os.path.isabs(r)
    return v.get("layout") in ("flat", "csv") and ("root_realpath" not in v
                                          or isinstance(r, str) and os.path.isabs(r))
def absent_ok(a):
    if not isinstance(a, list):
        return False
    keys = ("db", "table", "reason", "evidence")
    if not all(isinstance(e, dict) and set(e) == set(keys) | {"readers"}
               and all(isinstance(e[k], str) and e[k].strip() for k in keys)
               and type(e["readers"]) is int and e["readers"] == 0 for e in a):
        return False
    pairs = [(e["db"], e["table"].lower()) for e in a]
    return len(set(pairs)) == len(pairs)
ok = (p.get("schema") == "easyicu.required_layout/2" and p.get("complete") is True
      and not set(p) - {"schema", "complete", "databases", "declared_absent"}
      and absent_ok(p.get("declared_absent", []))
      and isinstance(d, dict)
      and all(isinstance(d.get(db), dict) and d[db]
              and all(valid(v) for v in d[db].values()) for db in sys.argv[2:]))
sys.exit(0 if ok else 1)
PYEOF
    echo "refusing: $RL is not a complete required-layout list for: ${DBS[*]}" >&2
    echo "required_layout_invalid" > "$R/RUN_INVALID"; echo DONE >> "$R/status"; exit 2; }
  [ -e "$R/REQUIRED_LAYOUT.json" ] || cp "$RL" "$R/REQUIRED_LAYOUT.json" || exit 1
  chmod a-w "$R/REQUIRED_LAYOUT.json"
  echo "required_layout sha256=$(sha256sum "$R/REQUIRED_LAYOUT.json" | cut -d' ' -f1)" >> "$R/status"
fi
IFS=: read -r _db _dir IC <<< "$1"
inputs() {  # inputs <snapshot|verify> ARGS...: the sealed formal_inputs owner, admitted, pinned, capped
  local need=$((4 * 1073741824)) pid rc
  while :; do
    exec 9>"$B/memgate4.lock"; flock 9
    $PY $B/memgate.py check "$need" "inputs" 2>> "$R/queue.log" && break
    exec 9>&-; sleep 60
  done
  ( exec 9>&-; cd "$R" || exit 1; INC=$(( ${IC#*-} - ${IC%-*} + 1 ))
    timeout ${BENCH_INPUT_TIMEOUT:-4h} systemd-run --user --scope -q -p MemoryMax=4G -p MemorySwapMax=0 \
      env -i HOME=$HOME PATH=/usr/bin:/bin PYTHONPATH=$C/src EASYICU_HOME=$R/.inputs_home \
      OMP_NUM_THREADS=$INC OPENBLAS_NUM_THREADS=$INC MKL_NUM_THREADS=$INC NUMEXPR_MAX_THREADS=$INC \
      POLARS_MAX_THREADS=$INC RAYON_NUM_THREADS=$INC EASYICU_DUCKDB_THREADS=$INC \
      taskset -c "$IC" $PY $B/inputs_bench_local.py "$@" ) >> "$R/inputs.log" 2>&1 &
  pid=$!
  sleep 20; exec 9>&-  # the scope is visible to memgate before the lock is released
  wait "$pid"; rc=$?
  return $rc; }
inputs snapshot --out "$R/INPUT_SNAPSHOT.json" "$@"; PRE=$?
echo "inputs_pre exit=$PRE" >> "$R/status"
if [ "$PRE" != 0 ]; then echo "inputs_pre_failed" > "$R/RUN_INVALID"; echo DONE >> "$R/status"; exit 1; fi
if [ "$T" = auto ]; then TENV=(); else
  TENV=(EASYICU_PARALLEL_MAX_WORKERS=$WK EASYICU_DUCKDB_THREADS=$T EASYICU_ARROW_THREADS=$T); fi
busy_cores() {  # cores held by pinned bench processes; wrappers allowed on all cores are skipped
  local p a n=0
  for p in $(pgrep -u "$(id -u)" -f "[b]ench_local.py"); do
    a=$(awk '/Cpus_allowed_list/{print $2}' /proc/$p/status 2>/dev/null)
    case "$a" in *-*) w=$((${a#*-} - ${a%-*} + 1)); [ $w -le $SEG_W ] && n=$((n + w));; "") ;; *) n=$((n + 1));; esac
  done; echo $n; }
overlaps() {  # 1 when [lo,hi] intersects a running bench process's cores
  local p a
  for p in $(pgrep -u "$(id -u)" -f "[b]ench_local.py"); do
    a=$(awk '/Cpus_allowed_list/{print $2}' /proc/$p/status 2>/dev/null)
    case "$a" in *-*) [ $((${a#*-} - ${a%-*} + 1)) -le $SEG_W ] && [ "$1" -le "${a#*-}" ] && [ "$2" -ge "${a%-*}" ] && { echo 1; return; };; esac
  done; echo 0; }
for spec in "$@"; do
  IFS=: read -r db dir cores <<< "$spec"
  lo=${cores%-*}; hi=${cores#*-}; NC=$((hi - lo + 1))
  while :; do
    gate=$(cat "$GATE_FILE" 2>/dev/null || echo 340)
    load=$(awk '{print int($1)}' /proc/loadavg)
    SEG_W=$(seg_width); used=$(busy_cores)
    if [ -z "$(seg_of "$lo" "$hi")" ]; then echo "$db exit=refused_window_closed" >> "$R/status"; continue 2; fi
    if [ "$load" -le "$gate" ] && [ $((used + NC)) -le $SEG_W ] && [ "$(overlaps "$lo" "$hi")" = 0 ]; then break; fi
    echo "$(date -u +%H:%M:%SZ) wait $db load=$load gate=$gate used=$used need=$NC" >> "$R/queue.log"
    sleep 60
  done
  CAP=(OMP_NUM_THREADS=$NC OPENBLAS_NUM_THREADS=$NC MKL_NUM_THREADS=$NC NUMEXPR_MAX_THREADS=$NC
       POLARS_MAX_THREADS=$NC RAYON_NUM_THREADS=$NC BENCH_SEG=$(seg_of $lo $hi) BENCH_NC=$NC)
  MM=${BENCH_MEMMAX:-14G}; MMB=$(( ${MM%G} * 1073741824 ))
  NEED=$MMB
  # Never hold the lock while waiting for budget: a large job must not block smaller ones.
  while :; do
    exec 9>"$B/memgate4.lock"; flock 9
    $PY $B/memgate.py check "$NEED" "$MODE" 2>> "$R/queue.log" && break
    exec 9>&-; sleep 60
  done
  ( exec 9>&-; W=$R/$db; mkdir -p "$W/home" "$W/spill"; cd "$W" || exit 1
    timeout ${BENCH_TIMEOUT:-5h} systemd-run --user --scope -q -p MemoryMax=${BENCH_MEMMAX:-14G} -p MemorySwapMax=0 env -i HOME=$HOME PATH=/usr/bin:/bin \
      BENCH_GATED=1 BENCH_BATCH=${BENCH_BATCH:-} BENCH_MODULES=${BENCH_MODULES:-} ${BENCH_DUCKMEM:+EASYICU_DUCKDB_MEMORY_LIMIT=$BENCH_DUCKMEM} PYTHONPATH=$C/src EASYICU_HOME=$W/home EASYICU_DUCKDB_TEMP_DIR=$W/spill \
      "${CAP[@]}" "${TENV[@]}" taskset -c "$cores" $PY $B/bench_local.py "$db" "$dir" "$W" "$MODE" > "$W/run.log" 2>&1
    echo "$db exit=$?" >> "$R/status"; rm -rf "${W:?}/spill" ) &
  sleep 20  # let the scope start so the next spec sees its cores as busy
  sleep 70; exec 9>&-  # let RSS show before the next admission
done
wait
inputs verify --pre "$R/INPUT_SNAPSHOT.json" --out "$R/INPUT_SNAPSHOT_POST.json"; POST=$?
echo "inputs_post exit=$POST" >> "$R/status"
rm -rf "${R:?}/.inputs_home"
if [ "$POST" != 0 ]; then
  echo "inputs_changed_during_run" > "$R/RUN_INVALID"; echo DONE >> "$R/status"; exit 1; fi
if [ "${BENCH_FORMAL_RECEIPT:-0}" = 1 ]; then
  IFS=: read -r _db _dir RC <<< "$1"; RNC=$(( ${RC#*-} - ${RC%-*} + 1 ))
  RS=$R/.receipt_scratch; mkdir -p "$RS/home" || exit 1
  ( cd "$RS" || exit 1
    timeout 2h systemd-run --user --scope -q -p MemoryMax=8G -p MemorySwapMax=0 env -i HOME=$HOME PATH=/usr/bin:/bin \
      PYTHONPATH=$C/src EASYICU_HOME=$RS/home EASYICU_DUCKDB_TEMP_DIR=$RS OMP_NUM_THREADS=$RNC OPENBLAS_NUM_THREADS=$RNC \
      MKL_NUM_THREADS=$RNC NUMEXPR_MAX_THREADS=$RNC POLARS_MAX_THREADS=$RNC RAYON_NUM_THREADS=$RNC EASYICU_CONV_WORKERS=1 \
      taskset -c "$RC" $PY $C/tools/formal_run_receipt.py --run-root "$R" --sealed-sha "$SHA" --scratch "$RS" \
      --started-at "$START" --param mode=$MODE --param threads=$T --param workers=$WK \
      --param memmax=${BENCH_MEMMAX:-14G} --param batch=${BENCH_BATCH:-default} \
      --param modules=${BENCH_MODULES:-all} --param timeout=${BENCH_TIMEOUT:-5h} "$@" > "$R/receipt.log" 2>&1
    echo "receipt exit=$?" >> "$R/status" )
  rm -rf "${RS:?}/home"
fi
echo DONE >> "$R/status"
