"""Local-profile extraction benchmark: one database, all modules, native-v2.

Run inside a systemd scope (MemoryMax=14G, 4 pinned cores, engine threads 4).
Records wall time, the sampled peak RSS of this process tree, the cgroup peak
and OOM kills, and the sha256 of every export file. Aggregates only.

usage: bench_local.py DB DATA_PATH OUT (sample|full)
"""
import hashlib
import json
import random
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

import os

import psutil


def _cgroup_dir():
    rel = Path("/proc/self/cgroup").read_text().strip().split(":", 2)[2]
    return Path("/sys/fs/cgroup" + rel)


def _read(path):
    try:
        return path.read_text().strip()
    except OSError:
        return None


class TreeRss(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.peak = 0
        self.stop = threading.Event()

    def run(self):
        me = psutil.Process()
        while not self.stop.wait(1.0):
            total = 0
            try:
                procs = [me, *me.children(recursive=True)]
            except psutil.Error:
                continue
            for proc in procs:
                try:
                    total += proc.memory_info().rss
                except psutil.Error:
                    pass
            self.peak = max(self.peak, total)


def _affinity_selfcheck():
    """Exit unless pinned inside BENCH_SEG with exactly BENCH_NC cores and capped pools."""
    seg = os.environ.get("BENCH_SEG", "188-255")
    lo, hi = (int(x) for x in seg.split("-"))
    cpus = sorted(os.sched_getaffinity(0))
    want = int(os.environ.get("BENCH_NC", "0") or 0)
    bad = [c for c in cpus if not lo <= c <= hi]
    caps = {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "POLARS_MAX_THREADS", "RAYON_NUM_THREADS", "OPENBLAS_NUM_THREADS")}
    if bad or (want and len(cpus) != want) or any(v != str(len(cpus)) for v in caps.values()):
        sys.stderr.write(f"bench self-check failed: affinity={cpus[:4]}..({len(cpus)}) seg={seg} want={want} caps={caps}\n")
        sys.exit(3)
    return len(cpus)


def main():
    ncpu = _affinity_selfcheck()
    db, data_path, out, mode = sys.argv[1], sys.argv[2], Path(sys.argv[3]), sys.argv[4]
    # Account memory gate (peer, 2026-11-06T06:41Z): wait until RSS + this job fits 420 GiB.
    subprocess.run([sys.executable, str(Path(__file__).with_name("memgate.py")), "self", mode], check=True)
    import easyicu
    from easyicu.api import extract_database
    from easyicu.api.extraction import EXTRACT_MODULE_ORDER, _get_all_patient_ids

    summary = {"db": db, "mode": mode, "easyicu": str(Path(easyicu.__file__).parent),
               "affinity": len(psutil.Process().cpu_affinity())}
    patient_ids = None
    if mode == "sample":
        ids, col = _get_all_patient_ids(data_path, db, None)
        ids = sorted(ids)
        random.Random(20261104).shuffle(ids)
        patient_ids = {col: sorted(ids[:3000])}
    cg = _cgroup_dir()
    summary["cgroup_memory_max"] = _read(cg / "memory.max")
    try:
        from easyicu.runtime.parallel_config import get_global_config
        from easyicu.datasource import _default_duckdb_threads
        summary["effective_workers"] = get_global_config().max_workers
        summary["effective_duckdb_threads"] = _default_duckdb_threads()
        if int(summary["effective_duckdb_threads"]) > ncpu:
            sys.stderr.write("bench self-check failed: DuckDB threads exceed the pinned cores\n")
            sys.exit(3)
    except Exception as exc:
        summary["effective_workers"] = repr(exc)[:200]
    sampler = TreeRss()
    sampler.start()
    t0 = time.time()
    try:
        batch = os.environ.get("BENCH_BATCH")
        summary["batch_size"] = int(batch) if batch else None
        res = extract_database(db, data_path=data_path, output_dir=out / "export",
                               modules=[m for m in os.environ.get("BENCH_MODULES", "").split(",") if m] or list(EXTRACT_MODULE_ORDER), patient_ids=patient_ids,
                               native_export_v2=True, stream_output_batches=True, verbose=False,
                               batch_size=int(batch) if batch else None)
        summary["errors"] = [str(e)[:300] for e in (res.get("errors") or [])]
        summary["worker_failures"] = len(res.get("worker_failures") or [])
    except Exception:
        summary["exception"] = traceback.format_exc()[-1500:]
    summary["secs"] = round(time.time() - t0, 1)
    sampler.stop.set()
    sampler.join()
    summary["peak_tree_rss_gib"] = round(sampler.peak / 2**30, 2)
    peak = _read(cg / "memory.peak")
    summary["cgroup_peak_gib"] = round(int(peak) / 2**30, 2) if peak and peak.isdigit() else None
    events = _read(cg / "memory.events") or ""
    summary["memory_events"] = dict(line.split() for line in events.splitlines() if line)
    files = {}
    for path in sorted((out / "export").rglob("*")):
        if path.is_file():
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1 << 20), b""):
                    digest.update(block)
            files[str(path.relative_to(out / "export"))] = digest.hexdigest()
    summary["files"] = files
    (out / "summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
