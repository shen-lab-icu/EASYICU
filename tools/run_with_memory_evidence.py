#!/usr/bin/env python3
"""Run one command while recording process-tree RSS/PSS timing evidence."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import psutil


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _process_tree_memory_mb(process: psutil.Process) -> tuple[float, float, int]:
    """Return summed RSS/PSS for a live process tree.

    RSS intentionally sums per-process resident sets to match the historical
    EasyICU timing contract. PSS apportions shared pages and is therefore the
    preferred estimate of the tree's physical working set.
    """

    try:
        processes = [process, *process.children(recursive=True)]
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        processes = [process]
    rss_kb = 0.0
    pss_kb = 0.0
    seen: set[int] = set()
    for member in processes:
        if member.pid in seen:
            continue
        seen.add(member.pid)
        try:
            rss_kb += member.memory_info().rss / 1024.0
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        try:
            with Path(f"/proc/{member.pid}/smaps_rollup").open(
                encoding="utf-8"
            ) as handle:
                for line in handle:
                    if line.startswith("Pss:"):
                        pss_kb += float(line.split()[1])
                        break
        except (OSError, ValueError, IndexError):
            # smaps_rollup is Linux-specific. Keep the evidence portable by
            # falling back to RSS for this process on macOS/Windows or when
            # procfs access is restricted.
            try:
                pss_kb += member.memory_info().rss / 1024.0
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
    return rss_kb / 1024.0, pss_kb / 1024.0, len(seen)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def run(command: Sequence[str], *, output: Path, interval: float) -> int:
    if not command:
        raise ValueError("a command is required after --")
    started_wall = time.monotonic()
    started_at = _utc_now()
    child = subprocess.Popen(list(command))
    root = psutil.Process(child.pid)
    peak_rss_mb = 0.0
    peak_pss_mb = 0.0
    peak_process_count = 0
    samples = 0
    while True:
        rss_mb, pss_mb, process_count = _process_tree_memory_mb(root)
        peak_rss_mb = max(peak_rss_mb, rss_mb)
        peak_pss_mb = max(peak_pss_mb, pss_mb)
        peak_process_count = max(peak_process_count, process_count)
        samples += 1
        exit_code = child.poll()
        if exit_code is not None:
            break
        time.sleep(max(0.05, interval))
    payload = {
        "schema_version": "easyicu_process_tree_memory_evidence_v1",
        "command": list(command),
        "started_at_utc": started_at,
        "ended_at_utc": _utc_now(),
        "elapsed_seconds": round(time.monotonic() - started_wall, 3),
        "process_exit_code": int(exit_code),
        "peak_process_tree_rss_mb": round(peak_rss_mb, 1),
        "peak_process_tree_pss_mb": round(peak_pss_mb, 1),
        "peak_process_count": peak_process_count,
        "sample_interval_seconds": max(0.05, interval),
        "samples": samples,
    }
    _atomic_write_json(output, payload)
    return int(exit_code)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--interval", type=float, default=0.25)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("a command is required after --")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    return run(args.command, output=args.output, interval=args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
