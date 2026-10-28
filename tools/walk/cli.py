"""Command line entry for an already-running EasyICU HTTP server."""

from __future__ import annotations

import argparse
from pathlib import Path

from .contracts import PHASES
from .runner import Options, WalkRunner


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--base", default="http://127.0.0.1:8780", help="Single HTTP origin; redirects are refused")
    result.add_argument("--source", required=True, help="Exact configured source id or unambiguous name")
    result.add_argument("--out", required=True, type=Path, help="Report directory outside the repository")
    result.add_argument("--only", default=",".join(PHASES), help="Comma-separated phases: " + ",".join(PHASES))
    result.add_argument("--max-workflows", type=int, help="Maximum runnable examples, including method examples")
    result.add_argument("--resume", action="store_true", help="Reuse this server/source's finished receipts")
    result.add_argument("--approve-test-runs", action="store_true", help="Approve current host-offered plans in walk projects")
    result.add_argument("--request-timeout", type=float, default=30, metavar="SECONDS")
    result.add_argument("--poll-interval", type=float, default=1, metavar="SECONDS")
    for phase, seconds in Options(source="unused", out=Path("unused")).timeouts.items():
        result.add_argument(f"--timeout-{phase}", type=float, default=seconds, metavar="SECONDS",
                            help="Per-item deadline, including polling and HTTP requests")
    return result


def main(argv=None, *, runner_factory=WalkRunner):
    arguments = parser()
    parsed = arguments.parse_args(argv)
    try:
        options = Options(source=parsed.source, out=parsed.out, base=parsed.base,
                          only=tuple(part.strip() for part in parsed.only.split(",")),
                          max_workflows=parsed.max_workflows, resume=parsed.resume,
                          approve_test_runs=parsed.approve_test_runs,
                          timeouts={phase: getattr(parsed, f"timeout_{phase}") for phase in PHASES},
                          request_timeout=parsed.request_timeout, poll_interval=parsed.poll_interval)
        report = runner_factory(options).run()
    except (ValueError, OSError):
        arguments.exit(2, "walk refused: invalid options, resume scope, or output directory\n")
    counts = report["status_counts"]
    print("; ".join(f"{status}: {count}" for status, count in sorted(counts.items())))
    return 1 if counts.get("failed") or counts.get("timed-out") else 0
