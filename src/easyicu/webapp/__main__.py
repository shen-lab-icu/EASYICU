"""Command-line entry point for the EasyICU webapp."""

from __future__ import annotations

import argparse
from typing import Sequence

from . import run_app, status_app, stop_app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the EasyICU web application.")
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Start the EasyICU web application.")
    run_parser.add_argument("--host", default="0.0.0.0", help="Host interface to bind.")
    run_parser.add_argument("--port", type=int, default=8501, help="Port to bind.")
    run_parser.add_argument("--debug", action="store_true", help="Enable verbose Streamlit logging.")
    run_parser.add_argument("--daemon", action="store_true", help="Restart the process if it crashes.")
    run_parser.add_argument("--background", action="store_true", help="Run in the background.")

    stop_parser = subparsers.add_parser("stop", help="Stop a background EasyICU webapp.")
    stop_parser.add_argument("--port", type=int, default=8501, help="Unused placeholder for symmetry.")

    status_parser = subparsers.add_parser("status", help="Check whether EasyICU is running.")
    status_parser.add_argument("--port", type=int, default=8501, help="Port to probe.")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    command = args.command or "run"

    if command == "run":
        if args.command is None:
            run_app()
            return 0

        run_app(
            host=args.host,
            port=args.port,
            debug=args.debug,
            daemon=args.daemon,
            background=args.background,
        )
        return 0

    if command == "stop":
        stop_app()
        return 0

    if command == "status":
        status_app(port=args.port)
        return 0

    parser.error(f"Unsupported command: {command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
