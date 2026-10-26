#!/usr/bin/env python3
"""Export one existing study wrapper as an aggregate-only static replay."""

from __future__ import annotations

import argparse
from pathlib import Path

from easyicu.webserver.study_replay import ReplayExportError, export_study_replay


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    try:
        export_study_replay(args.run_dir, args.destination)
    except ReplayExportError as exc:
        print(exc.code)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
