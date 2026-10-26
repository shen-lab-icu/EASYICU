#!/usr/bin/env python3
"""Build the registered aggregate replay gallery without running studies."""

from __future__ import annotations

import argparse
from pathlib import Path

from easyicu.webserver.study_replay import ReplayExportError
from easyicu.webserver.study_replay_gallery import export_replay_gallery


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--registry", default="replay_registry.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        export_replay_gallery(args.root, args.registry, args.output)
    except ReplayExportError as exc:
        print(exc.code)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
