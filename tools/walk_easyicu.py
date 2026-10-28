#!/usr/bin/env python3
"""Walk an existing EasyICU server through its HTTP gates; save aggregates only."""

from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.walk.cli import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
