#!/usr/bin/env python3
"""CLI entry point for the pyricu vs ricu feature comparison.

The heavy lifting now lives in :mod:`pyricu.feature_compare` so that other
tools can reuse the same logic.  This wrapper only ensures the repository's
``src`` directory is on ``sys.path`` when invoked directly from the repo root.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_SRC = Path(__file__).resolve().parent / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from pyricu.feature_compare import main


if __name__ == "__main__":
    main()
