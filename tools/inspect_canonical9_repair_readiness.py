#!/usr/bin/env python
"""Print the offline Canonical9 development-repair readiness report.

This command does not read cohort data, call a provider, materialize an export,
or start a benchmark.  It is a deterministic preflight report only.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from benchmarks.figure2_canonical9.development_repair_framework import (  # noqa: E402
    evaluate_development_repair_readiness,
    render_development_repair_report,
)


def main() -> int:
    rows, protocol_sha, binding_sha = evaluate_development_repair_readiness()
    print(
        json.dumps(
            render_development_repair_report(
                rows,
                repair_protocol_sha256=protocol_sha,
                input_binding_sha256=binding_sha,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
