"""Load every concept alone through the public API and list the ones that fail.

Extraction loads concepts in module groups, so a concept that only fails when a
user loads it on its own through ``easyicu.load_concepts`` (for example a
window table whose producer never declared its ``dur_var`` unit) is not
exercised there. This sweep loads each concept separately, for every stay of
the source, and records the exception class and message of each failure.

Output is aggregate only: concept names, targets, row counts and exception
messages; no rows are written.

    python tools/public_api_concept_sweep.py --database eicu \
        --data-path ~/.easyicu/demo_sources/eicu_demo_v2_0_1/raw --out sweep_eicu.json
"""

from __future__ import annotations

import argparse
import json
import logging
import time
import traceback
import warnings
from pathlib import Path
from typing import Dict, List


def _concepts(database: str, *, win_tbl_only: bool) -> List[str]:
    from easyicu.resources import load_dictionary

    dictionary = load_dictionary(source_name=database, include_sofa2=True)
    names = []
    for name, definition in dictionary.items():
        sources = getattr(definition, "sources", None) or {}
        derived = bool(getattr(definition, "sub_concepts", None))
        if database not in sources and not derived:
            continue
        if win_tbl_only and str(getattr(definition, "target", "") or "") != "win_tbl":
            continue
        names.append(name)
    return sorted(names)


def _target(database: str, name: str) -> str:
    from easyicu.resources import load_dictionary

    definition = load_dictionary(source_name=database, include_sofa2=True).get(name)
    return str(getattr(definition, "target", "") or "")


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--database", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--concepts", help="comma-separated subset")
    parser.add_argument("--win-tbl-only", action="store_true")
    args = parser.parse_args(argv)

    warnings.filterwarnings("ignore")
    logging.disable(logging.CRITICAL)
    import easyicu

    names = (
        args.concepts.split(",") if args.concepts
        else _concepts(args.database, win_tbl_only=args.win_tbl_only)
    )
    results: Dict[str, dict] = {}
    for name in names:
        started = time.time()
        entry: dict = {"target": _target(args.database, name)}
        try:
            frame = easyicu.load_concepts(
                [name], database=args.database, data_path=args.data_path,
                merge=True, verbose=False,
            )
            entry["rows"] = int(len(frame)) if hasattr(frame, "__len__") else None
        except Exception as exc:  # recorded per concept, never hidden
            entry["error"] = type(exc).__name__
            entry["message"] = str(exc).splitlines()[0][:300] if str(exc) else ""
            frames = traceback.extract_tb(exc.__traceback__)
            entry["raised_at"] = (
                f"{Path(frames[-1].filename).name}:{frames[-1].lineno}" if frames else ""
            )
        entry["seconds"] = round(time.time() - started, 1)
        results[name] = entry

    failures = {n: e for n, e in results.items() if "error" in e}
    summary = {
        "database": args.database,
        "concepts": len(results),
        "failed": len(failures),
        "failures_by_error": {},
        "results": results,
    }
    for name, entry in sorted(failures.items()):
        summary["failures_by_error"].setdefault(entry["error"], []).append(name)
    Path(args.out).write_text(json.dumps(summary, indent=1, sort_keys=True))
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}, indent=1))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
