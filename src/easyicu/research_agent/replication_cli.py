"""CLI for deterministic EasyICU research-agent replication packages."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional, Sequence


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="easyicu-research-replication",
        description=(
            "Build deterministic lactate-MAP-vasopressor replication tables "
            "from one or more EasyICU concept-export directories."
        ),
    )
    parser.add_argument(
        "--target",
        action="append",
        default=[],
        metavar="DB=EXPORT_DIR",
        help="Database/export pair, e.g. miiv=/path/to/easyicu_export. Repeatable.",
    )
    parser.add_argument(
        "--build-target",
        action="append",
        default=[],
        metavar="DB=DATA_PATH",
        help=(
            "Build a minimal shock-case EasyICU export from a prepared database path, "
            "then include it as a replication target. Repeatable."
        ),
    )
    parser.add_argument(
        "--discover-root",
        action="append",
        default=[],
        metavar="PATH",
        help="Search this directory recursively for EasyICU export manifests.",
    )
    parser.add_argument(
        "--pending",
        action="append",
        default=[],
        metavar="DB",
        help="Planned replication database with no local EasyICU export yet.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for cohorts, tables, manifest and appendix.",
    )
    parser.add_argument("--window-start", type=float, default=0.0)
    parser.add_argument("--window-end", type=float, default=24.0)
    parser.add_argument(
        "--export-root",
        default=None,
        help="Where --build-target should write generated EasyICU exports.",
    )
    parser.add_argument(
        "--max-patients",
        type=int,
        default=None,
        help="Optional patient cap passed to EasyICU concept loading for --build-target.",
    )
    parser.add_argument(
        "--minimal-export",
        action="store_true",
        help="For --build-target, export only required shock concepts and skip optional circ/sepsis derivations.",
    )
    return parser


def _parse_targets(raw_targets: Sequence[str], pending: Sequence[str]) -> Dict[str, Optional[Path]]:
    targets: Dict[str, Optional[Path]] = {}
    for raw in raw_targets:
        if "=" not in raw:
            raise SystemExit(f"--target must be DB=EXPORT_DIR, got: {raw!r}")
        database, path = raw.split("=", 1)
        database = database.strip()
        path = path.strip()
        if not database or not path:
            raise SystemExit(f"--target must be DB=EXPORT_DIR, got: {raw!r}")
        targets[database] = Path(path)
    for database in pending:
        database = database.strip()
        if database:
            targets.setdefault(database, None)
    if not targets:
        raise SystemExit("Provide at least one --target or --pending database.")
    return targets


def _parse_pairs(raw_pairs: Sequence[str], flag: str) -> Dict[str, Path]:
    pairs: Dict[str, Path] = {}
    for raw in raw_pairs:
        if "=" not in raw:
            raise SystemExit(f"{flag} must be DB=PATH, got: {raw!r}")
        database, path = raw.split("=", 1)
        database = database.strip()
        path = path.strip()
        if not database or not path:
            raise SystemExit(f"{flag} must be DB=PATH, got: {raw!r}")
        pairs[database] = Path(path)
    return pairs


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)

    from .replication import (
        LACTATE_MAP_VASO_MINIMAL_EXPORT_GROUPS,
        discover_easyicu_exports,
        export_lactate_map_vaso_concepts_from_easyicu,
        run_lactate_map_vaso_replication,
    )

    targets = _parse_targets(args.target, []) if args.target else {}
    if args.discover_root:
        targets.update(discover_easyicu_exports([Path(p) for p in args.discover_root]))

    build_targets = _parse_pairs(args.build_target, "--build-target")
    export_root = Path(args.export_root) if args.export_root else Path(args.output) / "generated_exports"
    for database, data_path in build_targets.items():
        generated = export_lactate_map_vaso_concepts_from_easyicu(
            database=database,
            data_path=data_path,
            output_dir=export_root / database,
            max_patients=args.max_patients,
            groups=LACTATE_MAP_VASO_MINIMAL_EXPORT_GROUPS if args.minimal_export else None,
        )
        targets[database] = generated

    for database in args.pending:
        database = database.strip()
        if database:
            targets.setdefault(database, None)
    if not targets:
        raise SystemExit("Provide at least one --target, --build-target, --discover-root, or --pending database.")

    paths = run_lactate_map_vaso_replication(
        targets,
        args.output,
        window=(args.window_start, args.window_end),
    )
    for name, path in paths.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
