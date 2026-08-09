#!/usr/bin/env python3
"""Build verified GitHub Release assets for prepared official ICU demos."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from easyicu import demo_release_pack
from easyicu.webserver import demo_source_storage
from easyicu.webserver.demo_source_contracts import SOURCE_BY_ID


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        action="append",
        choices=sorted(SOURCE_BY_ID),
        required=True,
        help="Allowlisted official demo source ID; repeat for both demos.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dist/demo-data"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    receipts = []
    for source_id in args.source:
        source = SOURCE_BY_ID[source_id]
        paths = demo_source_storage.source_paths(source)
        receipt = demo_release_pack.build_release_pack(
            source,
            paths,
            args.output_dir,
        )
        receipts.append(asdict(receipt))
    print(json.dumps({"ok": True, "receipts": receipts}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
