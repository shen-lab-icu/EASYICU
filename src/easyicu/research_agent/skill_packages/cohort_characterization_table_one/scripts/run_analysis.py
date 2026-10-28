"""Execute the declared Table 1 with the host's summary and testing kernel."""

import json
from pathlib import Path

from ....methods.table_one import build_grouped_table_one
from ...reference_io import LoadedCohort
from ..spec import TableOneReferenceSpec


def run_analysis(loaded: LoadedCohort, spec: TableOneReferenceSpec, out_dir: str | Path) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    table = build_grouped_table_one(loaded.frame, spec.table_one)
    table.to_csv(out / "table_one.csv", index=False)
    summary = {
        "status": "ok", "cohort_n": len(loaded.frame), "table_one_result_rows": len(table),
        "output_files": {"table:table_one": "table_one.csv"},
    }
    (out / "step_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
