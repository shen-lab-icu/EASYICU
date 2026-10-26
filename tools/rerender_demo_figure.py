"""Re-render the workflow demo's figures with the current renderers.

Usage: PYTHONPATH=src python tools/rerender_demo_figure.py RUN_DIR

The demo's source run registered each figure's input table and the figure drawn
from it. This tool binds the same tables (each verified against its registered
digest), hands each to the current renderer of its step with the approved
plan's display labels, and writes the PNGs beside the originals in the demo
assets. It reads nothing else from the run: no cohort, no patient rows. The
original figures and their digests stay as registered.

The run predates three columns the current distribution table contract
carries. They hold values only under the patient-weighted Hoeffding design, so
for that table's declared counts-only design they are structurally empty; the
tool adds them empty and refuses any other design rather than invent
uncertainty fields.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from easyicu.research_agent.execution.runners.cohort_flow_figure_executor import (
    run_cohort_flow_figure,
)
from easyicu.research_agent.execution.runners.exposure_outcome_distribution_executor import (
    EXPOSURE_OUTCOME_DISTRIBUTION_COLUMNS,
)
from easyicu.research_agent.execution.runners.exposure_outcome_distribution_render import (
    run_exposure_outcome_distribution_figure,
)
from easyicu.research_agent.execution.runners.missingness_measurement_figure_executor import (
    run_measurement_missingness_figure,
)

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "src/easyicu/webserver/static/assets/demo"
LATER_EMPTY_COLUMNS = {
    "exposure_interval_cluster_size_square_sum",
    "outcome_interval_cluster_size_square_sum",
    "patient_weighted_interval_assumptions",
}


def _distribution_upgrade(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    added = [name for name in EXPOSURE_OUTCOME_DISTRIBUTION_COLUMNS if name not in frame.columns]
    if added and set(frame["interval_method"].dropna()) != {"none_counts_only"}:
        raise SystemExit("only a counts-only table may gain the later, empty columns")
    if set(added) - LATER_EMPTY_COLUMNS:
        raise SystemExit(f"the table lacks columns it should carry: {sorted(set(added) - LATER_EMPTY_COLUMNS)}")
    for name in added:
        frame[name] = pd.NA
    return frame[list(EXPOSURE_OUTCOME_DISTRIBUTION_COLUMNS)], added


#: One row per demo figure: its step, the registered table it draws, and the
#: current renderer of that step.
FIGURES: tuple[dict[str, Any], ...] = (
    {
        "key": "main", "renderer": "exposure_outcome_distribution_render",
        "step_id": "04_visualize_exposure_outcome_distribution",
        "input_key": "table:exposure_outcome_distribution",
        "source_table": "table_step_artifact_257971d9f784c410",
        "product": "sep3_sofa2_mortality_distribution",
        "png": "sofa2-phenotype-mortality-rerendered.png",
        "render": run_exposure_outcome_distribution_figure, "labels": True,
        "upgrade": _distribution_upgrade,
    },
    {
        "key": "cohort_flow", "renderer": "cohort_flow_figure_executor",
        "step_id": "05_visualize_cohort_accounting",
        "input_key": "table:cohort_flow",
        "source_table": "cohort_flow_execute_repair",
        "product": "cohort_flow",
        "png": "sofa2-cohort-flow-rerendered.png",
        "render": run_cohort_flow_figure, "labels": False,
    },
    {
        "key": "data_quality", "renderer": "missingness_measurement_figure_executor",
        "step_id": "06_visualize_data_quality",
        "input_key": "table:missingness_data_quality",
        "source_table": "table_step_artifact_2edb4f312cc279ab",
        "product": "data_quality",
        "png": "sofa2-data-quality-rerendered.png",
        "render": run_measurement_missingness_figure, "labels": True,
        "extra": {"input_key": "table:missingness_data_quality"},
    },
)


def _render_one(run_dir: Path, evidence: dict[str, Any], labels: dict[str, str], spec: dict[str, Any]) -> dict[str, Any]:
    row = evidence.get(spec["source_table"]) or {}
    table = run_dir / str(row.get("relative_path") or "")
    if not row.get("sha256") or not table.is_file():
        raise SystemExit(f"{spec['source_table']} is not a registered table of this run")
    digest = hashlib.sha256(table.read_bytes()).hexdigest()
    if digest != row["sha256"]:
        raise SystemExit(f"{spec['source_table']} does not match its registered digest")
    upgrade: Callable[[pd.DataFrame], tuple[pd.DataFrame, list[str]]] = spec.get(
        "upgrade", lambda frame: (frame, [])
    )
    with tempfile.TemporaryDirectory() as scratch:
        work = Path(scratch).resolve()
        (work / "inputs").mkdir()
        name = table.name.split("__", 1)[-1]
        bound = work / "inputs" / name
        frame, added = upgrade(pd.read_csv(table))
        frame.to_csv(bound, index=False)
        bound_digest = hashlib.sha256(bound.read_bytes()).hexdigest()
        rows = int(len(frame))
        product = spec["input_key"].partition(":")[2]
        manifest = {
            "step_id": spec["step_id"],
            "inputs": {
                spec["input_key"]: {
                    "relative_path": f"inputs/{name}",
                    "sha256": bound_digest,
                    "declared_kind": "table",
                    "evidence_kind": "table",
                    "product": product,
                    "evidence_id": spec["source_table"],
                    "identity_row": {
                        "input_key": spec["input_key"], "declared_kind": "table",
                        "product": product, "evidence_id": spec["source_table"],
                        "sha256": bound_digest,
                    },
                    "product_contract": {"columns": list(frame.columns), "row_count": rows},
                    "consumption_contract": {
                        "input_key": spec["input_key"], "mode": "all_rows",
                        "artifact_sha256": bound_digest, "verified_row_count": rows,
                    },
                }
            },
        }
        out = work / "out"
        options = dict(spec.get("extra") or {})
        if spec["labels"]:
            options["display_labels"] = labels
        with contextlib.redirect_stdout(io.StringIO()):
            spec["render"](
                out_dir=out, run_dir=work, resolved_inputs=manifest,
                step_id=spec["step_id"], figure_product=spec["product"], **options,
            )
        output = ASSETS / spec["png"]
        shutil.copyfile(out / f"{spec['product']}.png", output)
    record = {
        "renderer": spec["renderer"],
        "step_id": spec["step_id"],
        "source_table": spec["source_table"],
        "source_sha256": digest,
        "png": spec["png"],
        "png_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }
    if added:
        record["added_empty_columns"] = added
    return record


def rerender_all(run_dir: Path) -> dict[str, dict[str, Any]]:
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    evidence = {row["evidence_id"]: row for row in manifest["evidence"]}
    plan = json.loads((run_dir / "approved_executable_plan_revision_1.json").read_text(encoding="utf-8"))
    labels = dict(plan["plan_payload"].get("display_labels") or {})
    return {spec["key"]: _render_one(run_dir, evidence, labels, spec) for spec in FIGURES}


if __name__ == "__main__":
    print(json.dumps(rerender_all(Path(sys.argv[1])), indent=2))
