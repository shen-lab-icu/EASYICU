"""A declared parent name must resolve by identity, not by one spelling.

Measured on a real run (2026-07-29): a figure whose source-data rows reconcile
value-for-value against *both* of its digest-verified parents was refused with
``declared source table event_timing_audit was not found in an upstream step``
-- while ``event_timing_audit.csv`` sat in the parent step's outputs and was
bound to the figure step under ``table:event_timing_audit``.

The host names one bound artifact four ways: the typed input key it was
declared under, the typed product id inside that key, the file the producing
step wrote, and the evidence copy taken of it.  Three of the four resolved; the
bare product id did not.  Replaying the real step with only the ``.csv`` suffix
appended to ``source_table`` turned three errors into zero, so the spelling was
the whole of it.

Widening the vocabulary is not widening the authority.  An alias is a *filter*
over tables this step already bound: it can never introduce a table the step
did not bind, a name that resolves to more than one artifact is still refused
as ambiguous lineage, and every matched row is still value-verified against
the parent's bytes.  The negative tests below pin each of those.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from easyicu.research_agent.audits.validators import FigureSourceDataValidator
from easyicu.research_agent.figures.publication import make_figure_contract
from easyicu.research_agent.schema import AnalysisStep

PARENT_STEP = "05_missingness_event_timing_audit"
FIGURE_STEP = "05_missingness_event_timing_audit_figure"
FIGURE_PRODUCT = "event_timing_diagnostic"

# The real two-parent shape, one row kind each, trimmed to what the comparison
# needs: a stable identifier plus values that must match exactly.
_PARENT_TABLES: dict[str, pd.DataFrame] = {
    "event_timing_audit": pd.DataFrame(
        [
            {"variable": "susp_inf", "metric": "first_time_h", "count": 533},
            {"variable": "sep3_sofa2", "metric": "first_time_h", "count": 486},
        ]
    ),
    "measurement_process_audit": pd.DataFrame(
        [
            {"variable": "susp_inf", "metric": "measurement_status", "count": 533},
            {"variable": "sofa2_renal", "metric": "measurement_status", "count": 912},
        ]
    ),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_source_data(
    frame: pd.DataFrame,
    *,
    path: Path,
    declared_name: str,
) -> None:
    """Write a faithful positional projection naming its parent as given."""

    export = frame.copy()
    export["source_row_index"] = range(len(export))
    export["source_table"] = declared_name
    export.to_csv(path, index=False)


def _build_run(
    tmp_path: Path,
    *,
    declared_names: dict[str, str],
    mutate: Any = None,
    parent_step: str = PARENT_STEP,
) -> tuple[Path, Path, dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """Materialise a minimal but real-shaped run for the figure validator."""

    run_dir = tmp_path / "run"
    evidence_dir = run_dir / "evidence"
    parent_out = run_dir / "steps" / parent_step / "outputs"
    figure_out = run_dir / "steps" / FIGURE_STEP / "outputs"
    for directory in (evidence_dir, parent_out, figure_out):
        directory.mkdir(parents=True, exist_ok=True)

    evidence_records: list[dict[str, Any]] = []
    bindings: dict[str, Any] = {}
    for product, frame in _PARENT_TABLES.items():
        output_path = parent_out / f"{product}.csv"
        frame.to_csv(output_path, index=False)
        digest = _sha256(output_path)
        evidence_id = f"table_step_artifact_{product}"
        evidence_path = evidence_dir / f"{evidence_id}__{product}.csv"
        evidence_path.write_bytes(output_path.read_bytes())
        evidence_records.append(
            {
                "evidence_id": evidence_id,
                "kind": "table",
                "relative_path": f"evidence/{evidence_path.name}",
                "sha256": digest,
                "produced_by_step": parent_step,
            }
        )
        bindings[f"table:{product}"] = {
            "declared_kind": "table",
            "evidence_kind": "table",
            "product": product,
            "evidence_id": evidence_id,
            "produced_by_step": parent_step,
            "sha256": digest,
            "relative_path": f"evidence/{evidence_path.name}",
            "absolute_path": str(evidence_path),
        }

        exported = frame if mutate is None else mutate(product, frame)
        _write_source_data(
            exported,
            path=figure_out / f"{FIGURE_PRODUCT}_{product}_source_data.csv",
            declared_name=declared_names[product],
        )

    (figure_out / f"{FIGURE_PRODUCT}.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    (figure_out / f"{FIGURE_PRODUCT}.figure_contract.json").write_text(
        make_figure_contract(
            figure_id=f"figure:{FIGURE_PRODUCT}",
            core_claim=(
                "Audited availability and event-timing context are rendered "
                "from the registered parent audit tables."
            ),
            archetype="quantitative_grid",
            width_mm=183.0,
            height_mm=90.0,
            panels=[
                {
                    "panel_id": panel_id,
                    "title": "Measurement process",
                    "role": "data_quality",
                    "claim": "Parent-reported counts, unaltered.",
                    "evidence_ids": [f"{FIGURE_PRODUCT}_{product}_source_data.csv"],
                    "metadata": {
                        "chart_type": "horizontal_bar",
                        "source_data": [f"{FIGURE_PRODUCT}_{product}_source_data.csv"],
                    },
                }
                for panel_id, product in zip("AB", _PARENT_TABLES)
            ],
            source_data=[
                f"{FIGURE_PRODUCT}_{product}_source_data.csv"
                for product in _PARENT_TABLES
            ],
        ).model_dump_json(),
        encoding="utf-8",
    )
    summary = {
        "status": "success",
        "rendering_only": True,
        "figure_files": [f"{FIGURE_PRODUCT}.png"],
        "output_files": {f"figure:{FIGURE_PRODUCT}": f"{FIGURE_PRODUCT}.png"},
        "source_data_files": [
            f"{FIGURE_PRODUCT}_{product}_source_data.csv" for product in _PARENT_TABLES
        ],
    }
    (figure_out / "step_summary.json").write_text(json.dumps(summary), encoding="utf-8")

    records = [
        {
            "step_id": parent_step,
            "status": "ok",
            "evidence_ids": [record["evidence_id"] for record in evidence_records],
        }
    ]
    (run_dir / "manifest.json").write_text(
        json.dumps({"evidence": evidence_records, "per_step_records": records}),
        encoding="utf-8",
    )
    return run_dir, figure_out, bindings, records, summary


def _figure_step(inputs: list[str] | None = None) -> AnalysisStep:
    declared = inputs or [f"table:{product}" for product in _PARENT_TABLES]
    return AnalysisStep.model_validate(
        {
            "step_id": FIGURE_STEP,
            "intent": "Render the audited availability and timing panels.",
            "method": "visualization",
            "inputs": declared,
            "expected_outputs": [f"figure:{FIGURE_PRODUCT}"],
            "planned_analysis_role": "auxiliary",
            "input_consumption_contracts": [
                {
                    "schema_version": "easyicu.artifact_consumption/1",
                    "input_key": key,
                    "mode": "all_rows",
                    "role_column": None,
                    "expected_roles": [],
                }
                for key in declared
            ],
        }
    )


def _audit(
    run_dir: Path,
    figure_out: Path,
    bindings: dict[str, Any],
    records: list[dict[str, Any]],
    summary: dict[str, Any],
    *,
    step: AnalysisStep | None = None,
) -> list:
    return FigureSourceDataValidator().audit(
        step=step or _figure_step(),
        out_dir=figure_out,
        run_dir=run_dir,
        step_summary=summary,
        completed_step_records=records,
        resolved_input_bindings=bindings,
    )


def test_the_typed_product_id_names_its_bound_parent(tmp_path: Path) -> None:
    """The measured false rejection: the product id is a name for the artifact."""

    run_dir, figure_out, bindings, records, summary = _build_run(
        tmp_path,
        declared_names={product: product for product in _PARENT_TABLES},
    )

    findings = _audit(run_dir, figure_out, bindings, records, summary)

    assert not findings, [(finding.message, finding.detail) for finding in findings]


@pytest.mark.parametrize(
    "spelling",
    [
        pytest.param(lambda product: product, id="product_id"),
        pytest.param(lambda product: f"{product}.csv", id="output_basename"),
        pytest.param(lambda product: f"table:{product}", id="typed_input_key"),
        pytest.param(
            lambda product: f"table_step_artifact_{product}__{product}.csv",
            id="evidence_copy_name",
        ),
    ],
)
def test_every_name_the_host_itself_uses_resolves(
    tmp_path: Path, spelling: Any
) -> None:
    """One artifact, several host-published names, one verdict.

    Accepting a proper subset of them makes the verdict depend on which
    spelling the producer happened to pick, which is not a property of the
    figure.
    """

    run_dir, figure_out, bindings, records, summary = _build_run(
        tmp_path,
        declared_names={product: spelling(product) for product in _PARENT_TABLES},
    )

    findings = _audit(run_dir, figure_out, bindings, records, summary)

    assert not findings, [(finding.message, finding.detail) for finding in findings]


def test_a_forged_value_under_the_product_id_is_still_rejected(
    tmp_path: Path,
) -> None:
    """The alias routes the comparison; it must not replace it."""

    def _forge(product: str, frame: pd.DataFrame) -> pd.DataFrame:
        if product != "event_timing_audit":
            return frame
        forged = frame.copy()
        forged.loc[0, "count"] = 99999
        return forged

    run_dir, figure_out, bindings, records, summary = _build_run(
        tmp_path,
        declared_names={product: product for product in _PARENT_TABLES},
        mutate=_forge,
    )

    findings = _audit(run_dir, figure_out, bindings, records, summary)

    assert findings
    reasons = {
        (finding.detail.get("best_mismatch") or {}).get("reason")
        for finding in findings
    }
    assert "source_values_disagree" in reasons


def test_a_product_id_this_step_never_bound_is_still_not_found(
    tmp_path: Path,
) -> None:
    """An alias may only name a table the step actually bound.

    This is what keeps the widening honest: the vocabulary grows, the set of
    reachable parents does not.
    """

    run_dir, figure_out, bindings, records, summary = _build_run(
        tmp_path,
        declared_names={
            "event_timing_audit": "some_other_teams_audit",
            "measurement_process_audit": "measurement_process_audit",
        },
    )

    findings = _audit(run_dir, figure_out, bindings, records, summary)

    mismatches = [finding.detail.get("best_mismatch") or {} for finding in findings]
    assert any(
        item.get("reason") == "declared_source_table_not_found"
        and item.get("declared_source_table") == "some_other_teams_audit"
        for item in mismatches
    ), [finding.detail for finding in findings]


def test_a_name_resolving_to_two_bound_artifacts_is_refused(
    tmp_path: Path,
) -> None:
    """Ambiguity stays a refusal, and says so rather than picking a winner.

    Both parents are declared under the *same* name. Resolving that to one of
    them arbitrarily would authenticate a figure against a table it did not
    claim, which is exactly the laundering the declaration exists to prevent.
    """

    run_dir, figure_out, bindings, records, summary = _build_run(
        tmp_path,
        declared_names={product: "shared_audit_name" for product in _PARENT_TABLES},
    )
    # Give the shared spelling two distinct bound artifacts to resolve to.
    for product in _PARENT_TABLES:
        bindings[f"table:{product}"]["product"] = "shared_audit_name"
    bindings = {
        "table:shared_audit_name": bindings["table:event_timing_audit"],
        "table:measurement_process_audit": bindings["table:measurement_process_audit"],
    }
    bindings["table:measurement_process_audit"]["product"] = "measurement_process_audit"

    findings = _audit(
        run_dir,
        figure_out,
        bindings,
        records,
        summary,
        step=_figure_step(
            inputs=["table:shared_audit_name", "table:measurement_process_audit"]
        ),
    )

    assert findings, "two artifacts under one declared name were silently accepted"
