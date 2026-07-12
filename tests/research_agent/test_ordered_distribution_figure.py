"""Case-neutral ordered-distribution figure contract and routing tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from easyicu.research_agent.audits.validators import (
    FigureContractQualityValidator,
    FigureSourceDataValidator,
)
from easyicu.research_agent.figures.ordered_distribution import (
    render_ordered_distribution_bundle_from_prior_outputs,
)
from easyicu.research_agent.evidence import EvidenceStore
from easyicu.research_agent.pipeline import (
    _render_publication_bundle_from_prior_outputs_for_step,
    _resolve_upstream_figure_data_family,
    deterministic_figure_family_supported_for_upstream,
)
from easyicu.research_agent.schema import AnalysisStep


PARENT_STEP = "04_ordered_quality"
FIGURE_STEP = f"{PARENT_STEP}_figure"


def _write_generic_parent(run_dir: Path) -> Path:
    parent = run_dir / "steps" / PARENT_STEP / "outputs"
    parent.mkdir(parents=True, exist_ok=True)
    locked_n = 104
    observed_n = 100
    rows = []
    for level, label, count in (
        (0, "Low severity", 60),
        (1, "Intermediate severity", 30),
        (2, "High severity", 10),
    ):
        rows.append(
            {
                "row_type": "ordered_level",
                "source_status": "valid observed",
                "level": level,
                "level_label": label,
                "n": count,
                "percentage_of_locked_cohort": 100.0 * count / locked_n,
                "fraction_of_locked_cohort": count / locked_n,
                "percentage_of_valid_observed": 100.0 * count / observed_n,
                "fraction_of_valid_observed": count / observed_n,
            }
        )
    for status, count in (
        ("valid observed", 100),
        ("no source", 4),
        ("source present but summary missing", 0),
        ("contradictory or invalid", 0),
    ):
        rows.append(
            {
                "row_type": "availability_status",
                "source_status": status,
                "level": None,
                "level_label": None,
                "n": count,
                "percentage_of_locked_cohort": 100.0 * count / locked_n,
                "fraction_of_locked_cohort": count / locked_n,
                "percentage_of_valid_observed": None,
                "fraction_of_valid_observed": None,
            }
        )
    pd.DataFrame(rows).to_csv(parent / "severity_distribution.csv", index=False)
    (parent / "step_summary.json").write_text(
        json.dumps(
            {
                "method": "source_quality_audit",
                "analysis_family": "association_study",
                "figure_data_family": "ordered_category_distribution",
                "primary_exposure": "severity_band",
                "n_analysis_cohort": locked_n,
                "valid_observed_n": observed_n,
            }
        ),
        encoding="utf-8",
    )
    evidence = EvidenceStore(run_dir)
    table_record = evidence.register_file(
        kind="table",
        description="Ordered category distribution source.",
        source_path=parent / "severity_distribution.csv",
        evidence_id="ordered_distribution_table",
        produced_by_step=PARENT_STEP,
        producer="coder",
        generation_mode="llm",
    )
    summary_record = evidence.register_file(
        kind="statistic",
        description="Ordered category distribution summary.",
        source_path=parent / "step_summary.json",
        evidence_id="ordered_distribution_summary",
        produced_by_step=PARENT_STEP,
        producer="runner",
        generation_mode="llm",
    )
    (run_dir / "manifest_partial.json").write_text(
        json.dumps(
            {
                "per_step_records": [
                    {
                        "step_id": PARENT_STEP,
                        "status": "ok",
                        "evidence_ids": [
                            table_record.evidence_id,
                            summary_record.evidence_id,
                        ],
                        "step_summary_evidence_id": summary_record.evidence_id,
                    }
                ],
                "evidence": [
                    record.model_dump(mode="json") for record in evidence.records()
                ],
            }
        ),
        encoding="utf-8",
    )
    return parent


def _render(run_dir: Path, out_dir: Path) -> str | None:
    return render_ordered_distribution_bundle_from_prior_outputs(
        run_dir=run_dir,
        current_step_id=FIGURE_STEP,
        out_dir=out_dir,
    )


def test_explicit_artifact_contract_precedes_ordered_and_quality_name_tokens(
    tmp_path: Path,
) -> None:
    parent = _write_generic_parent(tmp_path)
    out = tmp_path / "steps" / FIGURE_STEP / "outputs"

    assert (
        _resolve_upstream_figure_data_family(tmp_path, FIGURE_STEP)
        == "ordered_category_distribution"
    )
    assert deterministic_figure_family_supported_for_upstream(tmp_path, FIGURE_STEP)
    repair_id = _render_publication_bundle_from_prior_outputs_for_step(
        run_dir=tmp_path,
        current_step_id=FIGURE_STEP,
        out_dir=out,
    )

    assert repair_id == "ordered_category_distribution_publication_bundle_v1"
    for suffix in ("png", "svg", "pdf", "tiff"):
        assert (out / f"severity_distribution.{suffix}").is_file()
    assert "<text" in (out / "severity_distribution.svg").read_text(encoding="utf-8")

    source_path = out / "severity_distribution_source_data.csv"
    source = pd.read_csv(source_path)
    expected = 100.0 * source["count"] / source["denominator"]
    assert (source["percentage"] - expected).abs().max() < 1e-12
    panel_a = source[source["panel_id"] == "A"]
    panel_b = source[source["panel_id"] == "B"]
    assert set(panel_a["denominator"]) == {100}
    assert set(panel_b["denominator"]) == {104}
    assert panel_a["percentage"].sum() == pytest.approx(100.0)
    assert panel_b["percentage"].sum() == pytest.approx(100.0)
    assert panel_a["source_row_index"].tolist() == [0, 1, 2]
    assert panel_b["source_row_index"].tolist() == [3, 4, 5, 6]

    contract_path = out / "severity_distribution.figure_contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    assert [panel["role"] for panel in contract["panels"]] == [
        "distribution",
        "data_quality",
    ]
    assert (
        FigureContractQualityValidator().audit_contract_file(
            contract_path,
            manuscript_facing=True,
        )
        == []
    )

    summary = json.loads((out / "step_summary.json").read_text(encoding="utf-8"))
    assert summary["valid_observed_n"] == 100
    assert summary["locked_analysis_cohort_n"] == 104
    assert summary["unavailable_n"] == 4
    assert (
        FigureSourceDataValidator().audit(
            step=AnalysisStep(
                step_id=FIGURE_STEP,
                intent=f"Render the figure declared by step '{PARENT_STEP}'.",
            ),
            out_dir=out,
            run_dir=tmp_path,
            step_summary=summary,
        )
        == []
    )
    assert parent.is_dir()


@pytest.mark.parametrize(
    "mutation",
    (
        "valid_status_mismatch",
        "locked_total_mismatch",
        "conditional_percentage_mismatch",
        "locked_percentage_mismatch",
        "duplicate_status_role",
    ),
)
def test_denominator_and_status_invariants_fail_closed(
    tmp_path: Path,
    mutation: str,
) -> None:
    parent = _write_generic_parent(tmp_path)
    table_path = parent / "severity_distribution.csv"
    table = pd.read_csv(table_path)
    if mutation == "valid_status_mismatch":
        table.loc[
            table["source_status"].eq("valid observed") & table["level"].isna(), "n"
        ] = 99
    elif mutation == "locked_total_mismatch":
        table.loc[table["source_status"].eq("no source"), "n"] = 5
    elif mutation == "conditional_percentage_mismatch":
        table.loc[table["level"].eq(0), "percentage_of_valid_observed"] += 1.0
    elif mutation == "locked_percentage_mismatch":
        table.loc[table["level"].eq(0), "percentage_of_locked_cohort"] += 1.0
    elif mutation == "duplicate_status_role":
        table.loc[
            table["source_status"].eq("contradictory or invalid"),
            "source_status",
        ] = "no source"
    table.to_csv(table_path, index=False)

    out = tmp_path / "steps" / FIGURE_STEP / "outputs"
    assert _render(tmp_path, out) is None
    assert not out.exists()


def test_ambiguous_or_incomplete_parent_tables_are_not_claimed(tmp_path: Path) -> None:
    parent = _write_generic_parent(tmp_path)
    table = pd.read_csv(parent / "severity_distribution.csv")
    table.to_csv(parent / "second_ordered_distribution.csv", index=False)
    out = tmp_path / "steps" / FIGURE_STEP / "outputs"
    assert _render(tmp_path, out) is None
    assert not out.exists()

    (parent / "second_ordered_distribution.csv").unlink()
    table = table[table["level"].notna()].copy()
    table.to_csv(parent / "severity_distribution.csv", index=False)
    assert _render(tmp_path, out) is None
    assert not out.exists()


def test_exact_method_adapter_supports_pre_contract_runs(tmp_path: Path) -> None:
    parent = _write_generic_parent(tmp_path)
    summary_path = parent / "step_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.pop("figure_data_family")
    summary["method"] = "ordinal_exposure_derivation_and_quality_control"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    manifest_path = tmp_path / "manifest_partial.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary_evidence = next(
        item
        for item in manifest["evidence"]
        if item["evidence_id"] == "ordered_distribution_summary"
    )
    summary_copy = tmp_path / summary_evidence["relative_path"]
    summary_copy.write_bytes(summary_path.read_bytes())
    summary_evidence["sha256"] = hashlib.sha256(summary_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    assert deterministic_figure_family_supported_for_upstream(tmp_path, FIGURE_STEP)
    out = tmp_path / "steps" / FIGURE_STEP / "outputs"
    assert _render(tmp_path, out) == (
        "ordered_category_distribution_publication_bundle_v1"
    )


def test_ambiguous_explicit_contract_blocks_method_and_name_fallback(
    tmp_path: Path,
) -> None:
    parent = _write_generic_parent(tmp_path)
    summary_path = parent / "step_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["method"] = "ordinal_exposure_derivation_and_quality_control"
    summary["figure_data_contracts"] = [{"family": "another_figure_data_family"}]
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    assert not deterministic_figure_family_supported_for_upstream(tmp_path, FIGURE_STEP)
    out = tmp_path / "steps" / FIGURE_STEP / "outputs"
    assert (
        _render_publication_bundle_from_prior_outputs_for_step(
            run_dir=tmp_path,
            current_step_id=FIGURE_STEP,
            out_dir=out,
        )
        is None
    )
    assert not out.exists()


def test_distribution_contract_cannot_claim_result_bearing_parent(
    tmp_path: Path,
) -> None:
    _write_generic_parent(tmp_path)
    (tmp_path / "manifest_partial.json").write_text(
        json.dumps(
            {
                "per_step_records": [
                    {
                        "step_id": PARENT_STEP,
                        "status": "ok",
                        "analysis_request": {
                            "step": {
                                "method": "ordinal_stratified_descriptive_analysis",
                                "expected_outputs": [
                                    "table:stage_stratified_outcomes",
                                    "test:ordinal_trend",
                                ],
                            }
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    assert not deterministic_figure_family_supported_for_upstream(
        tmp_path, FIGURE_STEP
    )
    out = tmp_path / "steps" / FIGURE_STEP / "outputs"
    assert (
        _render_publication_bundle_from_prior_outputs_for_step(
            run_dir=tmp_path,
            current_step_id=FIGURE_STEP,
            out_dir=out,
        )
        is None
    )
    assert not out.exists()


def test_renderer_and_prompt_remain_case_neutral() -> None:
    import easyicu.research_agent.figures.ordered_distribution as module
    from easyicu.research_agent.prompts import load_prompt_pack

    source = Path(module.__file__).read_text(encoding="utf-8").lower()
    assert "kdigo" not in source
    assert "e3" not in source
    coder_prompt = load_prompt_pack()["coder"]
    assert 'step_summary["figure_data_family"]' in coder_prompt
    assert '"ordered_category_distribution"' in coder_prompt
    assert "do not infer it from a clinical variable name" in coder_prompt
