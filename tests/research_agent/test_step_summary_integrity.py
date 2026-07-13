from __future__ import annotations

from pathlib import Path

import pandas as pd

from easyicu.research_agent.audits import (
    StepSummaryFractionValidator,
    StepSummaryIntegrityValidator,
)
from easyicu.research_agent.schema import AnalysisStep


REFERENCE_KEY = "artifact:reference_rows"
SUBSET_KEY = "artifact:subset_rows"


def _step() -> AnalysisStep:
    return AnalysisStep(step_id="04_verify_subset", intent="Verify a typed subset.")


def _resolved_bindings(
    tmp_path: Path,
    *,
    subset_values: tuple[float, float] = (10.0, 30.0),
    subset_secondary_values: tuple[float, float] = (100.0, 300.0),
) -> dict[str, dict[str, str]]:
    reference_path = tmp_path / "reference.parquet"
    subset_path = tmp_path / "subset.parquet"
    pd.DataFrame(
        {
            "record_id": [1, 2, 3],
            "measurement": [10.0, 20.0, 30.0],
            "secondary_measurement": [100.0, 200.0, 300.0],
        }
    ).to_parquet(reference_path, index=False)
    pd.DataFrame(
        {
            "record_id": [1, 3],
            "measurement": list(subset_values),
            "secondary_measurement": list(subset_secondary_values),
        }
    ).to_parquet(subset_path, index=False)
    return {
        REFERENCE_KEY: {
            "absolute_path": str(reference_path),
            "evidence_id": "table_reference",
            "sha256": "a" * 64,
        },
        SUBSET_KEY: {
            "absolute_path": str(subset_path),
            "evidence_id": "table_subset",
            "sha256": "b" * 64,
        },
    }


def _truthful_summary() -> dict:
    return {
        "input_bindings": [
            {
                "input_key": REFERENCE_KEY,
                "evidence_id": "table_reference",
                "sha256": "a" * 64,
                "loaded": True,
                "row_count": 3,
            },
            {
                "input_key": SUBSET_KEY,
                "evidence_id": "table_subset",
                "sha256": "b" * 64,
                "loaded": True,
                "row_count": 2,
            },
        ],
        "subset_input": {
            "artifact": SUBSET_KEY,
            "loaded": True,
            "n_rows": 2,
        },
        "subset_reconciliation": {
            "status": "checked",
            "reference_artifact": REFERENCE_KEY,
            "subset_artifact": SUBSET_KEY,
            "key_columns": ["record_id"],
            "value_columns_checked": ["measurement", "secondary_measurement"],
            "value_mismatch_n": 0,
        },
    }


def test_integrity_rejects_nested_loaded_and_row_count_contradiction(
    tmp_path: Path,
) -> None:
    summary = _truthful_summary()
    summary["subset_input"].update({"loaded": False, "n_rows": 7})

    findings = StepSummaryIntegrityValidator().audit(
        step=_step(),
        step_summary=summary,
        resolved_input_bindings=_resolved_bindings(tmp_path),
    )

    issues = {finding.detail["issue"] for finding in findings}
    assert "nested_input_declaration_contradiction" in issues
    assert "checked_reconciliation_unloaded_input" in issues


def test_integrity_rejects_checked_reconciliation_without_value_evidence(
    tmp_path: Path,
) -> None:
    summary = _truthful_summary()
    summary["subset_reconciliation"].pop("value_columns_checked")
    summary["subset_reconciliation"].pop("value_mismatch_n")

    findings = StepSummaryIntegrityValidator().audit(
        step=_step(),
        step_summary=summary,
        resolved_input_bindings=_resolved_bindings(
            tmp_path, subset_values=(10.0, 999.0)
        ),
    )

    incomplete = [
        finding
        for finding in findings
        if finding.detail["issue"] == "checked_reconciliation_evidence_incomplete"
    ]
    assert len(incomplete) == 1
    assert set(incomplete[0].detail["invalid_fields"]) == {
        "value_columns_checked",
        "value_mismatch_n",
    }


def test_integrity_rejects_self_comparison_as_subset_reconciliation(
    tmp_path: Path,
) -> None:
    summary = _truthful_summary()
    summary["subset_reconciliation"]["subset_artifact"] = REFERENCE_KEY

    findings = StepSummaryIntegrityValidator().audit(
        step=_step(),
        step_summary=summary,
        resolved_input_bindings=_resolved_bindings(tmp_path),
    )

    incomplete = [
        finding
        for finding in findings
        if finding.detail["issue"] == "checked_reconciliation_evidence_incomplete"
    ]
    assert len(incomplete) == 1
    assert (
        "distinct_reference_and_subset_artifacts"
        in incomplete[0].detail["invalid_fields"]
    )


def test_integrity_host_replay_rejects_same_keys_with_wrong_values(
    tmp_path: Path,
) -> None:
    findings = StepSummaryIntegrityValidator().audit(
        step=_step(),
        step_summary=_truthful_summary(),
        resolved_input_bindings=_resolved_bindings(
            tmp_path, subset_values=(10.0, 999.0)
        ),
    )

    mismatch = [
        finding
        for finding in findings
        if finding.detail["issue"] == "checked_reconciliation_host_value_mismatch"
    ]
    assert len(mismatch) == 1
    assert mismatch[0].detail["value_mismatch_cell_n"] == 1
    assert mismatch[0].detail["value_mismatch_row_n"] == 1


def test_integrity_rejects_a_convenient_subset_of_shared_value_columns(
    tmp_path: Path,
) -> None:
    summary = _truthful_summary()
    summary["subset_reconciliation"]["value_columns_checked"] = ["measurement"]

    findings = StepSummaryIntegrityValidator().audit(
        step=_step(),
        step_summary=summary,
        resolved_input_bindings=_resolved_bindings(tmp_path),
    )

    incomplete = [
        finding
        for finding in findings
        if finding.detail["issue"] == "checked_reconciliation_value_scope_incomplete"
    ]
    assert len(incomplete) == 1
    assert incomplete[0].detail["omitted_value_columns"] == ["secondary_measurement"]


def test_integrity_accepts_truthful_host_verified_reconciliation(
    tmp_path: Path,
) -> None:
    findings = StepSummaryIntegrityValidator().audit(
        step=_step(),
        step_summary=_truthful_summary(),
        resolved_input_bindings=_resolved_bindings(tmp_path),
    )

    assert findings == []


def test_integrity_does_not_claim_unrelated_checked_qc() -> None:
    findings = StepSummaryIntegrityValidator().audit(
        step=_step(),
        step_summary={
            "component_qc": {
                "status": "checked",
                "comparison_n": 100,
                "discordant_n": 0,
            }
        },
        resolved_input_bindings={},
    )

    assert findings == []


def test_integrity_does_not_claim_untyped_component_reconciliation() -> None:
    findings = StepSummaryIntegrityValidator().audit(
        step=_step(),
        step_summary={
            "component_reconciliation": {
                "status": "checked",
                "comparison_n": 100,
                "discordant_n": 0,
            }
        },
        resolved_input_bindings={},
    )

    assert findings == []


def test_non_tabular_input_binding_does_not_require_row_count(tmp_path: Path) -> None:
    figure_path = tmp_path / "figure.svg"
    figure_path.write_text("<svg/>", encoding="utf-8")

    findings = StepSummaryIntegrityValidator().audit(
        step=_step(),
        step_summary={
            "input_bindings": [
                {
                    "input_key": "figure:registered_plot",
                    "evidence_id": "figure_registered",
                    "sha256": "c" * 64,
                    "loaded": True,
                }
            ]
        },
        resolved_input_bindings={
            "figure:registered_plot": {
                "absolute_path": str(figure_path),
                "evidence_id": "figure_registered",
                "sha256": "c" * 64,
                "evidence_kind": "figure",
            }
        },
    )

    assert findings == []


def _measurement_step(*, expected_outputs: list[str] | None = None) -> AnalysisStep:
    return AnalysisStep(
        step_id="04_report_measurement_status",
        intent="Report source-status results from the locked cohort.",
        inputs=["artifact:locked_rows", "signal_measured"],
        expected_outputs=(
            ["table:source_status_summary"]
            if expected_outputs is None
            else expected_outputs
        ),
    )


def _write_measurement_cohort(
    tmp_path: Path,
    *,
    measured: list[object] | None = None,
    counts: list[object] | None = None,
) -> Path:
    measured = [1, 1, 0] if measured is None else measured
    payload: dict[str, list[object]] = {
        "record_id": list(range(1, len(measured) + 1)),
        "signal_measured": measured,
    }
    if counts is not None:
        payload["signal_n"] = counts
    path = tmp_path / "locked_cohort.parquet"
    pd.DataFrame(payload).to_parquet(path, index=False)
    return path


def _measurement_summary(
    *,
    status: str = "checked",
    comparison_n: int | None = 3,
    invalid_pair_n: int | None = 0,
    discordant_n: int | None = 0,
    source: str = "COHORT_PARQUET",
    reason: str | None = None,
    count_column: str = "signal_n",
) -> dict:
    check = {
        "measured_column": "signal_measured",
        "count_column": count_column,
        "status": status,
        "comparison_n": comparison_n,
        "invalid_pair_n": invalid_pair_n,
        "discordant_n": discordant_n,
        "role": "audit_only",
    }
    if reason is not None:
        check["reason"] = reason
    return {
        "source_status_counts": {"observed": 2, "not_observed": 1},
        "measurement_provenance_audit": {
            "source": source,
            "checks": [check],
        },
    }


def test_measurement_provenance_missing_audit_fails_closed(tmp_path: Path) -> None:
    findings = StepSummaryIntegrityValidator().audit(
        step=_measurement_step(),
        step_summary={"source_status_counts": {"observed": 2}},
        resolved_input_bindings={},
        cohort_path=_write_measurement_cohort(tmp_path, counts=[1, 1, 0]),
    )

    assert [finding.detail["issue"] for finding in findings] == [
        "measurement_provenance_source_invalid"
    ]


def test_measurement_provenance_false_zero_discordance_is_rejected(
    tmp_path: Path,
) -> None:
    findings = StepSummaryIntegrityValidator().audit(
        step=_measurement_step(),
        step_summary=_measurement_summary(discordant_n=0),
        resolved_input_bindings={},
        cohort_path=_write_measurement_cohort(tmp_path, counts=[1, 0, 0]),
    )

    mismatch = [
        finding
        for finding in findings
        if finding.detail["issue"] == "measurement_provenance_host_count_mismatch"
    ]
    assert len(mismatch) == 1
    assert mismatch[0].detail["host"]["discordant_n"] == 1


def test_measurement_provenance_rejects_non_cohort_source(tmp_path: Path) -> None:
    findings = StepSummaryIntegrityValidator().audit(
        step=_measurement_step(),
        step_summary=_measurement_summary(source="artifact:convenient_subset"),
        resolved_input_bindings={},
        cohort_path=_write_measurement_cohort(tmp_path, counts=[1, 1, 0]),
    )

    assert [finding.detail["issue"] for finding in findings] == [
        "measurement_provenance_source_invalid"
    ]


def test_measurement_provenance_truthful_checked_passes(tmp_path: Path) -> None:
    findings = StepSummaryIntegrityValidator().audit(
        step=_measurement_step(),
        step_summary=_measurement_summary(),
        resolved_input_bindings={},
        cohort_path=_write_measurement_cohort(tmp_path, counts=[1, 1, 0]),
    )

    assert findings == []


def test_measurement_provenance_resolves_unique_count_column_case_variant(
    tmp_path: Path,
) -> None:
    cohort_path = tmp_path / "case_variant_cohort.parquet"
    pd.DataFrame({"Signal_Measured": [1, 0], "Signal_N": [2, 0]}).to_parquet(
        cohort_path, index=False
    )
    step = _measurement_step().model_copy(
        update={"inputs": ["artifact:locked_rows", "Signal_Measured"]}
    )
    summary = _measurement_summary(comparison_n=2)
    summary["measurement_provenance_audit"]["checks"][0].update(
        {
            "measured_column": "Signal_Measured",
            "count_column": "Signal_N",
        }
    )

    findings = StepSummaryIntegrityValidator().audit(
        step=step,
        step_summary=summary,
        resolved_input_bindings={},
        cohort_path=cohort_path,
    )

    assert findings == []


def test_measurement_provenance_preserves_window_suffix_in_count_pairing(
    tmp_path: Path,
) -> None:
    for measured_column, count_column in (
        ("lactate_measured_24h", "lactate_n_24h"),
        ("novel_signal_measured_6h", "novel_signal_n_6h"),
        ("marker_measured_first_24h", "marker_n_first_24h"),
    ):
        cohort_path = tmp_path / f"{measured_column}.parquet"
        pd.DataFrame({measured_column: [1, 0], count_column: [2, 0]}).to_parquet(
            cohort_path, index=False
        )
        step = _measurement_step().model_copy(
            update={"inputs": ["artifact:locked_rows", measured_column]}
        )
        summary = _measurement_summary(comparison_n=2)
        summary["measurement_provenance_audit"]["checks"][0].update(
            {
                "measured_column": measured_column,
                "count_column": count_column,
            }
        )

        findings = StepSummaryIntegrityValidator().audit(
            step=step,
            step_summary=summary,
            resolved_input_bindings={},
            cohort_path=cohort_path,
        )

        assert findings == []


def test_measurement_provenance_truthful_unavailable_passes(tmp_path: Path) -> None:
    findings = StepSummaryIntegrityValidator().audit(
        step=_measurement_step(),
        step_summary=_measurement_summary(
            status="unavailable",
            comparison_n=None,
            invalid_pair_n=None,
            discordant_n=None,
            reason="Companion count column is absent from the locked cohort.",
        ),
        resolved_input_bindings={},
        cohort_path=_write_measurement_cohort(tmp_path),
    )

    assert findings == []


def test_measurement_provenance_unavailable_still_checks_measured_values(
    tmp_path: Path,
) -> None:
    findings = StepSummaryIntegrityValidator().audit(
        step=_measurement_step(),
        step_summary=_measurement_summary(
            status="unavailable",
            comparison_n=None,
            invalid_pair_n=None,
            discordant_n=None,
            reason="Companion count column is absent from the locked cohort.",
        ),
        resolved_input_bindings={},
        cohort_path=_write_measurement_cohort(
            tmp_path,
            measured=[0, 2, None, float("inf")],
        ),
    )

    invalid = [
        finding
        for finding in findings
        if finding.detail["issue"] == "measurement_provenance_invalid_measured_values"
    ]
    assert len(invalid) == 1
    assert invalid[0].detail["invalid_measured_n"] == 3


def test_measurement_provenance_rejects_semantic_count_dtypes(
    tmp_path: Path,
) -> None:
    for suffix, counts in (
        ("bool", pd.Series([True, False], dtype=bool)),
        ("datetime", pd.to_datetime([1, 0], unit="ns")),
    ):
        cohort_path = tmp_path / f"semantic_count_{suffix}.parquet"
        pd.DataFrame({"signal_measured": [1, 0], "signal_n": counts}).to_parquet(
            cohort_path, index=False
        )
        findings = StepSummaryIntegrityValidator().audit(
            step=_measurement_step(),
            step_summary=_measurement_summary(
                comparison_n=0,
                invalid_pair_n=2,
                discordant_n=0,
            ),
            resolved_input_bindings={},
            cohort_path=cohort_path,
        )

        assert any(
            finding.detail["issue"] == "measurement_provenance_invalid_pairs"
            for finding in findings
        )


def test_measurement_provenance_rejects_datetime_measured_dtype(
    tmp_path: Path,
) -> None:
    cohort_path = tmp_path / "datetime_measured.parquet"
    pd.DataFrame(
        {
            "signal_measured": pd.to_datetime([0, 1], unit="ns"),
            "signal_n": [0, 1],
        }
    ).to_parquet(cohort_path, index=False)

    findings = StepSummaryIntegrityValidator().audit(
        step=_measurement_step(),
        step_summary=_measurement_summary(
            comparison_n=0,
            invalid_pair_n=2,
            discordant_n=0,
        ),
        resolved_input_bindings={},
        cohort_path=cohort_path,
    )

    assert any(
        finding.detail["issue"] == "measurement_provenance_invalid_measured_values"
        for finding in findings
    )


def test_locked_measurement_preflight_blocks_missing_or_ambiguous_host_columns(
    tmp_path: Path,
) -> None:
    missing_path = tmp_path / "missing_measured.parquet"
    pd.DataFrame({"signal_n": [0, 1]}).to_parquet(missing_path, index=False)
    missing = StepSummaryIntegrityValidator.audit_locked_measurement_data_quality(
        step=_measurement_step(),
        cohort_path=missing_path,
    )
    assert [finding.detail["issue"] for finding in missing] == [
        "measurement_provenance_measured_column_missing"
    ]

    ambiguous_path = tmp_path / "ambiguous_count.parquet"
    pd.DataFrame(
        {
            "signal_measured": [0, 1],
            "signal_N": [0, 1],
            "SIGNAL_N": [0, 1],
        }
    ).to_parquet(ambiguous_path, index=False)
    ambiguous = StepSummaryIntegrityValidator.audit_locked_measurement_data_quality(
        step=_measurement_step(),
        cohort_path=ambiguous_path,
    )
    assert [finding.detail["issue"] for finding in ambiguous] == [
        "measurement_provenance_count_column_ambiguous"
    ]


def test_measurement_provenance_unavailable_requires_explicit_null_counts(
    tmp_path: Path,
) -> None:
    summary = _measurement_summary(
        status="unavailable",
        comparison_n=None,
        invalid_pair_n=None,
        discordant_n=None,
        reason="Companion count column is absent from the locked cohort.",
    )
    summary["measurement_provenance_audit"]["checks"][0].pop("invalid_pair_n")

    findings = StepSummaryIntegrityValidator().audit(
        step=_measurement_step(),
        step_summary=summary,
        resolved_input_bindings={},
        cohort_path=_write_measurement_cohort(tmp_path),
    )

    invalid = [
        finding
        for finding in findings
        if finding.detail["issue"] == "measurement_provenance_check_invalid"
    ]
    assert len(invalid) == 1
    assert invalid[0].detail["invalid_fields"] == ["invalid_pair_n"]


def test_measurement_provenance_rejects_false_unavailable(tmp_path: Path) -> None:
    findings = StepSummaryIntegrityValidator().audit(
        step=_measurement_step(),
        step_summary=_measurement_summary(
            status="unavailable",
            comparison_n=None,
            invalid_pair_n=None,
            discordant_n=None,
            reason="Count was not loaded by generated code.",
        ),
        resolved_input_bindings={},
        cohort_path=_write_measurement_cohort(tmp_path, counts=[1, 0, 0]),
    )

    assert any(
        finding.detail["issue"] == "measurement_provenance_unavailable_contradicted"
        for finding in findings
    )


def test_measurement_provenance_rejects_wrong_companion_column(
    tmp_path: Path,
) -> None:
    findings = StepSummaryIntegrityValidator().audit(
        step=_measurement_step(),
        step_summary=_measurement_summary(count_column="other_n"),
        resolved_input_bindings={},
        cohort_path=_write_measurement_cohort(tmp_path, counts=[1, 0, 0]),
    )

    invalid = [
        finding
        for finding in findings
        if finding.detail["issue"] == "measurement_provenance_check_invalid"
    ]
    assert len(invalid) == 1
    assert invalid[0].detail["invalid_fields"] == ["count_column"]


def test_measurement_provenance_ignores_convenient_resolved_subset(
    tmp_path: Path,
) -> None:
    cohort_path = _write_measurement_cohort(tmp_path, counts=[1, 0, 0])
    subset_path = tmp_path / "convenient_subset.parquet"
    pd.DataFrame({"signal_measured": [1], "signal_n": [1]}).to_parquet(
        subset_path, index=False
    )
    findings = StepSummaryIntegrityValidator().audit(
        step=_measurement_step(),
        step_summary=_measurement_summary(comparison_n=1, discordant_n=0),
        resolved_input_bindings={
            "artifact:convenient_subset": {"absolute_path": str(subset_path)}
        },
        cohort_path=cohort_path,
    )

    mismatch = [
        finding
        for finding in findings
        if finding.detail["issue"] == "measurement_provenance_host_count_mismatch"
    ]
    assert len(mismatch) == 1
    assert mismatch[0].detail["host"] == {
        "comparison_n": 3,
        "invalid_pair_n": 0,
        "discordant_n": 1,
    }


def test_measurement_provenance_blocks_truthfully_reported_invalid_pairs(
    tmp_path: Path,
) -> None:
    findings = StepSummaryIntegrityValidator().audit(
        step=_measurement_step(),
        step_summary=_measurement_summary(
            comparison_n=1, invalid_pair_n=3, discordant_n=0
        ),
        resolved_input_bindings={},
        cohort_path=_write_measurement_cohort(
            tmp_path,
            measured=[1, 2, None, 0],
            counts=[1, 1, -1, None],
        ),
    )

    invalid = [
        finding
        for finding in findings
        if finding.detail["issue"] == "measurement_provenance_invalid_pairs"
    ]
    assert len(invalid) == 1
    assert invalid[0].detail["invalid_pair_n"] == 3


def test_measurement_provenance_blocks_truthfully_reported_discordance(
    tmp_path: Path,
) -> None:
    findings = StepSummaryIntegrityValidator().audit(
        step=_measurement_step(),
        step_summary=_measurement_summary(discordant_n=1),
        resolved_input_bindings={},
        cohort_path=_write_measurement_cohort(tmp_path, counts=[1, 0, 0]),
    )

    discordance = [
        finding
        for finding in findings
        if finding.detail["issue"] == "measurement_provenance_count_flag_discordance"
    ]
    assert len(discordance) == 1
    assert discordance[0].detail["discordant_n"] == 1


def test_measurement_provenance_normalises_boolean_and_numeric_string_pairs(
    tmp_path: Path,
) -> None:
    for measured, counts in (
        ([True, False], [1, 0]),
        (["1", "0"], ["2", "0"]),
    ):
        findings = StepSummaryIntegrityValidator().audit(
            step=_measurement_step(),
            step_summary=_measurement_summary(
                comparison_n=2, invalid_pair_n=0, discordant_n=0
            ),
            resolved_input_bindings={},
            cohort_path=_write_measurement_cohort(
                tmp_path,
                measured=measured,
                counts=counts,
            ),
        )

        assert findings == []


def test_measurement_provenance_requires_every_planned_flag(tmp_path: Path) -> None:
    cohort_path = _write_measurement_cohort(tmp_path, counts=[1, 0, 0])
    frame = pd.read_parquet(cohort_path)
    frame["secondary_measured"] = [1, 0, 0]
    frame["secondary_n"] = [1, 0, 0]
    frame.to_parquet(cohort_path, index=False)
    step = _measurement_step()
    step = step.model_copy(update={"inputs": [*step.inputs, "secondary_measured"]})

    findings = StepSummaryIntegrityValidator().audit(
        step=step,
        step_summary=_measurement_summary(),
        resolved_input_bindings={},
        cohort_path=cohort_path,
    )

    missing = [
        finding
        for finding in findings
        if finding.detail["issue"] == "measurement_provenance_check_missing"
    ]
    assert len(missing) == 1
    assert missing[0].detail["measured_column"] == "secondary_measured"


def test_measurement_provenance_non_result_step_does_not_trigger(
    tmp_path: Path,
) -> None:
    findings = StepSummaryIntegrityValidator().audit(
        step=_measurement_step(expected_outputs=[]),
        step_summary={"distribution": {"n": 3}},
        resolved_input_bindings={},
        cohort_path=_write_measurement_cohort(tmp_path, counts=[1, 0, 0]),
    )

    assert findings == []


def test_measurement_provenance_component_qc_cannot_bypass_host_replay(
    tmp_path: Path,
) -> None:
    cohort_path = _write_measurement_cohort(tmp_path, counts=[1, 1, 0])
    for expected_outputs in (
        ["table:signal_component_qc"],
        ["table:outcome_incidence", "table:signal_component_qc"],
    ):
        step = _measurement_step(expected_outputs=expected_outputs)
        missing = StepSummaryIntegrityValidator().audit(
            step=step,
            step_summary={"component_qc": {"count_flag_discordant_n": 0}},
            resolved_input_bindings={},
            cohort_path=cohort_path,
        )
        assert any(
            finding.detail["issue"] == "measurement_provenance_source_invalid"
            for finding in missing
        )

        truthful = StepSummaryIntegrityValidator().audit(
            step=step,
            step_summary=_measurement_summary(),
            resolved_input_bindings={},
            cohort_path=cohort_path,
        )
        assert truthful == []


def test_measurement_provenance_render_only_step_does_not_trigger(
    tmp_path: Path,
) -> None:
    for expected_outputs in (
        ["figure:source_status_plot"],
        ["publication_figure"],
        ["forest_plot"],
        ["missingness_heatmap"],
        ["figure:publication_figure", "log:figure_contract"],
        ["log:analysis_protocol"],
    ):
        findings = StepSummaryIntegrityValidator().audit(
            step=_measurement_step(expected_outputs=expected_outputs),
            step_summary={"source_status_counts": {"observed": 2}},
            resolved_input_bindings={},
            cohort_path=_write_measurement_cohort(tmp_path, counts=[1, 1, 0]),
        )

        assert findings == []


def test_measurement_provenance_mixed_result_and_render_outputs_trigger(
    tmp_path: Path,
) -> None:
    findings = StepSummaryIntegrityValidator().audit(
        step=_measurement_step(
            expected_outputs=[
                "table:source_status_summary",
                "figure:source_status_plot",
                "log:figure_contract",
            ]
        ),
        step_summary={"source_status_counts": {"observed": 2}},
        resolved_input_bindings={},
        cohort_path=_write_measurement_cohort(tmp_path, counts=[1, 1, 0]),
    )

    assert [finding.detail["issue"] for finding in findings] == [
        "measurement_provenance_source_invalid"
    ]


def test_measurement_provenance_manifest_result_triggers(tmp_path: Path) -> None:
    for result_output in (
        "manifest:cluster_selection",
        "contract:effect_estimates",
        "protocol:result_table",
        "figure_source_data",
        "publication_figure_skill_summary",
        "model_plot_statistics",
    ):
        findings = StepSummaryIntegrityValidator().audit(
            step=_measurement_step(expected_outputs=[result_output]),
            step_summary={"result": {"status": "complete"}},
            resolved_input_bindings={},
            cohort_path=_write_measurement_cohort(tmp_path, counts=[1, 1, 0]),
        )

        assert [finding.detail["issue"] for finding in findings] == [
            "measurement_provenance_source_invalid"
        ]


def test_probability_ci_roundoff_overflow_is_not_silently_accepted() -> None:
    findings = StepSummaryFractionValidator().audit(
        step=_step(),
        step_summary={"prevalence_ci_high": 1.0000000000000002},
    )

    assert len(findings) == 1
    assert findings[0].severity == "error"
    assert findings[0].detail["issue"] == "bounded_metric_out_of_range"
    assert findings[0].detail["metric_kind"] == "prevalence"
    assert findings[0].detail["roundoff_sized_overflow"] is True


def test_nested_risk_ci_roundoff_overflow_is_not_silently_accepted() -> None:
    findings = StepSummaryFractionValidator().audit(
        step=_step(),
        step_summary={
            "risk": {
                "estimate": 0.2,
                "ci_low": 0.1,
                "ci_high": 1.0000000000000002,
            }
        },
    )

    assert len(findings) == 1
    assert findings[0].detail["summary_path"] == "risk.ci_high"
    assert findings[0].detail["metric_kind"] == "risk"


def test_effect_scale_estimates_and_intervals_are_not_probability_bounded() -> None:
    findings = StepSummaryFractionValidator().audit(
        step=_step(),
        step_summary={
            "risk_ratio": 1.8,
            "risk_ratio_ci_low": 1.4,
            "risk_ratio_ci_high": 2.2,
            "relative_risk": 1.8,
            "relative_risk_ci_high": 2.2,
            "odds_ratio": 2.4,
            "hazard_ratio": 1.7,
            "risk_difference": -0.2,
            "number_at_risk": 120,
            "patients_at_risk": 118,
            "at_risk": "reported",
        },
    )

    assert findings == []


def test_generic_interval_is_not_assigned_to_risk_when_effect_scale_is_present() -> (
    None
):
    findings = StepSummaryFractionValidator().audit(
        step=_step(),
        step_summary={
            "absolute_risk": 0.2,
            "risk_ratio": 1.8,
            "ci_low": 1.4,
            "ci_high": 2.2,
        },
    )

    assert findings == []
