"""Outcome-blind real-data concept feasibility probes."""

from __future__ import annotations

import importlib
from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError


def _availability_module():
    return importlib.import_module("easyicu.research_agent.concept_availability")


def _allow_all_concepts(monkeypatch: pytest.MonkeyPatch):
    ca = _availability_module()

    def fake_explain_concept_availability(*, concept, database, requested_concept=None):
        canonical = ca.normalize_concept_name(concept)
        return ca.ConceptDatabaseAvailability(
            concept=canonical,
            requested_concept=requested_concept or concept,
            database=ca.normalize_database_name(database),
            status="full",
            available=True,
            direct_source=True,
            reason="direct_source_available",
        )

    monkeypatch.setattr(ca, "explain_concept_availability", fake_explain_concept_availability)
    return ca


def _write_joint_fixture(tmp_path: Path) -> Path:
    path = tmp_path / "joint_fixture.parquet"
    pd.DataFrame(
        {
            "stay_id": [1, 2, 3, 4, 5, 6],
            "group": ["a", "a", "a", "b", "b", "b"],
            "crea": [1.0, None, 1.2, 1.3, None, 1.4],
            "map": [None, 70.0, 80.0, None, 90.0, 100.0],
            "death": [0, 1, 1, 0, 1, 0],
        }
    ).to_parquet(path, index=False)
    return path


def test_real_data_feasibility_model_validates_counts(ra):
    record = ra.RealDataConceptFeasibility(
        concept="crea",
        database="miiv",
        analytic_unit="stay",
        denominator_n=4,
        n_present=3,
        fraction_missing=0.25,
        n_joint_complete=2,
        joint_fraction_complete=0.50,
        missingness_severity="medium",
    )

    assert "outcome_rate" in record.non_outcome_blind_fields_checked
    dumped = record.model_dump()
    assert "outcome_rate" not in dumped.keys()
    assert "stratified_outcome" not in dumped.keys()

    with pytest.raises(ValidationError, match="n_present cannot exceed"):
        ra.RealDataConceptFeasibility(
            concept="crea",
            database="miiv",
            denominator_n=1,
            n_present=2,
            fraction_missing=0.0,
            n_joint_complete=0,
            joint_fraction_complete=0.0,
            missingness_severity="low",
        )


def test_real_data_feasibility_reports_single_and_joint_completeness(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    ca = _allow_all_concepts(monkeypatch)
    path = _write_joint_fixture(tmp_path)

    result = ca.real_data_concept_feasibility(
        ["crea", "map"],
        "miiv",
        path,
        analytic_unit="stay",
    )

    crea = result["crea"]
    mean_arterial_pressure = result["map"]
    assert crea.denominator_n == 6
    assert crea.n_present == 4
    assert crea.fraction_missing == pytest.approx(1.0 / 3.0)
    assert mean_arterial_pressure.n_present == 4
    assert crea.n_joint_complete == 2
    assert crea.joint_fraction_complete == pytest.approx(1.0 / 3.0)
    assert crea.joint_fraction_complete < (1.0 - crea.fraction_missing)
    assert "outcome_rate" in crea.non_outcome_blind_fields_checked
    assert "death" not in crea.model_dump().keys()


def _write_contrast_fixture(tmp_path: Path) -> Path:
    path = tmp_path / "contrast_fixture.parquet"
    pd.DataFrame(
        {
            "stay_id": [1, 2, 3, 4, 5, 6],
            "vaso": [1, 1, 1, 1, 1, 1],  # single-valued exposure: no contrast
            "crea": [1.0, 1.2, 0.9, 1.5, 2.0, 1.1],  # varying exposure
            "death": [0, 1, 1, 0, 1, 0],
        }
    ).to_parquet(path, index=False)
    return path


def test_exposure_contrast_computed_only_for_requested_predictor(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    ca = _allow_all_concepts(monkeypatch)
    path = _write_contrast_fixture(tmp_path)

    result = ca.real_data_concept_feasibility(
        ["crea", "death"],
        "miiv",
        path,
        analytic_unit="stay",
        contrast_concepts=["crea"],
    )

    # The predictor gets an exposure-contrast value; the outcome NEVER does, so
    # its modal share (== event rate) cannot leak through the outcome-blind guard.
    assert result["crea"].value_contrast_fraction == pytest.approx(1.0 - 1.0 / 6.0)
    assert result["death"].value_contrast_fraction is None


def test_single_valued_exposure_reports_zero_contrast(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    ca = _allow_all_concepts(monkeypatch)
    path = _write_contrast_fixture(tmp_path)

    result = ca.real_data_concept_feasibility(
        ["vaso", "death"],
        "miiv",
        path,
        analytic_unit="stay",
        contrast_concepts=["vaso"],
    )

    assert result["vaso"].value_contrast_fraction == 0.0


def test_contrast_is_not_computed_without_optin(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    ca = _allow_all_concepts(monkeypatch)
    path = _write_contrast_fixture(tmp_path)

    result = ca.real_data_concept_feasibility(
        ["crea", "death"],
        "miiv",
        path,
        analytic_unit="stay",
    )

    assert result["crea"].value_contrast_fraction is None


def test_event_default_false_concepts_treat_sparse_nan_as_observed_false(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    ca = _allow_all_concepts(monkeypatch)
    path = tmp_path / "event_default_false_fixture.parquet"
    pd.DataFrame(
        {
            "stay_id": [1, 2, 3, 4, 5],
            "peep": [8.0, 10.0, None, 12.0, 14.0],
            # Sparse positive event export: NaN encodes no AKI, not unobserved AKI.
            "aki": [1.0, None, None, None, 1.0],
            "lact": [1.2, None, 2.4, None, 3.1],
        }
    ).to_parquet(path, index=False)

    result = ca.real_data_concept_feasibility(
        ["peep", "aki"],
        "miiv",
        path,
    )

    peep = result["peep"]
    aki = result["aki"]
    assert peep.denominator_n == 5
    assert peep.n_present == 4
    assert aki.concept == "kdigo_aki"
    assert aki.n_present == 5
    assert aki.fraction_missing == 0.0
    assert peep.n_joint_complete == 4
    assert peep.joint_fraction_complete == pytest.approx(4.0 / 5.0)

    dumped = aki.model_dump()
    assert "outcome_rate" not in dumped
    assert "event_rate" not in dumped
    assert "effect_estimate" not in dumped


def test_canonical_concept_resolves_exported_alias_column(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    ca = _allow_all_concepts(monkeypatch)
    path = tmp_path / "canonical_alias_fixture.parquet"
    pd.DataFrame(
        {
            "stay_id": [1, 2, 3, 4, 5],
            "peep": [8.0, 10.0, None, 12.0, 14.0],
            # The prepared wide table may carry the original short column while
            # the idea resolver requests the canonical concept name.
            "aki": [1.0, None, None, None, 1.0],
        }
    ).to_parquet(path, index=False)

    result = ca.real_data_concept_feasibility(
        ["peep", "kdigo_aki"],
        "miiv",
        path,
    )

    kdigo = result["kdigo_aki"]
    assert kdigo.concept == "kdigo_aki"
    assert kdigo.n_present == 5
    assert kdigo.fraction_missing == 0.0
    assert kdigo.note is None
    assert result["peep"].n_joint_complete == 4
    assert result["peep"].joint_fraction_complete == pytest.approx(4.0 / 5.0)


def test_available_concept_missing_export_column_records_resolution_note(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    ca = _allow_all_concepts(monkeypatch)
    path = tmp_path / "missing_alias_column_fixture.parquet"
    pd.DataFrame(
        {
            "stay_id": [1, 2, 3],
            "peep": [8.0, 10.0, 12.0],
        }
    ).to_parquet(path, index=False)

    result = ca.real_data_concept_feasibility(
        ["peep", "kdigo_aki"],
        "miiv",
        path,
    )

    kdigo = result["kdigo_aki"]
    assert kdigo.availability_status == "full"
    assert kdigo.n_present == 0
    assert kdigo.n_joint_complete == 0
    assert "no matching exported wide-table column" in (kdigo.note or "")
    assert "aki" in (kdigo.note or "")
    dumped = kdigo.model_dump()
    assert "outcome_rate" not in dumped
    assert "effect_estimate" not in dumped


def test_measurement_nan_still_reduces_joint_completeness(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    ca = _allow_all_concepts(monkeypatch)
    path = tmp_path / "measurement_missing_fixture.parquet"
    pd.DataFrame(
        {
            "stay_id": [1, 2, 3, 4, 5],
            "peep": [8.0, 10.0, None, 12.0, 14.0],
            "lact": [1.2, None, 2.4, None, 3.1],
        }
    ).to_parquet(path, index=False)

    result = ca.real_data_concept_feasibility(
        ["peep", "lact"],
        "miiv",
        path,
    )

    lact = result["lact"]
    assert lact.n_present == 3
    assert lact.fraction_missing == pytest.approx(2.0 / 5.0)
    assert lact.n_joint_complete == 2
    assert lact.joint_fraction_complete == pytest.approx(2.0 / 5.0)


def test_screened_binary_concepts_do_not_default_nan_to_false(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    ca = _allow_all_concepts(monkeypatch)
    path = tmp_path / "screened_binary_fixture.parquet"
    pd.DataFrame(
        {
            "stay_id": [1, 2, 3, 4, 5],
            "peep": [8.0, 10.0, None, 12.0, 14.0],
            # Assessed/screened-style binary: NaN means not assessed.
            "delirium_positive": [1.0, None, None, None, None],
        }
    ).to_parquet(path, index=False)

    result = ca.real_data_concept_feasibility(
        ["peep", "delirium_positive"],
        "miiv",
        path,
    )

    delirium = result["delirium_positive"]
    assert delirium.n_present == 1
    assert delirium.fraction_missing == pytest.approx(4.0 / 5.0)
    assert delirium.n_joint_complete == 1
    assert delirium.joint_fraction_complete == pytest.approx(1.0 / 5.0)


def test_caller_supplied_event_default_false_hint_controls_custom_events(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    ca = _allow_all_concepts(monkeypatch)
    path = tmp_path / "custom_event_hint_fixture.parquet"
    pd.DataFrame(
        {
            "stay_id": [1, 2, 3, 4],
            "marker": [1.0, 2.0, None, 4.0],
            "custom_event": [None, 1.0, None, None],
        }
    ).to_parquet(path, index=False)

    default_result = ca.real_data_concept_feasibility(
        ["marker", "custom_event"],
        "miiv",
        path,
    )
    hinted_result = ca.real_data_concept_feasibility(
        ["marker", "custom_event"],
        "miiv",
        path,
        event_default_false_concepts={"custom_event"},
    )

    assert default_result["custom_event"].n_present == 1
    assert default_result["custom_event"].n_joint_complete == 1
    assert hinted_result["custom_event"].n_present == 4
    assert hinted_result["custom_event"].fraction_missing == 0.0
    assert hinted_result["custom_event"].n_joint_complete == 3


def test_time_window_and_aggregation_are_recorded_as_requested_only(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    ca = _allow_all_concepts(monkeypatch)
    path = _write_joint_fixture(tmp_path)

    result = ca.real_data_concept_feasibility(
        ["crea"],
        "miiv",
        path,
        time_window="first_24h",
        aggregation="median",
    )

    crea = result["crea"]
    assert crea.denominator_n == 6
    assert crea.n_present == 4
    assert crea.time_window_requested == "first_24h"
    assert crea.aggregation_requested == "median"
    dumped = crea.model_dump()
    assert "time_window_applied" not in dumped
    assert "aggregation_rule" not in dumped


def test_real_data_feasibility_supports_mapping_cohort_filter(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    ca = _allow_all_concepts(monkeypatch)
    path = _write_joint_fixture(tmp_path)

    result = ca.real_data_concept_feasibility(
        ["crea", "map"],
        "miiv",
        path,
        cohort={"group": "a"},
    )

    crea = result["crea"]
    assert crea.denominator_n == 3
    assert crea.n_present == 2
    assert crea.n_joint_complete == 1
    assert crea.cohort_filter_summary == "group=a"


def test_real_data_feasibility_counts_patient_level_presence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    ca = _allow_all_concepts(monkeypatch)
    path = tmp_path / "patient_fixture.parquet"
    pd.DataFrame(
        {
            "patient_id": [10, 10, 20, 30],
            "crea": [None, 1.1, 1.5, None],
            "map": [72.0, 73.0, None, 88.0],
            "death": [0, 0, 1, 1],
        }
    ).to_parquet(path, index=False)

    result = ca.real_data_concept_feasibility(
        ["crea", "map"],
        "miiv",
        path,
        analytic_unit="patient",
    )

    crea = result["crea"]
    assert crea.denominator_n == 3
    assert crea.n_present == 2
    assert crea.fraction_missing == pytest.approx(1.0 / 3.0)
    assert crea.n_joint_complete == 1
    assert crea.joint_fraction_complete == pytest.approx(1.0 / 3.0)
    assert "effect_estimate" in crea.non_outcome_blind_fields_checked


def test_blocked_dictionary_concept_short_circuits_without_reading_data(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    ca = _availability_module()

    def fake_blocked_availability(*, concept, database, requested_concept=None):
        canonical = ca.normalize_concept_name(concept)
        return ca.ConceptDatabaseAvailability(
            concept=canonical,
            requested_concept=requested_concept or concept,
            database=ca.normalize_database_name(database),
            status="blocked",
            available=False,
            reason="concept_not_found",
        )

    def fail_if_read(_path):
        raise AssertionError("blocked concepts must not read prepared data")

    monkeypatch.setattr(ca, "explain_concept_availability", fake_blocked_availability)
    monkeypatch.setattr(pd, "read_parquet", fail_if_read)

    result = ca.real_data_concept_feasibility(
        ["unmapped_future_marker"],
        "miiv",
        tmp_path / "missing.parquet",
    )

    blocked = result["unmapped_future_marker"]
    assert blocked.denominator_n == 0
    assert blocked.n_present == 0
    assert blocked.fraction_missing == 0.0
    assert blocked.missingness_applicable is False
    assert blocked.structural_unavailable is True
    assert blocked.n_joint_complete == 0
    assert blocked.structural_unavailable_concepts == ["unmapped_future_marker"]
    assert blocked.joint_denominator_concepts == []
    assert blocked.cohort_filter_summary.startswith("structural_unavailable:")
    assert "excluded from missingness denominator" in (blocked.note or "")
    assert "p_value" in blocked.non_outcome_blind_fields_checked


def test_structural_unavailable_concept_is_excluded_from_joint_missingness_denominator(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    ca = _availability_module()
    path = _write_joint_fixture(tmp_path)

    def fake_mixed_availability(*, concept, database, requested_concept=None):
        canonical = ca.normalize_concept_name(concept)
        if canonical == "unmapped_future_marker":
            return ca.ConceptDatabaseAvailability(
                concept=canonical,
                requested_concept=requested_concept or concept,
                database=ca.normalize_database_name(database),
                status="blocked",
                available=False,
                reason="concept_not_found",
            )
        return ca.ConceptDatabaseAvailability(
            concept=canonical,
            requested_concept=requested_concept or concept,
            database=ca.normalize_database_name(database),
            status="full",
            available=True,
            direct_source=True,
            reason="direct_source_available",
        )

    monkeypatch.setattr(ca, "explain_concept_availability", fake_mixed_availability)

    result = ca.real_data_concept_feasibility(
        ["crea", "unmapped_future_marker"],
        "miiv",
        path,
    )

    crea = result["crea"]
    structural = result["unmapped_future_marker"]
    assert crea.denominator_n == 6
    assert crea.n_present == 4
    assert crea.n_joint_complete == 4
    assert crea.joint_fraction_complete == pytest.approx(2.0 / 3.0)
    assert crea.joint_denominator_concepts == ["crea"]
    assert crea.structural_unavailable_concepts == ["unmapped_future_marker"]
    assert "unmapped_future_marker" in (crea.note or "")
    assert structural.missingness_applicable is False
    assert structural.fraction_missing == 0.0


def test_source_unavailable_runtime_cell_is_structural_not_data_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    ca = _availability_module()

    def fake_source_unavailable(*, concept, database, requested_concept=None):
        canonical = ca.normalize_concept_name(concept)
        return ca.ConceptDatabaseAvailability(
            concept=canonical,
            requested_concept=requested_concept or concept,
            database=ca.normalize_database_name(database),
            status="blocked",
            available=False,
            reason="source_unavailable",
            runtime_reason="source_unavailable",
            structural_unavailable=True,
            source_missing_tables=["inputevents"],
        )

    def fail_if_read(_path):
        raise AssertionError("structural source_unavailable must not read prepared data")

    monkeypatch.setattr(ca, "explain_concept_availability", fake_source_unavailable)
    monkeypatch.setattr(pd, "read_parquet", fail_if_read)

    result = ca.real_data_concept_feasibility(
        ["norepi_rate"],
        "mimic",
        tmp_path / "missing.parquet",
    )

    norepi = result["norepi_rate"]
    assert norepi.structural_unavailable is True
    assert norepi.missingness_applicable is False
    assert norepi.fraction_missing == 0.0
    assert norepi.availability_reason == "source_unavailable"
    assert norepi.source_missing_tables == ["inputevents"]
