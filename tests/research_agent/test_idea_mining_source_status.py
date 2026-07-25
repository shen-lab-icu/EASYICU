"""Contracts for provider-free Idea Mining source-status profiling."""

from __future__ import annotations

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from easyicu.research_agent.discovery.idea_mining_source_status import (
    ComparisonSourceSpec,
    MeasurementAuditCriteria,
    PairAnswerabilityCriteria,
    RowwiseDerivedConceptSpec,
    profile_rowwise_derived_concept,
)


def _formula(columns):
    return pc.add(
        pc.cast(columns["ca"], pa.float64()),
        pc.multiply(0.8, pc.subtract(4.0, pc.cast(columns["alb"], pa.float64()))),
    )


def _spec(*, comparison: bool = True) -> RowwiseDerivedConceptSpec:
    return RowwiseDerivedConceptSpec(
        concept_name="test_corrected_calcium",
        source_table="chemistry",
        component_columns=("ca", "alb"),
        formula_id="test_formula_v1",
        valid_range=(4.0, 16.0),
        materialized_column="corrected_calcium",
        comparison_source=(
            ComparisonSourceSpec(
                table="blood_gas", column="cai", valid_range=(0.1, 5.0)
            )
            if comparison
            else None
        ),
    )


def _write_miiv(
    root,
    *,
    demographics: dict[str, list],
    chemistry: dict[str, list] | None,
    blood_gas: dict[str, list] | None = None,
):
    database = root / "miiv"
    database.mkdir(parents=True)
    pq.write_table(pa.table(demographics), database / "demographics.parquet")
    if chemistry is not None:
        pq.write_table(pa.table(chemistry), database / "chemistry.parquet")
    if blood_gas is not None:
        pq.write_table(pa.table(blood_gas), database / "blood_gas.parquet")


def test_same_row_components_are_required_and_status_partition_is_exclusive(tmp_path):
    _write_miiv(
        tmp_path,
        demographics={"stay_id": [1, 2, 3]},
        chemistry={
            "stay_id": [1, 1, 2, 3],
            "charttime": [10.0, 20.0, 10.0, 10.0],
            # Stay 1 has both components, but never on the same prepared row.
            "ca": [8.0, None, 9.0, 20.0],
            "alb": [None, 3.0, 3.0, 4.0],
            "corrected_calcium": [None, None, 9.8, 20.0],
        },
        blood_gas={
            "stay_id": [1, 2, 3],
            "cai": [1.1, None, 8.0],
        },
    )

    report = profile_rowwise_derived_concept(
        tmp_path, databases=["miiv"], spec=_spec(), formula=_formula
    )
    row = report.databases[0]
    assert row.exact_component_rows == 2
    assert row.exact_component_stays == 2
    assert row.recomputed_valid_stays == 1
    assert row.source_status.model_dump() == {
        "structural_no_source": 0,
        "source_present_unmeasured": 1,
        "contradictory_or_out_of_range": 1,
        "valid_observed": 1,
    }
    assert row.source_status.total == row.denominator_stays
    assert row.comparison_coverage is not None
    assert row.comparison_coverage.observed_stays == 1
    assert row.predictor_outcome_pair_coverage is not None
    assert row.predictor_outcome_pair_coverage.joint_valid_stays == 0
    assert report.analysis_authorized is False
    assert report.paper_authorized is False


def test_missing_component_is_structural_no_source_but_present_component_is_visible(
    tmp_path,
):
    _write_miiv(
        tmp_path,
        demographics={"stay_id": [1, 2]},
        chemistry={
            "stay_id": [1, 2],
            "charttime": [10.0, 10.0],
            "ca": [8.0, 9.0],
            "corrected_calcium": [8.0, 9.0],
        },
    )

    row = profile_rowwise_derived_concept(
        tmp_path, databases=["miiv"], spec=_spec(), formula=_formula
    ).databases[0]
    coverage = {item.column: item for item in row.component_coverage}
    assert row.data_readiness == "structural_no_source"
    assert row.source_status.structural_no_source == 2
    assert coverage["ca"].source_present is True
    assert coverage["ca"].observed_stays == 2
    assert coverage["alb"].source_present is False
    assert "required source column missing: alb" in row.warnings


def test_denominator_identity_quality_is_reported(tmp_path):
    _write_miiv(
        tmp_path,
        demographics={"stay_id": [1, 1, 2, None]},
        chemistry={
            "stay_id": [1, 2, 99],
            "charttime": [10.0, 10.0, 10.0],
            "ca": [8.0, 9.0, 10.0],
            "alb": [4.0, 4.0, 4.0],
            "corrected_calcium": [8.0, 9.0, 10.0],
        },
        blood_gas={"stay_id": [1], "cai": [1.0]},
    )

    row = profile_rowwise_derived_concept(
        tmp_path, databases=["miiv"], spec=_spec(), formula=_formula
    ).databases[0]
    assert row.denominator_rows == 4
    assert row.denominator_stays == 2
    assert row.missing_denominator_ids == 1
    assert row.duplicate_denominator_ids == 1
    assert row.source_stays_outside_denominator == 1
    assert "denominator contains missing stay identifiers" in row.warnings
    assert "denominator contains duplicate stay identifiers" in row.warnings


def test_materialized_formula_disagreement_is_fail_visible(tmp_path):
    _write_miiv(
        tmp_path,
        demographics={"stay_id": [1, 2]},
        chemistry={
            "stay_id": [1, 2],
            "charttime": [10.0, 10.0],
            "ca": [8.0, 9.0],
            "alb": [4.0, 3.0],
            "corrected_calcium": [8.0, 11.0],
        },
    )

    row = profile_rowwise_derived_concept(
        tmp_path,
        databases=["miiv"],
        spec=_spec(comparison=False),
        formula=_formula,
    ).databases[0]
    assert row.formula_agreement is not None
    assert row.formula_agreement.comparable_rows == 2
    assert row.formula_agreement.within_tolerance_rows == 1
    assert row.formula_agreement.mismatch_rows == 1
    assert row.formula_agreement.material_difference_rows == 1
    assert row.formula_agreement.material_difference_fraction == 0.5
    assert (
        "materialized values materially differ from host recomputation" in row.warnings
    )
    assert all(binding.sha256 for binding in row.input_files)
    assert all(binding.schema_sha256 for binding in row.input_files)


def test_materialized_predictor_authority_handles_post_aggregation_non_equivalence(
    tmp_path,
):
    _write_miiv(
        tmp_path,
        demographics={"stay_id": [1, 2]},
        chemistry={
            "stay_id": [1, 2],
            "charttime": [10.0, 10.0],
            "ca": [8.0, 9.0],
            "alb": [4.0, 3.0],
            # Only stay 1 has an authoritative materialized predictor.
            "corrected_calcium": [10.0, None],
        },
        blood_gas={"stay_id": [1, 2], "cai": [1.0, 1.1]},
    )
    spec = _spec().model_copy(
        update={
            "predictor_authority": "materialized_column",
            "materialized_comparison_semantics": (
                "nonlinear_post_aggregation_not_equivalent"
            ),
        }
    )

    row = profile_rowwise_derived_concept(
        tmp_path, databases=["miiv"], spec=spec, formula=_formula
    ).databases[0]
    pair = row.predictor_outcome_pair_coverage
    assert pair is not None
    assert row.recomputed_valid_stays == 2
    assert pair.predictor_valid_stays == 1
    assert pair.joint_valid_stays == 1
    assert not any("materially differ" in item for item in row.warnings)
    assert any("nonlinear derivation precedes" in item for item in row.warnings)


def test_missing_source_table_fails_closed_without_extraction(tmp_path):
    _write_miiv(
        tmp_path,
        demographics={"stay_id": [1, 2]},
        chemistry=None,
        blood_gas={"stay_id": [1, 2], "cai": [1.1, None]},
    )

    row = profile_rowwise_derived_concept(
        tmp_path, databases=["miiv"], spec=_spec(), formula=_formula
    ).databases[0]
    assert row.data_readiness == "structural_no_source"
    assert row.source_rows == 0
    assert row.source_status.structural_no_source == 2
    assert row.input_files[0].relative_path == "miiv/demographics.parquet"
    assert row.comparison_coverage is not None
    assert row.comparison_coverage.observed_stays == 1


def test_duplicate_database_request_is_rejected(tmp_path):
    _write_miiv(
        tmp_path,
        demographics={"stay_id": [1]},
        chemistry=None,
    )

    try:
        profile_rowwise_derived_concept(
            tmp_path,
            databases=["miiv", "miiv"],
            spec=_spec(),
            formula=_formula,
        )
    except ValueError as exc:
        assert "must not contain duplicates" in str(exc)
    else:  # pragma: no cover - explicit failure branch
        raise AssertionError("duplicate databases were accepted")


def test_measurement_audit_requires_enough_databases_and_coverage_contrast(tmp_path):
    _write_miiv(
        tmp_path,
        demographics={"stay_id": [1, 2, 3]},
        chemistry={
            "stay_id": [1, 2, 3],
            "charttime": [10.0, 10.0, 10.0],
            "ca": [8.0, 9.0, None],
            "alb": [4.0, 4.0, None],
            "corrected_calcium": [8.0, 9.0, None],
        },
    )
    database = tmp_path / "mimic"
    database.mkdir()
    pq.write_table(
        pa.table({"icustay_id": [10, 11, 12]}),
        database / "demographics.parquet",
    )
    pq.write_table(
        pa.table(
            {
                "icustay_id": [10, 11, 12],
                "charttime": [10.0, 10.0, 10.0],
                "ca": [8.0, None, None],
                "alb": [4.0, None, None],
                "corrected_calcium": [8.0, None, None],
            }
        ),
        database / "chemistry.parquet",
    )
    criteria = MeasurementAuditCriteria(
        min_databases_with_valid_observations=2,
        min_valid_stays_per_database=1,
        min_cross_database_coverage_range=0.20,
    )

    report = profile_rowwise_derived_concept(
        tmp_path,
        databases=["miiv", "mimic"],
        spec=_spec(comparison=False),
        formula=_formula,
        measurement_audit_criteria=criteria,
    )
    answerability = report.measurement_audit_answerability
    assert answerability is not None
    assert answerability.status == "answerable_requires_human_confirmation"
    assert answerability.eligible_databases == ("miiv", "mimic")
    assert answerability.coverage_range == 1 / 3
    assert answerability.analysis_authorized is False
    assert answerability.paper_authorized is False

    blocked = profile_rowwise_derived_concept(
        tmp_path,
        databases=["miiv", "mimic"],
        spec=_spec(comparison=False),
        formula=_formula,
        measurement_audit_criteria=criteria.model_copy(
            update={"min_valid_stays_per_database": 2}
        ),
    ).measurement_audit_answerability
    assert blocked is not None
    assert blocked.status == "insufficient_database_coverage"


def test_pair_answerability_requires_joint_predictor_outcome_stays(tmp_path):
    _write_miiv(
        tmp_path,
        demographics={"stay_id": [1, 2, 3]},
        chemistry={
            "stay_id": [1, 2, 3],
            "charttime": [10.0, 10.0, 10.0],
            "ca": [8.0, 9.0, None],
            "alb": [4.0, 4.0, None],
            "corrected_calcium": [8.0, 9.0, None],
        },
        blood_gas={"stay_id": [1, 2, 3], "cai": [1.0, None, 1.2]},
    )
    database = tmp_path / "mimic"
    database.mkdir()
    pq.write_table(
        pa.table({"icustay_id": [10, 11, 12]}),
        database / "demographics.parquet",
    )
    pq.write_table(
        pa.table(
            {
                "icustay_id": [10, 11, 12],
                "charttime": [10.0, 10.0, 10.0],
                "ca": [8.0, 9.0, 10.0],
                "alb": [4.0, 4.0, 4.0],
                "corrected_calcium": [8.0, 9.0, 10.0],
            }
        ),
        database / "chemistry.parquet",
    )
    pq.write_table(
        pa.table({"icustay_id": [10, 11, 12], "cai": [1.0, 1.1, None]}),
        database / "blood_gas.parquet",
    )

    report = profile_rowwise_derived_concept(
        tmp_path,
        databases=["miiv", "mimic"],
        spec=_spec(),
        formula=_formula,
        pair_answerability_criteria=PairAnswerabilityCriteria(
            min_databases_with_joint_observations=2,
            min_joint_stays_per_database=1,
            min_joint_fraction_per_database=0.30,
        ),
    )
    pairs = {
        row.database: row.predictor_outcome_pair_coverage for row in report.databases
    }
    assert pairs["miiv"] is not None
    assert pairs["miiv"].joint_valid_stays == 1
    assert pairs["mimic"] is not None
    assert pairs["mimic"].joint_valid_stays == 2
    assert report.pair_answerability is not None
    assert report.pair_answerability.status == "answerable_requires_temporal_protocol"
    assert report.pair_answerability.analysis_authorized is False
    assert report.pair_answerability.paper_authorized is False

    blocked = profile_rowwise_derived_concept(
        tmp_path,
        databases=["miiv", "mimic"],
        spec=_spec(),
        formula=_formula,
        pair_answerability_criteria=PairAnswerabilityCriteria(
            min_databases_with_joint_observations=2,
            min_joint_stays_per_database=2,
            min_joint_fraction_per_database=0.30,
        ),
    ).pair_answerability
    assert blocked is not None
    assert blocked.status == "insufficient_joint_coverage"
    assert blocked.eligible_databases == ("mimic",)
