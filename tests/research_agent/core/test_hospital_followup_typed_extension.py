"""Typed-lineage extension of a sealed cohort by verified hospital follow-up.

The follow-up child must be an exact ordered parent-row subset for every
carried column, carry a typed event/censor axis bound to the parent's event
status, and stay verifiable after the run stage copies it into a run directory.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from easyicu.concept.metadata_projection import ConceptColumnRole
from easyicu.research_agent.acquisition.foundation import AcquisitionResult
from easyicu.research_agent.acquisition.hospital_mortality_followup import (
    derive_mimic_iv_hospital_mortality_followup,
)
from easyicu.research_agent.acquisition.hospital_outcome_materialization import (
    materialize_hospital_followup_acquisition,
)
from easyicu.research_agent.cohort import materializer as cohort_materializer
from easyicu.research_agent.intake import materialized_metadata as materialized
from easyicu.research_agent.intake.materialized_metadata import (
    HOSPITAL_FOLLOWUP_EXTENSION_PRODUCER,
    MaterializedMetadataError,
    implementation_bundle_sha256,
    load_verified_materialized_cohort_authority,
    publish_hospital_followup_materialized_cohort,
    stage_materialized_cohort_authority,
)

from tests.support.typed_export import typed_export


_RAW_SOURCE_RECEIPT = {
    "schema_version": "easyicu.registered_export_raw_source_authority/2",
    "authority_kind": "export_manifest_data_path",
    "database": "mimic_iv",
    "tables": {"icustays": "a" * 64, "admissions": "b" * 64},
}


def _followup_frames(stay_ids, *, exclude=()):
    rows = [
        {
            "stay_id": stay_id,
            "hospital_death": stay_id % 2 == 0,
            "death_time_hours": float(30 + stay_id) if stay_id % 2 == 0 else None,
            "hospital_followup_time_hours": float(30 + stay_id),
        }
        for stay_id in stay_ids
        if stay_id not in exclude
    ]
    followup = pd.DataFrame(
        rows,
        columns=[
            "stay_id",
            "hospital_death",
            "death_time_hours",
            "hospital_followup_time_hours",
        ],
    )
    exclusions = pd.DataFrame(
        [{"stay_id": stay_id, "reason": "hospital_mortality_status_missing"} for stay_id in exclude],
        columns=["stay_id", "reason"],
    )
    return followup, exclusions


def _typed_parent(tmp_path: Path):
    source = typed_export(
        tmp_path / "export",
        labs=pd.DataFrame(
            {
                "stay_id": [1, 2, 3, 3],
                "charttime": [1.0, 1.0, 1.0, 2.0],
                "age": [50, 60, 70, 70],
                "lact": [1.0, 2.0, 3.0, 4.0],
                "mech_vent": [False, True, False, True],
            }
        ),
        outcomes=pd.DataFrame(
            {"stay_id": [1, 2, 3], "death": [False, True, False]}
        ),
    )
    paths = cohort_materializer.materialize_to_parquet(
        tmp_path / "materialized",
        stem="universe",
        data_path=source,
        database="miiv",
        static_concepts=("age",),
        feature_concepts=("lact",),
        outcome_concepts=("death",),
    )
    parent = load_verified_materialized_cohort_authority(paths["parquet"])
    assert parent is not None
    return paths, parent


def _publish(tmp_path: Path, *, exclude=(3,)):
    paths, parent = _typed_parent(tmp_path)
    followup, exclusions = _followup_frames([1, 2, 3], exclude=exclude)
    target = paths["parquet"].parent / "hospital_followup_cohort.parquet"
    child = publish_hospital_followup_materialized_cohort(
        paths["parquet"],
        target,
        followup=followup,
        exclusions=exclusions,
        followup_receipt={"schema_version": "easyicu.mimic_iv_hospital_mortality_followup/1"},
        raw_source_receipt=_RAW_SOURCE_RECEIPT,
        producer_implementation_sha256=implementation_bundle_sha256(
            (Path(cohort_materializer.__file__),)
        ),
        producer_parameters={"adapter": "test"},
        expected_parent_authority=parent.reference,
    )
    assert child is not None
    return paths, parent, target, child


def test_followup_child_is_parent_bound_and_typed(tmp_path: Path) -> None:
    paths, parent, target, child = _publish(tmp_path)

    frame = pd.read_parquet(target)
    assert frame["stay_id"].tolist() == [1, 2]
    assert frame["death"].tolist() == [False, True]
    assert frame["death_time_hours"].tolist()[0] != frame["death_time_hours"].tolist()[0]
    assert frame["death_time_hours"].tolist()[1] == 32.0
    assert frame["hospital_followup_time_hours"].tolist() == [31.0, 32.0]
    assert "death_time" not in frame.columns
    assert child.authority.producer == HOSPITAL_FOLLOWUP_EXTENSION_PRODUCER
    assert child.authority.parent_authority_sha256 == parent.reference.sha256
    assert child.authority.cohort_columns == tuple(
        [c for c in parent.authority.cohort_columns if c != "death_time"]
        + ["death_time_hours", "hospital_followup_time_hours"]
    )
    binding = child.sidecar.files[0]
    assert binding.columns["death"].metadata.role is ConceptColumnRole.EVENT_STATUS
    assert binding.columns["death"].representation_transform == "hospital_followup_event_status"
    assert binding.columns["death_time_hours"].metadata.role is ConceptColumnRole.EVENT_TIME
    assert binding.columns["death_time_hours"].metadata.time_unit == "h"
    assert (
        binding.columns["hospital_followup_time_hours"].metadata.role
        is ConceptColumnRole.LAST_OBSERVATION_TIME
    )
    assert binding.columns["age"] == parent.sidecar.files[0].columns["age"]
    transforms = {
        item.output_column: item.transform_id
        for item in child.authority.output_derivations
    }
    assert transforms["death"] == "hospital_followup_event_status"
    assert transforms["age"] == "ordered_row_subset"
    assert all(
        item.sources[0].column == "death"
        for item in child.authority.output_derivations
        if item.output_column in {"death", "death_time_hours", "hospital_followup_time_hours"}
    )
    parameters = child.authority.producer_parameters
    assert parameters["excluded_row_count"] == 1
    assert parameters["exclusions"] == {"hospital_mortality_status_missing": 1}
    assert parameters["raw_source"]["authority_kind"] == "export_manifest_data_path"
    provenance = json.loads(
        (target.parent / "hospital_followup_cohort_provenance.json").read_text()
    )
    receipt = provenance["hospital_followup_materialization"]
    assert receipt["source_stays"] == 3 and receipt["analysis_stays"] == 2
    assert receipt["source_metadata_kind"] == "typed_materialized_authority"
    # This fixture export carries no event time, so nothing is invalidated.
    assert receipt["invalidated_parent_columns"] == []
    # Re-verification reads the parent parquet and proves the carried subset.
    assert load_verified_materialized_cohort_authority(target) is not None


def test_followup_requires_complete_coverage_and_a_typed_event_status(
    tmp_path: Path,
) -> None:
    paths, parent = _typed_parent(tmp_path)
    followup, exclusions = _followup_frames([1, 2])
    target = paths["parquet"].parent / "hospital_followup_cohort.parquet"
    with pytest.raises(MaterializedMetadataError, match="coverage is incomplete"):
        publish_hospital_followup_materialized_cohort(
            paths["parquet"],
            target,
            followup=followup,
            exclusions=exclusions,
            followup_receipt={},
            raw_source_receipt=_RAW_SOURCE_RECEIPT,
            producer_implementation_sha256="c" * 64,
            producer_parameters={},
        )
    assert not target.exists()
    assert not (target.parent / "hospital_followup_cohort_provenance.json").exists()
    # A censored stay carrying a death time is not an event/censor contract.
    bad, exclusions = _followup_frames([1, 2, 3])
    bad.loc[bad["stay_id"] == 1, "death_time_hours"] = 5.0
    with pytest.raises(MaterializedMetadataError, match="censored stays"):
        publish_hospital_followup_materialized_cohort(
            paths["parquet"],
            target,
            followup=bad,
            exclusions=exclusions,
            followup_receipt={},
            raw_source_receipt=_RAW_SOURCE_RECEIPT,
            producer_implementation_sha256="c" * 64,
            producer_parameters={},
        )
    assert not target.exists()


def test_followup_child_rejects_carried_value_or_axis_tampering(
    tmp_path: Path,
) -> None:
    paths, parent, target, child = _publish(tmp_path)
    table = pq.read_table(target)
    tampered = table.set_column(
        table.schema.get_field_index("age"),
        "age",
        pa.array([99, 60], type=table.schema.field("age").type),
    )
    pq.write_table(tampered, target)
    with pytest.raises(MaterializedMetadataError, match="no longer matches authority"):
        load_verified_materialized_cohort_authority(target)


def test_followup_child_rejects_resigned_parent_drift(tmp_path: Path) -> None:
    paths, parent, target, child = _publish(tmp_path)
    # Re-sign the child against a different (unrelated) parent digest: the
    # loader must refuse the chain even though the child's own digests agree.
    resigned = materialized.MaterializedCohortAuthority.from_dict(
        {
            **child.authority.to_dict(),
            "parent_authority_sha256": "d" * 64,
            "producer_parameters": {
                **dict(child.authority.producer_parameters),
                "parent_authority_sha256": "d" * 64,
            },
            "producer_parameters_sha256": materialized.canonical_parameters_sha256(
                {
                    **dict(child.authority.producer_parameters),
                    "parent_authority_sha256": "d" * 64,
                }
            ),
        }
    )
    authority_ref = materialized._write_authority(target.parent, resigned)
    provenance_path = materialized.materialized_provenance_path(target)
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["column_metadata"]["authority"] = authority_ref.to_dict()
    materialized._atomic_write_json(provenance_path, provenance)
    with pytest.raises(MaterializedMetadataError):
        load_verified_materialized_cohort_authority(target)


def test_followup_child_can_be_staged_and_re_verified_offline(tmp_path: Path) -> None:
    paths, parent, target, child = _publish(tmp_path)
    run_dir = tmp_path / "run"
    staged = stage_materialized_cohort_authority(
        target,
        run_dir / "cohort.parquet",
        producer_implementation_sha256=implementation_bundle_sha256(
            (Path(cohort_materializer.__file__),)
        ),
    )
    assert staged is not None
    assert staged.authority.parent_authority_sha256 == child.reference.sha256
    assert (run_dir / child.reference.file).exists()
    assert (run_dir / parent.reference.file).exists()
    assert (run_dir / parent.authority.column_metadata.file).exists()
    # The staged copy verifies the whole typed chain from snapshots alone.
    assert load_verified_materialized_cohort_authority(run_dir / "cohort.parquet") is not None
    (run_dir / parent.reference.file).unlink()
    with pytest.raises(MaterializedMetadataError):
        load_verified_materialized_cohort_authority(run_dir / "cohort.parquet")


def test_typed_acquisition_is_extended_not_rewritten(tmp_path: Path) -> None:
    paths, parent = _typed_parent(tmp_path)
    acquisition = AcquisitionResult(
        universe_path=paths["parquet"],
        provenance_path=paths["provenance"],
        selection=None,
        coverage=None,
        materialized_concepts=["age", "lact", "death"],
        cohort_authority_path=paths["parquet"].parent / parent.reference.file,
        cohort_authority_ref=parent.reference,
        materialized_columns=tuple(parent.authority.cohort_columns),
    )
    icustays = pd.DataFrame(
        {
            "stay_id": [1, 2, 3],
            "hadm_id": [10, 20, 30],
            "intime": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
        }
    )
    admissions = pd.DataFrame(
        {
            "hadm_id": [10, 20, 30],
            "dischtime": pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-04"]),
            "deathtime": [pd.NaT, pd.Timestamp("2020-01-03"), pd.NaT],
            "hospital_expire_flag": [0, 1, pd.NA],
        }
    )
    followup = derive_mimic_iv_hospital_mortality_followup(icustays, admissions)
    extended = materialize_hospital_followup_acquisition(
        acquisition, followup=followup, raw_source_receipt=_RAW_SOURCE_RECEIPT
    )
    assert extended.universe_path.name == "hospital_followup_cohort.parquet"
    assert extended.cohort_authority_ref is not None
    assert extended.cohort_authority_ref != parent.reference
    assert set(extended.materialized_columns) >= {
        "death",
        "death_time_hours",
        "hospital_followup_time_hours",
    }
    verified = load_verified_materialized_cohort_authority(
        extended.universe_path, expected_authority=extended.cohort_authority_ref
    )
    assert verified is not None
    assert verified.authority.producer == HOSPITAL_FOLLOWUP_EXTENSION_PRODUCER
    frame = pd.read_parquet(extended.universe_path)
    assert frame["stay_id"].tolist() == [1, 2]
    assert frame["death"].tolist() == [False, True]
    # The parent artifact and its authority stay untouched.
    assert load_verified_materialized_cohort_authority(paths["parquet"]) is not None
    with pytest.raises(ValueError, match="hospital_followup_artifact_exists"):
        materialize_hospital_followup_acquisition(
            acquisition, followup=followup, raw_source_receipt=_RAW_SOURCE_RECEIPT
        )
