"""Cross-concept cohort columns and the lineage they are allowed to claim.

The cohort materializer binds one output column to one source concept, which
is what keeps a materialized column from claiming provenance it does not have.
A few scientific readings genuinely need several concepts at once -- an
observability-preserving KDIGO stage is the worked example, where "stage 0" and
"never assessed" are told apart by three evidence receipts together.  These
tests fix both halves: the reading itself, and the rule that a multi-source
receipt is only ever accepted for a transform the contract declares.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from easyicu.concept.export_metadata import build_export_file_metadata_binding
from easyicu.concept.metadata_projection import declares_physical_numeric_domain
from easyicu.concept.metadata_sidecar import (
    EXPORT_PHYSICAL_SCOPE,
    ColumnMetadataBinding,
    ColumnMetadataFileBinding,
    ColumnMetadataSidecar,
    write_content_addressed_sidecar,
)
from easyicu.resources import load_dictionary
from easyicu.research_agent.cohort.materializer import (
    materialize_cohort,
    materialize_to_parquet,
)
from easyicu.research_agent.contracts.host_derivations import (
    HOST_DERIVATIONS,
    STRICT_KDIGO_DERIVATION_ID,
    HostDerivationError,
    host_derivation,
    host_derivation_producing,
    host_derived_transform,
)
from easyicu.research_agent.intake import export_package as intake
from easyicu.research_agent.intake.export_package import open_export_package
from easyicu.research_agent.intake.materialized_metadata import (
    MaterializedMetadataError,
    load_verified_materialized_cohort_authority,
)


_RENAL_CONCEPTS = (
    "aki_stage_creat_reference",
    "aki_stage_uo_reference",
    "aki_stage_rrt_reference",
    "creatinine_evidence_status",
    "urine_evidence_status",
    "rrt_evidence_status",
)


def _file_binding(
    *, relative_path: str, module: str, frame: pd.DataFrame, concepts: list[str]
) -> ColumnMetadataFileBinding:
    """Bind one physical file exactly as the module exporter does.

    The renal bundle's receipts are producer-owned callback outputs rather than
    source-dictionary concepts, so the export's own binder is the only honest
    way to type them here.
    """

    return build_export_file_metadata_binding(
        relative_path=relative_path,
        module=module,
        frame=frame,
        concept_ids=concepts,
        database="miiv",
        database_class_prefixes=(),
        dictionary=load_dictionary(include_sofa2=True),
    )


def _column_binding(concept: str) -> ColumnMetadataBinding:
    frame = _renal_rows() if concept in _RENAL_CONCEPTS else None
    assert frame is not None, concept
    binding = _file_binding(
        relative_path="renal.parquet",
        module="renal",
        frame=frame,
        concepts=list(_RENAL_CONCEPTS),
    )
    return binding.columns[concept]


def _renal_rows() -> pd.DataFrame:
    """Four stays that exercise every reading the window rule distinguishes."""

    return pd.DataFrame(
        {
            "stay_id": [1, 2, 2, 3, 4],
            "charttime": [2.0, 1.0, 30.0, 3.0, 4.0],
            # stay 1: every component observed and negative -> a real stage 0
            # stay 2: complete negative inside the window, positive only after
            # stay 3: urine positive at stage 2 inside the window
            # stay 4: creatinine negative, urine never assessed -> unknown
            "aki_stage_creat_reference": [0, 0, 3, 0, 0],
            "aki_stage_uo_reference": [0, 0, 0, 2, None],
            "aki_stage_rrt_reference": [0, 0, 0, 0, 0],
            "creatinine_evidence_status": [
                "negative",
                "negative",
                "positive",
                "negative",
                "negative",
            ],
            "urine_evidence_status": [
                "negative",
                "negative",
                "negative",
                "positive",
                "indeterminate",
            ],
            "rrt_evidence_status": [
                "negative",
                "negative",
                "negative",
                "negative",
                "negative",
            ],
        }
    ).astype({column: "Int64" for column in _RENAL_CONCEPTS[:3]})


def _typed_renal_export(
    root: Path,
    *,
    renal: pd.DataFrame | None = None,
    renal_concepts: list[str] | None = None,
    stays: list[int] | None = None,
) -> Path:
    root.mkdir()
    renal = _renal_rows() if renal is None else renal
    renal_concepts = list(renal_concepts or _RENAL_CONCEPTS)
    stays = list(stays or [1, 2, 3, 4])
    statics = pd.DataFrame(
        {"stay_id": stays, "age": [50 + 10 * index for index in range(len(stays))]}
    )
    outcomes = pd.DataFrame(
        {"stay_id": stays, "death": [index % 2 == 1 for index in range(len(stays))]}
    )
    renal.to_parquet(root / "renal.parquet", index=False)
    statics.to_parquet(root / "demographics.parquet", index=False)
    outcomes.to_parquet(root / "outcome.parquet", index=False)
    renal_binding = _file_binding(
        relative_path="renal.parquet",
        module="renal",
        frame=renal,
        concepts=renal_concepts,
    )
    demographics_binding = _file_binding(
        relative_path="demographics.parquet",
        module="demographics",
        frame=statics,
        concepts=["age"],
    )
    outcome_binding = _file_binding(
        relative_path="outcome.parquet",
        module="outcome",
        frame=outcomes,
        concepts=["death"],
    )
    sidecar = ColumnMetadataSidecar(
        source_database="miiv",
        source_database_class_prefixes=(),
        scope=EXPORT_PHYSICAL_SCOPE,
        files=(renal_binding, demographics_binding, outcome_binding),
    )
    reference = write_content_addressed_sidecar(root, sidecar)
    (root / intake.NATIVE_MANIFEST).write_text(
        json.dumps(
            {
                "schema_version": intake.NATIVE_MANIFEST_SCHEMA_V2,
                "database": "miiv",
                "format": "parquet",
                "concept_selection": {
                    "mode": "explicit",
                    "modules": {
                        "renal": list(renal_concepts),
                        "demographics": ["age"],
                        "outcome": ["death"],
                    },
                },
                "files": [
                    {
                        "file": "renal.parquet",
                        "module": "renal",
                        "concepts": len(renal_concepts),
                        "concept_ids": list(renal_concepts),
                        "rows": len(renal),
                        "column_metadata_columns": list(renal_binding.columns),
                    },
                    {
                        "file": "demographics.parquet",
                        "module": "demographics",
                        "concepts": 1,
                        "concept_ids": ["age"],
                        "rows": len(statics),
                        "column_metadata_columns": ["age"],
                    },
                    {
                        "file": "outcome.parquet",
                        "module": "outcome",
                        "concepts": 1,
                        "concept_ids": ["death"],
                        "rows": len(outcomes),
                        "column_metadata_columns": ["death"],
                    },
                ],
                "feature_definitions": {"included": False},
                "column_metadata": reference.to_dict(),
            }
        ),
        encoding="utf-8",
    )
    return root


def _materialize(root: Path, **overrides: object):
    kwargs: dict[str, object] = {
        "feature_concepts": [],
        "database": "miiv",
        "data_path": str(root),
        "cohort_window": (0.0, 24.0),
        "outcome_concepts": ["death"],
        "static_concepts": ["age"],
        "host_derivations": [STRICT_KDIGO_DERIVATION_ID],
    }
    kwargs.update(overrides)
    return materialize_cohort(**kwargs)  # type: ignore[arg-type]


def test_the_declaration_is_closed_and_self_consistent() -> None:
    for derivation in HOST_DERIVATIONS.values():
        for output in derivation.outputs:
            assert output.primary_concept in derivation.source_concepts
            assert host_derived_transform(output.transform_id) == (
                derivation,
                output,
            )
            assert host_derivation_producing(output.column) == (derivation, output)
    assert host_derived_transform("window_numeric_max") is None
    assert host_derivation_producing("lact_max") is None
    with pytest.raises(HostDerivationError, match="undeclared host derivation"):
        host_derivation("not_a_declared_derivation")


def test_stage_zero_requires_observed_complete_negative_evidence(
    tmp_path: Path,
) -> None:
    """The whole point of the reading: an unassessed stay is not a stage 0."""

    root = _typed_renal_export(tmp_path / "export")

    cohort, provenance = _materialize(root)

    by_stay = cohort.set_index("stay_id")
    assert provenance["host_derivations"] == [STRICT_KDIGO_DERIVATION_ID]
    # stay 1 ruled injury out; stay 2's positive falls outside the window
    assert by_stay.loc[1, "aki_stage_strict"] == 0
    assert by_stay.loc[2, "aki_stage_strict"] == 0
    assert by_stay.loc[3, "aki_stage_strict"] == 2
    # stay 4 never had a urine assessment, so it is unknown -- not stage 0
    assert pd.isna(by_stay.loc[4, "aki_stage_strict"])
    assert by_stay.loc[4, "aki_ascertainment"] == "partial_no_observed_positive"
    assert by_stay["kidney_complete_negative_observed"].tolist() == [1, 1, 0, 0]
    assert by_stay["kidney_window_row_count"].tolist() == [1, 1, 1, 1]


def test_a_stay_with_no_source_rows_is_unknown_with_a_real_zero_count(
    tmp_path: Path,
) -> None:
    # A fifth stay is in the cohort denominator but has no renal row at all.
    root = _typed_renal_export(tmp_path / "export", stays=[1, 2, 3, 4, 5])

    cohort, _ = _materialize(root)

    row = cohort.set_index("stay_id").loc[5]
    assert pd.isna(row["aki_stage_strict"])
    assert pd.isna(row["aki_ascertainment"])
    # The window was searched and was empty; that is a measured zero, and it is
    # not the same statement as "kidney injury was ruled out".
    assert row["kidney_window_row_count"] == 0
    assert row["kidney_complete_negative_observed"] == 0


def test_the_sealed_authority_binds_every_concept_the_reading_used(
    tmp_path: Path,
) -> None:
    root = _typed_renal_export(tmp_path / "export")
    out = tmp_path / "materialized"
    out.mkdir()

    with open_export_package(root) as package:
        paths = materialize_to_parquet(
            out,
            stem="cohort",
            source_package=package,
            feature_concepts=[],
            database="miiv",
            data_path=str(root),
            cohort_window=(0.0, 24.0),
            outcome_concepts=["death"],
            static_concepts=["age"],
            host_derivations=[STRICT_KDIGO_DERIVATION_ID],
        )

    verified = load_verified_materialized_cohort_authority(Path(paths["parquet"]))
    assert verified is not None
    receipts = {
        item.output_column: item for item in verified.authority.output_derivations
    }
    stage = receipts["aki_stage_strict"]
    assert stage.transform_id == "strict_kdigo_window_stage"
    assert {source.column for source in stage.sources} == set(_RENAL_CONCEPTS)
    assert tuple(verified.authority.producer_parameters["host_derivations"]) == (
        STRICT_KDIGO_DERIVATION_ID,
    )
    binding = verified.sidecar.files[0].columns["aki_stage_strict"]
    # Unit and bounds come from the declared primary; the rest of the reading's
    # inputs are named so a reader can see what the column depends on.
    assert binding.metadata.source_concept == "aki_stage_creat_reference"
    assert set(binding.metadata.derived_from_concepts) == set(_RENAL_CONCEPTS) - {
        "aki_stage_creat_reference"
    }
    assert binding.derivation_window is not None
    assert (
        binding.derivation_window.start_hours,
        binding.derivation_window.end_hours,
    ) == (0.0, 24.0)


def test_an_undeclared_transform_may_not_widen_its_own_lineage(
    tmp_path: Path,
) -> None:
    """A multi-source receipt is a declared shape, never a self-granted one."""

    from easyicu.research_agent.intake import materialized_metadata as materialized

    root = _typed_renal_export(tmp_path / "export")
    out = tmp_path / "materialized"
    out.mkdir()
    with open_export_package(root) as package:
        paths = materialize_to_parquet(
            out,
            stem="cohort",
            source_package=package,
            feature_concepts=[],
            database="miiv",
            data_path=str(root),
            cohort_window=(0.0, 24.0),
            outcome_concepts=["death"],
            static_concepts=["age"],
            host_derivations=[STRICT_KDIGO_DERIVATION_ID],
        )
    verified = load_verified_materialized_cohort_authority(Path(paths["parquet"]))
    assert verified is not None
    stage = next(
        item
        for item in verified.authority.output_derivations
        if item.output_column == "aki_stage_strict"
    )
    forged = materialized.OutputDerivation(
        output_column="age",
        sources=stage.sources,
        transform_id="stay_level_unique_value",
    )
    authority = materialized.MaterializedCohortAuthority(
        **{
            **{
                field: getattr(verified.authority, field)
                for field in verified.authority.__dataclass_fields__
            },
            "output_derivations": tuple(
                forged if item.output_column == "age" else item
                for item in verified.authority.output_derivations
            ),
        }
    )
    with pytest.raises(MaterializedMetadataError, match="cardinality mismatch"):
        materialized._validate_derivation_contract(
            authority,
            file_binding=verified.sidecar.files[0],
            source_sidecar=materialized.read_content_addressed_sidecar(
                root / json.loads(
                    (root / intake.NATIVE_MANIFEST).read_text(encoding="utf-8")
                )["column_metadata"]["file"],
                expected_sha256=json.loads(
                    (root / intake.NATIVE_MANIFEST).read_text(encoding="utf-8")
                )["column_metadata"]["sha256"],
                expected_size=json.loads(
                    (root / intake.NATIVE_MANIFEST).read_text(encoding="utf-8")
                )["column_metadata"]["size"],
            ),
        )


def test_a_categorical_receipt_is_never_coerced_to_a_number() -> None:
    """A 'category' unit is the dictionary saying "codes", not a quantity.

    Coercing one turns every observed evidence status into a lossy-conversion
    failure, which is how a categorical receipt becomes unreadable to a typed
    cohort and the strict reading becomes impossible to materialize.
    """

    status = _column_binding("creatinine_evidence_status")
    stage = _column_binding("aki_stage_creat_reference")

    assert status.metadata.canonical_unit == "category"
    assert declares_physical_numeric_domain(status.metadata) is False
    assert stage.metadata.canonical_unit == "0-3"
    assert declares_physical_numeric_domain(stage.metadata) is True


def test_a_host_derivation_needs_every_concept_it_declares(tmp_path: Path) -> None:
    """An export without one receipt cannot produce a partial strict reading."""

    root = _typed_renal_export(
        tmp_path / "export",
        renal=_renal_rows().drop(columns=["rrt_evidence_status"]),
        renal_concepts=[
            concept
            for concept in _RENAL_CONCEPTS
            if concept != "rrt_evidence_status"
        ],
    )

    with pytest.raises(MaterializedMetadataError, match="rrt_evidence_status"):
        _materialize(root)
