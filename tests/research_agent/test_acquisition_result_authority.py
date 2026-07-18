from __future__ import annotations

from pathlib import Path

import pytest

from easyicu.research_agent.data_catalog import CoverageReport
from easyicu.research_agent.data_foundation import AcquisitionResult, ConceptSelection
from easyicu.research_agent.intake.materialized_metadata import (
    MaterializedCohortAuthorityRef,
    MaterializedMetadataError,
)
from easyicu.research_agent.intake.materialized_trajectory import (
    MaterializedTrajectoryAuthorityRef,
    MaterializedTrajectoryError,
)


def _acquisition_result(**overrides) -> AcquisitionResult:
    coverage = CoverageReport(
        requested=["death"],
        resolved={"death": "death"},
        available=["death"],
        missing=[],
    )
    values = {
        "universe_path": Path("universe.parquet"),
        "provenance_path": Path("provenance.json"),
        "selection": ConceptSelection(
            selected_concepts=["death"],
            coverage=coverage,
        ),
        "materialized_concepts": ["death"],
        "coverage": coverage,
    }
    values.update(overrides)
    return AcquisitionResult(**values)


def _cohort_authority_ref() -> MaterializedCohortAuthorityRef:
    return MaterializedCohortAuthorityRef(
        file=f"cohort_authority.sha256-{'a' * 64}.json",
        sha256="b" * 64,
        size=123,
    )


def _trajectory_authority_ref() -> MaterializedTrajectoryAuthorityRef:
    return MaterializedTrajectoryAuthorityRef(
        file=f"trajectory_authority.sha256-{'c' * 64}.json",
        sha256="c" * 64,
        size=456,
    )


@pytest.mark.parametrize(
    ("authority_path", "authority_ref"),
    [
        (Path("cohort_authority.json"), None),
        (None, _cohort_authority_ref()),
    ],
)
def test_acquisition_result_rejects_half_bound_cohort_authority_at_construction(
    authority_path, authority_ref
):
    with pytest.raises(
        MaterializedMetadataError,
        match="cohort authority path and reference must be present together",
    ):
        _acquisition_result(
            cohort_authority_path=authority_path,
            cohort_authority_ref=authority_ref,
        )


def test_acquisition_result_valid_authority_pair_preserves_serialization_contract():
    authority_path = Path("cohort_authority.json")
    authority_ref = _cohort_authority_ref()

    payload = _acquisition_result(
        cohort_authority_path=authority_path,
        cohort_authority_ref=authority_ref,
    ).to_dict()

    assert payload["cohort_authority_path"] == str(authority_path)
    assert payload["cohort_authority_ref"] == authority_ref.to_dict()


def test_acquisition_result_without_authority_preserves_legacy_serialization_contract():
    payload = _acquisition_result().to_dict()

    assert "cohort_authority_path" not in payload
    assert "cohort_authority_ref" not in payload
    assert "trajectory_path" not in payload
    assert "trajectory_authority_path" not in payload
    assert "trajectory_authority_ref" not in payload


@pytest.mark.parametrize(
    ("trajectory_path", "provenance_path"),
    [
        (Path("trajectory.parquet"), None),
        (None, Path("trajectory_provenance.json")),
    ],
)
def test_acquisition_result_rejects_half_bound_trajectory_artifact(
    trajectory_path,
    provenance_path,
):
    with pytest.raises(MaterializedTrajectoryError, match="path and provenance"):
        _acquisition_result(
            trajectory_path=trajectory_path,
            trajectory_provenance_path=provenance_path,
        )


@pytest.mark.parametrize(
    ("authority_path", "authority_ref"),
    [
        (Path("trajectory_authority.json"), None),
        (None, _trajectory_authority_ref()),
    ],
)
def test_acquisition_result_rejects_half_bound_trajectory_authority(
    authority_path,
    authority_ref,
):
    with pytest.raises(MaterializedTrajectoryError, match="present together"):
        _acquisition_result(
            trajectory_path=Path("trajectory.parquet"),
            trajectory_provenance_path=Path("trajectory_provenance.json"),
            trajectory_authority_path=authority_path,
            trajectory_authority_ref=authority_ref,
        )


def test_acquisition_result_rejects_trajectory_authority_without_artifact():
    with pytest.raises(MaterializedTrajectoryError, match="requires the selected"):
        _acquisition_result(
            trajectory_authority_path=Path("trajectory_authority.json"),
            trajectory_authority_ref=_trajectory_authority_ref(),
        )


def test_acquisition_result_serializes_complete_trajectory_authority():
    trajectory_path = Path("trajectory.parquet")
    provenance_path = Path("trajectory_provenance.json")
    authority_path = Path("trajectory_authority.json")
    authority_ref = _trajectory_authority_ref()

    payload = _acquisition_result(
        trajectory_path=trajectory_path,
        trajectory_provenance_path=provenance_path,
        trajectory_authority_path=authority_path,
        trajectory_authority_ref=authority_ref,
    ).to_dict()

    assert payload["trajectory_path"] == str(trajectory_path)
    assert payload["trajectory_provenance_path"] == str(provenance_path)
    assert payload["trajectory_authority_path"] == str(authority_path)
    assert payload["trajectory_authority_ref"] == authority_ref.to_dict()
