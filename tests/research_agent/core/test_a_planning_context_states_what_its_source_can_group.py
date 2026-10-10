"""A metadata-only planning context states what its source can group, checked.

A metadata-only planning context binds no patient grouping unless its design
reads one, so the host states beside it what the source can provide
(``contracts.patient_grouping_need``).  The context builder projects that
status into the cohort's provenance only when it is one closed value, says
``bound`` exactly when the catalog binds a replacement row identity, and
carries an invalid authority's typed code exactly beside
``authority_invalid``.  A catalog that states neither projects neither.

Synthetic catalogs only.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from easyicu.research_agent.contracts.patient_grouping_need import (
    PATIENT_GROUPING_AUTHORITY_ERROR_KEY,
    PATIENT_GROUPING_STATUS_KEY,
)
from easyicu.research_agent.intake.materialized_metadata import (
    MaterializedMetadataError,
)

STATUS = PATIENT_GROUPING_STATUS_KEY
ERROR = PATIENT_GROUPING_AUTHORITY_ERROR_KEY
_BRIDGE = {
    "output_identity_column": "patient_stay_id",
    "mapping_file_sha256": "a" * 64,
    "mapped_cohort_rows": 0,
    "patient_group_derivation": {"algorithm": "prefix_before_:s", "delimiter": ":s"},
    "authority_coordinates": {
        "schema_version": "easyicu.patient_grouping_runtime_authority/1",
        "authority_ref": "test/identity-bridge/v1",
        "database": "miiv",
        "mapping_sha256": "a" * 64,
        "grouping_derivation": "prefix_before_:s",
        "provider_visible_values": False,
    },
}


def _context(ra: Any, **authority: Any) -> Any:
    frame = pd.DataFrame(
        {
            "stay_id": pd.Series(dtype="int64"),
            "patient_stay_id": pd.Series(dtype="string"),
            "lact_max": pd.Series(dtype="float64"),
            "death": pd.Series(dtype="float64"),
        }
    )
    frame.attrs["easyicu_planning_authority"] = {
        "kind": "metadata_only_planning_catalog",
        "patient_rows_read": False,
        **authority,
    }
    return ra.build_research_context(
        research_question="Is lactate associated with hospital mortality?",
        cohort=frame,
        cohort_name="metadata-only",
        database="miiv",
        target_outcome="death",
        primary_exposure="lact_max",
        **(
            {"id_columns": ["patient_stay_id"]}
            if "replacement_row_identity" in authority
            else {}
        ),
    )


@pytest.mark.parametrize(
    "status", ["available_unbound", "not_carried_by_trajectory", "source_has_none"]
)
def test_an_unbound_status_reaches_the_cohort_provenance(ra: Any, status: str) -> None:
    provenance = _context(ra, **{STATUS: status}).cohort.provenance

    assert provenance[STATUS] == status
    assert ERROR not in provenance
    assert "replacement_row_identity" not in provenance


def test_bound_is_stated_beside_the_row_identity_it_binds(ra: Any) -> None:
    provenance = _context(
        ra, **{STATUS: "bound", "replacement_row_identity": _BRIDGE}
    ).cohort.provenance

    assert provenance[STATUS] == "bound"
    assert provenance["replacement_row_identity"]["output_identity_column"] == (
        "patient_stay_id"
    )


def test_an_invalid_authority_states_its_typed_code(ra: Any) -> None:
    provenance = _context(
        ra,
        **{STATUS: "authority_invalid", ERROR: "patient_grouping_authority_mapping_mismatch"},
    ).cohort.provenance

    assert provenance[STATUS] == "authority_invalid"
    assert provenance[ERROR] == "patient_grouping_authority_mapping_mismatch"


def test_a_catalog_that_states_nothing_projects_nothing(ra: Any) -> None:
    provenance = _context(ra).cohort.provenance

    assert STATUS not in provenance and ERROR not in provenance


@pytest.mark.parametrize(
    ("authority", "match"),
    [
        ({STATUS: "maybe"}, "not a known status"),
        ({ERROR: "patient_grouping_authority_mapping_mismatch"}, "not a known status"),
        ({STATUS: "bound"}, "row identity"),
        ({STATUS: "available_unbound", "replacement_row_identity": _BRIDGE}, "row identity"),
        ({STATUS: "authority_invalid"}, "authority error"),
        ({STATUS: "authority_invalid", ERROR: "  "}, "authority error"),
        ({STATUS: "authority_invalid", ERROR: 7}, "authority error"),
        ({STATUS: "source_has_none", ERROR: "patient_grouping_x"}, "authority error"),
        ({STATUS: "source_has_none", ERROR: ""}, "authority error"),
    ],
)
def test_a_status_that_disagrees_is_refused(
    ra: Any, authority: dict[str, Any], match: str
) -> None:
    with pytest.raises(MaterializedMetadataError, match=match):
        _context(ra, **authority)
