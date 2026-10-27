"""A trajectory design on measurements counts the windows its values were recorded in.

The signed representation owner counted a stay's eligible windows on SOFA-2
evidence only, so a design on charted vital signs or laboratory values had no
eligible stay.  A design without a SOFA-2 coordinate now counts a window when
any coordinate has a recorded value in it, and its rule outcome, method text
and exclusion reason say so; a SOFA-2 design is counted and worded as before.
Synthetic panel: one heart-rate coordinate over 0-48 h on an 8-hour grid.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from easyicu.research_agent.authority.manuscript_method_facts import _design_text
from easyicu.research_agent.authority.scientific_claims import (
    derive_scientific_claim_drafts,
)
from easyicu.research_agent.contracts.executed_method_design import (
    validate_executed_method_design,
)
from easyicu.research_agent.contracts.trajectory_design import (
    load_trajectory_design,
    sealed_trajectory_authority_body,
)
from easyicu.research_agent.execution.runners.trajectory_scientific_representation_executor import (
    run_trajectory_scientific_representation,
)
from easyicu.research_agent.trajectory.scientific_runtime_authority import (
    build_trajectory_scientific_runtime_authority,
)

STARTS = (0, 8, 16, 24, 32, 40)


def _authority():
    design = load_trajectory_design(
        {
            "coordinate_concepts": ["hr"],
            "descriptive_only_concepts": [],
            "window_start_hours": 0,
            "window_end_hours": 48,
            "grid_width_hours": 8,
            "minimum_available_windows": 3,
        }
    )
    return build_trajectory_scientific_runtime_authority(
        sealed_trajectory_authority_body(design, protocol_content_sha256="3" * 64)
    )


def _panel_rows(stays) -> list[dict]:
    rng = np.random.default_rng(7)
    rows = []
    for stay in stays:
        # Stays 1-6 have heart rate recorded in two windows only.
        recorded = STARTS[:2] if stay <= 6 else STARTS
        for start in recorded:
            rows.append({"stay_id": stay, "charttime": start + 2.0, "concept": "hr",
                         "value_num": float(rng.normal(90.0, 12.0)),
                         "evidence_state": "direct_observed",
                         "owner_observed": 1, "owner_available": 1})
    return rows


def test_a_measured_design_counts_recorded_windows_and_says_so(tmp_path: Path) -> None:
    panel = tmp_path / "panel.parquet"
    pd.DataFrame(_panel_rows(range(1, 41))).to_parquet(panel, index=False)
    cohort = tmp_path / "cohort.parquet"
    pd.DataFrame({"stay_id": range(1, 41)}).to_parquet(cohort, index=False)

    summary = run_trajectory_scientific_representation(
        authority=_authority(), runtime_projection_sha256="4" * 64,
        trajectory_path=panel, cohort_path=cohort, out_dir=tmp_path / "out",
    )

    [outcome] = summary["reportable_rule_outcomes"]
    assert outcome["window_evidence"] == "any_recorded_coordinate_value"
    assert (outcome["input_n"], outcome["included_n"], outcome["excluded_n"]) == (40, 34, 6)
    design = summary["executed_method_design"]
    assert design["window_evidence"] == "any_recorded_coordinate_value"
    membership = pd.read_csv(tmp_path / "out" / "trajectory_membership.csv")
    excluded = membership.loc[membership["stay_id"].le(6), "exclusion_reason"]
    assert excluded.eq("fewer_than_signed_minimum_recorded_coordinate_windows").all()
    clustered = pd.read_parquet(tmp_path / "out" / "trajectory_representation.parquet")
    assert set(clustered["stay_id"]) == set(range(7, 41))

    [draft] = derive_scientific_claim_drafts(summary)
    assert draft.rule_outcome.result_sentence() == (
        "Of 40 records in the study cohort, 34 had at least one recorded value of a "
        "model coordinate in at least 3 of the 6 prespecified 8-hour windows from 0 "
        "to 48 hours after ICU admission and entered the class model; 6 were excluded."
    )
    assert "SOFA-2" not in draft.rule_outcome.conclusion_sentence()
    method = _design_text(validate_executed_method_design(design))
    assert "at least one model coordinate was recorded in at least 3 windows" in method
    assert "SOFA-2" not in method
