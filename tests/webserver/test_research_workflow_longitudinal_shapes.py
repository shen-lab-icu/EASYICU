"""Longitudinal workflows show their family-owned step graphs, never a generic chain."""

from __future__ import annotations

import pytest

from easyicu.webserver.research_workflow_shapes import canonical_workflow_shape
from easyicu.webserver.research_workflows import research_workflow_catalog

EXPECTED = {
    "target-trial-emulation": ("target_trial_emulation", "target_trial_suite"),
    "time-varying-exposure-survival-study": ("time_varying_exposure_survival", "time_varying_cox"),
    "dynamic-landmark-prediction": ("dynamic_landmark_prediction", "model_fit"),
}


@pytest.mark.parametrize("skill_id", sorted(EXPECTED))
def test_a_longitudinal_workflow_publishes_its_family_graph(skill_id):
    family, primary = EXPECTED[skill_id]
    row = next(item for item in research_workflow_catalog(enabled=True)["items"] if item["id"] == skill_id)
    assert row["graph_source"] == family and row["canonical"] is True
    ids = [step["id"] for step in row["steps"]]
    assert ids[0] == "cohort_accounting" and primary in ids and ids[-1] == "report"
    # Every step is connected; the graph is not a list of unrelated boxes.
    linked = {edge["source"] for edge in row["edges"]} | {edge["target"] for edge in row["edges"]}
    assert set(ids) <= linked
    # The catalog's generic package chain is not what is shown.
    assert "exposure_outcome_distribution" not in ids


def test_unknown_workflows_keep_no_canonical_shape():
    assert canonical_workflow_shape("not-a-workflow") is None
