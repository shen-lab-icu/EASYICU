"""An outcome-free phenotype-discovery question gets a clustering plan.

The run used to stop before planning because the hypothesis blueprint asked
every question for a target outcome.  Discovery asks how stays group, so the
pipeline must plan it in the clustering family.  Synthetic stays only; the
mock Provider stands in for the Planner.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.slow]


def test_an_outcome_free_discovery_question_is_planned(ra, tmp_path: Path):
    cohort = pd.DataFrame(
        {
            "stay_id": range(1, 61),
            "age": [45 + (i % 25) for i in range(60)],
            "lact_t0": [1.2 + (i % 5) * 0.3 for i in range(60)],
            "lact_t6": [1.1 + (i % 5) * 0.25 for i in range(60)],
            "map_t0": [75 + (i % 7) * 2 for i in range(60)],
            "map_t6": [78 + (i % 7) * 2 for i in range(60)],
        }
    )
    pipeline = ra.ResearchAgentPipeline(workdir=tmp_path, llm=ra.MockLLMClient())

    result = pipeline.run(
        question=(
            "Do first-day lactate and blood-pressure trajectories cluster into "
            "distinct subgroups of ICU patients?"
        ),
        cohort=cohort,
        cohort_name="outcome_free_discovery",
        database="synthetic",
    )

    manifest = json.loads(Path(result.manifest_path).read_text(encoding="utf-8"))
    assert manifest["notes"] != "aborted: hypothesis_blueprint_blocked"
    assert not any(
        finding["validator"] == "hypothesis_blueprint" for finding in manifest["findings"]
    )
    blueprint = json.loads(
        (Path(result.manifest_path).parent / "hypothesis_blueprint.json").read_text(
            encoding="utf-8"
        )
    )
    assert blueprint["feasibility_status"] == "ready"
    plan = json.loads(Path(result.plan_path).read_text(encoding="utf-8"))
    assert plan["analysis_type"] == "trajectory_clustering"
    assert plan["steps"]
