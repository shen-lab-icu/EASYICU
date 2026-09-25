"""A run whose cohort filters one of its own materialized columns completes.

End to end with a scripted provider.  The Planner's cohort names a column of
the run's cohort that is no dictionary concept, as a package-bound plan names
``<concept>_max``.  Planning, the probe replan, execution and finalization
validate that cohort again, some after the run's concept scope closed, and a
development replay reads the locked plan once more.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from easyicu.research_agent.planning.cohort_contract import (
    cohort_concept_id_scope,
    concept_id_exists,
)
from easyicu.research_agent.providers.mocks import (
    PatternScriptedMockLLMClient,
    ScriptedMockLLMClient,
)
from tests.support.pipeline_contracts import (
    disable_article_contract,
    stable_plan_rules,
)

pytestmark = [pytest.mark.integration, pytest.mark.slow]

COLUMN = "fixture_flag_max"
QUESTION = "Does age describe ICU mortality among flagged stays?"
COHORT = {
    "name": "flagged_stays",
    "inclusion": [
        {
            "concept_id": COLUMN,
            "time_window": {
                "anchor": "icu_admission",
                "start_offset_hours": 0.0,
                "end_offset_hours": 24.0,
            },
            "aggregation": "any",
            "op": "==",
            "value": 1.0,
        }
    ],
    "exclusion": [],
    "selection_mode": "predicate_filtered",
}
STEP = {
    "step_id": "01_table_one",
    "planned_analysis_role": "auxiliary",
    "intent": "Write a compact cohort table.",
    "inputs": ["death"],
    "expected_outputs": ["table:table_one"],
    "method": "descriptive",
    "icu_rule_refs": ["aggregation_rule_for"],
}
CODE = """
import json
import os
import pandas as pd

df = pd.read_parquet(os.environ["COHORT_PARQUET"])
out = os.environ["STEP_OUT_DIR"]
pd.DataFrame({"n": [int(len(df))]}).to_csv(os.path.join(out, "table_one.csv"), index=False)
summary = {"n": int(len(df)), "output_files": {"table:table_one": "table_one.csv"}}
with open(os.path.join(out, "step_summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f)
print(json.dumps(summary))
"""


def _cohort() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    n = 60
    return pd.DataFrame(
        {
            "stay_id": np.arange(1, n + 1),
            "age": rng.integers(30, 90, n),
            "death": rng.integers(0, 2, n),
            COLUMN: np.r_[np.ones(40), np.zeros(20)],
        }
    )


def test_a_run_filtering_its_own_column_completes_and_binds_its_plan(
    ra, tmp_path: Path, monkeypatch
) -> None:
    disable_article_contract(monkeypatch)
    plan = json.dumps(
        {
            "research_question": QUESTION,
            "cohort": COHORT,
            "steps": [STEP],
            "rationale": "Describe the flagged stays.",
        }
    )
    llm = PatternScriptedMockLLMClient(
        [
            *stable_plan_rules(plan),
            ("WRITE THE PYTHON CODE FOR STEP", [CODE] * 8),
            ("REPAIR THE PYTHON CODE FOR STEP", [CODE] * 8),
            (
                "INTERPRET THE RESULTS OF STEP",
                ["The cohort table was produced {evidence:table_one}."] * 8,
            ),
            (
                "WRITE A MANUSCRIPT SCAFFOLD",
                [
                    "# Title\n\n## Results\n\nThe cohort table was produced "
                    "{evidence:table_one}.\n\n(left to the human author)"
                ]
                * 8,
            ),
        ]
    )
    pipeline = ra.ResearchAgentPipeline(
        config=ra.PipelineConfig(workdir=tmp_path, enable_literature=False),
        services=ra.PipelineServices(llm=llm),
    )

    result = pipeline.run(
        question=QUESTION,
        cohort=_cohort(),
        cohort_name="flagged",
        database="synthetic",
        target_outcome="death",
    )

    run_dir = Path(result.workdir)
    registered = json.loads(Path(result.plan_path).read_text(encoding="utf-8"))
    assert [p["concept_id"] for p in registered["cohort"]["inclusion"]] == [COLUMN]
    manifest = json.loads(Path(result.manifest_path).read_text(encoding="utf-8"))
    assert manifest["current_plan_authority"]["evidence_id"] == "analysis_plan"
    partial = json.loads(
        (run_dir / "manifest_partial.json").read_text(encoding="utf-8")
    )
    assert "current_plan_authority_error" not in partial
    analysis = json.loads(
        (run_dir / "cohort_analysis_provenance.json").read_text(encoding="utf-8")
    )
    assert analysis["n_analysis_cohort"] == 40
    assert not concept_id_exists(COLUMN)


def test_a_development_replay_of_such_a_plan_reaches_review(
    ra, tmp_path: Path
) -> None:
    with cohort_concept_id_scope([COLUMN]):
        plan = ra.AnalysisPlan(
            research_question=QUESTION,
            cohort=COHORT,
            steps=[STEP],
            rationale="Describe the flagged stays.",
        )
    plan_path = tmp_path / "locked_plan.json"
    plan_path.write_text(plan.model_dump_json(indent=2), encoding="utf-8")
    client = ScriptedMockLLMClient([])
    pipeline = ra.ResearchAgentPipeline(
        config=ra.PipelineConfig(
            workdir=tmp_path / "replay",
            planner_only=True,
            development_diagnostic=True,
            require_human_plan_review=True,
            development_locked_analysis_plan_path=plan_path,
            development_locked_analysis_plan_sha256=hashlib.sha256(
                plan_path.read_bytes()
            ).hexdigest(),
            enable_memory=False,
            enable_replanning=False,
            enable_literature=False,
        ),
        services=ra.PipelineServices(llm=client),
    )

    outcome = pipeline.run(
        question=QUESTION,
        cohort=_cohort(),
        cohort_name="flagged",
        database="synthetic",
        target_outcome="death",
        stop_after_analysis=True,
    )

    assert type(outcome).__name__ == "HumanReviewPending"
    assert not client.calls
    assert not concept_id_exists(COLUMN)
