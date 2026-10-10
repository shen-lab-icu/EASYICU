"""A run plans, pauses and resumes on the exposure groups its study formed.

End to end with the offline graph and a typed synthetic export.  The host
plans exposure groupings (``PipelineConfig.enable_exposure_grouping``): the
Planner's grouping request is answered with one grouping of the first day's
lactate maximum, and every other request by the contextual mock.  The run's
cohort is restaged with the grouped column under the name its input capsule
seals, the paused plan's context names the grouped variable, and a new
process resumes the approved run without asking again.  A host that plans no
groupings asks nothing.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from easyicu.research_agent.authority.evidence_store import EvidenceStore
from easyicu.research_agent.cohort import materializer as cohort_materializer
from easyicu.research_agent.contracts.exposure_group_rules import (
    EXPOSURE_GROUP_CONTRASTS_KEY,
)
from easyicu.research_agent.intake.materialized_metadata import (
    EXPOSURE_GROUP_STAGE_PRODUCER,
    load_verified_materialized_cohort_authority,
)
from easyicu.research_agent.orchestration.human_review_checkpoint import (
    load_checkpoint,
)
from easyicu.research_agent.orchestration.workflow import HumanReviewDecision
from easyicu.research_agent.pipeline import ResearchAgentPipeline
from easyicu.research_agent.providers.mocks import (
    MockLLMClient,
    PatternScriptedMockLLMClient,
)
from tests.support.typed_export import typed_export

pytestmark = [pytest.mark.integration, pytest.mark.slow]

_ASKED = "Values the input holds that a grouping can read"
_CONTRASTS = EXPOSURE_GROUP_CONTRASTS_KEY
_QUESTION = "Compare hospital death across lactate groups in the first day."
_ANSWER = json.dumps(
    {
        "groupings": [
            {
                "id": "x1",
                "concept": "lact",
                "window": {"start_hours": 0, "end_hours": 24},
                "scale": "nominal",
                "groups": [
                    {
                        "id": "g1",
                        "label": "lactate below 2.5",
                        "rule": {
                            "summary": "max",
                            "op": "<",
                            "value": 2.5,
                            "unit": "mmol/L",
                        },
                    },
                    {"id": "g2", "label": "lactate 2.5 or above", "rule": "otherwise"},
                ],
                "unmeasured": {"handling": "own_group", "label": "not measured"},
                "reference": "g1",
                "contrast": "g2",
                "quote": "lactate groups",
                "source": "question",
            }
        ]
    }
)


def _approvable_plan_review(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the plan's own review request; drop unrelated mock findings."""

    import easyicu.research_agent.orchestration.workflow as workflow_module

    real = workflow_module.human_review_requests_for_plan

    def plan_only(**kwargs):
        request = dict(kwargs)
        request["findings"] = []
        request["require_plan_review"] = True
        return real(**request)

    monkeypatch.setattr(workflow_module, "human_review_requests_for_plan", plan_only)


def _universe(tmp_path: Path) -> Path:
    paths = cohort_materializer.materialize_to_parquet(
        tmp_path / "materialized",
        data_path=typed_export(tmp_path / "export"),
        database="miiv",
        static_concepts=("age",),
        feature_concepts=("lact",),
        outcome_concepts=("death",),
    )
    return paths["parquet"]


def _pipeline(root: Path, llm, **overrides) -> ResearchAgentPipeline:
    options = {
        "workdir": root,
        "llm": llm,
        "require_human_plan_review": True,
        "enable_visual_qa": False,
        "enable_publication_figure_skill": False,
        "enable_nature_writing_skill": False,
        "enable_exposure_grouping": True,
    }
    options.update(overrides)
    return ResearchAgentPipeline(**options)


def _asked(llm) -> int:
    return sum(
        1
        for messages, _options in llm.calls
        if any(_ASKED in str(message.content or "") for message in messages)
    )


def test_a_run_plans_and_resumes_on_the_groups_its_study_formed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _approvable_plan_review(monkeypatch)
    llm = PatternScriptedMockLLMClient([(_ASKED, [_ANSWER])], contextual_default=True)
    workdir = tmp_path / "runs"

    pending = _pipeline(workdir, llm).run(
        question=_QUESTION, cohort=_universe(tmp_path), target_outcome="death"
    )

    assert _asked(llm) == 1
    run_dir = Path(pending.run_dir)
    cohort = run_dir / "cohort.parquet"
    staged = load_verified_materialized_cohort_authority(cohort)
    assert staged is not None
    assert staged.authority.producer == EXPOSURE_GROUP_STAGE_PRODUCER
    capsule = json.loads((run_dir / "run_input_capsule.json").read_text())
    assert capsule["cohort_sha256"] == hashlib.sha256(cohort.read_bytes()).hexdigest()
    handoff = load_checkpoint(run_dir / "human_review_checkpoint.json").plan_handoff
    assert "lact_group_x1" in [item["name"] for item in handoff["context"]["variables"]]
    # The plan compares the groups the study stated, as their level codes.
    compared = [{"variable": "lact_group_x1", "reference": 1, "contrast": 2}]
    assert handoff["context"]["cohort"]["provenance"][_CONTRASTS] == compared
    assert EvidenceStore(root=run_dir).get("exposure_groupings") is not None

    # A fresh object stands in for a restarted host approving the paused plan;
    # the sealed groups are read, never asked for again.
    again = PatternScriptedMockLLMClient([(_ASKED, [])], contextual_default=True)
    result = _pipeline(workdir, again).resume_human_review(
        [
            HumanReviewDecision(
                review_id=request.review_id,
                authority_sha256=request.authority_sha256,
                decision="approved",
                reviewer="test reviewer",
                decided_at="2026-10-09T23:00:00Z",
            )
            for request in pending.requests
        ],
        run_id=pending.run_id,
    )

    assert result.run_id == pending.run_id
    assert _asked(again) == 0
    resumed = json.loads((run_dir / "research_context.json").read_text())
    assert resumed["cohort"]["provenance"][_CONTRASTS] == compared
    assert load_verified_materialized_cohort_authority(cohort).reference == (
        staged.reference
    )


def test_a_host_that_plans_no_groupings_asks_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _approvable_plan_review(monkeypatch)
    llm = MockLLMClient()

    pending = _pipeline(tmp_path / "runs", llm, enable_exposure_grouping=None).run(
        question=_QUESTION, cohort=_universe(tmp_path), target_outcome="death"
    )

    assert _asked(llm) == 0
    staged = load_verified_materialized_cohort_authority(
        Path(pending.run_dir) / "cohort.parquet"
    )
    assert staged is not None
    assert staged.authority.producer == "research_agent_run_stage"
