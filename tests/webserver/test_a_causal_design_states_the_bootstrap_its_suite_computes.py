"""A causal design states the bootstrap its suite computes.

The target trial suite reports percentile-bootstrap intervals: it resamples
stays, or patients with all their stays when a patient may contribute
several.  A causal study states that estimator.  With repeat ICU stays kept,
a bootstrap of stays leaves within-patient dependence open, and the store
names the patient bootstrap that closes it.  A patient bootstrap reads the
verified patient grouping at launch, as cluster-robust variance does, and
binds no model's clustering into the plan.  A researcher who names the
bootstrap in conversation has it written, as any variance choice they name;
one they did not name leaves the estimator they had.  Synthetic studies
only.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from easyicu.research_agent.planning.dependence_authority import (
    DependenceAuthorityError,
    context_dependence_authority,
)
from easyicu.webserver import research_launch_scientific, source_identity_authority
from easyicu.webserver import study_contexts as context_store
from easyicu.webserver.pi_copilot import tools as tool_module
from easyicu.webserver.pi_copilot.contracts import (
    AuthorityBinding,
    PiSessionRecord,
    ToolExecutionContext,
)
from easyicu.webserver.research_pipeline_run_errors import ResearchPipelineRunError

_STAYS = {
    "analysis_family": "causal_inference",
    "analysis_unit": "icu_stay",
    "variance_estimator": "bootstrap",
}
_PATIENTS = {**_STAYS, "cluster_unit": "patient"}
_REPEAT_STAYS = {"label": "All adult ICU stays", "exclude_readmissions": False}


def _context(design: dict[str, str]) -> Any:
    return SimpleNamespace(
        user_preferences=SimpleNamespace(
            data_constraints=json.dumps({"analysis_design": design})
        )
    )


def _study(design: dict[str, str], tmp_path: Path) -> dict[str, Any]:
    return {
        "analysis_design": dict(design),
        "data_source": {"path": str(tmp_path / "export"), "database": "miiv"},
    }


def test_a_causal_design_states_a_bootstrap_of_stays_or_of_patients() -> None:
    for design in (_STAYS, _PATIENTS):
        assert context_store.normalize_analysis_design(dict(design)) == design
        # The suite reads the grouping it resamples from the context; no
        # model step of the plan clusters.
        assert context_dependence_authority(_context(design)) is None


@pytest.mark.parametrize("unit", ["hospital_admission", "site", "custom"])
def test_a_bootstrap_resamples_no_unit_but_the_patient(unit: str) -> None:
    design = {**_STAYS, "cluster_unit": unit}

    with pytest.raises(context_store.StudyContextError) as setup:
        context_store.normalize_analysis_design(design)
    assert setup.value.detail == {
        "error": "study_cluster_unit_unsupported",
        "field": "analysis_design.cluster_unit",
        "allowed": ["patient"],
    }
    with pytest.raises(DependenceAuthorityError) as binding:
        context_dependence_authority(_context(design))
    assert binding.value.code == "analysis_dependence_contract_invalid"


@pytest.mark.parametrize("estimator", ["bootstrap", "model_based"])
def test_kept_repeat_stays_need_the_patient_bootstrap(estimator: str) -> None:
    design = {**_STAYS, "variance_estimator": estimator}

    finding = context_store.analysis_dependence_finding(
        {"cohort": dict(_REPEAT_STAYS), "analysis_design": design}
    )

    assert finding is not None
    assert finding["error"] == "study_repeated_stay_dependence_unaddressed"
    assert finding["required_design"] == {
        "variance_estimator": "bootstrap",
        "cluster_unit": "patient",
    }
    assert (
        context_store.analysis_dependence_finding(
            {"cohort": dict(_REPEAT_STAYS), "analysis_design": dict(_PATIENTS)}
        )
        is None
    )


def test_another_family_keeps_its_cluster_robust_closure() -> None:
    design = {
        "analysis_family": "association_study",
        "analysis_unit": "icu_stay",
        "variance_estimator": "model_based",
    }

    finding = context_store.analysis_dependence_finding(
        {"cohort": dict(_REPEAT_STAYS), "analysis_design": design}
    )

    assert finding is not None
    assert finding["required_design"] == {
        "variance_estimator": "cluster_robust",
        "cluster_unit": "patient",
    }


@pytest.mark.parametrize(
    ("design", "reads"),
    [
        ({"variance_estimator": "model_based"}, False),
        ({"variance_estimator": "none_counts_only"}, False),
        (_STAYS, False),
        (_PATIENTS, True),
        ({"variance_estimator": "cluster_robust", "cluster_unit": "patient"}, True),
        # The launch refuses a unit other than the patient after reading it.
        ({"variance_estimator": "cluster_robust", "cluster_unit": "site"}, True),
    ],
)
def test_which_designs_read_the_patient_grouping(
    design: dict[str, str], reads: bool
) -> None:
    assert context_store.analysis_design_reads_patient_grouping(design) is reads


def test_a_bootstrap_of_stays_launches_as_stated(tmp_path: Path) -> None:
    assert research_launch_scientific.validate_analysis_design_for_execution(
        _study(_STAYS, tmp_path)
    ) == {"analysis_unit": "icu_stay", "variance_estimator": "bootstrap"}


def test_a_patient_bootstrap_launches_on_the_verified_patient_grouping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    resolved: list[tuple[str, str]] = []

    def resolve(*, export_path: str, database: str) -> Any:
        resolved.append((export_path, database))
        return SimpleNamespace(output_identity_column="patient_stay_id")

    monkeypatch.setattr(
        source_identity_authority, "resolve_study_patient_grouping", resolve
    )
    study = _study(_PATIENTS, tmp_path)

    assert research_launch_scientific.validate_analysis_design_for_execution(
        study
    ) == {
        "analysis_unit": "icu_stay",
        "variance_estimator": "bootstrap",
        "cluster_unit": "patient",
        "grouping_coordinate": "patient_stay_id",
    }
    assert resolved == [(str(tmp_path / "export"), "miiv")]

    # Without a verified grouping the launch refuses, as it does for
    # cluster-robust variance.
    monkeypatch.setattr(
        source_identity_authority,
        "resolve_study_patient_grouping",
        lambda **_kwargs: None,
    )
    with pytest.raises(ResearchPipelineRunError) as refused:
        research_launch_scientific.validate_analysis_design_for_execution(study)
    assert refused.value.code == "research_pipeline_cluster_variance_unsupported"
    assert refused.value.details["variance_estimator"] == "bootstrap"


def _current(**fields: Any) -> dict[str, Any]:
    return {
        "id": "study-causal-bootstrap",
        "revision": 2,
        "question": "Does starting a vasopressor early change death by day 28?",
        "active_job_id": None,
        **fields,
    }


def _update(
    monkeypatch: pytest.MonkeyPatch,
    current: dict[str, Any],
    proposal: dict[str, Any],
    message: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run the conversation's study update with the researcher's message."""

    writes: list[dict[str, Any]] = []
    monkeypatch.setattr(tool_module, "_bound_context", lambda _binding: dict(current))
    monkeypatch.setattr(
        tool_module.study_contexts,
        "upsert_context",
        lambda raw, **_kwargs: writes.append(dict(raw))
        or {**raw, "revision": current["revision"] + 1},
    )
    session = PiSessionRecord(
        session_id="pi-causal-bootstrap",
        binding=AuthorityBinding(
            study_context_id=current["id"], study_revision=current["revision"]
        ),
    )
    result = tool_module.execute_tool(
        "easyicu_update_study_context",
        proposal,
        ToolExecutionContext(
            session=session, user_message=message, allowed_actions={"configure"}
        ),
    )
    return result, writes


_MODEL_BASED = {**_STAYS, "variance_estimator": "model_based"}


@pytest.mark.parametrize(
    "message",
    [
        "请用 bootstrap 区间报告这个效应。",
        "Report the effect with bootstrap intervals.",
        "区间改用自助法。",
        "效应的区间用重抽样来估计。",
    ],
)
def test_a_researcher_who_names_the_bootstrap_has_it_written(
    monkeypatch: pytest.MonkeyPatch, message: str
) -> None:
    current = _current(cohort={"preset": "all_icu"}, analysis_design=dict(_MODEL_BASED))

    result, writes = _update(
        monkeypatch, current, {"analysis_design": dict(_STAYS)}, message
    )

    assert result["code"] == "study_context_updated"
    assert writes[-1]["analysis_design"] == _STAYS


def test_an_estimator_the_researcher_did_not_name_stays_as_it_was(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current = _current(cohort={"preset": "all_icu"}, analysis_design=dict(_MODEL_BASED))

    result, writes = _update(
        monkeypatch,
        current,
        {"analysis_design": dict(_STAYS)},
        "把结局改成 28 天死亡。",
    )

    assert result["code"] == "study_context_updated"
    assert writes[-1]["analysis_design"]["variance_estimator"] == "model_based"


def test_a_patient_bootstrap_is_written_only_when_the_researcher_names_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current = _current(cohort={"preset": "all_icu"}, analysis_design={})
    proposal = {"analysis_design": dict(_PATIENTS)}

    blocked, writes = _update(
        monkeypatch,
        current,
        proposal,
        "纳入所有符合条件的成人 ICU stays，包括重复 ICU 入住。",
    )
    assert blocked["code"] == "study_patient_clustering_confirmation_required"
    assert writes == []

    # Named, it reaches the launch's grouping check, which this synthetic
    # study, bound to no source, cannot pass.
    for message in ("因果效应的区间按患者重抽样。", "Bootstrap by patient, please."):
        checked, writes = _update(monkeypatch, current, proposal, message)
        assert checked["code"] == "research_pipeline_cluster_variance_unsupported"
        assert writes == []
