"""A descriptive study reports counts only when its researcher's words ask for it.

A descriptive design that reports counts without any interval
(``none_counts_only``) leaves a proportion question without its confidence
interval.  The study setup keeps counts only when the researcher's own words
ask for them (``design_variance_basis.counts_only_request``); counts only the
turn proposes, or the host's descriptive default would have recorded, on no
such request are replaced by the variance the source's patient grouping
supports (``study_family_design.descriptive_variance``), and the record says
why.  Counts only the study already records stay, with the conflict shown on
the study card; nothing stops.  Synthetic questions and studies only.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from easyicu.webserver import source_identity_authority
from easyicu.webserver import study_contexts as context_store
from easyicu.webserver.design_variance_basis import (
    DESIGN_VARIANCE_BASIS_FIELD,
    counts_only_request,
    normalize_design_variance_basis,
)
from easyicu.webserver.pi_copilot import tools as tool_module
from easyicu.webserver.pi_copilot.contracts import (
    AuthorityBinding,
    PiSessionRecord,
    ToolExecutionContext,
)
from easyicu.webserver.study_family_design import descriptive_variance

_QUESTION = (
    "Among adult ICU stays, what proportion had a serum sodium below 130 mmol/L in the "
    "first 24 hours, and how did hospital mortality differ between those stays and the rest?"
)
_COUNTS_ONLY = {
    "analysis_family": "descriptive_epidemiology",
    "analysis_unit": "icu_stay",
    "variance_estimator": "none_counts_only",
}
_CLUSTERED = {
    "analysis_family": "descriptive_epidemiology",
    "analysis_unit": "icu_stay",
    "variance_estimator": "cluster_robust",
    "cluster_unit": "patient",
}
_MODEL_BASED = {
    "analysis_family": "descriptive_epidemiology",
    "analysis_unit": "icu_stay",
    "variance_estimator": "model_based",
}


def _bound(tmp_path) -> dict[str, Any]:
    return {"data_source": {"path": str(tmp_path / "export"), "database": "miiv"}}


def _grouping(monkeypatch: pytest.MonkeyPatch, grouped: bool) -> None:
    binding = SimpleNamespace(output_identity_column="patient_stay_id") if grouped else None
    monkeypatch.setattr(
        source_identity_authority,
        "resolve_study_patient_grouping",
        lambda **_kwargs: binding,
    )


def _current(**fields: Any) -> dict[str, Any]:
    return {"id": "study-counts-only", "revision": 3, "active_job_id": None, **fields}


def _update(monkeypatch, current: dict[str, Any], proposal: dict[str, Any], message: str):
    writes: list[dict[str, Any]] = []
    monkeypatch.setattr(tool_module, "_bound_context", lambda _binding: dict(current))
    monkeypatch.setattr(
        tool_module.study_contexts,
        "upsert_context",
        lambda raw, **_kwargs: writes.append(dict(raw))
        or {**current, **raw, "revision": current["revision"] + 1},
    )
    session = PiSessionRecord(
        session_id="pi-counts-only",
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


@pytest.mark.parametrize("grouped", [True, False])
def test_counts_only_nobody_asked_for_take_the_sources_variance(tmp_path, monkeypatch, grouped) -> None:
    _grouping(monkeypatch, grouped)
    current = _current(**_bound(tmp_path), question=_QUESTION, analysis_design={})

    result, writes = _update(
        monkeypatch, current, {"analysis_design": dict(_COUNTS_ONLY)}, "这是一项描述性研究"
    )

    assert result["code"] == "study_context_updated"
    assert writes[-1]["analysis_design"] == (_CLUSTERED if grouped else _MODEL_BASED)
    record = writes[-1][DESIGN_VARIANCE_BASIS_FIELD]
    assert record == {
        "variance_estimator": "cluster_robust" if grouped else "model_based",
        "basis": "source_patient_grouping" if grouped else "source_without_patient_grouping",
        "replaced": "none_counts_only",
    }
    assert result["details"][DESIGN_VARIANCE_BASIS_FIELD] == record
    assert "Do not propose counts only" in result["summary"]


@pytest.mark.parametrize(
    "message",
    [
        "只报告人数和比例，不需要置信区间。",
        "这是描述性研究，不做统计推断，只计数。",
        "Counts only, please: no confidence intervals.",
        "Report the proportions without confidence intervals.",
        "Counts only, please.",
        "Please report only the counts in each group.",
    ],
)
def test_counts_only_the_researchers_words_ask_for_are_kept(tmp_path, monkeypatch, message) -> None:
    _grouping(monkeypatch, True)
    current = _current(**_bound(tmp_path), question=_QUESTION, analysis_design={})

    result, writes = _update(monkeypatch, current, {"analysis_design": dict(_COUNTS_ONLY)}, message)

    assert writes[-1]["analysis_design"] == _COUNTS_ONLY
    record = writes[-1][DESIGN_VARIANCE_BASIS_FIELD]
    assert record["basis"] == "user_words" and record["variance_estimator"] == "none_counts_only"
    assert record["evidence"] in message
    assert "as the researcher asked" in result["summary"]


@pytest.mark.parametrize(
    "message",
    ["不需要回归，也不做校正。", "No adjustment and no regression model.", "这是一项描述性研究"],
)
def test_declining_a_model_does_not_ask_for_counts_only(message) -> None:
    assert counts_only_request(message) is None


def test_the_descriptive_default_takes_the_same_rule(tmp_path, monkeypatch) -> None:
    _grouping(monkeypatch, True)
    question = "估计成人 ICU 住院中低钠血症的患病率，并描述其与院内死亡的关系。"
    current = _current(**_bound(tmp_path), analysis_design={})

    _, writes = _update(monkeypatch, current, {"question": question}, question)
    assert writes[-1]["analysis_design"] == _CLUSTERED
    assert writes[-1][DESIGN_VARIANCE_BASIS_FIELD]["replaced"] == "none_counts_only"

    asked = question + "只计数即可。"
    _, writes = _update(monkeypatch, current, {"question": asked}, asked)
    assert writes[-1]["analysis_design"] == _COUNTS_ONLY
    assert writes[-1][DESIGN_VARIANCE_BASIS_FIELD]["evidence"] == "只计数"


def test_a_recorded_counts_only_design_is_kept_and_its_conflict_shown(tmp_path, monkeypatch) -> None:
    _grouping(monkeypatch, True)
    current = _current(**_bound(tmp_path), question=_QUESTION, analysis_design=dict(_COUNTS_ONLY))

    result, writes = _update(monkeypatch, current, {"comparator": "the remaining stays"}, "继续")

    assert writes[-1].get("analysis_design", _COUNTS_ONLY) == _COUNTS_ONLY
    assert writes[-1][DESIGN_VARIANCE_BASIS_FIELD] == {
        "variance_estimator": "none_counts_only",
        "conflict": "counts_only_not_requested",
    }
    assert "the design is kept and the study card shows it" in result["summary"]

    # Counts only the researcher asked for earlier show no conflict.
    asked = {"variance_estimator": "none_counts_only", "basis": "user_words", "evidence": "只计数"}
    current = {**current, DESIGN_VARIANCE_BASIS_FIELD: asked}
    result, writes = _update(monkeypatch, current, {"comparator": "the remaining stays"}, "继续")
    assert DESIGN_VARIANCE_BASIS_FIELD not in writes[-1]
    assert DESIGN_VARIANCE_BASIS_FIELD not in result["details"]


def test_an_association_design_with_counts_only_is_still_refused(tmp_path, monkeypatch) -> None:
    _grouping(monkeypatch, True)
    current = _current(**_bound(tmp_path), question=_QUESTION, analysis_design={})
    proposal = {"analysis_design": {**_COUNTS_ONLY, "analysis_family": "association_study"}}

    result, writes = _update(monkeypatch, current, proposal, "这是一项关联研究")

    # The family ceiling refuses it; the variance rule does not rewrite it.
    assert result["code"] == "analysis_design_counts_only_family_conflict"
    assert writes == []


def test_only_a_descriptive_design_with_counts_only_is_decided() -> None:
    study = {"data_source": {"database": "miiv"}}
    association = {**_COUNTS_ONLY, "analysis_family": "association_study"}
    for design in (association, _MODEL_BASED, _CLUSTERED, None):
        assert descriptive_variance(study, design=design, saved=None, message="") is None


def test_the_record_is_the_hosts_and_is_cleared_with_its_variance() -> None:
    record = {"variance_estimator": "none_counts_only", "conflict": "counts_only_not_requested"}
    saved = context_store.upsert_context(
        {
            "id": "study-counts-only-stale",
            "question": _QUESTION,
            "analysis_design": dict(_COUNTS_ONLY),
            DESIGN_VARIANCE_BASIS_FIELD: record,
        },
        _server_design_variance_basis_write=True,
    )
    assert saved[DESIGN_VARIANCE_BASIS_FIELD] == record
    assert context_store.get_context("study-counts-only-stale")[DESIGN_VARIANCE_BASIS_FIELD] == record

    with pytest.raises(context_store.StudyContextError) as refused:
        context_store.upsert_context(
            {"id": "study-counts-only-stale", DESIGN_VARIANCE_BASIS_FIELD: record}
        )
    assert refused.value.detail["error"] == "design_variance_basis_server_owned"

    changed = context_store.upsert_context(
        {"id": "study-counts-only-stale", "analysis_design": dict(_MODEL_BASED)}
    )
    assert changed[DESIGN_VARIANCE_BASIS_FIELD] is None


@pytest.mark.parametrize(
    "value",
    [
        {"variance_estimator": "model_based", "basis": "user_words", "evidence": "只计数"},
        {"variance_estimator": "none_counts_only", "basis": "user_words"},
        {"variance_estimator": "none_counts_only", "basis": "source_patient_grouping", "replaced": "none_counts_only"},
        {"variance_estimator": "cluster_robust", "basis": "source_patient_grouping"},
        {"variance_estimator": "model_based", "conflict": "counts_only_not_requested"},
        {"variance_estimator": "none_counts_only", "basis": "a model's choice"},
    ],
)
def test_a_record_states_one_of_its_closed_shapes(value) -> None:
    with pytest.raises(ValueError):
        normalize_design_variance_basis(value)
