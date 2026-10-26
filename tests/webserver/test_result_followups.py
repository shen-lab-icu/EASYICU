"""Completed synthetic results offer only host-compiled, source-bound next steps."""
from types import SimpleNamespace
from dataclasses import replace
import json

import pytest

from tests.support.result_append import completed_source
from easyicu.webserver.research_append_source import seal_completed_source
from easyicu.webserver.research_followup_source import load_completed_result
from easyicu.webserver.research_followups import result_followups, suggest_followups


@pytest.fixture
def sealed(tmp_path):
    wrapper, run, _, _, baseline = completed_source(tmp_path)
    return seal_completed_source(wrapper=wrapper, run_dir=run, study={"id": "study"},
                                 baseline=baseline.model_dump(mode="json"), prepared_package_binding={"synthetic": True})


def _loaded(sealed):
    return load_completed_result(sealed.wrapper, run_id="source", study_id="study")


def test_real_result_has_host_owned_followups(tmp_path, monkeypatch):
    from easyicu.webserver import study_contexts
    from easyicu.webserver.research_append_actions import append_options

    wrapper, run, _, _, baseline = completed_source(tmp_path)
    source = seal_completed_source(
        wrapper=wrapper, run_dir=run, study={"id": "study"},
        baseline=baseline.model_dump(mode="json"), prepared_package_binding={"synthetic": True},
    )
    service = SimpleNamespace(_research_run_row=lambda *_: {"project_dir": str(wrapper), "study_id": "study"})
    monkeypatch.setattr(study_contexts, "get_context", lambda _: {"id": "study"})
    result = append_options(service, project_id="project", run_id="source")
    followups = result["followups"]
    assert followups["source_run_sha256"] == source.digest
    assert 0 < len(followups["suggestions"]) <= 4
    assert "multi_database_unavailable" in followups["unavailable_actions"]
    assert all(row["delta"] and row["reason_codes"] for row in followups["suggestions"])


def test_each_append_delta_is_the_exact_t5_proposal(sealed):
    from easyicu.research_agent.planning.family_spec.baseline_association_template import compile_result_append

    result = suggest_followups(_loaded(sealed))
    assert [row.kind for row in result.suggestions] == ["dose_response", "functional_form_check", "article_extension"]
    for row in result.suggestions:
        assert row.target_action == "append"
        assert row.delta.plan == compile_result_append(kind=row.delta.kind, source=sealed.plan,
                                                       context=sealed.context, authority=sealed.baseline).plan


def test_ids_bind_the_result_digest_and_duplicate_candidates_collapse(sealed, monkeypatch):
    from easyicu.webserver import research_followups as owner

    source = _loaded(sealed)
    first = suggest_followups(source)
    changed = suggest_followups(replace(source, source_sha256="b" * 64))
    assert {row.suggestion_id for row in first.suggestions}.isdisjoint(row.suggestion_id for row in changed.suggestions)
    rows = list(owner._append_suggestions(source))
    monkeypatch.setattr(owner, "_append_suggestions", lambda _: [*rows, *rows])
    assert suggest_followups(source) == first


def test_a_sensitivity_is_called_prespecified_only_when_the_design_names_it(sealed):
    from easyicu.research_agent.planning.sensitivity_authority import PrespecifiedSensitivitySpec

    source = _loaded(sealed)
    context = source.context.model_copy(deep=True)
    context.user_preferences.sensitivity_specs = [PrespecifiedSensitivitySpec(
        spec_id="curvature", axis="functional_form", strategy="restricted_cubic_spline", execution_variables=("adjuster",),
    )]
    result = suggest_followups(replace(source, context=context))
    row = result.suggestions[0]
    assert row.kind == "prespecified_sensitivity"
    assert row.reason_codes[0] == "prespecified_not_run"
    assert row.delta.plan.steps[0].sensitivity_spec_ids == ["curvature"]


def test_a_named_subgroup_uses_only_the_subgroup_t5_actually_compiles(sealed):
    source = _loaded(sealed)
    context = source.context.model_copy(deep=True)
    context.user_preferences.data_constraints = json.dumps({"stated_subgroups": [{
        "variable": "stratum", "concept": "stratum", "words": "the stated strata", "source_status": "available",
    }]})
    named = replace(source, context=context)
    row = next(row for row in suggest_followups(named).suggestions if row.kind == "named_subgroup")
    assert [term.name for term in row.delta.plan.steps[0].model_requirements[0].effect_modifiers] == ["stratum"]
    assert not any(row.kind == "named_subgroup" for row in suggest_followups(source).suggestions)
    completed = replace(named, completed_steps=(*named.completed_steps, *row.delta.plan.steps))
    assert not any(row.kind == "named_subgroup" for row in suggest_followups(completed).suggestions)


def test_completed_sensitivities_and_append_kinds_are_not_suggested_again(sealed):
    source = _loaded(sealed)
    offered = suggest_followups(source)
    sensitivity = next(row for row in offered.suggestions if row.kind == "functional_form_check")
    completed = replace(source, completed_steps=(*source.completed_steps, *sensitivity.delta.plan.steps))
    assert not any(row.delta.kind == "sensitivity" for row in suggest_followups(completed).suggestions)
    completed = replace(source, completed_append_kinds=frozenset({"sensitivity", "exposure", "subgroup", "article"}))
    assert suggest_followups(completed).suggestions == []


def test_a_missing_source_product_cannot_be_promised_to_the_append_owner(sealed):
    source = replace(_loaded(sealed), products=frozenset())
    assert suggest_followups(source).suggestions == []


def test_renaming_a_completed_functional_form_does_not_make_it_new(sealed):
    source = _loaded(sealed)
    row = next(row for row in suggest_followups(source).suggestions if row.kind == "functional_form_check")
    step = row.delta.plan.steps[0].model_copy(update={"step_id": "earlier_check", "sensitivity_spec_ids": ["earlier_name"]})
    source = replace(source, completed_steps=(*source.completed_steps, step))
    assert not any(row.kind == "functional_form_check" for row in suggest_followups(source).suggestions)


def test_a_completed_spec_id_remains_completed_without_optional_form_metadata(sealed):
    source = _loaded(sealed)
    row = next(row for row in suggest_followups(source).suggestions if row.kind == "functional_form_check")
    step = row.delta.plan.steps[0].model_copy(update={"functional_form_spec": None})
    source = replace(source, completed_steps=(*source.completed_steps, step))
    assert not any(row.kind == "functional_form_check" for row in suggest_followups(source).suggestions)


def test_nominal_groups_are_not_described_as_ordered_dose_response(sealed):
    from easyicu.research_agent.contracts.exposure_group_rules import EXPOSURE_GROUP_LABELS_KEY

    source = _loaded(sealed)
    context = source.context.model_copy(deep=True)
    context.cohort.provenance[EXPOSURE_GROUP_LABELS_KEY][0]["parameters"]["scale"] = "nominal"
    assert not any(row.kind == "dose_response" for row in suggest_followups(replace(source, context=context)).suggestions)


def test_generic_completed_result_admission_works_without_a_t5_seal(tmp_path):
    wrapper, _, plan, context, _ = completed_source(tmp_path)
    result = load_completed_result(wrapper, run_id="source", study_id="study")
    assert result.plan == plan and result.context == context
    assert "table:adjusted_association_estimates" in result.products
    assert result.baseline is None
    assert result == load_completed_result(wrapper, run_id="source", study_id="study")


@pytest.mark.parametrize("damage", ["incomplete", "review", "primary", "checkpoint", "context", "missing_primary"])
def test_generic_result_admission_requires_verified_completion(tmp_path, damage):
    wrapper, run, _, _, _ = completed_source(tmp_path)
    if damage == "incomplete":
        path = run / "run_status.json"
        payload = json.loads(path.read_text())
        payload["gates"]["analysis_validated"] = False
        path.write_text(json.dumps(payload))
    elif damage == "review":
        path = wrapper / "scientific_plan_review.json"
        payload = json.loads(path.read_text())
        payload["plan_sha256"] = "b" * 64
        path.write_text(json.dumps(payload))
    elif damage == "primary":
        from easyicu.research_agent.authority.evidence_store import EvidenceStore

        record = next(r for r in EvidenceStore(run).records() if r.produced_by_step == "adjusted_association")
        path = run / record.relative_path
        path.write_bytes(path.read_bytes() + b"\n")
    elif damage == "missing_primary":
        path = run / "manifest.json"
        payload = json.loads(path.read_text())
        payload["per_step_records"] = [r for r in payload["per_step_records"] if r["step_id"] != "adjusted_association"]
        path.write_text(json.dumps(payload))
    else:
        from easyicu.research_agent.authority.evidence_store import EvidenceStore

        path = run / ("human_review_checkpoint.json" if damage == "checkpoint" else EvidenceStore(run).get("research_context").relative_path)
        path.write_text("{}")
    service = SimpleNamespace(_research_run_row=lambda *_: {"project_dir": str(wrapper), "study_id": "study"})
    result = result_followups(service, project_id="project", run_id="source")
    assert result["suggestions"] == [] and result["source_run_sha256"] is None
    assert "followup_source_unavailable" in result["unavailable_actions"]


def test_bad_append_seal_never_falls_back_to_generic_admission(sealed):
    (sealed.wrapper / "result_append_source.json").write_text("{}")
    service = SimpleNamespace(_research_run_row=lambda *_: {"project_dir": str(sealed.wrapper), "study_id": "study"})
    assert result_followups(service, project_id="project", run_id="source")["source_run_sha256"] is None


def test_project_authorization_precedes_any_source_read(monkeypatch):
    def refused(*_):
        raise PermissionError("different project")
    monkeypatch.setattr("easyicu.webserver.research_followups.load_completed_result", lambda *_args, **_kwargs: pytest.fail("source read"))
    with pytest.raises(PermissionError):
        result_followups(SimpleNamespace(_research_run_row=refused), project_id="other", run_id="source")


def test_projection_privacy_failure_removes_every_action(sealed, monkeypatch):
    from easyicu.webserver import run_artifact_disclosure

    monkeypatch.setattr(run_artifact_disclosure, "scan_browser_projection", lambda _: {"passed": False})
    service = SimpleNamespace(_research_run_row=lambda *_: {"project_dir": str(sealed.wrapper), "study_id": "study"})
    result = result_followups(service, project_id="project", run_id="source")
    assert result["suggestions"] == []
    assert "followup_projection_unavailable" in result["unavailable_actions"]


def test_completed_append_lineage_suppresses_the_analysis_just_run(sealed):
    from easyicu.webserver.research_append_run import approve_append_run, propose_append_run

    child = propose_append_run(source=sealed, kind="sensitivity", wrapper=sealed.wrapper.parent / "run_child",
                               study={"id": "study"}, registered_source_changed=False)
    approve_append_run(child.wrapper, decision="approved", reviewer="synthetic")
    source = load_completed_result(child.wrapper, run_id=child.checkpoint.run_id, study_id="study")
    assert source.completed_append_kinds == frozenset({"sensitivity"})
    assert not any(row.kind in {"prespecified_sensitivity", "functional_form_check"} for row in suggest_followups(source).suggestions)


def test_http_adapter_delegates_to_the_project_scoped_owner(monkeypatch):
    from easyicu.webserver.routes import research_append as routes

    service = object()
    monkeypatch.setattr(routes, "get_pi_copilot_service", lambda: service)
    calls = []
    monkeypatch.setattr(routes, "result_followups", lambda actual, **kw: calls.append((actual, kw)) or {"suggestions": []})
    assert routes.get_result_followups("project", "source") == {"suggestions": []}
    assert calls == [(service, {"project_id": "project", "run_id": "source"})]
