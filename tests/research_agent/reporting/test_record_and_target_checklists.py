"""Synthetic coverage for RECORD/TARGET selection and report emission."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from easyicu.research_agent.reporting import reporting_checklist as checklist


def _records(*names: str):
    return [SimpleNamespace(evidence_id=name) for name in names]


def _items(report):
    return {item.item_id: item for item in report.items}


def test_record_template_and_artifact_evidence():
    empty = checklist.build_record_checklist(evidence_records=[], bound_manuscript="")
    assert (empty.name, empty.version, len(empty.items)) == ("RECORD", "2015", 13)
    assert set(_items(empty)) == {
        "1.1",
        "1.2",
        "1.3",
        "6.1",
        "6.2",
        "6.3",
        "7.1",
        "12.1",
        "12.2",
        "12.3",
        "13.1",
        "14.1",
        "19.1",
    }
    assert all(item.status == "open" for item in empty.items)
    assert "author" in _items(empty)["12.1"].rationale.lower()

    report = checklist.build_record_checklist(
        evidence_records=_records("research_context", "cohort_attrition"),
        bound_manuscript="Routinely collected electronic health records were analysed.",
    )
    items = _items(report)
    assert items["13.1"].status == "addressed"
    assert "cohort_attrition" in items["13.1"].evidence_ids
    # A generic context and flow table do not establish code lists or linkage.
    assert items["6.1"].status == "open"
    assert items["12.3"].status == "open"
    assert report.summary()["coverage"] > empty.summary()["coverage"]


def test_target_template_and_artifact_evidence():
    empty = checklist.build_target_checklist(evidence_records=[], bound_manuscript="")
    assert (empty.name, empty.version, len(empty.items)) == ("TARGET", "2025", 22)
    assert set(_items(empty)) == {f"T{number}" for number in range(1, 23)}
    assert all(item.status == "open" for item in empty.items)
    assert "author" in _items(empty)["T4"].rationale.lower()

    report = checklist.build_target_checklist(
        evidence_records=_records("target_trial_protocol", "cohort_attrition"),
        bound_manuscript="We aligned eligibility and treatment at time zero.",
    )
    items = _items(report)
    assert items["T5"].status == "addressed"
    assert items["T19"].status == "addressed"
    assert items["T4"].status == "open"
    assert items["T20"].status == "open"
    assert report.summary()["coverage"] > empty.summary()["coverage"]
    placeholder = checklist.build_target_checklist(
        evidence_records=_records("target_trial_protocol"),
        bound_manuscript="## Time zero\nTime zero requires author verification.",
    )
    assert _items(placeholder)["T5"].status == "partial"
    assert "author" in _items(placeholder)["T5"].rationale.lower()
    result = checklist.build_target_checklist(
        evidence_records=_records("table_target_trial_effect_estimates_abcdef"),
        bound_manuscript="",
    )
    assert _items(result)["T20"].status == "addressed"


def test_generic_artifact_prefix_cannot_prove_a_specific_result_or_protocol():
    report = checklist.build_target_checklist(
        evidence_records=_records("target_trial", "cohort"),
        bound_manuscript="The analysis starts at time zero.",
    )
    items = _items(report)
    assert items["T5"].status == "partial"
    assert items["T19"].status == "open"
    assert items["T20"].status == "open"


@pytest.mark.parametrize(
    ("builder", "item_id", "artifact"),
    [
        ("build_record_checklist", "6.1", "cohort_code_list"),
        ("build_record_checklist", "7.1", "variable_code_list"),
        ("build_record_checklist", "12.2", "data_cleaning_log"),
        ("build_record_checklist", "14.1", "linkage_counts"),
        ("build_target_checklist", "T17", "target_trial_weight_models"),
        ("build_target_checklist", "T19", "target_trial_eligibility_flow"),
        ("build_target_checklist", "T20", "target_trial_effect_estimates"),
    ],
)
def test_specific_artifacts_credit_their_matching_items(builder, item_id, artifact):
    report = getattr(checklist, builder)(
        evidence_records=_records(artifact), bound_manuscript=""
    )
    item = _items(report)[item_id]
    assert item.status == "addressed"
    assert item.evidence_ids


@pytest.mark.parametrize(
    ("family", "database", "expected"),
    [
        ("causal_inference", "miiv", ("strobe", "target", "record")),
        ("association_study", "eicu", ("strobe", "record")),
        ("prediction_model", "miiv", ("strobe", "tripod_ai", "record")),
        ("trajectory_clustering", "eicu", ("internal_phenotype", "record")),
        ("descriptive_epidemiology", None, ("strobe",)),
        ("causal_inference", None, ("strobe", "target")),
        ("target_trial_emulation", "eicu", ("strobe", "target", "record")),
    ],
)
def test_checklist_routing_uses_family_and_database(family, database, expected):
    assert checklist.choose_checklist(family, database=database) == expected


def test_target_routing_does_not_scan_unstructured_prose():
    assert "target" not in checklist.choose_checklist(
        "a target trial might be useful", database="miiv"
    )
    assert checklist.checklist_names_for_kind("causal_inference") == (
        "strobe",
        "target",
    )
    assert checklist.choose_checklist("consider a prediction model") == ("strobe",)
    assert checklist.choose_checklist("describe the trajectory after care") == (
        "strobe",
    )
    assert checklist.choose_checklist(
        "causal_inference", database="miiv", requested_names=("target", "record")
    ) == ("target", "record")


def test_new_builders_are_available_from_the_public_api():
    import easyicu.research_agent as ra

    assert ra.build_record_checklist is checklist.build_record_checklist
    assert ra.build_target_checklist is checklist.build_target_checklist


@pytest.mark.parametrize(
    ("plan_family", "preference", "database", "override", "expected"),
    [
        (
            "causal_inference",
            "association_study",
            "miiv",
            None,
            {"strobe", "target", "record"},
        ),
        (
            "causal_inference",
            "association_study",
            "miiv",
            ["strobe"],
            {"strobe", "target", "record"},
        ),
        (
            "association_study",
            "causal_inference",
            "eicu",
            ["strobe"],
            {"strobe", "record"},
        ),
        ("prediction_model", None, "miiv", None, {"strobe", "tripod_ai", "record"}),
        ("descriptive_epidemiology", None, "", None, {"strobe"}),
        (None, "causal_inference", "miiv", None, {"strobe", "target", "record"}),
        (
            "signed_trial",
            "association_study",
            "miiv",
            ["strobe"],
            {"strobe", "target", "record"},
        ),
        ("kind_only", None, "miiv", None, {"strobe", "target", "record"}),
    ],
)
def test_write_phase_publishes_selected_checklists(
    tmp_path, monkeypatch, plan_family, preference, database, override, expected
):
    from easyicu.research_agent.reporting import write_phase
    from easyicu.research_agent.review import methodological_rigor

    monkeypatch.setattr(
        methodological_rigor.MethodologicalRigorAuditor, "audit", lambda *a, **k: []
    )
    monkeypatch.setattr(
        methodological_rigor,
        "extract_method_signals",
        lambda *a, **k: SimpleNamespace(family="synthetic"),
    )

    class Evidence:
        def __init__(self):
            self.registered = {}

        def current_verified_records(self, _):
            return _records("cohort_attrition", "target_trial_protocol")

        def get(self, key):
            return self.registered.get(key)

        def register_json(self, **kwargs):
            self.registered[kwargs["evidence_id"]] = kwargs

        def register_file(self, **kwargs):
            self.registered[kwargs["evidence_id"]] = kwargs

    evidence = Evidence()
    bound_path = tmp_path / "bound.md"
    bound_path.write_text("Eligibility and treatment aligned at time zero.")
    pipeline = SimpleNamespace(
        _enable_latex=False,
        _enable_causal_audit=False,
        _enable_reporting_checklist=True,
        _enable_reviewer_round=False,
        _reporting_checklist_names=override,
        _benchmark_task_kind="causal_inference" if plan_family == "kind_only" else None,
    )
    context = SimpleNamespace(
        cohort=SimpleNamespace(database=database),
        user_preferences=SimpleNamespace(inferred_analysis_family=preference),
    )
    findings = []
    write_phase._publish_and_audit_manuscript(
        pipeline,
        bound=bound_path.read_text(),
        bound_path=bound_path,
        context=context,
        plan=SimpleNamespace(
            analysis_type=None
            if plan_family in {"signed_trial", "kind_only"}
            else plan_family,
            steps=[
                SimpleNamespace(
                    planned_analysis_role="primary",
                    method="signed_target_trial_suite",
                )
            ]
            if plan_family == "signed_trial"
            else [],
        ),
        current_verified_evidence_records=[],
        evidence=evidence,
        findings=findings,
        literature=None,
        manuscript_authors=None,
        manuscript_title=None,
        per_step_records=[],
        repro_envelope=None,
        run_dir=tmp_path,
        run_id="synthetic",
        emit_progress=lambda *a, **k: None,
    )
    actual = {
        path.name.removeprefix("reporting_checklist_").removesuffix(".json")
        for path in tmp_path.glob("reporting_checklist_*.json")
    }
    assert actual == expected
    for name in expected:
        assert (tmp_path / f"reporting_checklist_{name}.md").exists()
        assert f"reporting_checklist_{name}" in evidence.registered
        assert f"reporting_checklist_{name}_json" in evidence.registered
        assert json.loads((tmp_path / f"reporting_checklist_{name}.json").read_text())[
            "items"
        ]
    if "record" in expected:
        saved = json.loads((tmp_path / "reporting_checklist_record.json").read_text())
        flow = next(item for item in saved["items"] if item["item_id"] == "13.1")
        assert flow["status"] == "addressed"


def test_signed_primary_trial_method_routes_target_when_family_is_missing():
    plan = SimpleNamespace(
        steps=[
            SimpleNamespace(
                planned_analysis_role="primary",
                method="signed_target_trial_suite",
            )
        ]
    )
    auxiliary = SimpleNamespace(
        steps=[
            SimpleNamespace(
                planned_analysis_role="auxiliary",
                method="signed_target_trial_suite",
            )
        ]
    )
    capability = SimpleNamespace(
        steps=[
            SimpleNamespace(
                planned_analysis_role="primary",
                scientific_capability="causal_target_trial_ccw_v1",
            )
        ]
    )
    assert checklist.plan_has_target_trial(plan)
    assert checklist.plan_has_target_trial(capability)
    assert not checklist.plan_has_target_trial(auxiliary)
    assert checklist.choose_checklist(
        None, target_trial=checklist.plan_has_target_trial(plan)
    ) == ("strobe", "target")
