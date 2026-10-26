"""Synthetic recorded results; the review reads facts without running analysis."""

from copy import deepcopy
import hashlib
import json

import pytest


def records(family="association", *, append=False):
    fields = {
        "association": ("estimate", "1.25"),
        "survival": ("hazard_ratio", "1.18"),
        "prediction": ("auroc", "0.81"),
        "descriptive": ("median", "12.4"),
    }
    field, value = fields[family]
    source = {"database": "synthetic", "version": "v-test", "export_id": "export-test"}
    return {
        "run_context.json": {"run_id": "run_child" if append else "run_test", "source": source},
        "agent_plan.json": {
            "analysis_type": family, "cohort": {"selection_mode": "all_input_rows"},
            "endpoint": {"name": "outcome", "time_origin": "icu_admit"},
            "steps": [{"step_id": "primary", "method": family, "planned_analysis_role": "primary",
                       "model_requirements": [{"exposure_source": "marker", "outcome": "outcome"}]}],
        },
        "quality_gate.json": {"gate": {"status": "analysis_only", "reportable": False,
                                       "checks": [{"id": "execution_complete", "passed": True},
                                                  {"id": "paper_authorized", "passed": False}]}},
        "scientific_readiness.json": {"claim_ceiling": "association_only", "findings": [
            {"code": "RECORDED_DIAGNOSTIC", "severity": "warning", "message": "Recorded diagnostic."}]},
        "cohort_summary.json": {"summary": {"n": 120, "n_excluded": 15}},
        "source_run_manifest.json": {"run_id": "run_child" if append else "run_test",
                                     "status": "analysis_only", **({"append": {"source_run_id": "run_parent", "kind": "subgroup"}} if append else {})},
        "result_tables.json": {"tables": [{
            "evidence_id": "result", "produced_by_step": "primary", "sha256": "a" * 64,
            "source_status": "current", "name": "result.csv", "headers": [field, "n"],
            "rows": [[value, "120"]],
        }]},
        "manuscript_draft.json": {"markdown_preview": "Recorded draft."},
        "manuscript_provenance.json": {
            "manuscript_sha256": hashlib.sha256(b"Recorded draft.").hexdigest(),
            "claims": [{"claim_id": "claim_1", "display_value": value, "source_value": value,
                        "source_field": field, "source_json_pointer": "/" + field,
                        "step_id": "primary", "occurrence_count": 2, "status": "current",
                        "evidence": {"evidence_id": "result", "sha256": "a" * 64}}],
            "number_binding": {"status": "recorded", "untraced": []},
        },
    }


@pytest.mark.parametrize("family", ["association", "survival", "prediction", "descriptive"])
def test_card_preserves_recorded_values_and_sources(family):
    from easyicu.webserver.scientific_review import project_scientific_review

    payloads = records(family)
    before = deepcopy(payloads)
    card = project_scientific_review(payloads)
    assert card["schema_version"] == "easyicu.run-scientific-review/1"
    assert card["analysis_only"] is True
    assert card["publication_authorized"] is False
    assert card["number_check"]["counts"] == {"bound": 4, "unbound": 0, "suppressed": 0}
    numbers = card["number_check"]["items"]
    assert numbers[0]["value"] == payloads["result_tables.json"]["tables"][0]["rows"][0][0]
    assert numbers[0]["source"]["evidence_id"] == "result"
    assert numbers[0]["source"]["step_id"] == "primary"
    assert numbers[0]["source"]["json_pointer"] == "/tables/0/rows/0/0"
    assert any(item["code"] == "SR_APPROACH_MODEL" and item["source"]["step_id"] == "primary"
               for item in card["approach"]["items"])
    assert any(item["code"] == "SR_CLAIM_CEILING" and item["value"] == "association_only"
               for item in card["limitations"]["items"])
    assert payloads == before


def test_unbound_numbers_are_defects_even_when_the_first_page_is_full():
    from easyicu.webserver.scientific_review import project_scientific_review

    payloads = records()
    payloads["manuscript_provenance.json"]["number_binding"]["untraced"] = ["987.65"]
    card = project_scientific_review(payloads, limit=1)
    check = card["number_check"]
    assert check["counts"]["unbound"] == 1
    assert check["status"] == "defect"
    assert check["defect_codes"] == ["SR_NUMBER_UNBOUND"]
    assert check["items"][0]["code"] == "SR_NUMBER_UNBOUND"
    assert check["items"][0]["value"] == "987.65"
    assert check["next_offset"] == 1


def test_substitutions_keep_the_recorded_decision_and_source():
    from easyicu.webserver.scientific_review import project_scientific_review

    payloads = records()
    substitution = {"capability": "threshold:age", "original": {"concept": "age", "operator": ">=", "value": 65},
                    "substitute": {"concept": "age", "operator": ">=", "value": 60},
                    "reason_code": "numeric_threshold_aligned_to_band"}
    payloads["scientific_plan_review.json"] = {"facts": {"database_capability_review": {
        "decision": {"status": "substitution", "database": "synthetic", "substitutions": [substitution]},
        "unapplied_substitutions": [],
    }}}
    card = project_scientific_review(payloads)
    row = next(item for item in card["materials"]["items"] if item["code"] == "SR_SOURCE_SUBSTITUTION")
    assert row["value"]["substitute"] == substitution["substitute"]
    assert row["value"]["applied"] is True
    assert row["source"]["artifact"] == "scientific_plan_review.json"
    assert any(item["code"] == "SR_SOURCE_SUBSTITUTION" for item in card["limitations"]["items"])


def test_append_card_uses_child_records_and_keeps_parent_coordinate():
    from easyicu.webserver.scientific_review import project_scientific_review

    card = project_scientific_review(records(append=True))
    assert card["run_id"] == "run_child"
    assert card["append_source_run_id"] == "run_parent"
    assert card["analysis_only"] is True


def test_missing_binding_is_unknown_and_small_cells_use_the_existing_owner():
    from easyicu.webserver.scientific_review import project_scientific_review

    payloads = records()
    payloads.pop("manuscript_provenance.json")
    payloads["result_tables.json"]["tables"][0]["rows"][0][1] = "3"
    card = project_scientific_review(payloads)
    assert card["number_check"]["coverage"]["manuscript"] == "unavailable"
    assert card["number_check"]["counts"]["suppressed"] == 1
    assert any(row["value"] == "suppressed" for row in card["number_check"]["items"])


def test_read_api_uses_existing_run_record_and_is_read_only(tmp_path):
    from easyicu.webserver.scientific_review import read_scientific_review

    payloads = records()
    payloads["evidence_ledger.json"] = {"run_id": "run_test", "run_type": "full"}
    for name, payload in payloads.items():
        (tmp_path / name).write_text(json.dumps(payload))
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    result = read_scientific_review(str(tmp_path))
    assert result["ok"] is True
    assert result["review"]["run_id"] == "run_test"
    assert all(row["source"]["artifact_sha256"] for row in result["review"]["approach"]["items"])
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before


def test_an_injected_number_without_current_binding_is_a_visible_defect():
    from easyicu.webserver.scientific_review import project_scientific_review

    payloads = records()
    payloads["manuscript_draft.json"]["markdown_preview"] += " Unverified 987.65."
    card = project_scientific_review(payloads)
    assert card["number_check"]["counts"]["unbound"] == 1
    assert card["number_check"]["items"][0]["code"] == "SR_NUMBER_UNBOUND"
    assert card["number_check"]["items"][0]["value"] == "987.65"
    assert card["number_check"]["coverage"]["manuscript"] == "unavailable"


def test_equal_values_cannot_rebind_a_missing_evidence_coordinate():
    from easyicu.webserver.scientific_review import project_scientific_review

    payloads = records()
    payloads["result_tables.json"]["tables"][0]["sha256"] = ""
    card = project_scientific_review(payloads)
    assert card["number_check"]["counts"]["unbound"] == 2
    assert card["number_check"]["counts"]["bound"] == 2


def test_typed_answer_blocks_keep_the_values_used_by_prediction_and_descriptive_readers():
    from easyicu.webserver.scientific_review import project_scientific_review

    payloads = records("prediction")
    payloads["result_tables.json"] = {"prediction_performance": {
        "schema_version": "easyicu.web-prediction-performance/1", "evidence_id": "performance",
        "step_id": "prediction", "auroc": 0.81, "auroc_ci_low": 0.74, "auroc_ci_high": 0.88,
        "validation_n": 120, "calibration_status": "unavailable",
    }}
    card = project_scientific_review(payloads)
    answer = [row for row in card["number_check"]["items"] if row["surface"] == "answer"]
    assert [row["value"] for row in answer] == [0.81, 0.74, 0.88, 120]
    assert all(row["source"]["evidence_id"] == "performance" for row in answer)
    payloads = records("descriptive")
    table = payloads["result_tables.json"]["tables"][0]
    table["descriptive_summary"] = {"continuous": [{"variable": "marker", "median": "12.4", "q25": "11.2", "q75": "15.1"}]}
    answer = [row for row in project_scientific_review(payloads)["number_check"]["items"] if row["surface"] == "answer"]
    assert [row["value"] for row in answer] == ["12.4", "11.2", "15.1"]


def test_pages_are_bounded_and_counts_cover_every_page():
    from easyicu.webserver.scientific_review import project_scientific_review

    payloads = records()
    payloads["manuscript_provenance.json"]["number_binding"]["untraced"] = [str(1000 + i) for i in range(125)]
    card = project_scientific_review(payloads, limit=50)
    assert len(card["number_check"]["items"]) == 50
    assert card["number_check"]["counts"]["unbound"] == 125
    last = project_scientific_review(payloads, offset=100)
    assert last["number_check"]["next_offset"] is None
    assert last["number_check"]["total"] == 128
    assert len(json.dumps(card).encode()) < 1024 * 1024
    for offset, limit in ((-1, 50), (False, 50), (0, 51), (0, 0), (0, "50")):
        with pytest.raises(ValueError, match="scientific_review_page_invalid"):
            project_scientific_review(payloads, offset=offset, limit=limit)


def test_unapplied_substitution_is_not_presented_as_applied():
    from easyicu.webserver.scientific_review import project_scientific_review

    payloads = records()
    payloads["scientific_plan_review.json"] = {"facts": {"database_capability_review": {
        "decision": {"status": "substitution", "substitutions": [{
            "capability": "endpoint", "original": "a", "substitute": "b", "reason_code": "recorded_substitution"}]},
        "unapplied_substitutions": ["endpoint"],
    }}}
    materials = project_scientific_review(payloads)["materials"]["items"]
    assert next(row for row in materials if row["code"] == "SR_SOURCE_SUBSTITUTION")["value"]["applied"] is False


def test_read_only_route_accepts_run_ref_and_keeps_errors_path_free(tmp_path, monkeypatch):
    from easyicu.webserver.routes import agent

    payloads = records()
    payloads["evidence_ledger.json"] = {"run_id": "run_test", "run_type": "full"}
    for name, payload in payloads.items():
        (tmp_path / name).write_text(json.dumps(payload))
    monkeypatch.setattr(agent.run_refs, "resolve", lambda ref: str(tmp_path) if ref == "owned" else None)
    result = agent.post_agent_run_scientific_review({"run_ref": "owned", "limit": 1})
    assert result["ok"] is True
    assert result["review"]["run_id"] == "run_test"
    assert str(tmp_path) not in json.dumps(result)
    with pytest.raises(agent.HTTPException) as exc:
        agent.post_agent_run_scientific_review({"run_ref": "owned", "limit": 51})
    assert exc.value.status_code == 400
    with pytest.raises(agent.HTTPException) as exc:
        agent.post_agent_run_scientific_review({"run_ref": "other"})
    assert exc.value.status_code == 404


@pytest.mark.parametrize("large", [False, True])
def test_card_does_not_copy_host_paths_or_large_unbounded_values(large):
    from easyicu.webserver.scientific_review import project_scientific_review

    payloads = records()
    payloads["scientific_readiness.json"]["findings"][0]["detail"] = {
        "source_path": "/home/other/private", "note": "/tmp/private-data", "large": ["x" * 1000] * 1000 if large else []}
    card = project_scientific_review(payloads)
    encoded = json.dumps(card)
    assert "/home/other" not in encoded and "/tmp/private" not in encoded
    assert len(encoded.encode()) < 1024 * 1024
    assert next(row for row in card["limitations"]["items"] if row["code"] == "SR_FINDING")["value_truncated"] is large


def test_append_answer_copies_without_value_coordinates_are_explicit_defects():
    from easyicu.webserver.scientific_review import project_scientific_review

    payloads = records(append=True)
    payloads["result_tables.json"]["append"] = {"strata": [{"estimate": 1.25, "ci_low": 1.1}]}
    check = project_scientific_review(payloads)["number_check"]
    assert check["counts"]["unbound"] == 2
    assert check["items"][0]["reason_code"] == "answer_source_coordinate_missing"


def test_typed_prediction_supersedes_its_preview_and_manuscript_keeps_field_pointer():
    from easyicu.webserver.scientific_review import project_scientific_review

    payloads = records("prediction")
    payloads["result_tables.json"]["prediction_performance"] = {
        "schema_version": "easyicu.web-prediction-performance/1", "evidence_id": "result", "step_id": "primary", "auroc": 0.81}
    check = project_scientific_review(payloads)["number_check"]
    answer = [row for row in check["items"] if row["surface"] == "answer"]
    assert len(answer) == 1
    assert answer[0]["value"] == 0.81
    manuscript = next(row for row in check["items"] if row["surface"] == "manuscript")
    assert manuscript["source"]["evidence_json_pointer"] == "/auroc"


def test_a_bounded_draft_preview_keeps_its_full_manuscript_binding():
    from easyicu.webserver.scientific_review import project_scientific_review

    payloads = records()
    preview = payloads["manuscript_draft.json"]["markdown_preview"]
    full_sha = hashlib.sha256((preview + " Remaining synthetic text.").encode()).hexdigest()
    payloads["manuscript_provenance.json"]["manuscript_sha256"] = full_sha
    payloads["source_run_manifest.json"]["scientific_review"] = {"manuscript_preview": {
        "sha256": hashlib.sha256(preview.encode()).hexdigest(), "manuscript_sha256": full_sha}}
    card = project_scientific_review(payloads)
    assert card["number_check"]["counts"]["bound"] == 4
    assert card["number_check"]["coverage"]["manuscript"] == "recorded"
    payloads["manuscript_draft.json"]["markdown_preview"] += " Unverified 987.65."
    assert project_scientific_review(payloads)["number_check"]["counts"]["unbound"] == 1


def test_substitution_source_pointer_survives_filtered_invalid_rows():
    from easyicu.webserver.scientific_review import project_scientific_review

    payloads = records()
    payloads["scientific_plan_review.json"] = {"facts": {"database_capability_review": {
        "decision": {"status": "substitution", "substitutions": [{}, {
            "capability": "endpoint", "original": "a", "substitute": "b", "reason_code": "recorded_substitution"}]},
        "unapplied_substitutions": [],
    }}}
    materials = project_scientific_review(payloads)["materials"]["items"]
    assert next(row for row in materials if row["code"] == "SR_SOURCE_SUBSTITUTION")["source"]["json_pointer"].endswith("/1")
