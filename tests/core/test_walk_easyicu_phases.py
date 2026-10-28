"""Catalogue, dictionary, literature and interaction walks use public shapes."""

from __future__ import annotations

from dataclasses import replace
import json

import pytest

from tools.walk.contracts import PHASES
from tools.walk.http import HttpAPI
from tools.walk.runner import WalkRunner
from tests.support.walk_http import PhaseAPI, walk_options


def test_all_phases_project_aggregates_and_exercise_owned_interactions(tmp_path):
    api = PhaseAPI()
    options = replace(walk_options(tmp_path, approve_test_runs=True), only=PHASES)
    report = WalkRunner(options, api=api).run()
    by_item = {row["item_id"]: row for row in report["items"]}
    mapping = by_item["dictionary-availability"]["metrics"]
    assert (mapping["mapped"], mapping["missing"], mapping["changed_meaning"]) == (1, 1, 1)
    assert mapping["families"] == [
        {"id": "basic", "mapped": 0, "missing": 0, "changed_meaning": 1},
        {"id": "labs", "mapped": 1, "missing": 1, "changed_meaning": 0},
    ]
    assert by_item["extraction-smoke"]["metrics"]["module_count"] == 19
    assert by_item["extraction-smoke"]["status"] == "skipped-with-reason"
    assert by_item["build-identity"]["metrics"]["version"] == "0.1.0"
    assert by_item["build-identity"]["metrics"]["commit"] == "123abcdef"
    assert by_item["synthetic-abstract"]["metrics"]["feasibility_verdicts"] == {"T1_metadata_only": 1}
    assert by_item["synthetic-abstract"]["metrics"]["evidence_verdicts"] == {"not_checked": 1}
    literature = next(row for row in report["items"] if row["phase"] == "literature")["metrics"]
    assert (literature["retrieved"], literature["screened"], literature["included"]) == (4, 3, 2)
    assert literature["role_coverage"] == {"direct_comparator": 1}
    assert by_item["finished-run-followup"]["status"] == "passed"
    assert by_item["composer-skill-attach"]["status"] == "passed"
    assert by_item["material-mention"]["status"] == "passed"
    assert by_item["rename-conversation"]["status"] == "passed"
    assert by_item["delete-conversation"]["status"] == "passed"
    assert len(api.deleted) == 1 and api.drafts == 2
    for path in options.out.rglob("*"):
        if path.is_file():
            assert "NEVER-SAVE-THIS" not in path.read_text()
    assert not any(path == "/api/settings" and method != "GET" for method, path, _, _ in api.calls)


def test_dictionary_pagination_is_complete_and_cycles_fail(tmp_path):
    class PagedAPI(PhaseAPI):
        cycle = False

        def request(self, method, path, body=None, **kwargs):
            payload = super().request(method, path, body, **kwargs)
            if path == "/api/data-map":
                offset = kwargs["params"].get("concept_offset", 0)
                concepts = payload["databases"][0]["concepts"]
                payload["databases"][0]["concepts"] = concepts[offset:offset + 1]
                payload["concept_next_offset"] = (offset if self.cycle else offset + 1) if offset < 2 else None
            return payload

    api = PagedAPI()
    options = replace(walk_options(tmp_path), only=("mapping",))
    report = WalkRunner(options, api=api).run()
    assert report["items"][1]["metrics"]["concept_count"] == 3
    api.cycle = True
    report = WalkRunner(replace(options, out=tmp_path / "cycle"), api=api).run()
    assert report["items"][1]["status"] == "failed"
    assert report["items"][1]["reason_code"] == "dictionary_pagination_invalid"


def test_method_prompt_prefix_is_skipped_instead_of_inventing_a_question(tmp_path):
    class MethodsAPI(PhaseAPI):
        def request(self, method, path, body=None, **kwargs):
            result = super().request(method, path, body, **kwargs)
            if path == "/api/capabilities":
                result["capabilities"]["method_skills"]["components"] = [
                    {"id": "some.method", "prompt": "My research question: "}]
            return result

    api = MethodsAPI()
    report = WalkRunner(walk_options(tmp_path), api=api).run()
    assert report["items"][-1]["reason_code"] == "runnable_example_not_exposed"
    assert len(api.sessions) == 1


def test_review_stop_can_be_resumed_with_approval_without_repeating_the_question(tmp_path):
    api = PhaseAPI()
    options = walk_options(tmp_path)
    WalkRunner(options, api=api).run()
    report = WalkRunner(replace(options, resume=True, approve_test_runs=True), api=api).run()
    assert report["items"][-1]["status"] == "passed"
    assert len(api.sessions) == 1
    assert sum(path.endswith("/message") for _, path, _, _ in api.calls) == 1


def test_increasing_resume_limit_visits_previously_skipped_examples(tmp_path):
    class TwoExamplesAPI(PhaseAPI):
        def request(self, method, path, body=None, **kwargs):
            if path.endswith("/second-study/package"):
                path = path.replace("second-study", "simple-study")
            result = super().request(method, path, body, **kwargs)
            if path == "/api/capabilities/research-workflows":
                result["items"].append({"id": "second-study", "steps": [{"module_id": "summary"}]})
            return result

    api = TwoExamplesAPI()
    options = replace(walk_options(tmp_path), max_workflows=1)
    WalkRunner(options, api=api).run()
    report = WalkRunner(replace(options, resume=True, max_workflows=2), api=api).run()
    assert len(api.sessions) == 2
    assert report["items"][-1]["reason_code"] == "approval_flag_required"


def test_resume_revalidates_source_name_binding_before_reusing_results(tmp_path):
    class ChangedSourceAPI(PhaseAPI):
        changed = False

        def request(self, method, path, body=None, **kwargs):
            result = super().request(method, path, body, **kwargs)
            if path == "/api/workspaces/registry" and self.changed:
                result["sources"][0].update(id="other", label="demo")
            return result

    api = ChangedSourceAPI()
    options = walk_options(tmp_path)
    WalkRunner(options, api=api).run()
    api.changed = True
    calls = len(api.calls)
    report = WalkRunner(replace(options, resume=True), api=api).run()
    assert report["items"][0]["reason_code"] == "resume_source_binding_changed"
    assert not any(method != "GET" for method, _, _, _ in api.calls[calls:])


def test_host_design_card_is_selected_then_replanned_before_approval(tmp_path):
    class DesignAPI(PhaseAPI):
        selected = False
        replanned = False

        def request(self, method, path, body=None, **kwargs):
            if path.endswith("/plan-decision-selection"):
                self.calls.append((method, path, body, kwargs.get("params")))
                assert body["option_id"] == "fixed_landmark_survival"
                assert body["expected_revision"] == 1
                assert body["card_sha256"] == "c" * 64
                self.selected = True
                return {"ok": True, "next_action": "replan"}
            if path.startswith("/api/study-contexts/"):
                return {"context": {"id": path.rsplit("/", 1)[-1], "question": "Describe fictional admission data?",
                                    "data_source": {"database": "sic", "path": "/synthetic/server-only"}}}
            if path == "/api/jobs/agent-run":
                self.calls.append((method, path, body, None))
                assert body["candidate_plan_only"] is True
                assert not self.executed
                self.replanned = True
                return {"job_id": "candidate-job"}
            result = super().request(method, path, body, **kwargs)
            if path.endswith("/workflow") and not self.replanned:
                review = result["workflow"]["plan_review_summary"]
                review["status"] = "changes_required"
                if not self.selected:
                    result["workflow"]["plan_decision"] = {
                        "decision_code": "SURVIVAL_LANDMARK_OWNER_NOT_SEALED", "option_id": "fixed_landmark_survival",
                        "source_run_id": review["run_id"], "card_sha256": "c" * 64,
                    }
                else:
                    result["workflow"]["next_action_code"] = "plan_configuration_superseded"
                    result["host_decisions"] = {"plan_transition": {
                        "family": "plan_transition", "next_action_code": "plan_configuration_superseded",
                        "scientific_configuration_sha256": "a" * 64, "source_run_id": review["run_id"],
                    }}
            return result

    api = DesignAPI()
    options = replace(walk_options(tmp_path, approve_test_runs=True), only=("workflows", "interactions"))
    report = WalkRunner(options, api=api).run()
    row = next(row for row in report["items"] if row["item_id"] == "design-choice-rerun")
    assert row["status"] == "passed", row
    assert api.selected and api.replanned and len(api.executed) == 1
    assert "/synthetic/server-only" not in (options.out / "walk_state.json").read_text()


def test_enabled_catalogue_owner_status_is_audited_in_every_layer(tmp_path):
    class UnownedAPI(PhaseAPI):
        def request(self, method, path, body=None, **kwargs):
            result = super().request(method, path, body, **kwargs)
            if path == "/api/capabilities":
                result["capabilities"]["method_skills"]["items"].append({"id": "unowned-aid", "layer": "planning_aid"})
            return result

    options = replace(walk_options(tmp_path), only=("catalogue",))
    report = WalkRunner(options, api=UnownedAPI()).run()
    assert report["items"][-1]["reason_code"] == "catalogue_owner_binding_not_reported"
    assert report["items"][-1]["metrics"]["unreported_owner_count"] == 1


def test_material_mention_uses_the_selected_source_when_project_has_no_files(tmp_path):
    class NoFilesAPI(PhaseAPI):
        def request(self, method, path, body=None, **kwargs):
            if path.endswith("/mentions"):
                return {"items": []}
            if path.endswith("/draft-mentions"):
                assert body["mentions"][0]["source"] == "registered_source"
                return {"mentions": body["mentions"]}
            return super().request(method, path, body, **kwargs)

    report = WalkRunner(replace(walk_options(tmp_path), only=("interactions",)), api=NoFilesAPI()).run()
    row = next(row for row in report["items"] if row["item_id"] == "material-mention")
    assert row["status"] == "passed"
    assert row["metrics"]["scope"] == "registered_source_mention"


def test_http_json_exchange_runs_a_reduced_walk_without_a_socket(tmp_path):
    from urllib.parse import parse_qs, urlsplit

    fixture = PhaseAPI()

    def exchange(method, url, body, timeout):
        parsed = urlsplit(url)
        assert parsed.netloc == "localhost:8780"
        params = {k: v[-1] for k, v in parse_qs(parsed.query).items()}
        result = fixture.request(method, parsed.path, body, params=params, timeout=timeout)
        return 200, {}, json.dumps(result).encode()

    options = replace(walk_options(tmp_path, approve_test_runs=True), base="http://localhost:8780")
    report = WalkRunner(options, api=HttpAPI(options.base, exchange=exchange)).run()
    assert report["status_counts"] == {"passed": 3}


@pytest.mark.parametrize("code,status", [("blocked", "refused-with-reason"), ("incomplete", "failed")])
def test_a_finished_job_does_not_imply_a_successful_analysis(tmp_path, code, status):
    class FailedGateAPI(PhaseAPI):
        def request(self, method, path, body=None, **kwargs):
            result = super().request(method, path, body, **kwargs)
            if path.endswith("/workflow"):
                result["latest_run"]["gate_status"] = code
                if code == "incomplete":
                    result["latest_run"]["execution_complete"] = False
            return result

    report = WalkRunner(walk_options(tmp_path, approve_test_runs=True), api=FailedGateAPI()).run()
    assert report["items"][-1]["status"] == status
