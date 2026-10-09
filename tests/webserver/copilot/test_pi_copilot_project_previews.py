"""Run artifacts and project previews reach the browser path free and bound.

The run artifact tools emit clickable resources without host paths. Each
project preview (artifact, evidence, document, data package) resolves its
authority within its project and is pinned to the current digest or revision.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from easyicu.webserver.pi_copilot.contracts import (
    AuthorityBinding,
    PiCopilotError,
    PiSessionRecord,
    ToolExecutionContext,
)
from easyicu.webserver.pi_copilot.service import PiCopilotService
from easyicu.webserver.pi_copilot import service as service_module
from easyicu.webserver.pi_copilot import tools as tool_module
from tests.webserver.copilot.pi_copilot_contract_fixtures import FakeGateway


def test_run_artifact_tools_emit_path_free_clickable_resources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = ToolExecutionContext(
        session=PiSessionRecord(
            session_id="pi-artifacts",
            binding=AuthorityBinding(run_id="run_20260808"),
        )
    )
    monkeypatch.setattr(
        tool_module.agent_runs,
        "list_run_history",
        lambda **kwargs: {
            "runs": [
                {
                    "run_id": "run_20260808",
                    "project_dir": "/private/owner-only/run",
                }
            ]
        },
    )
    monkeypatch.setattr(
        tool_module.agent_runs,
        "read_run_review",
        lambda project_dir: {
            "ok": True,
            "artifacts": [
                {
                    "name": "table1_summary.json",
                    "bytes": 412,
                    "sha256": "a" * 64,
                    "kind": "json",
                    "path": "/private/owner-only/run/table1_summary.json",
                },
                {
                    "name": "quality_gate.json",
                    "bytes": 233,
                    "sha256": "b" * 64,
                    "kind": "json",
                },
            ],
            "artifact_payloads": {},
        },
    )

    result = tool_module.execute_tool("easyicu_list_artifacts", {}, context)

    assert result["details"]["artifacts"][0]["size"] == 412
    assert result["details"]["artifacts"][0]["media_type"] == "application/json"
    assert result["details"]["resources"][0] == {
        "kind": "research_artifact",
        "run_id": "run_20260808",
        "artifact": "table1_summary.json",
        "label": "table1_summary.json",
        "media_type": "application/json",
        "sha256": "a" * 64,
    }
    assert "project_dir" not in json.dumps(result)
    assert "/private/" not in json.dumps(result)


def test_project_artifact_preview_resolves_authority_and_scrubs_host_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = PiCopilotService(
        store_path=tmp_path / "sessions.json",
        gateway=FakeGateway(),
    )
    service.project_store.bind("project-a", "study-a")
    service.project_store.bind("project-b", "study-b")

    def history(*, study_id: str, **kwargs: Any) -> dict[str, Any]:
        return {
            "runs": (
                [{"run_id": "run_20260808", "project_dir": "/private/run-a"}]
                if study_id == "study-a"
                else []
            )
        }

    monkeypatch.setattr(service_module.agent_runs, "list_run_history", history)
    monkeypatch.setattr(
        service_module.agent_runs,
        "read_run_artifact",
        lambda project_dir, artifact_name: {
            "ok": True,
            "artifact": {
                "name": artifact_name,
                "path": f"{project_dir}/{artifact_name}",
                "bytes": 120,
                "sha256": "c" * 64,
                "kind": "json",
            },
            "payload": {
                "status": "ready",
                "source": {"path": "/private/export", "database": "mimiciv"},
                "future_paths": {
                    "artifact_path": "/private/future.json",
                    "output_dir": "/private/output",
                    "cache_file": "/private/cache.bin",
                    "cwd": "/private/work",
                },
                "figures": [{"relative_path": "figures/roc.svg"}],
            },
            "privacy_scan": {"passed": True},
        },
    )
    monkeypatch.setattr(
        service_module.agent_runs,
        "read_run_review",
        lambda project_dir: {
            "ok": True,
            "gate": {"status": "analysis_only"},
            "readiness": {
                "status": "awaiting_human_signoff",
                "signed": False,
                "signoff_stale": False,
                "reportable": False,
            },
        },
    )

    payload = service.get_research_artifact(
        project_id="project-a",
        run_id="run_20260808",
        artifact_name="table1_summary.json",
        expected_sha256="c" * 64,
    )
    encoded = json.dumps(payload)
    assert payload["payload"]["source"] == {"database": "mimiciv"}
    assert payload["payload"]["future_paths"] == {}
    assert payload["payload"]["figures"][0]["relative_path"] == "figures/roc.svg"
    assert payload["governance"] == {
        "authority_class": "easyicu_run_artifact",
        "artifact_integrity": "unsigned",
        "gate_status": "analysis_only",
        "readiness_status": "awaiting_human_signoff",
        "human_signoff": "required",
        "reportable": False,
        "claim_ceiling": "analysis_only",
    }
    assert "project_dir" not in encoded
    assert "/private/" not in encoded

    review_export = service.get_research_artifact_download(
        project_id="project-a",
        run_id="run_20260808",
        artifact_name="table1_summary.json",
        expected_sha256="c" * 64,
    )
    exported = json.loads(review_export["content"])
    assert exported["kind"] == "easyicu_review_export"
    assert exported["source_artifact"]["sha256"] == "c" * 64
    assert exported["payload"]["source"] == {"database": "mimiciv"}
    assert "/private/" not in review_export["content"].decode("utf-8")

    with pytest.raises(PiCopilotError) as digest_mismatch:
        service.get_research_artifact(
            project_id="project-a",
            run_id="run_20260808",
            artifact_name="table1_summary.json",
            expected_sha256="d" * 64,
        )
    assert digest_mismatch.value.code == "pi_research_artifact_digest_mismatch"

    with pytest.raises(PiCopilotError) as wrong_project:
        service.get_research_artifact(
            project_id="project-b",
            run_id="run_20260808",
            artifact_name="table1_summary.json",
        )
    assert wrong_project.value.code == "pi_research_run_not_found"

    monkeypatch.setattr(
        service_module.agent_runs,
        "read_run_artifact",
        lambda project_dir, artifact_name: {
            "ok": False,
            "error": "artifact_privacy_scan_failed",
        },
    )
    with pytest.raises(PiCopilotError) as privacy_blocked:
        service.get_research_artifact(
            project_id="project-a",
            run_id="run_20260808",
            artifact_name="table1_summary.json",
        )
    assert privacy_blocked.value.code == "pi_research_artifact_privacy_blocked"


def test_project_evidence_preview_is_project_scoped_digest_pinned_and_governed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = PiCopilotService(
        store_path=tmp_path / "sessions.json",
        gateway=FakeGateway(),
    )
    service.project_store.bind("project-a", "study-a")
    digest = "d" * 64
    wrapper = tmp_path / "wrapper"
    pipeline_run = wrapper / "pipeline" / "run_1"
    pipeline_run.mkdir(parents=True)
    monkeypatch.setattr(
        service_module.agent_runs,
        "list_run_history",
        lambda **kwargs: {
            "runs": [{"run_id": "run_1", "project_dir": str(wrapper)}]
            if kwargs.get("study_id") == "study-a"
            else []
        },
    )
    preview_paths: list[str] = []
    monkeypatch.setattr(
        service_module.agent_runs,
        "read_run_evidence_preview",
        lambda project_dir, evidence_id, expected_sha256: preview_paths.append(
            project_dir
        )
        or {
            "ok": True,
            "payload": {
                "schema_version": "easyicu.web-evidence-preview/1",
                "evidence_id": evidence_id,
                "sha256": expected_sha256,
                "display_name": "analysis.py",
                "renderer": "code",
                "previewable": True,
                "text": "estimate = 1.25\n",
            },
            "privacy_scan": {"passed": True},
        },
    )
    review_paths: list[str] = []
    monkeypatch.setattr(
        service_module.agent_runs,
        "read_run_review",
        lambda project_dir: review_paths.append(project_dir)
        or {
            "ok": True,
            "gate": {"status": "analysis_only"},
            "readiness": {
                "status": "awaiting_human_signoff",
                "signed": False,
                "signoff_stale": False,
                "reportable": False,
            },
        },
    )

    payload = service.get_research_evidence_preview(
        project_id="project-a",
        run_id="run_1",
        evidence_id="code_analysis_1",
        expected_sha256=digest,
    )
    encoded = json.dumps(payload)
    assert payload["payload"]["text"] == "estimate = 1.25\n"
    assert payload["payload"]["sha256"] == digest
    assert payload["governance"]["claim_ceiling"] == "analysis_only"
    assert preview_paths == [str(pipeline_run.resolve())]
    assert review_paths == [str(wrapper)]
    assert "project_dir" not in encoded and "/private/" not in encoded

    with pytest.raises(PiCopilotError) as wrong_project:
        service.get_research_evidence_preview(
            project_id="project-b",
            run_id="run_1",
            evidence_id="code_analysis_1",
            expected_sha256=digest,
        )
    assert wrong_project.value.code == "pi_project_not_initialized"

    monkeypatch.setattr(
        service_module.agent_runs,
        "read_run_evidence_preview",
        lambda *args: {
            "ok": False,
            "error": "evidence_preview_privacy_scan_failed",
        },
    )
    with pytest.raises(PiCopilotError) as blocked:
        service.get_research_evidence_preview(
            project_id="project-a",
            run_id="run_1",
            evidence_id="code_analysis_1",
            expected_sha256=digest,
        )
    assert blocked.value.code == "pi_research_evidence_privacy_blocked"


def test_project_document_preview_requires_the_current_ledger_digest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = PiCopilotService(
        store_path=tmp_path / "sessions.json",
        gateway=FakeGateway(),
    )
    service.project_store.bind("project-a", "study-a")
    document = b"<!doctype html><title>Bound report</title>"
    digest = hashlib.sha256(document).hexdigest()
    monkeypatch.setattr(
        service_module.agent_runs,
        "list_run_history",
        lambda **_kwargs: {
            "runs": [{"run_id": "run_20260808", "project_dir": "/private/run-a"}]
        },
    )
    monkeypatch.setattr(
        service_module.agent_runs,
        "read_run_artifact_bytes",
        lambda _project_dir, name: {
            "ok": True,
            "name": name,
            "content": document,
            "media_type": "text/html; charset=utf-8",
        },
    )
    review = {
        "ok": True,
        "gate": {"status": "blocked"},
        "readiness": {"status": "blocked", "reportable": False},
        "artifacts": [{"name": "system_validation_report.html", "sha256": digest}],
        "artifact_payloads": {
            "evidence_ledger.json": {
                "artifacts": [
                    {"name": "system_validation_report.html", "sha256": digest}
                ]
            }
        },
    }
    monkeypatch.setattr(
        service_module.agent_runs,
        "read_run_review",
        lambda _project_dir: review,
    )

    loaded = service.get_research_document(
        project_id="project-a",
        run_id="run_20260808",
        document_name="system_validation_report.html",
    )
    assert loaded["content"] == document
    assert loaded["claim_ceiling"] == "engineering_validation_only"

    assert service.get_research_document(
        project_id="project-a", run_id="run_20260808",
        document_name="system_validation_report.html", expected_sha256=digest,
    )["content"] == document
    for stale in ("f" * 64, "invalid", ""):
        with pytest.raises(PiCopilotError) as stale_revision:
            service.get_research_document(
                project_id="project-a", run_id="run_20260808",
                document_name="system_validation_report.html", expected_sha256=stale,
            )
        assert stale_revision.value.code == "pi_research_document_digest_mismatch"

    review["artifact_payloads"]["evidence_ledger.json"]["artifacts"][0][
        "sha256"
    ] = "0" * 64
    with pytest.raises(PiCopilotError) as mismatch:
        service.get_research_document(
            project_id="project-a",
            run_id="run_20260808",
            document_name="system_validation_report.html",
        )
    assert mismatch.value.code == "pi_research_document_digest_mismatch"


def test_project_data_package_preview_is_revision_and_digest_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = PiCopilotService(
        store_path=tmp_path / "sessions.json",
        gateway=FakeGateway(),
    )
    service.project_store.bind("project-a", "study-a")
    study = {
        "id": "study-a",
        "revision": 7,
        "data_source": {"path": "/private/export", "database": "miiv"},
    }
    monkeypatch.setattr(service_module.study_contexts, "get_context", lambda _id: study)
    from easyicu.webserver import data_package_review as review_owner

    review_payload = {
        "schema_version": "easyicu.data-package-review/1",
        "study_context_id": "study-a",
        "study_context_revision": 7,
        "status": "ready_for_plan",
        "source": {"database": "miiv"},
        "privacy": {
            "raw_rows_returned": False,
            "host_paths_returned": False,
        },
        "analysis_results_withheld": True,
    }
    review_digest = hashlib.sha256(
        json.dumps(
            review_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    review_payload["review_sha256"] = review_digest
    monkeypatch.setattr(
        review_owner,
        "build_registered_data_package_review",
        lambda _study: dict(review_payload),
    )

    prepared = service.prepare_data_package_review(project_id="project-a")
    assert prepared["resource"] == {
        "kind": "data_package_review",
        "study_context_id": "study-a",
        "study_revision": 7,
        "review_sha256": review_digest,
        "label": "Analysis data preview",
        "media_type": "application/json",
    }
    assert prepared["governance"]["analysis_results_withheld"] is True

    payload = service.get_data_package_review(
        project_id="project-a",
        study_revision=7,
        review_sha256=review_digest,
    )
    assert payload["payload"]["source"] == {"database": "miiv"}
    assert payload["governance"]["claim_ceiling"] == "pre_analysis_review"
    assert "/private/" not in json.dumps(payload)

    with pytest.raises(PiCopilotError) as stale:
        service.get_data_package_review(
            project_id="project-a", study_revision=6, review_sha256=review_digest
        )
    assert stale.value.code == "pi_data_package_review_snapshot_missing"

    with pytest.raises(PiCopilotError) as drift:
        service.get_data_package_review(
            project_id="project-a", study_revision=7, review_sha256="e" * 64
        )
    assert drift.value.code == "pi_data_package_review_digest_mismatch"


@pytest.mark.parametrize("failure_code", [
    None, "plan_bound_data_preview_context_mismatch",
    "plan_bound_data_preview_context_unreadable",
    "plan_bound_data_preview_files_unavailable",
    "pi_research_run_not_found",
])
def test_project_data_package_preview_uses_plan_bound_analysis_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_code: str | None,
) -> None:
    service = PiCopilotService(
        store_path=tmp_path / "sessions.json",
        gateway=FakeGateway(),
    )
    service.project_store.bind("project-a", "study-a")
    study = {"id": "study-a", "revision": 7, "data_source": {"database": "miiv"}}
    monkeypatch.setattr(service_module.study_contexts, "get_context", lambda _id: study)
    monkeypatch.setattr(
        service,
        "_latest_run_id",
        lambda _study_id, *, project_id: "run-plan",
    )
    wrapper = tmp_path / "wrapper"
    monkeypatch.setattr(
        service,
        "_research_run_row",
        lambda _project_id, _run_id: {"project_dir": str(wrapper)},
    )
    from easyicu.webserver import data_package_review as review_owner

    captured: dict[str, Path] = {}
    payload = {
        "schema_version": "easyicu.data-package-review/2",
        "study_context_id": "study-a",
        "study_context_revision": 7,
        "review_stage": "post_plan",
        "status": "ready_for_analysis",
        "source": {"database": "miiv"},
        "privacy": {"raw_rows_returned": False, "host_paths_returned": False},
        "analysis_results_withheld": True,
    }
    digest = hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    payload["review_sha256"] = digest

    def _build(_study: dict, *, cohort_file: Path, plan_file: Path, context_file: Path) -> dict:
        captured["cohort_file"] = cohort_file
        captured["plan_file"] = plan_file
        captured["context_file"] = context_file
        if failure_code == "pi_research_run_not_found":
            raise PiCopilotError(failure_code, "Project binding failure", status_code=404)
        if failure_code:
            raise review_owner.DataPackageReviewError(failure_code, "Exact-source failure")
        return dict(payload)

    def _registered(_study):
        assert failure_code == "plan_bound_data_preview_files_unavailable"
        return dict(payload)

    monkeypatch.setattr(review_owner, "build_plan_bound_data_package_review", _build)
    monkeypatch.setattr(
        review_owner,
        "build_registered_data_package_review",
        _registered,
    )

    if failure_code and failure_code != "plan_bound_data_preview_files_unavailable":
        with pytest.raises(PiCopilotError) as error:
            service.prepare_data_package_review(project_id="project-a")
        assert error.value.code == failure_code
        return

    prepared = service.prepare_data_package_review(project_id="project-a")

    assert prepared["resource"]["review_sha256"] == digest
    assert captured == {
        "cohort_file": wrapper / "pipeline" / "run-plan" / "cohort.parquet",
        "plan_file": wrapper / "pipeline" / "run-plan" / "analysis_plan.json",
        "context_file": wrapper / "pipeline" / "run-plan" / "research_context.json",
    }
