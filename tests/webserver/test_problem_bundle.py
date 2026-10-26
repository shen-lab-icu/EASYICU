import io
import json
import zipfile

import pytest

from easyicu.webserver.problem_bundle import build_problem_bundle, scrub_diagnostic_logs


def test_logs_drop_secrets_paths_rows_and_prompts_even_in_metadata_fields():
    logs = [{"level": "ERROR", "code": "provider_request_failed", "state": "failed",
             "message": "SYNTHETIC_ROW", "api_key": "sk-synthetic-private-key",
             "path": "/home/synthetic/private.csv", "rows": [{"value": 17}],
             "prompt": "SYNTHETIC_PROMPT", "traceback": "C:\\private\\report.csv"},
            {"code": "sk-synthetic-private-key", "level": "/home/synthetic",
             "state": "SYNTHETIC_ROW"},
            "password=private-value /private/report.csv SYNTHETIC_PROMPT"]
    clean = scrub_diagnostic_logs(logs)
    text = json.dumps(clean)
    for forbidden in ("SYNTHETIC", "sk-synthetic", "private", "rows", "prompt", "path", "api_key"):
        assert forbidden not in text
    assert clean[0] == {"level": "ERROR", "code": "provider_request_failed", "state": "failed"}
    assert clean[1]["code"] == "diagnostic_detail_withheld"


def test_bundle_is_downloadable_metadata_only_and_does_not_copy_input(tmp_path):
    from tests.support.trust_help import call, ledger
    ledger(tmp_path / ".runtime/provider_hard_stop_ledger.json",
           [call(payload="SYNTHETIC_PRIVATE_VALUE")])
    (tmp_path / "patient.csv").write_text("SYNTHETIC_DO_NOT_OPEN")
    result = build_problem_bundle(tmp_path)
    with zipfile.ZipFile(io.BytesIO(result)) as bundle:
        assert sorted(bundle.namelist()) == ["egress.json", "events.json", "manifest.json"]
        manifest = json.loads(bundle.read("manifest.json"))
        assert manifest["schema_version"] == "easyicu.problem-bundle/1"
        assert manifest["upload_performed"] is False
        assert manifest["versions"]["python"]
        assert manifest["run_state"] == "failed"
        assert manifest["run_state_basis"] == "provider_task_ledger"
        assert manifest["reason_codes"] == ["provider_request_failed"]
        for name in bundle.namelist():
            raw = bundle.read(name).decode()
            assert "SYNTHETIC" not in raw and str(tmp_path) not in raw


def test_bundle_uses_run_owner_state_when_execution_fails_after_provider_success(tmp_path):
    from tests.support.trust_help import call, ledger
    ledger(tmp_path / ".runtime/provider_hard_stop_ledger.json", [call(state="completed")], state="completed")
    (tmp_path / "source_run_manifest.json").write_text(json.dumps({
        "status": "failed", "failure_code": "research_agent_pipeline_failed_closed",
        "diagnostic": "SYNTHETIC_PRIVATE", "provider": {"api_key": "private-value"},
    }))
    with zipfile.ZipFile(io.BytesIO(build_problem_bundle(tmp_path))) as bundle:
        manifest = json.loads(bundle.read("manifest.json"))
        assert manifest["run_state"] == "failed"
        assert manifest["run_state_basis"] == "run_manifest"
        assert "research_agent_pipeline_failed_closed" in manifest["reason_codes"]
        assert "SYNTHETIC_PRIVATE" not in bundle.read("events.json").decode()


@pytest.mark.parametrize("damage", ["json", "symlink"])
def test_invalid_run_metadata_does_not_report_provider_success_as_run_success(tmp_path, damage):
    from tests.support.trust_help import call, ledger
    ledger(tmp_path / ".runtime/provider_hard_stop_ledger.json", [call()], state="completed")
    path = tmp_path / "source_run_manifest.json"
    if damage == "json":
        path.write_text("{incomplete")
    else:
        other = tmp_path / "other.json"
        other.write_text('{"status":"completed"}')
        path.symlink_to(other)
    with zipfile.ZipFile(io.BytesIO(build_problem_bundle(tmp_path))) as bundle:
        manifest = json.loads(bundle.read("manifest.json"))
        assert manifest["run_state"] == "unknown"
        assert "run_metadata_invalid" in manifest["reason_codes"]


def test_log_projection_is_bounded():
    assert len(scrub_diagnostic_logs({"code": "provider_request_failed"} for _ in range(1001))) == 1000
