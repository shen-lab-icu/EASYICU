"""Only constructed aggregates; no recorded clinical data or ASGI transport."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import struct
import zlib

import pytest


def encoded(value):
    return json.dumps(value, sort_keys=True).encode()


def write_json(root, name, value):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded(value))


def png_chunk(kind, data):
    return (
        struct.pack("!I", len(data))
        + kind
        + data
        + struct.pack("!I", zlib.crc32(kind + data))
    )


def register(root, evidence_id, kind, name, raw, **extra):
    path = root / "evidence" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    index = root / "evidence/evidence_index.json"
    records = json.loads(index.read_bytes()) if index.exists() else []
    record = {
        "evidence_id": evidence_id,
        "kind": kind,
        "relative_path": f"evidence/{name}",
        "sha256": hashlib.sha256(raw).hexdigest(),
        "produced_by_step": "analysis",
        "script_evidence_id": "analysis_code",
        "producer": "standard_executor",
        "metadata": {},
        **extra,
    }
    records.append(record)
    write_json(root, "evidence/evidence_index.json", records)
    return record


@pytest.fixture
def recorded(tmp_path):
    root = tmp_path / "recorded"
    source = {"database": "synthetic", "label": "Synthetic fixture"}
    write_json(
        root,
        "run_context.json",
        {
            "run_id": "fixture-run",
            "study_id": "fixture-study",
            "question": "How do aggregate outcomes differ between groups?",
            "source": source,
        },
    )
    write_json(
        root,
        "source_run_manifest.json",
        {
            "run_id": "fixture-run",
            "source": source,
            "status": "analysis_only",
            "trace_timings": [
                {"step_id": "analysis", "status": "done", "elapsed_seconds": 2.5}
            ],
            "provider": {"api_key": "fixture-secret-never-publish"},
        },
    )
    write_json(
        root,
        "agent_plan.json",
        {
            "title": "Synthetic group comparison",
            "study_type": "association",
            "steps": [
                {"step_id": "analysis", "description": "Compare aggregate outcomes"}
            ],
        },
    )
    write_json(
        root,
        "manuscript_draft.json",
        {
            "run_id": "fixture-run",
            "markdown_preview": "The aggregate estimate is 1.2.",
            "claims": [],
            "status": "locked_pending_human_review",
        },
    )
    write_json(
        root,
        "scientific_readiness.json",
        {
            "schema_version": "easyicu.web-scientific-readiness/1",
            "run_id": "fixture-run",
            "status": "analysis_only",
            "claim_ceiling": "analysis_only",
            "publication_ready": False,
            "paper_authorized": False,
            "human_review_required": True,
            "domains": [],
            "findings": [],
        },
    )
    table = register(
        root,
        "aggregate",
        "table",
        "aggregate.csv",
        b"group,n,events,estimate\nA,40,20,1.2\nB,60,25,1.1\n",
    )
    write_json(
        root,
        "result_tables.json",
        {
            "tables": [
                {
                    "evidence_id": table["evidence_id"],
                    "sha256": table["sha256"],
                    "label": "Aggregate table",
                    # This preview is deliberately incomplete: the exporter must inspect the file.
                    "headers": ["group", "n"],
                    "rows": [["A", "40"]],
                    "preview_truncated": True,
                }
            ]
        },
    )
    write_json(root, "figure_gallery.json", {"figures": []})
    return root


def add_figure(root):
    from easyicu.research_agent.gates.figure_privacy import (
        FIGURE_PRIVACY_AUDIT_VERSION,
        FIGURE_PRIVACY_RECEIPT_SCHEMA,
    )

    table = json.loads((root / "evidence/evidence_index.json").read_bytes())[0]
    sources = {"aggregate": table["sha256"]}
    register(
        root,
        "contract",
        "log",
        "contract.json",
        encoded(
            {
                "figure_id": "comparison",
                "core_claim": "Synthetic groups",
                "panels": [
                    {"title": "Aggregate comparison", "claim": "Synthetic fixture"}
                ],
            }
        ),
        producer="publication_figure_skill",
    )
    register(
        root,
        "privacy",
        "log",
        "privacy.json",
        encoded(
            {
                "schema": FIGURE_PRIVACY_RECEIPT_SCHEMA,
                "audit_version": FIGURE_PRIVACY_AUDIT_VERSION,
                "figure_id": "comparison",
                "aggregate_only": True,
                "source_sha256": sources,
                "mark_count_verified": False,
            }
        ),
        producer="publication_figure_skill",
    )
    png = b"\x89PNG\r\n\x1a\n" + png_chunk(
        b"IHDR", struct.pack("!2I5B", 1, 1, 8, 2, 0, 0, 0)
    )
    png += png_chunk(b"tEXt", b"Comment\x00/home/private/path api_key=do-not-share")
    png += png_chunk(b"IDAT", zlib.compress(b"\x00\xff\xff\xff")) + png_chunk(
        b"IEND", b""
    )
    figure = register(
        root,
        "figure",
        "figure",
        "figure.png",
        png,
        producer="publication_figure_skill",
        metadata={
            "figure_id": "comparison",
            "aggregate_only": True,
            "aggregate_only_basis": "host_privacy_audit",
            "aggregate_only_audit_version": FIGURE_PRIVACY_AUDIT_VERSION,
            "figure_privacy_audit_evidence_id": "privacy",
            "contract_evidence_id": "contract",
            "source_evidence_ids": ["aggregate"],
            "source_evidence_sha256": sources,
        },
    )
    write_json(
        root,
        "figure_gallery.json",
        {
            "figures": [
                {
                    "relative_path": figure["relative_path"],
                    "label": "Aggregate figure",
                    "caption": "Synthetic groups",
                }
            ]
        },
    )
    return figure


def export(root, **kwargs):
    from easyicu.webserver.study_replay import build_study_replay

    return build_study_replay(root, **kwargs)


def test_replay_contains_existing_contracts_and_is_byte_stable(recorded):
    add_figure(recorded)
    conversation = {
        "schema_version": "easyicu.pi-conversation-replay/1",
        "turns": [
            {
                "kind": "host_action",
                "action_code": "approve_plan",
                "status": "done",
                "child_job_id": "job-a",
            }
        ],
        "child_jobs": [{"job_id": "job-a", "result": {"run_id": "fixture-run"}}],
    }
    bundle = export(recorded, conversation=conversation)
    assert bundle == export(recorded, conversation=copy.deepcopy(conversation))
    replay = json.loads(bundle["replay.json"])
    assert replay["question"].startswith("How do")
    assert replay["reviewed_plan"]["steps"][0]["step_id"] == "analysis"
    assert replay["trace_steps"][0]["elapsed_seconds"] == 2.5
    assert replay["decisions"][0]["action_code"] == "approve_plan"
    assert (
        replay["result_answer"]["markdown_preview"] == "The aggregate estimate is 1.2."
    )
    assert replay["scientific_review"]["claim_ceiling"] == "analysis_only"
    assert len(replay["aggregate_tables"][0]["rows"]) == 2
    assert "figures/figure.png" in bundle
    assert b"/home/private" not in b"".join(bundle.values())
    assert b"fixture-secret" not in b"".join(bundle.values())
    assert b"do-not-share" not in b"".join(bundle.values())
    manifest = json.loads(bundle["manifest.json"])
    assert manifest["source"]["database"] == "synthetic"
    for row in manifest["files"]:
        assert hashlib.sha256(bundle[row["path"]]).hexdigest() == row["sha256"]
        assert len(bundle[row["path"]]) == row["bytes"]
    assert set(bundle) == {
        "index.html",
        "replay.json",
        "manifest.json",
        "figures/figure.png",
    }


def replace_table(root, raw, **changes):
    records = json.loads((root / "evidence/evidence_index.json").read_bytes())
    record = records[0]
    record.update(changes)
    (root / record["relative_path"]).write_bytes(raw)
    record["sha256"] = hashlib.sha256(raw).hexdigest()
    write_json(root, "evidence/evidence_index.json", records)
    write_json(
        root,
        "result_tables.json",
        {
            "tables": [
                {"evidence_id": record["evidence_id"], "sha256": record["sha256"]}
            ]
        },
    )


@pytest.mark.parametrize(
    "raw,code",
    [
        (b"patient_id,n\nsynthetic-person,40\n", "replay_patient_level_forbidden"),
        (b"patientNumber,n\n77,40\n", "replay_patient_level_forbidden"),
        (b"label,n\npatient_123456,40\n", "replay_patient_level_forbidden"),
        (b"label,n\n2025-01-01T12:30,40\n", "replay_patient_level_forbidden"),
        (b"age,value\n55,2.0\n", "replay_aggregate_unproven"),
        (b"group,n\nA,19\n", "replay_small_cell_forbidden"),
        (b"group,at_risk,n\nA,3.0,40\n", "replay_small_cell_forbidden"),
        (b"group,n\nA,40\nB,1\n", "replay_small_cell_forbidden"),
        (b"group,n\nA,unknown\n", "replay_aggregate_unproven"),
        (b"group,n\nA,40,extra\n", "replay_table_invalid"),
    ],
)
def test_replay_refuses_unsafe_tables(recorded, raw, code):
    from easyicu.webserver.study_replay import ReplayExportError

    replace_table(recorded, raw)
    with pytest.raises(ReplayExportError) as error:
        export(recorded)
    assert error.value.code == code
    assert str(error.value) == code  # Diagnostics cannot echo rejected values.


@pytest.mark.parametrize(
    "text,code",
    [
        ("subject_id: synthetic-identifier", "replay_patient_level_forbidden"),
        ("There were 3 patients.", "replay_small_cell_forbidden"),
        ("n=19", "replay_small_cell_forbidden"),
    ],
)
def test_replay_scans_prose(recorded, text, code):
    from easyicu.webserver.study_replay import ReplayExportError

    write_json(recorded, "manuscript_draft.json", {"markdown_preview": text})
    with pytest.raises(ReplayExportError) as error:
        export(recorded)
    assert error.value.code == code


def test_replay_scrubs_nested_paths_and_credentials_and_escapes_html(recorded):
    plan = json.loads((recorded / "agent_plan.json").read_bytes())
    plan.update(
        {
            "description": 'Read /srv/private/run and C:\\private\\run. api_key="secret value" '
            "https://u:pw@example.invalid/a?token=abc <script>alert(1)</script>",
            "provider": {"api_key": "opaque-key"},
            "export_path": "/mnt/clinical",
        }
    )
    write_json(recorded, "agent_plan.json", plan)
    bundle = export(recorded)
    raw = b"".join(bundle.values())
    projected_plan = json.loads(bundle["replay.json"])["reviewed_plan"]
    assert "provider" not in projected_plan
    assert "export_path" not in projected_plan
    for token in (
        b"/srv/private",
        b"C:\\",
        b"secret value",
        b"u:pw",
        b"token=abc",
        b"opaque-key",
        b"/mnt/clinical",
    ):
        assert token not in raw
    assert b"<script>" not in bundle["index.html"]
    assert b"&lt;script&gt;" in bundle["index.html"]


def test_credentialed_source_uses_the_same_aggregate_boundary(recorded):
    for name in ("run_context.json", "source_run_manifest.json"):
        payload = json.loads((recorded / name).read_bytes())
        payload["source"] = {"database": "miiv", "label": "Credentialed source"}
        write_json(recorded, name, payload)
    register(recorded, "raw-private", "table", "raw.parquet", b"NEVER OPEN THIS")
    bundle = export(recorded)
    assert json.loads(bundle["manifest.json"])["source"]["kind"] == "credentialed"
    assert b"NEVER OPEN" not in b"".join(bundle.values())
    replace_table(recorded, b"group,n\nA,3\n")
    from easyicu.webserver.study_replay import ReplayExportError

    with pytest.raises(ReplayExportError, match="replay_small_cell_forbidden"):
        export(recorded)


def test_replay_rejects_digest_changes_and_unsafe_classes(recorded):
    from easyicu.webserver.study_replay import ReplayExportError

    (recorded / "evidence/aggregate.csv").write_bytes(b"group,n\nA,100\n")
    with pytest.raises(ReplayExportError, match="replay_digest_mismatch"):
        export(recorded)
    replace_table(recorded, b"group,n\nA,40\n", kind="code")
    with pytest.raises(ReplayExportError, match="replay_evidence_class_forbidden"):
        export(recorded)


@pytest.mark.parametrize(
    "relative", ["../outside.csv", "/etc/passwd", "evidence/link.csv"]
)
def test_replay_rejects_escaping_and_symlinked_evidence(recorded, relative):
    from easyicu.webserver.study_replay import ReplayExportError

    outside = recorded.parent / "outside.csv"
    outside.write_text("group,n\nA,40\n")
    (recorded / "evidence/link.csv").symlink_to(outside)
    records = json.loads((recorded / "evidence/evidence_index.json").read_bytes())
    records[0]["relative_path"] = relative
    write_json(recorded, "evidence/evidence_index.json", records)
    with pytest.raises(ReplayExportError, match="replay_path_forbidden"):
        export(recorded)


def test_replay_rechecks_figure_sources_and_refuses_declared_only_clearance(recorded):
    from easyicu.webserver.study_replay import ReplayExportError

    add_figure(recorded)
    records = json.loads((recorded / "evidence/evidence_index.json").read_bytes())
    records[-1]["metadata"]["aggregate_only_basis"] = "producer_claim"
    write_json(recorded, "evidence/evidence_index.json", records)
    with pytest.raises(ReplayExportError, match="replay_figure_uncleared"):
        export(recorded)
    records[-1]["metadata"]["aggregate_only_basis"] = "host_privacy_audit"
    write_json(recorded, "evidence/evidence_index.json", records)
    replace_table(recorded, b"group,n\nA,80\n")
    with pytest.raises(ReplayExportError, match="replay_figure_uncleared"):
        export(recorded)


def test_export_failure_leaves_no_partial_destination(recorded):
    from easyicu.webserver.study_replay import ReplayExportError, export_study_replay

    replace_table(recorded, b"group,n\nA,2\n")
    target = recorded.parent / "share"
    with pytest.raises(ReplayExportError):
        export_study_replay(recorded, target)
    assert not target.exists()


def test_replay_keeps_optional_review_absent(recorded):
    (recorded / "scientific_readiness.json").unlink()
    assert json.loads(export(recorded)["replay.json"])["scientific_review"] is None


@pytest.mark.parametrize(
    "payload",
    [
        {"rows": [[55, 80], [70, 90]]},
        {"detail": {"value": 12345678}},
        {"detail": {"patient_number": "synthetic-person"}},
        {"detail": {"charttime": "2025-01-01T12:30"}},
    ],
)
def test_plan_cannot_smuggle_rows_or_identifiers(recorded, payload):
    from easyicu.webserver.study_replay import ReplayExportError

    write_json(recorded, "agent_plan.json", payload)
    with pytest.raises(ReplayExportError, match="replay_patient_level_forbidden"):
        export(recorded)


@pytest.mark.parametrize(
    "text", ["There were three patients.", "Events: 3.0", "共有3例。"]
)
def test_text_small_counts_cannot_bypass_the_policy(recorded, text):
    from easyicu.webserver.study_replay import ReplayExportError

    write_json(recorded, "manuscript_draft.json", {"markdown_preview": text})
    with pytest.raises(ReplayExportError, match="replay_small_cell_forbidden"):
        export(recorded)


@pytest.mark.parametrize(
    "name",
    ["scientific_readiness.json", "source_run_manifest.json", "manuscript_draft.json"],
)
def test_source_identity_must_match_all_records(recorded, name):
    from easyicu.webserver.study_replay import ReplayExportError

    value = json.loads((recorded / name).read_bytes())
    value["run_id"] = "foreign-run"
    write_json(recorded, name, value)
    with pytest.raises(ReplayExportError, match="replay_source_identity_mismatch"):
        export(recorded)


def test_foreign_conversation_turns_are_never_exported(recorded):
    conversation = {
        "schema_version": "easyicu.pi-conversation-replay/1",
        "turns": [
            {
                "kind": "host_action",
                "action_code": "foreign_action",
                "status": "done",
                "child_job_id": "foreign-job",
                "assistant_reply": "patient_id: do-not-export",
            }
        ],
        "child_jobs": [{"id": "foreign-job", "result": {"run_id": "foreign-run"}}],
    }
    assert (
        json.loads(export(recorded, conversation=conversation)["replay.json"])[
            "decisions"
        ]
        == []
    )


def test_missing_job_coordinates_do_not_bind_a_decision(recorded):
    conversation = {
        "schema_version": "easyicu.pi-conversation-replay/1",
        "turns": [
            {
                "kind": "host_action",
                "action_code": "unbound_action",
                "status": "done",
            }
        ],
        "child_jobs": [{"result": {"run_id": "fixture-run"}}],
    }
    assert (
        json.loads(export(recorded, conversation=conversation)["replay.json"])[
            "decisions"
        ]
        == []
    )


def test_trace_uses_recorded_timestamps_without_inventing_missing_durations(recorded):
    manifest = json.loads((recorded / "source_run_manifest.json").read_bytes())
    manifest["trace_timings"] = [
        {
            "step_id": "analysis",
            "timestamp": "2026-01-01T00:00:00Z",
            "status": "running",
        },
        {"step_id": "analysis", "timestamp": "2026-01-01T00:00:04Z", "status": "done"},
        {"step_id": "report", "status": "done"},
    ]
    write_json(recorded, "source_run_manifest.json", manifest)
    trace = json.loads(export(recorded)["replay.json"])["trace_steps"]
    assert [row["elapsed_seconds"] for row in trace] == [None, 4, None]


@pytest.mark.parametrize("count", ["0", "20", "100000"])
def test_safe_counts_are_retained_exactly(recorded, count):
    replace_table(recorded, f"group,n\nA,{count}\n".encode())
    assert json.loads(export(recorded)["replay.json"])["aggregate_tables"][0][
        "rows"
    ] == [["A", count]]


def test_bundle_writes_once_and_preserves_existing_destination(recorded):
    from easyicu.webserver.study_replay import ReplayExportError, export_study_replay

    target = recorded.parent / "share"
    export_study_replay(recorded, target)
    snapshot = {
        p.relative_to(target).as_posix(): p.read_bytes()
        for p in target.rglob("*")
        if p.is_file()
    }
    assert snapshot == export(recorded)
    with pytest.raises(ReplayExportError, match="replay_destination_exists"):
        export_study_replay(recorded, target)
    assert (target / "replay.json").read_bytes() == snapshot["replay.json"]
    with pytest.raises(ReplayExportError, match="replay_destination_forbidden"):
        export_study_replay(recorded, recorded / "share")


def test_unselected_patient_file_is_never_opened(recorded, monkeypatch):
    import easyicu.webserver.study_replay_io as file_owner

    record = register(
        recorded, "private", "table", "private.csv", b"forbidden synthetic marker"
    )
    real_open = file_owner.os.open

    def guarded(path, *args, **kwargs):
        assert str(path) != Path(record["relative_path"]).name
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(file_owner.os, "open", guarded)
    export(recorded)


def test_replay_rejects_oversize_and_duplicate_registry(recorded, monkeypatch):
    import easyicu.webserver.study_replay_io as file_owner
    from easyicu.webserver.study_replay import ReplayExportError

    records = json.loads((recorded / "evidence/evidence_index.json").read_bytes())
    write_json(recorded, "evidence/evidence_index.json", records * 2)
    with pytest.raises(ReplayExportError, match="replay_evidence_ambiguous"):
        export(recorded)
    write_json(recorded, "evidence/evidence_index.json", records)
    monkeypatch.setattr(file_owner, "MAX_FILE_BYTES", 5)
    with pytest.raises(ReplayExportError, match="replay_size_limit"):
        export(recorded)


def test_pipeline_registry_symlink_is_refused(recorded):
    from easyicu.webserver.study_replay import ReplayExportError

    target = recorded.parent / "moved-evidence"
    (recorded / "evidence").rename(target)
    pipeline = recorded / "pipeline" / "fixture-run"
    pipeline.mkdir(parents=True)
    (pipeline / "evidence").symlink_to(target, target_is_directory=True)
    with pytest.raises(ReplayExportError, match="replay_path_forbidden"):
        export(recorded)


def test_figure_refuses_corrupt_audit_and_pixels(recorded):
    from easyicu.webserver.study_replay import ReplayExportError

    add_figure(recorded)
    audit_path = recorded / "evidence/privacy.json"
    original = audit_path.read_bytes()
    audit_path.write_bytes(original + b" ")
    with pytest.raises(ReplayExportError, match="replay_digest_mismatch"):
        export(recorded)
    audit_path.write_bytes(original)
    records = json.loads((recorded / "evidence/evidence_index.json").read_bytes())
    records[-1]["sha256"] = hashlib.sha256(b"invalid PNG").hexdigest()
    (recorded / "evidence/figure.png").write_bytes(b"invalid PNG")
    write_json(recorded, "evidence/evidence_index.json", records)
    with pytest.raises(ReplayExportError, match="replay_figure_invalid"):
        export(recorded)


@pytest.mark.parametrize(
    "claim,code",
    [
        ("api_key=fixture-secret", "replay_figure_text_unsafe"),
        ("Read /home/private/figure", "replay_figure_text_unsafe"),
        ("n=3", "replay_small_cell_forbidden"),
    ],
)
def test_figure_checks_the_text_rendered_into_pixels(recorded, claim, code):
    from easyicu.webserver.study_replay import ReplayExportError

    add_figure(recorded)
    path = recorded / "evidence/contract.json"
    contract = json.loads(path.read_bytes())
    contract["core_claim"] = claim
    path.write_bytes(encoded(contract))
    records = json.loads((recorded / "evidence/evidence_index.json").read_bytes())
    next(row for row in records if row["evidence_id"] == "contract")["sha256"] = (
        hashlib.sha256(path.read_bytes()).hexdigest()
    )
    write_json(recorded, "evidence/evidence_index.json", records)
    with pytest.raises(ReplayExportError, match=code):
        export(recorded)


def test_figure_refuses_source_text_that_needed_scrubbing(recorded):
    from easyicu.webserver.study_replay import ReplayExportError

    replace_table(recorded, b"group,n\napi_key=fixture-secret,40\n")
    add_figure(recorded)
    with pytest.raises(ReplayExportError, match="replay_figure_text_unsafe"):
        export(recorded)


def test_provider_secret_echoed_in_scientific_text_is_scrubbed(recorded):
    write_json(
        recorded,
        "manuscript_draft.json",
        {
            "markdown_preview": "Recorded note: fixture-secret-never-publish",
        },
    )
    assert b"fixture-secret-never-publish" not in b"".join(export(recorded).values())


def test_review_authorization_and_scope_are_preserved(recorded):
    review = {
        "status": "changes_required",
        "review_scope": "pre_execution_plan",
        "rendered_outputs_assessed": False,
        "findings": [
            {
                "code": "fixture_review_required",
                "requires_user_authorization": True,
                "authorization_question": "Confirm the recorded design.",
            }
        ],
    }
    write_json(recorded, "scientific_plan_review.json", review)
    assert json.loads(export(recorded)["replay.json"])["plan_review"] == review


@pytest.mark.parametrize(
    "field,value",
    [
        ("aggregate_only", False),
        ("aggregate_only_audit_version", "obsolete"),
        ("figure_id", "different-figure"),
        ("source_evidence_ids", ["aggregate", "missing"]),
    ],
)
def test_every_figure_clearance_binding_is_checked(recorded, field, value):
    from easyicu.webserver.study_replay import ReplayExportError

    add_figure(recorded)
    records = json.loads((recorded / "evidence/evidence_index.json").read_bytes())
    records[-1]["metadata"][field] = value
    write_json(recorded, "evidence/evidence_index.json", records)
    with pytest.raises(ReplayExportError, match="replay_figure_uncleared"):
        export(recorded)


def test_export_reads_the_concrete_pipeline_and_is_location_independent(recorded):
    import shutil

    before = export(recorded)
    pipeline = recorded / "pipeline/fixture-run"
    pipeline.mkdir(parents=True)
    (recorded / "evidence").rename(pipeline / "evidence")
    wrapped = export(recorded)
    assert json.loads(wrapped["replay.json"]) == json.loads(before["replay.json"])
    relocated = recorded.parent / "relocated"
    shutil.copytree(recorded, relocated)
    assert wrapped == export(relocated)


def test_invalid_record_has_a_stable_non_echoing_code(recorded):
    from easyicu.webserver.study_replay import ReplayExportError

    write_json(recorded, "source_run_manifest.json", ["private-malformed-value"])
    with pytest.raises(ReplayExportError, match="^replay_record_invalid$"):
        export(recorded)


def test_host_run_coordinates_and_digests_are_not_patient_identifiers(recorded):
    run_id = "run_20261026_123456"
    for name in (
        "run_context.json",
        "source_run_manifest.json",
        "manuscript_draft.json",
        "scientific_readiness.json",
    ):
        record = json.loads((recorded / name).read_bytes())
        record["run_id"] = run_id
        write_json(recorded, name, record)
    plan = json.loads((recorded / "agent_plan.json").read_bytes())
    plan["context_sha256"] = "1234567890abcdef" * 4
    write_json(recorded, "agent_plan.json", plan)
    replay = json.loads(export(recorded)["replay.json"])
    assert replay["scientific_review"]["run_id"] == run_id
    assert replay["reviewed_plan"]["context_sha256"] == plan["context_sha256"]


def test_selected_table_requires_its_displayed_digest(recorded):
    from easyicu.webserver.study_replay import ReplayExportError

    write_json(
        recorded,
        "result_tables.json",
        {"tables": [{"evidence_id": "aggregate", "sha256": "0" * 64}]},
    )
    with pytest.raises(ReplayExportError, match="replay_digest_mismatch"):
        export(recorded)


def test_columnar_file_is_refused_before_reading(recorded):
    from easyicu.webserver.study_replay import ReplayExportError

    replace_table(
        recorded, b"group,n\nA,40\n", relative_path="evidence/aggregate.parquet"
    )
    with pytest.raises(ReplayExportError, match="replay_evidence_class_forbidden"):
        export(recorded)


def test_extreme_count_exponents_are_refused_before_integer_conversion(recorded):
    from easyicu.webserver.study_replay import ReplayExportError

    replace_table(recorded, b"group,n\nA,1e1000\n")
    with pytest.raises(ReplayExportError, match="replay_aggregate_unproven"):
        export(recorded)


def test_write_failures_expose_only_a_stable_code(recorded, monkeypatch):
    from easyicu.webserver.study_replay import ReplayExportError, export_study_replay

    target = recorded.parent / "unwritable" / "share"
    original = Path.mkdir

    def denied(path, *args, **kwargs):
        if path == target.parent:
            raise PermissionError("/home/private/synthetic-path")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", denied)
    with pytest.raises(ReplayExportError, match="^replay_write_failed$"):
        export_study_replay(recorded, target)
    assert not target.exists()
