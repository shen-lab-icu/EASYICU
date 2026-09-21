from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from easyicu.webserver.research_evidence_preview import (
    EvidencePreviewError,
    build_evidence_preview,
    project_figure_source_code,
    project_review_evidence_refs,
)


def _register(
    run_dir: Path,
    *,
    evidence_id: str,
    kind: str,
    name: str,
    content: bytes,
    produced_by_step: str = "analysis_step",
    script_evidence_id: str | None = None,
    inputs: list[str] | None = None,
    producer: str = "runner",
    generation_mode: str = "system",
    metadata: dict | None = None,
) -> tuple[Path, str]:
    evidence = run_dir / "evidence"
    evidence.mkdir(parents=True, exist_ok=True)
    path = evidence / name
    path.write_bytes(content)
    digest = hashlib.sha256(content).hexdigest()
    index_path = evidence / "evidence_index.json"
    records = json.loads(index_path.read_text()) if index_path.exists() else []
    records.append(
        {
            "evidence_id": evidence_id,
            "kind": kind,
            "description": "registered test evidence",
            "relative_path": f"evidence/{name}",
            "sha256": digest,
            "produced_by_step": produced_by_step,
            "script_evidence_id": script_evidence_id,
            "inputs": list(inputs or []),
            "producer": producer,
            "generation_mode": generation_mode,
            "prompt_pack_version": "test-prompts/v1",
            "metadata": dict(metadata or {}),
        }
    )
    index_path.write_text(json.dumps(records), encoding="utf-8")
    return path, digest


def test_review_file_labels_resolve_to_digest_pinned_registry_records(
    tmp_path: Path,
) -> None:
    _, report_digest = _register(
        tmp_path,
        evidence_id="reviewer_report_json",
        kind="log",
        name="reviewer_report_json__reviewer_report.json",
        content=b'{"status":"blocked"}',
    )
    _, literature_digest = _register(
        tmp_path,
        evidence_id="preplan_literature_bundle",
        kind="log",
        name="preplan_literature_bundle__preplan_literature_bundle.json",
        content=b'{"screening_decisions":[]}',
    )

    projected = project_review_evidence_refs(
        tmp_path,
        [
            "reviewer_report.json",
            "preplan_literature_bundle.json.screening_decisions",
            "StudyContext.idea_handoff",
        ],
    )

    assert projected == [
        {
            "reference": "reviewer_report.json",
            "evidence_id": "reviewer_report_json",
            "sha256": report_digest,
            "kind": "log",
            "label": "reviewer_report.json",
            "description": "registered test evidence",
        },
        {
            "reference": "preplan_literature_bundle.json.screening_decisions",
            "evidence_id": "preplan_literature_bundle",
            "sha256": literature_digest,
            "kind": "log",
            "label": "preplan_literature_bundle.json.screening_decisions",
            "description": "registered test evidence",
            "pointer": "/screening_decisions",
        },
    ]


def test_figure_gallery_projects_only_verified_unique_source_code(tmp_path: Path) -> None:
    supporting_path, supporting_digest = _register(
        tmp_path,
        evidence_id="code_supporting",
        kind="code",
        name="code_supporting__analysis.py",
        content=b"print('supporting')\n",
        produced_by_step="11_cohort_figure",
        producer="standard_executor",
        generation_mode="deterministic_standard",
    )
    _path, primary_digest = _register(
        tmp_path,
        evidence_id="publication_renderer",
        kind="code",
        name="publication_renderer__skill.py",
        content=b"print('publication')\n",
        produced_by_step="",
        producer="publication_figure_skill",
        generation_mode="deterministic_figure_skill",
        metadata={"figure_id": "main_figure", "artifact_role": "figure_renderer"},
    )
    _register(
        tmp_path,
        evidence_id="figure_supporting",
        kind="figure",
        name="figure_supporting__cohort.png",
        content=b"png",
        produced_by_step="11_cohort_figure",
        script_evidence_id="code_supporting",
    )
    gallery = {
        "figures": [
            {
                "figure_id": "figure:cohort",
                "label": "Cohort flow",
                "relative_path": "steps/11_cohort_figure/outputs/cohort.png",
            },
            {
                "figure_id": "main_figure",
                "label": "Primary figure",
                "relative_path": "publication_figures/main_figure.png",
            },
            {
                "label": "Presented cohort flow",
                "name": "figure_supporting__cohort.png",
                "source_evidence_id": "figure_supporting",
            },
        ]
    }

    projected = project_figure_source_code(tmp_path, gallery)

    assert "source_code" not in gallery["figures"][0]
    assert projected["figures"][0]["source_code"] == {
        "evidence_id": "code_supporting",
        "sha256": supporting_digest,
        "display_name": "analysis.py",
        "language": "python",
        "description": "registered test evidence",
        "produced_by_step": "11_cohort_figure",
        "producer": "standard_executor",
        "generation_mode": "deterministic_standard",
    }
    assert projected["figures"][1]["source_code"]["sha256"] == primary_digest
    assert projected["figures"][1]["source_code"]["display_name"] == "skill.py"
    assert projected["figures"][2]["source_code"]["evidence_id"] == "code_supporting"
    assert projected["figures"][2]["source_code"]["produced_by_step"] == "11_cohort_figure"
    assert str(tmp_path) not in json.dumps(projected)

    supporting_path.write_text("print('tampered')\n", encoding="utf-8")
    tampered = project_figure_source_code(tmp_path, gallery)
    assert "source_code" not in tampered["figures"][0]


def test_figure_gallery_resolves_declared_wrapper_pipeline_run(tmp_path: Path) -> None:
    wrapper = tmp_path / "wrapper"
    pipeline_run = wrapper / "pipeline" / "run_test"
    pipeline_run.mkdir(parents=True)
    (wrapper / "source_run_manifest.json").write_text(
        json.dumps({"run_id": "run_test"}), encoding="utf-8"
    )
    _path, code_digest = _register(
        pipeline_run,
        evidence_id="code_figure",
        kind="code",
        name="code_figure__analysis.py",
        content=b"print('figure')\n",
        produced_by_step="figure_step",
    )
    _register(
        pipeline_run,
        evidence_id="figure_registered",
        kind="figure",
        name="figure_registered__plot.png",
        content=b"png",
        produced_by_step="figure_step",
        script_evidence_id="code_figure",
    )

    projected = project_figure_source_code(
        wrapper,
        {"figures": [{"source_evidence_id": "figure_registered", "name": "plot.png"}]},
    )

    assert projected["figures"][0]["source_code"]["evidence_id"] == "code_figure"
    assert projected["figures"][0]["source_code"]["sha256"] == code_digest
    assert str(pipeline_run) not in json.dumps(projected)


def test_figure_gallery_focuses_large_renderer_on_registered_figure_id(
    tmp_path: Path,
) -> None:
    lines = [f"line_{index} = {index}" for index in range(1, 60)]
    lines[34] = "figure_id = 'main_figure'"
    _path, digest = _register(
        tmp_path,
        evidence_id="publication_renderer",
        kind="code",
        name="publication_renderer__skill.py",
        content=("\n".join(lines) + "\n").encode(),
        metadata={"figure_id": "main_figure"},
    )

    projected = project_figure_source_code(
        tmp_path,
        {"figures": [{"figure_id": "main_figure", "relative_path": "publication_figures/main.png"}]},
    )

    source = projected["figures"][0]["source_code"]
    assert source["sha256"] == digest
    assert source["focus_start_line"] == 23
    assert source["focus_end_line"] == 59


def test_figure_gallery_uses_renderer_metadata_when_gallery_omits_figure_id(
    tmp_path: Path,
) -> None:
    lines = [f"line_{index} = {index}" for index in range(1, 70)]
    lines[41] = "figure_id = 'easyicu_publication_figure'"
    _path, digest = _register(
        tmp_path,
        evidence_id="publication_renderer",
        kind="code",
        name="publication_renderer__skill.py",
        content=("\n".join(lines) + "\n").encode(),
        metadata={"figure_id": "easyicu_publication_figure"},
    )
    _register(
        tmp_path,
        evidence_id="publication_png",
        kind="figure",
        name="publication_png__easyicu_publication_figure.png",
        content=b"png",
        script_evidence_id="publication_renderer",
        metadata={"figure_id": "easyicu_publication_figure"},
    )

    projected = project_figure_source_code(
        tmp_path,
        {"figures": [{"source_evidence_id": "publication_png", "name": "figure.png"}]},
    )

    source = projected["figures"][0]["source_code"]
    assert source["sha256"] == digest
    assert source["focus_start_line"] == 30
    assert source["focus_end_line"] == 66


def test_registered_code_preview_is_digest_pinned_and_host_path_free(
    tmp_path: Path,
) -> None:
    _path, digest = _register(
        tmp_path,
        evidence_id="code_analysis_1",
        kind="code",
        name="code_analysis_1__analysis.py",
        content=b"value = 42\nprint(value)\n",
    )

    preview = build_evidence_preview(tmp_path, "code_analysis_1", digest)

    assert preview["renderer"] == "code"
    assert preview["language"] == "python"
    assert preview["line_count"] == 2
    assert preview["text"] == "value = 42\nprint(value)\n"
    assert preview["display_name"] == "analysis.py"
    assert preview["relative_path"] == "evidence/code_analysis_1__analysis.py"
    assert str(tmp_path) not in json.dumps(preview)


def test_statistic_json_preview_is_structured(tmp_path: Path) -> None:
    _path, code_digest = _register(
        tmp_path,
        evidence_id="code_analysis_1",
        kind="code",
        name="code_analysis_1__analysis.py",
        content=b"estimate = 1.25\n",
        producer="coder",
        generation_mode="llm",
    )
    _path, input_digest = _register(
        tmp_path,
        evidence_id="analysis_cohort_1",
        kind="table",
        name="analysis_cohort_1__cohort.parquet",
        content=b"PAR1 cohort placeholder",
        produced_by_step="cohort_definition",
        producer="cohort",
    )
    _path, plan_digest = _register(
        tmp_path,
        evidence_id="analysis_plan",
        kind="log",
        name="analysis_plan__analysis_plan.json",
        content=b'{"steps": ["analysis_step"]}',
        produced_by_step="",
        producer="planner",
        generation_mode="llm",
    )
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": "run_test",
                "code_version": {"git_sha": "abc123", "git_dirty": False},
                "execution_identity": {
                    "runner": "docker",
                    "runner_image_digest": "sha256:image",
                    "environment_identity_sha256": "e" * 64,
                    "prompt_pack_sha256": "p" * 64,
                    "identity_sha256": "i" * 64,
                    "paper_eligible": False,
                },
                "prompt_pack_version": "test-prompts/v1",
                "current_plan_authority": {
                    "evidence_id": "analysis_plan",
                    "sha256": plan_digest,
                },
            }
        ),
        encoding="utf-8",
    )
    _path, digest = _register(
        tmp_path,
        evidence_id="statistic_summary_1",
        kind="statistic",
        name="statistic_summary_1__summary.json",
        content=b'{"estimate": 1.25, "ci": [1.1, 1.4]}',
        script_evidence_id="code_analysis_1",
        inputs=["analysis_cohort_1"],
    )

    preview = build_evidence_preview(tmp_path, "statistic_summary_1", digest)

    assert preview["renderer"] == "json"
    assert preview["value"]["estimate"] == 1.25
    assert preview["relative_path"] == "evidence/statistic_summary_1__summary.json"
    assert [row["relation"] for row in preview["declared_lineage"]] == [
        "analysis_code",
        "input_data",
    ]
    assert preview["declared_lineage"][0]["sha256"] == code_digest
    assert preview["declared_lineage"][1]["sha256"] == input_digest
    assert preview["run_authority"]["run_id"] == "run_test"
    assert preview["run_authority"]["git_sha"] == "abc123"
    assert preview["run_authority"]["links"][0]["relation"] == ("run_plan_authority")
    assert preview["run_authority"]["links"][0]["sha256"] == plan_digest


def test_missing_declared_parent_is_shown_as_unregistered(tmp_path: Path) -> None:
    _path, digest = _register(
        tmp_path,
        evidence_id="statistic_summary_1",
        kind="statistic",
        name="statistic_summary_1__summary.json",
        content=b'{"estimate": 1.25}',
        script_evidence_id="missing_code",
    )

    preview = build_evidence_preview(tmp_path, "statistic_summary_1", digest)

    assert preview["declared_lineage"] == [
        {
            "relation": "analysis_code",
            "evidence_id": "missing_code",
            "status": "unregistered",
        }
    ]
    assert preview["run_authority"] == {"status": "not_recorded", "links": []}


def test_result_csv_is_bounded_and_identifier_csv_is_withheld(tmp_path: Path) -> None:
    _path, digest = _register(
        tmp_path,
        evidence_id="table_result_1",
        kind="table",
        name="table_result_1__result.csv",
        content=b"contrast,estimate\n5 vs 2.1,1.89\n",
        script_evidence_id="code_analysis_1",
    )
    preview = build_evidence_preview(tmp_path, "table_result_1", digest)
    assert preview["renderer"] == "table"
    assert preview["headers"] == ["contrast", "estimate"]
    assert preview["rows"] == [["5 vs 2.1", "1.89"]]

    _path, sensitive_digest = _register(
        tmp_path,
        evidence_id="table_sensitive_1",
        kind="table",
        name="table_sensitive_1__result.csv",
        content=b"estimate,subject_id\n1.2,123\n",
        script_evidence_id="code_analysis_1",
    )
    sensitive = build_evidence_preview(tmp_path, "table_sensitive_1", sensitive_digest)
    assert sensitive["renderer"] == "metadata"
    assert sensitive["withheld_reason"] == "direct_identifier_columns_withheld"
    assert "rows" not in sensitive


def test_raw_cohort_is_metadata_only(tmp_path: Path) -> None:
    _path, digest = _register(
        tmp_path,
        evidence_id="analysis_cohort_1",
        kind="table",
        name="analysis_cohort_1__cohort.parquet",
        content=b"PAR1 patient-level placeholder",
        produced_by_step="cohort_definition",
    )

    preview = build_evidence_preview(tmp_path, "analysis_cohort_1", digest)

    assert preview["renderer"] == "metadata"
    assert preview["withheld_reason"] == "patient_level_rows_withheld"
    assert "text" not in preview and "rows" not in preview


def test_expected_and_actual_digests_fail_closed(tmp_path: Path) -> None:
    path, digest = _register(
        tmp_path,
        evidence_id="code_analysis_1",
        kind="code",
        name="code_analysis_1__analysis.py",
        content=b"value = 42\n",
    )
    with pytest.raises(EvidencePreviewError) as wrong:
        build_evidence_preview(tmp_path, "code_analysis_1", "0" * 64)
    assert wrong.value.code == "evidence_preview_sha_mismatch"

    path.write_text("value = 43\n", encoding="utf-8")
    with pytest.raises(EvidencePreviewError) as tampered:
        build_evidence_preview(tmp_path, "code_analysis_1", digest)
    assert tampered.value.code == "evidence_preview_digest_mismatch"


def test_path_escape_and_host_path_text_are_withheld(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    outside = tmp_path.parent / "outside-evidence.py"
    outside.write_text("safe = True\n", encoding="utf-8")
    digest = hashlib.sha256(outside.read_bytes()).hexdigest()
    (evidence / "evidence_index.json").write_text(
        json.dumps(
            [
                {
                    "evidence_id": "code_escape_1",
                    "kind": "code",
                    "relative_path": "../outside-evidence.py",
                    "sha256": digest,
                }
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(EvidencePreviewError) as escaped:
        build_evidence_preview(tmp_path, "code_escape_1", digest)
    assert escaped.value.code == "evidence_preview_path_invalid"

    _path, host_digest = _register(
        tmp_path,
        evidence_id="code_host_path_1",
        kind="code",
        name="code_host_path_1__analysis.py",
        content=b"source = '/Users/example/private.csv'\n",
    )
    withheld = build_evidence_preview(tmp_path, "code_host_path_1", host_digest)
    assert withheld["renderer"] == "metadata"
    assert withheld["withheld_reason"] == "evidence_preview_host_path_detected"


def test_registered_symlink_is_rejected_even_when_target_is_inside_run(
    tmp_path: Path,
) -> None:
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    target = evidence / "target.py"
    target.write_text("value = 1\n", encoding="utf-8")
    link = evidence / "linked.py"
    link.symlink_to(target.name)
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    (evidence / "evidence_index.json").write_text(
        json.dumps(
            [
                {
                    "evidence_id": "code_link_1",
                    "kind": "code",
                    "relative_path": "evidence/linked.py",
                    "sha256": digest,
                }
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(EvidencePreviewError) as blocked:
        build_evidence_preview(tmp_path, "code_link_1", digest)
    assert blocked.value.code == "evidence_preview_symlink_forbidden"
