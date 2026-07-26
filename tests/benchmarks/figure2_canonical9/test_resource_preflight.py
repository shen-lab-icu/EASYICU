from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.figure2_canonical9.evaluator.rubric_v1 import FIGURE2_TASK_IDS
from benchmarks.figure2_canonical9 import resource_preflight as preflight
from easyicu.research_agent.authority.evidence_store import EvidenceStore
from easyicu.research_agent.contracts.execution_result import RunResult
from easyicu.research_agent.intake.materialized_metadata import (
    MaterializedCohortAuthorityRef,
)
from easyicu.research_agent.intake.materialized_trajectory import (
    MaterializedTrajectoryAuthorityRef,
    StagedTrajectoryBinding,
)


def _case(root: Path, task_id: str, index: int) -> preflight.ResourceCase:
    case_dir = root / task_id
    case_dir.mkdir()
    cohort = case_dir / "cohort.parquet"
    cohort.write_bytes(f"cohort-{task_id}".encode())
    cohort_authority = case_dir / f"cohort_authority_{index}.json"
    cohort_authority.write_text("{}", encoding="utf-8")
    cohort_ref = MaterializedCohortAuthorityRef(
        file=cohort_authority.name,
        sha256=f"{index + 1:064x}",
        size=cohort_authority.stat().st_size,
    )
    trajectory_binding = None
    trajectory_rows = None
    trajectory_columns: tuple[str, ...] = ()
    if task_id in {"h2_vasopressor_causal", "h3_trajectory_clustering"}:
        trajectory = case_dir / "cohort_trajectory.parquet"
        trajectory.write_bytes(f"trajectory-{task_id}".encode())
        trajectory_authority = case_dir / f"trajectory_authority_{index}.json"
        trajectory_authority.write_text("{}", encoding="utf-8")
        trajectory_ref = MaterializedTrajectoryAuthorityRef(
            file=trajectory_authority.name,
            sha256=f"{index + 100:064x}",
            size=trajectory_authority.stat().st_size,
        )
        trajectory_binding = StagedTrajectoryBinding(
            path=trajectory,
            sha256=f"{index + 200:064x}",
            size=trajectory.stat().st_size,
            authority_ref=trajectory_ref,
        )
        trajectory_rows = 200 + index
        trajectory_columns = (
            "stay_id",
            "charttime",
            "concept",
            "value_num",
            "value_str",
        )
    return preflight.ResourceCase(
        task_id=task_id,
        case_dir=case_dir,
        cohort_path=cohort,
        cohort_authority_ref=cohort_ref,
        cohort_rows=100 + index,
        cohort_columns=("stay_id", "age", "sex", "death"),
        trajectory_binding=trajectory_binding,
        trajectory_rows=trajectory_rows,
        trajectory_columns=trajectory_columns,
    )


def _probe_payload(
    case: preflight.ResourceCase,
    *,
    h3_sample_stays: int | None = None,
) -> dict[str, object]:
    trajectory = None
    if case.trajectory_binding is not None:
        sampled = h3_sample_stays is not None
        trajectory = {
            "compressed_size_bytes": case.trajectory_binding.size,
            "authority_rows": case.trajectory_rows,
            "authority_columns": len(case.trajectory_columns),
            "loaded_rows": 20 if sampled else case.trajectory_rows,
            "loaded_columns": 3 if sampled else len(case.trajectory_columns),
            "load_mode": "development_sample" if sampled else "full",
            "sample_stays_requested": h3_sample_stays,
            "loaded_stays": min(h3_sample_stays, 10) if sampled else None,
            "load_seconds": 0.25,
            "aggregated_rows": 12,
            "authority_sha256": case.trajectory_binding.authority_ref.sha256,
        }
    return {
        "schema_version": preflight.RESOURCE_PROBE_SCHEMA_VERSION,
        "status": "passed",
        "development_only": True,
        "paper_authorized": False,
        "provider_calls": 0,
        "task_id": case.task_id,
        "full_input_resource_qualified": h3_sample_stays is None,
        "family": preflight._FAMILY_BY_TASK[case.task_id],
        "family_executor": {
            "name": "fake_family",
            "status": "passed",
            "input_rows": 10,
        },
        "table_one": {"status": "passed", "result_rows": 3},
        "cohort": {
            "compressed_size_bytes": case.cohort_path.stat().st_size,
            "rows": case.cohort_rows,
            "columns": len(case.cohort_columns),
            "load_seconds": 0.1,
            "authority_sha256": case.cohort_authority_ref.sha256,
        },
        "trajectory": trajectory,
        "packages": {name: "1.0" for name in preflight._REQUIRED_IMPORTS},
        "mounts": {
            "source_read_only": True,
            "cohort_read_only": True,
            "output_writable": True,
        },
        "scratch": {
            "tmp_write": "passed",
            "tmp_capacity_bytes": 1024,
            "shm_write": "passed",
            "shm_capacity_bytes": 1024,
        },
        "peak_rss_bytes": 4096,
    }


class _FakeRunner:
    def __init__(self, case: preflight.ResourceCase, **kwargs: object) -> None:
        self.case = case
        self.workdir = Path(str(kwargs["workdir"]))
        self.network = str(kwargs["network"])

    def run(self, *, step_id: str, code: str) -> RunResult:
        compile(code, "<resource-probe>", "exec")
        out_dir = self.workdir / "steps" / step_id / "outputs"
        out_dir.mkdir(parents=True)
        payload = _probe_payload(self.case)
        artifact = out_dir / "resource_probe.json"
        artifact.write_text(
            json.dumps(payload, ensure_ascii=False, allow_nan=False),
            encoding="utf-8",
        )
        record = EvidenceStore(out_dir / "temporary_evidence_store").register_file(
            kind="log",
            description="test",
            source_path=artifact,
            evidence_id=f"resource_{self.case.task_id}",
        )
        (out_dir / "resource_probe_registration.json").write_text(
            json.dumps(
                {
                    "evidence_id": record.evidence_id,
                    "sha256": record.sha256,
                    "reopened": True,
                }
            ),
            encoding="utf-8",
        )
        script = self.workdir / "script.py"
        script.write_text(code, encoding="utf-8")
        return RunResult(
            step_id=step_id,
            script_path=script,
            cwd=self.workdir,
            out_dir=out_dir,
            stdout="passed",
            stderr="",
            returncode=0,
            duration_seconds=0.5,
            artefacts=[artifact],
            requested_network_policy=f"docker:{self.network}",
            effective_isolation=f"docker_network_{self.network}",
            runtime_provenance={
                "image_id": f"sha256:{'a' * 64}",
                "repo_digests": [f"image@sha256:{'a' * 64}"],
                "requirements_sha256": "b" * 64,
            },
        )


def test_probe_source_compiles_and_contains_complete_zero_provider_contract(
    tmp_path: Path,
) -> None:
    case = _case(tmp_path, "h3_trajectory_clustering", 8)
    source = preflight._probe_code(case)

    compile(source, "<resource-probe>", "exec")
    assert "pd.read_parquet" in source
    assert "load_verified_materialized_trajectory_authority" in source
    assert "build_grouped_table_one" in source
    assert "EvidenceStore" in source
    assert 'provider_calls": 0' in source
    assert "OpenAI" not in source


def test_h3_sample_probe_is_explicit_bounded_and_not_full_input_qualified(
    tmp_path: Path,
) -> None:
    case = _case(tmp_path, "h3_trajectory_clustering", 8)
    source = preflight._probe_code(case, h3_sample_stays=512)
    payload = _probe_payload(case, h3_sample_stays=512)

    compile(source, "<resource-probe>", "exec")
    assert "TRAJECTORY_SAMPLE_STAYS = 512" in source
    assert "SEMI JOIN selected_stays" in source
    assert "pd.read_parquet(TRAJECTORY_PATH)" not in source
    preflight._validate_probe(case, payload, h3_sample_stays=512)
    assert payload["full_input_resource_qualified"] is False


def test_h3_sample_probe_rejects_false_full_input_claim(tmp_path: Path) -> None:
    case = _case(tmp_path, "h3_trajectory_clustering", 8)
    payload = _probe_payload(case, h3_sample_stays=512)
    payload["full_input_resource_qualified"] = True

    with pytest.raises(preflight.ResourcePreflightError, match="changed"):
        preflight._validate_probe(case, payload, h3_sample_stays=512)


def test_probe_validation_fails_closed_on_authority_dimension_drift(
    tmp_path: Path,
) -> None:
    case = _case(tmp_path, "e1_sepsis3_prevalence_mortality", 0)
    payload = _probe_payload(case)
    payload["cohort"]["rows"] = case.cohort_rows + 1  # type: ignore[index]

    with pytest.raises(preflight.ResourcePreflightError, match="cohort metrics"):
        preflight._validate_probe(case, payload)


def test_resource_preflight_is_sequential_zero_provider_and_source_preserving(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    cases = [
        _case(source_root, task_id, index)
        for index, task_id in enumerate(FIGURE2_TASK_IDS)
    ]
    case_by_cohort = {case.cohort_path: case for case in cases}
    jsonl = tmp_path / "canonical9.jsonl"
    jsonl.write_text("{}\n", encoding="utf-8")
    before = {
        path: path.read_bytes()
        for case in cases
        for path in case.case_dir.iterdir()
        if path.is_file()
    }
    observed_order: list[str] = []

    def factory(**kwargs: object) -> _FakeRunner:
        case = case_by_cohort[Path(str(kwargs["cohort_parquet"]))]
        observed_order.append(case.task_id)
        return _FakeRunner(case, **kwargs)

    monkeypatch.setattr(preflight, "_load_cases", lambda _path: cases)
    monkeypatch.setattr(
        preflight,
        "_git_identity",
        lambda _root: ("c" * 40, ""),
    )
    monkeypatch.setattr(
        preflight,
        "_container_ids",
        lambda _docker: frozenset({"preexisting"}),
    )

    report = preflight.run_canonical9_resource_preflight(
        jsonl_path=jsonl,
        output_dir=tmp_path / "out",
        image="easyicu:test",
        runner_factory=factory,
    )

    assert observed_order == list(FIGURE2_TASK_IDS)
    assert report["status"] == "passed"
    assert report["provider_calls"] == 0
    assert report["source_zero_write_verified"] is True
    assert report["container_cleanup_verified"] is True
    assert report["task_count"] == 9
    assert report["docker_image_id"] == f"sha256:{'a' * 64}"
    assert all(
        task["evidence_registration"] == "passed" for task in report["tasks"]
    )
    assert {
        path: path.read_bytes()
        for path in before
    } == before
    assert (
        tmp_path
        / "out"
        / preflight.RESOURCE_REPORT_FILENAME
    ).is_file()
