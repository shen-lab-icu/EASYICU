"""Post-QC development sampling stays deterministic and non-paper."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from easyicu.research_agent.authority.execution_input import (
    ExecutionInputAuthorityState,
)
from easyicu.research_agent.execution.development_sample import (
    DEVELOPMENT_COHORT_FILENAME,
    DEVELOPMENT_SAMPLE_FILENAME,
    DEVELOPMENT_TRAJECTORY_FILENAME,
    DevelopmentSampleError,
    materialize_development_execution_sample,
)
from easyicu.research_agent.intake.materialized_trajectory import (
    StagedTrajectoryBinding,
)
from easyicu.research_agent.orchestration.config import PipelineConfig
from easyicu.research_agent.pipeline import ResearchAgentPipeline


def _sha(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _locked_analysis(run_dir: Path, *, rows: int = 20) -> Path:
    path = run_dir / "cohort_analysis.parquet"
    pd.DataFrame(
        {
            "stay_id": list(range(1, rows + 1)),
            "exposure": [index % 3 for index in range(rows)],
            "death": [index % 5 == 0 for index in range(rows)],
        }
    ).to_parquet(path, index=False)
    return path


def _trajectory(run_dir: Path, *, rows: int = 20) -> StagedTrajectoryBinding:
    path = run_dir / "cohort_trajectory.parquet"
    pd.DataFrame(
        {
            "stay_id": [stay for stay in range(1, rows + 1) for _ in range(3)],
            "charttime": [float(index % 3) for index in range(rows * 3)],
            "concept": ["score" for _ in range(rows * 3)],
            "value_num": [float(index) for index in range(rows * 3)],
            "value_str": ["" for _ in range(rows * 3)],
        }
    ).to_parquet(path, index=False)
    return StagedTrajectoryBinding(
        path=path,
        sha256=_sha(path),
        size=path.stat().st_size,
    )


def test_samples_only_after_locked_analysis_cohort_exists(tmp_path: Path) -> None:
    with pytest.raises(DevelopmentSampleError, match="locked, materialized"):
        materialize_development_execution_sample(
            run_dir=tmp_path,
            target_rows=5,
            seed=7,
            declared_id_columns=("stay_id",),
            trajectory_binding=None,
        )


def test_sample_is_deterministic_unstratified_and_non_paper(tmp_path: Path) -> None:
    parent = _locked_analysis(tmp_path, rows=20)
    parent_bytes = parent.read_bytes()

    binding = materialize_development_execution_sample(
        run_dir=tmp_path,
        target_rows=7,
        seed=42,
        declared_id_columns=("stay_id",),
        trajectory_binding=None,
    )

    assert parent.read_bytes() == parent_bytes
    assert binding.cohort_path.name == DEVELOPMENT_COHORT_FILENAME
    assert binding.selected_rows == 7
    assert len(pd.read_parquet(binding.cohort_path)) == 7
    manifest = json.loads(
        (tmp_path / DEVELOPMENT_SAMPLE_FILENAME).read_text(encoding="utf-8")
    )
    assert manifest["paper_authority"] is False
    assert manifest["algorithm"] == "sha256_identity_rank_v1"
    # Sampling is deliberately not stratified on exposure/outcome. Those
    # scientific columns are preserved but never consulted by the selector.
    assert "strat" not in json.dumps(manifest).casefold()

    resumed = materialize_development_execution_sample(
        run_dir=tmp_path,
        target_rows=7,
        seed=42,
        declared_id_columns=("stay_id",),
        trajectory_binding=None,
    )
    assert resumed == binding


def test_trajectory_is_stream_filtered_to_sampled_stays(tmp_path: Path) -> None:
    _locked_analysis(tmp_path, rows=20)
    trajectory = _trajectory(tmp_path, rows=20)

    binding = materialize_development_execution_sample(
        run_dir=tmp_path,
        target_rows=6,
        seed=9,
        declared_id_columns=("stay_id",),
        trajectory_binding=trajectory,
    )

    selected = set(pd.read_parquet(binding.cohort_path)["stay_id"])
    filtered = pd.read_parquet(tmp_path / DEVELOPMENT_TRAJECTORY_FILENAME)
    assert set(filtered["stay_id"]) == selected
    assert len(filtered) == 3 * len(selected)
    assert binding.trajectory_binding is not None
    assert binding.trajectory_binding.sha256 == _sha(
        tmp_path / DEVELOPMENT_TRAJECTORY_FILENAME
    )


def test_manifest_or_parent_tamper_fails_closed_on_resume(tmp_path: Path) -> None:
    parent = _locked_analysis(tmp_path, rows=10)
    materialize_development_execution_sample(
        run_dir=tmp_path,
        target_rows=4,
        seed=2,
        declared_id_columns=("stay_id",),
        trajectory_binding=None,
    )
    frame = pd.read_parquet(parent)
    frame.loc[0, "death"] = not bool(frame.loc[0, "death"])
    frame.to_parquet(parent, index=False)

    with pytest.raises(DevelopmentSampleError, match="changed across"):
        materialize_development_execution_sample(
            run_dir=tmp_path,
            target_rows=4,
            seed=2,
            declared_id_columns=("stay_id",),
            trajectory_binding=None,
        )


def test_coordinated_sample_and_manifest_forgery_fails_closed(
    tmp_path: Path,
) -> None:
    _locked_analysis(tmp_path, rows=10)
    binding = materialize_development_execution_sample(
        run_dir=tmp_path,
        target_rows=4,
        seed=2,
        declared_id_columns=("stay_id",),
        trajectory_binding=None,
    )
    forged = pd.read_parquet(binding.cohort_path)
    forged.loc[0, "stay_id"] = 999999
    forged.to_parquet(binding.cohort_path, index=False)
    manifest_path = tmp_path / DEVELOPMENT_SAMPLE_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["sample"]["sha256"] = _sha(binding.cohort_path)
    manifest["sample"]["size"] = binding.cohort_path.stat().st_size
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(DevelopmentSampleError, match="deterministic parent subset"):
        materialize_development_execution_sample(
            run_dir=tmp_path,
            target_rows=4,
            seed=2,
            declared_id_columns=("stay_id",),
            trajectory_binding=None,
        )


def test_execution_authority_selects_sample_but_keeps_full_universe(
    tmp_path: Path,
) -> None:
    universe = tmp_path / "cohort.parquet"
    pd.DataFrame({"stay_id": list(range(1, 21))}).to_parquet(universe, index=False)
    _locked_analysis(tmp_path, rows=20)
    binding = materialize_development_execution_sample(
        run_dir=tmp_path,
        target_rows=5,
        seed=3,
        declared_id_columns=("stay_id",),
        trajectory_binding=None,
    )
    state = ExecutionInputAuthorityState.bind(
        universe_path=universe,
        analysis_path=tmp_path / "cohort_analysis.parquet",
        trajectory_binding=None,
        run_dir=tmp_path,
        legacy_trajectory_verifier=None,
        plan=SimpleNamespace(),
        context=SimpleNamespace(),
        development_sample=binding,
    )

    assert state.selected_path == binding.cohort_path
    assert state.universe_path == universe
    assert len(pd.read_parquet(state.selected_path)) == 5


def test_pipeline_config_round_trips_development_sample(tmp_path: Path) -> None:
    config = PipelineConfig(
        workdir=tmp_path,
        development_sample_size=1000,
        development_sample_seed=17,
    )
    pipeline = ResearchAgentPipeline.from_config(config)
    assert pipeline._development_sample_size == 1000
    assert pipeline._development_sample_seed == 17


def test_submission_profile_cannot_enable_development_sample(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="non-paper authority"):
        ResearchAgentPipeline(
            workdir=tmp_path,
            submission_profile_name="npj_dm",
            development_sample_size=1000,
        )


def test_missing_post_qc_cohort_blocks_scientific_steps_without_coder_calls(
    ra, synthetic_cohort, tmp_path: Path
) -> None:
    class CountingLLM(ra.MockLLMClient):
        def __init__(self) -> None:
            super().__init__()
            self.coder_calls = 0

        def complete(self, messages, **kwargs):
            user = next(
                (
                    message.content
                    for message in reversed(messages)
                    if message.role == "user"
                ),
                "",
            )
            if "WRITE THE PYTHON CODE" in user.upper():
                self.coder_calls += 1
            return super().complete(messages, **kwargs)

    llm = CountingLLM()
    result = ResearchAgentPipeline(
        workdir=tmp_path,
        llm=llm,
        development_sample_size=100,
    ).run(
        question="Is admission SOFA associated with ICU mortality?",
        cohort=synthetic_cohort,
        cohort_name="development_sample_requires_post_qc_cohort",
        database="synthetic",
        target_outcome="death",
    )

    manifest = json.loads(Path(result.manifest_path).read_text(encoding="utf-8"))
    blocked = [
        finding
        for finding in manifest.get("findings", [])
        if finding.get("validator") == "development_sample_authority"
        and (finding.get("detail") or {}).get("stage")
        == "blocked_before_scientific_execution"
    ]
    assert blocked
    assert llm.coder_calls == 0
    assert not (Path(result.workdir) / DEVELOPMENT_COHORT_FILENAME).exists()
    assert not [
        record
        for record in manifest.get("per_step_records", [])
        if record.get("step_id") != "00_probe"
    ]
