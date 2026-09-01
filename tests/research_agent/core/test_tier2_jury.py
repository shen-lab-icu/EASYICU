from __future__ import annotations

import os
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import pytest

from easyicu.research_agent.evaluation.tier2_jury import (
    REAL_JUDGE_SPECS,
    JudgeClient,
    JudgeIdentity,
    JudgeScore,
    JuryRunner,
    OpenAIJudge,
    default_mock_judges,
    krippendorff_alpha,
    make_real_judges,
)
from easyicu.research_agent.evaluation.tier2_rubric import NPJ_DM_RUBRIC_V1


class ScriptedJudge(JudgeClient):
    def __init__(self, identity: JudgeIdentity, scores_by_run: Dict[str, int]):
        super().__init__(identity=identity)
        self.scores_by_run = scores_by_run

    def score_run(
        self,
        *,
        run_artifact_bundle: Dict[str, str],
        rubric,
        prompt_text: str,
        prompt_hash: str,
    ) -> List[JudgeScore]:
        run_id = run_artifact_bundle["__run_id__"]
        return [
            JudgeScore(
                run_id=run_id,
                judge_id=self.identity.judge_id,
                dimension_id=dimension.dimension_id,
                score=self.scores_by_run[run_id],
                rationale="scripted",
                prompt_sha256=prompt_hash,
            )
            for dimension in rubric.dimensions
        ]


def _bundle(run_id: str, text: str = "Manuscript text.") -> Dict[str, str]:
    return {
        "__run_id__": run_id,
        "manifest.json": json.dumps({"run_id": run_id, "gates": {"evidence_complete": True}}),
        "manuscript_scaffold_bound.md": text,
    }


def _tier_protocol_path() -> Path | None:
    configured = os.environ.get("EASYICU_TIER_PROTOCOL_PATH")
    candidates = []
    if configured:
        candidates.append(Path(configured))
    candidates.append(
        Path(__file__).resolve().parents[3]
        / "easyicu写作"
        / "00_当前投稿_20260516"
        / "02_npj_Digital_Medicine"
        / "tier_evaluation_protocol_20260527.md"
    )
    for path in candidates:
        if path.is_file():
            return path
    return None


def test_rubric_definition_is_complete():
    assert NPJ_DM_RUBRIC_V1.version == "npj_dm_rubric/20260527"
    assert NPJ_DM_RUBRIC_V1.dimension_ids == [
        "plan_completeness",
        "evidence_binding",
        "missingness_handling",
        "overclaim_avoidance",
    ]
    for dimension in NPJ_DM_RUBRIC_V1.dimensions:
        assert set(dimension.anchors) == {0, 1, 2, 3}
        assert all(anchor.strip() for anchor in dimension.anchors.values())


def test_rubric_anchors_match_protocol_doc():
    protocol_path = _tier_protocol_path()
    if protocol_path is None:
        pytest.skip("Tier-2 protocol manuscript document is not available in this checkout")
    protocol = protocol_path.read_text(encoding="utf-8")
    for dimension in NPJ_DM_RUBRIC_V1.dimensions:
        assert dimension.label in protocol
        for anchor in dimension.anchors.values():
            assert anchor in protocol


def test_mock_judge_deterministic(monkeypatch):
    monkeypatch.delenv("EASYICU_MOCK_JUDGE_OVERRIDES", raising=False)
    judge = default_mock_judges()[0]
    kwargs = {
        "run_artifact_bundle": _bundle("run-a"),
        "rubric": NPJ_DM_RUBRIC_V1,
        "prompt_text": "same prompt",
        "prompt_hash": "abc",
    }
    first = judge.score_run(**kwargs)
    second = judge.score_run(**kwargs)
    assert [score.score for score in first] == [2, 2, 2, 2]
    assert first == second


def test_mock_judge_env_override_lets_judges_disagree(monkeypatch):
    monkeypatch.setenv(
        "EASYICU_MOCK_JUDGE_OVERRIDES",
        json.dumps({
            "mock_judge_1|run-a|plan_completeness": 0,
            "mock_judge_2|run-a|plan_completeness": 1,
            "mock_judge_3|run-a|plan_completeness": 3,
        }),
    )
    report = JuryRunner(
        judges=default_mock_judges(),
        rubric=NPJ_DM_RUBRIC_V1,
        seed=1,
    ).score_runs([_bundle("run-a")])
    plan_scores = [
        score.score
        for score in report.scores
        if score.dimension_id == "plan_completeness"
    ]
    assert sorted(plan_scores) == [0, 1, 3]


def test_jury_requires_disjoint_families():
    judges = [
        ScriptedJudge(JudgeIdentity("a", "anthropic", "a", "anthropic"), {"run-a": 2}),
        ScriptedJudge(JudgeIdentity("b", "anthropic", "b", "anthropic"), {"run-a": 2}),
        ScriptedJudge(JudgeIdentity("c", "anthropic", "c", "anthropic"), {"run-a": 2}),
    ]
    with pytest.raises(ValueError, match="disjoint model families"):
        JuryRunner(judges=judges, rubric=NPJ_DM_RUBRIC_V1, seed=1)


def test_jury_mock_path_does_not_enforce_families():
    runner = JuryRunner(judges=default_mock_judges(), rubric=NPJ_DM_RUBRIC_V1, seed=1)
    report = runner.score_runs([_bundle("run-a")])
    assert {judge.family for judge in report.judges} == {"mock"}


def test_prompt_byte_identical_across_judges():
    report = JuryRunner(
        judges=default_mock_judges(),
        rubric=NPJ_DM_RUBRIC_V1,
        seed=1,
    ).score_runs([_bundle("run-a")])
    hashes = {score.prompt_sha256 for score in report.scores}
    assert len(hashes) == 1


def test_position_randomization_is_seeded():
    runs = [_bundle(f"run-{idx}") for idx in range(6)]
    first = JuryRunner(default_mock_judges(), NPJ_DM_RUBRIC_V1, seed=7).score_runs(runs)
    second = JuryRunner(default_mock_judges(), NPJ_DM_RUBRIC_V1, seed=7).score_runs(runs)
    different = JuryRunner(default_mock_judges(), NPJ_DM_RUBRIC_V1, seed=11).score_runs(runs)
    assert first.run_order == second.run_order
    assert first.run_order != different.run_order
    assert sorted(first.run_order) == [f"run-{idx}" for idx in range(6)]


def test_krippendorff_alpha_perfect_agreement():
    assert krippendorff_alpha([[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]]) == 1.0


def test_krippendorff_alpha_krippendorff_c_data_ordinal_example():
    # The "C" data from Krippendorff, as reproduced in the R irr::kripp.alpha
    # documentation. R's matrix() fills by column, so each row below is a rater.
    na = None
    scores_by_judge = [
        [1, 2, 3, 3, 2, 1, 4, 1, 2, na, na, na],
        [1, 2, 3, 3, 2, 2, 4, 1, 2, 5, na, na],
        [na, 3, 3, 3, 2, 3, 4, 2, 2, 5, 1, 3],
        [1, 2, 3, 3, 2, 4, 4, 1, 2, 5, 1, na],
    ]
    assert krippendorff_alpha(scores_by_judge) == pytest.approx(0.8154, abs=0.01)


def test_alpha_below_0_4_flagged_below_0_2_retired():
    run_ids = [f"run-{idx}" for idx in range(4)]
    judges = [
        ScriptedJudge(JudgeIdentity("j1", "anthropic", "a", "anthropic"), {rid: 0 for rid in run_ids}),
        ScriptedJudge(JudgeIdentity("j2", "openai", "o", "openai"), {rid: 3 for rid in run_ids}),
        ScriptedJudge(JudgeIdentity("j3", "google", "g", "google"), {rid: 3 for rid in run_ids}),
    ]
    report = JuryRunner(judges=judges, rubric=NPJ_DM_RUBRIC_V1, seed=1).score_runs(
        [_bundle(rid) for rid in run_ids]
    )
    assert set(report.flagged_dimensions) == set(NPJ_DM_RUBRIC_V1.dimension_ids)
    assert set(report.retired_dimensions) == set(NPJ_DM_RUBRIC_V1.dimension_ids)


def test_real_judge_requires_explicit_flag(monkeypatch):
    monkeypatch.delenv("EASYICU_ENABLE_REAL_JUDGES", raising=False)
    judge = OpenAIJudge(
        JudgeIdentity("gpt_5_5", "openai", "gpt-5.5", "openai"),
        model="gpt-5.5",
        api_key_env="EASYICU_JUDGE_GPT_5_5_API_KEY",
        api_key="dummy",
    )
    with pytest.raises(RuntimeError, match="EASYICU_ENABLE_REAL_JUDGES=1"):
        judge.score_run(
            run_artifact_bundle=_bundle("run-a"),
            rubric=NPJ_DM_RUBRIC_V1,
            prompt_text="prompt",
            prompt_hash="abc",
        )


def test_openrouter_judges_are_single_key_disjoint_families():
    """The OpenRouter judge bundle must let one OPENROUTER_API_KEY drive a
    valid 3-judge jury: disjoint families, free slugs, and no per-judge
    frontier-provider env vars. This is what makes Tier-2 actually runnable."""
    or_ids = ["or_llama_70b", "or_qwen3_next", "or_gpt_oss_120b"]
    for jid in or_ids:
        spec = REAL_JUDGE_SPECS[jid]
        assert spec["api_key_env"] == "OPENROUTER_API_KEY"
        assert spec["base_url_env"] == "OPENROUTER_BASE_URL"
        assert spec["snapshot"].endswith(":free")

    judges = make_real_judges(or_ids)
    families = [j.identity.family for j in judges]
    assert sorted(families) == ["meta", "openai", "qwen"]  # disjoint
    # Must satisfy the runner's disjoint-family invariant without raising.
    JuryRunner(judges=judges, rubric=NPJ_DM_RUBRIC_V1, seed=7)


def test_real_judge_parser_tolerates_reasoning_preamble():
    """Judge responses with thinking/preamble before the JSON must still
    parse (reasoning models emit prose first). Covers fenced, prefixed, and
    bare-array forms."""
    from easyicu.research_agent.evaluation.tier2_jury import _parse_real_judge_response

    preamble = (
        'The user wants scores. Let me think... here is my answer:\n'
        '{"scores": [{"dimension_id": "plan_completeness", "score": 2, '
        '"rationale": "ok"}]}\nDone.'
    )
    out = _parse_real_judge_response(preamble)
    assert out == [{"dimension_id": "plan_completeness", "score": 2, "rationale": "ok"}]

    fenced = '```json\n[{"dimension_id": "x", "score": 1}]\n```'
    assert _parse_real_judge_response(fenced) == [{"dimension_id": "x", "score": 1}]

    with pytest.raises(ValueError):
        _parse_real_judge_response("no json here at all")


def test_cli_smoke_with_mock_judges(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    if not (repo_root / "tools" / "run_tier2_jury.py").exists():
        # tools/run_tier2_jury.py is excluded from the slimmed public
        # repository surface (see 874db1c); the CLI smoke can only run where
        # the script is kept.
        pytest.skip("tools/run_tier2_jury.py not present (excluded from the public repo surface)")
    for idx in range(2):
        run_dir = tmp_path / f"run_{idx}"
        run_dir.mkdir()
        (run_dir / "manifest.json").write_text(
            json.dumps({"run_id": f"run_{idx}", "gates": {"evidence_complete": True}}),
            encoding="utf-8",
        )
        (run_dir / "manuscript_scaffold_bound.md").write_text(
            "The run has evidence-bound text.",
            encoding="utf-8",
        )
    out = tmp_path / "tier2_jury_report.json"
    result = subprocess.run(
        [
            sys.executable,
            "tools/run_tier2_jury.py",
            "--run-dirs",
            str(tmp_path / "run_*"),
            "--rubric",
            "npj_dm_rubric/20260527",
            "--out",
            str(out),
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["rubric_version"] == "npj_dm_rubric/20260527"
    assert {judge["family"] for judge in payload["judges"]} == {"mock"}
    assert len(payload["scores"]) == 2 * 3 * 4
