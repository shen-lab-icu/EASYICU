"""Strict JSON-object decoding regressions for benchmark handoff rows."""

from __future__ import annotations

import json

import pandas as pd
import pytest


def _completed_score(item_key: str) -> dict[str, object]:
    return {
        "item_key": item_key,
        "aware": {
            "execution_complete": True,
            "step_scientific_requirements_complete": True,
            "failed_step_ids": [],
            "missing_step_ids": [],
        },
    }


def test_strict_jsonl_decoder_preserves_valid_nested_payload() -> None:
    import tools.run_research_agent_bench as bench

    payload = {
        "key": "valid-item",
        "question": "Estimate the declared association.",
        "cohort_path": "/tmp/cohort.parquet",
        "target_outcome": "death",
        "criteria": {"minimum_age": 18, "labels": ["ICU", "adult"]},
        "enabled": True,
        "optional": None,
    }

    assert bench._decode_jsonl_object(json.dumps(payload)) == payload


@pytest.mark.parametrize(
    "raw,duplicate_key",
    [
        ('{"key":"first","key":"second"}', "key"),
        ('{"criteria":{"window":6,"window":12}}', "window"),
    ],
)
def test_strict_jsonl_decoder_rejects_duplicate_keys(
    raw: str,
    duplicate_key: str,
) -> None:
    import tools.run_research_agent_bench as bench

    with pytest.raises(
        bench._JSONLObjectDecodeError,
        match=rf"duplicate JSON key: '{duplicate_key}'",
    ):
        bench._decode_jsonl_object(raw)


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_strict_jsonl_decoder_rejects_nonfinite_constants(constant: str) -> None:
    import tools.run_research_agent_bench as bench

    with pytest.raises(
        bench._JSONLObjectDecodeError,
        match="non-finite JSON constant is forbidden",
    ):
        bench._decode_jsonl_object(f'{{"threshold":{constant}}}')


@pytest.mark.parametrize("raw", ["[1, 2]", '"text"', "null", "3"])
def test_strict_jsonl_decoder_requires_top_level_object(raw: str) -> None:
    import tools.run_research_agent_bench as bench

    with pytest.raises(
        bench._JSONLObjectDecodeError,
        match="benchmark JSONL row must be an object",
    ):
        bench._decode_jsonl_object(raw)


def test_external_jsonl_duplicate_key_is_reported_without_execution(
    tmp_path,
    monkeypatch,
) -> None:
    import tools.run_research_agent_bench as bench

    jsonl = tmp_path / "items.jsonl"
    jsonl.write_text(
        '{"key":"first","key":"second","question":"Q",'
        '"cohort_path":"/tmp/unused.parquet","target_outcome":"death"}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        bench,
        "_run_one_item_from_cohort",
        lambda **_kwargs: pytest.fail("invalid JSONL must not execute"),
    )

    assert (
        bench._run_ehrflowbench_jsonl(
            jsonl_path=jsonl,
            out_root=tmp_path / "out",
            seed=7,
            arms=["aware"],
            provider="openai",
            model="model",
        )
        # The item was rejected at intake and never ran, which is what the
        # rest of this test proves. Exit 0 said that was a passing benchmark.
        == bench._PENDING_ITEMS_EXIT_CODE
    )

    payload = json.loads(
        (tmp_path / "out" / "ehrflowbench_results.json").read_text(encoding="utf-8")
    )
    assert payload["scores"] == []
    assert payload["pending"] == [
        {
            "key": "ehrflowbench_000",
            "status": "invalid_json",
            "error": "duplicate JSON key: 'key'",
            "raw": (
                '{"key":"first","key":"second","question":"Q",'
                '"cohort_path":"/tmp/unused.parquet","target_outcome":"death"}'
            ),
            "line": 1,
        }
    ]


def test_valid_row_status_field_cannot_impersonate_decoder_failure(
    tmp_path,
    monkeypatch,
) -> None:
    import tools.run_research_agent_bench as bench

    cohort = tmp_path / "cohort.parquet"
    pd.DataFrame({"stay_id": [1], "death": [0]}).to_parquet(cohort, index=False)
    jsonl = tmp_path / "items.jsonl"
    jsonl.write_text(
        json.dumps(
            {
                "key": "valid-status-field",
                "status": "invalid_json",
                "question": "Q",
                "cohort_path": str(cohort),
                "target_outcome": "death",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    seen = {}

    def fake_run_one(**kwargs):
        seen.update(kwargs)
        return _completed_score("valid-status-field")

    monkeypatch.setattr(bench, "_run_one_item_from_cohort", fake_run_one)
    monkeypatch.setattr(bench, "_aggregate", lambda _scores: {"aware": {}})
    monkeypatch.setattr(bench, "_render_markdown", lambda **_kwargs: "ok")

    assert (
        bench._run_ehrflowbench_jsonl(
            jsonl_path=jsonl,
            out_root=tmp_path / "out",
            seed=7,
            arms=["aware"],
            provider="openai",
            model="model",
        )
        == 0
    )
    assert seen["item"].key == "valid-status-field"


def test_longitudinal_jsonl_can_run_without_invented_target_outcome(
    tmp_path,
    monkeypatch,
) -> None:
    import tools.run_research_agent_bench as bench

    cohort = tmp_path / "cohort.parquet"
    pd.DataFrame({"stay_id": [1, 2]}).to_parquet(cohort, index=False)
    jsonl = tmp_path / "items.jsonl"
    jsonl.write_text(
        json.dumps(
            {
                "key": "sofa2-trajectory",
                "kind": "longitudinal_trajectory_analysis",
                "question": "Are SOFA-2 trajectories reproducible?",
                "cohort_path": str(cohort),
                "analysis_concepts": ["sofa2"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    seen = {}

    def fake_run_one(**kwargs):
        seen.update(kwargs)
        return _completed_score("sofa2-trajectory")

    monkeypatch.setattr(bench, "_run_one_item_from_cohort", fake_run_one)
    monkeypatch.setattr(bench, "_aggregate", lambda _scores: {"aware": {}})
    monkeypatch.setattr(bench, "_render_markdown", lambda **_kwargs: "ok")

    assert (
        bench._run_ehrflowbench_jsonl(
            jsonl_path=jsonl,
            out_root=tmp_path / "out",
            seed=7,
            arms=["aware"],
            provider="openai",
            model="model",
        )
        == 0
    )
    assert seen["item"].target_outcome is None
    assert seen["item"].kind == "longitudinal_trajectory_analysis"


def test_external_jsonl_binds_exact_question_reviewed_literature(
    tmp_path,
    monkeypatch,
) -> None:
    import tools.run_research_agent_bench as bench

    cohort = tmp_path / "cohort.parquet"
    pd.DataFrame({"stay_id": [1], "outcome": [0]}).to_parquet(cohort, index=False)
    question = "Does the declared exposure predict the outcome?"
    bound_literature = {
        "research_question": question,
        "citations": [],
    }
    jsonl = tmp_path / "items.jsonl"
    jsonl.write_text(
        json.dumps(
            {
                "key": "literature-bound-item",
                "question": question,
                "cohort_path": str(cohort),
                "target_outcome": "outcome",
                "bound_preplan_literature": bound_literature,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    seen = {}

    def fake_run_one(**kwargs):
        seen.update(kwargs)
        return _completed_score("literature-bound-item")

    monkeypatch.setattr(bench, "_run_one_item_from_cohort", fake_run_one)
    monkeypatch.setattr(bench, "_aggregate", lambda _scores: {"aware": {}})
    monkeypatch.setattr(bench, "_render_markdown", lambda **_kwargs: "ok")

    assert (
        bench._run_ehrflowbench_jsonl(
            jsonl_path=jsonl,
            out_root=tmp_path / "out",
            seed=7,
            arms=["aware"],
            provider="openai",
            model="model",
        )
        == 0
    )
    bound = seen["pipeline_options"]["bound_preplan_literature"]
    assert bound["research_question"] == question
    assert bound["citations"] == []


def test_external_jsonl_rejects_literature_for_a_different_question(
    tmp_path,
    monkeypatch,
) -> None:
    import tools.run_research_agent_bench as bench

    cohort = tmp_path / "cohort.parquet"
    pd.DataFrame({"stay_id": [1], "outcome": [0]}).to_parquet(cohort, index=False)
    jsonl = tmp_path / "items.jsonl"
    jsonl.write_text(
        json.dumps(
            {
                "key": "mismatched-literature-item",
                "question": "Question A",
                "cohort_path": str(cohort),
                "target_outcome": "outcome",
                "bound_preplan_literature": {
                    "research_question": "Question B",
                    "citations": [],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        bench,
        "_run_one_item_from_cohort",
        lambda **_kwargs: pytest.fail("mismatched literature must not execute"),
    )

    assert (
        bench._run_ehrflowbench_jsonl(
            jsonl_path=jsonl,
            out_root=tmp_path / "out",
            seed=7,
            arms=["aware"],
            provider="openai",
            model="model",
        )
        == bench._PENDING_ITEMS_EXIT_CODE
    )
    payload = json.loads(
        (tmp_path / "out" / "ehrflowbench_results.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload["pending"] == [
        {
            "key": "mismatched-literature-item",
            "status": "bound_preplan_literature_question_mismatch",
        }
    ]


def test_explicit_resume_reopens_existing_hard_stop_ledger(
    tmp_path,
    monkeypatch,
) -> None:
    import tools.run_research_agent_bench as bench
    from easyicu.research_agent.authority.provider_hard_stop import (
        ProviderHardStopLedger,
        load_provider_hard_stop_ledger,
    )

    cohort = tmp_path / "cohort.parquet"
    pd.DataFrame({"stay_id": [1], "death": [0]}).to_parquet(
        cohort,
        index=False,
    )
    jsonl = tmp_path / "items.jsonl"
    jsonl.write_text(
        json.dumps(
            {
                "key": "resume-item",
                "question": "Resume the declared analysis.",
                "cohort_path": str(cohort),
                "target_outcome": "death",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    out_root = tmp_path / "out"
    out_root.mkdir()
    options = bench._benchmark_pipeline_options(
        max_total_steps=None,
        disable_replanning=False,
        max_code_repair_attempts=None,
    )
    limits = bench._provider_hard_stop_limits(options)
    assert limits is not None
    ledger = ProviderHardStopLedger(
        path=(out_root / "ehrflowbench_progress.json").resolve(),
        task_ids=("resume-item",),
        limits=limits,
    )
    ledger.start_task("resume-item").finish(error="RuntimeError: interrupted")

    seen = {}

    def fake_run_one(**kwargs):
        seen.update(kwargs)
        return _completed_score("resume-item")

    monkeypatch.setattr(bench, "_run_one_item_from_cohort", fake_run_one)
    monkeypatch.setattr(bench, "_aggregate", lambda _scores: {"aware": {}})
    monkeypatch.setattr(bench, "_render_markdown", lambda **_kwargs: "ok")

    assert (
        bench._run_ehrflowbench_jsonl(
            jsonl_path=jsonl,
            out_root=out_root,
            seed=7,
            arms=["aware"],
            provider="openai",
            model="model",
            pipeline_options=options,
            resume_run_id="run_existing",
            resume_from_step_id="02_table_one",
        )
        == 0
    )
    assert seen["resume_run_id"] == "run_existing"
    assert seen["resume_from_step_id"] == "02_table_one"
    resumed = load_provider_hard_stop_ledger(
        out_root / "ehrflowbench_progress.json"
    )
    task = resumed["tasks"][0]
    assert task["status"] == "completed"
    assert task["resume_count"] == 1
    assert task["terminal_attempts"][0]["status"] == "failed"


def test_benchmark_rejects_impossible_provider_token_ceiling_before_run() -> None:
    import tools.run_research_agent_bench as bench

    options = bench._benchmark_pipeline_options(
        max_total_steps=None,
        disable_replanning=False,
        max_code_repair_attempts=None,
        max_total_tokens_per_run=120_000,
        max_total_tokens_per_batch=120_000,
    )
    with pytest.raises(
        ValueError,
        match="cannot fund one minimum Provider transport reservation",
    ):
        bench._provider_hard_stop_limits(options)
