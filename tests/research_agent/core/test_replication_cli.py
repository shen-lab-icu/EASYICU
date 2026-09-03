"""CLI tests for easyicu-research-replication."""

from __future__ import annotations

from pathlib import Path

import pytest


def test_replication_cli_paper_mode_dispatches_to_pipeline(
    monkeypatch, tmp_path: Path
):
    from easyicu.research_agent import replication_cli
    import easyicu.research_agent.providers.llm as llm_mod
    import easyicu.research_agent.providers.mocks as mocks_mod
    import easyicu.research_agent.pipeline as pipeline_mod

    calls = {}

    class FakeLLM:
        def __init__(self, *args, **kwargs):
            calls["llm_init"] = {"args": args, "kwargs": kwargs}

    class FakePipeline:
        def __init__(self, **kwargs):
            calls["pipeline_init"] = kwargs

        def reproduce_paper(self, **kwargs):
            calls["reproduce_paper"] = kwargs

            class Result:
                run_id = "run_test"
                workdir = str(tmp_path / "out")
                manifest_path = str(tmp_path / "out" / "manifest.json")
                report_path = str(tmp_path / "out" / "results_report.md")
                replication_report_path = str(tmp_path / "out" / "replication_report.md")
                manuscript_path = str(tmp_path / "out" / "manuscript_scaffold_bound.md")

            return Result()

    monkeypatch.setattr(mocks_mod, "MockLLMClient", FakeLLM)
    monkeypatch.setattr(llm_mod, "OpenAIClient", FakeLLM)
    monkeypatch.setattr(pipeline_mod, "ResearchAgentPipeline", FakePipeline)

    rc = replication_cli.main(
        [
            "--paper",
            "Title: SOFA paper",
            "--cohort",
            str(tmp_path / "cohort.parquet"),
            "--database",
            "synthetic",
            "--mode",
            "replication",
            "--llm",
            "mock",
            "--output",
            str(tmp_path / "out"),
        ]
    )
    assert rc == 0
    assert calls["pipeline_init"]["workdir"] == str(tmp_path / "out")
    assert calls["reproduce_paper"]["database"] == "synthetic"
    assert calls["reproduce_paper"]["mode"] == "replication"


def test_replication_cli_blocks_external_provider_before_construction(
    monkeypatch, tmp_path: Path
):
    from easyicu.research_agent import replication_cli
    import easyicu.research_agent.providers.factory as factory_module

    constructed = []

    def forbidden_builder(**_kwargs):
        constructed.append(True)
        raise AssertionError("provider construction must be unreachable")

    monkeypatch.setattr(factory_module, "build_provider_client", forbidden_builder)

    with pytest.raises(SystemExit, match="AI features are disabled"):
        replication_cli.main(
            [
                "--paper",
                "Title: cohort report",
                "--cohort",
                str(tmp_path / "cohort.parquet"),
                "--database",
                "synthetic",
                "--llm",
                "openai",
                "--output",
                str(tmp_path / "out"),
            ]
        )

    assert constructed == []
