from __future__ import annotations

from types import SimpleNamespace

import pytest


def test_parse_cohort_map_accepts_multiple_pairs(ra):
    from easyicu.research_agent.cli import _parse_cohort_map

    mapping = _parse_cohort_map(["miiv=/tmp/a.parquet", "eicu=/tmp/b.parquet"])
    assert mapping["miiv"].endswith("/tmp/a.parquet")
    assert mapping["eicu"].endswith("/tmp/b.parquet")


def test_parse_cohort_map_rejects_invalid_pair(ra):
    from easyicu.research_agent.cli import _parse_cohort_map

    try:
        _parse_cohort_map(["miiv"])
    except SystemExit as exc:
        assert "--cohort-map must be DB=PATH" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected SystemExit for invalid cohort-map input")


def test_public_cli_reaches_the_research_agent_pipeline(monkeypatch, tmp_path):
    """The console capability is real only if it reaches the live pipeline API."""

    from easyicu.research_agent import cli
    import easyicu.research_agent.pipeline as pipeline_module

    calls = {}

    class FakePipeline:
        def __init__(self, **kwargs):
            calls["init"] = kwargs

        def run(self, **kwargs):
            calls["run"] = kwargs
            return SimpleNamespace(
                run_id="run-cli-reachability",
                workdir=str(tmp_path),
                context_path=str(tmp_path / "context.json"),
                plan_path=str(tmp_path / "plan.json"),
                manifest_path=str(tmp_path / "manifest.json"),
                report_path=str(tmp_path / "report.md"),
                manuscript_path=str(tmp_path / "manuscript.md"),
                evidence_count=1,
                findings_count=0,
            )

    monkeypatch.setattr(pipeline_module, "ResearchAgentPipeline", FakePipeline)

    rc = cli.main(
        [
            "--llm",
            "mock",
            "--question",
            "Is the declared exposure associated with the outcome?",
            "--cohort",
            str(tmp_path / "cohort.parquet"),
            "--workdir",
            str(tmp_path),
        ]
    )

    assert rc == 0
    assert calls["run"]["question"].startswith("Is the declared exposure")
    assert calls["run"]["cohort"] == str(tmp_path / "cohort.parquet")


def test_external_cli_is_blocked_before_provider_construction_without_opt_in(
    monkeypatch, tmp_path
):
    from easyicu.research_agent import cli
    import easyicu.research_agent.providers.factory as factory_module

    constructed = []

    def forbidden_builder(**_kwargs):
        constructed.append(True)
        raise AssertionError("provider construction must be unreachable")

    monkeypatch.setattr(factory_module, "build_provider_client", forbidden_builder)

    with pytest.raises(SystemExit, match="AI features are disabled"):
        cli.main(
            [
                "--llm",
                "openai",
                "--question",
                "Describe this cohort.",
                "--cohort",
                str(tmp_path / "cohort.parquet"),
            ]
        )

    assert constructed == []


def test_external_cli_calls_canonical_opt_in_before_provider_construction(
    monkeypatch, tmp_path
):
    from easyicu import ai_optin
    from easyicu.research_agent import cli
    import easyicu.research_agent.pipeline as pipeline_module
    import easyicu.research_agent.providers.factory as factory_module

    events = []
    llm = object()

    def check_opt_in(choice, *, ai_enabled, language):
        events.append(("gate", choice, ai_enabled, language))

    def build_provider(**_kwargs):
        events.append(("provider",))
        return llm

    class FakePipeline:
        def __init__(self, **kwargs):
            events.append(("pipeline", kwargs["llm"] is llm))

        def run(self, **_kwargs):
            return SimpleNamespace(
                run_id="run-cli-opt-in",
                workdir=str(tmp_path),
                context_path=str(tmp_path / "context.json"),
                plan_path=str(tmp_path / "plan.json"),
                manifest_path=str(tmp_path / "manifest.json"),
                report_path=str(tmp_path / "report.md"),
                manuscript_path=str(tmp_path / "manuscript.md"),
                evidence_count=0,
                findings_count=0,
            )

    monkeypatch.setattr(ai_optin, "check_external_llm_opt_in", check_opt_in)
    monkeypatch.setattr(factory_module, "build_provider_client", build_provider)
    monkeypatch.setattr(pipeline_module, "ResearchAgentPipeline", FakePipeline)

    rc = cli.main(
        [
            "--llm",
            "openai",
            "--external-llm-opt-in",
            "--question",
            "Describe this cohort.",
            "--cohort",
            str(tmp_path / "cohort.parquet"),
            "--workdir",
            str(tmp_path),
        ]
    )

    assert rc == 0
    assert events == [
        ("gate", "openai", True, "en"),
        ("provider",),
        ("pipeline", True),
    ]
