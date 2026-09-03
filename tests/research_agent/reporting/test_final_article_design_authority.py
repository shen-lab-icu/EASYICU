"""Focused tests for Planner-final article-design materialization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from easyicu.research_agent.planning import final_article_design as owner


@dataclass(frozen=True)
class _Payload:
    name: str
    analysis_family: str = "association"

    def model_dump_json(self, *, indent: int) -> str:
        assert indent == 2
        return f'{{"name": "{self.name}"}}'


class _Evidence:
    def __init__(self) -> None:
        self.registered: list[dict[str, object]] = []

    def get(self, _evidence_id: str):
        return None

    def register_file(self, **kwargs) -> None:
        self.registered.append(kwargs)


def test_final_article_design_owner_persists_one_consistent_bundle(
    tmp_path: Path,
    monkeypatch,
) -> None:
    brief = _Payload("brief")
    contract = _Payload("contract")
    strategy = _Payload("strategy")
    blueprint = _Payload("blueprint")
    monkeypatch.setattr(owner, "build_study_design_brief", lambda *_a, **_k: brief)
    monkeypatch.setattr(
        owner, "build_article_analysis_contract", lambda *_a, **_k: contract
    )
    monkeypatch.setattr(
        owner, "build_article_figure_strategy", lambda *_a, **_k: strategy
    )
    monkeypatch.setattr(owner, "build_analysis_blueprint", lambda *_a, **_k: blueprint)
    evidence = _Evidence()

    result = owner.materialize_final_article_design_authority(
        context=object(),
        analysis_type="association_study",
        run_dir=tmp_path,
        evidence=evidence,
    )

    assert result == owner.FinalArticleDesignAuthority(
        brief=brief,
        contract=contract,
        figure_strategy=strategy,
        blueprint=blueprint,
    )
    assert {path.name for path in tmp_path.iterdir()} == {
        "study_design_brief.final.json",
        "article_analysis_contract.final.json",
        "article_figure_strategy.final.json",
        "analysis_blueprint.final.json",
    }
    assert len(evidence.registered) == 4
    assert {row["producer"] for row in evidence.registered} == {
        "planner_contract_finalizer"
    }
