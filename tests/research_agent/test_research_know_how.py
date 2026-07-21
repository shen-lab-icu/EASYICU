"""Contracts for the opt-in research know-how retrieval layer."""

from __future__ import annotations

import hashlib
import json
from importlib import resources
from pathlib import Path

import pytest
import pandas as pd
from pydantic import ValidationError

from easyicu.research_agent.agents.core import PlannerAgent, ReplannerAgent
from easyicu.research_agent.authority.evidence_store import EvidenceStore
from easyicu.research_agent.know_how import (
    KnowHowCard,
    KnowHowIntegrityError,
    KnowHowRegistry,
)
from easyicu.research_agent.know_how.registry import MAX_CARD_BYTES
from easyicu.research_agent.planning.preplan_know_how import (
    PlannerKnowHowBinding,
    PreplanKnowHow,
    prepare_preplan_know_how,
)
from easyicu.research_agent.pipeline import _load_compatible_resume_plan
from easyicu.research_agent.providers.structured_retry import (
    StructuredResponseFailure,
)
from easyicu.research_agent.schema import (
    AnalysisPlan,
    AnalysisStep,
    CohortDescriptor,
    ConceptDescriptor,
    ResearchContext,
)


def _context(
    question: str = "Predict acute kidney injury in ICU patients",
) -> ResearchContext:
    return ResearchContext(
        research_question=question,
        cohort=CohortDescriptor(
            cohort_name="synthetic",
            database="miiv",
            n_patients=10,
            n_stays=10,
        ),
        variables=[
            ConceptDescriptor(
                name="creatinine",
                dtype="float64",
                source_concept="creatinine",
            ),
            ConceptDescriptor(
                name="urine_output",
                dtype="float64",
                source_concept="urine_output",
            ),
        ],
    )


def _card_payload() -> dict[str, object]:
    registry = KnowHowRegistry.load()
    return registry.get("aki_onset_prediction").model_dump(mode="json")


def _plan_with_one_adopted_aki_claim() -> AnalysisPlan:
    registry = KnowHowRegistry.load()
    hit = registry.retrieve(
        query="acute kidney injury prediction",
        study_family="prediction",
        top_k=1,
    )[0]
    claim = next(
        item
        for item in registry.get(hit.card_id).claims
        if item.claim_id == "prediction_landmark"
    )
    return AnalysisPlan(
        research_question="Predict AKI",
        steps=[],
        know_how_decisions=[
            {
                "card_id": hit.card_id,
                "card_version": hit.version,
                "card_sha256": hit.file_sha256,
                "claim_id": claim.claim_id,
                "disposition": "adopted",
                "reason_code": "fits_prediction_landmark",
                "rationale": "The question requires a pre-outcome prediction landmark.",
                "citation_ids": claim.citation_ids,
            }
        ],
    )


def test_builtin_registry_contains_nine_curated_cards() -> None:
    from easyicu import research_agent as public_api

    registry = KnowHowRegistry.load()

    assert public_api.KnowHowRegistry is KnowHowRegistry
    assert len(registry.cards) == 9
    assert {card.review_status for card in registry.cards} == {"curated_mvp"}
    assert {card.trust_level for card in registry.cards} == {"built_in_reviewed"}
    assert all(len(card.claims) == 21 for card in registry.cards)
    assert all(len(card.citations) >= 2 for card in registry.cards)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda item: item.update(version="v1"), "version"),
        (lambda item: item.update(unknown_field=True), "extra"),
        (lambda item: item.update(citations=item["citations"][:1]), "citations"),
        (
            lambda item: item["citations"][0].update(url="not-a-url"),
            "url",
        ),
    ],
)
def test_card_schema_rejects_invalid_contracts(mutation, match: str) -> None:
    payload = _card_payload()
    mutation(payload)

    with pytest.raises(ValidationError, match=match):
        KnowHowCard.model_validate(payload)


def test_registry_rejects_duplicate_ids_and_oversized_cards(tmp_path: Path) -> None:
    payload = _card_payload()
    payload["trust_level"] = "user_supplied_unreviewed"
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text(json.dumps(payload), encoding="utf-8")
    second.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(KnowHowIntegrityError, match="duplicate"):
        KnowHowRegistry.load([tmp_path], include_builtin=False)

    first.unlink()
    second.write_bytes(b"{" + b" " * MAX_CARD_BYTES + b"}")
    with pytest.raises(KnowHowIntegrityError, match="exceeds"):
        KnowHowRegistry.load([second], include_builtin=False)


def test_retrieval_is_deterministic_bounded_and_relevance_gated() -> None:
    registry = KnowHowRegistry.load()
    kwargs = dict(
        query="acute kidney injury creatinine urine output prediction",
        study_family="prediction",
        database="miiv",
        available_concepts=["creatinine", "urine_output"],
        top_k=3,
        min_score=0.15,
    )

    first = registry.retrieve(**kwargs)
    second = registry.retrieve(**kwargs)

    assert first == second
    assert first[0].card_id == "aki_onset_prediction"
    assert first[0].topic_applicable is True
    assert first[0].data_readiness == "partial"
    assert "death" in first[0].unresolved_concepts
    assert len(first) <= 3
    assert (
        registry.retrieve(
            query="quantum lattice entanglement",
            study_family="prediction",
            database="miiv",
        )
        == []
    )
    with pytest.raises(ValueError, match="between 0 and 5"):
        registry.retrieve(query="AKI", top_k=6)


def test_prompt_projection_obeys_per_card_and_total_budgets() -> None:
    registry = KnowHowRegistry.load()
    hits = registry.retrieve(
        query="ICU mortality prediction external validation cross database",
        study_family="prediction",
        database="miiv",
        available_concepts=["death", "age", "sex"],
        top_k=5,
        min_score=0.01,
    )

    prompt = registry.render_prompt(hits)

    assert len(prompt) <= 8_000
    projection = json.loads(prompt[prompt.index('{"cards"') :])
    assert len(projection["cards"]) <= 5
    assert "unresolved_concepts" in prompt
    assert "stop_condition" in prompt
    assert "requires_confirmation" in prompt
    assert "citations" in prompt
    assert "truncated" not in prompt
    for card in projection["cards"]:
        assert (
            len(
                json.dumps(
                    card, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                )
            )
            <= 3_500
        )


def test_zero_hit_binding_adds_no_planner_context_or_false_authority() -> None:
    registry = KnowHowRegistry.load()
    prepared = PreplanKnowHow(
        registry=registry,
        hits=(),
        prompt=registry.render_prompt([]),
    )

    binding = PlannerKnowHowBinding.from_prepared(prepared)

    assert binding.enabled is True
    assert binding.prompt == ""
    assert binding.planner_kwargs == {
        "allowed_know_how_decisions": {},
        "know_how_context": "",
    }


def test_preplan_artifacts_are_registered_resume_safe_and_tamper_evident(
    tmp_path: Path,
) -> None:
    evidence = EvidenceStore(tmp_path)
    prepared = prepare_preplan_know_how(
        context=_context(),
        run_dir=tmp_path,
        evidence=evidence,
        database="miiv",
    )

    assert prepared.selected_ids[0] == "aki_onset_prediction"
    receipt_path = tmp_path / "know_how_retrieval.json"
    prompt_path = tmp_path / "know_how_prompt.md"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["selected"][0]["version"] == "1.0.0"
    assert len(receipt["selected"][0]["file_sha256"]) == 64
    assert receipt["selected"][0]["match_reasons"]
    assert (
        evidence.get("know_how_retrieval").sha256
        == hashlib.sha256(receipt_path.read_bytes()).hexdigest()
    )
    assert (
        evidence.get("know_how_prompt").sha256
        == hashlib.sha256(prompt_path.read_bytes()).hexdigest()
    )

    prepare_preplan_know_how(
        context=_context(),
        run_dir=tmp_path,
        evidence=evidence,
        database="miiv",
    )
    prompt_path.write_text("tampered", encoding="utf-8")
    with pytest.raises(KnowHowIntegrityError, match="does not match"):
        prepare_preplan_know_how(
            context=_context(),
            run_dir=tmp_path,
            evidence=evidence,
            database="miiv",
        )


def test_registry_detects_card_source_tampering(tmp_path: Path) -> None:
    payload = _card_payload()
    payload["trust_level"] = "user_supplied_unreviewed"
    card_path = tmp_path / "card.json"
    card_path.write_text(json.dumps(payload), encoding="utf-8")
    registry = KnowHowRegistry.load([card_path], include_builtin=False)
    hit = registry.retrieve(
        query="acute kidney injury prediction",
        study_family="prediction",
        top_k=1,
        min_score=0.01,
        allowed_trust_levels=["user_supplied_unreviewed"],
    )[0]
    card_path.write_text(
        json.dumps({**payload, "summary": "changed"}), encoding="utf-8"
    )

    with pytest.raises(KnowHowIntegrityError, match="digest changed"):
        registry.verify_hit_source(hit)


def test_planner_accepts_only_this_retrievals_claim_authority() -> None:
    planner = PlannerAgent.__new__(PlannerAgent)
    planner.last_dropped_plan_keys = {"top_level": [], "steps": []}
    hit = KnowHowRegistry.load().retrieve(
        query="acute kidney injury prediction",
        study_family="prediction",
        top_k=1,
    )[0]
    card = KnowHowRegistry.load().get(hit.card_id)
    claim = next(item for item in card.claims if item.claim_id == "prediction_landmark")
    authority = {
        hit.card_id: {
            "version": hit.version,
            "file_sha256": hit.file_sha256,
            "claims": {claim.claim_id: tuple(claim.citation_ids)},
        }
    }
    decision = {
        "card_id": hit.card_id,
        "card_version": hit.version,
        "card_sha256": hit.file_sha256,
        "claim_id": claim.claim_id,
        "disposition": "adopted",
        "reason_code": "fits_prediction_landmark",
        "rationale": "The question requires a pre-outcome prediction landmark.",
        "citation_ids": claim.citation_ids,
    }
    base = {"research_question": "Predict AKI", "steps": []}
    raw = json.dumps(
        {
            **base,
            "know_how_decisions": [decision],
        }
    )

    plan = planner._parse(
        raw,
        _context(),
        allowed_know_how_decisions=authority,
    )
    assert plan.know_how_decisions[0].claim_id == "prediction_landmark"
    with pytest.raises(ValueError, match="unretrieved card"):
        planner._parse(
            raw,
            _context(),
            allowed_know_how_decisions={},
        )


def test_replanner_cannot_remove_claim_decisions() -> None:
    class RemovingLLM:
        name = "removing"

        def complete(self, messages, **kwargs):
            return json.dumps({"research_question": "Predict AKI", "steps": []})

    current = _plan_with_one_adopted_aki_claim()
    with pytest.raises(StructuredResponseFailure, match="claim-level disposition"):
        ReplannerAgent(RemovingLLM()).run(context=_context(), current_plan=current)


def test_empty_decisions_do_not_change_legacy_plan_serialization() -> None:
    plan = AnalysisPlan(research_question="No know-how", steps=[])

    assert "know_how_decisions" not in plan.model_dump()


def test_resume_plan_loader_preserves_claim_decisions(tmp_path: Path) -> None:
    plan = _plan_with_one_adopted_aki_claim().model_copy(
        update={
            "steps": [
                AnalysisStep(step_id="01_prepare", intent="Prepare analysis data.")
            ]
        }
    )
    plan_path = tmp_path / "analysis_plan.json"
    plan_path.write_text(plan.model_dump_json(indent=2), encoding="utf-8")
    EvidenceStore(tmp_path).register_file(
        kind="log",
        description="Planner output.",
        source_path=plan_path,
        evidence_id="analysis_plan",
        producer="planner",
        generation_mode="llm",
    )

    restored, _ = _load_compatible_resume_plan(run_dir=tmp_path, resume_state={})

    assert restored is not None
    assert restored.know_how_decisions == plan.know_how_decisions


def test_builtin_cards_are_package_resources() -> None:
    card_dir = resources.files("easyicu").joinpath("data", "research_know_how")

    assert (
        len([item for item in card_dir.iterdir() if item.name.endswith(".json")]) == 9
    )


def test_opt_in_pipeline_smoke_adopts_card_without_extra_provider_calls(
    tmp_path: Path,
) -> None:
    from easyicu import research_agent as ra

    class CountingPlanner(ra.MockLLMClient):
        def __init__(self, *, adopt: bool) -> None:
            super().__init__()
            self.adopt = adopt
            self.calls = 0
            self.planner_prompts: list[str] = []
            registry = KnowHowRegistry.load()
            self.hit = registry.retrieve(
                query="Build an ICU mortality prediction model.",
                study_family="prediction",
                database="miiv",
                top_k=1,
            )[0]
            self.claim = next(
                item
                for item in registry.get(self.hit.card_id).claims
                if item.claim_id == "prediction_time"
            )

        def complete(self, messages, *, max_tokens=2048, temperature=0.2):
            self.calls += 1
            user_prompt = next(
                (
                    message.content
                    for message in reversed(messages)
                    if message.role == "user"
                ),
                "",
            )
            raw = super().complete(
                messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            if "ICU-AWARE RESEARCH PLAN" in user_prompt:
                self.planner_prompts.append(user_prompt)
                if self.adopt:
                    payload = json.loads(raw)
                    payload["know_how_decisions"] = [
                        {
                            "card_id": self.hit.card_id,
                            "card_version": self.hit.version,
                            "card_sha256": self.hit.file_sha256,
                            "claim_id": self.claim.claim_id,
                            "disposition": "adopted",
                            "reason_code": "fits_prediction_landmark",
                            "rationale": "The prediction task requires an explicit landmark.",
                            "citation_ids": self.claim.citation_ids,
                        }
                    ]
                    return json.dumps(payload)
            return raw

    cohort = pd.DataFrame(
        {
            "stay_id": range(1, 41),
            "age": [50 + index % 20 for index in range(40)],
            "sex": ["M", "F"] * 20,
            "death": [1 if index % 7 == 0 else 0 for index in range(40)],
        }
    )
    common = dict(
        enable_literature=False,
        enable_visual_qa=False,
        enable_publication_figure_skill=False,
        enable_reviewer_round=False,
        enable_fairness_subgroups=False,
        enable_reporting_checklist=False,
        enable_replanning=False,
        runner_kind="subprocess",
    )
    enabled_planner = CountingPlanner(adopt=True)
    enabled = ra.ResearchAgentPipeline(
        workdir=tmp_path / "enabled",
        llm=ra.LLMRouter(default=ra.MockLLMClient(), planner=enabled_planner),
        enable_know_how=True,
        **common,
    ).run(
        question="Build an ICU mortality prediction model.",
        cohort=cohort,
        cohort_name="know_how_smoke",
        database="miiv",
        target_outcome="death",
        stop_after_analysis=True,
    )
    disabled_planner = CountingPlanner(adopt=False)
    disabled = ra.ResearchAgentPipeline(
        workdir=tmp_path / "disabled",
        llm=ra.LLMRouter(default=ra.MockLLMClient(), planner=disabled_planner),
        enable_know_how=False,
        **common,
    ).run(
        question="Build an ICU mortality prediction model.",
        cohort=cohort,
        cohort_name="know_how_smoke",
        database="miiv",
        target_outcome="death",
        stop_after_analysis=True,
    )

    enabled_dir = Path(enabled.plan_path).parent
    disabled_dir = Path(disabled.plan_path).parent
    plan = json.loads(Path(enabled.plan_path).read_text(encoding="utf-8"))
    assert "know_how_refs" not in plan
    assert plan["know_how_decisions"][0]["card_id"] == "icu_mortality_prediction"
    assert plan["know_how_decisions"][0]["claim_id"] == "prediction_time"
    assert "RESEARCH KNOW-HOW CONTEXT" in enabled_planner.planner_prompts[0]
    assert "Retrieved Research Know-How" in enabled_planner.planner_prompts[0]
    assert (enabled_dir / "know_how_retrieval.json").exists()
    assert (enabled_dir / "know_how_prompt.md").exists()
    metrics = json.loads(
        (enabled_dir / "planner_prompt_metrics.json").read_text(encoding="utf-8")
    )
    assert 0 < metrics["know_how_added_bytes"] <= 8_000
    assert metrics["total_bytes"] <= metrics["limit_bytes"]
    assert metrics["know_how_selected_count"] == 1
    assert not (disabled_dir / "know_how_retrieval.json").exists()
    assert not (disabled_dir / "know_how_prompt.md").exists()
    assert disabled_planner.calls == enabled_planner.calls
