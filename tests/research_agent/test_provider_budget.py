from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from easyicu.research_agent.audits.validators import LLMConceptAuditor
from easyicu.research_agent.provider_budget import (
    PROVIDER_CALL_BUDGET_RECEIPT_SCHEMA_VERSION,
    ProviderCallBudgetExhausted,
    ProviderCallBudgetReceiptError,
    StepProviderCallBudget,
    complete_with_provider_budget,
    consume_active_transport_attempt,
    load_provider_call_budget_receipt,
    provider_call_budget_receipt_path,
)
from easyicu.research_agent.pipeline_execute import (
    _HOST_COHORT_TRANSLATION_BUDGET_STEP_ID,
    _cohort_translation_budget_owner_step_id,
    _extract_cohort_definition_with_provider_budget,
)
from easyicu.research_agent.pipeline_config import PipelineConfig
from easyicu.research_agent.schema import (
    AnalysisStep,
    CohortDescriptor,
    ResearchContext,
)


class _AuditLLM:
    name = "audit-budget-test"

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, messages, **kwargs):  # noqa: ANN001, ANN003
        self.calls += 1
        return '{"findings":[]}'


def _context() -> ResearchContext:
    return ResearchContext(
        research_question="Audit the planner-owned analysis.",
        cohort=CohortDescriptor(
            cohort_name="demo",
            database="synthetic",
            n_stays=3,
            n_patients=3,
        ),
        variables=[],
    )


def test_pre_step_cohort_translation_has_durable_shared_provider_budget(tmp_path):
    class _CohortTranslationLLM:
        name = "cohort-translation-budget-test"

        def __init__(self) -> None:
            self.calls = 0

        def complete(self, messages, **kwargs):  # noqa: ANN001, ANN003
            del messages, kwargs
            self.calls += 1
            return json.dumps(
                {
                    "inclusion": [{"concept_id": "age", "op": ">=", "value": 18}],
                    "exclusion": [],
                }
            )

    owner_step_id = "01_cohort_definition"
    llm = _CohortTranslationLLM()
    first, first_snapshot = _extract_cohort_definition_with_provider_budget(
        run_dir=tmp_path,
        budget_owner_step_id=owner_step_id,
        configured_limit=2,
        cohort_prose="Include adults age 18 years or older.",
        universe_columns=["stay_id", "age"],
        llm=llm,
        name="adult_icu",
    )
    second, second_snapshot = _extract_cohort_definition_with_provider_budget(
        run_dir=tmp_path,
        budget_owner_step_id=owner_step_id,
        configured_limit=2,
        cohort_prose="Include adults age 18 years or older.",
        universe_columns=["stay_id", "age"],
        llm=llm,
        name="adult_icu",
    )

    assert first is not None
    assert second is not None
    assert first_snapshot["step_provider_call_attempts"] == 1
    assert second_snapshot["step_provider_call_attempts"] == 2
    assert second_snapshot["step_provider_call_categories"] == [
        "cohort_definition_translation",
        "cohort_definition_translation",
    ]
    assert (
        second_snapshot["step_provider_call_receipt_version"]
        == PROVIDER_CALL_BUDGET_RECEIPT_SCHEMA_VERSION
    )
    receipt_path = provider_call_budget_receipt_path(
        tmp_path,
        step_id=owner_step_id,
    )
    assert second_snapshot["step_provider_call_receipt"] == str(
        receipt_path.relative_to(tmp_path)
    )
    stored_limit, stored_categories = load_provider_call_budget_receipt(
        receipt_path,
        step_id=owner_step_id,
    )
    assert stored_limit == 2
    assert json.loads(receipt_path.read_text(encoding="utf-8"))["schema_version"] == (
        PROVIDER_CALL_BUDGET_RECEIPT_SCHEMA_VERSION
    )
    assert stored_categories == (
        "cohort_definition_translation",
        "cohort_definition_translation",
    )

    with pytest.raises(ProviderCallBudgetExhausted):
        _extract_cohort_definition_with_provider_budget(
            run_dir=tmp_path,
            budget_owner_step_id=owner_step_id,
            configured_limit=2,
            cohort_prose="Include adults age 18 years or older.",
            universe_columns=["stay_id", "age"],
            llm=llm,
            name="adult_icu",
        )
    assert llm.calls == 2

    receipt_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ProviderCallBudgetReceiptError):
        _extract_cohort_definition_with_provider_budget(
            run_dir=tmp_path,
            budget_owner_step_id=owner_step_id,
            configured_limit=2,
            cohort_prose="Include adults age 18 years or older.",
            universe_columns=["stay_id", "age"],
            llm=llm,
            name="adult_icu",
        )
    assert llm.calls == 2


def test_cohort_translation_budget_owner_is_structural_not_prose_routed():
    cohort_only = SimpleNamespace(
        step_id="01_cohort",
        expected_outputs=["table:analysis_cohort"],
        intent="Any prose is irrelevant to budget ownership.",
    )
    model = SimpleNamespace(
        step_id="02_model",
        expected_outputs=["table:effect_estimates"],
        intent="cohort cohort cohort",
    )
    assert (
        _cohort_translation_budget_owner_step_id(
            SimpleNamespace(steps=[cohort_only, model])
        )
        == "01_cohort"
    )

    mixed = SimpleNamespace(
        step_id="01_mixed",
        expected_outputs=["table:analysis_cohort", "table:cohort_flow"],
        intent="Define the cohort.",
    )
    assert (
        _cohort_translation_budget_owner_step_id(SimpleNamespace(steps=[mixed]))
        == _HOST_COHORT_TRANSLATION_BUDGET_STEP_ID
    )


def test_step_provider_call_budget_reservation_is_atomic():
    budget = StepProviderCallBudget(17, step_id="atomic")

    def consume_once(index: int) -> bool:
        try:
            budget.consume(f"call_{index % 3}")
        except ProviderCallBudgetExhausted:
            return False
        return True

    with ThreadPoolExecutor(max_workers=8) as pool:
        accepted = list(pool.map(consume_once, range(64)))

    assert sum(accepted) == 17
    assert budget.used == 17
    assert budget.remaining == 0
    assert budget.exhausted is True
    assert len(budget.categories) == 17


def test_final_concept_audit_slot_cannot_be_spent_by_generation_or_repair():
    budget = StepProviderCallBudget(
        3,
        step_id="01_model",
        reserved_final_category="concept_audit",
    )

    budget.consume("initial_generation")
    budget.consume("contract_repair_patch")
    with pytest.raises(ProviderCallBudgetExhausted) as exc_info:
        budget.consume("contract_repair_full_rewrite")

    assert exc_info.value.reserved_for == "concept_audit"
    assert budget.remaining == 1
    assert budget.categories == (
        "initial_generation",
        "contract_repair_patch",
    )
    budget.consume("concept_audit")
    assert budget.remaining == 0


def test_default_budget_fits_two_semantic_repairs_final_audit_and_analyzer(
    tmp_path,
):
    limit = PipelineConfig(workdir=tmp_path).max_step_provider_calls
    assert limit == 7
    budget = StepProviderCallBudget(
        limit,
        step_id="01_model",
        reserved_final_category="concept_audit",
    )

    for category in (
        "initial_generation",
        "concept_audit",
        "post_mutation_concept_repair_patch",
        "concept_audit",
        "post_mutation_concept_repair_patch",
    ):
        budget.consume(category)
    budget.bind_reserved_category("concept_audit", token="final-digest-authority")
    budget.consume("concept_audit")
    budget.complete_reserved_category(
        "concept_audit",
        token="final-digest-authority",
    )
    budget.release_reserved_category(
        "concept_audit",
        token="final-digest-authority",
    )
    budget.consume("analyzer")

    assert budget.used == 7
    assert budget.categories == (
        "initial_generation",
        "concept_audit",
        "post_mutation_concept_repair_patch",
        "concept_audit",
        "post_mutation_concept_repair_patch",
        "concept_audit",
        "analyzer",
    )


def test_exact_final_audit_token_releases_reserved_slot_for_analyzer():
    budget = StepProviderCallBudget(
        2,
        step_id="01_model",
        reserved_final_category="concept_audit",
    )
    budget.consume("initial_generation")
    budget.bind_reserved_category("concept_audit", token="audit-A-authority-1")
    budget.complete_reserved_category("concept_audit", token="audit-A-authority-1")

    assert budget.can_consume("analyzer") is False
    with pytest.raises(ValueError, match="completed"):
        budget.release_reserved_category(
            "concept_audit",
            token="audit-B-authority-1",
        )
    budget.release_reserved_category(
        "concept_audit",
        token="audit-A-authority-1",
    )
    assert budget.can_consume("analyzer") is True
    budget.consume("analyzer")
    assert budget.categories == ("initial_generation", "analyzer")


def test_repair_from_cached_code_a_to_code_b_rearms_final_audit_slot():
    budget = StepProviderCallBudget(
        3,
        step_id="01_model",
        reserved_final_category="concept_audit",
    )
    budget.consume("initial_generation")
    budget.bind_reserved_category("concept_audit", token="audit-A-authority-1")
    budget.complete_reserved_category("concept_audit", token="audit-A-authority-1")

    # A cache hit is not the final boundary: a repair may still create B.
    budget.consume("post_mutation_concept_repair")
    budget.bind_reserved_category("concept_audit", token="audit-B-authority-1")

    assert budget.can_consume("analyzer") is False
    with pytest.raises(ProviderCallBudgetExhausted) as exc_info:
        budget.consume("analyzer")
    assert exc_info.value.reserved_for == "concept_audit"
    budget.consume("concept_audit")
    assert budget.categories[-1] == "concept_audit"


def test_resume_history_with_old_audit_does_not_release_slot_for_new_code(tmp_path):
    path = provider_call_budget_receipt_path(tmp_path, step_id="01_model")
    first = StepProviderCallBudget(
        3,
        step_id="01_model",
        receipt_path=path,
        reserved_final_category="concept_audit",
    )
    first.consume("initial_generation")
    first.bind_reserved_category("concept_audit", token="audit-A-authority-1")
    first.consume("concept_audit")

    stored_limit, stored_categories = load_provider_call_budget_receipt(
        path,
        step_id="01_model",
        expected_reserved_final_category="concept_audit",
    )
    resumed = StepProviderCallBudget(
        stored_limit,
        step_id="01_model",
        consumed_categories=stored_categories,
        receipt_path=path,
        reserved_final_category="concept_audit",
    )
    resumed.bind_reserved_category("concept_audit", token="audit-B-authority-1")

    assert resumed.can_consume("analyzer") is False
    assert resumed.can_consume("concept_audit") is True


def test_receipt_rejects_final_audit_policy_drift(tmp_path):
    path = provider_call_budget_receipt_path(tmp_path, step_id="01_model")
    budget = StepProviderCallBudget(
        2,
        step_id="01_model",
        receipt_path=path,
        reserved_final_category="concept_audit",
    )
    budget.consume("initial_generation")

    with pytest.raises(ProviderCallBudgetReceiptError, match="policy changed"):
        load_provider_call_budget_receipt(
            path,
            step_id="01_model",
            expected_reserved_final_category=None,
        )


def test_llm_concept_auditor_charges_shared_budget_and_fails_closed_when_empty():
    llm = _AuditLLM()
    auditor = LLMConceptAuditor(llm)
    budget = StepProviderCallBudget(1, step_id="audit")

    findings = auditor.audit(
        context=_context(),
        script_text="print('checked')",
        provider_budget=budget,
    )

    assert findings == []
    assert llm.calls == 1
    assert budget.categories == ("concept_audit",)

    with pytest.raises(ProviderCallBudgetExhausted) as exc_info:
        auditor.audit(
            context=_context(),
            script_text="print('unchanged')",
            provider_budget=budget,
        )

    assert exc_info.value.category == "concept_audit"
    assert llm.calls == 1


def test_analyzer_charges_the_same_step_budget_and_stops_when_exhausted():
    from easyicu.research_agent.agents import AnalyzerAgent

    llm = _AuditLLM()
    analyzer = AnalyzerAgent(llm)
    budget = StepProviderCallBudget(1, step_id="analysis")
    step = AnalysisStep(step_id="03_model", intent="Interpret the fitted model.")

    analyzer.run(
        context=_context(),
        step=step,
        step_summary={"estimate": 1.2},
        evidence_ids=["summary_03"],
        provider_budget=budget,
    )

    assert llm.calls == 1
    assert budget.categories == ("analyzer",)
    with pytest.raises(ProviderCallBudgetExhausted) as exc_info:
        analyzer.run(
            context=_context(),
            step=step,
            step_summary={"estimate": 1.2},
            evidence_ids=["summary_03"],
            provider_budget=budget,
        )
    assert exc_info.value.category == "analyzer"
    assert llm.calls == 1


def test_openai_transport_retries_consume_the_same_provider_budget(monkeypatch):
    from easyicu.research_agent.llm import LLMMessage, OpenAIClient

    class _Completions:
        def __init__(self) -> None:
            self.calls = 0

        def create(self, **kwargs):  # noqa: ANN003
            self.calls += 1
            if self.calls == 1:
                return SimpleNamespace(choices=[], usage=None)
            message = SimpleNamespace(content="ok")
            return SimpleNamespace(
                choices=[SimpleNamespace(message=message, finish_reason="stop")],
                usage=None,
            )

    completions = _Completions()
    client = OpenAIClient.__new__(OpenAIClient)
    client._client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    client._model = "gpt-test"
    client._timeout = 1.0
    client._extra_body = {}
    client._local_noauth_mode = False
    client._max_retries = 2
    monkeypatch.setattr("time.sleep", lambda _seconds: None)
    budget = StepProviderCallBudget(2, step_id="transport")

    result = complete_with_provider_budget(
        budget=budget,
        category="repair_patch",
        call=lambda: client.complete([LLMMessage(role="user", content="return ok")]),
    )

    assert result == "ok"
    assert completions.calls == 2
    assert budget.categories == ("repair_patch", "repair_patch")


def test_openai_transport_retry_stops_before_exceeding_budget(monkeypatch):
    from easyicu.research_agent.llm import LLMMessage, OpenAIClient

    class _Completions:
        def __init__(self) -> None:
            self.calls = 0

        def create(self, **kwargs):  # noqa: ANN003
            self.calls += 1
            return SimpleNamespace(choices=[], usage=None)

    completions = _Completions()
    client = OpenAIClient.__new__(OpenAIClient)
    client._client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    client._model = "gpt-test"
    client._timeout = 1.0
    client._extra_body = {}
    client._local_noauth_mode = False
    client._max_retries = 3
    sleeps = []
    monkeypatch.setattr("time.sleep", sleeps.append)
    budget = StepProviderCallBudget(1, step_id="transport")

    with pytest.raises(ProviderCallBudgetExhausted):
        complete_with_provider_budget(
            budget=budget,
            category="repair_patch",
            call=lambda: client.complete(
                [LLMMessage(role="user", content="return ok")]
            ),
        )

    assert completions.calls == 1
    assert budget.used == 1
    assert sleeps == []


def test_budget_receipt_is_atomic_restorable_and_survives_lower_limit(tmp_path):
    path = provider_call_budget_receipt_path(tmp_path, step_id="07_balance")
    budget = StepProviderCallBudget(
        5,
        step_id="07_balance",
        receipt_path=path,
    )
    for category in ("initial_generation", "concept_audit", "repair_patch", "retry"):
        budget.consume(category)

    stored_limit, categories = load_provider_call_budget_receipt(
        path,
        step_id="07_balance",
    )
    assert stored_limit == 5
    assert categories == budget.categories

    restored = StepProviderCallBudget(
        3,
        step_id="07_balance",
        consumed_categories=categories,
        receipt_path=path,
    )
    assert restored.used == 4
    assert restored.remaining == 0
    assert restored.exhausted is True
    with pytest.raises(ProviderCallBudgetExhausted):
        restored.consume("must_not_run")


def test_receipt_persistence_failure_prevents_provider_call(monkeypatch, tmp_path):
    path = provider_call_budget_receipt_path(tmp_path, step_id="audit")
    budget = StepProviderCallBudget(1, step_id="audit", receipt_path=path)
    calls = []

    def fail_replace(*_args):
        raise OSError("disk full")

    monkeypatch.setattr(
        "easyicu.research_agent.provider_budget.os.replace",
        fail_replace,
    )

    with pytest.raises(ProviderCallBudgetReceiptError, match="persist"):
        complete_with_provider_budget(
            budget=budget,
            category="concept_audit",
            call=lambda: calls.append("called"),
        )

    assert calls == []
    assert budget.used == 0
    assert not path.exists()


def test_concept_auditor_does_not_downgrade_receipt_failure(monkeypatch, tmp_path):
    path = provider_call_budget_receipt_path(tmp_path, step_id="audit")
    budget = StepProviderCallBudget(1, step_id="audit", receipt_path=path)
    llm = _AuditLLM()

    def fail_replace(*_args):
        raise OSError("read-only filesystem")

    monkeypatch.setattr(
        "easyicu.research_agent.provider_budget.os.replace",
        fail_replace,
    )

    with pytest.raises(ProviderCallBudgetReceiptError):
        LLMConceptAuditor(llm).audit(
            context=_context(),
            script_text="print('checked')",
            provider_budget=budget,
        )

    assert llm.calls == 0


def test_fallback_children_each_consume_a_real_provider_attempt():
    from easyicu.research_agent.llm import FallbackLLMClient, LLMMessage

    class _Child:
        def __init__(self, result=None, error=None):
            self.result = result
            self.error = error
            self.calls = 0

        def complete(self, messages, **kwargs):  # noqa: ANN001, ANN003
            self.calls += 1
            if self.error is not None:
                raise self.error
            return self.result

    first = _Child(error=RuntimeError("429 rate limit"))
    second = _Child(result="ok")
    client = FallbackLLMClient(first, second)
    budget = StepProviderCallBudget(2, step_id="fallback")

    result = complete_with_provider_budget(
        budget=budget,
        category="repair_patch",
        call=lambda: client.complete([LLMMessage(role="user", content="fix")]),
    )

    assert result == "ok"
    assert first.calls == 1
    assert second.calls == 1
    assert budget.categories == ("repair_patch", "repair_patch")


def test_fallback_does_not_double_charge_transparent_transport_aware_wrapper():
    from easyicu.research_agent.llm import FallbackLLMClient, LLMMessage

    class _AwareClient:
        provider_attempt_budget_aware = True

        def complete(self, messages, **kwargs):  # noqa: ANN001, ANN003
            consume_active_transport_attempt()
            return "ok"

    class _Wrapper:
        def __init__(self, inner):
            self._inner = inner

        def complete(self, messages, **kwargs):  # noqa: ANN001, ANN003
            return self._inner.complete(messages, **kwargs)

    client = FallbackLLMClient(_Wrapper(_AwareClient()))
    budget = StepProviderCallBudget(1, step_id="wrapped")

    result = complete_with_provider_budget(
        budget=budget,
        category="concept_audit",
        call=lambda: client.complete([LLMMessage(role="user", content="audit")]),
    )

    assert result == "ok"
    assert budget.categories == ("concept_audit",)


def test_pipeline_resume_restores_durable_budget_and_blocks_without_new_call(
    ra,
    tmp_path: Path,
):
    draft_code = """
import json
import os
import pandas as pd

df = pd.read_parquet(os.environ["COHORT_PARQUET"])
out = os.environ["STEP_OUT_DIR"]
summary = {
    "n": int(len(df)),
    "output_files": [
        {"kind": "table", "name": "cohort_summary", "path": "cohort_summary.csv"}
    ],
}
pd.DataFrame([summary]).to_csv(os.path.join(out, "cohort_summary.csv"), index=False)
with open(os.path.join(out, "step_summary.json"), "w", encoding="utf-8") as handle:
    json.dump(summary, handle)
"""

    class _BudgetedPipelineLLM:
        name = "provider-budget-pipeline-test"

        def __init__(self) -> None:
            self.plan_calls = 0
            self.write_calls = 0
            self.audit_calls = 0
            self.repair_calls = 0

        def complete(self, messages, *, max_tokens=2048, temperature=0.2):
            del max_tokens, temperature
            system = "\n".join(
                str(message.content or "")
                for message in messages
                if message.role == "system"
            )
            user = next(
                (
                    str(message.content or "")
                    for message in reversed(messages)
                    if message.role == "user"
                ),
                "",
            )
            upper = user.upper()
            if "ICU-AWARE RESEARCH PLAN" in upper:
                self.plan_calls += 1
                return json.dumps(
                    {
                        "research_question": "Summarize the ICU cohort.",
                        "steps": [
                            {
                                "step_id": "01_summary",
                                "intent": "Produce a descriptive cohort summary.",
                                "inputs": ["stay_id"],
                                "expected_outputs": ["table:cohort_summary"],
                                "method": "descriptive_summary",
                                "icu_rule_refs": [],
                            }
                        ],
                        "rationale": "provider budget resume regression",
                    }
                )
            if "WRITE THE PYTHON CODE" in upper:
                self.write_calls += 1
                return draft_code
            if "CONSERVATIVE ICU CONCEPT-USE AUDITOR" in system.upper():
                self.audit_calls += 1
                return json.dumps(
                    {
                        "findings": [
                            {
                                "severity": "error",
                                "message": "A binding concept error requires repair.",
                                "detail": {
                                    "issue_code": "provider_budget_resume_test",
                                },
                            }
                        ]
                    }
                )
            if "REPAIR" in upper:
                self.repair_calls += 1
                return draft_code.replace('"n": int(len(df))', '"n": int(len(df))')
            return "{}"

    cohort = pd.DataFrame({"stay_id": [1, 2, 3], "death": [0, 1, 0]})
    first_llm = _BudgetedPipelineLLM()
    first_pipeline = ra.ResearchAgentPipeline(
        workdir=tmp_path,
        llm=first_llm,
        enable_literature=False,
        enable_visual_qa=False,
        enable_latex=False,
        enable_llm_concept_audit=True,
        enable_deterministic_code_fallback=False,
        enable_deterministic_runner_repair=False,
        max_step_provider_calls=2,
    )
    first = first_pipeline.run(
        question="Summarize the ICU cohort.",
        cohort=cohort,
        cohort_name="provider_budget_resume_test",
        database="synthetic",
        target_outcome="death",
        stop_after_step_id="01_summary",
        stop_after_analysis=True,
    )
    run_dir = Path(first.workdir)
    receipt_path = provider_call_budget_receipt_path(run_dir, step_id="01_summary")
    stored_limit, stored_categories = load_provider_call_budget_receipt(
        receipt_path,
        step_id="01_summary",
    )

    assert stored_limit == 2
    assert stored_categories == ("initial_generation", "concept_audit")
    assert first_llm.write_calls == 1
    assert first_llm.audit_calls == 1
    assert first_llm.repair_calls == 0

    resumed_llm = _BudgetedPipelineLLM()
    resumed_pipeline = ra.ResearchAgentPipeline(
        workdir=tmp_path,
        llm=resumed_llm,
        enable_literature=False,
        enable_visual_qa=False,
        enable_latex=False,
        enable_llm_concept_audit=True,
        enable_deterministic_code_fallback=False,
        enable_deterministic_runner_repair=False,
        max_step_provider_calls=2,
    )
    resumed_pipeline.run(
        question="Summarize the ICU cohort.",
        cohort=cohort,
        cohort_name="provider_budget_resume_test",
        database="synthetic",
        target_outcome="death",
        resume_run_id=first.run_id,
        resume_from_step_id="01_summary",
        stop_after_step_id="01_summary",
        stop_after_analysis=True,
    )

    partial = json.loads(
        (run_dir / "manifest_partial.json").read_text(encoding="utf-8")
    )
    latest = [
        record
        for record in partial["per_step_records"]
        if record.get("step_id") == "01_summary"
    ][-1]
    assert latest["status"] != "ok"
    assert latest["step_provider_call_budget_exhausted"] is True
    assert latest["step_provider_call_attempts"] == 2
    assert latest["step_provider_call_categories"] == [
        "initial_generation",
        "concept_audit",
    ]
    assert latest.get("step_llm_repair_attempts", 0) == 0
    assert latest["step_provider_call_repair_unavailable"] is True
    assert resumed_llm.write_calls == 0
    assert resumed_llm.audit_calls == 0
    assert resumed_llm.repair_calls == 0


def test_pipeline_default_budget_executes_two_semantic_repairs_and_final_audit(
    ra,
    tmp_path: Path,
):
    def script(marker: str) -> str:
        return f'''\
import json
import os
import pandas as pd

# {marker}
df = pd.read_parquet(os.environ["COHORT_PARQUET"])
out = os.environ["STEP_OUT_DIR"]
summary = {{
    "n": int(len(df)),
    "output_files": [
        {{"kind": "table", "name": "cohort_summary", "path": "cohort_summary.csv"}}
    ],
}}
pd.DataFrame([summary]).to_csv(os.path.join(out, "cohort_summary.csv"), index=False)
with open(os.path.join(out, "step_summary.json"), "w", encoding="utf-8") as handle:
    json.dump(summary, handle)
'''

    class _TwoRepairLLM:
        name = "two-semantic-repair-budget-test"

        def __init__(self) -> None:
            self.audit_calls = 0
            self.repair_calls = 0

        def complete(self, messages, *, max_tokens=2048, temperature=0.2):
            del max_tokens, temperature
            system = "\n".join(
                str(message.content or "")
                for message in messages
                if message.role == "system"
            )
            user = next(
                (
                    str(message.content or "")
                    for message in reversed(messages)
                    if message.role == "user"
                ),
                "",
            )
            upper = user.upper()
            if "ICU-AWARE RESEARCH PLAN" in upper:
                return json.dumps(
                    {
                        "research_question": "Summarize the ICU cohort.",
                        "steps": [
                            {
                                "step_id": "01_summary",
                                "intent": "Produce a descriptive cohort summary.",
                                "inputs": ["stay_id"],
                                "expected_outputs": ["table:cohort_summary"],
                                "method": "descriptive_summary",
                                "icu_rule_refs": [],
                            }
                        ],
                        "rationale": "two semantic repair budget regression",
                    }
                )
            if "WRITE THE PYTHON CODE" in upper:
                return script("SEMANTIC_REPAIR_ROUND_1")
            if "CONSERVATIVE ICU CONCEPT-USE AUDITOR" in system.upper():
                self.audit_calls += 1
                marker = next(
                    (
                        value
                        for value in (
                            "SEMANTIC_REPAIR_ROUND_1",
                            "SEMANTIC_REPAIR_ROUND_2",
                        )
                        if value in user
                    ),
                    None,
                )
                return json.dumps(
                    {
                        "findings": (
                            [
                                {
                                    "severity": "error",
                                    "message": f"{marker} requires one repair.",
                                    "detail": {"issue_code": "other"},
                                }
                            ]
                            if marker
                            else []
                        )
                    }
                )
            if "REPAIR THE PYTHON CODE" in upper:
                self.repair_calls += 1
                return script(
                    "SEMANTIC_REPAIR_ROUND_2"
                    if self.repair_calls == 1
                    else "SEMANTIC_AUDIT_SAFE"
                )
            if "INTERPRET THE RESULTS" in upper:
                return "Cohort summary completed {evidence:cohort_summary}."
            return "{}"

    llm = _TwoRepairLLM()
    pipeline = ra.ResearchAgentPipeline(
        workdir=tmp_path,
        llm=llm,
        enable_literature=False,
        enable_visual_qa=False,
        enable_latex=False,
        enable_llm_concept_audit=True,
        enable_deterministic_code_fallback=False,
        enable_deterministic_runner_repair=False,
    )
    result = pipeline.run(
        question="Summarize the ICU cohort.",
        cohort=pd.DataFrame({"stay_id": [1, 2, 3], "death": [0, 1, 0]}),
        cohort_name="two_semantic_repair_budget_test",
        database="synthetic",
        target_outcome="death",
        stop_after_step_id="01_summary",
        stop_after_analysis=True,
    )
    partial = json.loads(
        (Path(result.workdir) / "manifest_partial.json").read_text(encoding="utf-8")
    )
    record = [
        item
        for item in partial["per_step_records"]
        if item.get("step_id") == "01_summary"
    ][-1]

    assert record["status"] == "ok"
    assert llm.repair_calls == 2
    assert llm.audit_calls == 3
    assert record["step_llm_repair_attempts"] == 2
    assert record["step_provider_call_categories"] == [
        "initial_generation",
        "concept_audit",
        "post_mutation_concept_repair_patch",
        "concept_audit",
        "post_mutation_concept_repair_patch",
        "concept_audit",
        "analyzer",
    ]
