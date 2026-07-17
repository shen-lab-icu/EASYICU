"""Tests for Patch C — post-codegen pre-execution forbidden-pattern check.

These tests do NOT call a real LLM. They use a scripted client that
returns pre-fabricated code strings — first a "bad" script that uses
a forbidden estimator over an ordinal variable, then a "clean"
script. The expected behaviour:

* ``CoderAgent.run`` detects the violation
* ``CoderAgent.run`` invokes ``CoderAgent.repair`` with a structured
  error message in the ``run_log`` field
* the repaired (clean) code is returned
* ``last_compatibility_violations`` records the violation, and
  ``last_compatibility_repair_attempts`` counts the actual repairs.

If the repair budget is exhausted without producing a clean script,
the last (still-violating) script is returned unchanged so the
post-hoc validator in ``audits/patterns.py`` records the issue.
"""

from __future__ import annotations

import json
from typing import List

import pytest

from easyicu.research_agent.agents import (
    CoderAgent,
    _MAX_PRE_EXEC_COMPATIBILITY_REPAIRS,
)
from easyicu.research_agent.code_patch import PATCH_FORMAT
from easyicu.research_agent.llm import LLMMessage
from easyicu.research_agent.method_compatibility import (
    detect_forbidden_pattern_usage,
    format_violation_message,
)
from easyicu.research_agent.provider_budget import (
    ProviderCallBudgetExhausted,
    StepProviderCallBudget,
)
from easyicu.research_agent.schema import (
    AnalysisStep,
    CohortDescriptor,
    ConceptDescriptor,
    ResearchContext,
    VariableRole,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _ordinal_context() -> ResearchContext:
    """ResearchContext with one ordinal (gcs) and one continuous variable."""
    return ResearchContext(
        research_question="Cluster ICU stays by physiology.",
        cohort=CohortDescriptor(
            cohort_name="t",
            database="miiv",
            n_patients=10,
            n_stays=10,
            id_columns=["stay_id"],
            outcome_columns=["death"],
        ),
        variables=[
            ConceptDescriptor(
                name="gcs", dtype="int64", role=VariableRole.ORDINAL_SCORE
            ),
            ConceptDescriptor(name="hr", dtype="float64", role=VariableRole.VITAL),
        ],
    )


_BAD_SCRIPT = """\
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
df = pd.read_parquet("cohort.parquet")
X = df[["gcs", "hr"]].fillna(0).values
clusters = MiniBatchKMeans(n_clusters=3, random_state=0).fit_predict(X)
"""

_CLEAN_SCRIPT = """\
import pandas as pd
from scipy.stats import spearmanr
df = pd.read_parquet("cohort.parquet")
rho, p = spearmanr(df["gcs"], df["hr"])
"""


def _patch(*edits: tuple[str, str]) -> str:
    return json.dumps(
        {
            "format": PATCH_FORMAT,
            "edits": [
                {"old": old, "new": new, "expected_count": 1} for old, new in edits
            ],
        }
    )


_CLEAN_PATCH = _patch(
    (
        "from sklearn.cluster import MiniBatchKMeans",
        "from scipy.stats import spearmanr",
    ),
    (
        'X = df[["gcs", "hr"]].fillna(0).values\n'
        "clusters = MiniBatchKMeans(n_clusters=3, random_state=0).fit_predict(X)",
        'rho, p = spearmanr(df["gcs"], df["hr"])',
    ),
)


def _still_bad_patch(index: int) -> str:
    return _patch((f"random_state={index}", f"random_state={index + 1}"))


class _ScriptedLLM:
    name = "scripted"

    def __init__(self, replies: List[str]) -> None:
        self.replies = list(replies)
        self.calls: List[List[LLMMessage]] = []

    def complete(self, messages, *, max_tokens=2048, temperature=0.2):
        self.calls.append(list(messages))
        if not self.replies:
            raise RuntimeError("scripted LLM ran out of replies")
        return self.replies.pop(0)


# ---------------------------------------------------------------------------
# Detection unit tests (no agent involvement)
# ---------------------------------------------------------------------------


def test_detector_flags_minibatchkmeans_over_ordinal():
    ctx = _ordinal_context()
    violations = detect_forbidden_pattern_usage(_BAD_SCRIPT, ctx)
    assert len(violations) == 1, violations
    v = violations[0]
    assert v["variable"] == "gcs"
    assert v["kind"] == "ordinal"
    assert any("minibatchkmeans" in p.lower() for p in v["matched_patterns"])


def test_detector_passes_clean_script():
    ctx = _ordinal_context()
    assert detect_forbidden_pattern_usage(_CLEAN_SCRIPT, ctx) == []


def test_detector_rejects_positional_sparse_event_helper_columns():
    script = """\
from easyicu.research_agent.methods.source_status import reconcile_binary_event_presence
result = reconcile_binary_event_presence(frame, count_col, measured_col, representative_col)
"""
    violations = detect_forbidden_pattern_usage(script, _ordinal_context())
    assert len(violations) == 1
    assert violations[0]["kind"] == "method_helper_call_contract"


def test_detector_accepts_keyword_only_sparse_event_helper_columns():
    script = """\
from easyicu.research_agent.methods.source_status import reconcile_binary_event_presence
result = reconcile_binary_event_presence(
    frame,
    count_column=count_col,
    measured_column=measured_col,
    representative_column=representative_col,
)
"""
    assert detect_forbidden_pattern_usage(script, _ordinal_context()) == []


def test_detector_ignores_forbidden_pattern_when_variable_not_referenced():
    """Forbidden pattern present but variable not used → no violation."""
    ctx = _ordinal_context()
    script = "from sklearn.cluster import KMeans\nX = [[1,2],[3,4]]\nKMeans(2).fit(X)"
    # 'gcs' is NOT in the script → no constrained variable is at risk
    assert detect_forbidden_pattern_usage(script, ctx) == []


def test_detector_uses_word_boundary_for_variable_name():
    """`gcs` should not match `gcsscaler` or arbitrary substrings."""
    ctx = _ordinal_context()
    # Use the forbidden pattern AND a similar-looking but distinct identifier
    script = "from sklearn.cluster import KMeans\ngcsscaler = 1\nKMeans(2).fit([[1,2]])"
    assert detect_forbidden_pattern_usage(script, ctx) == []


def test_format_violation_message_includes_preferred_alternatives():
    ctx = _ordinal_context()
    violations = detect_forbidden_pattern_usage(_BAD_SCRIPT, ctx)
    msg = format_violation_message(violations)
    assert "PRE-EXECUTION COMPATIBILITY CHECK FAILED" in msg
    assert "ordinal" in msg
    assert "spearman_correlation" in msg or "ordinal_logistic_regression" in msg
    # Must explicitly call out the bypass-via-variant pitfall
    assert "switching" in msg.lower() or "variant" in msg.lower()


# ---------------------------------------------------------------------------
# CoderAgent.run end-to-end behaviour with scripted LLM
# ---------------------------------------------------------------------------


def _step() -> AnalysisStep:
    return AnalysisStep(step_id="01_test", intent="cluster")


def test_coderagent_requests_full_token_budget():
    """Regression (E1 20260611): code generation must request the full
    ``_CODER_MAX_TOKENS`` so a verbose model's analysis.py is not truncated
    mid-expression (SyntaxError "'(' was never closed")."""
    from easyicu.research_agent.agents import _CODER_MAX_TOKENS

    captured = {}

    class _RecordingLLM:
        name = "rec"

        def complete(self, messages, *, max_tokens=2048, temperature=0.2):
            captured["max_tokens"] = max_tokens
            return _CLEAN_SCRIPT

    agent = CoderAgent(_RecordingLLM())
    agent.run(context=_ordinal_context(), step=_step())
    assert _CODER_MAX_TOKENS >= 8192
    assert captured["max_tokens"] == _CODER_MAX_TOKENS


def test_coderagent_run_triggers_repair_on_violation_then_returns_clean_code():
    llm = _ScriptedLLM([_BAD_SCRIPT, _CLEAN_PATCH])
    agent = CoderAgent(llm)
    code = agent.run(context=_ordinal_context(), step=_step())
    assert code == _CLEAN_SCRIPT.strip() or "spearman" in code
    # First call: initial. Second call: repair triggered by violation.
    assert len(llm.calls) == 2
    # The repair user message must include the violation block.
    repair_user_msg = llm.calls[1][-1].content
    assert "PRE-EXECUTION COMPATIBILITY CHECK FAILED" in repair_user_msg
    assert (
        "MiniBatchKMeans" in repair_user_msg
        or "minibatchkmeans" in repair_user_msg.lower()
    )
    # The agent records what happened.
    assert agent.last_compatibility_repair_attempts == 1
    # last_compatibility_violations reflects the FINAL state — clean now.
    assert agent.last_compatibility_violations == []


def test_coderagent_repair_rejects_non_script_output():
    # Minimal-patch mode gets one reply, then the full-rewrite fallback gets a
    # second. Both are invalid so the final non-script guard must reject them.
    llm = _ScriptedLLM(["{}", "{}"])
    agent = CoderAgent(llm)

    with pytest.raises(ValueError, match="non-script output"):
        agent.repair(
            context=_ordinal_context(),
            step=_step(),
            code=_BAD_SCRIPT,
            run_log="contract failed",
            attempt=1,
        )


def test_coderagent_repair_allows_only_contract_named_method_modules():
    llm = _ScriptedLLM([_CLEAN_PATCH])
    agent = CoderAgent(llm)

    agent.repair(
        context=_ordinal_context(),
        step=AnalysisStep(
            step_id="01_ordered",
            intent="summarise outcomes across ordered groups",
            method="ordinal_stratified_descriptive_analysis",
        ),
        code=_BAD_SCRIPT,
        run_log="contract failed",
        attempt=1,
    )

    repair_prompt = llm.calls[0][-1].content
    assert "easyicu.research_agent.methods.*" in repair_prompt
    assert "explicitly named by the code contract" in repair_prompt
    assert "All other project-local imports" in repair_prompt


def test_coderagent_run_no_repair_when_first_attempt_is_clean():
    llm = _ScriptedLLM([_CLEAN_SCRIPT])
    agent = CoderAgent(llm)
    code = agent.run(context=_ordinal_context(), step=_step())
    assert "spearman" in code
    # Only the initial call — no repair triggered.
    assert len(llm.calls) == 1
    assert agent.last_compatibility_repair_attempts == 0
    assert agent.last_compatibility_violations == []


def test_coderagent_run_gives_up_after_max_repairs_and_returns_last_attempt():
    """If the LLM keeps writing forbidden patterns, return the last attempt
    so the post-hoc validator can record it in the audit trail.

    The logical repair budget is bounded by
    _MAX_PRE_EXEC_COMPATIBILITY_REPAIRS. This fixture returns a successful
    minimal patch for each logical attempt, so each attempt uses one call.
    """
    # Initial generation returns a bad script. Each exact patch changes code but
    # deliberately leaves the forbidden method in place for the next check.
    n_attempts = 1 + _MAX_PRE_EXEC_COMPATIBILITY_REPAIRS
    llm = _ScriptedLLM(
        [
            _BAD_SCRIPT,
            *(
                _still_bad_patch(index)
                for index in range(_MAX_PRE_EXEC_COMPATIBILITY_REPAIRS)
            ),
        ]
    )
    agent = CoderAgent(llm)
    code = agent.run(context=_ordinal_context(), step=_step())
    # Returns last attempt (still bad)
    assert "MiniBatchKMeans" in code
    assert len(llm.calls) == n_attempts
    assert (
        agent.last_compatibility_repair_attempts == _MAX_PRE_EXEC_COMPATIBILITY_REPAIRS
    )
    # Violations are still recorded for the final state.
    assert len(agent.last_compatibility_violations) >= 1


def test_coderagent_compatibility_repairs_share_provider_budget():
    llm = _ScriptedLLM([_BAD_SCRIPT, _still_bad_patch(0)])
    agent = CoderAgent(llm)
    budget = StepProviderCallBudget(2, step_id="01_test")

    with pytest.raises(ProviderCallBudgetExhausted) as exc_info:
        agent.run(
            context=_ordinal_context(),
            step=_step(),
            provider_budget=budget,
        )

    assert exc_info.value.category == "compatibility_repair_patch"
    assert len(llm.calls) == 2
    assert budget.categories == (
        "initial_generation",
        "compatibility_repair_patch",
    )


def test_coderagent_run_does_not_loop_on_context_without_constrained_variables():
    """Continuous-only cohort → no compatibility constraint → no repair loop."""
    ctx = ResearchContext(
        research_question="hr vs map",
        cohort=CohortDescriptor(
            cohort_name="t",
            database="miiv",
            n_patients=1,
            n_stays=1,
            id_columns=["stay_id"],
            outcome_columns=["map"],
        ),
        variables=[
            ConceptDescriptor(name="hr", dtype="float64", role=VariableRole.VITAL),
            ConceptDescriptor(name="map", dtype="float64", role=VariableRole.VITAL),
        ],
    )
    # Even a "bad-looking" script is allowed when no ordinal var is in context.
    llm = _ScriptedLLM([_BAD_SCRIPT.replace("gcs", "hr")])
    agent = CoderAgent(llm)
    code = agent.run(context=ctx, step=_step())
    assert len(llm.calls) == 1, "no repair should have been triggered"
    assert agent.last_compatibility_repair_attempts == 0
