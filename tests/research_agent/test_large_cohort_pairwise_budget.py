"""Large-cohort pairwise evaluation must stay deterministic and bounded."""

from __future__ import annotations

from easyicu.research_agent.agents.core import CoderAgent
from easyicu.research_agent.gates.method_compatibility import (
    PAIRWISE_EVALUATION_FULL_COHORT_MAX_ROWS,
    PAIRWISE_EVALUATION_MAX_SAMPLE_SIZE,
    detect_forbidden_pattern_usage,
    render_computational_budget_constraints,
)
from easyicu.research_agent.gates.concept import deterministic_code_gate_findings
from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient
from easyicu.research_agent.schema import (
    AnalysisStep,
    CohortDescriptor,
    ConceptDescriptor,
    ResearchContext,
    VariableRole,
)


def _context(*, n_stays: int) -> ResearchContext:
    return ResearchContext(
        research_question="Discover reproducible physiological subgroups.",
        cohort=CohortDescriptor(
            cohort_name="large_general_cohort",
            database="example",
            n_patients=n_stays,
            n_stays=n_stays,
            id_columns=["stay_id"],
        ),
        variables=[
            ConceptDescriptor(
                name="measurement_a",
                dtype="float64",
                role=VariableRole.VITAL,
            ),
            ConceptDescriptor(
                name="measurement_b",
                dtype="float64",
                role=VariableRole.VITAL,
            ),
        ],
    )


def _step() -> AnalysisStep:
    return AnalysisStep(
        step_id="05_cluster_structure",
        intent="Select and characterize a clustering solution.",
        method="k-means candidate comparison with silhouette evaluation",
    )


def _budget_violations(code: str, *, n_stays: int) -> list[dict[str, object]]:
    return [
        item
        for item in detect_forbidden_pattern_usage(code, _context(n_stays=n_stays), _step())
        if item.get("reason_code") == "large_cohort_silhouette_unbounded"
    ]


def test_large_cohort_rejects_full_pairwise_silhouette() -> None:
    code = """\
from sklearn.metrics import silhouette_score
score = silhouette_score(feature_matrix, cluster_labels)
"""

    violations = _budget_violations(
        code,
        n_stays=PAIRWISE_EVALUATION_FULL_COHORT_MAX_ROWS + 1,
    )

    assert len(violations) == 1
    assert violations[0]["kind"] == "computational_budget"
    assert "sample_size" in violations[0]["matched_patterns"]
    assert "random_state" in violations[0]["matched_patterns"]


def test_shared_deterministic_gate_attributes_budget_failure_to_owner() -> None:
    code = """\
from sklearn.metrics import silhouette_score
score = silhouette_score(feature_matrix, cluster_labels)
"""

    findings = deterministic_code_gate_findings(
        context=_context(
            n_stays=PAIRWISE_EVALUATION_FULL_COHORT_MAX_ROWS + 1,
        ),
        step=_step(),
        script_text=code,
    )

    finding = next(item for item in findings if item.validator == "method_compatibility")
    violations = finding.detail["violations"]
    assert violations[0]["reason_code"] == "large_cohort_silhouette_unbounded"
    assert finding.detail["step_id"] == _step().step_id


def test_large_cohort_accepts_bounded_deterministic_silhouette() -> None:
    code = f"""\
from sklearn.metrics import silhouette_score
SILHOUETTE_ROWS = min({PAIRWISE_EVALUATION_MAX_SAMPLE_SIZE}, len(feature_matrix))
for seed in (17, 29, 41):
    score = silhouette_score(
        feature_matrix,
        cluster_labels,
        sample_size=SILHOUETTE_ROWS,
        random_state=seed,
    )
"""

    assert not _budget_violations(
        code,
        n_stays=PAIRWISE_EVALUATION_FULL_COHORT_MAX_ROWS + 50_000,
    )


def test_large_cohort_accepts_integer_wrapped_bounded_contract() -> None:
    code = f"""\
from sklearn.metrics import silhouette_score
SILHOUETTE_ROWS = min({PAIRWISE_EVALUATION_MAX_SAMPLE_SIZE}, len(feature_matrix))
SILHOUETTE_SEED = 1729
score = silhouette_score(
    feature_matrix,
    cluster_labels,
    sample_size=int(SILHOUETTE_ROWS),
    random_state=int(SILHOUETTE_SEED),
)
"""

    assert not _budget_violations(
        code,
        n_stays=PAIRWISE_EVALUATION_FULL_COHORT_MAX_ROWS + 1,
    )


def test_large_cohort_rejects_sample_larger_than_bound() -> None:
    code = f"""\
from sklearn.metrics import silhouette_score
score = silhouette_score(
    feature_matrix,
    cluster_labels,
    sample_size={PAIRWISE_EVALUATION_MAX_SAMPLE_SIZE + 1},
    random_state=17,
)
"""

    violations = _budget_violations(
        code,
        n_stays=PAIRWISE_EVALUATION_FULL_COHORT_MAX_ROWS + 1,
    )

    assert len(violations) == 1
    assert "sample_size" in violations[0]["matched_patterns"]
    assert "random_state" not in violations[0]["matched_patterns"]


def test_large_cohort_rejects_dynamic_random_state() -> None:
    code = f"""\
import time
from sklearn.metrics import silhouette_score
score = silhouette_score(
    feature_matrix,
    cluster_labels,
    sample_size={PAIRWISE_EVALUATION_MAX_SAMPLE_SIZE},
    random_state=int(time.time()),
)
"""

    violations = _budget_violations(
        code,
        n_stays=PAIRWISE_EVALUATION_FULL_COHORT_MAX_ROWS + 1,
    )

    assert len(violations) == 1
    assert "sample_size" not in violations[0]["matched_patterns"]
    assert "random_state" in violations[0]["matched_patterns"]


def test_small_cohort_may_use_full_silhouette() -> None:
    code = """\
from sklearn import metrics
score = metrics.silhouette_score(feature_matrix, cluster_labels)
"""

    assert not _budget_violations(
        code,
        n_stays=PAIRWISE_EVALUATION_FULL_COHORT_MAX_ROWS,
    )


def test_large_cohort_prompt_publishes_same_bounded_contract() -> None:
    prompt = render_computational_budget_constraints(
        _context(n_stays=PAIRWISE_EVALUATION_FULL_COHORT_MAX_ROWS + 1)
    )

    assert "silhouette_score" in prompt
    assert f"sample_size <= {PAIRWISE_EVALUATION_MAX_SAMPLE_SIZE}" in prompt
    assert "random_state" in prompt
    assert "full-data model fitting" in prompt
    assert "sample size and seed" in prompt


def test_small_cohort_prompt_does_not_add_large_cohort_rule() -> None:
    assert (
        render_computational_budget_constraints(
            _context(n_stays=PAIRWISE_EVALUATION_FULL_COHORT_MAX_ROWS)
        )
        == ""
    )


def test_coder_receives_large_cohort_budget_contract() -> None:
    safe_code = f"""\
from sklearn.metrics import silhouette_score
score = silhouette_score(
    feature_matrix,
    cluster_labels,
    sample_size=min({PAIRWISE_EVALUATION_MAX_SAMPLE_SIZE}, len(feature_matrix)),
    random_state=17,
)
"""
    llm = ScriptedMockLLMClient([safe_code])

    CoderAgent(llm).run(
        context=_context(
            n_stays=PAIRWISE_EVALUATION_FULL_COHORT_MAX_ROWS + 1,
        ),
        step=_step(),
    )

    initial_prompt = "\n".join(message.content for message in llm.calls[0][0])
    assert "LARGE-COHORT COMPUTATIONAL BUDGET" in initial_prompt
    assert f"sample_size <= {PAIRWISE_EVALUATION_MAX_SAMPLE_SIZE}" in initial_prompt
