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


def test_large_cohort_accepts_seed_derived_from_fixed_integer_range() -> None:
    code = f"""\
from sklearn.metrics import silhouette_score
SEED_BASE = 1729
for candidate_count in range(2, 7):
    candidate_seed = SEED_BASE + candidate_count
    score = silhouette_score(
        feature_matrix,
        cluster_labels,
        sample_size={PAIRWISE_EVALUATION_MAX_SAMPLE_SIZE},
        random_state=candidate_seed,
    )
"""

    assert not _budget_violations(
        code,
        n_stays=PAIRWISE_EVALUATION_FULL_COHORT_MAX_ROWS + 1,
    )


def test_large_cohort_rejects_seed_derived_from_dynamic_range() -> None:
    code = f"""\
from sklearn.metrics import silhouette_score
SEED_BASE = 1729
for candidate_count in range(2, runtime_candidate_limit()):
    candidate_seed = SEED_BASE + candidate_count
    score = silhouette_score(
        feature_matrix,
        cluster_labels,
        sample_size={PAIRWISE_EVALUATION_MAX_SAMPLE_SIZE},
        random_state=candidate_seed,
    )
"""

    violations = _budget_violations(
        code,
        n_stays=PAIRWISE_EVALUATION_FULL_COHORT_MAX_ROWS + 1,
    )

    assert len(violations) == 1
    assert "random_state" in violations[0]["matched_patterns"]


def test_large_cohort_accepts_provably_safe_silhouette_wrapper() -> None:
    code = f"""\
from sklearn.metrics import silhouette_score as _sklearn_silhouette_score

def silhouette_score(features, labels, sample_size=None, random_state=None, **kwargs):
    bounded_sample_size = (
        min({PAIRWISE_EVALUATION_MAX_SAMPLE_SIZE}, len(features))
        if sample_size is None
        else min(sample_size, {PAIRWISE_EVALUATION_MAX_SAMPLE_SIZE})
    )
    deterministic_random_state = 42 if random_state is None else random_state
    return _sklearn_silhouette_score(
        features,
        labels,
        sample_size=bounded_sample_size,
        random_state=deterministic_random_state,
        **kwargs,
    )

BASE_SEED = 1729
evaluation_rows = min({PAIRWISE_EVALUATION_MAX_SAMPLE_SIZE}, len(feature_matrix))
score = silhouette_score(
    feature_matrix,
    cluster_labels,
    sample_size=int(evaluation_rows),
    random_state=int(BASE_SEED + 300000),
)
"""

    assert not _budget_violations(
        code,
        n_stays=PAIRWISE_EVALUATION_FULL_COHORT_MAX_ROWS + 1,
    )


def test_large_cohort_accepts_provably_safe_kwargs_silhouette_wrapper() -> None:
    code = f"""\
from sklearn.metrics import silhouette_score as _sklearn_pairwise_silhouette_score

SILHOUETTE_SAMPLE_SIZE = {PAIRWISE_EVALUATION_MAX_SAMPLE_SIZE}
PRIMARY_SEED = 42

def _pairwise_silhouette_score(X, labels, *args, **kwargs):
    kwargs["sample_size"] = min(SILHOUETTE_SAMPLE_SIZE, int(len(X)))
    kwargs["random_state"] = PRIMARY_SEED
    return _sklearn_pairwise_silhouette_score(X, labels, *args, **kwargs)

score = _pairwise_silhouette_score(feature_matrix, cluster_labels)
"""

    assert not _budget_violations(
        code,
        n_stays=PAIRWISE_EVALUATION_FULL_COHORT_MAX_ROWS + 1,
    )

    dynamic_seed = code.replace(
        'kwargs["random_state"] = PRIMARY_SEED',
        'kwargs["random_state"] = runtime_seed()',
    )
    assert _budget_violations(
        dynamic_seed,
        n_stays=PAIRWISE_EVALUATION_FULL_COHORT_MAX_ROWS + 1,
    )


def test_large_cohort_rejects_wrapper_without_a_provable_sample_cap() -> None:
    code = """\
from sklearn.metrics import silhouette_score as _sklearn_silhouette_score

def silhouette_score(features, labels, sample_size=None, random_state=None):
    evaluation_rows = len(features) if sample_size is None else sample_size
    deterministic_seed = 42 if random_state is None else random_state
    return _sklearn_silhouette_score(
        features,
        labels,
        sample_size=evaluation_rows,
        random_state=deterministic_seed,
    )

score = silhouette_score(feature_matrix, cluster_labels, random_state=1729)
"""

    violations = _budget_violations(
        code,
        n_stays=PAIRWISE_EVALUATION_FULL_COHORT_MAX_ROWS + 1,
    )

    assert len(violations) == 1
    assert "sample_size" in violations[0]["matched_patterns"]


def test_large_cohort_rejects_wrapper_with_dynamic_random_state() -> None:
    code = f"""\
import time
from sklearn.metrics import silhouette_score as _sklearn_silhouette_score

def silhouette_score(features, labels, sample_size=None, random_state=None):
    evaluation_rows = min({PAIRWISE_EVALUATION_MAX_SAMPLE_SIZE}, len(features))
    return _sklearn_silhouette_score(
        features,
        labels,
        sample_size=evaluation_rows,
        random_state=random_state,
    )

score = silhouette_score(
    feature_matrix,
    cluster_labels,
    random_state=int(time.time()),
)
"""

    violations = _budget_violations(
        code,
        n_stays=PAIRWISE_EVALUATION_FULL_COHORT_MAX_ROWS + 1,
    )

    assert len(violations) == 1
    assert "random_state" in violations[0]["matched_patterns"]


def test_large_cohort_accepts_explicit_bounded_deterministic_subset() -> None:
    code = f"""\
import numpy as np
from sklearn.metrics import silhouette_score
SILHOUETTE_ROWS = {PAIRWISE_EVALUATION_MAX_SAMPLE_SIZE}
SILHOUETTE_SEED = 1729

def sampled_silhouette(feature_matrix, cluster_labels, seed):
    n_rows = len(cluster_labels)
    sample_size = min(int(SILHOUETTE_ROWS), n_rows)
    if sample_size < n_rows:
        rng = np.random.default_rng(int(seed))
        indices = rng.choice(n_rows, size=sample_size, replace=False)
        sampled_features = feature_matrix[indices]
        sampled_labels = cluster_labels[indices]
    else:
        sampled_features = feature_matrix
        sampled_labels = cluster_labels
    return silhouette_score(sampled_features, sampled_labels), sample_size

score, evaluation_n = sampled_silhouette(
    feature_matrix,
    cluster_labels,
    seed=SILHOUETTE_SEED,
)
"""

    assert not _budget_violations(
        code,
        n_stays=PAIRWISE_EVALUATION_FULL_COHORT_MAX_ROWS + 50_000,
    )


def test_large_cohort_accepts_index_first_bounded_deterministic_subset() -> None:
    code = f"""\
import numpy as np
from sklearn.metrics import silhouette_score as sklearn_silhouette_score
SILHOUETTE_ROWS = {PAIRWISE_EVALUATION_MAX_SAMPLE_SIZE}
SILHOUETTE_SEED = 1729

def deterministic_silhouette(feature_matrix, labels, sample_n, seed):
    n_rows = int(feature_matrix.shape[0])
    bounded_n = min(int(sample_n), n_rows)
    if bounded_n > {PAIRWISE_EVALUATION_MAX_SAMPLE_SIZE}:
        raise ValueError("sample exceeds the declared bound")
    rng = np.random.default_rng(int(seed))
    if bounded_n == n_rows:
        indices = np.arange(n_rows, dtype=int)
    else:
        indices = np.sort(rng.choice(n_rows, size=bounded_n, replace=False))
    sampled_labels = np.asarray(labels)[indices]
    return sklearn_silhouette_score(
        feature_matrix[indices, :],
        sampled_labels,
    )

evaluation_n = min(SILHOUETTE_ROWS, len(cluster_labels))
score = deterministic_silhouette(
    feature_matrix,
    cluster_labels,
    evaluation_n,
    SILHOUETTE_SEED,
)
"""

    assert not _budget_violations(
        code,
        n_stays=PAIRWISE_EVALUATION_FULL_COHORT_MAX_ROWS + 50_000,
    )


def test_large_cohort_rejects_unseeded_explicit_subset() -> None:
    code = f"""\
import numpy as np
from sklearn.metrics import silhouette_score
sample_size = min({PAIRWISE_EVALUATION_MAX_SAMPLE_SIZE}, len(cluster_labels))
rng = np.random.default_rng()
indices = rng.choice(len(cluster_labels), size=sample_size, replace=False)
score = silhouette_score(feature_matrix[indices], cluster_labels[indices])
"""

    violations = _budget_violations(
        code,
        n_stays=PAIRWISE_EVALUATION_FULL_COHORT_MAX_ROWS + 1,
    )

    assert len(violations) == 1
    assert "random_state" in violations[0]["matched_patterns"]


def test_large_cohort_rejects_subset_overwritten_with_full_cohort() -> None:
    code = f"""\
import numpy as np
from sklearn.metrics import silhouette_score
sample_size = min({PAIRWISE_EVALUATION_MAX_SAMPLE_SIZE}, len(cluster_labels))

def sampled_silhouette(feature_matrix, cluster_labels):
    rng = np.random.default_rng(1729)
    indices = rng.choice(len(cluster_labels), size=sample_size, replace=False)
    sampled_features = feature_matrix[indices]
    sampled_labels = cluster_labels[indices]
    sampled_features = feature_matrix
    sampled_labels = cluster_labels
    return silhouette_score(sampled_features, sampled_labels)

score = sampled_silhouette(feature_matrix, cluster_labels)
"""

    violations = _budget_violations(
        code,
        n_stays=PAIRWISE_EVALUATION_FULL_COHORT_MAX_ROWS + 1,
    )

    assert len(violations) == 1
    assert "sample_size" in violations[0]["matched_patterns"]


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
