"""``ols(`` matched ``matched_controls.append`` -- "contr-OLS".

The method-compatibility gate normalises a callable path to alphanumerics and
then asked ``token in path``. Every forbidden pattern that names a callable was
therefore a substring test, so any identifier containing the token matched.

MEASURED over the whole recorded corpus: this gate has produced exactly ONE
finding, ever, and it was this. In h2's propensity-score script,
``matched_controls.append(control_index)`` normalises to
``matchedcontrolsappend``, which contains ``ols``. The agent was told it had run
ordinary least squares on a binary outcome and should use logistic regression;
it spent BOTH of its repairs on a call that does not exist -- ``ols(`` appears
nowhere in that file, not even with whitespace stripped -- and the step died
``blocked_by_concept_audit`` with two repair classes both reading
``compatibility``.

So the rule's lifetime record was: 0 true positives, 1 false positive, 1 dead
step. The check is kept -- a binary outcome fitted by OLS is a real error worth
failing closed on -- and only its matching is corrected: a pattern that names a
callable must match a NAME, compared over the path's dotted segments.

The two prose patterns (``report mean``, ``kmeans on binary``) describe habits
rather than callables and keep their substring test.
"""

from __future__ import annotations

import ast
import json
import pathlib

import pytest

from easyicu.research_agent.gates.method_compatibility import (
    FORBIDDEN_METHOD_BY_KIND,
    _call_matches_forbidden_pattern,
)

_CORPUS = pathlib.Path("/Volumes/外置硬盘/easyicu_data/canonical9_runs")


def _matches(source: str, pattern: str) -> bool:
    call = next(
        node for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Call)
    )
    return _call_matches_forbidden_pattern(call, pattern, {})


# ---------------------------------------------------------------------------
# The false positives
# ---------------------------------------------------------------------------


def test_the_recorded_false_positive_no_longer_matches():
    """The exact line, from the quarantined script that died."""

    assert not _matches("matched_controls.append(control_index)", "ols(")


@pytest.mark.parametrize(
    "source",
    [
        "available_controls.remove(index)",  # the second hit in that same file
        "controls(frame)",
        "protocols(frame)",
        "symbols(frame)",
        "tools.build(frame)",
        "n_controls(frame)",
    ],
)
def test_no_identifier_merely_containing_the_token_matches(source: str):
    assert not _matches(source, "ols(")


def test_the_latent_twin_of_the_same_bug_is_gone():
    """``.mean()`` matched any helper whose name ended in ``_mean``.

    The h2 script defines and calls ``weighted_mean(...)`` -- correct for an
    IPTW analysis -- which the substring test would have flagged for any
    right-skewed variable reaching it.
    """

    assert not _matches("weighted_mean(values, weights)", ".mean()")
    assert not _matches("mean_absolute_error(a, b)", ".mean()")


# ---------------------------------------------------------------------------
# Every true positive still matches
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "source,pattern",
    [
        ("sm.OLS(y, X)", "ols("),
        ("smf.ols(formula, data)", "ols("),
        ("statsmodels.api.OLS(y, X)", "ols("),
        ("LinearRegression().fit(X, y)", "linearregression"),
        ("sklearn.linear_model.LinearRegression()", "linearregression"),
        ("linear_regression(y, X)", "linear_regression("),
        ("np.mean(values)", ".mean()"),
        ("numpy.mean(values)", "numpy.mean("),
        ("frame['x'].mean()", ".mean()"),
        ("KMeans(n_clusters=3)", "kmeans"),
        ("sklearn.cluster.KMeans(n_clusters=3)", "kmeans"),
        ("DBSCAN(eps=0.5)", "dbscan"),
        ("PCA(n_components=2)", "pca"),
        ("TSNE(n_components=2)", "tsne"),
    ],
)
def test_a_real_forbidden_call_still_matches(source: str, pattern: str):
    assert _matches(source, pattern)


def test_the_prose_patterns_keep_their_substring_test():
    """They name a habit, not a callable, and must not be segment-matched."""

    assert _matches("report_mean_and_sd(values)", "report mean")
    assert _matches("kmeans_cluster(frame)", "kmeans on binary")


def test_a_dotted_pattern_still_matches_its_whole_path():
    assert _matches("umap.umap(frame)", "umap.umap")


# ---------------------------------------------------------------------------
# The rule survives, and the corpus record is what motivated the change
# ---------------------------------------------------------------------------


def test_the_binary_rule_still_forbids_least_squares():
    """Deleting the check would be the wrong fix; only matching was broken."""

    rule = FORBIDDEN_METHOD_BY_KIND["binary"]
    assert "ols(" in rule["forbidden_patterns"]
    assert "logistic_regression" in rule["preferred"]


def test_the_recorded_script_really_contains_no_least_squares_call():
    """The premise: the agent was asked to remove something that was not there."""

    script = (
        _CORPUS
        / "batch_20260804_luna_miiv_FULL_4f93e9b_verify22"
        / "h2_vasopressor_causal"
        / "aware"
        / "run_20260804T124315_2934c2"
        / "steps"
        / "04_primary_causal_effect_and_diagnostics"
        / ".quarantine"
        / "concept_draft.py"
    )
    if not script.exists():
        pytest.skip("the recorded h2 draft is not mounted")

    source = script.read_text(encoding="utf-8", errors="replace")
    flattened = "".join(source.split())

    assert "ols(" not in source.lower()
    assert "ols(" not in flattened.lower()
    # And the identifier that did match is present.
    assert "matched_controls" in source


def test_the_recorded_corpus_carries_one_ols_finding_and_it_is_that_one():
    """The OLS rule had 0 true positives and 1 false positive in this corpus."""

    if not _CORPUS.exists():
        pytest.skip("recorded run corpus is not mounted")

    ols_findings: list[list[str]] = []
    for path in _CORPUS.glob("batch_*/*/aware/run_*/manifest.json"):
        try:
            manifest = json.loads(path.read_text())
        except Exception:  # noqa: BLE001 - a malformed manifest is not the subject
            continue
        for finding in manifest.get("findings", []):
            if str(finding.get("validator")) != "method_compatibility":
                continue
            for violation in (finding.get("detail") or {}).get("violations", []):
                patterns = [
                    str(pattern)
                    for pattern in violation.get("matched_patterns") or []
                ]
                if "ols(" in patterns:
                    ols_findings.append(patterns)

    if not ols_findings:
        pytest.skip("no recorded run carries an OLS method-compatibility finding")
    assert ols_findings == [["ols("]], ols_findings
