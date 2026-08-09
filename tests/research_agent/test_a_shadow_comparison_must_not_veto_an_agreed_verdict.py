"""The gate that catches "the two views disagree" had never caught one.

``StepSummaryFractionEnvelopeDualReader`` runs the legacy bounded-metric
validator and a canonical envelope view, and -- by its own docstring --
"retains the legacy bounded-metric decision only after exact comparison". When
the comparison is not exact it DISCARDS the legacy decision and returns a
single blocking error, failing the step ``contract_failed``.

The message it emits is addressed to the host's own developers: "Keep the
legacy consumer active until source, digest, normalization, scalar-tree and
finding decisions agree exactly." It is delivered to an LLM that can do nothing
about a disagreement between two host implementations.

MEASURED over every recorded run: 11 distinct steps across 5 of the 9 tasks
(e1 5, m1 2, m3 1, e2 1, m2 1) died here, ALL ``contract_failed``, 8 of them
without a single repair attempt while 3-9 provider calls sat unspent.

And in ALL ELEVEN, ``legacy_findings_sha256 == canonical_findings_sha256`` with
``legacy_finding_count == canonical_finding_count == 0``. **The two verdicts
were byte-identical every single time.** What entered ``mismatches`` was the
canonical NORMALIZER complaining about the input it was handed -- one side's
internal diagnostic, not a disagreement between two decisions.

Verified by hand on today's instance (m2 ``03_feature_missingness_and_leakage_
audit``): both of the table's count/fraction pairs reconcile to six decimals
(stratum 82/84992 = 0.096480 %, cohort 96/94458 = 0.101632 %). The canonical
normalizer picks the denominator from a fixed list of column names
(``n_full``/``n_total``/``cohort_n``), so it paired a stratum-level count with
the whole-cohort total. Rows 0-3 escaped only because a zero numerator makes
the two denominators indistinguishable.

The fix is typed, not a string set at the consumer: a mismatch now declares
whether it is DECISIVE (a comparison of two things) or an observation (one
side's complaint about its own input). ``normalization_error`` is the only
code of the eleven that is the latter. Every genuine blocking case -- digest
mismatch, status mismatch, missing/unexpected artifact, invalid scalar tree,
and the verdict mismatch itself -- still blocks.
"""

from __future__ import annotations

import ast
import collections
import csv
import inspect
import json
import pathlib

import pytest

from easyicu.research_agent.audits import envelope_shadow
from easyicu.research_agent.audits.envelope_shadow import (
    FractionScaleShadowComparison,
    ValidatorShadowMismatch,
    compare_fraction_scale_shadow,
    fraction_scale_shadow_observed_findings,
)
from easyicu.research_agent.audits.validators import StepSummaryFractionValidator
from easyicu.research_agent.contracts.result_envelope import (
    normalize_step_result_shadow,
)
from easyicu.research_agent.schema import AnalysisStep

_CORPUS = pathlib.Path("/Volumes/外置硬盘/easyicu_data/canonical9_runs")
_SHA = "a" * 64


def _normalization(detail: str = "row[4]: one denominator") -> ValidatorShadowMismatch:
    return ValidatorShadowMismatch(
        code="normalization_error", detail=detail, decisive=False
    )


def _comparison(*mismatches, agree: bool = True) -> FractionScaleShadowComparison:
    return FractionScaleShadowComparison(
        step_id="03_feature_missingness_and_leakage_audit",
        exact_match=not any(item.decisive for item in mismatches),
        legacy_finding_count=0,
        canonical_finding_count=0,
        legacy_findings_sha256=_SHA,
        canonical_findings_sha256=_SHA if agree else "b" * 64,
        mismatches=tuple(mismatches),
    )


# ---------------------------------------------------------------------------
# What is decisive, and what is only observed
# ---------------------------------------------------------------------------


def _real_comparison(tmp_path) -> FractionScaleShadowComparison:
    """Reproduce m2's exact table and drive the REAL producer.

    Building the comparison by hand tests nothing: three of four mutations
    survived a first version of this file because its helper constructed the
    mismatch objects itself and computed ``exact_match`` locally, so the
    production construction sites and the production rule were never run.
    """

    with open(tmp_path / "missingness_leakage_audit.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "role",
                "variable",
                "cohort_n",
                "outcome_group_n",
                "missing_n",
                "available_n",
                "missing_pct",
            ]
        )
        # The recorded row, verbatim: 82/84992 = 0.096480 %, exact to six
        # decimals against the STRATUM denominator the row itself declares.
        writer.writerow(["missingness", "hr_first", 94458, 84992, 82, 84910, 0.096480])

    summary = {
        "step": "STEP 03_feature_missingness_and_leakage_audit",
        "status": "success",
        "output_files": {
            "table:missingness_leakage_audit": "missingness_leakage_audit.csv"
        },
    }
    step = AnalysisStep(
        step_id="03_feature_missingness_and_leakage_audit",
        intent="Audit feature missingness.",
    )
    envelope = normalize_step_result_shadow(
        step_id=step.step_id,
        step_summary=summary,
        output_dir=tmp_path,
        status="ok",
    )
    return compare_fraction_scale_shadow(
        step=step,
        step_summary=summary,
        current_status="ok",
        envelope=envelope,
        legacy_findings=StepSummaryFractionValidator().audit(
            step=step, step_summary=summary
        ),
    )


def test_the_recorded_table_no_longer_blocks_its_own_step(tmp_path):
    """End to end on m2's real row, through the real producer."""

    comparison = _real_comparison(tmp_path)

    # The premise: the canonical normalizer really does complain about it.
    assert [item.code for item in comparison.mismatches] == ["normalization_error"]
    # And the two verdicts really are identical.
    assert comparison.legacy_finding_count == comparison.canonical_finding_count == 0
    assert comparison.legacy_findings_sha256 == comparison.canonical_findings_sha256
    # So it is an observation, and the step is not failed.
    assert comparison.decisive_mismatches == ()
    assert len(comparison.observed_mismatches) == 1
    assert comparison.exact_match is True


def test_only_the_one_sided_diagnostic_is_ever_built_non_decisive():
    """Read the production construction sites, not a hand-built object.

    A mutation that marks a real comparison mismatch (say the verdict
    disagreement itself) non-decisive changes only a construction site, and a
    test that builds its own objects cannot see it.
    """

    tree = ast.parse(inspect.getsource(envelope_shadow))
    non_decisive: set[str] = set()
    every: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        if node.func.id != "ValidatorShadowMismatch":
            continue
        code = next(
            (
                kw.value.value
                for kw in node.keywords
                if kw.arg == "code" and isinstance(kw.value, ast.Constant)
            ),
            None,
        )
        if code is None:
            continue
        every.add(code)
        if any(
            kw.arg == "decisive"
            and isinstance(kw.value, ast.Constant)
            and kw.value.value is False
            for kw in node.keywords
        ):
            non_decisive.add(code)

    assert len(every) > 5, every
    assert non_decisive == {"normalization_error"}, non_decisive


@pytest.mark.parametrize(
    "code",
    [
        "canonical_artifact_missing",
        "canonical_envelope_missing",
        "canonical_source_digest_mismatch",
        "canonical_status_mismatch",
        "canonical_fraction_view_mismatch",
        "canonical_scalar_tree_invalid",
        "canonical_table_presence_mismatch",
        "canonical_unexpected_artifact",
        "envelope_digest_invalid",
        "summary_not_mapping",
    ],
)
def test_every_other_code_still_blocks(code: str):
    """Only one of the eleven codes is one-sided. The rest compare two things."""

    mismatch = ValidatorShadowMismatch(code=code, detail="x")
    assert mismatch.decisive is True
    assert _comparison(mismatch).exact_match is False


def test_the_verdict_mismatch_itself_is_still_decisive():
    """This is the disagreement the comparison exists for. It must still fail."""

    disagreement = ValidatorShadowMismatch(
        code="canonical_fraction_view_mismatch",
        detail="Legacy and canonical findings were not byte-equivalent.",
    )
    comparison = _comparison(disagreement, agree=False)

    assert comparison.exact_match is False
    assert comparison.decisive_mismatches == (disagreement,)


def test_a_decisive_code_beside_an_observation_still_blocks():
    """Two of the eleven recorded steps also carried a missing artifact."""

    comparison = _comparison(
        _normalization(),
        ValidatorShadowMismatch(code="canonical_artifact_missing", detail="x"),
    )
    assert comparison.exact_match is False


# ---------------------------------------------------------------------------
# The observation is still recorded -- the migration stays measurable
# ---------------------------------------------------------------------------


def test_the_observation_is_reported_but_cannot_be_read_as_a_verdict():
    findings = fraction_scale_shadow_observed_findings(
        validator_name="step_summary_fraction_scale",
        step_id="03_feature_missingness_and_leakage_audit",
        comparison=_comparison(_normalization()),
    )

    assert len(findings) == 1
    finding = findings[0]
    assert finding.severity == "info"
    assert finding.detail["canonical_shadow_blocked"] is False
    assert finding.detail["decision_effect"] == "none"
    assert finding.detail["mismatch_codes"] == ["normalization_error"]
    # The cause is still in the message -- the channel an LLM actually reads.
    assert "row[4]" in finding.message


def test_an_agreeing_comparison_with_nothing_to_observe_says_nothing():
    assert (
        fraction_scale_shadow_observed_findings(
            validator_name="step_summary_fraction_scale",
            step_id="03_x",
            comparison=_comparison(),
        )
        == []
    )


# ---------------------------------------------------------------------------
# End to end through the consumer that owns the decision
# ---------------------------------------------------------------------------


def test_the_consumer_keeps_the_legacy_verdict_when_the_views_agree(monkeypatch):
    from easyicu.research_agent.audits import envelope_consumers
    from easyicu.research_agent.schema import AnalysisStep

    step = AnalysisStep(
        step_id="03_feature_missingness_and_leakage_audit",
        intent="Audit feature missingness.",
        method="descriptive_summary",
        planned_analysis_role="auxiliary",
        inputs=["artifact:analysis_cohort"],
        expected_outputs=["table:missingness_leakage_audit"],
    )
    monkeypatch.setattr(
        envelope_consumers,
        "compare_fraction_scale_shadow",
        lambda **_kwargs: _comparison(_normalization()),
    )

    findings = envelope_consumers.StepSummaryFractionEnvelopeDualReader().audit(
        step=step,
        step_summary={},
        envelope=None,
        current_status="completed",
        legacy_findings=[],
    )

    # The legacy verdict was a pass, and it survives.
    assert [f for f in findings if f.severity == "error"] == []
    assert [f.severity for f in findings] == ["info"]


def test_the_consumer_still_blocks_a_real_disagreement(monkeypatch):
    from easyicu.research_agent.audits import envelope_consumers
    from easyicu.research_agent.schema import AnalysisStep

    step = AnalysisStep(
        step_id="03_x",
        intent="Audit feature missingness.",
        method="descriptive_summary",
        planned_analysis_role="auxiliary",
        inputs=["artifact:analysis_cohort"],
        expected_outputs=["table:missingness_leakage_audit"],
    )
    monkeypatch.setattr(
        envelope_consumers,
        "compare_fraction_scale_shadow",
        lambda **_kwargs: _comparison(
            ValidatorShadowMismatch(
                code="canonical_fraction_view_mismatch", detail="x"
            ),
            agree=False,
        ),
    )

    findings = envelope_consumers.StepSummaryFractionEnvelopeDualReader().audit(
        step=step,
        step_summary={},
        envelope=None,
        current_status="completed",
        legacy_findings=[],
    )

    assert [f.severity for f in findings] == ["error"]
    assert findings[0].detail["canonical_shadow_blocked"] is True


# ---------------------------------------------------------------------------
# The corpus record: it never once caught a real disagreement
# ---------------------------------------------------------------------------


def test_no_recorded_block_was_ever_a_real_disagreement():
    if not _CORPUS.exists():
        pytest.skip("recorded run corpus is not mounted")

    blocked = {}
    for path in _CORPUS.glob("batch_*/*/aware/run_*/manifest*.json"):
        try:
            manifest = json.loads(path.read_text())
        except Exception:  # noqa: BLE001 - a malformed manifest is not the subject
            continue
        for record in manifest.get("per_step_records", []):
            for finding in (record.get("contract_findings") or []) + (
                record.get("usage_findings") or []
            ):
                detail = finding.get("detail")
                if not isinstance(detail, dict):
                    continue
                if not detail.get("canonical_shadow_blocked"):
                    continue
                blocked[(path.parts[-5], path.parts[-4], record.get("step_id"))] = detail

    if not blocked:
        pytest.skip("no recorded run was blocked by the shadow comparison")

    disagreements = [
        key
        for key, detail in blocked.items()
        if detail.get("legacy_findings_sha256") != detail.get("canonical_findings_sha256")
    ]
    tasks = collections.Counter(key[1] for key in blocked)

    assert len(tasks) >= 4, tasks
    assert disagreements == [], disagreements
