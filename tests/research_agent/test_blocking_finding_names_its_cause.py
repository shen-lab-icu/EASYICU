"""A blocking finding must name its cause in the channel the agent reads.

Live E1 blocker, 2026-07-29, ``run_20260729T062855_175303``, step
``06_missingness_event_timing_audit``. The step died ``contract_failed`` and
the agent was told only this::

    Canonical bounded fraction/percentage shadow could not safely replace the
    legacy view for step 06_missingness_event_timing_audit. Keep the legacy
    consumer active until source, digest, normalization, scalar-tree, and
    finding decisions agree exactly.

That names no defect. The real cause -- the canonical normalizer rejecting a
registered missingness partition on ``death_time`` -- lived on
``detail["mismatch_details"]``, and ``detail`` never reaches a prompt:
``_compact_findings`` projects every finding down to validator / severity /
message and clips the message to 240 characters. Deterministic repairs read
``detail``; LLM consumers read ``message`` and nothing else.

Two consequences the tests below lock:

* the cause belongs in ``message``, and
* it belongs at the **front** of it, because the only budget on that channel
  clips from the tail.
"""

from __future__ import annotations

from easyicu.research_agent.agents.core import (
    _REPLANNER_FINDING_MESSAGE_CHARS,
    _compact_findings,
)
from easyicu.research_agent.audits.envelope_shadow import (
    FractionScaleShadowComparison,
    ValidatorShadowMismatch,
    fraction_scale_shadow_blocking_finding,
)

_STEP_ID = "06_missingness_event_timing_audit"
_ZERO_SHA = "0" * 64

# The exact detail string ``compare_validator_shadow_inputs`` builds for the
# error that blocked the live step.
_REAL_DETAIL = (
    "Canonical normalization reported error "
    "'inconsistent_registered_missingness_partition'."
)


def _finding(*details: str):
    comparison = FractionScaleShadowComparison(
        step_id=_STEP_ID,
        exact_match=False,
        legacy_finding_count=0,
        canonical_finding_count=0,
        legacy_findings_sha256=_ZERO_SHA,
        canonical_findings_sha256=_ZERO_SHA,
        mismatches=tuple(
            ValidatorShadowMismatch(code="normalization_error", detail=detail)
            for detail in details
        ),
    )
    return fraction_scale_shadow_blocking_finding(
        validator_name="step_summary_fraction_envelope_dual_reader",
        step_id=_STEP_ID,
        comparison=comparison,
    )


def _as_agent_sees_it(finding) -> str:
    """Project the finding exactly as the replanner prompt builder does."""

    compact = _compact_findings([finding.model_dump(mode="json")])
    assert len(compact) == 1
    assert set(compact[0]) <= {"validator", "severity", "message"}, (
        "detail must stay out of the prompt projection; if it ever arrives "
        "here, this test is guarding the wrong channel"
    )
    return compact[0]["message"]


def test_the_agent_is_told_which_error_blocked_the_step() -> None:
    """The live regression: the inner code reaches the prompt."""

    message = _as_agent_sees_it(_finding(_REAL_DETAIL))

    assert "inconsistent_registered_missingness_partition" in message


def test_the_cause_survives_the_prompt_clip() -> None:
    """Ordering is load-bearing, not cosmetic.

    ``_compact_findings`` clips from the tail, so boilerplate-first would
    deliver the same contentless message the live run produced even though the
    cause was technically present in the untruncated string.
    """

    finding = _finding(_REAL_DETAIL)
    assert len(finding.message) > _REPLANNER_FINDING_MESSAGE_CHARS, (
        "this test only proves something while the message is long enough to "
        "be clipped; shorten it and the ordering guarantee goes untested"
    )

    assert "inconsistent_registered_missingness_partition" in _as_agent_sees_it(finding)


def test_every_distinct_cause_is_named() -> None:
    """A step blocked for two reasons must not report only the first."""

    message = _finding(
        _REAL_DETAIL,
        "The envelope was not compiled from these validator inputs.",
    ).message

    assert "inconsistent_registered_missingness_partition" in message
    assert "not compiled from these validator inputs" in message


def test_a_repeated_cause_is_stated_once() -> None:
    message = _finding(_REAL_DETAIL, _REAL_DETAIL).message

    assert message.count("inconsistent_registered_missingness_partition") == 1


def test_a_blocked_step_with_no_recorded_cause_says_so() -> None:
    """Silence is itself reportable -- never an empty accusation."""

    assert "No mismatch detail was recorded." in _finding().message


def test_the_structured_detail_is_still_carried_for_deterministic_readers() -> None:
    """Moving the cause into the message must not remove it from ``detail``."""

    detail = _finding(_REAL_DETAIL).detail or {}

    assert detail["mismatch_details"] == [_REAL_DETAIL]
    assert detail["mismatch_codes"] == ["normalization_error"]
    assert detail["canonical_shadow_blocked"] is True


# --- one level deeper: the cause must name the cell, not just the code -------
#
# canary7, 2026-07-31, step ``02_feature_availability_audit``. The step ran
# clean (returncode 0, all four promised tables present) and died reporting
#
#     Canonical normalization reported error 'invalid_registered_count'.
#
# The offending cell was one row of one emitted table: a reconciliation
# difference of -93458 written under a column called ``n``, which the host
# reads as a population count. ``NormalizationIssue`` carried the product and
# ``row[4].n`` the whole time; ``compare_validator_shadow_inputs`` collapsed
# the issues to a set of *codes* and dropped both. Finding it took a manual
# scan of four CSVs.


def _reconciliation_envelope(tmp_path, n_value: str):
    """Normalize the exact table shape the real step emitted."""

    from easyicu.research_agent.contracts.result_envelope import (
        normalize_step_result_shadow,
    )

    (tmp_path / "cohort_input_reconciliation.csv").write_text(
        "quantity,source,n,denominator_n,percent\n"
        "loaded_cohort_rows,COHORT_PARQUET,1000,1000,100.0\n"
        "observed_probe_denominator_minus_context_declared,"
        f"COHORT_PARQUET_vs_ResearchContext,{n_value},94458,-98.94\n",
        encoding="utf-8",
    )
    summary = {
        "status": "completed",
        "output_files": {
            "table:cohort_input_reconciliation": "cohort_input_reconciliation.csv"
        },
    }
    envelope = normalize_step_result_shadow(
        step_id="02_feature_availability_audit",
        step_summary=summary,
        output_dir=tmp_path,
        status="ok",
    )
    return summary, envelope


def test_a_normalization_error_names_the_product_and_the_cell(tmp_path) -> None:
    from easyicu.research_agent.audits.envelope_shadow import (
        compare_validator_shadow_inputs,
    )

    summary, envelope = _reconciliation_envelope(tmp_path, "-93458")
    comparison = compare_validator_shadow_inputs(
        step_summary=summary, envelope=envelope, current_status="ok"
    )
    details = [
        mismatch.detail
        for mismatch in comparison.mismatches
        if mismatch.code == "normalization_error"
    ]

    assert details, "the negative count did not reach the shadow comparison"
    detail = details[0]
    assert "invalid_registered_count" in detail
    assert (
        "table:cohort_input_reconciliation" in detail
    ), "the finding does not say which registered product carried the cell"
    assert (
        "row[1].n" in detail
    ), "the finding does not say which cell; the issue knew, the mismatch did not"


def test_a_clean_reconciliation_raises_no_normalization_error(tmp_path) -> None:
    """The same table with a plausible count must not be reported at all."""

    from easyicu.research_agent.audits.envelope_shadow import (
        compare_validator_shadow_inputs,
    )

    summary, envelope = _reconciliation_envelope(tmp_path, "93458")
    comparison = compare_validator_shadow_inputs(
        step_summary=summary, envelope=envelope, current_status="ok"
    )

    assert not [
        mismatch
        for mismatch in comparison.mismatches
        if mismatch.code == "normalization_error"
    ]
