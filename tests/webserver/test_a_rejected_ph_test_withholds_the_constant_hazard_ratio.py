"""A rejected PH test withholds the constant hazard ratio from the manuscript.

When the prespecified Schoenfeld test rejects proportional hazards the suite's
primary estimates are the interval-specific hazard ratios, and a question may
say outright "do not report a single constant HR".  The constant estimate still
sat at the summary's top level, in its runtime receipt and in the reporting
envelope, so every one of its digits was a registered leaf: a neutral Writer
sentence "The adjusted hazard ratio was …" bound and passed STRICT, and only a
prompt line stood in its way.  The suite now keeps it as a diagnostic row of
its Cox table and nowhere else, and the Writer digest offers no generic effect
headline beside the interval block.

Synthetic study and seeded synthetic rows only (renal replacement therapy and
90-day mortality); the crossing-hazard rows make the test reject.
"""

from __future__ import annotations

import copy
import json

import pandas as pd
import pytest

from easyicu.research_agent.authority.evidence_store import (
    EvidenceEnforcementError,
    EvidenceEnforcementMode,
    EvidenceStore,
)
from easyicu.research_agent.authority.survival_scientific_claims import (
    SurvivalReporting,
    derive_survival_claim_payloads,
)
from easyicu.research_agent.reporting.manuscript_post import bind_numeric_values
from easyicu.research_agent.reporting.writer_evidence import _render_writer_evidence_digest
from tests.support.survival_sealed import (
    run_signed_suite,
    sealed_survival,
    synthetic_crossing_hazard_rows,
    synthetic_survival_rows,
)

STEP = "primary_survival_suite"
EVIDENCE = "statistic_step_summary_primary_survival_suite"
SUMMARY_KEYS = ("hazard_ratio", "hazard_ratio_ci_low", "hazard_ratio_ci_high")
RECEIPT_KEYS = ("hazard_ratio", "ci_low", "ci_high")


def _run(tmp_path, rows):
    _context, authority = sealed_survival(tmp_path)
    out = tmp_path / "out"
    summary = json.loads(json.dumps(run_signed_suite(authority, rows, out)))
    cox = pd.read_csv(out / "landmark_cox_summary.csv")
    constant = cox.loc[cox["term"] == authority.derived_exposure_column].iloc[0]
    return summary, constant


def _bind(tmp_path, summary, sentence):
    run_dir = tmp_path / "run"
    source = run_dir / "steps" / STEP / "outputs" / "step_summary.json"
    source.parent.mkdir(parents=True)
    source.write_text(json.dumps(summary), encoding="utf-8")
    store = EvidenceStore(run_dir, enforcement_mode=EvidenceEnforcementMode.STRICT)
    store.register_file(
        kind="statistic", description="Signed survival suite summary", source_path=source,
        evidence_id=EVIDENCE, produced_by_step=STEP, producer="runner",
        generation_mode="deterministic_standard",
    )
    store.register_step_summary_numerics(step_id=STEP, evidence_id=EVIDENCE, summary=summary)
    records = [{"step_id": STEP, "status": "ok", "evidence_ids": [EVIDENCE]}]
    bound = store.bind_manuscript(f"## Results\n\n{sentence}\n", per_step_records=records)
    _, _binding, untraced = bind_numeric_values(bound, evidence=store, per_step_records=records)
    return untraced


def _neutral_sentence(constant) -> str:
    return (
        f"The adjusted hazard ratio was {constant['hazard_ratio']:.3f} "
        f"(95% CI, {constant['ci_low']:.3f} to {constant['ci_high']:.3f}) {{evidence:{EVIDENCE}}}."
    )


def test_a_rejected_test_keeps_the_constant_estimate_out_of_every_reportable_leaf(tmp_path):
    summary, constant = _run(tmp_path, synthetic_crossing_hazard_rows())
    reporting = summary["reportable_survival_results"]
    assert reporting["constant_hazard_ratio_authorized"] is False

    assert not set(SUMMARY_KEYS) & set(summary)
    assert not set(RECEIPT_KEYS) & set(summary["scientific_runtime_receipt"])
    assert "adjusted_hazard_ratio" not in reporting
    # It remains a diagnostic row of the suite's own Cox table.
    assert constant["hazard_ratio"] > 0

    payloads = derive_survival_claim_payloads(summary)
    assert all(payload["claim_id"] != "adjusted_hazard_ratio" for payload in payloads)
    intervals = [payload for payload in payloads if payload["claim_id"].startswith("interval_")]
    assert intervals and all(payload["analysis_role"] == "primary" for payload in intervals)


def test_a_neutral_constant_hazard_ratio_sentence_cannot_bind_after_a_rejected_test(tmp_path):
    summary, constant = _run(tmp_path, synthetic_crossing_hazard_rows())

    # STRICT refuses the manuscript: the withheld estimate is no bindable leaf.
    with pytest.raises(EvidenceEnforcementError, match="not traceable"):
        _bind(tmp_path, summary, _neutral_sentence(constant))


def test_an_unrejected_test_still_reports_and_binds_its_constant_estimate(tmp_path):
    summary, constant = _run(tmp_path, synthetic_survival_rows())
    reporting = summary["reportable_survival_results"]
    assert reporting["constant_hazard_ratio_authorized"] is True

    assert summary["hazard_ratio"] == constant["hazard_ratio"]
    assert reporting["adjusted_hazard_ratio"]["hazard_ratio"] == constant["hazard_ratio"]
    assert _bind(tmp_path, summary, _neutral_sentence(constant)) == []


def test_an_envelope_signed_before_the_fence_still_parses_and_claims_no_constant_estimate(tmp_path):
    summary, constant = _run(tmp_path, synthetic_crossing_hazard_rows())
    legacy = copy.deepcopy(summary)
    legacy["reportable_survival_results"]["adjusted_hazard_ratio"] = {
        "hazard_ratio": float(constant["hazard_ratio"]),
        "ci_low": float(constant["ci_low"]),
        "ci_high": float(constant["ci_high"]),
    }

    SurvivalReporting.model_validate(legacy["reportable_survival_results"])
    assert all(
        payload["claim_id"] != "adjusted_hazard_ratio"
        for payload in derive_survival_claim_payloads(legacy)
    )


def test_an_authorized_constant_estimate_must_be_reported(tmp_path):
    summary, _constant = _run(tmp_path, synthetic_survival_rows())
    forged = copy.deepcopy(summary["reportable_survival_results"])
    forged.pop("adjusted_hazard_ratio")

    with pytest.raises(ValueError, match="must be reported"):
        SurvivalReporting.model_validate(forged)


def _digest_row(tmp_path, summary) -> dict:
    digest = _render_writer_evidence_digest(
        [{"step_id": STEP, "status": "ok", "generation_mode": "deterministic_standard", "step_summary": summary}],
        run_dir=tmp_path, evidence=None,
    )
    lines = digest.splitlines()
    head = next(index for index, line in enumerate(lines) if line.startswith(f"- {STEP} ["))
    return json.loads(lines[head + 1])


@pytest.mark.parametrize("signed", ["after_the_fence", "before_the_fence"])
def test_the_writer_digest_offers_no_effect_headline_after_a_rejected_test(tmp_path, signed):
    """The interval and RMST block is the result, so no generic effect key is a headline.

    Before, the digest lifted one nested interval's bounds as ``ci_low`` and
    ``ci_high``, and from a summary signed before the fence the withheld
    constant estimate itself.
    """

    summary, constant = _run(tmp_path, synthetic_crossing_hazard_rows())
    if signed == "before_the_fence":
        summary = {
            **summary, "hazard_ratio": float(constant["hazard_ratio"]),
            "hazard_ratio_ci_low": float(constant["ci_low"]), "hazard_ratio_ci_high": float(constant["ci_high"]),
        }

    row = _digest_row(tmp_path, summary)

    assert not {"estimate", "effect_estimate", "hazard_ratio", "ci_low", "ci_high"} & set(row)
    assert "reportable_survival_results" in row


def test_an_unrejected_test_still_offers_its_constant_estimate_to_the_writer(tmp_path):
    summary, constant = _run(tmp_path, synthetic_survival_rows())

    assert _digest_row(tmp_path, summary)["hazard_ratio"] == pytest.approx(constant["hazard_ratio"])
