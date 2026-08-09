"""Publication handoffs must not invite the writer to state p=0."""

from __future__ import annotations

import json

from easyicu.research_agent.reporting.p_values import (
    MIN_NONZERO_REPORTED_P,
    is_p_value_field,
    prepare_p_values_for_writer,
    publication_p_value,
    render_claim_value_for_writer,
)
from easyicu.research_agent.reporting.writer_evidence import (
    _render_writer_evidence_digest,
    _render_writer_evidence_digest_v2,
)


def test_zero_p_value_is_bounded_only_for_publication_handoff():
    raw = {"primary_p_value": 0.0, "odds_ratio": 1.6}

    prepared = prepare_p_values_for_writer(raw)

    assert raw["primary_p_value"] == 0.0
    assert prepared["primary_p_value"] == MIN_NONZERO_REPORTED_P
    assert prepared["primary_p_value_reporting"] == "p < 1e-300"
    assert prepared["primary_p_value_source_underflow"] is True


def test_nonzero_and_invalid_p_values_are_not_rewritten():
    assert publication_p_value(0.02) == (0.02, None, False)
    assert publication_p_value(-1.0) == (-1.0, None, False)
    assert publication_p_value(None) == (None, None, False)
    assert is_p_value_field("group_value") is False
    assert is_p_value_field("p_value_bounded") is False


def test_primary_writer_digest_never_exposes_numeric_zero_p_value():
    records = [
        {
            "step_id": "05_primary_model",
            "status": "ok",
            "step_summary": {
                "odds_ratio": 1.608,
                "p_value": 0.0,
            },
        }
    ]

    digest = _render_writer_evidence_digest(records)
    payload = json.loads(digest.splitlines()[1].strip())

    assert payload["p_value"] == MIN_NONZERO_REPORTED_P
    assert payload["p_value_reporting"] == "p < 1e-300"
    assert payload["p_value_source_underflow"] is True
    assert '"p_value": 0.0' not in digest


def test_secondary_writer_digest_uses_underflow_reporting():
    records = [
        {
            "step_id": "05_primary_model",
            "status": "ok",
            "step_summary": {"exposure_p_value": 0.0},
        }
    ]

    digest = _render_writer_evidence_digest_v2(records)

    assert (
        "exposure_p_value=1e-300 "
        "(reporting=p < 1e-300; source_underflow=true)"
    ) in digest
    assert "exposure_p_value=0.0" not in digest


def test_secondary_claim_renderer_preserves_ordinary_numbers():
    assert (
        render_claim_value_for_writer(
            source_field="odds_ratio",
            value=1.608,
            canonical=1.608,
        )
        == "1.608 (canonical=1.608)"
    )
