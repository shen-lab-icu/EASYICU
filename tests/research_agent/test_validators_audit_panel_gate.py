"""Regression: the manuscript-facing result-figure ">= 2 panels" gate must not
over-fire on single-panel audit/overview figures.

Root cause (2026-07-06, H1 fix5 + H2 fix2): both benchmark runs blocked
`manuscript_ready` because ``FigureContractQualityValidator`` flagged a single-panel
figure whose only panel had ``role="audit"`` (H1 ``reporting_followup_distribution_final``,
H2 ``probe_overview``). ``_is_result_like_contract`` returned True because a
result-role word ("distribution", "effect", ...) appeared as a substring of the
figure_id / core_claim, even though the structured panel role was "audit".

Fix: an all-supporting-role figure (every panel is audit/diagnostic/overview/…) is
not a manuscript-facing PRIMARY result figure and is exempt from the >= 2 panel
rule. Genuine single-panel result figures MUST still be flagged.
"""

from __future__ import annotations

import json
from pathlib import Path

from easyicu.research_agent.audits.validators import FigureContractQualityValidator

_PANEL_MSG = "only 1 panel"


def _write_contract(tmp_path: Path, figure_id: str, panels, **extra) -> Path:
    contract = {"figure_id": figure_id, "panels": panels, **extra}
    path = tmp_path / f"{figure_id}.figure_contract.json"
    path.write_text(json.dumps(contract), encoding="utf-8")
    return path


# ---- classifier unit tests -------------------------------------------------


def test_is_result_like_false_for_single_audit_panel():
    raw = {
        "figure_id": "reporting_followup_distribution_final",
        "core_claim": "Summarizes the follow-up time distribution.",
    }
    panels = [{"panel_id": "A", "role": "audit", "title": "Follow-up Distribution"}]
    assert FigureContractQualityValidator._is_result_like_contract(raw, panels) is False


def test_is_result_like_false_for_probe_overview_shape():
    # Exact shape of H2's probe_overview: audit role, "manuscript figure" in text.
    raw = {
        "figure_id": "probe_overview",
        "core_claim": "Probe Overview summarizes the planned manuscript figure "
        "from registered source data.",
        "archetype": "asymmetric_mixed_modality",
    }
    panels = [{"panel_id": "A", "role": "audit", "title": "Probe Overview"}]
    assert FigureContractQualityValidator._is_result_like_contract(raw, panels) is False


def test_is_result_like_true_for_single_result_panel():
    raw = {"figure_id": "primary_forest"}
    panels = [{"panel_id": "A", "role": "forest_odds_ratio", "title": "Primary OR"}]
    assert FigureContractQualityValidator._is_result_like_contract(raw, panels) is True


def test_is_result_like_true_for_mixed_result_and_audit_panels():
    # If ANY panel is a result role, the figure stays result-like.
    raw = {"figure_id": "primary_with_audit"}
    panels = [
        {"panel_id": "A", "role": "survival_effect", "title": "KM"},
        {"panel_id": "B", "role": "audit", "title": "Follow-up"},
    ]
    assert FigureContractQualityValidator._is_result_like_contract(raw, panels) is True


def test_is_result_like_true_for_blank_role_result_text():
    # Blank panel role -> cannot prove supporting -> fall back to text signal.
    raw = {"figure_id": "assoc_effect_plot", "core_claim": "association effect"}
    panels = [{"panel_id": "A", "role": "", "title": "Effect"}]
    assert FigureContractQualityValidator._is_result_like_contract(raw, panels) is True


# ---- end-to-end audit_contract_file tests ----------------------------------


def test_audit_contract_file_does_not_flag_single_audit_panel(tmp_path: Path):
    path = _write_contract(
        tmp_path,
        "reporting_followup_distribution_final",
        [{"panel_id": "A", "role": "audit", "title": "Follow-up Distribution"}],
        core_claim="Summarizes the follow-up time distribution.",
    )
    findings = FigureContractQualityValidator().audit_contract_file(
        path, manuscript_facing=True
    )
    assert not any(_PANEL_MSG in f.message for f in findings), [
        f.message for f in findings
    ]


def test_audit_contract_file_still_flags_single_result_panel(tmp_path: Path):
    # The gate's real protection must remain: a lone forest panel is flagged.
    path = _write_contract(
        tmp_path,
        "primary_forest",
        [{"panel_id": "A", "role": "forest_odds_ratio", "title": "Primary OR"}],
    )
    findings = FigureContractQualityValidator().audit_contract_file(
        path, manuscript_facing=True
    )
    assert any(_PANEL_MSG in f.message for f in findings), [f.message for f in findings]
