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
from types import SimpleNamespace

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


# ---- step-aware exemption for a supporting audit STEP (M3 subphenotype) -----
#
# The M3 block: the LLM tagged a lone audit panel with role="robustness" (a
# RESULT role), so the role-based exemption above did not fire and a
# SUPPLEMENTARY audit figure hard-failed the whole run. A supporting/audit STEP
# is exempt from the primary-result ">= 2 panels" rule regardless of the role
# label; a primary result step is NOT.

_AUDIT_STEP = SimpleNamespace(
    step_id="03_audit_panel",
    intent=(
        "Render an audit panel that summarises the analysis's robustness: cohort "
        "attrition, data completeness / missingness, measurement-process handling."
    ),
    method="visualization",
)
_PRIMARY_STEP = SimpleNamespace(
    step_id="01_phenotype_structure",
    intent="Render the phenotype structure figure showing cluster centroid profiles.",
    method="visualization",
)


def test_is_supporting_figure_step_true_for_audit_panel():
    assert FigureContractQualityValidator._is_supporting_figure_step(_AUDIT_STEP)


def test_is_supporting_figure_step_false_for_primary_result_step():
    assert not FigureContractQualityValidator._is_supporting_figure_step(_PRIMARY_STEP)
    assert not FigureContractQualityValidator._is_supporting_figure_step(None)


def test_audit_step_single_robustness_panel_is_exempt(tmp_path: Path):
    # Exact M3 shape: one panel, role="robustness" (a result role), on an audit
    # STEP -> must NOT be flagged, so a supplementary figure cannot nuke the run.
    path = _write_contract(
        tmp_path,
        "audit_panel",
        [{"panel_id": "A", "role": "robustness", "title": "Audit Panel"}],
        core_claim="Displays the step result using registered source data.",
    )
    findings = FigureContractQualityValidator().audit_contract_file(
        path, step=_AUDIT_STEP, manuscript_facing=True
    )
    assert not any(_PANEL_MSG in f.message for f in findings), [
        f.message for f in findings
    ]


def test_primary_step_single_robustness_panel_is_still_flagged(tmp_path: Path):
    # The same lone-robustness contract on a PRIMARY step keeps failing: the
    # step-aware exemption must not weaken the primary-figure guard.
    path = _write_contract(
        tmp_path,
        "primary_result",
        [{"panel_id": "A", "role": "robustness", "title": "Primary"}],
        core_claim="Primary robustness result.",
    )
    findings = FigureContractQualityValidator().audit_contract_file(
        path, step=_PRIMARY_STEP, manuscript_facing=True
    )
    assert any(_PANEL_MSG in f.message for f in findings), [f.message for f in findings]


# ---- contract-based exemption (the real call sites pass NO step) -----------


def test_audit_contract_exempt_via_figure_id_without_step(tmp_path: Path):
    # Exact M3 production path: figure_id='audit_panel', a lone role='robustness'
    # panel, and NO step threaded. Detection must work from the contract alone.
    path = _write_contract(
        tmp_path,
        "audit_panel",
        [{"panel_id": "A", "role": "robustness", "title": "Audit Panel"}],
        core_claim="Displays the step result using registered source data.",
    )
    findings = FigureContractQualityValidator().audit_contract_file(
        path, manuscript_facing=True
    )
    assert not any(_PANEL_MSG in f.message for f in findings), [
        f.message for f in findings
    ]


def test_primary_forest_still_flagged_without_step(tmp_path: Path):
    # A primary figure_id with a lone robustness panel and NO step is still
    # flagged: the contract-based exemption must not weaken the guard.
    path = _write_contract(
        tmp_path,
        "primary_forest",
        [{"panel_id": "A", "role": "robustness", "title": "Primary"}],
        core_claim="Primary adjusted effect result.",
    )
    findings = FigureContractQualityValidator().audit_contract_file(
        path, manuscript_facing=True
    )
    assert any(_PANEL_MSG in f.message for f in findings), [f.message for f in findings]
