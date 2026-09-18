"""Checked-in Tool Card grants: loader validity + live reproducibility.

The four ``methods/tool_cards/*.card.json`` envelopes are the first
checked-in promotion set (lasso / SHAP / PSM / RCS). This file proves the loader
registers exactly them, refuses tampering, and — for PSM — that the
envelope's portable synthetic-origin digest reproduces from the documented
issuance fixture (a mirror of the kernel's own synthetic fixture, kept
inline so this test needs no cross-test imports).

Re-issuance procedure: rebuild the envelope with the documented ceremony
(fixtures referenced by each card's population assumption), replace the
JSON, and these tests re-verify.  The issuance script itself is not
checked in; the ceremony is fully described by the kernel determinism
tests plus the fixture below.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from easyicu.research_agent.methods import propensity_weighting as pw
from easyicu.research_agent.methods.tool_card import (
    ToolCard,
    synthetic_origin_sha256,
    tool_card_sha256,
)
from easyicu.research_agent.planning import capability_registry as registry_module

PSM_EXECUTOR_MODULE = "easyicu.research_agent.methods.propensity_weighting"
LASSO_EXECUTOR_MODULE = "easyicu.research_agent.methods.lasso_selection"
SHAP_EXECUTOR_MODULE = "easyicu.research_agent.methods.shap_attribution"
RCS_EXECUTOR_MODULE = "easyicu.research_agent.methods.rcs_dose_response"
EXPECTED_MODULES = (
    LASSO_EXECUTOR_MODULE,
    SHAP_EXECUTOR_MODULE,
    PSM_EXECUTOR_MODULE,
    RCS_EXECUTOR_MODULE,
)


@pytest.fixture
def _isolated_registry():
    saved = dict(registry_module._TOOL_CARD_GRANTS)
    saved_flag = registry_module._CHECKED_IN_TOOL_CARDS_LOADED
    registry_module._TOOL_CARD_GRANTS.clear()
    registry_module._CHECKED_IN_TOOL_CARDS_LOADED = False
    try:
        yield
    finally:
        registry_module._TOOL_CARD_GRANTS.clear()
        registry_module._TOOL_CARD_GRANTS.update(saved)
        registry_module._CHECKED_IN_TOOL_CARDS_LOADED = saved_flag


def _psm_issuance_fixture():
    """Mirror of the PSM card issuance fixture (seed 7, n_t=150, n_c=600)."""

    rng = np.random.default_rng(7)
    n_t, n_c = 150, 600
    treated_x = np.column_stack(
        [rng.normal(0.8, 1.0, n_t), rng.normal(-0.5, 1.0, n_t)]
    )
    control_x = np.column_stack(
        [rng.normal(0.0, 1.0, n_c), rng.normal(0.3, 1.0, n_c)]
    )
    covariates = np.vstack([treated_x, control_x])
    treated = np.array([1] * n_t + [0] * n_c)
    return treated, covariates


def _read_envelope(tool_name: str) -> dict:
    root = (
        Path(registry_module.__file__).resolve().parent.parent
        / "methods"
        / "tool_cards"
    )
    return json.loads((root / f"{tool_name}.card.json").read_text(encoding="utf-8"))


def _psm_card_origin_digest(treated, covariates, matched, iptw) -> str:
    matching_output = {k: v for k, v in matched.to_json().items() if k != "digest"}
    iptw_output = {k: v for k, v in iptw.to_json().items() if k != "digest"}
    return synthetic_origin_sha256(
        {
            "kind": "psm_confounding_adjustment_origin/1",
            "inputs": {
                "treatment": treated.tolist(),
                "covariates": covariates.tolist(),
            },
            "matching": matching_output,
            "iptw": iptw_output,
        }
    )


def test_checked_in_cards_register_all_verified_grants(_isolated_registry) -> None:
    records = registry_module.load_checked_in_tool_cards()
    assert sorted(records) == sorted(EXPECTED_MODULES)
    for module in EXPECTED_MODULES:
        assert (
            registry_module.granted_tool_identity(module) == "verified_tool"
        )


def test_checked_in_psm_origin_reproduces_live(_isolated_registry) -> None:
    treated, covariates = _psm_issuance_fixture()
    ps = pw.estimate_propensity_scores(
        treated, covariates, covariate_names=["x1", "x2"]
    )
    matched = pw.match_nearest_neighbor(
        treated, covariates, ps, covariate_names=["x1", "x2"]
    )
    iptw = pw.compute_iptw_weights(treated, ps)
    suite_digest = _psm_card_origin_digest(treated, covariates, matched, iptw)
    envelope = _read_envelope("psm_confounding_adjustment")
    assert suite_digest == envelope["card"]["origin_output_sha256"]
    card = ToolCard.model_validate(envelope["card"], strict=True)
    assert envelope["decision"]["card_sha256"] == tool_card_sha256(card)


def test_psm_origin_moves_when_weights_move(_isolated_registry) -> None:
    """Review finding: an origin over matching alone lets weight changes
    pass silently. The suite digest must move with either output."""

    treated, covariates = _psm_issuance_fixture()
    ps = pw.estimate_propensity_scores(
        treated, covariates, covariate_names=["x1", "x2"]
    )
    matched = pw.match_nearest_neighbor(
        treated, covariates, ps, covariate_names=["x1", "x2"]
    )
    plain = pw.compute_iptw_weights(treated, ps)
    truncated = pw.compute_iptw_weights(
        treated, ps, truncate_quantiles=(0.05, 0.95)
    )
    assert truncated.weights != plain.weights
    assert pw.adjustment_suite_digest(
        matched=matched, iptw=truncated
    ) != pw.adjustment_suite_digest(matched=matched, iptw=plain)
    assert _psm_card_origin_digest(
        treated, covariates, matched, truncated
    ) != _psm_card_origin_digest(treated, covariates, matched, plain)


def test_loader_rejects_tampered_envelope(_isolated_registry, tmp_path) -> None:
    src = (
        Path(registry_module.__file__).resolve().parent.parent
        / "methods"
        / "tool_cards"
    )
    dirty = tmp_path / "cards"
    shutil.copytree(src, dirty)
    target = dirty / "psm_confounding_adjustment.card.json"
    envelope = json.loads(target.read_text(encoding="utf-8"))
    digest = envelope["decision"]["card_sha256"]
    envelope["decision"]["card_sha256"] = (
        ("0" if digest[0] != "0" else "1") + digest[1:]
    )
    target.write_text(json.dumps(envelope), encoding="utf-8")
    with pytest.raises(ValueError, match="digest"):
        registry_module.load_checked_in_tool_cards(directory=dirty)


def test_loader_rejects_missing_and_empty_directories(
    _isolated_registry, tmp_path
) -> None:
    with pytest.raises(ValueError, match="missing"):
        registry_module.load_checked_in_tool_cards(
            directory=tmp_path / "absent"
        )
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="no tool card grant files"):
        registry_module.load_checked_in_tool_cards(directory=empty)


def test_lazy_grant_lookup_loads_once(_isolated_registry) -> None:
    assert registry_module._CHECKED_IN_TOOL_CARDS_LOADED is False
    assert (
        registry_module.granted_tool_identity(PSM_EXECUTOR_MODULE)
        == "verified_tool"
    )
    assert registry_module._CHECKED_IN_TOOL_CARDS_LOADED is True
    assert registry_module.granted_tool_identity("execution.runners.unknown") is None


def test_loader_rejects_repointed_executor(_isolated_registry, tmp_path) -> None:
    """Review finding: editing only the envelope's executor field must not
    promote an unrelated executor (e.g. a survival runner)."""

    import shutil

    src = (
        Path(registry_module.__file__).resolve().parent.parent
        / "methods"
        / "tool_cards"
    )
    dirty = tmp_path / "cards"
    shutil.copytree(src, dirty)
    target = dirty / "psm_confounding_adjustment.card.json"
    envelope = json.loads(target.read_text(encoding="utf-8"))
    envelope["executor_module"] = "execution.runners.survival_primary_executor"
    target.write_text(json.dumps(envelope), encoding="utf-8")
    with pytest.raises(ValueError, match="unrelated executor"):
        registry_module.load_checked_in_tool_cards(directory=dirty)
    assert (
        registry_module.granted_tool_identity(
            "execution.runners.survival_primary_executor"
        )
        is None
    )
