from __future__ import annotations

import ast

from easyicu.research_agent.repairs.model_contract import (
    patch_penalized_convergence_contract,
)
from easyicu.research_agent.repairs.source import deterministic_contract_repair
from easyicu.research_agent.schema import ValidationFinding


def _finding(model_id: str = "operational_proxy") -> ValidationFinding:
    return ValidationFinding(
        validator="primary_model_contract",
        severity="error",
        message="Penalized convergence evidence is incomplete.",
        detail={
            "issues": [
                {
                    "model_id": model_id,
                    "issue": "penalized_convergence_not_verified",
                }
            ]
        },
    )


def _script() -> str:
    return """\
model_id = "operational_proxy"
converged = bool(ridge.n_iter_[0] < max_iter)
model_contract = {
    "model_id": model_id,
    "fit_method": "sklearn_logistic_ridge_C=1.0_point_only",
    "penalized": True,
    "converged": converged,
}
"""


def test_penalized_convergence_contract_copies_existing_boolean_only() -> None:
    repaired = patch_penalized_convergence_contract(
        _script(),
        findings=[_finding()],
    )

    assert repaired != _script()
    assert '"convergence_method": "optimizer_success"' in repaired
    assert '"optimizer_success": bool(converged)' in repaired
    assert repaired.count("ridge.n_iter_") == 1
    ast.parse(repaired)


def test_penalized_convergence_contract_is_registered_deterministic_repair() -> None:
    repair = deterministic_contract_repair(
        code=_script(),
        findings=[_finding()],
    )

    assert repair is not None
    repair_id, repaired = repair
    assert repair_id == "penalized_convergence_contract_v1"
    assert '"optimizer_success": bool(converged)' in repaired


def test_penalized_convergence_contract_rejects_ambiguous_or_unbound_target() -> None:
    ambiguous = _script() + _script().replace("model_contract", "other_contract")
    assert (
        patch_penalized_convergence_contract(
            ambiguous,
            findings=[_finding()],
        )
        == ambiguous
    )
    assert (
        patch_penalized_convergence_contract(
            _script(),
            findings=[_finding("different_model")],
        )
        == _script()
    )


def test_penalized_convergence_contract_preserves_existing_authority_fields() -> None:
    code = _script().replace(
        '    "converged": converged,\n',
        '    "converged": converged,\n'
        '    "convergence_method": "kkt_residual",\n'
        '    "optimizer_success": optimizer_success,\n',
    )

    assert (
        patch_penalized_convergence_contract(
            code,
            findings=[_finding()],
        )
        == code
    )
