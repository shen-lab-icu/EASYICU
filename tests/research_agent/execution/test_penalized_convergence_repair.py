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


def _unproven_script() -> str:
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


def _proven_script() -> str:
    return """\
import scipy.optimize

model_id = "operational_proxy"
optimizer_result = scipy.optimize.minimize(objective, x0)
converged = bool(optimizer_result.success)
model_contract = {
    "model_id": model_id,
    "fit_method": "scipy_penalized_logistic",
    "penalized": True,
    "converged": converged,
}
"""


def test_penalized_convergence_contract_copies_proven_optimizer_status() -> None:
    repaired = patch_penalized_convergence_contract(
        _proven_script(),
        findings=[_finding()],
    )

    assert repaired != _proven_script()
    assert '"convergence_method": "optimizer_success"' in repaired
    assert '"optimizer_success": bool(converged)' in repaired
    assert repaired.count("optimizer_result.success") == 1
    ast.parse(repaired)


def test_penalized_convergence_contract_is_registered_deterministic_repair() -> None:
    repair = deterministic_contract_repair(
        code=_proven_script(),
        findings=[_finding()],
    )

    assert repair is not None
    repair_id, repaired = repair
    assert repair_id == "penalized_convergence_contract_v2"
    assert '"optimizer_success": bool(converged)' in repaired


def test_penalized_convergence_contract_rejects_ambiguous_or_unbound_target() -> None:
    ambiguous = _proven_script() + _proven_script().replace(
        "model_contract", "other_contract"
    )
    assert (
        patch_penalized_convergence_contract(
            ambiguous,
            findings=[_finding()],
        )
        == ambiguous
    )
    assert (
        patch_penalized_convergence_contract(
            _proven_script(),
            findings=[_finding("different_model")],
        )
        == _proven_script()
    )


def test_penalized_convergence_contract_rejects_iteration_and_literal_booleans() -> (
    None
):
    for code in (
        _unproven_script(),
        _unproven_script().replace(
            "bool(ridge.n_iter_[0] < max_iter)",
            "True",
        ),
        _proven_script()
        .replace(
            "import scipy.optimize\n\n",
            "",
        )
        .replace(
            "scipy.optimize.minimize(objective, x0)",
            "custom_optimizer(objective, x0)",
        ),
        _proven_script().replace(
            "model_contract = {\n",
            "converged = True\nmodel_contract = {\n",
        ),
        _proven_script().replace(
            "optimizer_result = scipy.optimize.minimize(objective, x0)",
            (
                "scipy.optimize.minimize = custom_optimizer\n"
                "optimizer_result = scipy.optimize.minimize(objective, x0)"
            ),
        ),
    ):
        assert (
            patch_penalized_convergence_contract(
                code,
                findings=[_finding()],
            )
            == code
        )


def test_penalized_convergence_contract_preserves_existing_authority_fields() -> None:
    code = _proven_script().replace(
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
