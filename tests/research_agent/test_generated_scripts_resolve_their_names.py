"""A host-generated script must not depend on names nobody bound.

fresh22 claimed ``06_primary_adjusted_association`` with a deterministic owner
-- the first time the primary result had one in a real run -- and the script
the host wrote for it died on line 42 with ``NameError: name 'hashlib' is not
defined``.  The shared plausibility-receipt fragment uses ``hashlib``, and it
is spliced into five executors' prologues.  Four of them import ``hashlib``
for their own reasons.  The fifth did not.

Compiling the script does not catch this: a module-level ``NameError`` is a
runtime event, so ``compile()`` is happy and the container is not.  What
follows is the cheapest check that would have: walk the generated module and
require every module-level name it loads to be bound somewhere in it, or be a
builtin.

The step then spent a runtime repair and a post-mutation concept repair on a
missing import, exhausted its two-repair budget and was blocked -- so this
also cost the run its whole repair allowance for a defect no model wrote.
"""

from __future__ import annotations

import ast
import builtins
from typing import Callable, Dict, List

import pytest

from easyicu.research_agent.authority.plausibility import FlagOnlyPlausibilityScope
from easyicu.research_agent.execution.runners.plausibility_receipt import (
    render_standard_plausibility_receipt_code,
)


def _module_level_unbound_names(source: str) -> List[str]:
    """Names loaded at module level that nothing in the module binds."""

    tree = ast.parse(source)
    bound = set(dir(builtins)) | {"__name__", "__file__", "__doc__"}
    loaded: Dict[str, int] = {}

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                bound.add(alias.asname or alias.name.split(".", 1)[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                bound.add(alias.asname or alias.name)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(node.name)
        elif isinstance(node, ast.arg):
            bound.add(node.arg)
        elif isinstance(node, ast.Name):
            if isinstance(node.ctx, (ast.Store, ast.Del)):
                bound.add(node.id)
            else:
                loaded.setdefault(node.id, node.lineno)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bound.add(node.name)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            bound.update(node.names)
        elif isinstance(node, ast.comprehension):
            for target in ast.walk(node.target):
                if isinstance(target, ast.Name):
                    bound.add(target.id)

    return sorted(name for name in loaded if name not in bound)


def _scope() -> FlagOnlyPlausibilityScope:
    return FlagOnlyPlausibilityScope(
        step_id="06_primary_adjusted_association",
        expected_columns=("age",),
        source_contracts_sha256="4d8bd1f3" + "0" * 56,
        authority_kind="raw_input_contract",
    )


def test_the_fragment_expects_exactly_one_name_from_its_caller() -> None:
    """The frame, and nothing else.

    ``frame_name`` is the declared parameter: the caller has already loaded
    its cohort and passes the variable holding it, so that name is the one
    legitimate expectation this fragment places on a prologue.  Every other
    name it uses is its own to bind -- which is the rule that ``hashlib`` and
    ``pandas`` were quietly breaking.
    """

    source = render_standard_plausibility_receipt_code(_scope(), frame_name="frame")

    assert source.strip(), "the fixture must produce a non-empty fragment"
    assert _module_level_unbound_names(source) == ["frame"]


def test_hashlib_specifically_is_bound_by_the_fragment() -> None:
    """The exact name the real run died on, so a regression is unambiguous."""

    source = render_standard_plausibility_receipt_code(_scope(), frame_name="frame")

    assert "hashlib.sha256(" in source
    assert "import hashlib" in source
    assert "hashlib" not in _module_level_unbound_names(source)


def _executor_scripts() -> Dict[str, Callable[[], str]]:
    """Every generator whose product is a whole standalone sandbox script."""

    from easyicu.research_agent.execution.runners.adjusted_association_executor import (
        adjusted_association_executor_code,
    )
    from easyicu.research_agent.execution.runners.deterministic_missingness import (
        missingness_measurement_audit_code,
    )
    from easyicu.research_agent.schema import AnalysisStep

    association_step = AnalysisStep.model_validate(
        {
            "step_id": "06_primary_adjusted_association",
            "method": "adjusted_association_models",
            "intent": "Estimate the adjusted association.",
            "inputs": [
                "artifact:analysis_cohort",
                "sep3_sofa2_max",
                "death",
                "age",
                "sex",
                "charlson_first",
            ],
            "expected_outputs": ["table:adjusted_association_estimates"],
            "model_requirements": [
                {
                    "requirement_id": "primary_full_cohort_logistic",
                    "outcome": "death",
                    "outcome_type": "binary",
                    "method_family": "logistic_regression",
                    "exposure_source": "sep3_sofa2_max",
                    "analysis_role": "primary",
                    "analysis_set": "source_aware",
                    "required_for_step_success": True,
                    "covariates": ["age", "sex", "charlson_first"],
                }
            ],
        }
    )
    audit_step = AnalysisStep.model_validate(
        {
            "step_id": "05_missingness_event_timing_audit",
            "method": "measurement_missingness_audit",
            "intent": "Count how often each concept was measured.",
            "inputs": ["artifact:analysis_cohort", "lact_max"],
            "expected_outputs": ["table:missingness_measurement_audit"],
            "measurement_audit_spec": {
                "products": [
                    {
                        "product_id": "missingness_measurement_audit",
                        "audit": "measurement_missingness",
                    }
                ]
            },
        }
    )
    return {
        "adjusted_association": lambda: adjusted_association_executor_code(
            association_step,
            plausibility_scope=_scope(),
        ),
        "missingness_audit": lambda: missingness_measurement_audit_code(audit_step),
    }


@pytest.mark.parametrize("name", sorted(_executor_scripts()))
def test_every_generated_script_binds_every_name_it_uses(name: str) -> None:
    """The whole script, not just the fragment: this is what Docker runs."""

    source = _executor_scripts()[name]()
    compile(source, f"<{name}>", "exec")

    assert _module_level_unbound_names(source) == []
