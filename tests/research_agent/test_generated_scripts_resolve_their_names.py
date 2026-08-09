"""A generated script must not depend on names nobody bound.

fresh22 claimed ``06_primary_adjusted_association`` with a deterministic owner
-- the first time the primary result had one in a real run -- and the script
the host wrote for it died on line 42 with ``NameError: name 'hashlib' is not
defined``.  The shared plausibility-receipt fragment uses ``hashlib``, and it
is spliced into five executors' prologues.  Four of them import ``hashlib``
for their own reasons.  The fifth did not.

Compiling the script does not catch this: a module-level ``NameError`` is a
runtime event, so ``compile()`` is happy and the container is not.  The step
then spent a runtime repair and a post-mutation concept repair on a missing
import, exhausted its two-repair budget and was blocked -- so this also cost
the run its whole repair allowance for a defect no model wrote.

The check that would have caught it lived here, as a test helper, and only
ever ran against the host's own fragments.  On 2026-07-30 the H1 canary proved
what that omission costs on the model's side: ``02_table_one`` died on
``NameError: name 'manifest' is not defined``, and the repair that replaced
the draft reintroduced the same class with ``table_one_spec``.  Two attempts,
one step, one defect class, and no gate looked.  The helper now lives in the
gate; this module imports the one implementation rather than keeping a second.
"""

from __future__ import annotations

import ast
from typing import Callable, Dict, List

import pytest

from easyicu.research_agent.authority.plausibility import FlagOnlyPlausibilityScope
from easyicu.research_agent.execution.runners.plausibility_receipt import (
    render_standard_plausibility_receipt_code,
)
from easyicu.research_agent.gates.preflight import (
    audit_mechanical_code_contracts,
    module_level_unbound_names,
    unresolvable_names,
)
from easyicu.research_agent.schema import AnalysisStep


def _unbound(source: str) -> List[str]:
    """Just the names, for assertions that do not care about the line."""

    return [name for name, _ in module_level_unbound_names(ast.parse(source))]


def _scope() -> FlagOnlyPlausibilityScope:
    return FlagOnlyPlausibilityScope(
        step_id="06_primary_adjusted_association",
        expected_columns=("age",),
        source_contracts_sha256="4d8bd1f3" + "0" * 56,
        authority_kind="raw_input_contract",
    )


def _step() -> AnalysisStep:
    return AnalysisStep.model_validate(
        {
            "step_id": "02_table_one",
            "method": "descriptive_summary",
            "intent": "Summarise the analysis cohort.",
            "inputs": ["artifact:analysis_cohort"],
            "expected_outputs": ["table:table_one"],
        }
    )


def _unbound_findings(source: str) -> List[dict]:
    return [
        finding.detail or {}
        for finding in audit_mechanical_code_contracts(source, _step())
        if (finding.detail or {}).get("reason") == "unresolvable_name"
    ]


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
    assert _unbound(source) == ["frame"]


def test_hashlib_specifically_is_bound_by_the_fragment() -> None:
    """The exact name the real run died on, so a regression is unambiguous."""

    source = render_standard_plausibility_receipt_code(_scope(), frame_name="frame")

    assert "hashlib.sha256(" in source
    assert "import hashlib" in source
    assert "hashlib" not in _unbound(source)


def _executor_scripts() -> Dict[str, Callable[[], str]]:
    """Every generator whose product is a whole standalone sandbox script."""

    from easyicu.research_agent.execution.runners.adjusted_association_executor import (
        adjusted_association_executor_code,
    )
    from easyicu.research_agent.execution.runners.deterministic_missingness import (
        missingness_measurement_audit_code,
    )

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
                    "model_terms": [
                        {
                            "name": "sep3_sofa2_max",
                            "role": "exposure",
                            "coding": "binary",
                            "levels": ["0", "1"],
                            "reference_level": "0",
                            "transform": "treatment_contrast",
                        },
                        {
                            "name": "age",
                            "role": "covariate",
                            "coding": "continuous",
                            "transform": "identity",
                        },
                        {
                            "name": "sex",
                            "role": "covariate",
                            "coding": "binary",
                            "levels": ["Female", "Male"],
                            "reference_level": "Female",
                            "transform": "treatment_contrast",
                        },
                        {
                            "name": "charlson_first",
                            "role": "covariate",
                            "coding": "continuous",
                            "transform": "identity",
                        },
                    ],
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

    assert unresolvable_names(ast.parse(source)) == []


# ---------------------------------------------------------------------------
# The model's script, not only the host's fragments.


def test_the_gate_reports_the_name_and_the_line_the_step_would_die_on() -> None:
    """The shape the H1 repair draft died on, reduced to its essentials.

    ``table_one_spec`` is read four times and stored nowhere.  A repair prompt
    that says only "the script failed" makes the model hunt; the finding has
    to name what to bind and where it is read.
    """

    source = (
        "from easyicu.research_agent.methods.table_one import build_grouped_table_one\n"
        "\n"
        "table_one = build_grouped_table_one(analysis, table_one_spec)\n"
    )

    details = _unbound_findings(source)

    assert len(details) == 1
    assert details[0]["names"] == [
        {"name": "analysis", "line": 3},
        {"name": "table_one_spec", "line": 3},
    ]


def test_the_message_carries_the_name_so_a_repair_prompt_can_use_it() -> None:
    """Findings become repair prompts; a reason code alone is not actionable."""

    source = "value = missing_binding + 1\n"

    messages = [
        finding.message
        for finding in audit_mechanical_code_contracts(source, _step())
        if (finding.detail or {}).get("reason") == "unresolvable_name"
    ]

    assert len(messages) == 1
    assert "missing_binding" in messages[0]
    assert "line 1" in messages[0]


def test_the_gate_refuses_the_script_rather_than_warning_about_it() -> None:
    """A NameError is certain, not advisory: the step cannot run at all."""

    findings = [
        finding
        for finding in audit_mechanical_code_contracts("x = nope\n", _step())
        if (finding.detail or {}).get("reason") == "unresolvable_name"
    ]

    assert [finding.severity for finding in findings] == ["error"]
    assert [finding.validator for finding in findings] == ["mechanical_code_preflight"]


def test_every_unbound_name_is_reported_not_just_the_first() -> None:
    """fresh22 died on ``hashlib`` and would then have died on ``pd``.

    Reporting one at a time turns one repair into two, and the step has a
    repair budget of two.
    """

    source = "digest = hashlib.sha256(b'')\nframe = pd.DataFrame()\n"

    details = _unbound_findings(source)

    assert details[0]["names"] == [
        {"name": "hashlib", "line": 1},
        {"name": "pd", "line": 2},
    ]

    # The message, not the detail, is what reaches a repair prompt: asserting
    # only the detail let a mutation that truncated the message survive.
    message = next(
        finding.message
        for finding in audit_mechanical_code_contracts(source, _step())
        if (finding.detail or {}).get("reason") == "unresolvable_name"
    )
    assert "hashlib (line 1)" in message
    assert "pd (line 2)" in message


def test_a_name_bound_anywhere_in_the_module_is_not_reported() -> None:
    """The class is "nobody binds it", not "bound in the wrong place"."""

    source = "spec = {'a': 1}\nresult = spec['a']\n"

    assert _unbound_findings(source) == []


def test_a_module_level_read_before_its_assignment_is_left_to_its_owner() -> None:
    """Deliberate boundary, not an oversight.

    ``_local_read_before_assignment_findings`` already walks ``tree.body`` and
    owns the ordering defect.  Reporting it here too would put one rule in two
    places, which is the failure that made the cohort ledger drift.
    """

    source = "result = spec\nspec = {'a': 1}\n"

    assert _unbound_findings(source) == []

    reasons = {
        (finding.detail or {}).get("reason")
        for finding in audit_mechanical_code_contracts(source, _step())
    }
    assert "local_read_before_assignment" in reasons or any(
        reason and "before" in str(reason) for reason in reasons
    ), f"the ordering owner must still fire; saw {sorted(map(str, reasons))}"


def test_a_local_of_one_function_is_not_visible_in_another() -> None:
    """canary4's death, reduced.

    This test previously asserted the opposite -- that a read inside a function
    body is never reported -- on the reasoning that such a name might be bound
    by the time the function is called.  On 2026-07-30 that narrowing let
    ``predicate_flow`` through: a local of ``validate_receipt``, read by
    ``main`` as a global, killing step 01 and the three steps behind it.

    The whole-program version would have missed it too, because it counted the
    sibling function's local as a binding.  Only the scope chain answers it.
    """

    source = (
        "def build():\n"
        "    flow = compute()\n"
        "    return flow\n"
        "\n"
        "def main():\n"
        "    return len(flow)\n"
    )

    details = _unbound_findings(source)

    assert len(details) == 1
    names = {item["name"] for item in details[0]["names"]}
    assert "flow" in names


def test_a_closure_reading_its_enclosing_scope_is_not_reported() -> None:
    """The legitimate shape the scope chain must keep admitting."""

    source = (
        "def outer():\n"
        "    total = 1\n"
        "    def inner():\n"
        "        return total\n"
        "    return inner()\n"
    )

    assert _unbound_findings(source) == []


def test_a_function_reading_a_module_level_name_is_not_reported() -> None:
    """The commonest shape in every generated script."""

    source = (
        "import os\n\nOUT = os.environ['STEP_OUT_DIR']\n\ndef main():\n    return OUT\n"
    )

    assert _unbound_findings(source) == []


def test_a_global_declaration_binds_the_name_for_the_reader() -> None:
    """``global`` is a binding statement; treating it otherwise is a false flag."""

    source = (
        "def setup():\n"
        "    global cache\n"
        "    cache = {}\n"
        "\n"
        "def main():\n"
        "    setup()\n"
        "    return cache\n"
    )

    assert _unbound_findings(source) == []


def test_builtins_and_module_dunders_are_bound() -> None:
    """The check must not invent failures out of the language itself."""

    source = "size = len([1, 2])\nwhere = __file__\ntag = str(size)\n"

    assert _unbound_findings(source) == []


def test_a_comprehension_target_binds_inside_its_own_expression() -> None:
    """Module-level comprehensions run on import; their targets are theirs."""

    source = "rows = [value * 2 for value in range(3)]\n"

    assert _unbound_findings(source) == []


def test_an_except_alias_and_a_with_target_count_as_bindings() -> None:
    """Both bind by statement, not by assignment, and both appear in scripts."""

    source = (
        "import pathlib\n"
        "try:\n"
        "    handle = pathlib.Path('x').open()\n"
        "except OSError as error:\n"
        "    message = str(error)\n"
        "with pathlib.Path('y').open() as stream:\n"
        "    body = stream.read()\n"
    )

    assert _unbound_findings(source) == []


def test_a_match_statement_makes_the_check_abstain_instead_of_guessing() -> None:
    """``case {"a": rest}`` binds ``rest`` with no ``Name`` store to see.

    Nothing in the 408 recorded generated scripts uses ``match``, so modelling
    its capture forms would be code that never runs.  Abstaining costs a
    missed catch; guessing costs a healthy step a repair, which is worse.
    """

    source = (
        "import json\n"
        "payload = json.loads('{}')\n"
        "match payload:\n"
        "    case {'a': found}:\n"
        "        answer = found\n"
        "    case _:\n"
        "        answer = None\n"
        "tail = never_bound_anywhere\n"
    )

    assert unresolvable_names(ast.parse(source)) == []


def test_the_check_runs_inside_the_mechanical_preflight_both_paths_use() -> None:
    """Wiring, not just the helper.

    Generation and repair both reach ``audit_mechanical_code_contracts``; the
    H1 canary proved the repair draft reintroduces this class, so a helper
    nobody calls from there is worth nothing.
    """

    reasons = [
        (finding.detail or {}).get("reason")
        for finding in audit_mechanical_code_contracts("x = never_bound\n", _step())
    ]

    assert "unresolvable_name" in reasons
