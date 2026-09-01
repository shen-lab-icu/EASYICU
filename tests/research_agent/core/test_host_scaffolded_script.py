"""The host's authority code must not be the model's to rewrite.

fresh17 step ``07_standard_robustness_sensitivity``. The executed script was
entirely host-generated. It could not satisfy the step's declared outputs, so
the host handed *its own* script to the model as a contract repair, and the
returned draft (quarantined, ``90ff3ec…``) changed two host-authored things:

    - plausibility_expected_columns = ('age',)      # sealed scope
    + plausibility_expected_columns = None          # re-derived at runtime

    - or declared_contracts_sha256 != '4d8bd1f3…'   # pin to step authority
    + if declared_contracts_sha256 != computed_contracts_sha256:

The mechanical preflight caught the first and blocked, correctly. Nothing
catches the second: ``source_contracts_sha256`` appears in the gates only as a
finding-detail field and is never read back out of the code, so a draft that
kept the scope tuple and dropped the pin would have executed with the authority
binding silently gone.

The original fresh17 digest remains historical evidence.  The receipt fragment
later became intentionally self-contained in ``04b19fa`` (it now carries its
own imports), so the live generator must not pretend to reproduce those older
bytes.  The golden below instead pins the current host scaffold, while the
region tests remain load-bearing for the authority boundary.
"""

from __future__ import annotations

import hashlib

from easyicu.research_agent.authority.plausibility import FlagOnlyPlausibilityScope
from easyicu.research_agent.contracts.host_scaffold import HostScaffoldedScript
from easyicu.research_agent.execution.runners.deterministic_robustness import (
    robustness_sensitivity_preflight_code,
    robustness_sensitivity_preflight_scaffold,
)
from easyicu.research_agent.schema import AnalysisStep

# The exact coordinates the fresh17 manifest recorded for that step.
_STEP_ID = "07_standard_robustness_sensitivity"
_CONTRACTS_SHA = "4d8bd1f3b81c0ad100bfc5b6f04f94acac7f74ba54ebf8f52d230fbae794c708"
_CURRENT_SCAFFOLD_SHA = (
    "8b6071825faebdb597fdaf4816f84280b2b5a1f942e372382fea8a92a4e498ae"
)


def _scope() -> FlagOnlyPlausibilityScope:
    return FlagOnlyPlausibilityScope(
        step_id=_STEP_ID,
        expected_columns=("age",),
        source_contracts_sha256=_CONTRACTS_SHA,
        authority_kind="resolved_raw_input_contracts",
    )


def _step() -> AnalysisStep:
    return AnalysisStep(step_id=_STEP_ID, intent="robustness", method="robustness")


def test_the_current_host_scaffold_has_an_explicit_byte_golden() -> None:
    """Intentional generator changes must update one reviewed byte golden."""

    code = robustness_sensitivity_preflight_code(_step(), plausibility_scope=_scope())

    assert hashlib.sha256(code.encode("utf-8")).hexdigest() == _CURRENT_SCAFFOLD_SHA


def test_the_assembled_scaffold_is_the_script() -> None:
    """A scaffold is never a second source of truth -- same bytes, with a seam."""

    scaffold = robustness_sensitivity_preflight_scaffold(
        _step(), plausibility_scope=_scope()
    )

    assert scaffold.assembled() == robustness_sensitivity_preflight_code(
        _step(), plausibility_scope=_scope()
    )


def test_the_authority_pin_and_sealed_scope_live_in_the_host_prologue() -> None:
    """Both things the model rewrote are host property, not body."""

    scaffold = robustness_sensitivity_preflight_scaffold(
        _step(), plausibility_scope=_scope()
    )

    assert _CONTRACTS_SHA in scaffold.prologue
    assert "plausibility_expected_columns = ('age',)" in scaffold.prologue
    assert _CONTRACTS_SHA not in scaffold.body
    assert "plausibility_audit" in scaffold.epilogue
    assert scaffold.body == "_run_robustness_preflight_from_env()"


def test_the_body_is_the_only_region_a_repair_may_replace() -> None:
    scaffold = robustness_sensitivity_preflight_scaffold(
        _step(), plausibility_scope=_scope()
    )

    replaced = scaffold.with_body("run_something_else()")

    assert replaced.body == "run_something_else()"
    assert replaced.prologue_sha256 == scaffold.prologue_sha256
    assert replaced.epilogue_sha256 == scaffold.epilogue_sha256
    assert _CONTRACTS_SHA in replaced.assembled()


def test_the_real_coder_draft_is_detected_as_rewritten() -> None:
    """The fresh17 draft, reduced to the two edits that matter.

    Reproducing the exact 137-line quarantined draft here would test the model,
    not the boundary; what matters is that a prologue the model touched at all
    stops being this scaffold's prologue.
    """

    scaffold = robustness_sensitivity_preflight_scaffold(
        _step(), plausibility_scope=_scope()
    )
    rewritten = scaffold.assembled().replace(
        "plausibility_expected_columns = ('age',)",
        "plausibility_expected_columns = None",
    )

    assert rewritten != scaffold.assembled()
    assert scaffold.host_regions_intact(rewritten) is False
    assert scaffold.body_of(rewritten) is None


def test_deleting_only_the_authority_pin_is_also_detected() -> None:
    """The edit nothing else in the system notices."""

    scaffold = robustness_sensitivity_preflight_scaffold(
        _step(), plausibility_scope=_scope()
    )
    depinned = scaffold.assembled().replace(f"!= {_CONTRACTS_SHA!r}", "!= None")

    assert _CONTRACTS_SHA not in depinned
    assert scaffold.host_regions_intact(depinned) is False


def test_an_untouched_script_round_trips() -> None:
    scaffold = robustness_sensitivity_preflight_scaffold(
        _step(), plausibility_scope=_scope()
    )

    assert scaffold.body_of(scaffold.assembled()) == scaffold.body
    assert scaffold.host_regions_intact(scaffold.assembled()) is True


def test_a_rewritten_prologue_is_never_silently_wrapped() -> None:
    """Wrapping would run the host's audit and the model's rewrite side by side.

    So ``body_of`` answers ``None`` rather than handing back something that
    could be re-wrapped, and the caller has to decide deliberately.
    """

    scaffold = robustness_sensitivity_preflight_scaffold(
        _step(), plausibility_scope=_scope()
    )
    whole_rewrite = "import json\nprint('mine')\n"

    assert scaffold.body_of(whole_rewrite) is None


def test_a_scaffold_without_host_regions_is_all_body() -> None:
    """No plausibility scope means nothing is host property in this script."""

    scaffold = robustness_sensitivity_preflight_scaffold()

    assert scaffold.prologue
    assert scaffold.epilogue == ""


def test_empty_scaffold_assembles_to_nothing() -> None:
    assert HostScaffoldedScript().assembled() == ""
    assert HostScaffoldedScript(body="x = 1").assembled() == "x = 1\n"
