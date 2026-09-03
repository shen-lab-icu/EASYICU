"""Six steps died calling a host function with a keyword it does not have.

``gates/preflight.py`` already owns the machinery to catch this -- scope
tracking, rebinding detection, positional and keyword rules -- driven by
``_HOST_HELPER_CALL_CONTRACTS``, a hand-transcribed table of four helpers whose
own comment records the cost of transcription:

    "a drifted copy is how this registry has produced wrong blocks before"

MEASURED over 1,068 recorded step logs: six steps died on
``TypeError: <helper>() got an unexpected keyword argument``, and every one was
a host-owned function absent from that table --

    run_adjusted_association_figure(dpi=...)
    run_adjusted_association_from_env(model_id=...)
    run_adjusted_association_from_env(fit_kwargs=...)
    _run_robustness_preflight_from_env(primary_exposure=...)   x2
    strict_numeric_input(name=...)

The most recent, on 2026-08-04, killed m1's ``07_adjusted_association_figure``
two code repairs deep with seven provider calls still unspent -- the step
between a nine-step-green run and its first manuscript. And ``dpi`` is a real
parameter: of ``save_publication_figure``, which the Coder prompt names two
paragraphs from the sentence that mentions it. The model transposed a
documented keyword onto the wrong callee. No prompt edit reliably prevents
that; comparing the call against the callee's signature catches it exactly.

So the contract is read from the function instead of copied beside it. Only the
unknown-keyword half is derived -- which parameters a given step MUST pass is a
scientific decision a signature does not encode, and stays with the hand table
where one exists. A callee taking ``**kwargs`` is skipped: nothing can be
unexpected there, which is why ``save_publication_figure`` itself is untouched
and ``dpi=`` on it stays legal.
"""

from __future__ import annotations

import ast
import inspect
import pathlib

import pytest

from easyicu.research_agent.gates.preflight import (
    _HOST_HELPER_CALL_CONTRACTS,
    _SIGNATURE_DERIVED_HOST_HELPERS,
    _host_helper_call_signature_findings,
)
from easyicu.research_agent.schema import AnalysisStep

_CORPUS = pathlib.Path("/Volumes/外置硬盘/easyicu_data/canonical9_runs")


def _step() -> AnalysisStep:
    return AnalysisStep(
        step_id="07_adjusted_association_figure",
        method="forest_plot",
        intent="Render the adjusted association.",
        inputs=["table:adjusted_association_estimates"],
        expected_outputs=["figure:adjusted_association"],
    )


def _findings(source: str):
    return _host_helper_call_signature_findings(ast.parse(source), _step())


_THE_CALL_THAT_KILLED_M1 = """
from easyicu.research_agent.execution.runners.adjusted_association_figure_executor import (
    run_adjusted_association_figure,
)

run_adjusted_association_figure(
    out_dir=OUT,
    run_dir=RUN,
    resolved_inputs=INPUTS,
    step_id="07_adjusted_association_figure",
    figure_product="adjusted_association",
    dpi=300,
)
"""


def test_the_call_that_killed_m1_is_refused_before_it_runs():
    findings = _findings(_THE_CALL_THAT_KILLED_M1)

    assert findings, "the unknown keyword reached execution"
    detail = findings[0].detail
    assert "dpi" in str(detail) or "dpi" in str(findings[0].message), detail


def test_the_same_call_without_the_stray_keyword_is_allowed():
    """The check must not block the call the host actually wants made."""

    findings = _findings(_THE_CALL_THAT_KILLED_M1.replace("    dpi=300,\n", ""))

    assert findings == [], findings


def test_every_derived_contract_matches_its_functions_real_signature():
    """The property transcription cannot hold: the table IS the signature."""

    import importlib

    for module_name, symbol in _SIGNATURE_DERIVED_HOST_HELPERS:
        contract = _HOST_HELPER_CALL_CONTRACTS.get((module_name, symbol))
        if contract is None:
            continue  # uninspectable helpers are skipped, never guessed
        function = getattr(importlib.import_module(module_name), symbol)
        signature = inspect.signature(function)
        expected = tuple(
            parameter.name
            for parameter in signature.parameters.values()
            if parameter.kind
            in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            )
        )
        assert tuple(contract["allowed_keywords"]) == expected, (module_name, symbol)


def test_a_helper_taking_kwargs_is_left_alone():
    """``save_publication_figure`` accepts ``**legacy_kwargs``.

    It is the helper the prompt's ``dpi=``/``formats=`` sentence is actually
    about, it really does take both, and nothing may be unexpected on a callee
    that accepts anything.
    """

    from easyicu.research_agent.figures.publication import save_publication_figure

    parameters = inspect.signature(save_publication_figure).parameters
    assert any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )
    assert "dpi" in parameters and "formats" in parameters
    assert (
        "easyicu.research_agent.figures.publication",
        "save_publication_figure",
    ) not in _HOST_HELPER_CALL_CONTRACTS


def test_the_kwargs_skip_actually_works_and_is_not_just_unexercised(monkeypatch):
    """Reachability, not decoration.

    None of the four listed helpers takes ``**kwargs``, so deleting the skip
    changes nothing today and a mutation survived until this existed. The skip
    is what makes the list safe to GROW -- add a ``**kwargs`` callee without it
    and every legal ``dpi=`` on it becomes a block. So it is exercised against
    the real function that would be the first such entry.
    """

    from easyicu.research_agent.gates import preflight

    # ``save_publication_figure`` is the motivating case but cannot prove this
    # branch: it also takes ``*legacy_args``, so the varargs guard beside this
    # one would skip it either way and a mutation survived. This callee takes
    # ``**kwargs`` and no ``*args``, so only the guard under test can skip it.
    key = (
        "easyicu.research_agent.cohort.materializer",
        "materialize_to_parquet",
    )
    monkeypatch.setattr(
        preflight,
        "_SIGNATURE_DERIVED_HOST_HELPERS",
        (*preflight._SIGNATURE_DERIVED_HOST_HELPERS, key),
    )
    preflight._compile_signature_derived_contracts()
    try:
        assert (
            key not in preflight._HOST_HELPER_CALL_CONTRACTS
        ), "a helper that accepts **kwargs was given a closed keyword contract"
    finally:
        preflight._HOST_HELPER_CALL_CONTRACTS.pop(key, None)


def test_the_hand_written_entries_are_not_overwritten():
    """Derivation fills gaps; it does not take over a curated contract.

    The hand table also encodes REQUIRED keywords, which a signature cannot
    express -- a parameter with a default is optional to Python and mandatory
    to the host.
    """

    contract = _HOST_HELPER_CALL_CONTRACTS[
        (
            "easyicu.research_agent.methods.descriptive_inputs",
            "closed_categorical_counts",
        )
    ]

    assert contract["required_keywords"] == ("declared_levels",)


def test_a_locally_defined_lookalike_is_not_governed():
    """Only an exact import from the registered module grants host authority."""

    findings = _findings(
        """
def run_adjusted_association_figure(**anything):
    return anything

run_adjusted_association_figure(dpi=300, whatever=1)
"""
    )

    assert findings == [], findings


def test_the_recorded_corpus_shows_this_failure_class():
    """Re-measures rather than restating: these were real deaths."""

    if not _CORPUS.exists():
        pytest.skip("recorded run corpus is not mounted")

    hits = 0
    logs = 0
    for path in _CORPUS.glob("batch_*/*/aware/run_*/steps/*/run.log"):
        logs += 1
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if "unexpected keyword argument" in text:
            hits += 1

    if not logs:
        pytest.skip("no recorded step log could be read")
    assert logs > 500, logs
    assert hits > 0, "the failure class has disappeared from the corpus"


_THE_HOSTS_OWN_SEALED_SCAFFOLD = """
from easyicu.research_agent.execution.runners.adjusted_association_executor import (
    run_adjusted_association_from_env,
)

summary = run_adjusted_association_from_env(
    frame=frame,
    cohort_path=cohort_path,
    typed_cohort_input=typed_cohort_input,
    emit_step_summary=False,
    **declared_model,
)
"""


def test_the_hosts_own_sealed_scaffold_is_not_refused():
    """The regression this file's own change caused, on 2026-08-04.

    Adding ``run_adjusted_association_from_env`` to the derived registry turned
    the ``expanded_keyword_arguments_unverifiable`` rule on against the call the
    HOST writes itself -- ``adjusted_association_executor`` builds this exact
    text and comments "The call is host property too". m1 died at
    ``05_primary_adjusted_association_model`` with
    ``deterministic_standard_blocked``, on the one path where no Coder repair is
    attempted, for using the host's own scaffold.

    The body below is copied from that generator; the next test proves the
    copy still resembles it, because a drifted copy would quietly stop
    testing anything.
    """

    findings = _findings(_THE_HOSTS_OWN_SEALED_SCAFFOLD)

    assert findings == [], findings


def test_the_copied_scaffold_still_matches_the_generator_it_stands_for():
    """A fixture copied from production has to be checked against production.

    Not the whole text -- the generator interpolates a plan-specific model
    dict -- but the two properties that make the copy representative: it calls
    this helper, and it passes a keyword expansion.
    """

    import inspect

    from easyicu.research_agent.execution.runners.adjusted_association_executor import (
        adjusted_association_executor_scaffold,
    )

    source = inspect.getsource(adjusted_association_executor_scaffold)
    assert "run_adjusted_association_from_env(" in source
    assert "**declared_model," in source


def test_a_literal_unknown_keyword_is_still_caught_beside_an_expansion():
    """Exempting the expansion must not exempt what IS readable."""

    findings = _findings(
        _THE_HOSTS_OWN_SEALED_SCAFFOLD.replace(
            "    **declared_model,", "    dpi=300,\n    **declared_model,"
        )
    )

    assert findings, "a readable unknown keyword slipped through"
    assert "dpi" in findings[0].detail["unknown_keywords"], findings[0].detail


def test_a_hand_written_contract_still_refuses_what_it_cannot_read():
    """The rule stays where it means something.

    A hand-written contract also encodes REQUIRED keywords, so an expansion
    could hide a missing one. Only the derived contracts are exempt.
    """

    findings = _findings(
        """
from easyicu.research_agent.methods.descriptive_inputs import closed_categorical_counts

result = closed_categorical_counts(series, **options)
"""
    )

    assert findings, "an unreadable call to a curated contract was allowed"
    assert "expanded_keyword_arguments_unverifiable" in findings[0].detail["violations"]
