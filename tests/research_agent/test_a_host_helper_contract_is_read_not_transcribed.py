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
