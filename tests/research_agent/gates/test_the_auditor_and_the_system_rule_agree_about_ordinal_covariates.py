"""Two host layers must not publish opposite contracts about the same practice.

``system.txt`` rule 1 tells the agent:

    If a total score is modelled as a numeric covariate, state that modelling
    choice explicitly and keep the interpretation guarded.

The LLM concept auditor's focus list said, with no exception:

    ordinal scores treated as continuous

MEASURED on a real run (h2, 2026-08-03): the Coder obeyed the system rule. Its
generated script carried, on the balance table's own rows,

    "summary_scale": "rank-preserving numeric representation for GCS"

-- an explicit statement of exactly the modelling choice rule 1 asks it to
state. The auditor blocked the step anyway: "used as a continuous numeric
covariate in propensity-score fitting and standardized mean differences ...
this imposes unjustified equal-interval assumptions." The step died
``blocked_by_concept_audit`` on the one contract the agent had followed.

The fix keeps the check and removes the contradiction: a DECLARED
rank-preserving representation used for ADJUSTMENT is not by itself the defect;
an undeclared one, an ordinal serving as the estimand without a declared
coding, and an averaged ordinal all still are.

This file is a contract test between the two prompt layers, not a test of the
auditor model's judgement. It cannot make an LLM comply; it can stop the two
texts from drifting into opposite instructions again, which is what happened.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from easyicu.research_agent.audits.validators import LLMConceptAuditor
from easyicu.research_agent.providers.prompts import load_prompt_pack

_CORPUS = pathlib.Path("/Volumes/外置硬盘/easyicu_data/canonical9_runs")


def _auditor_prompt() -> str:
    """The real prompt builder, over a minimal context it does not read here."""

    from easyicu.research_agent.schema import CohortDescriptor, ResearchContext

    auditor = LLMConceptAuditor.__new__(LLMConceptAuditor)
    context = ResearchContext(
        research_question="q",
        cohort=CohortDescriptor(cohort_name="c", database="miiv", n_stays=1),
        variables=[],
    )
    return auditor._prompt(context=context, script_text="pass\n", step=None)


def test_the_system_rule_still_permits_a_declared_numeric_coding():
    """The anchor. If this rule is ever removed, the carve-out below is wrong."""

    system = load_prompt_pack()["system"]

    assert "modelled as a numeric covariate" in " ".join(system.split())
    # The file wraps this rule across lines; compare on collapsed whitespace.
    collapsed = " ".join(system.split())
    assert "state that modelling choice explicitly" in collapsed


def test_the_auditor_no_longer_contradicts_that_rule():
    prompt = _auditor_prompt()

    assert (
        "ordinal scores treated as continuous" in prompt
    ), "the check itself must survive; only its unconditional form was the bug"
    assert "do not report it as an ordinal-treated-as-continuous defect" in prompt
    assert "ADJUSTMENT" in prompt


def test_the_auditor_still_names_what_it_must_report():
    """A carve-out that swallowed the whole check would be worse than the bug."""

    prompt = _auditor_prompt()

    for required in (
        "no such declaration",
        "primary exposure",
        "estimand",
        "averaged",
    ):
        assert required in prompt, required


def test_the_carve_out_is_case_neutral():
    """Prompt hygiene: no benchmark task, score, or database may appear.

    The run that exposed this was about one score in one task; the rule is
    about ordinal covariates in general.
    """

    prompt = _auditor_prompt()
    carve_out = prompt.split("The system rule permits", 1)[1].split(
        "A value returned by a direct call", 1
    )[0]

    for forbidden in (
        "gcs",
        "sofa",
        "kdigo",
        "h2_",
        "vasopressor",
        "mimic",
        "miiv",
        "canonical9",
    ):
        assert forbidden not in carve_out.lower(), forbidden


def test_the_recorded_script_really_did_declare_its_coding():
    """Anchors the whole file in the artifact, not in a reconstruction.

    Without this the premise -- that the agent COMPLIED and was blocked anyway
    -- is only an argument.
    """

    if not _CORPUS.exists():
        pytest.skip("recorded run corpus is not mounted")

    script = (
        _CORPUS
        / "batch_20260802_luna_miiv_FULL_fb9c159_verify02"
        / "h2_vasopressor_causal"
        / "aware"
        / "run_20260803T033415_f364c8"
        / "steps"
        / "04_primary_causal_estimation"
        / "analysis.py"
    )
    if not script.exists():
        pytest.skip("the recorded h2 script is not mounted")

    source = script.read_text(encoding="utf-8", errors="replace")

    # The declaration the system rule asks for, written by the agent itself.
    assert "summary_scale" in source
    assert "rank-preserving numeric representation" in source


def test_the_recorded_block_is_the_one_this_repairs():
    if not _CORPUS.exists():
        pytest.skip("recorded run corpus is not mounted")

    manifest = (
        _CORPUS
        / "batch_20260802_luna_miiv_FULL_fb9c159_verify02"
        / "h2_vasopressor_causal"
        / "aware"
        / "run_20260803T033415_f364c8"
        / "manifest.json"
    )
    if not manifest.exists():
        pytest.skip("the recorded h2 manifest is not mounted")

    document = json.loads(manifest.read_text())
    messages = [
        str(finding.get("message") or "")
        for finding in document.get("findings", [])
        if str(finding.get("validator")) == "llm_concept_auditor"
    ]

    assert any(
        "continuous numeric covariate" in message and "ordinal" in message.lower()
        for message in messages
    ), messages
