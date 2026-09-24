"""A host claim reads by the names the plan and the sealed context give its variables.

The claim keeps its coordinates; only the reader sentence changes.  Every
caller that expands claims, and every caller that checks the expanded
Results, uses one set of labels.  The study here is a lactate tertile and ICU
readmission, with an admission-type contrast for named levels.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import easyicu.research_agent as research_agent
from easyicu.research_agent.authority.manuscript_claim_policy import (
    expand_scientific_claim_tokens,
    missing_scientific_claims_in_results,
)
from easyicu.research_agent.authority.scientific_claims import (
    bind_scientific_claim_drafts,
    derive_scientific_claim_drafts,
)
from easyicu.research_agent.reporting.manuscript_labels import reader_claim_labels
from easyicu.research_agent.schema import (
    CohortDescriptor,
    ConceptDescriptor,
    ResearchContext,
)


def _summary(**overrides: Any) -> dict[str, Any]:
    summary = {
        "status": "ok",
        "interpretation_class": "adjusted_association",
        "exposure": "lactate_tertile",
        "outcome": "icu_readmission",
        "effect_scale": "odds_ratio",
        "analysis_role": "primary",
        "analysis_set": "complete_case",
        "adjustment_covariates": ["age", "sex", "sofa"],
        "primary_estimate": 1.8604174,
        "primary_estimate_interval": [1.2217491, 2.8331002],
        "primary_contrast": {"exposure_level": "3", "reference_level": "1"},
    }
    summary.update(overrides)
    summary.setdefault("adjusted_effect", summary["primary_estimate"])
    return summary


def _claim(**overrides: Any):
    [claim] = bind_scientific_claim_drafts(
        [
            draft.model_dump(mode="json")
            for draft in derive_scientific_claim_drafts(_summary(**overrides))
        ],
        step_id="readmission_model",
        evidence_id="readmission_model_summary",
    )
    return claim


def _variable(name: str, description: str, role: str, dtype: str = "float"):
    return ConceptDescriptor(name=name, description=description, role=role, dtype=dtype)


def _context() -> ResearchContext:
    return ResearchContext(
        research_question="Is the first lactate tertile associated with ICU readmission?",
        cohort=CohortDescriptor(
            cohort_name="readmission_fixture", database="fixture",
            n_patients=540, n_stays=540,
        ),
        variables=[
            _variable("patient_stay_id", "Host-verified stay identity", "id", "str"),
            _variable("lactate_tertile", "first lactate tertile", "lab"),
            _variable("icu_readmission", "ICU readmission", "outcome", "int"),
            _variable("age", "patient age", "demographic"),
            _variable("sex", "patient sex", "demographic", "str"),
            _variable("sofa", "SOFA-2 score", "composite_score"),
            _variable("pf_ratio", "PaO2/FiO2 below 300", "vital"),
            _variable("lactate_units", "lactate [mmol/L]", "lab"),
        ],
    )


def test_the_names_are_english_and_carry_no_value_or_markup() -> None:
    labels = reader_claim_labels(_context(), {"lactate_tertile": "首个乳酸三分位"})

    # A plan label in another script yields the sealed English description.
    assert labels["lactate_tertile"] == "first lactate tertile"
    assert labels["icu_readmission"] == "ICU readmission"
    # A versioned name carries no reportable value.
    assert labels["sofa"] == "SOFA-2 score"
    # A row identity, a name with a value the binder must trace, and a name
    # with Markdown brackets are left out; those claims keep their keys.
    for key in ("patient_stay_id", "pf_ratio", "lactate_units"):
        assert key not in labels
    assert reader_claim_labels(None, {"icu_readmission": "ICU 再入院"}) == {}


def test_a_labelled_claim_reads_by_its_names() -> None:
    claim = _claim()

    assert claim.render_reader_text(labels=reader_claim_labels(_context(), {})) == (
        "After adjustment for patient age, patient sex, and SOFA-2 score, first "
        "lactate tertile 3 versus 1 was positively associated with ICU readmission "
        "in the complete case analysis set (adjusted odds ratio, 1.860; 95% CI, "
        "1.222 to 2.833)."
    )
    # Without labels the sentence keeps its coordinates, so sealed runs replay.
    assert claim.render_reader_text() == (
        "After adjustment for age, sex, and sofa, lactate tertile=3 versus lactate "
        "tertile=1 was positively associated with icu readmission in the complete "
        "case analysis set (adjusted odds ratio, 1.860; 95% CI, 1.222 to 2.833)."
    )


def test_named_levels_read_by_their_names_and_open_the_sentence() -> None:
    claim = _claim(
        exposure="admission_type", adjustment_covariates=[],
        primary_estimate=1.41, primary_estimate_interval=[1.02, 1.95],
        adjusted_effect=1.41,
        primary_contrast={"exposure_level": "emergency", "reference_level": "elective"},
    )
    names = {"admission_type": "admission type", "icu_readmission": "ICU readmission"}

    # A plan may key a text level with or without its JSON quotes.
    assert claim.render_reader_text(labels={
        **names,
        "admission_type=emergency": "emergency admission",
        'admission_type="elective"': "elective admission",
    }).startswith(
        "Emergency admission versus elective admission was positively associated "
        "with ICU readmission"
    )
    assert claim.render_reader_text(labels=names).startswith(
        'Admission type "emergency" versus "elective" was positively associated'
    )
    # A mixed-case term keeps its case at the start of a sentence.
    assert claim.render_reader_text(labels={
        "admission_type": "eGFR category at admission",
    }).startswith('eGFR category at admission "emergency" versus "elective" was')


def test_terms_that_would_read_alike_keep_their_keys() -> None:
    claim = _claim(adjustment_covariates=["age", "prior_readmission"])
    labels = {
        "icu_readmission": "readmission",
        "prior_readmission": "Readmission",
        "age": "patient age",
    }

    text = claim.render_reader_text(labels=labels)

    assert text.startswith(
        "After adjustment for age and prior readmission, lactate tertile=3 versus "
        "lactate tertile=1 was positively associated with icu readmission in"
    )


def test_the_results_check_reads_the_sentence_the_expansion_wrote() -> None:
    claim = _claim()
    labels = reader_claim_labels(_context(), {})
    scaffold = "## Results\n\n### Primary association\n\n" + claim.placeholder + "\n"

    expanded = expand_scientific_claim_tokens(
        scaffold,
        resolve_claim={claim.claim_ref: claim}.get,
        reader_labels=labels,
    )

    assert "first lactate tertile 3 versus 1" in expanded.scaffold
    assert missing_scientific_claims_in_results(
        expanded.scaffold, claims=[claim], reader_labels=labels,
    ) == ()
    assert missing_scientific_claims_in_results(
        expanded.scaffold, claims=[claim], reader_labels=None,
    ) == (claim.claim_ref,)


def _registered(tmp_path):
    from easyicu.research_agent.authority.evidence_store import EvidenceStore

    summary = _summary()
    store = EvidenceStore(tmp_path, enforcement_mode="strict")
    record = store.register_json(
        kind="statistic", description="Adjusted association",
        payload=summary, filename="summary.json",
        evidence_id="readmission_model_summary", produced_by_step="readmission_model",
        generation_mode="deterministic_standard",
    )
    store.register_step_summary_numerics(
        step_id="readmission_model", evidence_id=record.evidence_id, summary=summary,
    )
    records = [{
        "step_id": "readmission_model", "status": "ok",
        "generation_mode": "deterministic_standard",
        "step_summary": summary, "step_summary_evidence_id": record.evidence_id,
        "evidence_ids": [record.evidence_id],
    }]
    return store, records


def test_a_labelled_sentence_survives_strict_numeric_binding(tmp_path) -> None:
    from easyicu.research_agent.reporting.manuscript_post import (
        bind_numeric_values,
        drop_untraceable_numeric_sentences,
    )

    store, records = _registered(tmp_path)
    [claim] = store.scientific_claims()
    labels = reader_claim_labels(_context(), {})

    bound = store.bind_manuscript(
        "## Results\n\n### Primary association\n\n" + claim.placeholder,
        per_step_records=records, reader_labels=labels,
    )
    filtered, removed = drop_untraceable_numeric_sentences(
        bound, evidence=store, per_step_records=records,
    )
    _, _, untraced = bind_numeric_values(
        bound, evidence=store, per_step_records=records,
    )

    assert "After adjustment for patient age, patient sex, and SOFA-2 score" in bound
    assert removed == [] and filtered == bound
    assert not untraced


def test_readiness_checks_the_results_with_the_names_the_run_used(tmp_path) -> None:
    from easyicu.research_agent.reporting.manuscript_gate_state import (
        current_manuscript_completion_state,
    )
    from easyicu.research_agent.reporting.readiness import current_validation_findings
    from easyicu.research_agent.schema import AnalysisPlan

    store, records = _registered(tmp_path)
    [claim] = store.scientific_claims()
    plan = AnalysisPlan(
        research_question="Is the first lactate tertile associated with ICU readmission?",
        steps=[], display_labels={"lactate_tertile": "首个乳酸三分位"},
    )
    labels = reader_claim_labels(_context(), plan.display_labels)
    bound = store.bind_manuscript(
        "## Results\n\n### Primary association\n\n" + claim.placeholder,
        per_step_records=records, reader_labels=labels,
    )

    def complete(reader_labels) -> bool:
        return current_manuscript_completion_state(
            run_dir=tmp_path, manuscript_text=bound, evidence=store,
            per_step_records=records, stop_after_analysis=False,
            writer_probe_mode=False, reader_labels=reader_labels,
        )["manuscript_result_claims_complete"]

    assert complete(labels) and not complete(None)
    for context, expected in ((_context(), True), (None, False)):
        _, _, gate_state = current_validation_findings(
            plan=plan, per_step_records=records, findings=[], evidence=store,
            run_dir=tmp_path, manuscript_text=bound, context=context,
        )
        assert gate_state["manuscript_result_claims_complete"] is expected


# Each production call that renders claims, or checks rendered claims, names
# its labels, so no caller renders one sentence and checks another.
_LABELLED_CALLS = frozenset({
    "_claim_reader_view",
    "_expand_scientific_claim_tokens",
    "bind_manuscript",
    "current_manuscript_completion_state",
    "expand_scientific_claim_tokens",
    "missing_scientific_claims_in_results",
})


def test_every_production_caller_names_its_claim_labels() -> None:
    root = Path(research_agent.__file__).parent
    unlabelled = []
    for path in sorted(root.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.Call):
                continue
            name = getattr(node.func, "attr", None) or getattr(node.func, "id", None)
            if name in _LABELLED_CALLS and not any(
                keyword.arg == "reader_labels" for keyword in node.keywords
            ):
                unlabelled.append(f"{path.relative_to(root)}:{node.lineno} {name}")

    assert unlabelled == []
