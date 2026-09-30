"""The Methods state the time grid and model an owner executed.

The run context's time window describes what the host materialized for the
cohort.  A longitudinal owner reads its own grid, and the strict Methods
grammar admits a numeric design detail only as an exact host fact, so a
manuscript either lost the executed grid or restated the context window as
the analysis window.  Owners now seal an ``executed_method_design`` block; the
host renders it as an exact Methods fact bound to that owner's summary, and
the Writer's method boundary carries it.  Synthetic designs only.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from easyicu.research_agent.authority.evidence_store import (
    EvidenceEnforcementError,
    EvidenceEnforcementMode,
    EvidenceStore,
)
from easyicu.research_agent.authority.manuscript_method_facts import (
    MethodFactAuthorityError,
)
from easyicu.research_agent.contracts.executed_method_design import (
    EXECUTED_METHOD_DESIGN_SCHEMA_VERSION,
)
from easyicu.research_agent.reporting.manuscript_post import bind_numeric_values
from easyicu.research_agent.reporting.writer_evidence import (
    _render_writer_evidence_digest,
)

PANEL = {
    "schema_version": EXECUTED_METHOD_DESIGN_SCHEMA_VERSION,
    "design_kind": "fixed_window_representation",
    "anchor": "icu_admission",
    "window_start_hours": 0,
    "window_end_hours": 48,
    "window_width_hours": 8,
    "n_windows": 6,
    "window_aggregation": "max",
    "minimum_observed_windows": 3,
}
MODEL = {
    "schema_version": EXECUTED_METHOD_DESIGN_SCHEMA_VERSION,
    "design_kind": "latent_class_model",
    "model_family": "latent_class_diagonal_gaussian_mixture",
    "coordinate_scaling": "pooled_coordinate_wise_z_score",
    "candidate_class_counts": [2, 3, 4, 5],
    "selection_criterion": "bic",
    "minimum_class_fraction": 0.05,
}


def _store(tmp_path: Path, designs: dict[str, dict], *, generation_mode="deterministic_standard"):
    store = EvidenceStore(tmp_path, enforcement_mode=EvidenceEnforcementMode.STRICT)
    for step_id, design in designs.items():
        source = tmp_path / "steps" / step_id / "outputs" / "step_summary.json"
        source.parent.mkdir(parents=True)
        summary = {"status": "ok", "executed_method_design": design}
        source.write_text(json.dumps(summary), encoding="utf-8")
        store.register_file(
            kind="statistic",
            source_path=source,
            description="Owner summary",
            evidence_id=f"{step_id}_summary",
            produced_by_step=step_id,
            producer="runner",
            generation_mode=generation_mode,
        )
        store.register_step_summary_numerics(
            step_id=step_id, evidence_id=f"{step_id}_summary", summary=summary,
        )
    return store


def test_the_executed_grid_and_model_are_exact_bound_method_facts(tmp_path) -> None:
    store = _store(tmp_path, {"00_panel": PANEL, "01_candidates": MODEL})

    panel, model = store.manuscript_method_facts()

    assert panel.scaffold == (
        "Executed time design: each coordinate was summarized by its maximum in 6 "
        "consecutive 8-hour windows from 0 to 48 hours after ICU admission, and a "
        "record entered the model with at least 3 observed windows "
        "{evidence:00_panel_summary}."
    )
    assert model.scaffold == (
        "Executed class model: a diagonal Gaussian latent class mixture fitted to "
        "pooled coordinate-wise z-scores for 2 to 5 classes, with the class count "
        "chosen by the minimum Bayesian information criterion and a prespecified "
        "minimum class proportion of 0.050 of records {evidence:01_candidates_summary}."
    )
    scaffold = "## Methods\n\n### Variables\n\n" + "\n\n".join(
        fact.scaffold for fact in (panel, model)
    )
    ledger = [
        {"step_id": step_id, "status": "ok", "evidence_ids": [f"{step_id}_summary"]}
        for step_id in ("00_panel", "01_candidates")
    ]
    safe, removed = store.enforce_evidence_bound_scaffold(scaffold, per_step_records=ledger)
    assert not removed
    bound = store.bind_manuscript(safe, per_step_records=ledger)
    _, _, untraced = bind_numeric_values(bound, evidence=store, per_step_records=ledger)
    assert not untraced


def test_the_wording_follows_the_design_it_is_given(tmp_path) -> None:
    store = _store(tmp_path, {
        "00_panel": {**PANEL, "anchor": "hospital_admission",
                     "window_start_hours": -12, "window_end_hours": 36},
        "01_candidates": {**MODEL, "candidate_class_counts": [2, 4, 6]},
    })

    panel, model = store.manuscript_method_facts()

    assert "from -12 to 36 hours relative to hospital admission" in panel.text
    assert "for 2, 4 and 6 classes" in model.text


@pytest.mark.parametrize(
    "forged",
    [
        lambda text: text.replace("0 to 48 hours", "0 to 24 hours"),
        lambda text: text.replace("at least 3", "at least 1"),
        lambda text: "- " + text,
    ],
)
def test_a_design_statement_cannot_be_forged(tmp_path, forged) -> None:
    store = _store(tmp_path, {"00_panel": PANEL})
    [panel] = store.manuscript_method_facts()

    with pytest.raises(EvidenceEnforcementError):
        store.enforce_evidence_bound_scaffold(
            "## Methods\n\n### Variables\n\n" + forged(panel.scaffold)
        )


def test_the_design_form_is_reserved_outside_methods_too(tmp_path) -> None:
    store = _store(tmp_path, {"01_candidates": MODEL})

    with pytest.raises(EvidenceEnforcementError):
        store.enforce_evidence_bound_scaffold(
            "## Discussion\n\nExecuted class model: an ordinal latent class model "
            "{evidence:01_candidates_summary}."
        )


def test_only_a_deterministic_owner_states_a_design(tmp_path) -> None:
    assert _store(tmp_path / "script", {"00_panel": PANEL}, generation_mode="llm") \
        .manuscript_method_facts() == ()

    broken = _store(tmp_path / "broken", {"00_panel": {**PANEL, "n_windows": 5}})
    with pytest.raises(MethodFactAuthorityError, match="cannot be reproduced"):
        broken.manuscript_method_facts()


def _record(step_id: str, summary: dict) -> dict:
    return {
        "step_id": step_id,
        "status": "ok",
        "generation_mode": "deterministic_standard",
        "deterministic_standard_analysis": "signed_owner",
        "writer_result_envelope_evidence_id": f"{step_id}_envelope",
        "step_summary": summary,
    }


def test_the_writer_boundary_carries_the_design_and_a_rejection_its_outcome(tmp_path) -> None:
    store = EvidenceStore(tmp_path)
    digest = _render_writer_evidence_digest(
        [
            _record("00_panel", {"status": "ok", "executed_method_design": PANEL}),
            _record("01_candidates", {
                "status": "ok",
                "n_clusters": 5,
                "silhouette": 0.12,
                "scientific_status": "failed_closed",
                "reason_code": "NO_INTERIOR_BIC_OPTIMUM",
                "reportable_result": "no_interior_solution_in_prespecified_candidate_range",
                "executed_method_design": MODEL,
            }),
            _record("06_selected_elsewhere", {"status": "ok", "n_clusters": 3}),
        ],
        run_dir=tmp_path,
        evidence=store,
    )

    rows = {
        line.split(" [", 1)[0][2:]: line for line in digest.splitlines()
        if line.startswith("- ") and " [" in line
    }
    lines = digest.splitlines()
    candidate_row = json.loads(lines[lines.index(rows["01_candidates"]) + 1])
    # A rejected owner's scalars are not findings; its formal outcome is.
    assert candidate_row == {
        "reason_code": "NO_INTERIOR_BIC_OPTIMUM",
        "reportable_result": "no_interior_solution_in_prespecified_candidate_range",
        "scientific_status": "failed_closed",
    }
    other_row = json.loads(lines[lines.index(rows["06_selected_elsewhere"]) + 1])
    assert other_row == {"n_clusters": 3}

    boundary = digest.split("## EXECUTED METHOD BOUNDARY", 1)[1]
    assert "describe that step's time grid, eligibility rule and model" in boundary
    panel_row = json.loads(next(
        line[2:] for line in boundary.splitlines() if '"00_panel"' in line
    ))
    assert panel_row["executed_method_design"] == {
        key: value for key, value in PANEL.items() if key != "schema_version"
    }
