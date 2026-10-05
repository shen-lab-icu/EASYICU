"""The figure privacy audit checks every subject count, whatever its column is called.

The audit cleared a figure for external upload when no source declared a group
size under the floor, but it read only thirteen declared names (``n``,
``n_patients``, ``stratum_n`` ...).  An at-risk table's ``at_risk``, a flow's
``count`` and ``excluded_since_prior_stage``, a group's ``group_events`` were
never compared with the floor, although the module said they were.  The audit
now reads count columns by the publication owner's name rule, keeps its own
floor of 20, and its version retires the receipts the narrower scan produced.

Synthetic aggregate tables only.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from easyicu.research_agent.authority.evidence_store import EvidenceStore
from easyicu.research_agent.gates.figure_privacy import (
    FIGURE_PRIVACY_AUDIT_VERSION,
    GROUP_SIZE_KEYS,
    MIN_DISCLOSED_GROUP_SIZE,
    TRUSTED_AUDIT_VERSIONS,
    audit_figure_privacy,
)
from easyicu.research_agent.gates.publication_disclosure import is_subject_count_name


def _audit(tmp_path, filename: str, text: str):
    store = EvidenceStore(tmp_path / "run")
    staged = tmp_path / filename
    staged.write_text(text, encoding="utf-8")
    store.register_file(
        kind="table", description="Figure source.", source_path=staged, evidence_id="src",
        producer="publication_figure_skill", generation_mode="deterministic_figure_skill",
    )
    contract = SimpleNamespace(
        figure_id="Figure2", core_claim="A claim.", statistics_note=None, image_integrity_note=None,
        panels=[SimpleNamespace(role="primary_estimand", title="Panel", claim="A claim.")],
        source_data=["src"],
    )
    return audit_figure_privacy(
        contract=contract, evidence=store, run_dir=Path(store.root), source_evidence_ids=["src"],
    )


def _small_cell_reasons(audit):
    return [reason for reason in audit.reasons if reason.startswith("src: group size(s) or subject count(s)")]


@pytest.mark.parametrize(("filename", "text", "cell"), [
    ("km.csv", "time,survival,at_risk,group_n\n0,1.0,250,250\n90,0.62,3,250\n", "at_risk=3"),
    (
        "flow.csv",
        "stage,count,excluded_since_prior_stage\nsource,500,\nlandmark,480,20\nanalysis,474,6\n",
        "excluded_since_prior_stage=6",
    ),
    ("groups.csv", "group,group_events,events_percent\nexposed,4,1.9\n", "group_events=4"),
    ("strata.json", json.dumps({"strata": [{"label": "a", "count": 7}, {"label": "b", "count": 90}]}), "count=7"),
])
def test_a_small_count_under_any_count_name_keeps_the_figure_local(tmp_path, filename, text, cell):
    audit = _audit(tmp_path, filename, text)

    assert audit.aggregate_only is False
    [reason] = _small_cell_reasons(audit)
    assert reason.endswith(cell)
    assert str(MIN_DISCLOSED_GROUP_SIZE) in reason


def test_counts_at_the_floor_and_settings_clear_the_figure(tmp_path):
    audit = _audit(
        tmp_path, "km.csv",
        "time,survival,at_risk,n_bootstrap,n_clusters\n0,1.0,250,5,3\n90,0.62,20,5,3\n",
    )

    assert _small_cell_reasons(audit) == []
    assert audit.aggregate_only is True


@pytest.mark.parametrize(("text", "finding"), [
    ("time,survival,at_risk\n0,1.0,patient_30042318\n", "src: identifier-shaped value(s) in at_risk"),
    ("stratum,n\nall,2150-03-01 14:22:00\n", "src: event timestamp value(s) in n"),
])
def test_a_count_column_still_has_its_values_scanned(tmp_path, text, finding):
    """The column name is the producer's choice; the cell is the disclosure."""
    audit = _audit(tmp_path, "source.csv", text)

    assert audit.aggregate_only is False
    assert any(reason.startswith(finding) for reason in audit.reasons)


def test_a_large_declared_group_size_is_a_count_not_an_identifier(tmp_path):
    audit = _audit(tmp_path, "cohort.csv", "stratum,n_patients,n_events\nall,200859,41203\n")

    assert audit.reasons == []
    assert audit.aggregate_only is True


def test_every_declared_group_size_is_read_as_a_subject_count():
    assert all(is_subject_count_name(name) for name in GROUP_SIZE_KEYS)


def test_receipts_of_the_narrower_scan_no_longer_clear_a_figure():
    assert FIGURE_PRIVACY_AUDIT_VERSION == "1.3.0"
    assert TRUSTED_AUDIT_VERSIONS == frozenset({"1.3.0"})
