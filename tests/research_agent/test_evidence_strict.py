"""STRICT enforcement mode for the EvidenceStore.

In SOFT mode (the default), unsupported result-like sentences are
silently filtered and unresolved ``{evidence:<id>}`` placeholders are
rendered as ``[evidence missing: <id>]`` markers. That is fine for
interactive iteration on the writer.

STRICT mode is meant for CI / final submission packaging: every guard
raises :class:`EvidenceEnforcementError` instead of repairing the
manuscript in place, so the run aborts loudly rather than shipping a
silently shortened or partially-bound manuscript.
"""

from __future__ import annotations

from pathlib import Path

import pytest


def test_soft_mode_is_default_and_filters_quietly(ra, tmp_path: Path):
    store = ra.EvidenceStore(root=tmp_path)
    assert store.enforcement_mode is ra.EvidenceEnforcementMode.SOFT

    scaffold = (
        "# Results\n\n"
        "Median age was 65 years.\n"
    )
    filtered, removed = store.enforce_evidence_bound_scaffold(scaffold)
    assert removed == ["Median age was 65 years."]
    assert "Median age" not in filtered


def test_strict_mode_raises_on_unsupported_result_sentence(ra, tmp_path: Path):
    store = ra.EvidenceStore(root=tmp_path, enforcement_mode="strict")
    scaffold = (
        "# Results\n\n"
        "Median age was 65 years.\n"
    )
    with pytest.raises(ra.EvidenceEnforcementError) as exc_info:
        store.enforce_evidence_bound_scaffold(scaffold)
    assert "STRICT evidence mode" in str(exc_info.value)
    assert exc_info.value.detail["removed_sentences"] == [
        "Median age was 65 years."
    ]


def test_strict_mode_accepts_evidence_bound_scaffold(ra, tmp_path: Path):
    store = ra.EvidenceStore(
        root=tmp_path, enforcement_mode=ra.EvidenceEnforcementMode.STRICT
    )
    scaffold = (
        "# Results\n\n"
        "Median age was 65 years {evidence:table_one}.\n"
    )
    filtered, removed = store.enforce_evidence_bound_scaffold(scaffold)
    assert removed == []
    assert "{evidence:table_one}" in filtered


def test_strict_mode_raises_on_bold_section_result_sentence(ra, tmp_path: Path):
    store = ra.EvidenceStore(root=tmp_path, enforcement_mode="strict")
    scaffold = (
        "**Background:** The relation between early SOFA-2 severity and ICU mortality "
        "remains sensitive to missingness and component-completeness artefacts.\n"
    )
    with pytest.raises(ra.EvidenceEnforcementError) as exc_info:
        store.enforce_evidence_bound_scaffold(scaffold)
    assert "STRICT evidence mode" in str(exc_info.value)
    assert any(
        "missingness and component-completeness artefacts" in sentence
        for sentence in exc_info.value.detail["removed_sentences"]
    )


def test_strict_mode_raises_on_unresolved_bind(ra, tmp_path: Path):
    store = ra.EvidenceStore(root=tmp_path, enforcement_mode="strict")
    with pytest.raises(ra.EvidenceEnforcementError) as exc_info:
        store.bind_manuscript("See {evidence:not_registered}.")
    assert "not_registered" in exc_info.value.detail["missing_evidence_ids"]


def test_strict_mode_binds_known_placeholder(ra, tmp_path: Path):
    src = tmp_path / "table_one.csv"
    src.write_text("a,b\n1,2\n", encoding="utf-8")
    store = ra.EvidenceStore(root=tmp_path, enforcement_mode="strict")
    store.register_file(kind="table", description="t1", source_path=src)
    bound = store.bind_manuscript("See {evidence:table_one}.")
    assert "table_one" in bound
    assert "[evidence missing:" not in bound


def test_unknown_enforcement_mode_rejected(ra, tmp_path: Path):
    with pytest.raises(ValueError):
        ra.EvidenceStore(root=tmp_path, enforcement_mode="paranoid")


def test_enforcement_mode_accepts_enum_value(ra, tmp_path: Path):
    store = ra.EvidenceStore(
        root=tmp_path, enforcement_mode=ra.EvidenceEnforcementMode.STRICT
    )
    assert store.enforcement_mode is ra.EvidenceEnforcementMode.STRICT
