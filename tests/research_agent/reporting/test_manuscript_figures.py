from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import pytest

from easyicu.research_agent.reporting.latex import scaffold_to_latex
from easyicu.research_agent.reporting.manuscript_figures import (
    ManuscriptFigureProjectionError,
    build_manuscript_figures,
    manuscript_figure_receipt_is_current,
)
from easyicu.research_agent.schema import EvidenceRecord


def _record(root, identifier, filename, content, *, step="plot", kind="figure", **kwargs):
    path = root / "evidence" / f"{identifier}__{filename}"
    path.parent.mkdir(exist_ok=True)
    path.write_bytes(content)
    return EvidenceRecord(
        evidence_id=identifier, kind=kind, description="Test source",
        relative_path=str(path.relative_to(root)), sha256=hashlib.sha256(content).hexdigest(),
        produced_by_step=step, producer="runner" if step else "publication_figure_skill",
        generation_mode="deterministic_standard" if step else "deterministic_figure_skill",
        **kwargs,
    )


def _bundle(root, key="result", *, step="plot", caption="Observed values. No inference.",
            placement="main", content=b"image", inputs=()):
    contract = _record(root, f"{key}_contract", f"{key}.figure_contract.json", json.dumps({
        "figure_id": f"figure:{key}", "core_claim": "Observed summaries",
        **({"reader_caption": caption} if caption is not None else {}),
        "panels": [{"panel_id": "A", "title": "Results", "claim": "Observed summaries",
                    "role": "descriptive_result", "metadata": {"placement": placement}}],
    }).encode(), step=step, kind="log")
    figure = _record(root, f"{key}_pdf", f"{key}.pdf", content, step=step,
                     inputs=[*inputs, contract.evidence_id], metadata={
                         "contract_evidence_id": contract.evidence_id,
                         "artifact_role": "manuscript_figure",
                     })
    return contract, figure


def _build(root, records):
    return build_manuscript_figures(evidence_records=records, run_dir=root)


def test_projects_registered_caption_and_explicit_supplementary_placement(tmp_path):
    contract, figure = _bundle(tmp_path, placement="supplementary")
    result = _build(tmp_path, [contract, figure])
    assert not result.findings
    assert len(result.figures) == 1
    rendered = result.figures[0]
    assert rendered.caption == "Observed values. No inference."
    assert rendered.placement == "supplementary"
    assert rendered.contract_evidence_id == contract.evidence_id
    assert rendered.contract_sha256 == contract.sha256
    assert rendered.figure_sha256 == figure.sha256
    assert rendered.relative_path == figure.relative_path


def test_never_reads_unregistered_contract_or_borrows_its_caption(tmp_path):
    figure = _record(tmp_path, "plot_pdf", "plot.pdf", b"plot")
    unregistered = tmp_path / "publication_figures" / "plot.figure_contract.json"
    unregistered.parent.mkdir()
    unregistered.write_text(json.dumps({"reader_caption": "Significant survival benefit"}))
    result = _build(tmp_path, [figure])
    assert "Significant" not in result.figures[0].caption
    assert result.figures[0].placement == "supplementary"
    assert result.findings[0].severity == "error"
    assert result.findings[0].detail["reason_code"] == "MANUSCRIPT_FIGURE_CONTRACT_MISSING"


def test_legacy_contract_does_not_promote_core_claim_into_a_legend(tmp_path):
    result = _build(tmp_path, _bundle(tmp_path, caption=None))
    assert "Observed summaries" not in result.figures[0].caption
    assert result.findings[0].detail["reason_code"] == "MANUSCRIPT_FIGURE_CAPTION_MISSING"


@pytest.mark.parametrize("target", ["contract", "figure"])
@pytest.mark.parametrize("change", ["tamper", "delete", "symlink"])
def test_rejects_changed_or_unavailable_registered_bytes(tmp_path, target, change):
    contract, figure = _bundle(tmp_path)
    path = tmp_path / (contract if target == "contract" else figure).relative_path
    if change == "tamper":
        path.write_bytes(b"changed")
    else:
        original = path.read_bytes()
        path.unlink()
        if change == "symlink":
            sibling = tmp_path / "different-source"
            sibling.write_bytes(original)
            path.symlink_to(sibling)
    with pytest.raises(ManuscriptFigureProjectionError, match="verified"):
        _build(tmp_path, [contract, figure])


def test_missing_explicit_contract_link_never_falls_back_to_name(tmp_path):
    contract, figure = _bundle(tmp_path)
    figure.metadata["contract_evidence_id"] = "retired_contract"
    with pytest.raises(ManuscriptFigureProjectionError, match="contract link"):
        _build(tmp_path, [contract, figure])


def test_ambiguous_contracts_do_not_take_last_matching_file(tmp_path):
    contract, figure = _bundle(tmp_path)
    figure.metadata.clear()
    duplicate = _record(tmp_path, "another_contract", "result.figure_contract.json",
                        (tmp_path / contract.relative_path).read_bytes(), kind="log")
    with pytest.raises(ManuscriptFigureProjectionError, match="ambiguous"):
        _build(tmp_path, [contract, duplicate, figure])


def test_contract_from_another_step_cannot_label_same_named_figure(tmp_path):
    contract, figure = _bundle(tmp_path)
    contract.produced_by_step = "different_step"
    with pytest.raises(ManuscriptFigureProjectionError, match="owner"):
        _build(tmp_path, [contract, figure])


def test_same_basename_in_two_steps_remains_two_figures(tmp_path):
    a_contract, a = _bundle(tmp_path, "one", step="one")
    b_contract, b = _bundle(tmp_path, "two", step="two")
    b_path = tmp_path / b.relative_path
    a_path = tmp_path / a.relative_path
    new_path = b_path.with_name("two_pdf__one.pdf")
    b_path.rename(new_path)
    b.relative_path = str(new_path.relative_to(tmp_path))
    assert a_path.exists()
    assert len(_build(tmp_path, [a_contract, a, b_contract, b]).figures) == 2


@pytest.mark.parametrize("source_link,same_bytes,expected", [
    (True, True, 1), (True, False, 2), (False, True, 2), (False, False, 2),
])
def test_deduplicates_only_exact_promoted_source_not_shared_panel_role(
    tmp_path, source_link, same_bytes, expected,
):
    contract, figure = _bundle(tmp_path)
    promoted_contract, promoted = _bundle(
        tmp_path, "promoted", step=None, content=b"image" if same_bytes else b"other",
        inputs=[figure.evidence_id] if source_link else [],
    )
    result = _build(tmp_path, [contract, figure, promoted_contract, promoted])
    assert len(result.figures) == expected
    assert result.figures[0].evidence_id == promoted.evidence_id


def test_latex_uses_plain_text_bound_caption_in_both_sections(tmp_path):
    result = _build(tmp_path, _bundle(tmp_path, caption="A: 40% & B: source_x. No CI."))
    figure = result.figures[0]
    tex = scaffold_to_latex(markdown="# Results\n\nObserved data.",
                            figures=[figure, replace(figure, placement="supplementary",
                                                      relative_path="evidence/supplement.pdf")])
    assert tex.count(r"\caption{A: 40\% \& B: source\_x. No CI.}") == 2
    assert r"\section*{Main figures}" in tex
    assert r"\section*{Supplementary figures}" in tex
    assert "Observed values" not in tex
    with pytest.raises(ValueError, match="exactly once"):
        scaffold_to_latex(markdown="# Results", figures=[figure, figure])


def test_projection_caption_error_is_a_manuscript_readiness_error(tmp_path):
    from easyicu.research_agent.reporting.readiness import _MANUSCRIPT_ERROR_VALIDATORS
    result = _build(tmp_path, _bundle(tmp_path, caption=None))
    assert all(f.validator in _MANUSCRIPT_ERROR_VALIDATORS for f in result.findings)


@pytest.mark.parametrize("changed", ["none", "caption", "membership", "receipt", "image"])
def test_old_errors_clear_only_with_exact_current_projection_receipt(tmp_path, changed):
    records = list(_bundle(tmp_path))
    projection = _build(tmp_path, records)
    receipt = _record(tmp_path, "projection", "manuscript_figure_projection.json",
                      json.dumps(projection.as_receipt()).encode(), step=None, kind="log",
                      metadata={"artifact_role": "manuscript_figure_projection"})
    receipt.producer, receipt.generation_mode = "pipeline", "system"
    records.append(receipt)
    if changed == "caption":
        path = tmp_path / records[0].relative_path
        payload = json.loads(path.read_text())
        payload["reader_caption"] = "A different source-bound explanation."
        path.write_text(json.dumps(payload))
        records[0].sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    elif changed == "membership":
        records.pop(1)
    elif changed == "receipt":
        (tmp_path / receipt.relative_path).write_text("{}")
    elif changed == "image":
        (tmp_path / records[1].relative_path).write_bytes(b"other image")
    assert manuscript_figure_receipt_is_current(
        run_dir=tmp_path, evidence_records=records,
    ) is (changed == "none")


@pytest.mark.parametrize("caption", ["   ", "Hidden\x00text", "A\nB"])
def test_non_plain_caption_is_not_accepted(tmp_path, caption):
    with pytest.raises(ManuscriptFigureProjectionError, match="invalid"):
        _build(tmp_path, _bundle(tmp_path, caption=caption))


def test_two_different_current_exports_are_not_resolved_by_list_order(tmp_path):
    contract, figure = _bundle(tmp_path)
    newer = _record(tmp_path, "newer", "result.pdf", b"different result",
                    inputs=figure.inputs, metadata=figure.metadata)
    with pytest.raises(ManuscriptFigureProjectionError, match="versions are ambiguous"):
        _build(tmp_path, [contract, figure, newer])


def test_legacy_evidence_relative_path_is_canonicalized_for_latex(tmp_path):
    contract, figure = _bundle(tmp_path)
    canonical = figure.relative_path
    figure.relative_path = canonical.removeprefix("evidence/")
    assert _build(tmp_path, [contract, figure]).figures[0].relative_path == canonical


@pytest.mark.parametrize('count', [127, 8501, 200000])
def test_single_denominator_becomes_source_bound_text_for_any_cohort(tmp_path, count):
    contract, figure = _bundle(tmp_path, key='accounting')
    path = tmp_path / contract.relative_path
    raw = json.loads(path.read_text())
    raw['panels'][0]['role'] = 'cohort_accounting'
    raw['panels'][0]['metadata'].update(accounting_completeness='analysis_denominator_only', source_data=['ledger.csv'])
    path.write_text(json.dumps(raw))
    contract = contract.model_copy(update={'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    table = _record(tmp_path, 'ledger', 'ledger.csv', f'accounting_completeness,n_remaining\nanalysis_denominator_only,{count}\n'.encode(), kind='table')
    projection = _build(tmp_path, [contract, figure, table])
    assert projection.figures == ()
    assert not projection.findings and not projection.omitted_evidence_ids
    note = projection.context_notes[0]
    assert f'{count:,} records' in note['text']
    assert note['source_sha256'] == table.sha256
    assert note['reason_code'] == 'SINGLE_DENOMINATOR_AS_TEXT'
    assert (tmp_path / figure.relative_path).exists()
    with pytest.raises(ManuscriptFigureProjectionError):
        _build(tmp_path, [contract, figure])
    (tmp_path / table.relative_path).write_text('changed')
    with pytest.raises(ManuscriptFigureProjectionError):
        _build(tmp_path, [contract, figure, table])
