"""A report revision owns presentation bytes, never old analysis authority."""

import hashlib
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from easyicu.research_agent.reporting import revision_figures as owner
from easyicu.research_agent.reporting.manuscript_figures import ManuscriptFigure, ManuscriptFigures
from easyicu.research_agent.schema import AnalysisStep, EvidenceRecord, ValidationFinding
from easyicu.research_agent.figures.publication import make_figure_contract


@pytest.fixture
def revision_source(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    (source / "evidence").mkdir()
    records = []
    def record(name, kind, content, step, inputs=()):
        path = source / "evidence" / name
        path.write_bytes(content)
        row = EvidenceRecord(evidence_id=name.replace('.', '_'), kind=kind,
                             relative_path='evidence/' + name, sha256=hashlib.sha256(content).hexdigest(),
                             description=name, produced_by_step=step, inputs=list(inputs))
        records.append(row)
        return row
    table = record("exposure_outcome_distribution.csv", "table", b"column,n\nx,20\n", "counts")
    pdf = record("display.pdf", "figure", b"old pdf", "plot", [table.evidence_id])
    png = record("display.png", "figure", b"old png", "plot", [table.evidence_id])
    execution = [{"step_id": "plot", "status": "ok", "resolved_input_evidence_ids": [table.evidence_id]}]
    (source / "manifest.json").write_text(json.dumps({"per_step_records": execution}))
    (source / "analysis_plan.json").write_text('{}')
    (source / "research_context.json").write_text('{}')
    step = AnalysisStep.model_validate({"step_id": "plot", "planned_analysis_role": "auxiliary",
        "method": "visualization", "intent": "Show the declared result", "inputs": ["table:exposure_outcome_distribution"],
        "expected_outputs": ["figure:display"], "input_consumption_contracts": [{"input_key": "table:exposure_outcome_distribution", "mode": "all_rows"}]})
    prepared = SimpleNamespace(source_run_dir=source, context=None,
                              plan=SimpleNamespace(steps=[step], display_labels={"column": "Clinical label"}))
    monkeypatch.setattr(owner, 'ReadOnlyReportEvidence', lambda _: SimpleNamespace(root=source, current_verified_records=lambda _: records))
    def projection(**kwargs):
        row = png if kwargs['prefer_png'] else pdf
        return ManuscriptFigures((ManuscriptFigure(row.evidence_id, row.relative_path, 'Old caption', 'main', row.sha256, 'contract', 'c'*64),), (), ())
    monkeypatch.setattr(owner, 'build_manuscript_figures', projection)
    calls = []
    def render(**kwargs):
        calls.append(kwargs)
        path = kwargs['out_dir']
        for suffix in ('png', 'pdf', 'svg'):
            (path / ('display.' + suffix)).write_bytes(('new ' + suffix).encode())
        contract = make_figure_contract(figure_id='display', core_claim='Verified result', archetype='quantitative_grid',
            panels=[{'panel_id':'a', 'title':'Result', 'role':'distribution', 'claim':'Verified result', 'evidence_ids':[table.evidence_id]}],
            reader_caption='Current source-bound caption')
        (path / 'display.figure_contract.json').write_text(contract.model_dump_json())
    monkeypatch.setattr(owner, 'run_exposure_outcome_distribution_figure', render)
    output = tmp_path / 'revision'
    output.mkdir()
    return SimpleNamespace(source=source, prepared=prepared, records=records, output=output, calls=calls)


def test_one_render_binds_both_exports_and_preserves_source(revision_source):
    source = revision_source
    before = {str(p): p.read_bytes() for p in source.source.rglob('*') if p.is_file()}
    bundle = owner.build_revision_figure_bundle(prepared=source.prepared, output=source.output)
    assert len(source.calls) == 1
    assert source.calls[0]['display_labels'] == {'column':'Clinical label'}
    assert bundle.png.figures[0].caption == bundle.pdf.figures[0].caption == 'Current source-bound caption'
    assert bundle.png.figures[0].evidence_id != source.records[-1].evidence_id
    assert bundle.receipt['entries'][0]['exports']['png']['source_evidence_id'] == source.records[-1].evidence_id
    assert before == {str(p): p.read_bytes() for p in source.source.rglob('*') if p.is_file()}
    owner.verify_revision_figure_bundle(bundle)
    with pytest.raises(owner.WriterOnlyMigrationError, match='another report revision'):
        owner.verify_revision_figure_bundle(bundle, revision_output=source.output / 'other')
    with pytest.raises(owner.WriterOnlyMigrationError, match='projections differ'):
        owner.verify_revision_figure_bundle(replace(bundle, png=replace(bundle.png, figures=())))
    (bundle.root / bundle.png.figures[0].relative_path).write_bytes(b'changed')
    with pytest.raises(owner.WriterOnlyMigrationError, match='export changed'):
        owner.verify_revision_figure_bundle(bundle)


def test_revision_projects_legacy_caption_from_registered_panel_contract(
    revision_source, monkeypatch
):
    source = revision_source
    contract = make_figure_contract(
        figure_id='display', core_claim='Observed summaries',
        archetype='quantitative_grid',
        panels=[{'panel_id': 'a', 'title': 'Result', 'role': 'distribution',
                 'claim': 'Observed source values without model refitting.',
                 'evidence_ids': [source.records[0].evidence_id]}],
        statistics_note='No inferential uncertainty is shown.',
    )
    content = contract.model_dump_json().encode()
    path = source.source / 'evidence' / 'legacy.figure_contract.json'
    path.write_bytes(content)
    contract_record = EvidenceRecord(
        evidence_id='legacy_contract', kind='log',
        relative_path='evidence/legacy.figure_contract.json',
        sha256=hashlib.sha256(content).hexdigest(), description='Legacy contract',
        produced_by_step='plot',
    )
    source.records.append(contract_record)

    def projection(**kwargs):
        row = source.records[2] if kwargs['prefer_png'] else source.records[1]
        finding = ValidationFinding(
            validator='manuscript_figure_projection', severity='error',
            message='Reader caption missing', evidence_ids=[row.evidence_id],
            detail={'reason_code': 'MANUSCRIPT_FIGURE_CAPTION_MISSING'},
        )
        figure = ManuscriptFigure(
            row.evidence_id, row.relative_path, 'Requires review', 'main', row.sha256,
            contract_record.evidence_id, contract_record.sha256,
        )
        return ManuscriptFigures((figure,), (), (finding,))

    monkeypatch.setattr(owner, 'build_manuscript_figures', projection)
    monkeypatch.setattr(owner, 'exposure_outcome_distribution_figure_owns_step', lambda _: False)
    bundle = owner.build_revision_figure_bundle(
        prepared=source.prepared, output=source.output
    )
    expected = (
        '(A) Result. Observed source values without model refitting. '
        'No inferential uncertainty is shown.'
    )
    assert bundle.pdf.figures[0].caption == expected
    assert bundle.png.figures[0].caption == expected
    assert bundle.pdf.findings == ()
    assert bundle.png.findings == ()
    assert bundle.receipt['pdf']['findings'] == []
    assert bundle.receipt['png']['findings'] == []
    assert bundle.receipt['entries'][0]['caption_origin'] == (
        'registered_panel_contract_projection'
    )
    owner.verify_revision_figure_bundle(bundle)


@pytest.mark.parametrize('mutation', ['digest', 'input_membership', 'consumption'])
def test_revision_rejects_changed_or_widened_inputs_before_render(revision_source, mutation):
    source = revision_source
    if mutation == 'digest':
        (source.source / source.records[0].relative_path).write_text('different bytes')
    elif mutation == 'input_membership':
        source.records[1].inputs = ['another_table']
    else:
        source.prepared.plan.steps[0].input_consumption_contracts = []
    # Even if a changed declaration no longer belongs to this renderer, it must
    # not invoke that renderer. Unsupported families retain original exports.
    if mutation == 'consumption':
        owner.build_revision_figure_bundle(prepared=source.prepared, output=source.output)
    else:
        with pytest.raises(owner.WriterOnlyMigrationError):
            owner.build_revision_figure_bundle(prepared=source.prepared, output=source.output)
    assert not source.calls
