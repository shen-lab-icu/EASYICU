"""Revision export must use the current bound text and verified aggregate inputs."""
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from easyicu.webserver import report_revision_export as owner


def test_export_uses_revision_text_and_declared_engine_without_rerunning_analysis(tmp_path, monkeypatch):
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'manifest.json').write_text('{"per_step_records":[{"step_id":"a"}]}')
    output = tmp_path / 'revision'
    output.mkdir()
    bound = '# Report\n\n## Results\nVerified text.\n'
    (output / 'manuscript_bound.md').write_text(bound)
    revision = {'revision_id': 'r2', 'source_run_id': 'r1', 'output_sha256': hashlib.sha256(bound.encode()).hexdigest()}
    seen = {}
    evidence = SimpleNamespace(root=source, current_verified_records=lambda rows: seen.setdefault('execution', rows))
    monkeypatch.setattr(owner, 'ReadOnlyReportEvidence', lambda root: evidence)
    monkeypatch.setattr(owner, 'build_manuscript_figures', lambda **kw: SimpleNamespace(figures=(), findings=(), omitted_evidence_ids=()))
    monkeypatch.setattr(owner, 'build_manuscript_tables', lambda **kw: ())
    monkeypatch.setattr(owner, 'scaffold_to_latex', lambda **kw: seen.setdefault('latex', kw) and 'TeX')
    def render(**kwargs):
        seen['render'] = kwargs
        (kwargs['output_dir'] / 'manuscript_revision.pdf').write_bytes(b'%PDF-test')
        (kwargs['output_dir'] / 'manuscript_pdf_receipt.json').write_text('{}')
        return SimpleNamespace(success=True)
    monkeypatch.setattr(owner, 'render_pdf_for_run', render)
    plan = SimpleNamespace(display_labels={}, model_copy=lambda **kw: None)
    result = owner.export_revision_pdf(prepared=SimpleNamespace(source_run_dir=source, context=None, plan=plan, literature=None),
        output=output, reader=owner.render_reader_manuscript(bound), revision=revision)
    assert seen['execution'] == [{'step_id': 'a'}]
    assert seen['latex']['markdown'] == owner.render_reader_manuscript(bound)
    assert 'Revision r2' in seen['latex']['authors'][0]
    assert seen['render']['draft_watermark'] is True
    assert result['manuscript_sha256'] == revision['output_sha256']
    assert result['sha256'] == hashlib.sha256(b'%PDF-test').hexdigest()
    assert list(source.iterdir()) == [source / 'manifest.json']


@pytest.mark.parametrize('changed', ['bound', 'reader'])
def test_export_rejects_cross_revision_content_before_render(tmp_path, changed):
    bound = 'Current report\n'
    (tmp_path / 'manuscript_bound.md').write_text(bound)
    revision = {'output_sha256': hashlib.sha256(bound.encode()).hexdigest()}
    if changed == 'bound':
        (tmp_path / 'manuscript_bound.md').write_text('Other report')
    with pytest.raises(owner.WriterOnlyMigrationError):
        owner.export_revision_pdf(prepared=None, output=tmp_path, reader='Other report' if changed == 'reader' else bound, revision=revision)
    assert not (tmp_path / 'pdf').exists()
