from pathlib import Path
from types import SimpleNamespace
import pytest
from easyicu.research_agent.reporting.write_phase import _causal_delivery_blockers


def rec(evidence_id, description, kind='table', metadata=None):
    return SimpleNamespace(evidence_id=evidence_id, description=description, kind=kind,
                           relative_path='evidence/' + evidence_id + '.csv', metadata=metadata or {})


def test_actual_audit_blocks_linked_association_only(tmp_path: Path):
    association = rec('primary_association', 'Logistic odds ratio')
    ref = '[primary_association](evidence/primary_association.csv "sha256=abcdef12")'
    assert _causal_delivery_blockers(bound='X caused Y ' + ref + '.', evidence_records=[association], run_dir=tmp_path)
    assert _causal_delivery_blockers(bound='X caused Y 0.84[^n1].', evidence_records=[association], run_dir=tmp_path, numeric_binding_map={'n1': SimpleNamespace(evidence_id='primary_association')})
    assert not _causal_delivery_blockers(bound='X was associated with Y ' + ref + '.', evidence_records=[association], run_dir=tmp_path)
    assert not _causal_delivery_blockers(bound='This association was not caused by X ' + ref + '.', evidence_records=[association], run_dir=tmp_path)
    assert _causal_delivery_blockers(bound='X was not caused by Y but causes Z ' + ref + '.', evidence_records=[association], run_dir=tmp_path)
    assert _causal_delivery_blockers(bound='结果为“X caused Y” ' + ref + '.', evidence_records=[association], run_dir=tmp_path)
    assert _causal_delivery_blockers(bound='X 导致 Y ' + ref + '。', evidence_records=[association], run_dir=tmp_path)
    assert _causal_delivery_blockers(bound='Y was not caused by X but by Z ' + ref + '.', evidence_records=[association], run_dir=tmp_path)
    assert not _causal_delivery_blockers(bound='X 未导致 Y ' + ref + '。', evidence_records=[association], run_dir=tmp_path)
    assert not _causal_delivery_blockers(bound='X caused Y [literature](evidence/literature.txt).', evidence_records=[association], run_dir=tmp_path)


def test_valid_causal_design_is_not_blocked(tmp_path: Path):
    causal = rec('primary_association_iptw', 'IPTW risk difference', metadata={'identification_strategy': {'method': 'iptw'}})
    records = [causal] + [rec(n, '', kind='log') for n in ('dag', 'positivity_diagnostic', 'negative_control', 'e_value')]
    assert not _causal_delivery_blockers(bound='X caused Y [primary_association_iptw](evidence/out.csv).', evidence_records=records, run_dir=tmp_path)


def test_real_write_boundary_keeps_draft_and_blocks_final_export(tmp_path, monkeypatch):
    from easyicu.research_agent.authority.evidence_store import EvidenceStore
    from easyicu.research_agent.reporting import write_phase
    from easyicu.research_agent.schema import AnalysisPlan, CritiqueReport
    from easyicu.webserver.agent_pipeline_runs import _validated_manuscript_documents
    import hashlib
    import json

    pdf = tmp_path / 'manuscript_scaffold.pdf'
    tex = tmp_path / 'manuscript_scaffold.tex'
    pdf.write_bytes(b'%PDF-old')
    tex.write_text('old manuscript')
    def row(path):
        return {'name': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    (tmp_path / 'manuscript_pdf_receipt.json').write_text(json.dumps({
        'schema_version':'easyicu.manuscript_pdf_receipt.v1', 'status':'rendered', 'draft_watermark':True,
        'security':{'network_allowed':False, 'shell_escape_allowed':False, 'untrusted_input_mode':True},
        'pdf':row(pdf), 'source':row(tex), 'bibliography':None}))
    assert len(_validated_manuscript_documents(tmp_path)[1]) == 2
    evidence = EvidenceStore(tmp_path)
    effect = evidence.register_json(kind='table', description='Primary association logistic odds ratio',
        payload={'estimate': 0.84}, filename='primary.json', evidence_id='primary_association', producer='host')
    raw = 'X caused Y {evidence:primary_association}.'
    bound = evidence.bind_manuscript(raw, per_step_records=[])
    draft_path = tmp_path / 'manuscript_scaffold_bound.md'
    draft_path.write_text(bound)
    plan = AnalysisPlan(research_question='Synthetic association question', steps=[])
    findings, calls = [], []
    monkeypatch.setattr(write_phase, 'execution_gate_status', lambda **_: {'execution_complete': True})
    monkeypatch.setattr(write_phase, '_activate_publication_figure', lambda *a, **kw: None)
    monkeypatch.setattr(write_phase, '_activate_publication_inputs', lambda *a, **kw: None)
    monkeypatch.setattr(write_phase, '_write_reproducibility_artifacts', lambda *a, **kw: None)
    monkeypatch.setattr(write_phase, '_draft_bind_and_repair_manuscript', lambda *a, **kw: (
        SimpleNamespace(), write_phase._BindingStageResult(bound, draft_path, CritiqueReport(status='pass', reviewer='test'))))
    monkeypatch.setattr(write_phase, '_append_deferred_source_flow', lambda *a, **kw: calls.append('late'))
    monkeypatch.setattr(write_phase, '_publish_and_audit_manuscript', lambda *a, **kw: calls.append('publish'))
    pipeline = SimpleNamespace(_user_extension_activation=SimpleNamespace(receipt={'activation_sha256':'a'*64, 'skills':[], 'mcp_servers':[]}))
    result = write_phase.run_write_phase(pipeline, plan_result=SimpleNamespace(context=None, agent_context=None,
        evidence=evidence, findings=findings, role_resolver=lambda _: object(), prompt_version='test',
        resume_state=None, repro_envelope=None), execute_result=SimpleNamespace(plan=plan, runtime_state=object(), per_step_records=[]),
        run_dir=tmp_path, run_id='synthetic', stop_after_analysis=False, manuscript_title=None,
        manuscript_authors=None, run_language='en', emit_progress=lambda *a, **kw: None)
    assert effect.evidence_id in findings[-1].evidence_ids
    assert findings[-1].severity == 'error' and findings[-1].detail['reportability_allowed'] is False
    assert result.manuscript_critique.status == 'blocked'
    assert draft_path.read_text() == bound and calls == []
    assert _validated_manuscript_documents(tmp_path) == (None, [])
    assert pdf.read_bytes() == b'%PDF-old' and tex.read_text() == 'old manuscript'
    critique_path = tmp_path / 'manuscript_critique.json'
    saved_critique = critique_path.read_bytes()
    for broken in (b'{', b'', b'[]', b'{"status": "unknown"}', b'{"status": "pass"}', b' ' * (2 * 1024 * 1024 + 1)):
        critique_path.write_bytes(broken)
        assert _validated_manuscript_documents(tmp_path) == (None, [])
    critique_path.write_bytes(saved_critique)
    original_read = Path.read_text
    def failing_read(path, *args, **kwargs):
        if path == critique_path:
            raise OSError('synthetic read failure')
        return original_read(path, *args, **kwargs)
    with monkeypatch.context() as m:
        m.setattr(Path, 'read_text', failing_read)
        assert _validated_manuscript_documents(tmp_path) == (None, [])
    from easyicu.research_agent.authority import evidence_store
    with monkeypatch.context() as m:
        m.setattr(evidence_store, '_atomic_write_text', lambda *a, **kw: (_ for _ in ()).throw(OSError('synthetic write failure')))
        with pytest.raises(OSError, match='synthetic write failure'):
            write_phase._persist_manuscript_critique(critique=CritiqueReport(status='blocked', reviewer='test'),
                run_dir=tmp_path, evidence=evidence, producer='test')
    assert critique_path.read_bytes() == saved_critique
    assert _validated_manuscript_documents(tmp_path) == (None, [])


def test_video_wording_with_verified_numeric_binding(tmp_path):
    from easyicu.research_agent.authority.evidence_store import EvidenceStore
    from easyicu.research_agent.reporting.manuscript_post import bind_numeric_values

    evidence = EvidenceStore(tmp_path)
    effect = evidence.register_json(kind='statistic', description='Primary association hazard ratio',
        payload={'registered_percent': 41}, filename='association.json',
        evidence_id='primary_association', producer='host')
    evidence.register_numeric_claim(value='41', canonical=41.0, evidence_id=effect.evidence_id,
        step_id='analysis', source_field='registered_percent')
    evidence.register_numeric_claim(value='28', canonical=28.0, evidence_id=effect.evidence_id,
        step_id='analysis', source_field='horizon_days')
    records = evidence.current_verified_records([])
    for sentence in ('X reduces 28-day mortality by 41%.',
                     'X降低死亡风险41%。', 'X减少死亡风险41%。'):
        bound, mapping, missing = bind_numeric_values(sentence, evidence=evidence)
        assert not missing and mapping
        assert _causal_delivery_blockers(bound=bound, evidence_records=records,
            run_dir=tmp_path, numeric_binding_map=mapping), sentence
    bound, mapping, missing = bind_numeric_values('X与较低死亡风险相关，差异为41%。', evidence=evidence)
    assert not missing and mapping
    assert not _causal_delivery_blockers(bound=bound, evidence_records=records,
        run_dir=tmp_path, numeric_binding_map=mapping)


def test_stale_wrapper_download_and_bundle_are_denied_after_failed_critique_write(tmp_path, monkeypatch):
    import hashlib
    import json
    import zipfile
    from io import BytesIO
    from easyicu.research_agent.authority import evidence_store
    from easyicu.research_agent.authority.evidence_store import EvidenceStore
    from easyicu.research_agent.reporting import write_phase
    from easyicu.research_agent.schema import CritiqueReport
    from easyicu.webserver import agent_runs, agent_pipeline_runs

    wrapper = tmp_path / 'wrapper'
    run = wrapper / 'pipeline' / 'synthetic_run'
    run.mkdir(parents=True)
    (wrapper / 'run_context.json').write_text(json.dumps({'run_id': 'synthetic_run'}))
    bound = run / 'manuscript_scaffold_bound.md'
    bound.write_text('Old safe draft')
    pdf = run / 'manuscript_scaffold.pdf'
    tex = run / 'manuscript_scaffold.tex'
    pdf.write_bytes(b'%PDF-old')
    tex.write_text('old tex')
    def row(path):
        return {'name': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    receipt = run / 'manuscript_pdf_receipt.json'
    receipt.write_text(json.dumps({'schema_version':'easyicu.manuscript_pdf_receipt.v1', 'status':'rendered',
        'draft_watermark':True, 'security':{'network_allowed':False, 'shell_escape_allowed':False,
        'untrusted_input_mode':True}, 'source':row(tex), 'pdf':row(pdf), 'bibliography':None}))
    write_phase._bind_pdf_receipt_to_bound_manuscript(receipt_path=receipt, bound_path=bound,
        bound=bound.read_text(), run_dir=run)
    evidence = EvidenceStore(run)
    write_phase._persist_manuscript_critique(critique=CritiqueReport(status='pass', reviewer='test'),
        run_dir=run, evidence=evidence, producer='test')
    for path in (pdf, tex, receipt):
        (wrapper / path.name).write_bytes(path.read_bytes())
    loaded_receipt, current_docs = agent_pipeline_runs._validated_manuscript_documents(run)
    assert loaded_receipt['bound_manuscript_sha256'] == hashlib.sha256(bound.read_bytes()).hexdigest()
    assert {item['name'] for item in current_docs} == {pdf.name, tex.name}
    assert agent_runs.read_run_artifact_bytes(str(wrapper), pdf.name)['ok'] is True
    good_bundle = agent_runs.build_run_bundle(str(wrapper))
    assert good_bundle['ok'] is True
    bound.write_text('Changed draft with unsupported causal claim')
    with monkeypatch.context() as m:
        m.setattr(evidence_store, '_atomic_write_text', lambda *a, **kw: (_ for _ in ()).throw(OSError('write failed')))
        with pytest.raises(OSError):
            write_phase._persist_manuscript_critique(critique=CritiqueReport(status='blocked', reviewer='test'),
                run_dir=run, evidence=evidence, producer='test')
    assert json.loads((run / 'manuscript_critique.json').read_text())['status'] == 'pass'
    assert agent_pipeline_runs._validated_manuscript_documents(run) == (None, [])
    assert agent_runs.read_run_artifact_bytes(str(wrapper), pdf.name)['ok'] is False
    assert agent_runs.read_run_artifact(str(wrapper), receipt.name)['ok'] is False
    assert pdf.read_bytes() == b'%PDF-old' and (wrapper / pdf.name).read_bytes() == b'%PDF-old'
    bundle = agent_runs.build_run_bundle(str(wrapper))
    if bundle['ok']:
        with zipfile.ZipFile(BytesIO(bundle['content'])) as archive:
            assert not any(name.endswith(pdf.name) for name in archive.namelist())


def test_pdf_receipt_binding_rejects_changed_draft(tmp_path):
    import json
    from easyicu.research_agent.reporting.write_phase import _bind_pdf_receipt_to_bound_manuscript

    bound = tmp_path / 'manuscript_scaffold_bound.md'
    bound.write_text('current changed')
    receipt = tmp_path / 'manuscript_pdf_receipt.json'
    original = json.dumps({'schema_version': 'easyicu.manuscript_pdf_receipt.v1'})
    receipt.write_text(original)
    with pytest.raises(ValueError, match='bound_changed'):
        _bind_pdf_receipt_to_bound_manuscript(receipt_path=receipt, bound_path=bound,
            bound='earlier expected', run_dir=tmp_path)
    assert receipt.read_text() == original


def test_missing_critique_never_restores_current_or_legacy_pdf_when_bound_exists(tmp_path):
    import hashlib
    import json
    from io import BytesIO
    import zipfile
    from easyicu.webserver import agent_runs, agent_pipeline_runs

    wrapper = tmp_path / 'wrapper'
    run = wrapper / 'pipeline' / 'current_run'
    run.mkdir(parents=True)
    (wrapper / 'run_context.json').write_text(json.dumps({'run_id': 'current_run'}))
    pdf, tex, bound = (run / 'manuscript_scaffold.pdf', run / 'manuscript_scaffold.tex',
                       run / 'manuscript_scaffold_bound.md')
    pdf.write_bytes(b'%PDF-old')
    tex.write_text('old tex')
    bound.write_text('changed manuscript')
    def row(path):
        return {'name':path.name, 'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    receipt = {'schema_version':'easyicu.manuscript_pdf_receipt.v1', 'status':'rendered',
        'draft_watermark':True, 'security':{'network_allowed':False, 'shell_escape_allowed':False,
        'untrusted_input_mode':True}, 'source':row(tex), 'pdf':row(pdf), 'bibliography':None}
    for declared in (None, hashlib.sha256(bound.read_bytes()).hexdigest(), 'a' * 64):
        payload = dict(receipt)
        if declared is not None:
            payload['bound_manuscript_sha256'] = declared
        raw = json.dumps(payload).encode()
        (run / 'manuscript_pdf_receipt.json').write_bytes(raw)
        for path in (pdf, tex):
            (wrapper / path.name).write_bytes(path.read_bytes())
        (wrapper / 'manuscript_pdf_receipt.json').write_bytes(raw)
        assert agent_pipeline_runs._validated_manuscript_documents(run) == (None, [])
        assert agent_runs.read_run_artifact_bytes(str(wrapper), pdf.name)['ok'] is False
        bundle = agent_runs.build_run_bundle(str(wrapper))
        if bundle['ok']:
            with zipfile.ZipFile(BytesIO(bundle['content'])) as archive:
                assert not any(name.endswith(pdf.name) for name in archive.namelist())
    # A truly historical receipt has neither a critique nor a bound draft.
    bound.unlink()
    (run / 'manuscript_pdf_receipt.json').write_text(json.dumps(receipt))
    assert len(agent_pipeline_runs._validated_manuscript_documents(run)[1]) == 2
