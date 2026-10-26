/* Owner: the shared artifact reader's view of a run's own record files. */
/* Reader views of a run's own record files: what the run was asked, its
   cohort, its verification, its source manifest, its withheld draft and its
   evidence ledger. Each view leads with what a researcher reads; identifiers,
   digests and host codes sit in one closed audit block, and the raw JSON stays
   in the preview's audit view. A file whose shape it does not know returns ''
   so the shared renderer keeps its generic table. */
(function () {
  const { esc } = window.EU_HTML;
  const t = window.t;

  const CHECKS = {
    execution_complete: ['Analysis steps executed', '分析步骤执行完成'],
    analysis_validated: ['Automated analysis validation', '自动分析校验'],
    evidence_complete: ['Every reported value cites its evidence', '每个报告值都引用了证据'],
    numeric_verified: ['Reported numbers match the results', '报告数值与结果一致'],
    manuscript_ready: ['Evidence-bound draft written', '已写出证据绑定的稿件'],
    publication_ready: ['Publication package complete', '投稿材料齐备'],
    paper_authorized: ['Paper authorized for this run', '本次运行已获论文授权'],
  };
  const RUN_TYPES = { full: ['Full run', '完整运行'], resume: ['Resumed run', '续跑'], dry_run: ['Dry run', '试运行'] };
  const DRAFT_STATUS = {
    locked_pending_human_review: ['Locked; awaiting human review', '已锁定，待人工审阅'],
    diagnostic_only: ['Diagnostic only', '仅诊断'],
  };
  const pick = row => (row ? t(row[0], row[1]) : '');
  const count = value => (Number.isFinite(Number(value)) && value !== null && value !== '' ? Number(value).toLocaleString() : '');
  const text = value => (typeof value === 'string' ? value.trim() : '');
  const list = value => (Array.isArray(value) ? value.filter(item => typeof item === 'string' && item.trim()) : []);

  function facts(rows) {
    const shown = rows.filter(row => row && row[1] !== '' && row[1] != null);
    if (!shown.length) return '';
    return `<dl class="ag-runfile-facts">${shown.map(([label, value, wide]) => `<div${wide ? ' class="is-wide"' : ''}><dt>${esc(label)}</dt><dd>${esc(value)}</dd></div>`).join('')}</dl>`;
  }
  function points(title, items) {
    const rows = list(items);
    return rows.length ? `<section class="ag-runfile-points"><h4>${esc(title)}</h4><ul>${rows.map(row => `<li>${esc(row)}</li>`).join('')}</ul></section>` : '';
  }
  function audit(helpers, rows) {
    const shown = rows.filter(row => row && row[1] !== '' && row[1] != null);
    if (!shown.length) return '';
    return helpers.artifactTable(t('Audit details', '审计信息'), [t('Field', '字段'), t('Value', '值')], shown, '', {
      disclosure: true, meta: t(`${shown.length} fields`, `${shown.length} 项`), formattedCells: true,
    });
  }
  function digest(value) {
    const raw = text(value);
    return /^[a-f0-9]{32,}$/i.test(raw) ? `${raw.slice(0, 12)}…` : raw;
  }
  function database(source) {
    const s = source && typeof source === 'object' ? source : {};
    const name = window.EU_DATABASE_NAMES.name(s.database) || text(s.database);
    return name || text(s.label);
  }
  function criterion(value) {
    const age = /^age range:\s*(\d+(?:\.\d+)?)\s*to\s*(\*|\d+(?:\.\d+)?)$/i.exec(value);
    if (!age) return value;
    return age[2] === '*' ? t(`Age ≥ ${age[1]}`, `年龄 ≥ ${age[1]}`) : t(`Age ${age[1]}–${age[2]}`, `年龄 ${age[1]}–${age[2]}`);
  }
  function wrap(name, body) {
    return `<div class="ag-artifact-readable ag-runfile" data-run-file="${esc(name)}">${body}</div>`;
  }

  function runContext(p, helpers) {
    const summary = p.summary && typeof p.summary === 'object' ? p.summary : {};
    const concepts = list(summary.selected_concepts).map(value => helpers.variable(value));
    const question = text(p.question);
    if (!question && !p.source && !concepts.length) return '';
    return wrap('run_context.json', `
      ${question ? `<blockquote class="ag-runfile-question"><span>${esc(t('Question this run answered', '本次运行回答的问题'))}</span><p>${esc(question)}</p></blockquote>` : ''}
      ${facts([
        [t('Data source', '数据来源'), database(p.source)],
        [t('Study variables', '研究变量'), concepts.join(t(', ', '、'))],
        [t('Run', '运行'), pick(RUN_TYPES[String(p.run_type || '')])],
      ])}
      ${audit(helpers, [
        [t('Run ID', '运行 ID'), text(p.run_id)],
        [t('Study ID', '研究 ID'), text(p.study_id)],
        [t('Prepared export', '准备后导出'), text(p.source && p.source.label)],
        [t('Scientific configuration', '科学配置摘要'), digest(p.scientific_configuration_sha256)],
        [t('Engine', '引擎'), text(p.engine)],
      ])}`);
  }

  function cohortSummary(p, helpers) {
    const cohort = p.cohort && typeof p.cohort === 'object' ? p.cohort : {};
    const summary = p.summary && typeof p.summary === 'object' ? p.summary : {};
    const hours = cohort.time_window_hours;
    const rows = [
      [t('Inclusion', '纳入条件'), list(cohort.criteria).map(criterion).join(t('; ', '；'))],
      [t('Exclusion', '排除条件'), list(cohort.exclusion_criteria).map(criterion).join(t('; ', '；'))],
      [t('Outcome', '研究结局'), text(cohort.target_outcome) ? helpers.variable(cohort.target_outcome) : ''],
      [t('Time window', '时间窗'), typeof hours === 'number' && Number.isFinite(hours) && hours > 0 ? t(`First ${hours} hours`, `前 ${hours} 小时`) : ''],
      [t('Study variables', '研究变量'), list(summary.selected_concepts).map(value => helpers.variable(value)).join(t(', ', '、'))],
    ];
    const body = facts(rows);
    return wrap('cohort_summary.json', body || `<p class="ag-runfile-empty">${esc(t('This cohort summary records no definition.', '这份队列摘要没有记录队列定义。'))}</p>`);
  }

  function qualityGate(p, helpers) {
    const gate = p.gate && typeof p.gate === 'object' ? p.gate : p;
    const checks = Array.isArray(gate.checks) ? gate.checks.filter(row => row && typeof row === 'object') : [];
    if (!checks.length) return '';
    const passed = row => row.passed === true || String(row.status || '').toLowerCase() === 'passed';
    const executed = checks.some(row => row.id === 'execution_complete' && passed(row));
    const verdict = gate.reportable === true
      ? t('Every check passed; the result may be reported.', '全部核验通过，结果可以报告。')
      : executed
        ? t('The analysis ran to completion; the checks below keep the result from being reported.', '分析已执行完成；下面未通过的核验使结果暂不可报告。')
        : t('The analysis has not completed every step.', '分析尚未完成全部步骤。');
    const items = checks.map(row => {
      const ok = passed(row);
      const id = String(row.id || row.name || '');
      const label = pick(CHECKS[id]) || helpers.gateCheckLabel(row);
      const reason = text(row.reason);
      const note = reason && reason !== `${id}_not_satisfied` ? `<small>${esc(reason)}</small>` : '';
      return `<li class="${ok ? 'is-pass' : 'is-open'}"><span aria-hidden="true">${ok ? '✓' : '–'}</span><div><strong>${esc(label)}</strong>${note}</div><em>${esc(ok ? t('Passed', '已通过') : t('Not met', '未满足'))}</em></li>`;
    }).join('');
    return wrap('quality_gate.json', `
      <p class="ag-runfile-lede">${esc(verdict)}</p>
      <ul class="ag-runfile-checks">${items}</ul>
      ${audit(helpers, [[t('Gate reason', '核验结论代码'), text(gate.reason)], [t('Gate status', '核验状态'), text(gate.status)]])}`);
  }

  function sourceManifest(p, helpers) {
    const r = p.readiness && typeof p.readiness === 'object' ? p.readiness : null;
    if (!r && p.evidence_count == null) return '';
    const readiness = r || {};
    const failed = list(readiness.failed_steps);
    const missing = list(readiness.missing_steps);
    const steps = readiness.execution_complete === true && !failed.length && !missing.length
      ? t('All steps completed', '全部步骤完成')
      : failed.length || missing.length
        ? t(`${failed.length} failed · ${missing.length} not run`, `${failed.length} 步失败 · ${missing.length} 步未运行`) : '';
    const provider = p.provider && typeof p.provider === 'object' ? p.provider : {};
    const open = list(readiness.analysis_errors).concat(list(readiness.display_suite_errors));
    return wrap('source_run_manifest.json', `
      ${facts([
        [t('Analysis steps', '分析步骤'), steps],
        [t('Result tables', '结果表'), count(p.result_table_count)],
        [t('Figures', '图'), count(p.figure_count)],
        [t('Evidence records', '证据记录'), count(p.evidence_count)],
        [t('Manuscript', '稿件'), readiness.manuscript_generated === true ? t('Written', '已生成') : readiness.manuscript_generated === false ? t('Not written', '未生成') : ''],
        [t('Model', '模型'), text(provider.model)],
      ])}
      ${points(t('Conditions not yet met', '尚未满足的条件'), open)}
      ${audit(helpers, [
        [t('Run ID', '运行 ID'), text(p.run_id)],
        [t('Run status', '运行状态'), text(p.status)],
        [t('Engine', '引擎'), text(p.engine)],
        [t('Provider gate', '模型授权检查'), text(provider.provider_gate)],
        [t('Host paths returned', '返回宿主路径'), p.path_values_returned === false ? t('No', '否') : p.path_values_returned === true ? t('Yes', '是') : ''],
      ])}`);
  }

  function manuscriptDraft(p, helpers) {
    const claims = Array.isArray(p.claims) ? p.claims.filter(row => row && typeof row === 'object') : [];
    const sentences = Array.isArray(p.sentences) ? p.sentences : [];
    if (!claims.length && !sentences.length && !text(p.markdown_preview)) return '';
    const reason = text(p.markdown_preview).split('\n').map(line => line.trim())
      .filter(line => line && !line.startsWith('#'));
    const generated = sentences.length > 0;
    return wrap('manuscript_draft.json', `
      <p class="ag-runfile-lede">${esc(generated
        ? t(`${sentences.length} evidence-bound sentences, locked until human review.`, `${sentences.length} 句证据绑定的稿件，人工审阅前保持锁定。`)
        : t('No formal manuscript was written for this run.', '这次运行没有生成正式稿件。'))}</p>
      ${facts([[t('Status', '状态'), pick(DRAFT_STATUS[String(p.status || '')]) || text(p.status)]])}
      ${points(t('Why', '原因'), reason)}
      ${audit(helpers, [[t('Run ID', '运行 ID'), text(p.run_id)], [t('Writer source', '写作来源'), text(p.source)]])}`);
  }

  function evidenceLedger(p, helpers) {
    const artifacts = Array.isArray(p.artifacts) ? p.artifacts.filter(row => row && typeof row === 'object') : [];
    if (!artifacts.length) return '';
    const privacy = p.privacy && typeof p.privacy === 'object' ? p.privacy : {};
    const clean = privacy.projection_scan_passed === true && privacy.patient_rows_in_projection === false && privacy.path_values_returned === false;
    return wrap('evidence_ledger.json', `
      ${facts([
        [t('Registered files', '登记文件'), count(artifacts.length)],
        [t('Evidence records', '证据记录'), count(p.pipeline_evidence_count)],
        [t('Browser copy', '浏览器副本'), clean ? t('Aggregates only; no patient rows or host paths', '仅聚合结果；无患者行与宿主路径') : ''],
      ])}
      ${helpers.artifactTable(t('Registered files', '登记文件'), [t('File', '文件'), t('SHA-256', 'SHA-256'), t('Size', '大小')],
        artifacts.map(row => [text(row.name || row.relative_path), digest(row.sha256), row.bytes == null ? '' : `${Number(row.bytes).toLocaleString()} B`]),
        '', { disclosure: true, meta: t(`${artifacts.length} files`, `${artifacts.length} 个文件`), formattedCells: true })}
      ${audit(helpers, [[t('Run ID', '运行 ID'), text(p.run_id)], [t('Engine', '引擎'), text(p.engine)]])}`);
  }

  const VIEWS = {
    'run_context.json': runContext,
    'cohort_summary.json': cohortSummary,
    'quality_gate.json': qualityGate,
    'source_run_manifest.json': sourceManifest,
    'manuscript_draft.json': manuscriptDraft,
    'evidence_ledger.json': evidenceLedger,
  };
  function view(name, payload, helpers) {
    const owner = VIEWS[String(name || '').toLowerCase()];
    return owner && payload && typeof payload === 'object' ? owner(payload, helpers) : '';
  }

  window.AGENT_RUN_FILE_READER = Object.freeze({ view });
})();
