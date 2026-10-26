/* Guided Pi demo traceability cards.
   Owner: the walkthrough's stops, rendered from the registered run's trace
   record (assets/demo/sofa2-run-trace.json, built by
   tools/build_demo_walkthrough_trace.py): the data dictionary behind the
   question's words, the literature behind the plan, the researcher's
   confirmation, the chain from a reported number to its evidence, and how
   far the result may be read. Rendering only; the walkthrough owns playback
   and selection state. Every value shown comes from the trace record. */
(function () {
  'use strict';

  const { esc } = window.EU_HTML;

  const TERMS = {
    phenotype: ['Experimental SOFA-2 sepsis phenotype', '实验性 SOFA-2 脓毒症表型'],
    death: ['In-hospital death', '院内死亡'],
    adult: ['Adult', '成人'],
  };
  // Reader names for the upper levels of the tree; leaves keep the
  // dictionary's own description.
  const CONCEPTS = {
    sep3_sofa2: ['Experimental SOFA-2 sepsis phenotype', '实验性 SOFA-2 脓毒症表型'],
    sofa2: ['SOFA-2 total score', 'SOFA-2 总分'],
    sofa2_resp: ['Respiratory', '呼吸'],
    sofa2_coag: ['Hemostasis', '凝血'],
    sofa2_liver: ['Liver', '肝脏'],
    sofa2_cardio: ['Cardiovascular', '心血管'],
    sofa2_cns: ['Brain', '神经'],
    sofa2_renal: ['Kidney', '肾脏'],
    susp_inf: ['Suspected infection', '疑似感染'],
    abx: ['Antibiotics', '抗生素'],
    samp: ['Body-fluid sampling', '体液标本采集'],
    death: ['In-hospital death', '院内死亡'],
    age: ['Age', '年龄'],
  };
  const DESIGN_ELEMENTS = {
    reporting: ['reporting', '报告'], time_zero: ['time zero', '时间零点'],
    exposure: ['exposure', '暴露'], estimand: ['estimand', '估计目标'],
  };
  const LIMITATIONS = {
    post_baseline_exposure_opportunity_unresolved: [
      'The phenotype is assessed over the first 24 hours after ICU admission, after the start of follow-up.',
      '表型在入 ICU 后 24 小时内才判定，晚于随访起点。',
    ],
  };
  const COLUMNS = {
    stays: ['n_rows', ['ICU stays', 'ICU 入住']],
    share: ['exposure_pct', ['Share', '占比']],
    deaths: ['outcome_events', ['Deaths', '死亡']],
    rate: ['outcome_rate_pct', ['In-hospital mortality', '院内死亡率']],
  };

  function create(deps) {
    const { tr, icon, button } = deps;
    const pair = value => tr(value[0], value[1]);
    const count = value => Number(value).toLocaleString('en-US');
    const pct = value => `${Number(value).toFixed(2)}%`;
    const short = (value, size) => String(value || '').slice(0, size || 8);
    const ids = value => [].concat(value == null ? [] : value);

    function card(kind, kicker, title, body, foot) {
      return `<section class="gpi-trace-card" data-gpi-trace-card="${esc(kind)}">
        <header><span class="gpi-trace-kicker">${icon('link', 13)}${esc(kicker)}</span><h3>${esc(title)}</h3></header>
        ${body}
        ${foot ? `<footer class="gpi-trace-foot">${foot}</footer>` : ''}
      </section>`;
    }
    function digest(label, value) {
      return `<span class="gpi-trace-digest">${esc(label)} <code>${esc(short(value, 9))}</code></span>`;
    }

    // ---- data dictionary -------------------------------------------------
    function sourceText(source) {
      const table = source.table || '';
      if (source.sub_var === 'itemid' && source.ids != null) return `${table} · itemid ${ids(source.ids).join(', ')}`;
      if (source.regex) return `${table}.${source.val_var || source.sub_var || ''} · ${tr('drug-name pattern', '药名匹配')}`;
      if (source.val_var) return `${table}.${source.val_var}${source.index_var ? ` · ${tr('time', '时间')} ${source.index_var}` : ''}`;
      return table;
    }
    function conceptNode(code, concepts, depth, seen) {
      const entry = concepts[code] || { missing: true };
      const name = CONCEPTS[code] ? pair(CONCEPTS[code]) : String(entry.description || '');
      const head = `<code>${esc(code)}</code><span>${esc(name)}</span>${entry.unit ? `<em>${esc(ids(entry.unit)[0])}</em>` : ''}`;
      const sources = (entry.miiv || []).map(source => `<li class="gpi-dict-source" title="${esc(source.callback || '')}">${icon('db', 12)}<span>${esc(sourceText(source))}</span></li>`).join('');
      const children = (entry.children || []).filter(child => !seen.has(child));
      if (!children.length) {
        const none = entry.children ? '' : `<p class="gpi-dict-none">${esc(tr('No MIMIC-IV source in this dictionary', '字典里没有 MIMIC-IV 来源'))}</p>`;
        return `<li class="gpi-dict-leaf"><div class="gpi-dict-head">${head}</div>${sources ? `<ul class="gpi-dict-sources">${sources}</ul>` : none}</li>`;
      }
      const next = new Set(seen).add(code);
      return `<li><details${depth < 1 ? ' open' : ''}><summary class="gpi-dict-head">${head}<span class="gpi-dict-count">${children.length}</span></summary>
        <ul>${children.map(child => conceptNode(child, concepts, depth + 1, next)).join('')}</ul>
        ${sources ? `<ul class="gpi-dict-sources">${sources}</ul>` : ''}</details></li>`;
    }
    // A term opens straight onto what its concept is made of, or its sources.
    function termBody(code, concepts) {
      const entry = concepts[code] || {};
      const children = (entry.children || []).map(child => conceptNode(child, concepts, 0, new Set([code]))).join('');
      const sources = (entry.miiv || []).map(source => `<li class="gpi-dict-source" title="${esc(source.callback || '')}">${icon('db', 12)}<span>${esc(sourceText(source))}</span></li>`).join('');
      return `${children}${sources ? `<li class="gpi-dict-leaf"><ul class="gpi-dict-sources">${sources}</ul></li>` : ''}`;
    }
    function termMeta(term, trace) {
      const entry = trace.dictionary.concepts[term.concept] || {};
      if (term.id === 'adult') {
        const rule = (trace.cohort.inclusion || []).find(row => row.concept === term.concept);
        return rule ? `${esc(term.concept)} ${esc(rule.op)} ${esc(rule.value)}` : '';
      }
      if (entry.clinical_status === 'experimental') {
        return `${esc(tr('experimental', '实验性'))} · ${esc(entry.definition_source || '')}`;
      }
      return '';
    }
    function dictionaryCard(trace) {
      const { dictionary, sources } = trace;
      const terms = dictionary.terms.map((term, index) => `<li class="gpi-dict-term"><details${index === 0 ? ' open' : ''}>
        <summary><b>${esc(pair(TERMS[term.id] || [term.id, term.id]))}</b>${icon('arrow', 12)}<code>${esc(term.concept)}</code><span class="gpi-trace-muted">${termMeta(term, trace)}</span></summary>
        <ul class="gpi-dict-tree">${termBody(term.concept, dictionary.concepts)}</ul>
      </details></li>`).join('');
      return card('dictionary', tr('Traceable · data dictionary', '可追溯 · 数据字典'),
        tr('Which MIMIC-IV data each term in the question uses', '问题里的每个词，用的是 MIMIC-IV 的哪些数据'),
        `<ul class="gpi-dict-terms">${terms}</ul>`,
        `${digest(tr('Dictionary', '字典'), sources.concept_dict.commit)}${digest('SOFA-2', sources.sofa2_dict.commit)}<span>${esc(tr('the versions this run fingerprinted', '这次运行锁定的版本'))}</span>`);
    }

    // ---- plan literature -------------------------------------------------
    function literatureCard(trace, ctx) {
      const primary = trace.plan.steps.find(step => step.role === 'primary') || {};
      const papers = new Map(trace.literature.map(row => [row.key, row]));
      const bindings = trace.plan.bindings.filter(row => row.step_id === primary.step_id);
      const items = bindings.map(row => {
        const paper = papers.get(row.key) || {};
        const elements = (row.elements || []).map(name => `<span class="gpi-trace-chip">${esc(pair(DESIGN_ELEMENTS[name] || [name, name]))}</span>`).join('');
        return `<li>
          <a href="${esc(paper.source_url || '#')}" target="_blank" rel="noopener noreferrer">${esc(paper.title || row.key)}</a>
          <span class="gpi-trace-muted">${esc([paper.venue, paper.year, paper.pmid ? `PMID ${paper.pmid}` : ''].filter(Boolean).join(' · '))}</span>
          <p>${elements}${esc(tr(row.application, row.application_zh))}</p>
          ${row.divergence ? `<p class="gpi-trace-muted">${esc(tr('Not adopted: ', '未照搬：'))}${esc(tr(row.divergence, row.divergence_zh))}</p>` : ''}
        </li>`;
      }).join('');
      const review = trace.plan.review;
      const findings = review.findings.map(row => `<li>${esc(tr(row.message, row.message_zh))}</li>`).join('');
      const rest = trace.literature.length - new Set(bindings.map(row => row.key)).size;
      const more = rest > 0 && ctx.resource ? button(ctx.resource('literature_evidence.json'), tr(`${rest} more references`, `其余 ${rest} 篇文献`)) : '';
      return card('literature', tr('Traceable · literature', '可追溯 · 文献依据'),
        tr('Why the main analysis (step 3) is designed this way', '主分析（第 3 步）为什么这样设计'),
        `<ol class="gpi-trace-papers">${items}</ol>
        <details class="gpi-trace-review"><summary>${esc(tr('Plan review', '计划科学审阅'))} <b>${esc(review.score)}/100</b> · ${esc(review.approval_allowed ? tr('ready to run', '可以执行') : tr('not ready', '不能执行'))} · ${esc(tr(`${review.findings.length} major limitations`, `${review.findings.length} 项主要局限`))}</summary><ul>${findings}</ul></details>`,
        more);
    }

    // ---- confirmation ----------------------------------------------------
    function confirmCard(trace, durationText) {
      const span = trace && trace.spans && trace.spans['review-wait'];
      const waited = span ? durationText(Date.parse(span[0]), Date.parse(span[1])) : '';
      return `<section class="gpi-trace-card gpi-walk-confirm" data-gpi-trace-card="confirm">
        <header><h3>${esc(tr('The analysis starts only after you confirm the plan', '确认计划后，分析才会开始'))}</h3></header>
        ${waited ? `<p>${esc(tr(`In the recorded run the researcher reviewed the plan for ${waited} before confirming.`, `在这次真实运行里，研究者看了 ${waited} 才确认。`))}</p>` : ''}
        <div class="gpi-walk-confirm-actions"><button class="btn primary" type="button" data-gpi-walk="confirm">${esc(tr('Confirm plan and prepare data', '确认方案并准备数据'))}</button></div>
      </section>`;
    }

    // ---- number to evidence ------------------------------------------------
    function orderedRows(trace) {
      const rows = trace.evidence.result_rows;
      return [rows.find(row => row.level === 1), rows.find(row => row.level === 0), rows.find(row => row.role === 'overall')].filter(Boolean);
    }
    function groupName(row) {
      if (row.role === 'overall') return tr('All stays', '全部入住');
      return row.level === 1 ? tr('With the phenotype', '有表型') : tr('Without the phenotype', '无表型');
    }
    function cellValue(row, column) {
      if (column === 'stays') return count(row.stays);
      if (column === 'share') return pct(row.share_pct);
      if (column === 'deaths') return count(row.deaths);
      return pct(row.rate_pct);
    }
    function arithmetic(row, column) {
      if (column === 'stays') return tr(`${count(row.stays)} of ${count(row.denominator)} ICU stays`, `${count(row.denominator)} 个 ICU 入住中的 ${count(row.stays)} 个`);
      if (column === 'share') return `${count(row.stays)} ÷ ${count(row.denominator)} = ${pct(row.share_pct)}`;
      if (column === 'deaths') return tr(`${count(row.deaths)} in-hospital deaths among ${count(row.outcome_denominator)} stays`, `${count(row.outcome_denominator)} 个入住中院内死亡 ${count(row.deaths)} 例`);
      return `${count(row.deaths)} ÷ ${count(row.outcome_denominator)} = ${pct(row.rate_pct)}`;
    }
    function selectionKey(selection) { return selection ? `${selection.target}` : ''; }
    function resultTable(trace, selection) {
      const head = ['group', 'stays', 'share', 'deaths', 'rate'].map(column => `<th scope="col">${esc(column === 'group' ? tr('Group', '分组') : pair(COLUMNS[column][1]))}</th>`).join('');
      const body = orderedRows(trace).map((row, index) => `<tr><th scope="row">${esc(groupName(row))}</th>${Object.keys(COLUMNS).map(column => {
        const target = `cell:${index}:${column}`;
        const selected = selectionKey(selection) === target;
        return `<td><button type="button" class="gpi-walk-cell${selected ? ' is-selected' : ''}" data-gpi-walk="select" data-gpi-walk-target="${target}" aria-pressed="${selected}">${esc(cellValue(row, column))}</button></td>`;
      }).join('')}</tr>`).join('');
      return `<div class="gpi-demo-table gpi-walk-table"><table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table></div>`;
    }
    function chainNode(kind, title, detail, extra) {
      return `<li class="gpi-chain-node" data-kind="${esc(kind)}"><span class="gpi-chain-kind">${esc(title)}</span><div>${detail}${extra || ''}</div></li>`;
    }
    function codeNode(trace, record, id) {
      const excerpt = id === trace.evidence.chain[1] ? trace.evidence.code_excerpt : null;
      const code = excerpt ? `<details class="gpi-chain-code"><summary>${esc(tr('Show the code', '看代码'))}</summary><pre><code>${esc(excerpt.join('\n'))}</code></pre></details>` : '';
      return chainNode('code', tr('Code', '代码'),
        `<p>${esc(tr(`Step ${Number(record.step.slice(0, 2))} · deterministic host code; the plan sets its specification`, `第 ${Number(record.step.slice(0, 2))} 步 · 宿主确定性代码，规格来自计划`))}</p>
        <p class="gpi-chain-ids"><code>${esc(id)}</code>${digest('sha256', record.sha256)}</p>`, code);
    }
    function tableNode(record, id, column, row, note) {
      const where = note ? ` · ${note}` : column ? tr(` · column ${COLUMNS[column][0]}, row ${row.role === 'overall' ? 'overall' : `exposure = ${row.level}`}`, ` · 列 ${COLUMNS[column][0]}，行 ${row.role === 'overall' ? 'overall' : `exposure = ${row.level}`}`) : '';
      return chainNode('table', tr('Result table', '结果表'),
        `<p><code>${esc(record.file)}</code>${esc(where)}</p>
        <p class="gpi-chain-ids"><code>${esc(id)}</code>${digest('sha256', record.sha256)}</p>`);
    }
    function cohortNode(trace) {
      return chainNode('cohort', tr('Analysis cohort', '分析队列'),
        `<p>${esc(tr(`${count(trace.cohort.n_stays)} adult ICU stays · columns sep3_sofa2_max, death`, `${count(trace.cohort.n_stays)} 个成人 ICU 入住 · 列 sep3_sofa2_max、death`))}</p>
        <p class="gpi-chain-ids">${digest(tr('cohort', '队列'), trace.cohort.cohort_sha256)}${digest(tr('MIMIC-IV export', 'MIMIC-IV 导出'), trace.cohort.export_manifest_sha256)}</p>`,
        `<button type="button" class="gpi-link" data-gpi-walk="dictionary">${esc(tr('Open the data dictionary', '打开数据字典'))}</button>`);
    }
    function chainHtml(trace, selection) {
      const records = trace.evidence.records;
      const [tableId, codeId, figureId, sourceId, figureCodeId] = trace.evidence.chain;
      const target = selectionKey(selection);
      let title;
      let nodes;
      if (target === 'figure') {
        title = tr('Where the figure comes from', '这张图从哪来');
        nodes = [
          chainNode('figure', tr('Figure', '图'), `<p>${esc(tr('The image above is the registered file', '上面这张图就是登记的文件'))} <code>${esc(records[figureId].file)}</code></p><p class="gpi-chain-ids"><code>${esc(figureId)}</code>${digest('sha256', records[figureId].sha256)}</p>`),
          tableNode(records[sourceId], sourceId, null, null, tr('data of panel B', 'B 图的数据')),
          codeNode(trace, records[figureCodeId], figureCodeId),
          tableNode(records[tableId], tableId),
          codeNode(trace, records[codeId], codeId),
          cohortNode(trace),
        ];
      } else {
        const match = /^cell:(\d+):(\w+)$/.exec(target) || ['', '0', 'rate'];
        const row = orderedRows(trace)[Number(match[1])] || orderedRows(trace)[0];
        const column = COLUMNS[match[2]] ? match[2] : 'rate';
        title = tr(`Where ${cellValue(row, column)} comes from`, `${cellValue(row, column)} 从哪来`);
        nodes = [
          chainNode('number', tr('Arithmetic', '计算'), `<p>${esc(arithmetic(row, column))}</p>`),
          tableNode(records[tableId], tableId, column, row),
          codeNode(trace, records[codeId], codeId),
          cohortNode(trace),
        ];
      }
      return `<div class="gpi-chain" aria-live="polite"><h4>${esc(title)}</h4><ol>${nodes.join('')}</ol></div>`;
    }
    function resultCard(trace, selection, demo) {
      const figureSelected = selectionKey(selection) === 'figure';
      return card('result', tr('Traceable · result evidence', '可追溯 · 结果证据'),
        tr('Click any number or the figure to see where it comes from', '点表格里的任意数字或图，看它从哪来'),
        `<figure class="gpi-run-answer-figure gpi-walk-figure">
          <button type="button" class="gpi-walk-figure-button${figureSelected ? ' is-selected' : ''}" data-gpi-walk="select" data-gpi-walk-target="figure" aria-pressed="${figureSelected}">
            <img src="assets/demo/sofa2-phenotype-mortality.png?v=20260922-demo2" alt="${esc(tr('Phenotype share and in-hospital mortality by phenotype', '表型占比与各组院内死亡率'))}" decoding="async">
          </button>
          <figcaption>${esc(demo.figureCaption())}</figcaption>
        </figure>
        ${resultTable(trace, selection)}
        ${chainHtml(trace, selection)}`,
        `<span>${esc(tr('Run', '运行'))} <code>${esc(trace.source_run_id)}</code></span><span>${esc(tr(`${trace.evidence.count} evidence records`, `${trace.evidence.count} 条证据`))}</span>`);
    }

    // ---- reading the result --------------------------------------------------
    function boundaryCard(trace) {
      const primary = trace.plan.steps.find(step => step.role === 'primary') || {};
      const open = (primary.limitations || []).map(code => `<li><b>${esc(tr('Open limitation', '未解决的局限'))}</b>${esc(LIMITATIONS[code] ? pair(LIMITATIONS[code]) : code)}</li>`).join('');
      return card('boundary', tr('Traceable · claim boundary', '可追溯 · 结论边界'),
        tr('How far this result can be read', '这个结果能读到哪一步'),
        `<ul class="gpi-trace-boundary">
          <li><b>${esc(tr('Descriptive only', '仅描述性'))}</b>${esc(tr('Counts and proportions; no adjusted estimate, test or confidence interval.', '只有计数和比例，没有调整估计、检验或置信区间。'))}</li>
          ${open}
        </ul>`,
        `<span>${esc(tr('The plan’s claim ceiling', '来自计划的结论上限'))}</span><code>${esc(primary.claim_ceiling || '')}</code>`);
    }
    function followUpsCard(questions, asked) {
      const rows = questions.map((text, index) => `<li><button type="button" class="gpi-walk-followup${asked === index ? ' is-selected' : ''}" data-gpi-walk="followup" data-gpi-walk-index="${index}" aria-pressed="${asked === index}"><span aria-hidden="true">↳</span>${esc(text)}</button>
        ${asked === index ? `<p class="gpi-trace-muted" role="status">${esc(tr('Connect a model to ask this in your own project.', '连接模型后，就能在你自己的项目里接着问。'))}</p>` : ''}</li>`).join('');
      return `<section class="gpi-followups gpi-walk-followups" data-gpi-trace-card="followups"><div class="gpi-followups-head">${esc(tr('Follow-up questions', '可以继续问'))}</div><ul>${rows}</ul></section>`;
    }

    return Object.freeze({ dictionaryCard, literatureCard, confirmCard, resultCard, boundaryCard, followUpsCard, chainHtml });
  }

  window.EasyICU.guidedPi.declare('demoTrace', { create });
})();
