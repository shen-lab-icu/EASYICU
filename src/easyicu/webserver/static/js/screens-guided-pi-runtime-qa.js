/* Owner: current-version browser QA receipts for isolated workspace pages. */
(function () {
  'use strict';
  function failureText(error) {
    const zh = window.EU_LANG === 'zh';
    const detail = error && error.details || {};
    const diagnosis = detail.runtime_qa_contract || detail.contract_diagnostics || detail;
    const issues = Array.isArray(diagnosis.issues) ? diagnosis.issues.filter(issue => issue && typeof issue === 'object').slice(0, 16) : [];
    const rawCode = error && error.code || diagnosis.code || '';
    const code = /^[a-z0-9_]{1,80}$/.test(rawCode) ? rawCode : '';
    const lead = issues.length || ['runtime_qa_contract_invalid', 'pi_workspace_runtime_contract_invalid'].includes(code)
      ? (zh ? '检查合同格式错误，浏览器检查尚未运行。' : 'The check contract is invalid; browser checks have not run.')
      : (zh ? '浏览器检查未能启动；此页尚未验证。' : 'Browser checks could not start; this page is unverified.');
    const paths = issues.slice(0, 3).map(issue => {
      const path = typeof issue.path === 'string' ? issue.path.slice(0, 240) : '';
      const expected = Array.isArray(issue.expected) ? issue.expected.join(', ') : String(issue.expected || '');
      return `${path || '/'}: ${expected.slice(0, 180)}`;
    });
    const more = issues.length > 3 ? (zh ? `另有 ${issues.length - 3} 处合同错误。` : `${issues.length - 3} more contract errors.`) : '';
    return [lead, code, ...paths, more].filter(Boolean).join(' ');
  }
  function create(currentKey, options) {
    let key = '', nonce = '', frame = null, slot = null, source = null, report = null;
    let generation = 0;
    const publish = options && options.publish;
    function reset() {
      generation += 1;
      if (slot) slot.replaceChildren();
      key = ''; nonce = ''; frame = null; slot = null; source = null; report = null;
    }
    function prepare(value, request) {
      reset();
      if (!value || !request || !/^[a-f0-9]{64}$/.test(request.nonce || '')) return '';
      key = value; nonce = request.nonce;
      return nonce;
    }
    const active = () => key && key === currentKey() && frame && frame.isConnected && slot && slot.isConnected;
    function display() {
      if (!slot) return;
      const node = document.createElement('span');
      node.setAttribute('role', 'status'); node.dataset.gpiRuntimeQa = report ? report.status : 'partial';
      const zh = window.EU_LANG === 'zh';
      node.textContent = !report
        ? (zh ? '浏览器检查待完成' : 'Browser checks pending')
        : report.contract_diagnostics && report.contract_diagnostics.status === 'invalid'
          ? (zh ? '检查合同格式错误，浏览器检查尚未运行' : 'Invalid check contract; browser checks have not run')
          : report.status === 'failed'
          ? (zh ? '页面运行检查未通过' : 'Page runtime checks failed')
          : report.status === 'complete'
            ? (zh ? '声明的页面检查已通过' : 'Declared page checks passed')
            : (zh ? '页面仅完成部分检查' : 'Page checks partially complete');
      if (report && report.csv_export_ready) node.textContent += zh ? ' · CSV 可保存' : ' · CSV ready to save';
      node.title = zh ? '仅声明的界面一致性检查；不代表科学验证或文件已保存。' : 'Declared UI consistency only; not scientific verification or proof of a saved file.';
      const parts = [node];
      if (report) {
        const meanings = {
          number_mismatch: zh ? '显示数值与声明不一致' : 'Displayed value differs from the contract',
          ratio_mismatch: zh ? '比例与分子、分母不一致' : 'Ratio differs from numerator and denominator',
          bar_geometry_mismatch: zh ? '柱形尺寸与声明比例尺不一致' : 'Bar size differs from its declared scale',
          select_cases_mismatch: zh ? '筛选后的联动结果不符合声明' : 'Filtered values differ from the contract',
          select_cases_partial: zh ? '筛选检查还有未完成项' : 'Filtered checks remain pending',
          csv_export_not_yet_observed: zh ? '尚未观察到 Host 接受可保存 CSV' : 'Host has not accepted a CSV ready to save',
          download_control_disabled: zh ? '下载控件不可用' : 'Download control is disabled',
          selector_not_unique: zh ? '选择器未定位到唯一元素' : 'Selector did not identify one element',
          number_missing: zh ? '元素中未找到数值' : 'No number found in the element',
          non_finite_number: zh ? '显示了无效数值' : 'A nonfinite number was displayed',
          zero_denominator: zh ? '分母为零，尚不能核对比例' : 'Ratio is pending because the denominator is zero',
          suspicious_percent_bar: zh ? '百分比柱过短；缺少比例尺合同，需进一步核对' : 'Short percentage bar needs a declared scale check',
          non_finite_display: zh ? '数值辅助属性无效' : 'A numeric accessibility attribute is nonfinite',
        };
        const incomplete = (report.checks || []).filter(check => check.status !== 'passed');
        const warnings = report.warnings || [];
        const missing = report.missing_check_ids || [];
        const diagnostic = report.contract_diagnostics;
        const contractIssues = diagnostic && Array.isArray(diagnostic.issues) ? diagnostic.issues.filter(issue => issue && typeof issue === 'object').slice(0, 16) : [];
        if (incomplete.length || warnings.length || missing.length || contractIssues.length || report.runtime_errors || !report.coverage || !report.coverage.length) {
          const details = document.createElement('details'), summary = document.createElement('summary');
          summary.textContent = zh ? '查看检查范围与问题' : 'Check coverage and issues';
          details.appendChild(summary);
          const list = document.createElement('ul');
          for (const issue of contractIssues) {
            const item = document.createElement('li');
            const expected = Array.isArray(issue.expected) ? issue.expected.join(', ') : String(issue.expected || '');
            item.textContent = `${String(issue.path || '/').slice(0, 240)}: ${expected.slice(0, 300)}`;
            list.appendChild(item);
          }
          for (const check of incomplete) {
            const item = document.createElement('li');
            item.textContent = `${check.id}: ${check.code}${meanings[check.code] ? ' — ' + meanings[check.code] : ''}`;
            if (Number.isFinite(check.observed) && Number.isFinite(check.expected)) item.textContent += ` (${check.observed} / ${check.expected})`;
            list.appendChild(item);
          }
          for (const warning of warnings) {
            const item = document.createElement('li'); item.textContent = `${warning}${meanings[warning] ? ' — ' + meanings[warning] : ''}`; list.appendChild(item);
          }
          for (const id of missing) {
            const item = document.createElement('li'); item.textContent = `${id}: ${zh ? '尚无浏览器观察' : 'No browser observation yet'}`; list.appendChild(item);
          }
          if (report.runtime_errors) {
            const item = document.createElement('li'); item.textContent = `runtime_errors: ${report.runtime_errors}`; list.appendChild(item);
          }
          if (!report.coverage || !report.coverage.length) {
            const item = document.createElement('li');
            item.textContent = diagnostic && diagnostic.status === 'invalid'
              ? (zh ? '先修复合同格式；不要轮询状态或修改数据来凑通过。' : 'Repair the contract first; do not poll or alter data to force a pass.')
              : (zh ? '未声明可核对的检查合同；交互、图表和下载尚未验证。' : 'No declared check contract; interactions, charts, and downloads remain unverified.');
            list.appendChild(item);
          }
          details.appendChild(list); parts.push(details);
        }
      }
      slot.replaceChildren(...parts);
    }
    function bind(targetFrame, targetSlot, coordinates) {
      frame = targetFrame; slot = targetSlot; source = coordinates;
      display();
    }
    async function receive(event) {
      const data = event.data;
      if (!active() || event.source !== frame.contentWindow || event.origin !== 'null'
        || !data || data.type !== 'easyicu:runtime-qa' || data.nonce !== nonce
        || !data.report || data.report.source_sha256 !== source.sha256) return;
      let bytes;
      try { bytes = new TextEncoder().encode(JSON.stringify(data.report)).length; } catch (_) { return; }
      if (bytes > 24576 || typeof publish !== 'function') return;
      const ticket = generation, currentFrame = frame, currentNonce = nonce;
      const coordinates = { ...source };
      let result;
      try { result = await publish({ ...coordinates, nonce: currentNonce, report: data.report }); } catch (_) { return; }
      if (!active() || ticket !== generation || currentFrame !== frame || currentNonce !== nonce || !result) return;
      // Only the Host-normalized response controls the visible completion state.
      if (!['complete', 'partial', 'failed'].includes(result.status) || (result.source_sha256 && result.source_sha256 !== source.sha256)) return;
      report = result;
      display();
    }
    function noteCsvReady(value) {
      if (!active() || !value || value.file !== source.file || value.sha256 !== source.sha256) return;
      // The backend CSV owner records readiness; read it back through publish's
      // integration rather than upgrading an untrusted child report locally.
      if (options && typeof options.refresh === 'function') {
        const ticket = generation;
        options.refresh({ ...source }).then(result => {
          if (active() && ticket === generation && result && ['complete', 'partial', 'failed'].includes(result.status)) {
            report = result; display();
          }
        }).catch(() => {});
      }
    }
    window.addEventListener('message', receive);
    return { prepare, bind, reset, snapshot: () => report ? { ...report } : null, noteCsvReady, failureText };
  }
  window.EasyICU.guidedPi.declare('runtimeQa', { create, failureText });
})();
