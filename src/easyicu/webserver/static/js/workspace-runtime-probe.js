/* Injected only into the opaque workspace document. No Host or network API. */
(function (config) {
  'use strict';
  if (!config) return;
  window.__easyicuRuntimeQaRunning = true;
  const schema = 'easyicu.web-runtime-qa/1', results = [], warnings = [];
  let runtimeErrors = 0, sent = false;
  addEventListener('error', () => { runtimeErrors = Math.min(100, runtimeErrors + 1); });
  addEventListener('unhandledrejection', () => { runtimeErrors = Math.min(100, runtimeErrors + 1); });
  const settle = () => new Promise(resolve => {
    let done = false;
    const finish = () => { if (!done) { done = true; resolve(); } };
    setTimeout(finish, 120);
    requestAnimationFrame(() => requestAnimationFrame(finish));
  });
  function one(selector) {
    const nodes = document.querySelectorAll(selector);
    if (nodes.length !== 1) throw new Error('selector_not_unique');
    return nodes[0];
  }
  function number(selector) {
    const text = one(selector).textContent.trim().replace(/,/g, '').replace(/\u2212/g, '-');
    if (/NaN|Infinity/.test(text)) throw new Error('non_finite_number');
    const found = text.match(/[+-]?(?:\d+(?:\.\d*)?|\.\d+)/);
    if (!found || !Number.isFinite(Number(found[0]))) throw new Error('number_missing');
    return Number(found[0]);
  }
  function result(check, status, code, observed, expected) {
    const row = { id: check.id, kind: check.kind, status, code };
    if (Number.isFinite(observed)) row.observed = observed;
    if (Number.isFinite(expected)) row.expected = expected;
    results.push(row);
    return row;
  }
  async function run(check) {
    try {
      if (check.kind === 'text_number') {
        const observed = number(check.selector);
        return result(check, Math.abs(observed - check.expected) <= check.tolerance ? 'passed' : 'failed',
          Math.abs(observed - check.expected) <= check.tolerance ? 'number_matches' : 'number_mismatch', observed, check.expected);
      }
      if (check.kind === 'ratio') {
        const denominator = number(check.denominator), numerator = number(check.numerator);
        if (!denominator) return result(check, 'pending', 'zero_denominator');
        const observed = number(check.output), expected = check.scale * numerator / denominator;
        const passed = Math.abs(observed - expected) <= check.tolerance;
        return result(check, passed ? 'passed' : 'failed', passed ? 'ratio_matches' : 'ratio_mismatch', observed, expected);
      }
      if (check.kind === 'bar_geometry') {
        const element = one(check.selector), value = number(check.value_selector);
        const observed = element.getBoundingClientRect()[check.axis];
        const expected = Math.max(check.min_extent, value / check.domain_max * check.extent);
        const passed = Number.isFinite(observed) && Math.abs(observed - expected) <= check.tolerance_px;
        return result(check, passed ? 'passed' : 'failed', passed ? 'bar_geometry_matches' : 'bar_geometry_mismatch', observed, expected);
      }
      if (check.kind === 'download_control') {
        const element = one(check.selector);
        if (element.disabled || element.getAttribute('aria-disabled') === 'true') return result(check, 'failed', 'download_control_disabled');
        return result(check, 'pending', 'csv_export_not_yet_observed');
      }
      if (check.kind === 'select_cases') {
        const element = one(check.selector);
        if (element.tagName !== 'SELECT') return result(check, 'failed', 'select_required');
        const original = element.value;
        let status = 'passed';
        try {
          for (const scenario of check.cases) {
            if (!Array.from(element.options).some(option => option.value === scenario.value)) {
              status = 'failed';
              continue;
            }
            element.value = scenario.value;
            element.dispatchEvent(new Event('input', { bubbles: true }));
            element.dispatchEvent(new Event('change', { bubbles: true }));
            await settle();
            for (const child of scenario.checks) {
              const row = await run(child);
              if (!row || row.status === 'failed') status = 'failed';
              else if (row.status === 'pending' && status !== 'failed') status = 'pending';
            }
          }
        } finally {
          element.value = original;
          element.dispatchEvent(new Event('input', { bubbles: true }));
          element.dispatchEvent(new Event('change', { bubbles: true }));
          await settle();
        }
        return result(check, status, status === 'passed' ? 'select_cases_match' : status === 'failed' ? 'select_cases_mismatch' : 'select_cases_partial');
      }
      return result(check, 'failed', 'unsupported_check');
    } catch (error) {
      const known = ['selector_not_unique', 'non_finite_number', 'number_missing'];
      return result(check, 'failed', known.includes(error.message) ? error.message : 'dom_check_error');
    }
  }
  function observations() {
    // A missing contract cannot establish the intended chart scale. Surface
    // this as a warning, never a generic proof that every short bar is wrong.
    const nodes = document.querySelectorAll('[role="meter"],[role="progressbar"],[class~="bar"]');
    for (const element of Array.from(nodes).slice(0, 64)) {
      const text = element.textContent.trim();
      const percent = text.match(/^\s*(\d+(?:\.\d+)?)%\s*$/);
      const rect = element.getBoundingClientRect();
      if (percent && Number(percent[1]) >= 5 && rect.height > 0 && rect.height <= 2.5 && rect.width >= 10) {
        if (!warnings.includes('suspicious_percent_bar')) warnings.push('suspicious_percent_bar');
      }
    }
    const numericNodes = document.querySelectorAll('[aria-valuenow]');
    if (Array.from(numericNodes).slice(0, 64).some(element => !Number.isFinite(Number(element.getAttribute('aria-valuenow'))))) warnings.push('non_finite_display');
  }
  async function inspect() {
    if (sent) return;
    try {
      await settle();
      const checks = config.contract && config.contract.checks;
      if (checks) {
        for (const check of checks) await run(check);
      } else observations();
    } catch (_) {
      runtimeErrors = Math.min(100, runtimeErrors + 1);
    } finally {
      window.__easyicuRuntimeQaRunning = false;
      document.dispatchEvent(new CustomEvent('easyicu:runtime-qa-finished'));
    }
    if (sent) return;
    sent = true;
    parent.postMessage({ type: 'easyicu:runtime-qa-child', nonce: config.nonce,
      report: { schema, source_sha256: config.source_sha256, checks: results,
        runtime_errors: runtimeErrors, warnings, scientific_verified: false,
        browser_download_verified: false, csv_export_ready: false } }, '*');
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', inspect, { once: true });
  else inspect();
})(/*EASYICU_RUNTIME_QA_CONFIG*/null);
