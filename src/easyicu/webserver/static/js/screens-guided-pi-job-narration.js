/* Owner: the host's own sentences about one EasyICU job in the conversation.

   - The opening sentence before a job's trace says what EasyICU is about to
     do. It is a template keyed by the code that started the job: the host
     action of a transcript replay, or the tool result a live watch received.
   - The muted lines between trace rows say what changed, and only when a job
     event states it: a wait for the model service, a retry, steps already
     done before this run.
   - The run's notes paragraph counts the same receipts after the run.

   No model call writes any of it and nothing is a placeholder: a sentence is
   absent when no receipt states its fact. Decision-record lines (owner:
   screens-guided-pi-decision-records.js) keep the automatic continuations;
   this owner does not repeat them as trace narration. */
(function () {
  'use strict';

  // [codes, en, zh]: the coordinator's copy.
  const OPENINGS = [
    [['prepare_analysis_data', 'easyicu_full_run_upgrade_submitted'],
      'I will first prepare the data as the confirmed design states, then plan the analysis again on the prepared data.',
      '我先按确认的方案准备数据，然后在准备好的数据上重新规划分析。'],
    [['execute_plan', 'easyicu_review_submitted'],
      'I will run the analysis as the approved plan states and write up the results.',
      '我按批准的计划运行分析，并写出结果说明。'],
    [['retry_analysis'],
      'I will reuse the steps already completed, run the unfinished approved steps and check the results again.',
      '我复用已完成的步骤，补跑未完成的批准步骤，并重新校验结果。'],
    [['generate_plan', 'auto_generate_plan', 'easyicu_full_run_submitted'],
      'I will check the question and the data, then draft a research plan for your review; no analysis runs yet.',
      '我先核对研究问题和数据，再生成研究计划供你审阅；这一步不运行分析。'],
    [['auto_revise_plan'],
      'I will revise the research plan after its scientific checks; no analysis runs yet.',
      '我按科学检查的结果修订研究计划；这一步不运行分析。'],
    [['easyicu_full_run_resume_submitted'],
      'I will continue this approved run from its saved checkpoint.',
      '我从已保存的检查点继续这次获批的运行。'],
    [['easyicu_full_run_report_resume_submitted'],
      'I will restore the manuscript and evidence checks without re-running the analysis.',
      '我恢复稿件与证据校验，不重跑分析。'],
    [['easyicu_report_repair_submitted'],
      'I will rewrite the report from the sealed analysis results without re-running the analysis.',
      '我基于封存的分析结果重写报告，不重跑分析。'],
    [['easyicu_extraction_submitted'],
      'I will extract the study data from the source as configured and check its quality.',
      '我按确认的研究配置从数据源提取数据，并核查质量。'],
    [['easyicu_target_trial_compile_submitted', 'target-trial-compile'],
      'I will compile the target trial on this study’s data.',
      '我用这项研究的数据编译目标试验。'],
  ];

  function translate(tr) {
    return typeof tr === 'function' ? tr : (en, zh) => (window.EU_LANG === 'zh' ? zh : en);
  }
  function count(value) {
    const number = Number(value);
    return Number.isInteger(number) && number > 0 && number <= 100000 ? number : 0;
  }

  function opening(code, tr) {
    const value = String(code || '');
    const row = value && OPENINGS.find(([codes]) => codes.includes(value));
    return row ? translate(tr)(row[1], row[2]) : '';
  }

  // The provider-recovery owner names the continuation it waits for (n/limit).
  function waitLine(step, tr) {
    const t = translate(tr);
    const current = count(step && step.current);
    const total = count(step && step.total);
    if (current && total && current <= total) {
      return t(
        `The model service is temporarily unavailable; EasyICU waits for it and then continues from the checkpoint automatically (attempt ${current}/${total}).`,
        `模型服务暂时不可用，等待恢复后从检查点自动继续（第 ${current}/${total} 次）。`,
      );
    }
    // An older receipt names no attempt.
    return t('The model service is temporarily unavailable; EasyICU waits for it and then continues from the checkpoint automatically.',
      '模型服务暂时不可用，等待恢复后从检查点自动继续。');
  }
  // A model-transport retry the conversation runtime reported with its attempt number.
  function retryLine(step, tr) {
    const attempt = count(step && step.attempt);
    return attempt ? translate(tr)(`Retry ${attempt}.`, `重试第 ${attempt} 次。`) : '';
  }
  function precompletedLine(tr) {
    return translate(tr)('Preparation already finished; the checked data are used directly.', '准备阶段已完成，直接使用已核对的数据。');
  }
  // Steps a previous attempt of the same approved run completed.
  function reusedLine(tr) {
    return translate(tr)('Steps an earlier attempt completed are reused.', '上一次尝试已完成的步骤直接复用。');
  }

  // A wait for the model service and a label-less transport retry are
  // changes, not work: their host line takes the row's place.
  function replaces(step) {
    return Boolean(step) && ((step.kind === 'pipeline' && step.step === 'provider_recovery_wait')
      || (step.kind === 'retry' && !step.label && count(step.attempt) > 0));
  }

  /* Where a host line sits in a trace: a row the line replaces, or the line
     before a row. Returns null for an ordinary row. */
  function lineFor(step, state, tr) {
    if (!step) return null;
    if (replaces(step)) return { replaces: true, text: step.kind === 'retry' ? retryLine(step, tr) : waitLine(step, tr) };
    if (step.kind === 'pipeline' && step.step === 'plan_step' && step.precompleted) {
      const key = step.reused ? 'reused' : 'precompleted';
      if (state[key]) return null;
      state[key] = true;
      return { replaces: false, text: step.reused ? reusedLine(tr) : precompletedLine(tr) };
    }
    return null;
  }

  const PRECOMPLETED = /^Skipped step already completed by pre-execution:/;
  const REUSED = /^Skipped completed step from prior run:/;

  /* One short paragraph about how the run went, from its own receipts: the
     job's progress events and the run's gate. Empty when they record nothing
     worth saying. */
  function processNotes(job, latestRun, tr) {
    const t = translate(tr);
    const progress = window.EasyICU.guidedPi.jobProgress(job);
    const distinct = predicate => new Set(progress.filter(predicate)
      .map(event => [event.step, event.step_id, event.action, event.repair_kind, event.current, event.attempt,
        event.planning_unit, event.label].join('|'))).size;
    const resumes = distinct(event => event && event.step === 'provider_auto_resume');
    const corrections = distinct(event => event && event.retry_phase === 'rejected');
    const repairs = distinct(event => event && ['repair', 'repair_retry', 'deterministic_repair'].includes(String(event.action || '')));
    const labels = progress.map(event => String(event && event.label || ''));
    const precompleted = new Set(labels.filter(label => PRECOMPLETED.test(label))).size;
    const reused = new Set(labels.filter(label => REUSED.test(label))).size;
    const parts = [];
    if (resumes) parts.push(t(`The model service was interrupted; EasyICU continued from the checkpoint automatically ${resumes === 1 ? 'once' : `${resumes} times`}.`,
      `模型服务中断后，系统从检查点自动继续了 ${resumes} 次。`));
    if (corrections) parts.push(t(`The plan failed a validation check ${corrections === 1 ? 'once' : `${corrections} times`} and was corrected automatically.`,
      `计划有 ${corrections} 次没有通过校验，已自动修正。`));
    if (repairs) parts.push(t(`Analysis code was repaired ${repairs === 1 ? 'once' : `${repairs} times`} after a failed check or run.`,
      `分析代码在检查或运行失败后修复了 ${repairs} 次。`));
    if (precompleted) parts.push(t(`${precompleted} step(s) were already completed while the data were prepared and were not run again.`,
      `${precompleted} 个步骤已在准备数据时完成，没有重复运行。`));
    if (reused) parts.push(t(`${reused} step(s) completed by an earlier attempt were reused.`,
      `${reused} 个步骤沿用了上一次尝试的结果。`));
    if (latestRun && latestRun.present === true && latestRun.reportable !== true) {
      parts.push(t('The results are for analysis review only; publication use is not authorized.', '结果仅供分析审阅，未获投稿授权。'));
    }
    return parts.join(' ');
  }

  window.EasyICU.guidedPi.declare('jobNarration', { opening, lineFor, replaces, processNotes, waitLine, retryLine });
})();
