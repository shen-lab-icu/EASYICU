/* Guided Copilot new-session starter owner.
   Complete intents send one ordinary user message. Starters that require
   details only prefill and focus the composer; neither path grants data use. */
(function () {
  'use strict';
  const { esc } = window.EU_HTML;

  function button(attributes, title, note) {
    return `<button type="button" ${attributes}><strong>${esc(title)}</strong><small>${esc(note)}</small></button>`;
  }

  function render(options) {
    const tr = options.tr;
    const disabled = options.disabled ? 'disabled' : '';
    return `<section class="gpi-research-start" aria-label="${esc(tr('Start a research conversation', '开始研究对话'))}">
      <div class="gpi-research-start-copy">
        <h2>${esc(tr('Where would you like to start?', '你想从哪里开始？'))}</h2>
        <p>${esc(tr(
          'Start with a research question, relevant literature, or ICU data. EasyICU can help build cohorts, standardise features, prepare research plans, run statistical analyses, and produce figures and research reports.',
          '从研究问题、相关文献或 ICU 数据开始，EasyICU 可以协助完成队列构建、特征标准化、研究计划、统计分析、图表和研究报告。',
        ))}</p>
      </div>
      <div class="gpi-starter-actions">
        ${button(`class="primary" data-gpi-starter-compose="${esc(tr('My ICU research question is: ', '我的 ICU 研究问题是：'))}" ${disabled}`, tr('Start with a research question', '开始一个研究问题'), tr('Describe the population, disease, exposure, or outcome', '描述人群、疾病、暴露或结局'))}
        ${button(`data-gpi-starter-send="${esc(tr('I do not yet have a specific research question. Help me search the relevant literature and identify ICU research directions worth evaluating.', '我还没有明确的研究问题，请帮我检索相关文献，并筛选值得进一步评估的 ICU 研究方向。'))}" ${disabled}`, tr('Find a direction from the literature', '从文献寻找研究方向'), tr('Search related studies and screen candidate questions', '检索相关研究并筛选候选问题'))}
        ${button(`data-gpi-starter-send="${esc(tr('I want to start with ICU data. First ask what data I have and what I want to accomplish. Then help me build the cohort, standardise features, and continue with conversational analysis and figures after extraction. Ask me to confirm the data source only when the study needs to read it.', '我想从 ICU 数据开始。请先询问我现有的数据和希望完成的任务，再帮助我构建队列、标准化特征，并在提取后继续通过对话分析和绘图；只有研究需要读取数据时，再让我确认数据源。'))}" ${disabled}`, tr('Extract and analyse ICU data', '提取并分析 ICU 数据'), tr('Build cohorts, standardise features, then analyse and plot', '构建队列、标准化特征并继续分析绘图'))}
        ${button(`data-gpi-starter-compose="${esc(tr('I have a research idea to evaluate: ', '我想评估这个研究想法：'))}" ${disabled}`, tr('Evaluate a research idea', '评估一个研究想法'), tr('Check novelty, data feasibility, and study design', '检查新颖性、数据可行性和研究设计'))}
      </div>
      <small class="gpi-starter-privacy">${esc(tr(
        'Data stay local by default. EasyICU asks before reading or analysing them.',
        '数据默认保留在本机；需要读取或分析数据时，EasyICU 会先向你确认。',
      ))}</small>
    </section>`;
  }

  function actionFromEvent(event) {
    const compose = event.target.closest('[data-gpi-starter-compose]');
    if (compose) return { kind: 'compose', text: String(compose.dataset.gpiStarterCompose || '') };
    const send = event.target.closest('[data-gpi-starter-send]');
    if (send) return { kind: 'send', text: String(send.dataset.gpiStarterSend || '') };
    return null;
  }

  window.EU_GUIDED_PI_STARTERS = { render, actionFromEvent };
})();
