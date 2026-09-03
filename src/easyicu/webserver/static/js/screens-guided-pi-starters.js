/* Guided Copilot new-session starter owner.
   Complete intents send one ordinary user message. Starters that require
   details only prefill and focus the composer; neither path grants data use. */
(function () {
  'use strict';
  const { esc } = window.EU_HTML;

  function button(attributes, title, note) {
    return `<button type="button" ${attributes}><strong>${esc(title)}</strong><small>${esc(note)}</small></button>`;
  }

  function dataFirstMessage(tr) {
    return tr(
      'I want to start from existing ICU data. First ask what data I have and what I want to accomplish. Do not read or analyse it until I confirm the source.',
      '我想从现有 ICU 数据开始。请先询问我有什么数据、希望完成什么任务；在我确认数据源前不要读取或分析。',
    );
  }

  function render(options) {
    const tr = options.tr;
    const disabled = options.disabled ? 'disabled' : '';
    return `<section class="gpi-research-start" aria-label="${esc(tr('Start a research conversation', '开始研究对话'))}">
      <div class="gpi-research-start-copy">
        <h2>${esc(tr('Where would you like to start?', '你想从哪里开始？'))}</h2>
        <p>${esc(tr(
          'Start with one sentence, a PDF, or an article URL. Copilot will mine candidate directions and review the literature in the conversation.',
          '可以只说一句话，也可以附加 PDF 或粘贴文章链接；Copilot 会在对话中发掘候选方向并审阅文献。',
        ))}</p>
      </div>
      <div class="gpi-starter-actions">
        ${button(`class="primary" data-gpi-starter-intent="idea_discovery_entry" data-gpi-starter-send="${esc(tr('I do not have a specific research direction yet. Help me choose whether to start from a clinical uncertainty, an article, or existing ICU data. Do not read data or create a study plan yet.', '我目前还没有具体研究方向。请先帮我选择从临床困惑、已有文章还是现有 ICU 数据开始；现在不要读取数据或生成研究方案。'))}" ${disabled}`, tr('Help me find a direction', '我还没有方向'), tr('Choose the easiest starting point first', '先选择最容易提供的起点'))}
        ${button(`data-gpi-starter-intent="idea_mining_entry" data-gpi-starter-compose="${esc(tr('Use Idea Mining to explore this vague idea: ', '请用 Idea Mining 帮我发掘这个模糊想法：'))}" ${disabled}`, tr('Explore a vague idea', '发掘一个模糊想法'), tr('Start from a clinical observation or uncertainty', '从临床现象或困惑开始'))}
        ${button(`data-gpi-starter-intent="implement_scientific_question" data-gpi-starter-compose="${esc(tr('This is my decided scientific question. Do not run Idea Mining; enter the governed research-plan and data-preparation workflow: ', '这是我已经明确的研究问题。请不要进行 Idea Mining，进入研究方案与数据准备流程：'))}" ${disabled}`, tr('I have a clear research question', '我已有明确研究问题'), tr('Continue to the research plan and data preparation', '进入研究方案与数据准备'))}
        ${button(`data-gpi-starter-intent="data_first_entry" data-gpi-starter-send="${esc(dataFirstMessage(tr))}" ${disabled}`, tr('Start from existing ICU data', '从现有 ICU 数据开始'), tr('Understand the data and goal before reading it', '先了解数据与目标，再确认是否读取'))}
      </div>
      <small class="gpi-starter-privacy">${esc(tr(
        'Data stay local by default. EasyICU asks before reading or analysing them.',
        '数据默认保留在本机；需要读取或分析数据时，EasyICU 会先向你确认。',
      ))}</small>
    </section>`;
  }

  function actionFromEvent(event) {
    const compose = event.target.closest('[data-gpi-starter-compose]');
    if (compose) return { kind: 'compose', text: String(compose.dataset.gpiStarterCompose || ''), intent: String(compose.dataset.gpiStarterIntent || '') };
    const send = event.target.closest('[data-gpi-starter-send]');
    if (send) return { kind: 'send', text: String(send.dataset.gpiStarterSend || ''), intent: String(send.dataset.gpiStarterIntent || '') };
    return null;
  }

  function actionFromDiscoveryChoice(value, tr) {
    const choice = String(value || '').trim().replace(/[。.]$/, '');
    if (choice === '从临床困惑开始' || choice === 'Start from a clinical uncertainty') {
      return {
        kind: 'compose',
        text: tr(
          'Use Idea Mining to discover research directions from this clinical uncertainty: ',
          '请用 Idea Mining 从这个临床困惑中发掘研究方向：',
        ),
        intent: 'idea_mining_entry',
      };
    }
    if (choice === '从已有文章或 PDF 开始' || choice === 'Start from an article or PDF') {
      return {
        kind: 'compose',
        text: tr(
          'Mine research directions from the article or PDF I attach: ',
          '请从我附加的文章或 PDF 中发掘研究方向：',
        ),
        intent: 'idea_mining_entry',
      };
    }
    if (choice === '从现有 ICU 数据开始' || choice === 'Start from existing ICU data') {
      return { kind: 'send', text: dataFirstMessage(tr), intent: 'data_first_entry' };
    }
    return null;
  }

  window.EU_GUIDED_PI_STARTERS = { render, actionFromEvent, actionFromDiscoveryChoice };
})();
