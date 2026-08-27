/* Guided Copilot next-step owner.
   Converts one bounded, natural-language final block into host buttons. The
   host makes a demo-selection action explicit before sending it. Only the
   allowlisted demo/local-source controls below may project a one-turn grant;
   model-authored URLs, grant names, tool names, and HTML remain inert. */
(function () {
  'use strict';
  const { esc } = window.EU_HTML;

  const NEXT_STEP = /^\s*(?:#{1,6}\s*)?(?:\*\*|__)?(?:下一步|Next step)\s*[:：]\s*(?:\*\*|__)?\s*(.*)$/i;
  const CHOICE = /^\s*[-*]\s+(.+?)\s*$/;

  function plainLabel(value) {
    return String(value || '')
      .replace(/^\*\*(.+)\*\*$/, '$1')
      .replace(/^`(.+)`$/, '$1')
      .trim()
      .slice(0, 240);
  }

  function plainPrompt(value) {
    return String(value || '')
      .replace(/^(?:\*\*|__)/, '')
      .replace(/(?:\*\*|__)$/, '')
      .trim()
      .slice(0, 500);
  }

  function demoSubject(value) {
    const choice = plainLabel(value);
    const match = choice.match(/((?:mimic[\s-]*iv|eicu)[^。.;；]*?\bdemo(?:\s+data)?(?:\s*[（(][^）)]*[）)])?)/i);
    return match ? match[1].trim() : '';
  }

  function localSourceDatabase(value) {
    const choice = plainLabel(value);
    const localIntent = /(?:本机|本地|目录|文件夹|已经下载|已下载)|(?:local|folder|directory|already[\s-]+downloaded)/i.test(choice);
    if (!localIntent) return '';
    if (/mimic[\s-]*(?:iv|4)\b/i.test(choice)) return 'miiv';
    if (/mimic[\s-]*(?:iii|3)\b/i.test(choice)) return 'mimic';
    if (/\beicu\b/i.test(choice)) return 'eicu';
    if (/\b(?:amsterdamumcdb|aumc)\b/i.test(choice)) return 'aumc';
    if (/\bhirid\b/i.test(choice)) return 'hirid';
    if (/\b(?:sicdb|sic)\b/i.test(choice)) return 'sic';
    return '';
  }

  function choiceAction(value, language) {
    const choice = plainLabel(value);
    const consolidatedSetupAcceptance = language === 'zh'
      ? /^我(?:同意|确认|接受)/.test(choice)
        && /授权\s*EasyICU\s*准备数据/.test(choice)
      : /^(?:I\s+)?(?:agree to|approve|accept)\b/i.test(choice)
        && /authori[sz]e\s+EasyICU\s+to\s+prepare\s+(?:the\s+)?data/i.test(choice);
    if (consolidatedSetupAcceptance) {
      return {
        label: language === 'zh'
          ? '确认并准备数据'
          : 'Confirm and prepare data',
        message: choice,
        grants: [],
      };
    }
    const localDatabase = localSourceDatabase(choice);
    if (localDatabase) {
      return { label: choice, message: choice, grants: [], localDatabase };
    }
    const subject = demoSubject(choice);
    if (!subject) return { label: choice, message: choice, grants: [] };
    const declined = /(?:暂不|不要|不下载|无需|跳过|without|do not|don't|skip)/i.test(choice);
    if (declined) return { label: choice, message: choice, grants: [] };
    const localCopy = /(?:已经|已|本地).{0,8}(?:下载|准备|保存)|(?:选择|使用).{0,8}本地|already[\s-]+(?:downloaded|prepared)|local[\s-]+(?:copy|folder)/i.test(choice);
    if (localCopy) {
      return language === 'zh'
        ? {
            label: `使用已经下载好的 ${subject}`,
            message: `确认并授权本轮打开本地数据选择与扫描流程，使用已经下载好的 ${subject}；请直接打开目录选择，不要再次下载。`,
            grants: ['extract'],
          }
        : {
            label: `Use an already downloaded ${subject}`,
            message: `I confirm and authorize opening local data selection and scanning for the already downloaded ${subject} in this turn. Open the folder chooser and do not download it again.`,
            grants: ['extract'],
          };
    }
    return language === 'zh'
      ? {
          label: `下载并准备 ${subject}`,
          message: `确认并授权本轮准备并注册 ${subject}。`,
          grants: ['extract'],
        }
      : {
          label: `Download and prepare ${subject}`,
          message: `I confirm and authorize preparing and registering ${subject} for this turn.`,
          grants: ['extract'],
        };
  }

  function includeLocalDemoChoice(choices, language) {
    const rows = choices.slice(0, 4);
    const subject = rows.map(demoSubject).find(Boolean);
    if (!subject || rows.some(choice => /(?:已经|已|本地).{0,8}(?:下载|准备|保存)|already[\s-]+(?:downloaded|prepared)|local[\s-]+(?:copy|folder)/i.test(choice))) {
      return rows;
    }
    const localChoice = language === 'zh'
      ? `使用已经下载好的 ${subject}`
      : `Use an already downloaded ${subject}`;
    if (rows.length < 4) rows.splice(Math.min(1, rows.length), 0, localChoice);
    return rows.slice(0, 4);
  }

  function isDataSourceStep(step) {
    return Boolean(step && Array.isArray(step.choices) && step.choices.some((value) => {
      const choice = plainLabel(value);
      return Boolean(localSourceDatabase(choice) || demoSubject(choice))
        || /(?:数据来源|数据源|数据导出|数据库)|(?:data[\s-]+source|database|data[\s-]+export)/i.test(choice);
    }));
  }

  function isFormalPlanChoice(value) {
    const choice = plainLabel(value);
    return /(?:正式研究计划|研究计划|分析计划)|(?:formal\s+(?:research\s+)?plan|research\s+agent\s+(?:analysis\s+)?plan)/i.test(choice)
      && /(?:生成|开始|提交|授权)|(?:generate|start|submit|authorize)/i.test(choice);
  }

  function confirmedSourceLabel(authorization) {
    const source = authorization && authorization.source;
    const label = plainLabel(source && source.label) || 'EasyICU';
    const release = plainLabel(source && source.reference_release);
    if (!release || label.toLowerCase().includes(release.toLowerCase())) return label;
    return `${label} v${release}`;
  }

  function project(value) {
    const source = String(value == null ? '' : value).trim();
    if (!source) return null;
    const lines = source.split(/\r?\n/);
    let headingAt = -1;
    let headingText = '';
    for (let index = lines.length - 1; index >= 0; index -= 1) {
      const match = lines[index].match(NEXT_STEP);
      if (!match) continue;
      headingAt = index;
      headingText = plainPrompt(match[1]);
      break;
    }

    if (headingAt >= 0) {
      const promptLines = headingText ? [headingText] : [];
      const choices = [];
      for (const line of lines.slice(headingAt + 1)) {
        const match = line.match(CHOICE);
        if (match && choices.length < 4) {
          const label = plainLabel(match[1]);
          if (label && !choices.includes(label)) choices.push(label);
        } else if (line.trim() && !choices.length) {
          promptLines.push(plainPrompt(line));
        }
      }
      return {
        body: lines.slice(0, headingAt).join('\n').trim(),
        prompt: promptLines.join(' ').trim(),
        choices,
        explicit: true,
      };
    }

    return {
      body: source,
      prompt: '',
      choices: [],
      explicit: false,
      asking: /[?？](?:\*\*|__)?\s*$/.test(source),
    };
  }

  function render(step, options) {
    if (!step) return '';
    if (options && options.suppressFallback && !step.explicit) return '';
    const language = options && options.language === 'zh' ? 'zh' : 'en';
    const disabled = Boolean(options && options.disabled);
    const title = language === 'zh' ? '下一步' : 'Next step';
    const authorization = options && options.dataSourceAuthorization;
    const workflowActionCode = String(options && options.workflowActionCode || '');
    const prematurePlanChoice = workflowActionCode !== 'provider_ready_to_generate_plan'
      && step.choices.some(isFormalPlanChoice);
    if (prematurePlanChoice && authorization && authorization.status === 'confirmed') {
      const heading = language === 'zh' ? '先确认数据准备要求' : 'Confirm data-preparation requirements first';
      const detail = language === 'zh'
        ? '当前只确认数据准备所需的最少信息，不会提前生成研究计划。正式计划将在数据包与本地预检就绪后单独生成并交你审核。'
        : 'Only the minimum inputs for data preparation are confirmed here. The formal plan is generated and reviewed separately after the data package and local preflight are ready.';
      const action = disabled
        ? (language === 'zh' ? '正在汇总数据准备要求…' : 'Consolidating data-preparation requirements…')
        : (language === 'zh' ? '查看数据准备确认' : 'Review data-preparation confirmation');
      return `<section class="gpi-next-step is-resolved" data-gpi-premature-plan-guard aria-label="${heading}"><strong>${heading}</strong><p>${esc(detail)}</p><div class="gpi-next-actions"><button type="button" data-gpi-data-source-continue ${disabled ? 'disabled' : ''}><span>${action}</span><b aria-hidden="true">→</b></button></div></section>`;
    }
    if (authorization && authorization.status === 'confirmed' && isDataSourceStep(step)) {
      const source = confirmedSourceLabel(authorization);
      const heading = language === 'zh' ? '数据来源已确认' : 'Data source confirmed';
      const detail = language === 'zh'
        ? `已接入 ${source}。尚未开始数据提取或分析；EasyICU 只会先汇总数据准备所需的最少信息，不会在这里生成研究计划。`
        : `${source} is connected. Extraction and analysis have not started; EasyICU will confirm only the minimum data-preparation inputs here, not generate the research plan.`;
      const action = disabled
        ? (language === 'zh' ? '正在汇总数据准备要求…' : 'Consolidating data-preparation requirements…')
        : (language === 'zh' ? '查看数据准备确认' : 'Review data-preparation confirmation');
      return `<section class="gpi-next-step is-resolved" data-gpi-next-resolved-source aria-label="${heading}"><strong>${heading}</strong><p>${esc(detail)}</p><div class="gpi-next-actions"><button type="button" data-gpi-data-source-continue ${disabled ? 'disabled' : ''}><span>${action}</span><b aria-hidden="true">→</b></button></div></section>`;
    }
    const fallback = step.choices.length
      ? (language === 'zh'
        ? '请选择一个选项；EasyICU 会直接执行对应操作或继续对话。'
        : 'Choose an option; EasyICU will perform the action or continue the conversation.')
      : (step.asking
        ? (language === 'zh'
          ? '请在下方输入框回答这个问题。'
          : 'Answer this question in the composer below.')
        : (language === 'zh'
          ? '你可以继续补充要求，或让 EasyICU 进入下一项受治理步骤。'
          : 'Add a requirement or ask EasyICU to enter the next governed step.'));
    const prompt = step.prompt || fallback;
    const choices = includeLocalDemoChoice(step.choices, language);
    const customLabel = language === 'zh' ? '其他，我自己输入' : "Something else — I'll type it";
    const customPlaceholder = language === 'zh' ? '输入其他选择或补充说明' : 'Type another choice or add context';
    const customSend = language === 'zh' ? '发送' : 'Send';
    const customChoice = `
      <form class="gpi-next-custom" data-gpi-next-custom-form>
        <span>${customLabel}</span>
        <div class="gpi-next-custom-row">
          <input type="text" data-gpi-next-custom-input maxlength="12000" aria-label="${customLabel}" placeholder="${customPlaceholder}" ${disabled ? 'disabled' : ''} />
          <button type="submit" ${disabled ? 'disabled' : ''}><span>${customSend}</span></button>
        </div>
      </form>`;
    const controls = choices.length
      ? choices.map((choice) => {
          const action = choiceAction(choice, language);
          const grants = action.grants.length ? ` data-gpi-next-grants="${esc(action.grants.join(','))}"` : '';
          const localSource = action.localDatabase
            ? ` data-gpi-next-local-database="${esc(action.localDatabase)}"`
            : '';
          return `<button type="button" data-gpi-next-choice="${esc(action.message)}"${grants}${localSource} ${disabled ? 'disabled' : ''}><span>${esc(action.label)}</span><b aria-hidden="true">→</b></button>`;
        }).join('') + customChoice
      : `<button type="button" data-gpi-next-focus ${disabled ? 'disabled' : ''}>${step.asking ? (language === 'zh' ? '回答这个问题' : 'Answer this question') : (language === 'zh' ? '继续对话' : 'Continue')} <b aria-hidden="true">→</b></button>`;
    return `<section class="gpi-next-step" aria-label="${title}"><strong>${title}</strong><p>${esc(prompt)}</p><div class="gpi-next-actions">${controls}</div></section>`;
  }

  window.EU_GUIDED_PI_NEXT_ACTIONS = { project, render };
})();
