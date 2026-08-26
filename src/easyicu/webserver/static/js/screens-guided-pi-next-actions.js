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

  function choiceAction(value, language) {
    const choice = plainLabel(value);
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
    const language = options && options.language === 'zh' ? 'zh' : 'en';
    const disabled = Boolean(options && options.disabled);
    const title = language === 'zh' ? '下一步' : 'Next step';
    const fallback = step.choices.length
      ? (language === 'zh'
        ? '请选择一个选项；点击后将作为你的下一条消息发送。'
        : 'Choose an option; clicking it sends it as your next message.')
      : (step.asking
        ? (language === 'zh'
          ? '请在下方输入框回答这个问题。'
          : 'Answer this question in the composer below.')
        : (language === 'zh'
          ? '你可以继续补充要求，或让 EasyICU 进入下一项受治理步骤。'
          : 'Add a requirement or ask EasyICU to enter the next governed step.'));
    const prompt = step.prompt || fallback;
    const choices = includeLocalDemoChoice(step.choices, language);
    const controls = choices.length
      ? choices.map((choice) => {
          const action = choiceAction(choice, language);
          const grants = action.grants.length ? ` data-gpi-next-grants="${esc(action.grants.join(','))}"` : '';
          return `<button type="button" data-gpi-next-choice="${esc(action.message)}"${grants} ${disabled ? 'disabled' : ''}><span>${esc(action.label)}</span><b aria-hidden="true">→</b></button>`;
        }).join('')
      : `<button type="button" data-gpi-next-focus ${disabled ? 'disabled' : ''}>${step.asking ? (language === 'zh' ? '回答这个问题' : 'Answer this question') : (language === 'zh' ? '继续对话' : 'Continue')} <b aria-hidden="true">→</b></button>`;
    return `<section class="gpi-next-step" aria-label="${title}"><strong>${title}</strong><p>${esc(prompt)}</p><div class="gpi-next-actions">${controls}</div></section>`;
  }

  window.EU_GUIDED_PI_NEXT_ACTIONS = { project, render };
})();
