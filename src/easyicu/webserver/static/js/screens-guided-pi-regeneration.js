/* Guided Copilot in-place regeneration projection.
   The persisted transcript remains append-only; this owner only swaps the
   visible activity and assistant rows while a replacement branch streams. */
(function () {
  'use strict';

  const PLAN_ACTION = /^(?:重新生成研究计划|生成候选研究计划|生成正式研究计划|generate (?:a fresh|the candidate|the formal) research plan)[。.!！]?$/i;

  function isPlanActionText(value) {
    return PLAN_ACTION.test(String(value || '').trim());
  }

  function latestPlanRequest(rows) {
    const source = Array.isArray(rows) ? rows : [];
    for (let index = source.length - 1; index >= 0; index -= 1) {
      const row = source[index];
      if (
        !row || row.role !== 'user'
        || !String(row.entryId || '').trim()
        || !isPlanActionText(row.text)
      ) continue;
      let assistant = null;
      for (let cursor = index + 1; cursor < source.length; cursor += 1) {
        const item = source[cursor];
        if (item && item.role === 'user') break;
        if (item && item.role === 'assistant' && String(item.id || '').trim()) {
          assistant = item;
          break;
        }
      }
      return {
        userEntryId: String(row.entryId),
        // A terminal planning failure can project only a workflow card after
        // the user request while keeping the assistant receipt out of the
        // visible message list.  The persisted user entry is still the exact
        // branch coordinate; regenerate from it instead of appending a second
        // plan request below the failed conversation.
        targetMessageId: assistant ? String(assistant.id) : '',
      };
    }
    return null;
  }

  function create(rows, options) {
    const source = Array.isArray(rows) ? rows : [];
    const userEntryId = String(options && options.userEntryId || '');
    const requestedMessageId = String(options && options.targetMessageId || '');
    let targetIndex = requestedMessageId
      ? source.findIndex(row => String(row && row.id || '') === requestedMessageId && row.role === 'assistant')
      : -1;
    let userIndex = -1;

    if (targetIndex >= 0) {
      for (let index = targetIndex - 1; index >= 0; index -= 1) {
        const row = source[index];
        if (row && row.role === 'user') {
          userIndex = index;
          break;
        }
      }
    } else if (userEntryId) {
      userIndex = source.findIndex(row => row && row.role === 'user' && String(row.entryId || '') === userEntryId);
      if (userIndex >= 0) {
        targetIndex = source.findIndex((row, index) => index > userIndex && row && row.role === 'assistant');
      }
    }

    if (targetIndex < 0) return null;
    const target = source[targetIndex];
    let activity = null;
    for (let index = targetIndex - 1; index > userIndex; index -= 1) {
      if (source[index] && source[index].role === 'activity') {
        activity = source[index];
        break;
      }
    }
    const startedAt = Number(options && options.startedAt) || Date.now();
    return {
      targetMessageId: String(target.id || ''),
      targetActivityId: String(activity && activity.id || ''),
      message: {
        id: String(target.id || ''), role: 'assistant', text: '', complete: false,
        resources: [],
      },
      activity: {
        id: String(activity && activity.id || `regeneration-activity-${startedAt}`),
        role: 'activity', status: 'running', startedAt, steps: [], expanded: true,
      },
    };
  }

  function project(row, regeneration) {
    if (!row || !regeneration) return row;
    const id = String(row.id || '');
    if (id && id === regeneration.targetMessageId) return regeneration.message;
    if (id && id === regeneration.targetActivityId) return regeneration.activity;
    return row;
  }

  function visibleRows(rows, regeneration) {
    const source = Array.isArray(rows) ? rows : [];
    if (!regeneration) return source;
    const targetMessageId = String(regeneration.targetMessageId || '');
    const targetIndex = source.findIndex(row => (
      row && row.role === 'assistant' && String(row.id || '') === targetMessageId
    ));
    return targetIndex < 0 ? source : source.slice(0, targetIndex + 1);
  }

  window.EU_GUIDED_PI_REGENERATION = Object.freeze({
    create, project, visibleRows, isPlanActionText, latestPlanRequest,
  });
})();
