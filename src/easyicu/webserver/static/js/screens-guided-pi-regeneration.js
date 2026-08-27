/* Guided Copilot in-place regeneration projection.
   The persisted transcript remains append-only; this owner only swaps the
   visible activity and assistant rows while a replacement branch streams. */
(function () {
  'use strict';

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

  window.EU_GUIDED_PI_REGENERATION = Object.freeze({ create, project });
})();
