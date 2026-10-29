/* Owner: Guided Pi message-action widget. */
/* Guided Copilot message-action owner.
   Editing and regenerating both rewind to the target turn and create a
   recoverable Pi branch: an edited turn replaces everything after it rather
   than appending a new message to the bottom of the conversation. */
(function () {
  'use strict';
  const { esc } = window.EU_HTML;

  function create(context) {
    const tr = context.tr;
    const iconHtml = context.iconHtml;
    const unsafeRetryIds = new Set();
    const continuationTargets = new Map();
    const recoveryKinds = new Map();
    const liftedDrafts = new Map();

    function action(kind, label, icon, disabled) {
      return `<button type="button" data-gpi-message-${kind} title="${esc(label)}" aria-label="${esc(label)}" ${disabled ? 'disabled' : ''}>${iconHtml(icon, 14)}</button>`;
    }

    function render(row, options) {
      const text = String(row && row.text || '');
      const role = row && row.role;
      const id = String(row && row.id || '');
      liftedDrafts.delete(id);
      if (id && role === 'assistant' && options && typeof options.liftedDraftSource === 'string'
        && typeof options.liftedDraftRenderedText === 'string') {
        liftedDrafts.set(id, { source: options.liftedDraftSource, rendered: options.liftedDraftRenderedText });
      }
      if (!text || row.complete === false || !['user', 'assistant'].includes(role)) {
        return { editorHtml: '', actionsHtml: '' };
      }
      const completedFailure = role === 'assistant' && Boolean(row.errorCode || row.error_code || options && options.hostFailedAttempt)
        && Boolean(options && (options.workspaceMode || options.hasCompletedAction || options.hasSideEffectAttempt || options.canContinueReply || options.continuationReasonCode));
      const pendingContinuation = Boolean(options && options.continuationPending);
      if (completedFailure || row.replyContinuation) unsafeRetryIds.add(id);
      else unsafeRetryIds.delete(id);
      const continuationId = options && options.canContinueReply
        ? String(options.sourceErrorEntryId || '') : '';
      const recoveryKind = String(options && options.recoveryKind || 'continue_explanation');
      recoveryKinds.set(id, recoveryKind);
      if (completedFailure && continuationId) continuationTargets.set(id, continuationId);
      else continuationTargets.delete(id);
      const editing = role === 'user' && Boolean(options && options.editing);
      if (editing) {
        return {
          editorHtml: `<form class="gpi-message-editor" data-gpi-message-edit-form="${esc(id)}">
            <textarea rows="3" maxlength="12000" aria-label="${tr('Edit message', '编辑消息')}">${esc(text)}</textarea>
            <div class="gpi-message-editor-foot"><small>${tr('Replies after this message are replaced. The original branch stays recoverable in history.', '这条之后的回答会被替换；原分支仍保留在历史中可恢复。')}</small><span><button type="button" data-gpi-message-edit-cancel>${tr('Cancel', '取消')}</button><button type="submit">${tr('Send edit', '发送修改')}</button></span></div>
          </form>`,
          actionsHtml: '',
        };
      }

      const copy = action('copy', tr('Copy message', '复制消息'), 'copy', false);
      const edit = role === 'user' && options && options.allowEdit
        ? action('edit', tr('Edit and resend from here', '编辑并从这里重新发送'), 'edit', !options.canEdit)
        : '';
      const retry = role === 'assistant' && options && options.canRetry && options.retryUserEntryId && !completedFailure && !row.replyContinuation
        ? action('retry', tr('Regenerate response', '重新生成回答'), 'refresh', false)
        : '';
      const retryUnstarted = recoveryKind === 'retry_unstarted';
      const recoveryLabel = retryUnstarted ? tr('Retry task safely', '安全重试任务') : tr('Continue explanation', '继续完成说明');
      const recoveryNote = retryUnstarted
        ? tr('No file operation started. Retry uses the original model and remaining budget; the failed attempt stays in history.', '尚未开始文件操作；使用原模型和剩余预算重试，原失败记录保留。')
        : tr('File operations are complete. Continue only the explanation, without repeating those operations.', '文件操作已完成；只继续说明，不重复执行文件操作。');
      const continueReply = completedFailure && (continuationId || pendingContinuation)
        && typeof (retryUnstarted ? context.retryUnstarted : context.continueReply) === 'function'
        ? `<button type="button" class="gpi-message-recovery" ${retryUnstarted ? 'data-gpi-message-retry-unstarted' : 'data-gpi-message-continue-reply'} ${pendingContinuation ? 'disabled aria-busy="true"' : ''} title="${esc(recoveryNote)}" aria-label="${esc(recoveryLabel)}">${iconHtml('refresh', 14)}<span>${pendingContinuation ? (retryUnstarted ? tr('Retrying safely…', '正在安全重试…') : tr('Continuing explanation…', '正在继续完成说明…')) : recoveryLabel}</span></button>`
        : '';
      const blockedCode = completedFailure && !continuationId && !pendingContinuation && options
        ? String(options.continuationReasonCode || '') : '';
      const blockedReason = /budget_exhausted/.test(blockedCode) || options && options.budgetState === 'exhausted'
        ? tr('This message has no remaining budget. Completed files and execution records are preserved; this turn cannot continue. Review the existing artifacts and records.', '本条消息的预算已用尽。已完成的文件和执行记录已保留，本轮无法继续。请查看现有产物和记录。')
        : blockedCode === 'pi_reply_continuation_tool_outcome_unknown'
          ? tr('A file operation may have started but its outcome is unconfirmed. Review the execution record and file state before retrying.', '文件操作可能已经开始，但结果尚未确认。请先查看执行记录并核对文件状态，暂不重复执行。')
          : blockedCode === 'pi_reply_continuation_model_changed'
            ? tr('The current model differs from the failed attempt. Restore that model and refresh the conversation to check recovery.', '当前模型与失败时不同。请恢复原模型并刷新对话，重新核对恢复状态。')
            : blockedCode === 'pi_reply_continuation_terminal_state_unconfirmed'
              ? tr('The late model reply does not match the Host task outcome. Review the retained execution record; do not repeat the operation until that outcome is confirmed.', '模型的晚到回复与主机任务终态尚未核对一致。请先查看保留的执行记录，暂不重复执行。')
            : blockedCode === 'pi_reply_continuation_workspace_only'
              ? tr('The model response was interrupted. Review the execution record and confirm the operation state before retrying.', '模型回复已中断。请先查看执行记录并确认执行状态，再决定是否重试。')
            : blockedCode === 'pi_reply_continuation_budget_unknown'
              ? tr('The original budget cannot be verified. Refresh the conversation to check it again; automatic replay is unavailable.', '原消息的剩余预算尚无法核对。请刷新对话重新核对，暂不自动重放。')
              : blockedCode === 'pi_reply_continuation_no_completed_write'
                ? tr('The Host has not confirmed a safe retry for this attempt. Refresh the conversation and review its execution record.', '本次尝试尚未确认可安全重试。请刷新对话并查看执行记录。')
                : options && options.mutationState === 'unconfirmed'
                  ? tr('The operation state is unconfirmed. Review the execution record and file state before retrying.', '执行状态尚未核实。请先查看执行记录并核对文件状态，暂不重复执行。')
                : tr('Recovery is not available for this attempt. Refresh the conversation and review the retained execution record for its cause.', '本次尝试暂无可用的恢复操作。请刷新对话，并查看保留的执行记录了解原因。');
      const blockedNote = blockedCode
        ? `<small class="gpi-message-continuation-note">${esc(blockedReason)}</small>`
        : continueReply ? `<small class="gpi-message-continuation-note">${esc(recoveryNote)}</small>` : '';
      // Copy, edit-and-resend, and regenerate each reach an owner. Rating
      // buttons were removed: EasyICU keeps no feedback ledger, so a thumb
      // that only toggled a class would promise a record it never wrote.
      return {
        editorHtml: '',
        actionsHtml: `<div class="gpi-message-actions" role="group" aria-label="${tr('Message actions', '消息操作')}">${copy}${edit}${retry}${continueReply}</div>${blockedNote}`,
      };
    }

    async function copyText(value) {
      const text = String(value || '');
      try {
        if (window.navigator && window.navigator.clipboard && window.navigator.clipboard.writeText) {
          await window.navigator.clipboard.writeText(text);
          return true;
        }
      } catch (error) {}
      const document = window.document;
      if (!document || !document.body || typeof document.execCommand !== 'function') return false;
      const textarea = document.createElement('textarea');
      textarea.value = text;
      textarea.setAttribute('readonly', '');
      textarea.style.position = 'fixed';
      textarea.style.opacity = '0';
      document.body.appendChild(textarea);
      textarea.select();
      let copied = false;
      try { copied = document.execCommand('copy'); } catch (error) {}
      textarea.remove();
      return copied;
    }

    // An icon-only button stays icon-only; a labelled one keeps its label.
    function markCopied(button, labelled) {
      if (!button) return;
      const label = tr('Copied', '已复制');
      button.dataset.copied = 'true';
      button.title = label;
      button.setAttribute('aria-label', label);
      button.innerHTML = labelled ? `${iconHtml('check', 14)}<span>${esc(label)}</span>` : iconHtml('check', 14);
    }

    function messageById(id) {
      return context.rows().find(row => String(row.id || '') === String(id || ''));
    }

    function precedingUser(messageId) {
      let user = null;
      for (const row of context.rows()) {
        if (String(row.id || '') === String(messageId || '')) return user;
        if (row.role === 'user') user = row;
      }
      return null;
    }

    function articleId(target) {
      const article = target && target.closest('[data-gpi-message-id]');
      return article && article.dataset.gpiMessageId;
    }

    function focusEditor() {
      const host = context.host();
      const input = host && host.querySelector('[data-gpi-message-edit-form] textarea');
      if (!input) return;
      input.focus();
      input.setSelectionRange(input.value.length, input.value.length);
      input.scrollIntoView({ block: 'nearest' });
    }

    // A finished run's answer is the run-outcome card's text, as shown.
    function answerText(button) {
      const card = button && button.closest('[data-gpi-run-outcome]');
      const answer = card && (card.querySelector('.gpi-run-answer-text') || card.querySelector('.gpi-run-outcome-copy > p'));
      return answer ? String(answer.innerText || answer.textContent || '').trim() : '';
    }

    function handleClick(event) {
      const answerCopy = event.target.closest('[data-gpi-run-answer-copy]');
      if (answerCopy) {
        const text = answerText(answerCopy);
        if (text) copyText(text).then(copied => { if (copied) markCopied(answerCopy, true); });
        return true;
      }
      const copy = event.target.closest('[data-gpi-message-copy]');
      if (copy) {
        const row = messageById(articleId(copy));
        if (row) {
          const lifted = liftedDrafts.get(String(row.id || ''));
          const drafts = window.EasyICU.guidedPi.optional('researchDraft');
          const text = lifted && lifted.source === row.text && drafts && drafts.accepts(row)
            && typeof drafts.stripCopied === 'function'
            ? drafts.stripCopied(row.text, lifted.rendered) : row.text;
          copyText(text).then(copied => { if (copied) markCopied(copy); });
        }
        return true;
      }
      const edit = event.target.closest('[data-gpi-message-edit]');
      if (edit) {
        const row = messageById(articleId(edit));
        if (row && row.role === 'user' && context.canEdit()) {
          context.setEditing(row.id);
          context.renderHost();
          window.requestAnimationFrame(focusEditor);
        }
        return true;
      }
      if (event.target.closest('[data-gpi-message-edit-cancel]')) {
        context.setEditing('');
        context.renderHost();
        return true;
      }
      const retryUnstarted = event.target.closest('[data-gpi-message-retry-unstarted]');
      const continueReply = retryUnstarted || event.target.closest('[data-gpi-message-continue-reply]');
      if (continueReply) {
        const id = String(articleId(continueReply) || '');
        const target = continuationTargets.get(id);
        const kind = recoveryKinds.get(id);
        const callback = retryUnstarted && kind === 'retry_unstarted' ? context.retryUnstarted
          : !retryUnstarted && kind === 'continue_explanation' ? context.continueReply : null;
        if (target && typeof callback === 'function') callback(target, id);
        return true;
      }
      const retry = event.target.closest('[data-gpi-message-retry]');
      if (retry) {
        if (unsafeRetryIds.has(String(articleId(retry) || ''))) return true;
        const user = precedingUser(articleId(retry));
        // Retrying the answer to a host-owned plan action must invoke the same
        // governed plan starter as editing/resubmitting that action. Sending
        // it through generic conversation regeneration drops the one-turn
        // provider authority and lets the model misroute into setup dialogue.
        if (
          user && typeof context.resubmitHostGenerated === 'function'
          && context.resubmitHostGenerated(user, user.text)
        ) return true;
        if (user) context.regenerate(user.entryId, user.text, '', articleId(retry));
        return true;
      }
      return false;
    }

    function followingAssistantId(id) {
      const list = context.rows();
      const at = list.findIndex(row => String(row.id || '') === String(id || ''));
      if (at < 0) return '';
      const next = list.slice(at + 1).find(row => row.role === 'assistant');
      return next ? String(next.id || '') : '';
    }

    function handleSubmit(event) {
      const form = event.target.closest('[data-gpi-message-edit-form]');
      if (!form) return false;
      event.preventDefault();
      const input = form.querySelector('textarea');
      const text = String((input && input.value) || '').trim();
      if (!text) return true;
      const id = form.dataset.gpiMessageEditForm;
      const row = messageById(id);
      const entryId = String((row && row.entryId) || '').trim();
      context.setEditing('');
      // A host plan action can acquire a persisted transcript entry after a
      // refresh.  Route it by its governed action identity before using the
      // generic conversation-regeneration path; otherwise the model merely
      // replays the sentence and never starts a new Planner run.
      if (
        row && typeof context.resubmitHostGenerated === 'function'
        && context.resubmitHostGenerated(row, text)
      ) return true;
      if (!entryId) {
        // Other projected messages still fall back to an ordinary message so
        // an edit is never silently dropped.
        context.sendText(text, undefined, undefined, true, 'typed');
        return true;
      }
      context.regenerate(entryId, text, 'user_edited_message', followingAssistantId(id));
      return true;
    }

    return { render, handleClick, handleSubmit };
  }

  window.EasyICU.guidedPi.declare('messageActions', { create });
})();
