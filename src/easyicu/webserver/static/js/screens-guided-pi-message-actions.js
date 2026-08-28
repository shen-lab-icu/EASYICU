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

    function action(kind, label, icon, disabled) {
      return `<button type="button" data-gpi-message-${kind} title="${esc(label)}" aria-label="${esc(label)}" ${disabled ? 'disabled' : ''}>${iconHtml(icon, 14)}</button>`;
    }

    function render(row, options) {
      const text = String(row && row.text || '');
      const role = row && row.role;
      if (!text || row.complete === false || !['user', 'assistant'].includes(role)) {
        return { editorHtml: '', actionsHtml: '' };
      }
      const id = String(row.id || '');
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
      const retry = role === 'assistant' && options && options.canRetry && options.retryUserEntryId
        ? action('retry', tr('Regenerate response', '重新生成回答'), 'refresh', false)
        : '';
      return {
        editorHtml: '',
        actionsHtml: `<div class="gpi-message-actions" role="group" aria-label="${tr('Message actions', '消息操作')}">${copy}${edit}${retry}</div>`,
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

    function markCopied(button) {
      if (!button) return;
      const label = tr('Copied', '已复制');
      button.dataset.copied = 'true';
      button.title = label;
      button.setAttribute('aria-label', label);
      button.innerHTML = iconHtml('check', 14);
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

    function handleClick(event) {
      const copy = event.target.closest('[data-gpi-message-copy]');
      if (copy) {
        const row = messageById(articleId(copy));
        if (row) copyText(row.text).then(copied => { if (copied) markCopied(copy); });
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
      const retry = event.target.closest('[data-gpi-message-retry]');
      if (retry) {
        const user = precedingUser(articleId(retry));
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
        context.sendText(text);
        return true;
      }
      context.regenerate(entryId, text, 'user_edited_message', followingAssistantId(id));
      return true;
    }

    return { render, handleClick, handleSubmit };
  }

  window.EU_GUIDED_PI_MESSAGE_ACTIONS = { create };
})();
