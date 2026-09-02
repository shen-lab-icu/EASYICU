/* Copilot Idea source owner.
   Reuses the existing local PDF/URL adapters and exposes only bounded source
   metadata to one conversational Idea Mining turn. Full PDF bytes are never
   stored in this module or sent to the model. */
(function () {
  'use strict';

  let pending = null;
  let reading = false;
  let error = '';

  function api() { return window.EU_API || {}; }
  function clone(value) { return value ? JSON.parse(JSON.stringify(value)) : null; }
  function firstUrl(text) {
    const match = String(text || '').match(/https?:\/\/[^\s<>"']+/i);
    return match ? match[0].replace(/[),.;，。；]+$/, '') : '';
  }
  function normalize(payload, fallback) {
    const data = payload || {};
    const suggested = data.suggested_payload || {};
    const source = data.resolved_source || {};
    const pdf = data.pdf || {};
    const sourceType = String(
      suggested.source_type || source.source_type || fallback.source_type || 'manual',
    );
    return {
      source_type: ['pdf', 'url', 'manual'].includes(sourceType) ? sourceType : 'manual',
      title: String(suggested.title || source.title || pdf.title || fallback.title || '').slice(0, 220),
      excerpt: String(suggested.excerpt || source.evidence_quote || fallback.excerpt || '').slice(0, 1200),
      journal: String(suggested.journal || source.journal || '').slice(0, 160),
      year: Number(suggested.year || source.year || 0) || null,
      doi: String(suggested.doi || source.doi || '').slice(0, 240),
      pmid: String(suggested.pmid || source.pmid || '').slice(0, 80),
      url: String(suggested.url || source.url || fallback.url || '').slice(0, 2048),
      source_file_name: String(pdf.filename || fallback.source_file_name || '').slice(0, 240),
      source_file_sha256: String(pdf.sha256 || fallback.source_file_sha256 || '').slice(0, 64) || null,
    };
  }
  function label(tr) {
    if (!pending) return '';
    if (pending.source_type === 'pdf') return pending.source_file_name || tr('Attached PDF', '已附加 PDF');
    return pending.title || pending.url || tr('Attached article', '已附加文章');
  }
  function status(options) {
    const { tr, esc } = options;
    if (!pending && !reading && !error) return '';
    return `<div class="gpi-idea-source-status${pending ? ' has-source' : ''}" aria-live="polite">
      ${reading ? `<span class="gpi-idea-source-reading"><span class="spin"></span>${esc(tr('Reading PDF…', '正在读取 PDF…'))}</span>` : ''}
      ${pending ? `<span class="gpi-idea-source-chip"><strong>${esc(label(tr))}</strong><small>${esc(tr('bounded source ready', '来源已就绪'))}</small><button type="button" data-gpi-idea-source-remove aria-label="${esc(tr('Remove source', '移除来源'))}">×</button></span>` : ''}
      ${error ? `<span class="gpi-idea-source-error" role="alert">${esc(error)}</span>` : ''}
    </div>`;
  }
  function controls(options) {
    const { tr, esc, icon, disabled } = options;
    return `<div class="gpi-idea-source">
      <input type="file" accept="application/pdf,.pdf" data-gpi-idea-pdf-input hidden ${disabled ? 'disabled' : ''} />
      <details class="gpi-idea-source-menu">
        <summary role="button" aria-haspopup="menu" aria-label="${esc(tr('Add source', '添加资料'))}" title="${esc(tr('Add source', '添加资料'))}">${icon('plus', 17)}</summary>
        <div class="gpi-idea-source-popover" role="menu" aria-label="${esc(tr('Add research source', '添加研究资料'))}">
          <button type="button" role="menuitem" data-gpi-idea-pdf-pick ${disabled || reading ? 'disabled' : ''}>
            ${icon('file', 17)}<span><strong>${esc(tr('Upload PDF', '上传 PDF'))}</strong><small>${esc(tr('Mine ideas from a paper', '从文章中发掘创新方向'))}</small></span>
          </button>
          <button type="button" role="menuitem" data-gpi-idea-url-focus ${disabled ? 'disabled' : ''}>
            ${icon('link', 17)}<span><strong>${esc(tr('Paste article link', '粘贴文章链接'))}</strong><small>${esc(tr('Paste it directly into the conversation', '直接粘贴到对话框中'))}</small></span>
          </button>
        </div>
      </details>
    </div>`;
  }
  function pick(host) {
    const input = host && host.querySelector('[data-gpi-idea-pdf-input]');
    if (input) input.click();
  }
  function readPdf(file, callbacks) {
    if (!file || reading) return;
    const tr = callbacks.tr;
    const name = String(file.name || 'source.pdf');
    if (!/\.pdf$/i.test(name) && file.type !== 'application/pdf') {
      error = tr('Choose a PDF file.', '请选择 PDF 文件。');
      callbacks.render();
      return;
    }
    if (!api().ingestIdeaPdf) {
      error = tr('PDF ingestion is unavailable.', 'PDF 解析暂不可用。');
      callbacks.render();
      return;
    }
    reading = true; error = ''; callbacks.render();
    const reader = new FileReader();
    reader.onload = async () => {
      try {
        const raw = String(reader.result || '');
        const contentBase64 = raw.includes(',') ? raw.split(',').pop() : raw;
        const payload = await api().ingestIdeaPdf({ filename: name, content_base64: contentBase64 });
        pending = normalize(payload, { source_type: 'pdf', source_file_name: name });
        if (callbacks.onReady) callbacks.onReady();
      } catch (caught) {
        error = caught && caught.message ? caught.message : String(caught);
      } finally {
        reading = false; callbacks.render();
      }
    };
    reader.onerror = () => {
      reading = false;
      error = tr('Could not read the selected PDF.', '无法读取所选 PDF。');
      callbacks.render();
    };
    reader.readAsDataURL(file);
  }
  async function prepareForMessage(text, intent) {
    if (pending) return clone(pending);
    const url = firstUrl(text);
    if (!url || intent !== 'idea_mining_entry' || !api().resolveIdeaSource) return null;
    const payload = await api().resolveIdeaSource({
      source_type: 'url', url, topic: String(text || '').slice(0, 600), allow_network: true,
    });
    pending = normalize(payload, { source_type: 'url', url });
    return clone(pending);
  }
  function handleClick(event, callbacks) {
    if (event.target.closest('[data-gpi-idea-pdf-pick]')) {
      const menu = event.target.closest('.gpi-idea-source-menu');
      if (menu) menu.removeAttribute('open');
      pick(callbacks.host()); return true;
    }
    if (event.target.closest('[data-gpi-idea-url-focus]')) {
      const menu = event.target.closest('.gpi-idea-source-menu');
      if (menu) menu.removeAttribute('open');
      const input = callbacks.host() && callbacks.host().querySelector('[data-gpi-input]');
      if (input) {
        input.focus();
        input.setSelectionRange(input.value.length, input.value.length);
        input.placeholder = callbacks.tr(
          'Paste an article URL, then optionally add what you want to discover.',
          '粘贴文章链接，也可以继续补充你想发掘的问题。',
        );
      }
      return true;
    }
    if (event.target.closest('[data-gpi-idea-source-remove]')) {
      pending = null; error = ''; callbacks.render(); return true;
    }
    return false;
  }
  function handleChange(event, callbacks) {
    if (!event.target.matches('[data-gpi-idea-pdf-input]')) return false;
    readPdf(event.target.files && event.target.files[0], callbacks);
    return true;
  }
  function hasSource() { return Boolean(pending); }
  function suggestsIdeaMining(text) { return hasSource() || Boolean(firstUrl(text)); }
  function consume() { const value = clone(pending); pending = null; error = ''; return value; }
  function reset() { pending = null; reading = false; error = ''; }

  window.EU_GUIDED_PI_IDEA_SOURCE = Object.freeze({
    consume, controls, handleChange, handleClick, hasSource, prepareForMessage, reset, status,
    suggestsIdeaMining,
  });
})();
