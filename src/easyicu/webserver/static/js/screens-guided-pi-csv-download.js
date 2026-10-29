/* Owner: bounded CSV attachments from an explicitly enabled workspace preview. */
(function () {
  'use strict';
  function create(currentKey) {
    let key = '', nonce = '', frame = null, slot = null, coordinates = null;
    let count = 0, last = 0, pending = false, revision = 0, request = null;
    let epoch = '', passive = false, acknowledged = false, ackTimer = 0, selectionVersion = -1, readyReceived = false, optionsReceived = false;
    const urls = new Set();
    function clearDelivery() {
      revision += 1;
      if (request) request.abort();
      request = null;
      clearTimeout(ackTimer); ackTimer = 0;
      urls.forEach(url => window.URL.revokeObjectURL(url));
      urls.clear();
      if (slot) slot.replaceChildren();
      pending = false;
    }
    function randomNonce() {
      if (!window.crypto || !window.crypto.getRandomValues) return '';
      const bytes = new Uint8Array(32);
      window.crypto.getRandomValues(bytes);
      return Array.from(bytes, byte => byte.toString(16).padStart(2, '0')).join('');
    }
    function command(action, selection = null) {
      if (frame && frame.contentWindow && typeof frame.contentWindow.postMessage === 'function')
        frame.contentWindow.postMessage({ type: 'easyicu:workspace-csv-control', nonce, epoch, action,
          ...(selection ? { token: selection.token, selection_version: selection.version } : {}) }, '*');
    }
    function revoke() {
      command('disable'); epoch = ''; acknowledged = false; selectionVersion = -1; optionsReceived = false; clearDelivery();
    }
    function reset() {
      revoke();
      key = ''; nonce = ''; frame = null; slot = null; coordinates = null;
      count = 0; last = 0; passive = false; readyReceived = false;
    }
    function prepare(value, options = {}) {
      reset();
      if (!value || !(nonce = randomNonce())) return '';
      passive = options.passive === true;
      key = value;
      return nonce;
    }
    function active() { return key && key === currentKey() && frame && frame.isConnected && slot && slot.isConnected; }
    function bind(targetFrame, targetSlot, source) { frame = targetFrame; slot = targetSlot; coordinates = source; }
    function note(en, zh) {
      if (!slot) return;
      const status = document.createElement('span'); status.setAttribute('role', 'status');
      status.textContent = window.EU_LANG === 'zh' ? zh : en; slot.replaceChildren(status);
    }
    function generate() {
      if (!active() || !passive) return false;
      clearDelivery(); epoch = randomNonce(); acknowledged = false; selectionVersion = -1; optionsReceived = false;
      if (!epoch) return false;
      const sourceEpoch = epoch;
      note('Preparing CSV export…', '正在准备 CSV 导出…');
      command('generate');
      ackTimer = setTimeout(() => {
        if (active() && epoch === sourceEpoch && !acknowledged && !pending)
          note('Use the page’s CSV export button, then click the save link here.', '请点击页面里的 CSV 导出按钮，再点击此处的保存链接。');
      }, 10000);
      return true;
    }
    async function receive(event) {
      const data = event.data;
      if (!active() || event.source !== frame.contentWindow || event.origin !== 'null'
        || !data || data.nonce !== nonce) return;
      if (passive && data.type === 'easyicu:workspace-csv-ready-control') {
        if (!readyReceived) { readyReceived = true; if (epoch && !acknowledged) command('generate'); }
        return;
      }
      if (passive && (!epoch || data.epoch !== epoch)) return;
      if (data.type === 'easyicu:workspace-csv-ack') {
        if (acknowledged || !['generated', 'manual', 'qa_busy'].includes(data.status)) return;
        acknowledged = true; clearTimeout(ackTimer);
        if (!pending && data.status === 'manual')
          note('CSV export is unavailable. Generate the export options again, or use the page’s export button.', '暂时无法生成 CSV；请重新生成导出选项，或使用页面里的导出按钮。');
        if (!pending && data.status === 'qa_busy')
          note('Page checks are still running. Export the current filters after they finish.', '页面检查尚未完成；完成后请在页面里导出当前筛选。');
        return;
      }
      if (passive) {
        if (!Number.isSafeInteger(data.selection_version) || data.selection_version < 0
          || data.selection_version > 1000000 || data.selection_version < selectionVersion) return;
        selectionVersion = data.selection_version;
      }
      if (passive && data.type === 'easyicu:workspace-csv-options') {
        if (pending || urls.size || optionsReceived || !Array.isArray(data.options) || data.options.length < 2 || data.options.length > 8) return;
        const tokens = new Set();
        for (const option of data.options) {
          if (!option || typeof option.token !== 'string' || !/^[a-f0-9]{32}$/.test(option.token)
            || tokens.has(option.token) || typeof option.label !== 'string' || !option.label.trim()
            || option.label.length > 80 || /[\u0000-\u001f\u007f-\u009f\u202a-\u202e\u2066-\u2069]/.test(option.label)) return;
          tokens.add(option.token);
        }
        clearDelivery(); acknowledged = true; optionsReceived = true;
        const ticket = revision, sourceFrame = frame, sourceNonce = nonce, sourceEpoch = epoch, version = selectionVersion;
        const prompt = document.createElement('span'); prompt.setAttribute('role', 'status');
        prompt.textContent = window.EU_LANG === 'zh' ? '请选择 CSV 导出内容：' : 'Choose a CSV export: ';
        const buttons = data.options.map(option => {
          const button = document.createElement('button'); button.type = 'button';
          button.dataset.gpiCsvChoice = ''; button.textContent = option.label;
          button.addEventListener('click', event => {
            event.preventDefault(); event.stopPropagation();
            if (!tokens.has(option.token) || ticket !== revision || !active() || pending
              || frame !== sourceFrame || nonce !== sourceNonce || epoch !== sourceEpoch || selectionVersion !== version) return;
            tokens.clear(); clearDelivery(); acknowledged = false;
            note('Preparing the selected CSV export…', '正在生成所选 CSV…');
            command('select', { token: option.token, version });
          });
          return button;
        });
        slot.replaceChildren(prompt, ...buttons);
        return;
      }
      if (data.type === 'easyicu:workspace-csv-changed') {
        clearDelivery();
        note('Filters changed. Generate CSV again with the current filters.', '筛选已更改，请按当前筛选重新生成 CSV。');
        return;
      }
      if (pending || data.type !== 'easyicu:workspace-csv'
        || data.media_type !== 'text/csv' || typeof data.filename !== 'string'
        || !/^[A-Za-z0-9][A-Za-z0-9_.-]{0,115}\.csv$/.test(data.filename)
        || !data.filename.endsWith('.csv')
        || typeof data.csv !== 'string' || !data.csv.length || data.csv.length > 1048576
        || data.csv.includes('\0') || new TextEncoder().encode(data.csv).length > 1048576
        || count >= 5 || Date.now() - last < 1000) return;
      count += 1; last = Date.now(); pending = true;
      const ticket = revision, sourceFrame = frame, sourceNonce = nonce, sourceEpoch = epoch;
      const source = { ...coordinates }, controller = new AbortController();
      request = controller;
      const bound = () => ticket === revision && active() && frame === sourceFrame && nonce === sourceNonce && epoch === sourceEpoch;
      const current = () => bound() && !controller.signal.aborted;
      const status = document.createElement('span');
      status.setAttribute('role', 'status');
      status.textContent = window.EU_LANG === 'zh' ? '正在准备 CSV 下载…' : 'Preparing CSV download…';
      slot.replaceChildren(status);
      const bytes = new TextEncoder().encode(data.csv);
      let binary = '';
      for (let i = 0; i < bytes.length; i += 8192) binary += String.fromCharCode(...bytes.subarray(i, i + 8192));
      const fields = { file: source.file, expected_sha256: source.sha256,
        filename: data.filename, media_type: 'text/csv', csv_base64: window.btoa(binary) };
      try {
        // no-referrer makes a native POST form's Origin null. CORS-mode fetch
        // sends the real Host origin without weakening either CSRF boundary.
        const response = await window.fetch('/api/copilot/pi/projects/' + encodeURIComponent(source.projectId) + '/workspace/download-csv', {
          method: 'POST', mode: 'cors', credentials: 'same-origin', redirect: 'error',
          referrerPolicy: 'no-referrer', signal: controller.signal,
          body: new URLSearchParams(fields),
        });
        if (!current()) { controller.abort(); return; }
        if (!response.ok || response.headers.get('X-EasyICU-Source-SHA256') !== source.sha256
          || response.headers.get('Content-Type').split(';')[0] !== 'application/octet-stream') throw new Error('CSV response rejected');
        if (!response.body) throw new Error('CSV response missing');
        const reader = response.body.getReader(), chunks = [];
        let total = 0;
        for (;;) {
          const chunk = await reader.read();
          if (!current()) { controller.abort(); return; }
          if (chunk.done) break;
          const start = total;
          total += chunk.value.byteLength;
          if (total > 1048576 || total > bytes.length) { controller.abort(); throw new Error('CSV response too large'); }
          for (let i = 0; i < chunk.value.byteLength; i += 1) {
            if (chunk.value[i] !== bytes[start + i]) { controller.abort(); throw new Error('CSV response changed'); }
          }
          chunks.push(chunk.value);
        }
        // Fetch streams are already decoded; Content-Length may describe gzip.
        if (total !== bytes.length) throw new Error('CSV response incomplete');
        const url = window.URL.createObjectURL(new Blob(chunks, { type: 'text/csv;charset=utf-8' }));
        urls.add(url);
        const link = document.createElement('a');
        link.href = url; link.download = data.filename; link.dataset.gpiCsvDownload = '';
        link.textContent = (window.EU_LANG === 'zh' ? 'CSV 已生成，点击保存：' : 'CSV ready — click to save: ') + data.filename;
        link.addEventListener('click', event => { if (!bound()) event.preventDefault(); });
        slot.replaceChildren(link);
        document.dispatchEvent(new CustomEvent('easyicu:workspace-csv-ready', {
          detail: { projectId: source.projectId, file: source.file, sha256: source.sha256 },
        }));
      } catch (_) {
        if (bound()) status.textContent = window.EU_LANG === 'zh'
          ? 'CSV 下载准备失败，请核对当前文件后再次导出。'
          : 'CSV download preparation failed. Check the current file and export again.';
      } finally {
        if (ticket === revision) { pending = false; request = null; }
      }
    }
    window.addEventListener('message', receive);
    return { prepare, bind, reset, generate, revoke };
  }
  window.EasyICU.guidedPi.declare('csvDownloads', { create });
})();
