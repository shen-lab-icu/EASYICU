/* Owner: bounded CSV attachments from an explicitly enabled workspace preview. */
(function () {
  'use strict';
  function create(currentKey) {
    let key = '', nonce = '', frame = null, slot = null, coordinates = null;
    let count = 0, last = 0, pending = false, revision = 0, request = null;
    const urls = new Set();
    function reset() {
      revision += 1;
      if (request) request.abort();
      request = null;
      urls.forEach(url => window.URL.revokeObjectURL(url));
      urls.clear();
      if (slot) slot.replaceChildren();
      key = ''; nonce = ''; frame = null; slot = null; coordinates = null;
      count = 0; last = 0; pending = false;
    }
    function prepare(value) {
      reset();
      if (!value || !window.crypto || !window.crypto.getRandomValues) return '';
      const bytes = new Uint8Array(32);
      window.crypto.getRandomValues(bytes);
      nonce = Array.from(bytes, byte => byte.toString(16).padStart(2, '0')).join('');
      key = value;
      return nonce;
    }
    function active() { return key && key === currentKey() && frame && frame.isConnected && slot && slot.isConnected; }
    function bind(targetFrame, targetSlot, source) { frame = targetFrame; slot = targetSlot; coordinates = source; }
    async function receive(event) {
      const data = event.data;
      if (!active() || pending || event.source !== frame.contentWindow || event.origin !== 'null'
        || !data || data.type !== 'easyicu:workspace-csv' || data.nonce !== nonce
        || data.media_type !== 'text/csv' || typeof data.filename !== 'string'
        || !/^[A-Za-z0-9][A-Za-z0-9_.-]{0,115}\.csv$/.test(data.filename)
        || !data.filename.endsWith('.csv')
        || typeof data.csv !== 'string' || !data.csv.length || data.csv.length > 1048576
        || data.csv.includes('\0') || new TextEncoder().encode(data.csv).length > 1048576
        || count >= 5 || Date.now() - last < 1000) return;
      count += 1; last = Date.now(); pending = true;
      const ticket = revision, sourceFrame = frame, sourceNonce = nonce;
      const source = { ...coordinates }, controller = new AbortController();
      request = controller;
      const bound = () => ticket === revision && active() && frame === sourceFrame && nonce === sourceNonce;
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
        link.textContent = (window.EU_LANG === 'zh' ? 'CSV 已生成，下载：' : 'CSV ready — Download CSV: ') + data.filename;
        link.addEventListener('click', event => { if (!bound()) event.preventDefault(); });
        slot.replaceChildren(link);
      } catch (_) {
        if (bound()) status.textContent = window.EU_LANG === 'zh'
          ? 'CSV 下载准备失败，请核对当前文件后再次导出。'
          : 'CSV download preparation failed. Check the current file and export again.';
      } finally {
        if (ticket === revision) { pending = false; request = null; }
      }
    }
    window.addEventListener('message', receive);
    return { prepare, bind, reset };
  }
  window.EasyICU.guidedPi.declare('csvDownloads', { create });
})();
