/* Owner: bounded CSV attachments from an explicitly enabled workspace preview. */
(function () {
  'use strict';
  function create(currentKey) {
    let key = '', nonce = '', frame = null, slot = null, coordinates = null;
    let count = 0, last = 0, pending = false, deliveryForm = null;
    function reset() {
      if (slot) slot.replaceChildren();
      key = ''; nonce = ''; frame = null; slot = null; coordinates = null;
      count = 0; last = 0; pending = false; deliveryForm = null;
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
    function receive(event) {
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
      // A real Host button submits a fixed same-origin form. The isolated page
      // receives neither this form nor any application URL/capability.
      const form = document.createElement('form');
      form.method = 'post';
      form.action = '/api/copilot/pi/projects/' + encodeURIComponent(coordinates.projectId) + '/workspace/download-csv';
      // Use a normal user-initiated attachment navigation. Some embedded
      // browsers do not deliver attachments submitted into hidden frames.
      form.target = '_self';
      const bytes = new TextEncoder().encode(data.csv);
      let binary = '';
      for (let i = 0; i < bytes.length; i += 8192) binary += String.fromCharCode(...bytes.subarray(i, i + 8192));
      const fields = { file: coordinates.file, expected_sha256: coordinates.sha256,
        filename: data.filename, media_type: 'text/csv', csv_base64: window.btoa(binary) };
      Object.entries(fields).forEach(([name, value]) => {
        // Base64 preserves CSV newlines; values never enter an HTML string.
        const input = document.createElement('textarea');
        input.name = name; input.value = value; input.hidden = true;
        form.appendChild(input);
      });
      const button = document.createElement('button');
      button.type = 'submit'; button.dataset.gpiCsvDownload = '';
      button.textContent = (window.EU_LANG === 'zh' ? 'CSV 已生成，下载：' : 'CSV ready — Download CSV: ') + data.filename;
      form.appendChild(button);
      const sourceFrame = frame, sourceNonce = nonce;
      form.addEventListener('submit', event => {
        if (!active() || sourceFrame !== frame || sourceNonce !== nonce) { event.preventDefault(); return; }
        // One pending export, one submission. A later page export may replace it.
        pending = false;
        button.disabled = true;
      }, { once: true });
      if (deliveryForm && deliveryForm.parentNode === slot) slot.removeChild(deliveryForm);
      slot.appendChild(form);
      deliveryForm = form;
    }
    window.addEventListener('message', receive);
    return { prepare, bind, reset };
  }
  window.EasyICU.guidedPi.declare('csvDownloads', { create });
})();
