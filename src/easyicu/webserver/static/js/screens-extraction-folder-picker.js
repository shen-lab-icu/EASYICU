/* Owner: local folder-picker dialog shared by Data Extraction setup steps. */
(function () {
  const t = window.t;
  const icon = window.icon;
  const escHtml = window.EU_HTML.esc;
  let pickerEl = null;

  function closePicker() {
    if (pickerEl) { pickerEl.remove(); pickerEl = null; }
    document.removeEventListener('keydown', pickerKey);
  }
  function pickerKey(e) { if (e.key === 'Escape') closePicker(); }
  function cleanFolderName(raw) {
    return String(raw || '').trim().replace(/[\\/]+/g, '-');
  }
  function joinLocalPath(parent, name) {
    const base = String(parent || '').trim();
    if (!base) return name;
    return base + (base.endsWith('/') ? '' : '/') + name;
  }
  function open(startPath, onPick, title, options) {
    closePicker();
    const opts = options || {};
    let cur = startPath || '';
    const pickerTitle = title || t('Choose a data folder', '选择数据文件夹');
    const back = document.createElement('div'); back.className = 'eu-pick-back';
    back.innerHTML = `
      <div class="eu-pick" role="dialog" aria-label="${escHtml(pickerTitle)}">
        <div class="eu-pick-h">
          <span style="color:var(--ink-3);">${icon('folder', 16)}</span>
          <span class="t">${escHtml(pickerTitle)}</span>
          <span class="grow" style="flex:1;"></span>
          <button class="btn sm ghost" data-pk-close>${icon('close', 13)}</button>
        </div>
        <div class="eu-pick-cur" data-pk-cur></div>
        <div class="eu-pick-sc" data-pk-sc></div>
        <div class="eu-pick-list" data-pk-list><div class="eu-pick-empty">${t('Loading…', '加载中…')}</div></div>
        ${opts.allowCreate ? `
          <div class="eu-pick-create">
            <input data-pk-new-name placeholder="${escHtml(t('New folder name', '新文件夹名称'))}" />
            <button class="btn sm" data-pk-new>${icon('plus', 13)} ${t('Create folder', '创建文件夹')}</button>
            <div class="eu-pick-msg" data-pk-msg>${t('Create inside the folder shown above, then use it as the export destination.', '会在上方当前目录内创建，并把它作为导出目录。')}</div>
          </div>` : ''}
        <div class="eu-pick-f">
          <button class="btn ghost sm" data-pk-up>${icon('back', 13)} ${t('Up', '上一级')}</button>
          <span style="flex:1;"></span>
          <button class="btn primary" data-pk-use>${icon('check', 13)} ${t('Use this folder', '选择此文件夹')}</button>
        </div>
      </div>`;
    document.body.appendChild(back); pickerEl = back;
    const listEl = back.querySelector('[data-pk-list]');
    const curEl = back.querySelector('[data-pk-cur]');
    const scEl = back.querySelector('[data-pk-sc]');
    const newNameEl = back.querySelector('[data-pk-new-name]');
    const newBtn = back.querySelector('[data-pk-new]');
    const msgEl = back.querySelector('[data-pk-msg]');
    back.addEventListener('click', e => { if (e.target === back) closePicker(); });
    back.querySelector('[data-pk-close]').addEventListener('click', closePicker);
    back.querySelector('[data-pk-use]').addEventListener('click', () => { closePicker(); if (cur) onPick(cur); });
    if (newBtn && newNameEl && msgEl) {
      newBtn.addEventListener('click', () => {
        const name = cleanFolderName(newNameEl.value);
        msgEl.classList.remove('err');
        if (!cur) {
          msgEl.textContent = t('Choose a parent folder first.', '请先选择父目录。');
          msgEl.classList.add('err');
          return;
        }
        if (!name || name === '.' || name === '..') {
          msgEl.textContent = t('Enter a valid folder name.', '请输入有效的文件夹名称。');
          msgEl.classList.add('err');
          return;
        }
        if (!(window.EU_API && window.EU_API.createDir)) {
          msgEl.textContent = t('Folder creation endpoint is unavailable.', '文件夹创建接口不可用。');
          msgEl.classList.add('err');
          return;
        }
        const target = joinLocalPath(cur, name);
        newBtn.disabled = true;
        msgEl.textContent = t('Creating local folder…', '正在创建本地文件夹…');
        window.EU_API.createDir(target).then(r => {
          if (!r || !r.ok) throw new Error((r && (r.error || r.message)) || 'mkdir_failed');
          const createdPath = r.path || target;
          if (opts.pickCreated) {
            closePicker();
            onPick(createdPath);
          } else {
            newNameEl.value = '';
            msgEl.textContent = t('Folder created.', '文件夹已创建。');
            load(createdPath);
          }
        }).catch(err => {
          msgEl.textContent = String(err && err.message || err);
          msgEl.classList.add('err');
        }).finally(() => {
          if (pickerEl === back) newBtn.disabled = false;
        });
      });
      newNameEl.addEventListener('keydown', e => {
        if (e.key === 'Enter') {
          e.preventDefault();
          newBtn.click();
        }
      });
    }
    document.addEventListener('keydown', pickerKey);

    function load(path) {
      listEl.innerHTML = `<div class="eu-pick-empty">${t('Loading…', '加载中…')}</div>`;
      window.EU_API.listDir(path).then(r => {
        cur = r.path || path || '';
        curEl.textContent = cur || '/';
        const up = back.querySelector('[data-pk-up]'); up.disabled = !r.parent;
        up.onclick = () => r.parent && load(r.parent);
        scEl.innerHTML = '';
        (r.shortcuts || []).forEach(s => {
          const b = document.createElement('button'); b.textContent = s.name;
          b.onclick = () => load(s.path); scEl.appendChild(b);
        });
        if (!r.entries || !r.entries.length) {
          listEl.innerHTML = `<div class="eu-pick-empty">${r.ok === false ? t('Cannot read this folder.', '无法读取该文件夹。') : t('No sub-folders here.', '此处没有子文件夹。')}</div>`;
          return;
        }
        listEl.innerHTML = '';
        r.entries.forEach(en => {
          const b = document.createElement('button'); b.className = 'eu-pick-row';
          const folderIcon = document.createElement('span');
          folderIcon.style.cssText = 'color:var(--ink-3);flex:none;';
          folderIcon.innerHTML = icon('folder', 15);
          const name = document.createElement('span'); name.className = 'nm';
          name.textContent = String(en.name || '');
          b.appendChild(folderIcon); b.appendChild(name);
          if (en.hint) {
            const hint = document.createElement('span'); hint.className = 'hint';
            hint.textContent = String(en.hint);
            b.appendChild(hint);
          }
          b.onclick = () => load(en.path); listEl.appendChild(b);
        });
      }).catch(err => {
        listEl.innerHTML = '';
        const failure = document.createElement('div'); failure.className = 'eu-pick-empty';
        failure.textContent = `${t('Failed to list folder', '列目录失败')}: ${String(err && err.message || err)}`;
        listEl.appendChild(failure);
      });
    }
    load(cur);
  }

  window.EU_EXTRACTION_FOLDER_PICKER = Object.freeze({ open });
})();
