'use strict';
const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
const data = window.EASYICU_CATALOG;
const escapeHtml = value => String(value ?? '').replace(/[&<>"']/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));
const menu = $('.menu-toggle');
menu?.addEventListener('click', () => {
  const open = menu.getAttribute('aria-expanded') !== 'true';
  menu.setAttribute('aria-expanded', String(open));
  menu.setAttribute('aria-label', open ? '收起导航' : '展开导航');
  $('#main-nav').classList.toggle('open', open);
});
$$('#main-nav a').forEach(link => link.addEventListener('click', () => {
  menu?.setAttribute('aria-expanded', 'false');
  menu?.setAttribute('aria-label', '展开导航');
  $('#main-nav').classList.remove('open');
}));
document.addEventListener('keydown', event => {
  if (event.key === 'Escape' && menu?.getAttribute('aria-expanded') === 'true') {
    menu.click();
    menu.focus();
  }
});
let toastTimer;
function toast(message) {
  const element = $('.toast');
  element.textContent = message;
  element.classList.add('visible');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => element.classList.remove('visible'), 3000);
}
$$('[data-copy-target]').forEach(button => button.addEventListener('click', async () => {
  try {
    await navigator.clipboard.writeText(document.getElementById(button.dataset.copyTarget).textContent.trim());
    toast('已复制');
  } catch {
    toast('无法自动复制，请选中文字后手动复制。');
  }
}));
const imageDialog = $('#image-dialog');
document.addEventListener('click', event => {
  const button = event.target.closest('[data-lightbox]');
  if (!button) return;
  $('#dialog-image').src = button.dataset.lightbox;
  $('#dialog-image').alt = button.dataset.imageTitle;
  $('#image-dialog-title').textContent = button.dataset.imageTitle;
  $('#image-dialog-caption').textContent = button.dataset.caption || '';
  imageDialog.showModal();
});
$$('[data-close-dialog]').forEach(button => button.addEventListener('click', () => button.closest('dialog').close()));
$$('dialog').forEach(dialog => dialog.addEventListener('click', event => {
  if (event.target === dialog) {
    const rect = dialog.getBoundingClientRect();
    if (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom) dialog.close();
  }
}));
const screens = {
  workspace: {image:'assets/workspace-current.jpg',title:'从研究对话进入当前成果',description:'项目、方案状态、研究对话和产物入口集中呈现。打开已有成果不会启动新的分析。',alt:'当前研究工作台真实截图',source:'当前工作台界面 · 2026-09-16 核对'},
  reader: {image:'assets/reader-current.jpg',title:'把问题与研究产物放在一起读',description:'对话和成果并排呈现，在结果表、图表、文章、文献与 PDF 之间切换，也可以进入专注阅读。',alt:'当前研究对话与成果并排阅读截图',source:'当前工作台界面 · 2026-09-16 核对'},
};
$$('[data-hero-image]').forEach(button => button.addEventListener('click', () => {
  const screen = screens[button.dataset.heroImage];
  $('#hero-image').src = screen.image;
  $('#hero-image').alt = screen.alt;
  $('#hero-caption').textContent = screen.title;
  const trigger = $('#hero-image').closest('[data-lightbox]');
  trigger.dataset.lightbox = screen.image;
  trigger.dataset.imageTitle = screen.title;
  trigger.dataset.caption = screen.description;
  $$('[data-hero-image]').forEach(item => item.setAttribute('aria-pressed', String(item === button)));
}));
function selectAccessibleTab(button) {
  $$('[role="tab"]', button.closest('[role="tablist"]')).forEach(item => {
    item.setAttribute('aria-selected', String(item === button));
    item.tabIndex = item === button ? 0 : -1;
  });
  const panel = document.getElementById(button.getAttribute('aria-controls'));
  panel?.setAttribute('aria-labelledby', button.id);
}
$$('[data-demo]').forEach(button => button.addEventListener('click', () => {
  selectAccessibleTab(button);
  const screen = screens[button.dataset.demo];
  $('#demo-image').src = screen.image;
  $('#demo-image').alt = screen.alt;
  $('#demo-title').textContent = screen.title;
  $('#demo-description').textContent = screen.description;
  $('#demo-source').textContent = screen.source;
  const trigger = $('#demo-lightbox');
  trigger.dataset.lightbox = screen.image;
  trigger.dataset.imageTitle = screen.title;
  trigger.dataset.caption = `${screen.description} ${screen.source}`;
}));
$$('[role="tablist"]').forEach(list => list.addEventListener('keydown', event => {
  const tabs = $$('[role="tab"]', list);
  const current = tabs.indexOf(event.target);
  if (current < 0) return;
  let next;
  if (event.key === 'ArrowRight') next = (current + 1) % tabs.length;
  if (event.key === 'ArrowLeft') next = (current - 1 + tabs.length) % tabs.length;
  if (event.key === 'Home') next = 0;
  if (event.key === 'End') next = tabs.length - 1;
  if (next === undefined) return;
  event.preventDefault();
  tabs[next].click();
  tabs[next].focus();
}));
const tourSteps = [
  {image:'assets/workspace-current.jpg',title:'在工作台里推进研究',text:'从研究问题出发，查看方案、执行状态和当前成果。此图为当前研究工作台的真实界面。'},
  {image:'assets/reader-current.jpg',title:'并排审阅分析产物',text:'在保留研究对话的同时打开成果，切换结果表、图表、文章与 PDF。研究者可以继续追问和审阅。'},
];
let tourIndex = 0;
let tourTimer = null;
const tourDialog = $('#tour-dialog');
function stopTour() {
  clearInterval(tourTimer);
  tourTimer = null;
  $('#tour-play').textContent = '自动播放';
  $('#tour-play').setAttribute('aria-pressed','false');
}
function showTourStep() {
  const step = tourSteps[tourIndex];
  $('#tour-image').src = step.image;
  $('#tour-image').alt = step.title;
  $('#tour-counter').textContent = `${String(tourIndex + 1).padStart(2,'0')} / ${String(tourSteps.length).padStart(2,'0')}`;
  $('#tour-step-title').textContent = step.title;
  $('#tour-step-description').textContent = step.text;
  $('#tour-prev').disabled = tourIndex === 0;
  $('#tour-next').textContent = tourIndex === tourSteps.length - 1 ? '重新观看 ↻' : '下一步 →';
}
$$('[data-open-tour]').forEach(button => button.addEventListener('click', () => {
  stopTour(); tourIndex = 0; showTourStep(); tourDialog.showModal();
}));
$('#tour-prev').addEventListener('click', () => { stopTour(); tourIndex = Math.max(0,tourIndex-1); showTourStep(); });
$('#tour-next').addEventListener('click', () => { stopTour(); tourIndex = (tourIndex+1)%tourSteps.length; showTourStep(); });
$('#tour-play').addEventListener('click', () => {
  if (tourTimer) { stopTour(); return; }
  if (tourIndex === tourSteps.length - 1) { tourIndex = 0; showTourStep(); }
  $('#tour-play').textContent = '暂停播放';
  $('#tour-play').setAttribute('aria-pressed','true');
  tourTimer = setInterval(() => {
    tourIndex += 1; showTourStep();
    if (tourIndex >= tourSteps.length - 1) stopTour();
  }, 8000);
});
tourDialog.addEventListener('close', stopTour);
document.addEventListener('visibilitychange', () => { if (document.hidden) stopTour(); });
const installSuffix = 'python -m pip install "easyicu[webapp] @ git+https://github.com/shen-lab-icu/EASYICU.git"\neasyicu-webapp';
$$('[data-os]').forEach(button => button.addEventListener('click', () => {
  selectAccessibleTab(button);
  $('#install-code').textContent = (button.dataset.os === 'windows' ? 'py -m venv .venv\n.venv\\Scripts\\activate.bat\n' : 'python3 -m venv .venv\nsource .venv/bin/activate\n') + installSuffix;
  $('.code-block>div>span').textContent = button.dataset.os === 'windows' ? 'Windows · 命令提示符 (cmd)' : 'Terminal';
}));
if (data && $('#concept-grid')) {
  const sourceNames = Object.fromEntries(data.sources.map(source => [source.id,source.name]));
  const descriptions = {miiv:'重症医疗记录与派生研究变量。',mimic:'支持既有 MIMIC-III 数据研究流程。',eicu:'多中心重症监护数据的概念映射。',aumc:'AmsterdamUMCdb 数据适配与提取。',hirid:'高时间分辨率 ICU 数据的适配。',sic:'SICdb 数据的概念与表结构适配。'};
  $('#source-grid').innerHTML = data.sources.map(source => `<article class="source-card"><span class="source-id">${escapeHtml(source.id)}</span><h3>${escapeHtml(source.name)}</h3><p>${descriptions[source.id]}</p><dl><dt>参考版本</dt><dd>${escapeHtml(source.release || '按具体数据发行核对')}</dd></dl><a href="${escapeHtml(source.url || 'guide.html#data')}" ${source.url?'target="_blank" rel="noopener"':''}>${source.url?'数据提供方入口 ↗':'查看数据准备说明 ↗'}</a></article>`).join('');
  $('#module-filter').innerHTML += data.modules.map(module => `<option value="${escapeHtml(module.id)}">${escapeHtml(module.name)}</option>`).join('');
  $('#source-filter').innerHTML += data.sources.map(source => `<option value="${escapeHtml(source.id)}">${escapeHtml(source.name)}</option>`).join('');
  let page = 0;
  const pageSize = 18;
  function renderConcepts() {
    const query = $('#concept-search').value.trim().toLocaleLowerCase();
    const module = $('#module-filter').value;
    const source = $('#source-filter').value;
    const filtered = data.concepts.filter(concept => (!query || [concept.id,concept.name,concept.english,concept.description].join(' ').toLocaleLowerCase().includes(query)) && (!module || concept.modules.includes(module)) && (!source || concept.directSources.includes(source)));
    const pages = Math.max(1,Math.ceil(filtered.length/pageSize));
    page = Math.min(page,pages-1);
    const shown = filtered.slice(page*pageSize,(page+1)*pageSize);
    $('#concept-count').textContent = `找到 ${filtered.length} 项概念 · 字典共 ${data.counts.concepts} 项`;
    $('#concept-grid').innerHTML = shown.length ? shown.map(concept => `<article class="concept-card"><div class="concept-card-top"><code>${escapeHtml(concept.id)}</code><span>${escapeHtml(concept.unit || '单位见定义')}</span></div><h3>${escapeHtml(concept.name)}</h3>${concept.english!==concept.name?`<p class="english-name">${escapeHtml(concept.english)}</p>`:''}${concept.description && ![concept.name,concept.english].some(label=>label.toLowerCase()===concept.description.toLowerCase())?`<p class="concept-description">${escapeHtml(concept.description)}</p>`:''}<div class="concept-sources">${concept.directSources.length?`<b>直接映射</b><br>${concept.directSources.map(id=>escapeHtml(sourceNames[id])).join(' · ')}`:'<b>派生或流程概念</b><br>可用性取决于组成变量与计算条件'}${concept.derived && concept.directSources.length?'<br>另含派生计算逻辑':''}</div></article>`).join('') : '<div class="empty-results"><h3>没有找到匹配的概念</h3><p>试试中文名称、英文名称或短代码，也可以放宽筛选条件。</p></div>';
    $('#concept-page').textContent = `${page+1} / ${pages}`;
    $('#concept-prev').disabled = page === 0;
    $('#concept-next').disabled = page >= pages-1;
  }
  ['#concept-search','#module-filter','#source-filter'].forEach(selector => $(selector).addEventListener(selector==='#concept-search'?'input':'change', () => {page=0;renderConcepts();}));
  $('#clear-filters').addEventListener('click', () => {$('#concept-search').value='';$('#module-filter').value='';$('#source-filter').value='';page=0;renderConcepts();$('#concept-search').focus();});
  $('#concept-prev').addEventListener('click', () => {page--;renderConcepts();$('.catalog-controls').scrollIntoView({block:'start'});});
  $('#concept-next').addEventListener('click', () => {page++;renderConcepts();$('.catalog-controls').scrollIntoView({block:'start'});});
  function renderMethods(family='') {
    const methods = data.methods.filter(method => !family || method.family === family);
    $('#method-count').textContent = `${methods.length} 项能力登记 · 快照更新于 ${data.updated}`;
    $('#method-grid').innerHTML = methods.map(method => `<article class="method-card"><div class="method-tags"><span>${escapeHtml(method.path)}</span><span class="${method.validation==='分析级能力'?'exploratory':''}">${escapeHtml(method.validation)}</span></div><h3>${escapeHtml(method.name)}</h3><p>${escapeHtml(method.description)}</p><details><summary>使用条件与范围</summary><p>${escapeHtml(method.boundary)}</p></details></article>`).join('');
  }
  $$('[data-method-filter]').forEach(button => button.addEventListener('click', () => {$$('[data-method-filter]').forEach(item=>item.setAttribute('aria-pressed',String(item===button)));renderMethods(button.dataset.methodFilter);}));
  renderConcepts(); renderMethods();
}
