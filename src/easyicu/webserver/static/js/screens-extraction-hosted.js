/* Owner: Data Extraction copy on a hosted instance.
   A hosted export goes to the clinician's own workspace on the server: the
   request omits out_dir (the backend's instance-home default) and the
   administrator's data folders stay read-only, so the page names no folder to
   choose, converts no raw files and opens nothing on the server's desktop.
   screens-extraction.js asks for a line by key and keeps its local wording. */
(function () {
  'use strict';
  const t = (en, zh) => window.t(en, zh);
  const TEXT = {
    workspace: () => t('your workspace on the server', '服务器上你的工作区'),
    destinationHint: () => t(
      'Each extraction creates a new folder in your workspace; the data the administrator prepared is not changed.',
      '每次抽取都会在你的工作区新建一个文件夹；管理员准备的数据不会被改动。'
    ),
    savedSub: () => t('saved in your workspace', '保存在你的工作区'),
    readyNote: () => t('saved in your workspace on the server', '保存在服务器上你的工作区'),
    dockLine: () => t('saved in your workspace · reproducible manifest', '保存在你的工作区 · 可复现清单'),
    running: () => t('on the server · writing a new folder in your workspace', '在服务器上 · 写入你工作区中的新文件夹'),
    cancelTail: () => t('. The next extraction writes a new folder.', '。下次抽取会另建新文件夹。'),
    doneTail: () => t('The data never left the hospital server.', '数据始终留在医院服务器上。'),
    exportNote: () => t(
      'This is an EasyICU export the administrator prepared. You can review it in Patient Review and Cohort Statistics.',
      '这是管理员准备好的 EasyICU 导出，可以直接在患者审阅和队列统计中查看。'
    ),
    rawNote: () => t(
      'These raw files have not been converted yet. The administrator converts data on the server: ask them, or choose a folder that is already prepared.',
      '这些原始文件还没有转换。数据由管理员在服务器上统一转换：请联系管理员，或换一个已准备好的文件夹。'
    ),
    folderHint: () => t(
      'Choose a folder from the data the administrator prepared; EasyICU then identifies its database and layout automatically.',
      '从管理员准备的数据中选择一个文件夹，EasyICU 会自动判断数据库和目录结构。'
    ),
    folderHintNamed: name => t(
      `Choose the ${name} folder from the data the administrator prepared; EasyICU then identifies its layout automatically.`,
      `从管理员准备的数据中选择 ${name} 文件夹，EasyICU 会自动判断目录结构。`
    ),
    placeholder: () => t('Browse the prepared data folders', '浏览选择已准备的数据文件夹'),
    definitionPath: () => t('the project in your workspace', '你工作区中的项目'),
  };
  function text(key, arg) {
    const line = TEXT[key];
    if (!line) throw new Error(`hosted extraction copy has no line '${key}'`);
    return line(arg);
  }
  window.EU_EXTRACTION_HOSTED = Object.freeze({ text });
})();
