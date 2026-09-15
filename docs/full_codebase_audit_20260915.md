# EasyICU 全量逐行审阅 — 问题文档(2026-09-15)

> 用途：原始问题清单及本轮修复对账记录。
> 范围：Wave A–J(`src/easyicu` 数据层/agent authority/execution/planning、`research_agent/` 全子包、`webserver/` Python 129 文件、`webserver/static/` 112 JS + 58 CSS + HTML/assets、`tests/` ~1,230 文件轻量波、`tools/`/`benchmarks/`/`desktop/`/`examples/`/`scripts/`/`.github/`/根配置)。
> 方法：逐行 + 跨文件契约双向核对;文档生成后对全部 P1/P2 与关键 P3/潜伏项做了第二轮针对性复验(回读引用行+跨文件 grep 确认),每条"复验"行记录结论。
> **第二轮复验结论:全部 19 项 P2、3 项 P1、潜伏项抽验、关键 P3 均成立,无推翻/降级。** 该结论描述修复前快照；当前修复状态见下方第零节。严重度:P1=安全/正确性边界破坏;P2=坏控件/误导状态/分母丢失;P3=次要/观察。

---

## 零、本轮修复与 Codex 复审回执

- 3 项 P1 与 19 项活跃 P2 均已修复；每项行为由 Python 或 Node 回归覆盖。正式 JS 契约清单已纳入本轮新增测试。
- 已退役且未被 `index.html` 加载的 `screens-agent.js` 已删除，避免潜伏缺陷被误恢复；共享 artifact renderer 与 Guided run-file 审阅 owner 保留。
- Codex 复审额外修复一处竞态：正式计划请求若在创建任务前因会话切换失效，会释放本次 transition guard；一旦任务已创建则继续保留去重，避免重复发起 Provider 任务。
- Figure 5 校验函数导入时不再修改全局 matplotlib 样式，避免测试或复用场景中的共享状态污染。
- 本节只记录工程修复，不授权 Provider、数据重提取、新研究或发表结论；P3 观察项仍按第四节单独排期。

---

## 一、P1 — 安全或正确性边界破坏

### P1-1 · `static/js/screens-extraction.js:709` — SSE 文件名未转义进 innerHTML

- **位置**:`screens-extraction.js:709`(convert 进度行)。
- **根因**:SSE `progress` 事件的 `p.file` 是 `data_converter.py` 发出的 `csv_path.name`——用户所选文件夹内的真实磁盘文件名(不可信内容),模板里写 `` `<span class="mono">${p.file}</span>` `` 未过 `escHtml`。同文件其他文件名(L1765)都有转义。
- **后果**:文件夹内一个形如 `x<img src=a onerror=…>` 的文件名可在应用壳内执行脚本(壳可调用本地 mkdir/extract API)。属 self-XSS 语境(目录是用户自己选的),但违反本库声明的"转义是安全原语"(`html-escape.js` 头注释)。
- **最小修复**:`escHtml(p.file)`。
- **复验**:✅ 已回读 `:700-718` 确认原样插值。

### P1-2 · `static/js/screens-guided.js` — 用户自由文本 → innerHTML → 持久化重放(存储型 self-XSS)

- **位置**:sink `screens-guided.js:1062`(`${frameFor(branch)}`)、`:1064`(`planFor` 的 `${v}`)、`:660`(`JSON.stringify(j)` 进 `.json-block`——`JSON.stringify` 不转义 `<>&`);源 `screens-guided-extract.js:882,890`(自定义结局/比较输入)→`:43,48`(`resolveOutcome`/`resolveComparator` 原样返回 `outcomeCustom`/`comparatorCustom`)→`:502-513` `commitStudyDesign` → `screens-guided.js:155-158` `applyStudyDesign` 写入 `studyParams`。
- **后果**:自定义结局里输 `<img src=x onerror=…>` 即在 question 卡与 `cohort_summary.json` 预览执行;且 `guidedSlotSnapshot().study_params`(`:2099-2102`)持久化、会话恢复(`:2196`)时重放——每次重开项目重新执行。
- **最小修复**:`frameFor`/`planFor` 的动态片段过 `esc`;`:660` 先 `esc(JSON.stringify(...))` 再做高亮;或在 `applyStudyDesign` 边界消毒。
- **复验**:✅ 已回读 `:155-158`、`:295-314`、`:1062-1064` 确认 `studyParams.outcome/exposure` 原样进 innerHTML 且来源是自由文本。

### P1-3 · `tools/build_top_level_mechanism_qc.py:612-628` — 文档化的 `--skip-smoke` 模式必然崩溃

- **位置**:`:612-620` skip 分支 `qc = support.copy()`(只有 dictionary/readiness 列);`:628` 无条件 `build_figures(qc,…)` → `:449-452` 访问 `qc["non_null"]` → `KeyError`。`write_report` 同样选 `["non_null","rows","patients"]`(`:489`)。
- **后果**:help(`:592-595`)声明的"只写字典/readiness 产物"模式写 3 个 CSV 后 traceback——响亮失败、无假产物,但该模式是死代码。
- **最小修复**:skip 分支在写完 CSV 后直接 return(与 help 一致),或补 `non_null/rows/patients=NA` 并让 figure/report 段条件化。
- **复验**:✅ 已回读 `:605-634` 确认 `build_figures` 无条件调用且 skip 分支不产 `non_null`。

---

## 二、P2 — 坏控件 / 误导状态 / 分母丢失(活跃)

### P2-1 · `static/js/screens-viz-cohort.js:257,285,297` — 三个点击处理器读不存在的 `data-state`

- **位置**:消费者 `:257`(`b.dataset.state.featureScope`)、`:285`(`…sofaMatrixMode`)、`:297`(`…featureModule`);生产者 `screens-viz-cohort-view.js:385,468-469,709-710` 只发射 `data-cohort-feature-scope`/`data-cohort-sofa-matrix-mode`/`data-cohort-comp` 等,无 `data-state`。
- **后果**:`b.dataset.state` 为 `undefined`,`.X` 抛 TypeError —— 特征范围 all/recommended、SOFA 矩阵 pct/count、模块 chip 三个控件点击静默无反应。缓解:异常发生在状态变更前,UI 保持显示真实当前状态,不误标。
- **最小修复**:改读 `b.dataset.cohortFeatureScope` 等(缺陷在消费端,勿在渲染端补 `data-state`)。
- **复验**:✅ 主线程亲验(渲染端属性与消费端读取名不匹配)。

### P2-2 · `static/js/screens-extraction.js:441` — 无守卫的 demo `setTimeout` 可盖假 "done"

- **位置**:`runExtract` demo 分支 `setTimeout(()=>{exView='done';…},1200)` 无 token/模式检查;`abandonExtractionContinuity`/`__euExtractReset`/`data-ex-reset` 均不取消它。
- **后果**:demo 抽取启动后 1.2s 内切 Real 或启动真任务,旧定时器仍把 `exView` 盖成 `'done'`——瞬时假"完成"/遮住真在跑任务直到其终态事件重绘。
- **最小修复**:调度时捕获模式/序号(`const mode=dataMode(); const token=exportJobSeq++`),回调里复核。
- **复验**:✅ 已回读 `:439-448` 确认定时器无守卫;`:190-205` 的 reset 路径不取消它。

### P2-3 · `static/js/screens-extraction.js:202-205` — `backgroundRepaint` 不含 Copilot 内嵌路径

- **位置**:`backgroundRepaint` 只在 hash 为 `extraction`/`icd` 时重绘;`screens-extraction-embedded.js` 的 `paint()`/repaint 路径存在但 `backgroundRepaint` 不调用。
- **后果**:从 Copilot 侧栏(`EU_EXTRACTION_EMBEDDED_WORKSPACE`,hash=`guided`)启动的抽取,SSE `applyEvent` 更新了状态但内嵌 DOM 永不重绘——进度条冻结在首帧,直到面板重挂。
- **最小修复**:`backgroundRepaint` 增加 `embedded.isMounted()` 分支。
- **复验**:✅ 已回读 `:190-205`:`repaint()` 有 embedded 分支而 `backgroundRepaint` 只按 hash 门控,embedded 挂载时 hash=`guided` 故早退。

### P2-4 · `static/js/screens-extraction.js:222` — 后台任务完成时无条件全壳 `__euRender()`

- **位置**:`rememberExportPath` → `registerWorkspaceSource(...).then` 内无条件 `window.__euRender()`;文件自身注释(`:198-201`)明确禁止后台事件触发全壳重绘。
- **后果**:后台抽取完成时用户若在 Copilot/Settings 输入,整个 app 重渲染,丢焦点与未提交输入。
- **最小修复**:注册状态静默更新,仅当 extraction/embedded 可见时重绘;eager `__euRender` 仅留给用户发起的调用(:1887)。
- **复验**:✅ 已回读 `:213-228` 确认 `.then` 内 `window.__euRender()` 无条件执行。

### P2-5 · `static/js/screens-agent-render.js:706-708` — finding 分 lane 丢 `runtime_capability`/`unclassified`(分母丢失)

- **位置**:`decisions`/`automatic`/`evidence` 三个 filter 合计不覆盖 `remediation_route ∈ {runtime_capability, unclassified}` 且 `requires_user_authorization=False` 的 finding;后端实际发射(`scientific_review.py:1549,1609,1684,2222`,默认 `:104`)。
- **后果**:只含此类 finding 的计划渲染 "0 decisions"+空 lane,而状态 chip 说 "Analysis paused"——计数与列表都漏,仅原始 JSON 视图可见。
- **最小修复**:加 remainder lane(`findings` 减去三 lane 的剩余,标 "System / runtime items")。
- **复验**:✅ 已回读 `:700-714` + 后端 grep 确认发射。

### P2-6 · `static/js/screens-agent-render.js:526,540,588` — `data-gpi-evidence-open`/`data-gpi-display` 在 run-files 宿主内无消费者

- **位置**:`claimEvidenceAttrs`/`displayAnchorButton`/`evidenceButton` 发射;`screens-guided-pi-run-files.js` `handleClick`(L255-305)只处理 `data-gpi-reference`/`data-gpi-claim*`,事件委托(events.js)也无分支。
- **后果**:conversation "Run files" 区里 `manuscript_draft.json`/provenance 载荷的"Open registered evidence"/"Locate display"按钮可点但静默无效(`data-gpi-claim` 本身有效)。
- **最小修复**:仅在有契约的宿主发射,或 run-files 增加路由到 preview owner 的处理。
- **复验**:✅ 已回读 `screens-guided-pi-run-files.js:255-305`:`handleClick` 只处理 `data-gpi-reference`/`data-gpi-claim*`,无 evidence-open/display 分支;repo-wide grep 确认两文件内无这些属性的消费。

### P2-7 · `static/js/screens-guided.js:2252` — `@folderopen` chip 无 handler 臂

- **位置**:chip 发射 `:2252`;handler 表 `:3568-3623` 无 `@folderopen`,落 `go('@folderopen')` → `STATES` 无此键 → 静默无操作。
- **后果**:review_data 闸的"选择导出文件夹"chip 无效;缓解:`showGuidedDraftSetup(...,'open')` 对话框同时开着。
- **最小修复**:映射到 `showGuidedDraftSetup(label,'open')` 或删 chip。
- **复验**:✅ grep 确认 `@folderopen` 仅出现于 `:2252` 发射处,handler 表无对应臂。

### P2-8 · `static/js/screens-guided.js:1116-1127,3824-3825` — real 模式伪造队列匹配数

- **位置**:`data-act="strict"`(:1116)→ `_cohortEmpty`(:1119-1124)在 real 模式仍宣称 "empty in this export";`data-act="loosen"` handler(:3825)宣称 "now match ${patientN} stays",`patientN` 是 demo 计数器。
- **后果**:real 模式(经 `@activeExport`→`realConfirm`→`connect`→`detect`→`toCohort` 可达)点 Restrict 无查询即显示 0 匹配、Loosen 报 demo 数为真实匹配数——伪造结果。
- **最小修复**:real 模式隐藏 strict/loosen,或由 `snapshotSummary()` 实算。
- **复验**:✅ 已回读 `:1110-1134`、`:3818-3829` 确认。

### P2-9 · `static/js/screens-guided.js` — run-review 子系统自引用死链

- **位置**:`:68` 声明 `selectedGuidedRun`;`:3091-3137` `openGuidedRunReview` 只在内部给它赋值,而唯一调用点传入 `selectedGuidedRun`(首次必为 null→总是"cannot open");`openGuidedProjectMemory` 只以 `'draft'` 调用,`kind==='run'` 恢复路径不可达。
- **后果**:"Review local artifacts" 控件永远开不了(今日不可达);若 chip 渲染则误报错。
- **最小修复**:接真入口(rail 行设 `selectedGuidedRun` 再调用)或删链。
- **复验**:✅ grep 确认 `selectedGuidedRun` 仅在 `openGuidedRunReview`(:3100)内赋值,而唯一调用点(:3618)传入它自身;`kind==='run'` 恢复不可达(`openGuidedProjectMemory` 仅以 `'draft'` 调用)。

### P2-10 · `static/js/screens-guided.js:2452-2494` — `openGuidedProjectMemory` 无失效守卫 → 错会话写入

- **位置**:两次快速点击不同项目行 → 两个 `openGuidedProject` 在途,**最后 resolve 者**(非最后点击)赢得 `restoreGuidedProjectThread`/`bindProjectToPi`;rail 的 `active` 已标最后点击项;`saveGuidedSlotsNow`(`:2160-2168`)会把 slot 写进错误会话。
- **后果**:thread 显示项目 A 但 slot 持久化进项目 B。
- **最小修复**:请求序号/token(同 `gen`/`guidedRunChannel` 惯例),非当前则弃。
- **复验**:✅ 已回读 `:2452-2494`:`.then` 无条件 `bindProjectToPi`/`restoreGuidedProjectThread`,`active` class 在 :2461 同步标最后点击项,解析无序——竞态成立。

### P2-11 · `static/js/screens-guided-pi.js:1527-1535,1562-1571` — post-await `watchJob` 跨项目泄漏

- **位置**:`sendText`/`regenerateMessage` 在 `await api().sendPiCopilotMessage(...)` resolve 后直接 `state.jobId=payload.job_id; watchJob(...)`,不复核 `state.session`/`projectId()`;`bindProject`(:1384-1420)清理发生在 await 期间,迟到的 resolve 又开新 EventSource 往新项目的 `state.messages` 写旧 job 事件。同形:`plan-actions.js:232-279` `startFormalPlanGeneration`。
- **后果**:一个项目的在途 turn 可把 activity/assistant 行画进另一个项目的会话(权限不泄,但 UI 显示不属于当前会话的工作)。
- **最小修复**:await 前捕获 `expectedSessionId`/`expectedProjectId`,resolve 后不符即弃(同 `openSession` :961/977 惯例)。
- **复验**:✅ 已回读 `:1520-1574` 确认无守卫。

### P2-12 · `static/js/screens-guided-pi-preview.js:647-667` — `open()` 对 web/document 不递增 ticket → 旧 artifact 顶名显示

- **位置**:ticket(`state.request`)只在 `loadResource`(:540)与 `close`(:669)自增;`open()` 里 `mode==='web'||'document'` 时跳过 `loadResource`,旧在途请求 ticket 仍有效 → resolve 时写 `state.artifact/payload` 为**旧**资源;随后 "Code" 标签(:774)因 `state.artifact` 非空跳过重载。
- **后果**:新资源名下显示上一文件内容。
- **最小修复**:`open()` 状态重置处 `state.request += 1`(与 close 对称)。
- **复验**:✅ 已回读 `:640-674` 确认。

### P2-13 · `css/patient-tables.css:16` — `var(--accent-2)` 无定义 → Pareto 条形不可见

- **位置**:`.pt-pareto-track i` 的 `background:linear-gradient(90deg,var(--accent),var(--accent-2))`;`--accent-2` 全库无定义 → 整条 `background` 失效 → 条形无填充。
- **最小修复**:参考 `guided-pi-data-preview.css:15` 用 `color-mix`,或在 `tokens.css` 定义 `--accent-2`。
- **复验**:✅ grep 全 CSS 目录确认无 `--accent-2:` 定义。

### P2-14 · `css/cohort.css` 9 处 + `cohort-charts.css:5` — `--line`/`--ink-1`/`--mono` 只在 `.gd-pi-shell` 作用域定义

- **位置**:`guided-pi.css:13-16` 把这组变量定义在 `.gd-pi-shell{}` 内;Cohort 路由在其外 → `border:…var(--line)`(cohort.css:21,81,117,142,314,347,395,406;cohort-charts.css:5)整声明失效 → 卡片无边框、risk 表无行分隔线;`var(--ink-1)`/`var(--mono)` 同理(:46,154,392;:9-11)。
- **最小修复**:`tokens.css` `:root` 加别名(`--line:var(--hair)` 等)。
- **复验**:✅ 已回读 `guided-pi.css:10-16` 并 grep 全库确认定义点唯一且作用域受限。

### P2-15 · `css/guided-pi-literature.css:5,11,28,32` — `var(--good)` 无定义

- **后果**:`.searched` 状态点停在 warn 琥珀(误信号);`bound`/`direct`/`supported` 绿边退化为灰边(状态不可区分)。
- **最小修复**:`--good`→`--ok`(或 tokens 加别名)。
- **复验**:✅ grep 确认无 `--good:` 定义。

### P2-16 · `tools/build_top_level_mechanism_qc.py:517-521,545` — 报告结论硬编码、与自产 `summary.json` 可矛盾

- **位置**:`write_report` 写死 5 条编号结论(全部 ready/13 概念全保留/HiRID·SICdb 不支持名单/逐库裁决/"0 rows 全因 sample=10")——`--sample-size` 可调、`--skip-smoke` 存在;同产物 `summary.json`(:638-655)实算同量。
- **后果**:`crossdb_top_level_qc_report.md` 可宣称"全 ready"而同文件内的 readiness/status 表显示失败——误导性 QC 证据。
- **最小修复**:结论由 `readiness["ready"].all()`/`unsupported`/`status_counts`/`args.sample_size` 实算;或钉死 pinned-run 身份后标注为 run-specific adjudication。
- **复验**:✅ 已回读 `:510-549`:结论 1-5 与"复核说明"确为字面量;`summary.json`(:638-655)实算同量,两产物可矛盾。

### P2-17 · `tools/run_openrouter_fullflow_validation.py:611-625` — 无条件 `return 0`

- **位置**:`n_failed` 在 `:612` 计算、`:614` 写出,`:625` 无条件 `return 0`。
- **后果**:名为 validation 的工具任务失败仍 exit 0 → `tool && next` 假绿。仅手动可达(无 workflow/脚本调用)。
- **最小修复**:`return 0 if summary["n_failed"]==0 and summary["n_tasks"] else 1`。
- **复验**:✅ 已回读 `:605-629`:`n_failed` 计算+写出后 `:625` 无条件 `return 0`,`SystemExit(main())` 在 `:629`。

### P2-18 · `scripts/r5_fig5_nature.py:26-30,35,47-50` — 静默丢库 + 硬编码陈旧结论

- **位置**:`df[df.database==db].reindex(CATS)` → 缺席库成全 NaN 不画也不报错;`:35` "~6x between-DB spread"、`:47-48` "all 6 DBs"、suptitle "six harmonized" 均为字面量非实算;工作区笔记记载修正口径后 AUMC 32.4%→10.7%,"~6x" 已不成立。
- **后果**:图 5 可缺库仍宣称六库方向可移植;旧数字会被盖在修正后数据上。
- **最小修复**:绘图前 `assert set(df.database.unique())==set(DBS)` 且 `mortality_pct` 有限;spread 与 "<1 in all 6" 由 `df` 实算或改为打印。
- **复验**:✅ 已回读 `:24-53` 确认。

### P2-19 · `src/easyicu/research_agent/.../config.py` — profile pin 坐标缺一致性 enforcer

- **位置**:`planner_only`、`require_human_plan_review` 是 profile 的 pin 坐标(canary 钉 `planner_only=True`、QUALIFICATION12 钉 `require_human_plan_review=True`),但 `PipelineConfig.__post_init__` 对 know_how/curated/coder_resources/reviewed_memory/capability/planner_strategy/pubmed/literature_design/outline-stop 全部有一致性校验,唯独这两个没有。
- **后果**:直接构造 config 声称 canary profile 名却 `planner_only=False`,可绕过 "Planner 不得进 Execute" 边界且溯源仍记录该 profile。缓解:`*_dev` 名使 paper 闸仍关闭;宿主启动器(webserver)总是应用 `profile.pipeline_options()`。
- **最小修复**:`__post_init__` 补同形 enforcer(profile 名→坐标一致性断言)。
- **复验**:✅ 早前 Wave F 已验(构造器校验清单与 profile 表双向核对)。

---

## 三、潜伏缺陷(`screens-agent.js` — 已退役未加载)

> 前提:`index.html:128-131` 不加载该文件;`app.js` `normRoute('agent')→'guided'`;`tests/webserver/test_webserver_static_routes.py` 多处断言其缺席。**当前全部不可达;重新加 script 标签即激活。**

| ID | 严重度 | 位置 | 问题 |
|---|---|---|---|
| L-1 | 潜伏 P1 | `screens-agent.js:827,1284-1289` | 服务端 run 行字段(`s.runs[0][0]`、`r[0]`、`r[3]` 等,来自 project-index 载荷)原样进 innerHTML |
| L-2 | 潜伏 P1 | `:1174` | `contextStats` 的 `${v}` 未转义(含服务端 `benchmark.warnings`、`selectionDigest`、runStatusLabel 原始回退) |
| L-3 | 潜伏 P2 | `:820` | `data-ag-sel="${s.id}"` 服务端 id 未转义(引号截断) |
| L-4 | 潜伏 P2 | `render.js` 发射的 `data-gpi-*` | 本文件 wire 不处理 claim/evidence/display;`<a href="#gpi-reference-N">` 会改 hash → `resolveRoute` fallback → 跳去 `#guided`(引用点击劫持路由) |
| L-5 | 潜伏 P2 | `:2008` | `location.hash==='#agent'` 永假(normRoute 重写)→ StudyContext 变更不重绘 |
| L-6 | 潜伏 P3 | `:619-625` | `repaintBody` 在 `#agHost` 缺席时仍 `__euRender()` → 越界重绘当前路由 |
| L-7 | 潜伏 P3 | `:22,24,26,630-632,681-719` | `timer`/`errorRemedies`/`warning` 死字段;`rememberAgentJob` 无调用 → localStorage 续跑死路;`restoreAgentJobFromSnapshot` 不验 `snapshot.kind` |
| L-8 | 潜伏 P3 | `:1266,1841-1844` | `data-ag-history-open` 传行 index,history 刷新后可打开错误 run |
| L-9 | 潜伏 P3 | `:599-618,246,1798-1802` | 多处无守卫的异步写(openReview/requestArtifact/cancel catch/项目加载) |
| L-10 | 潜伏 P3 | `:1916-1924` | `[data-ag-signoff]` 在 `draft_unlocked` 为真时本地自封 `signed=true`(未写服务端;`signed`≠`draft_unlocked`) |

**建议**:删除该文件,或 revival 前按上表先修。

---

## 四、P3 — 次要项(按文件归组)

### `screens-extraction.js`
- `:1897` `[data-ex-resume]` 绑定无发射器(死代码,`resumeConvert` 不可达)。
- `:628,631-634,643,1765` 服务端常量(`db`/`layout`/`size`/`module` 键)原样插值——当前皆为后端常量无活利用面,但违本文件自身转义约定。
- `:587-602` 未映射扫描错误码(`export_layout_invalid`/`database_detection_unavailable`/`DatabaseDetectionError.code`)塌成通用文案;`:1150` `rememberExportPath` 注册失败 `.catch(()=>null)` 静默;`:1760` 与 `:1701/1726` 两枚 "Extract again" 语义不同;`:1913` 死表达式。

### `screens-agent-render.js`
- `:1207-1211` 结果表 meta 报切片后行数(500 行表显示 "30 rows" 无截断提示)。
- `:291-300` `scrubDataUrls` 大小写敏感(`Data_URL` 可漏进 JSON 预览;本地视图,无隐私破口)。
- `:301-307` `displayAnchorId` 允许空格进 `id` 属性(装饰性,无功能影响)。

### `screens-guided.js`
- 死 handler:`data-hint`(:3561)、`data-mode`(:3794)、`data-canceldraft`(:4039);死函数:`sendGuidedShortcut`(:2634)、`guidedDraftPayload`(:2659)、`isBeyondGoal`(:1301)。
- `:280-290` `loadWorkspaceSnapshot` 无序竞态(旧路径快照可写到新键)。
- `:915` SSE `onmessage` 无 `try/catch`(`:1588` 处有)。
- `:1549-1555` `runGuidedAgentPreflight` 对 `persistForRun→null` fail-open;同依赖在 `runLivePipeline`(:887-891)fail-closed —— 姿态不一致。
- `:1291` `summaryOf('draft')` 对 live 结果恒报 `locked·analysis_only`(gate-blocked 应为 `review_blocked`);`:3619` `@activeExport` 双推用户气泡;`:2420` `@noop` chip 回复文不对题;`:2352-2363` `ensureGuidedSession` 无在途去重;`:736` "Review artifacts" 链到 `#patient` 且 live artifacts 无 `data-artopen` 预览路径;`:782-796` `ONCE.detect` 无导出时仍称 "Detected…read from manifest";`:2374,2549` 后端 `reply.en/zh` 原样插入(约定 markup 通道,风险在服务端拼接)。
- `screens-guided-contracts.js:50,67` `guidedGateCheckLabel` 对未知 id 回退 `String(id)` 未转义(服务端自有 id,低风险)。

### `screens-guided-pi.js` 及兄弟 owner
- `:1726-1730` rebind 未守卫赋 `state.session`;`:974-977` `openSession` 多页 await 间赋值先于守卫;`:1072-1080` `switchMode` 合并未核项目;`:1638-1642`/`run-outcome.js:134-142` 用 post-await `projectId()` 打开 pre-await 资源;`data-binding.js:33,99-106` 未守卫 `setSession`——均瞬时自愈。
- `events.js:128,159-165,213` 三个 handler 无发射器(死代码);`confirmation.js:363-364,722-764` 不可达 decision 分支(该 code 被 `systemOwnedPlanFindingCodes` 过滤)。

### `screens-ideas.js`(尾)
- 全段无异步失效守卫:`data-idea-new`/srcType 切换/`mineIdeas` 可被在途 `loadIdeaRun`/`resolve`/`discover`/`ingestPdf`/`litScan` 的 resolve 覆写(瞬时但可见);Zotero widget 同形。
- `:1196` record 卡 `data-idea-record="${r.runId||r.id}"` 对缺 `run_id` 行退化为记录键 → 点击报 "not found"。

### `screens-guided-pi-preview.js`
- `:623` enrichment 响应未经 `safeResource` 白名单即 `{...state.resource,...payload}`(可覆盖 title/url/kind;下游 esc 与 `safeUrl` 兜底,无实演 XSS;`source_review_status:'reviewed'` 系本地自封)。
- `:132-133` `research_document`/`system_validation_document` 的 sha256 可缺省 → 预览不钉 digest(后端 CSP sandbox 兜底)。
- `:651` 两侧 id 有空值时 recents 跨项目残留。

### CSS 词表漂移簇(建议 `tokens.css` :root 一次性加别名)
- `--shadow-1/2/3` 11 处(guided/guided-projects/guided-panels/guided-pi-preview)阴影全失效(应 `--sh-*`)。
- `font:…var(--font)` ~17 处与 `var(--sans)`/`var(--mono)` 出域 ~10 处 → 整 shorthand 失效(应 `--font-sans`/`--font-mono`)。
- `--ink-5`(guided-projects:142 状态点透明)、`--hair-strong`(cohort:12/crossdb:272/deepdive:128 设计线消失)、`--danger`(guided-pi:264,289/idea-source:20 危险文本不红)、`--teal`(guided-pi:178 hover 边框+`:focus-visible` outline 失效——a11y)、`--muted`(cohort-eligibility:19,35)。
- `guided-pi-data-preview.css:2,3,7,15` 把 `.gd-pi-shell` 域内变量(`--line`/`--ink-1`)用在内嵌 aside 语境 → 边框失效。
- `agent.css:18-19` `.ag-title .editmk` hover-only 无 focus 路径(键盘可 tab 到不可见控件)。
- `cohort.css:11`/`crossdb.css:272` 固定最小宽图在窄屏被 `overflow-x:clip` 裁切不可达。
- 观察:`[aria-disabled]` 视觉门控普遍但一致(JS 端均有二次校验);`guided.css:1856-1893` 对 `[hidden]` 的 `display:flex !important` 移动端覆盖需确认与 `.gd-pipeline-disclosure` 的预期互动。

### tests(Wave I,4 项)
- `tests/research_agent/core/test_a_refusal_names_the_shape_that_caused_it.py:32` 硬编码 `/Volumes/外置硬盘/easyicu_data/...` → 记录回放测试在其它机器静默跳过(条件覆盖,非假绿)。
- `tests/research_agent/conftest.py:115` 会话级可变 `synthetic_cohort` DataFrame 无拷贝守卫(当前唯一改写者先拷贝,潜伏序依赖)。
- `execution/test_runner_timeout_bytes.py:262` 0.2s 真 SIGALRM;`providers/test_provider_hard_stop.py:375` 真 spawn+Barrier 无环境 skip —— 极端负载下可 flake。
- 源码文本守卫:`webserver/test_registered_export_selection.py:118` `count("resolve_registered_export(")==1`;`test_a_refusal_names_the_shape_that_caused_it.py:140-149` 断言源码窗口内 `"try:"`/`"except Exception"` —— 语义等价的重构会误伤。

### Wave F 其余 P3
- `robustness/primary_effect.py:345-360`:由 `primary_or`+SE 重建 Wald CI 进跨库比较/图面,无 `ci_source` 溯源标记——重建区间可能被呈现为模型报告的 CI。
- (另 3 项 P3 已在 Wave F 过程中记录;本条为其代表项,汇总见审阅史。)

### Wave J 其余 P3
- `tools/run_analysis_bench_overnight.py:466` 无条件 `return 0`(批次包装,逐项失败已入 `overnight_progress.json`;若作闸用需绑 `n_failed`)。
- `scripts/` 多个硬编码 `/Volumes/外置硬盘/databases…` 根;`warnings.filterwarnings("ignore")`;`EASYICU_FORCE_INPROCESS_BATCH`/`EASYICU_DISABLE_AUTO_CHUNK` 文档化快速路。
- `scripts/obesity_paradox_crossdb.py:37` 等 `death.fillna(0)` 视缺席死亡概念行为存活(隐含分析假设,未与封存契约回验)。
- `r4_crossdb_sofa2_extract.py:89-90` `(None,error)` 元组落 JSON 成 `[null,"…"]`(外观);`r4_fig4_nature.py:103-104` 缺 lookup 时标题印 "n = 0"(外观)。
- `pytest.ini:45-49` 默认 `--disable-warnings`。

---

## 五、审阅确认守住的边界(无缺陷记录,供讨论时区分"已验证"与"待修")

- EvidenceStore 唯一权威;续跑/重放全链 digest 自验(quarantine symlink/逃逸拒、preflight 双 digest、逐文件重哈希)。
- Planner 拥有 estimand/cohort/exposure/outcome/model/adjustment;修订永不原地改;`descriptive_only`/`matched_cohort=False`/`inferential_statistics_allowed=False` 贯穿 crossdb 与 review。
- Provider 双闸(canonical `ai_enabled`→per-run opt-in),凭据永不在闸前加载;伪造/未受管 mock 在回调前拒。
- PHI 哨兵测试穿透全部 agent prompt;provider 提示不含私有队列值/类别字面量;患者行不出本机。
- 发表权威:`reportable:false`/`draft_unlocked:false` 硬写持久化投影;签字不解锁草稿;`analysis_only` 不晋升 paper;`paper_authorized+publication_ready` 双闸。
- 分母保留:除 P2-5 一处真实漏洞外全栈落实(blocked/stale 行留数、`denominator_resolved===false→indeterminate`、unsupported fail-closed)。
- tests:post-mortem 回归钉死失败模式;mock 完整性 AST 元闸;无 mock 绕过被测闸。
- CI/benchmark/desktop:workflow `contents:read`+SHA pin;canonical9 evaluator 冻结模型+typed 拒绝码;desktop loopback+session token+严格 CSP。

## 六、未完全验证范围(诚实声明)

- tests 为轻量波次(35 全读+~55 实质/抽样;488k 行未逐行)。
- `screens-agent.js` 潜伏缺陷未做运行时复现(文件不加载)。
- 二进制资产(png/pdf)仅记录存在。
- `death.fillna(0)` 语义未对封存导出契约回验。
- P3 项以审阅报告为准;主线程对全部 P1/P2(19 项)、潜伏项关键点(ag:820/827/1284-1289)与关键 P3(pi-preview:623 白名单绕过、contracts:50 回退未转义)已回读源码逐条复验通过;其余 P3 为低风险观察项,供修复时顺带核对。

## 七、第二轮复验记录(2026-09-15)

| 复验项 | 方法 | 结论 |
|---|---|---|
| P1-1/2/3 | 回读引用行 | 全部成立 |
| P2-1,5,8,11,12,13,14,15,18,19 | 回读引用行+生产/消费端双向 grep | 全部成立 |
| P2-2 | 回读 `:439-448` + reset 路径 | 成立(定时器无守卫) |
| P2-3 | 回读 `:190-205` | 成立(`repaint()` 有 embedded 分支,`backgroundRepaint` 无) |
| P2-4 | 回读 `:213-228` | 成立(无条件 `__euRender`) |
| P2-6 | 回读 run-files `:255-305` + grep emitters | 成立(死控件) |
| P2-7 | grep `@folderopen` | 成立(仅发射无消费) |
| P2-9 | grep `selectedGuidedRun`/`openGuidedRunReview` | 成立(自引用死链) |
| P2-10 | 回读 `:2452-2494` | 成立(无序竞态) |
| P2-16 | 回读 `:510-549` | 成立(硬编码 prose 与实算 summary 可矛盾) |
| P2-17 | 回读 `:605-629` | 成立(无条件 return 0) |
| 潜伏 L-1/2/3 | 回读 agent.js `:818-831,:1284-1291` + index.html/script 清单 | 成立(且确认不可达) |
| pi-preview:623 / contracts:50 | 回读 | 成立(P3) |

**结论：本文档所列问题均为真问题；本轮已关闭全部活跃 P1/P2，第四节 P3 保留为后续低优先级观察项。**
