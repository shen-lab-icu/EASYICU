# 2026-07-29 EasyICU Web 端产品、可访问性与前后端契约审阅

## 结论

- 活跃模块：`web`
- 活跃阶段 / task ID：`WEBAPP-FASTAPI-NATIVE-QA · PATIENT-CROSSDB-VISUAL-PARITY`
- 审阅基线：共享分支 `fix/external-review-20260724-p0-p1@f0fdbce`；Web 功能基线仍为本地提交 `d52b05a`
- 审阅方式：只读代码审阅 + 真实 FastAPI 本地运行 + 1280×720 桌面浏览器主流程 + 聚焦自动化测试
- 总体判断：**未发现 P0；视觉层级、科学口径与 fail-closed 设计已经明显成熟，但当前仍不应判为 release-ready。**
- 发布阻塞主要集中在 6 个 P1：官方双 Demo 的稳定身份合同、SPA 路由焦点、外部字体请求、registered Cross-DB 同步无界扫描、取消/超时语义、科学产物路径租约。

本轮是审阅，不修改产品源代码，也不启动新的真实数据写入任务。

## 主流程审阅

| 步骤 | 场景 | 健康度 | 观察 |
|---|---|---|---|
| 1 | 首页入口 | 良好 | Guided 为推荐路径，Classic 为次级入口；真实数据与官方演示的选择关系清楚。 |
| 2 | Guided Copilot 起步 | 良好 / 有可访问性欠账 | 文件夹记忆、对话、进度与证据门同时可见；AI 明确 opt-in。SPA 路由仍缺标题与焦点迁移。 |
| 3 | Patient 官方 MIMIC Demo | 基本健康 | 显式打开后显示 140 entities、19 modules、281 features，来源标为“官方演示”，无页面横向溢出。直接重载时可因可变 label 失去 demo provenance。 |
| 4 | Cohort 官方 Demo | 可用但过密 | 真实聚合、KM/SOFA 展示和“不做推断统计”口径诚实；279 个特征的内部滚动 chip 列表没有搜索，键盘/读屏负担高。 |
| 5 | Cross-DB 官方双 Demo | **阻塞** | 后端两个 Demo 都是 `prepared + registered`，两张卡也都显示“已就绪”，但顶部仍为 `1 / 2 已就绪`，主 CTA 被禁用。 |

### 浏览器证据

- `EASYICU/output/ui-qa/20260729_web_review/01-home.png`
- `EASYICU/output/ui-qa/20260729_web_review/02-guided-start-viewport.png`
- `EASYICU/output/ui-qa/20260729_web_review/03-patient-real-source.png`
- `EASYICU/output/ui-qa/20260729_web_review/04-patient-loaded.png`
- `EASYICU/output/ui-qa/20260729_web_review/05-cohort-loaded.png`
- `EASYICU/output/ui-qa/20260729_web_review/05b-cohort-feature-catalog.png`
- `EASYICU/output/ui-qa/20260729_web_review/06-crossdb-setup.png`

`02-guided-start.png` 是 fixed-layout 页面使用 full-page capture 时产生的浏览器工具伪影，不作为产品缺陷或验收证据；实际视口证据使用 `02-guided-start-viewport.png`。

## P1 发现

### P1-1 · 官方 Demo 用可变展示 label 作为身份，阻塞 Cross-DB 主路径并破坏 Patient provenance 恢复

**运行时证据**

- `/api/demo-sources` 中 MIMIC-IV Demo 与 eICU Demo 均为 `state=prepared`、`registered=true`。
- `/api/workspaces/registry` 中两个实际导出路径均存在。
- MIMIC registry 的可变展示 label 当前为 `MIIV`，eICU label 则与 canonical catalog 标题一致。
- Cross-DB 页面把两个 catalog 卡都画成“已就绪”，但整体计数为 `1 / 2`，`开始一致性检查` 禁用。
- 直接进入 Patient 时，已打开的官方 MIMIC Demo 也可能被恢复成“真实”模式；再次显式打开来源后才回到“官方演示”。

**根因**

- `static/js/screens-viz-patient-demo-sources.js:29-47` 由 catalog title/version 生成 label，并用 `row.label === label` 找 registry row。
- 同文件 `92-100` 用 `active.label === value.registry_label` 恢复 provenance。
- `static/js/screens-viz-crossdb-source.js:114-140` 仅以 `officialPaths().length >= 2` 决定 readiness 和 CTA。
- 因此 display label 被错误承担了稳定主键职责。

**测试缺口**

- `tests/js/crossdb_source_choice.test.js:15-58` 直接 stub 出两个完美路径，并把两个 catalog source 都设为 active；没有覆盖“prepared/registered 但只有一个 active，且 label 被用户改名”的真实状态。

**建议合同**

- 后端 demo-source owner 提供不可变的 `source_id` / `registered_source_id` 或路径 digest；registry row 与 catalog source 通过稳定 ID 绑定。
- `display label` 只负责展示，不能再参与身份、readiness 或 provenance 判定。
- 官方 pair readiness 应由 `prepared && registered` 判断，而不是要求两个来源同时 `active`。
- 增加改名后的 negative contract test、只有一个 active 的 pair test，以及真实浏览器 happy path。

### P1-2 · SPA 路由改变后页面标题不变、焦点落回 `BODY`，关键 Agent / Extraction 交互仍是鼠标优先

**运行时证据**

- 从 Ideas 点击 Patient 后，URL 已变为 `#patient`，但 `document.title` 仍为 `EasyICU — ICU Research Workspace`。
- 当前焦点落在 `BODY`；Patient 页面没有可聚焦的 route heading，也没有路由变更 live announcement。

**代码证据**

- `static/index.html:6` 固定 document title。
- `static/js/app.js:278-312` 每次 render 重建 shell DOM；`333-350` 的 nav/hash 路径没有 title、main heading focus 或 route announcement。
- `static/js/screens-agent.js:1860-1870,2213` 把可展开证据行实现为 click-only `div`，没有 button/role/tabindex/键盘处理。
- `static/js/screens-extraction.js:1508-1511,2017-2031` 的自定义 switch 有 `role=switch` 和 tabindex，但没有 accessible name，也没有 Enter/Space 激活。
- Extraction 长任务的进度/失败区域 `1804-1827` 没有 `status`、`alert` 或 `progressbar` 语义。
- `static/js/app.js:353-370` 的裸 `1–5` 与 `L` 全局快捷键只排除 input/textarea/contenteditable，在 select、button 和自定义控件上仍可能误触。

**建议**

- 统一 `navigate()` 合同：路由变化时更新标题、焦点移到 route H1/main、提供 polite route announcement；同路由局部 repaint 保留当前焦点。
- 把可操作 `div` 改为原生 button；switch 补 visible/ARIA label 和 Enter/Space；进度/失败补 live region 与 progressbar。
- 裸快捷键改为带修饰键的命令，或仅在明确的 command surface 中启用。

### P1-3 · “本地优先 / 不外发”承诺与每次启动访问 Google Fonts 冲突

**证据**

- `static/index.html:7-9` 对 `fonts.googleapis.com`、`fonts.gstatic.com` 做 preconnect 并加载 Google Fonts。
- Guided / Copilot 产品文案明确承诺本地优先、不开启 AI 时不向外部服务发送数据。

**影响**

- 即使临床数据没有进入请求体，客户端 IP、User-Agent、访问时间等元数据仍会离开本机。
- 离线、医院防火墙和 air-gapped 场景会发生字体失败或等待，削弱本地桌面产品定位。

**建议**

- 将 IBM Plex 字体随包 self-host，或使用经过设计验证的系统字体栈。
- 增加静态资源回归：生产 `index.html` 不允许非白名单 `http(s)` 资产。

### P1-4 · Registered Cross-DB summary 仍是同步、无界的前台全量扫描

**证据**

- `routes/reviews.py:76-82` 在请求线程直接执行 registered summary。
- `crossdb_review.py:309` 起串行处理所有源；当前合同没有 registered source 数量上限。
- 每个模块的 feature 列会被读取，底层 `pd.read_parquet/read_csv` 全行加载后又转换为 Python list / NumPy 数组。
- raw 路由仍可绕开已有异步 job 路径。

**影响**

- 多个大型 registered export 会长时间占据 HTTP worker，内存峰值随行数、列数和来源数增长。
- 当前 `bounded_column_reads` 只约束列投影，不约束行数；产品也没有一致的进度、deadline、取消和容量反馈。

**建议**

- registered summary 与 raw summary 使用同一个 JobManager owner，固定返回 `202 + job_id`。
- 给来源数量、行批量/采样、峰值内存、deadline、取消和 429 统一合同。
- 重型 raw 请求不得保留同步旁路。

### P1-5 · 取消只写标志；convert 不合作，卡死任务可永久占据全部 job 容量

**证据**

- `jobs.py:93-109` 的取消只设置 `cancel_requested`。
- `jobs.py:137-175` 没有 deadline、watchdog 或强制终止边界。
- `dataio.py:436-455` 的 convert runner 在 `convert_all()` 返回前没有检查取消。
- 最小复现：取消被接受后任务仍为 `running`，第二个提交返回 `JobCapacityError`；底层 runner 释放后才变为 `cancelled`。

**影响**

- 用户看到 cancelled 语义，但后台仍可能继续读写。
- 8 个阻塞任务可永久占满容量；每个 convert 还可能创建自己的 worker。

**建议**

- 将 cooperative token 下沉到文件/分块循环。
- 为不合作的底层调用提供进程隔离、deadline 与 grace-period kill。
- 取消后的终态必须在固定时间内释放 slot，且不得继续生成 Parquet/manifest。

### P1-6 · Convert / Extract 缺少规范化路径租约，并发任务可能写入同一科学产物

**证据**

- `routes/jobs.py:39-48` 不拒绝相同规范化输入/输出路径的重复 convert。
- `dataio.py:744-752` 的导出目录分配是“先 exists、后返回”，真正 mkdir 更晚发生；同一秒并发可拿到相同目录。
- DataConverter 的锁属于单实例，不能阻止两个 job 操作同一 shard/output 目录。

**影响**

- 不同任务可能互相覆盖模块文件、删除同一 manifest，并把竞争后的目录自动注册为成功结果。

**建议**

- Job owner 引入规范化路径 lease；冲突提交返回结构化 `409 + stable reason_code`。
- Extract 以 `mkdir(exist_ok=False)` 循环原子预留目录。
- 租约覆盖任务终态、取消 grace period 与失败清理。

## P2 / 结构性风险

1. **Cohort 特征选择器信息密度过高。** 官方 MIMIC Demo 展开后有 279 个 feature chip 和大量模块按钮，内部滚动但没有搜索。建议改为可搜索 combobox/list、模块过滤、虚拟化或 roving-focus，并持续显示已选摘要。
2. **中文界面混入大量英文模块和指标。** 例如 `Blood Gas`、`Chemistry`、`Median LACT`；应在 i18n owner 中定义稳定临床术语，而不是逐屏硬编码。
3. **文件夹 modal 合同不一致。** Cross-DB 已有 focus trap/Escape/restore；Extraction `screens-extraction.js:969-1018` 和 Settings/Guided 的 picker 缺完整 trap、初始焦点和关闭后的焦点恢复。
4. **Cohort cache 未绑定实际模块文件签名。** `cohort_review.py:420-435` 的 key 只含路径、manifest mtime 和 summary 字段，原地替换模块文件后可返回旧 summary，并与当前 feature distribution 混用。可复用 Patient coverage 的 `{mtime_ns,size}` 文件签名。
5. **Loopback 防护不等于浏览器 CSRF 防护。** `app.py:121-135` 验证 peer/Host/proxy，但没有 Origin、Sec-Fetch-Site 或 CSRF token；`POST /api/settings/reset` 可被跨源 HTML form 触发。写接口需统一 same-origin / CSRF header 合同。
6. **源漂移与 Arrow/Pandas/OSError 缺 typed diagnostic。** 当前可退化成裸 500，后台 job 又把异常压成 `str(exc)`，既不含 owner/phase/reason_code，也可能泄漏本地绝对路径。
7. **确定性 CSS 跨路由碰撞。** Guided 给建议 chip 使用 `.express` modifier，而更晚加载的 `redesign.css:32-45` 把全局 `.express` 定义成 Extraction 大卡片；应 namespace 并增加 owner presence/absence regression。
8. **大 owner 与全量首屏资产仍是持续耦合风险。** `screens-guided.js` 5,789 行、`screens-viz.js` 4,818 行、`screens-extraction.js` 2,106 行；首页 Entry 还与 Extraction 同文件。`index.html` 同时加载约 35 个 CSS、51 个 script，ECharts 约 715 KB；静态 JS/CSS 当前 no-store、无路由懒加载。
9. **启动 hydration all-or-nothing。** `api.js:203-216` 的 `Promise.all` 中任一 catalog/settings/registry 请求失败，会跳过 capability hydration 与 rerender，整页退回 mock。应由各 owner 独立 readiness，使用 `allSettled` 或等价的 typed partial-state。
10. **窄屏 shell 会影响 200% zoom 场景。** `<=860px` 隐藏 sidebar/topbar，而替代导航没有 cohort/crossdb/settings/dictionary/help；即使当前 QA 不要求手机，也应把桌面缩放纳入可访问性回归。

## 做得好的地方

- 首页层级清楚：Guided 推荐、Classic 次级、真实/演示来源显式分开。
- Guided 将对话、项目文件夹记忆、进度和证据 gate 放在同一工作面，符合 Copilot 内完成配置的方向。
- Patient 显式打开官方 Demo 后，来源标记、真实计数、伪名实体、有界 preview 与 281-feature 全目录口径都诚实。
- Cohort 明确阻止 p-value/推断性结论，Cross-DB 也把 synthetic fallback 标成“仅界面演练，不是科学结果”。
- Patient、Cohort、Cross-DB 的图表 owner 与 backend aggregate contract 分工总体清楚；Patient coverage 已有基于真实文件签名的缓存实现。
- Cross-DB / Patient 已有 owner presence-and-absence regression，现有拆分方向正确。
- FastAPI 默认 loopback-only、Host allowlist、代理 fail-closed、Job SSE 终态顺序与 429 backpressure 都是扎实基础。
- 本轮代表性页面在 1280×720 没有 document 横向溢出；浏览器 console 为 0 error / 0 warning。

## 验证结果与边界

聚焦门：

```text
tests/test_webserver_shell_accessibility.py
tests/test_webserver_home_flow.py
tests/test_webserver_patient_browse_frontend.py
tests/test_webserver_cohort_profile_ui.py
tests/test_webserver_crossdb_job_continuity.py
tests/test_webserver_patient_demo_data.py

25 passed in 1.70s
```

这些测试全绿，但没有覆盖 P1-1 的真实 registry-label drift，也没有证明 screen-reader、键盘全流程或 200% zoom 合规。

本轮没有执行：

- VoiceOver/NVDA 全流程
- axe / Lighthouse 自动扫描
- 色觉模拟和对比度全量计算
- 手机/平板响应式 QA（当前项目协议只要求桌面/笔记本）
- 大型 registered export 的真实长时压力测试
- 数据转换/抽取写入或取消破坏性测试

## 建议修复顺序

1. **Release gate patch：**稳定 demo identity、Cross-DB pair negative contract、Patient provenance restore、self-host fonts、SPA title/focus/live announcement、Agent/Extraction 关键键盘语义。
2. **长任务边界：**registered summary 全部进入 async owner；实现 deadline、合作取消、slot 释放、规范化路径 lease 和原子目录预留。
3. **数据一致性与安全：**Cohort 文件签名 cache、same-origin/CSRF 写接口、typed source-drift diagnostics。
4. **结构与 UX：**Cohort 可搜索特征选择、Guided `.express` namespace、拆分 Entry/Extraction 与超预算 owner、路由级资源装载和 partial hydration。

