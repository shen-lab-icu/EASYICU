# WebApp StudyContext 与真实用户动线修复（2026-07-10）

## 范围

- 模块：`web`
- 主控任务：`WEBAPP-FASTAPI-NATIVE-QA`
- 分支：`fix/easyicu-concept-bounds-enforcement`
- 目标：让首页、Guided、Extraction、Patient/Cohort/Cross-DB 与 Agent Projects 共享同一研究上下文；刷新、切换项目和失败路径不得丢状态、串项目或伪装成功。

## 已落地

### 1. StudyContext 成为跨模块唯一研究上下文

- 新增 metadata-only StudyContext store/API，持久化 question、source、cohort、modules、outcome、time window、comparator、stage 与 active job pointer。
- 严格字段 allowlist、节点/大小上限和 row-level marker 拦截；私有配置采用原子写入。
- 服务端 `revision` + CAS 防止两个页面或 A/B 项目的旧快照覆盖新状态；普通 metadata save 不得修改服务端 lifecycle 字段。
- Extraction、Patient、Cohort、Cross-DB、Guided 与 Agent 通过各自 owner adapter 映射同一个 context id。Cross-DB 只允许创建分析计划，未绑定可执行单一 export 时后端 fail-closed。

### 2. Agent 执行与隐私边界

- Agent run 真正绑定 StudyContext 的 source/question，并明确区分“运行时已应用”与“仍为信息性上下文”的字段。
- 提交前 revision 冲突返回 409，后台 job 标记失败且不执行分析；旧 job 不能清除新 job pointer。
- 隐私扫描先 canonicalize 为实际 JSON 形态，堵住 tuple、非字符串 key 等 scanner/writer gap。
- 隐私失败只写固定 schema 的 `quality_gate.json` 与 `evidence_ledger.json`；原始 payload、strict audit、numeric audit 全部 withheld，ledger 如实记录检测/持久化状态。

### 3. 用户动线与真实性

- 首页按用户已有材料提供三个主入口：文章/主题 → Idea Mining；明确问题 → Guided；本地数据 → Extraction。Demo 降为二级探索入口。
- 导航按 Discovery & Plan → Data & Review → Analysis & Evidence 排列，Patient Review 与 Agent Projects 使用真实目的命名。
- 首页和 Guided 的问题/输入草稿在中英文切换后保留；modal 支持 Escape、焦点圈和焦点恢复。
- Real mode 的 extraction、convert、raw scan 与 Guided 缺 source/backend 时全部 fail-closed，不再回退到 seeded demo 成功态。
- Guided 只有完整 `preflight + analysis_only` gate contract 才显示 preflight 完成；missing/unknown/malformed gate 或任一 hard check 失败都进入 `review_blocked`，不前进到 Findings。
- Agent/Guided 使用 immutable run token；A/B 项目交错提交或 SSE 回调只能更新自己的 `contextId + jobId + revision`，旧回调不能覆盖当前 UI、last run 或 remembered job。

### 4. 长任务刷新恢复与 Cross-DB 来源选择

- Extraction/Convert 与 Cross-DB raw distribution job 只在 localStorage 保存有界 metadata pointer。
- 刷新后先 GET snapshot：running 自动重连 SSE，terminal 恢复真实结果，404 明确报错并清理 pointer；显式 reset/rescan/continue 才丢弃旧 terminal pointer。
- Cross-DB Real mode 明确分成 A）已注册 EasyICU exports 的有界聚合，B）raw ICU root 的有界抽样；A 路径不会误触发 raw-root scan。

## Owner 边界

新增独立 owner：

- `static/js/study-context.js`
- `static/js/screens-*-study-context.js`
- `static/js/screens-extraction-job-continuity.js`
- `static/js/screens-viz-crossdb-job-continuity.js`
- `static/js/screens-viz-crossdb-source.js`

`app.js` 只新增共享 shell/导航/可访问性；`api.js` 只新增 transport；未把 route workflow 塞入 catch-all。ownership presence/absence、装载顺序、CSS brace/comment 扫描均通过。

仍有明确架构债：`screens-guided.js`、`screens-viz.js`、`screens-agent.js`、`screens-extraction.js` 仍分别约 5.7k/5.8k/2.4k/2.0k 行。它们不阻断本轮真实性修复，但下一次结构性改动必须继续沿 sub-flow/widget seam 拆 sibling，不得再扩 catch-all。

## 验证证据

- 后端 StudyContext/Agent/隐私/route 聚焦回归：`158 passed, 1 warning`。
- 前端十文件最终矩阵：`109 passed, 1 warning`。
- owner/装载顺序/route-purity 终审：`88 passed`；CSS `app.css` 279/279 braces、37/37 comments，`tokens.css` 48/48 braces、24/24 comments。
- 15 个本轮 Web JS 文件全部 `node --check` 通过。
- 可执行竞态：`run_context_race.test.js` → `{"ok":true,"ui_events":2,"patches":4}`。
- 可执行 gate 边界：`guided_gate_state.test.js` → `{"ok":true,"cases":8}`。
- 桌面浏览器：
  - 1440×1000 首页视觉检查通过；三个主入口可见，无 console error。
  - 1180×800 的 home/guided/agent/crossdb/extraction 五路由均 `scrollWidth == clientWidth`、0 可见控件 clipping、0 console warning/error。
  - 首页问题与 Guided composer 在 EN/中文切换后保留。
  - Agent tabs 可用 ArrowRight 切换且 `aria-selected/tabindex` 正确。
  - Cross-DB 不存在的 raw root 与 Extraction 不存在的数据目录均显示真实错误，未生成成功状态。

本轮采用受影响测试选择，没有重复运行全仓 3,000+ 测试。真实 9.4 万实体 Patient Review 与六库 Cross-DB density/n×n 仍未执行，不能把本日志当作大数据量验收完成证据。

## Git

- `8becc2e feat(web): bind study context to safe agent runs`
- `17143db fix(web): preserve truthful research workflow state`
- 未 push；并发 research-agent 工作区改动未纳入上述提交。

## 下一步

1. 用 9.4 万实体真实 export 完成 Patient Review 桌面验收。
2. 用至少两个、最终六个真实注册 export 验证 Cross-DB aggregate/density/n×n 与刷新恢复。
3. 下一次触碰超预算 screen owner 时继续拆分 Guided/Viz/Agent/Extraction 内部 seam。
