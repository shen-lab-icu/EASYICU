# EasyICU Web Copilot 低改动优化执行方案

> 任务 ID：`WEB-COPILOT-COCKPIT-LITE`  
> 日期：2026-07-29  
> 模块：web / Guided Copilot  
> 状态：方案冻结，待用户确认后实施；本轮未修改产品源代码  
> 目标：保留 EasyICU 的 ICU 科研语义、本地优先、真实数据 owner 与 evidence-bound 优势，用最小前端改动降低 chat-first 交互成本

## 一句话决策

采用 **Research Cockpit Lite（侧挂式研究工作台）**，不重写 Copilot：

- 现有三栏壳保留；
- 中央聊天和现有 inline workflow 保留；
- 右栏从 `0/8` 进度升级为持久的 `Study Brief → Run → Scientific Review` 状态面板；
- 左栏只提升项目可恢复信息，路径退到二级；
- 前两轮不新增后端 route、调度器、数据库 schema 或执行协议。

这能用约两次小型、可独立回退的前端 patch，先解决“研究状态散落在聊天里”的主要问题，而不破坏已有执行链。

## 为什么这是最小且正确的改法

### 已有能力足够支撑第一版

当前产品已经具备：

- 三栏全屏 Guided shell；
- 项目文件夹记忆、`goal / step / slots / messages`；
- metadata-only、local-first 的 Guided session；
- `guidedSlotSnapshot()` 对数据源、队列、modules、outcome、window、comparator、export、Agent 状态的有界快照；
- Guided → Data Extraction / Patient / Cohort / Cross-DB / Agent 的 handoff；
- StudyContext revision/CAS；
- Agent job、SSE、gate checks、review、science workbench 与 artifact API。

因此第一版 `Study Brief` 和 `Review` 都应是**现有真相的只读/可导航投影**，而不是新建第二套状态源。

### 必须诚实表达的后端边界

当前 Agent runner 实际绑定到执行的主要字段只有：

- `data_source.path`
- `question`

以下内容虽然能进入 StudyContext，但当前仍属于 informational context：

- cohort
- modules
- outcome
- time window
- comparator

所以第一版不能使用“协议已批准并执行”“该 outcome 已被 pipeline 强制应用”等文案。按钮应叫：

- `保存研究简报`
- `检查实际执行输入`
- `进入 Agent 预检`
- `Ready for scientific review · not reportable`

这条边界是保证 EasyICU 科学诚实性的核心。

## 保住哪些优势

EasyICU 不应靠“更像通用聊天 Agent”竞争。要保住的定位是：

> **ICU-native Study Compiler + 真实本地数据审阅 + evidence-bound reportability**

| EasyICU 优势 | 第一版界面护栏 | 发布验收 |
|---|---|---|
| ICU 临床研究语义 | Study Brief 固定显示 question、source、cohort、outcome、time anchor/window、comparator、modules、export、analysis goal | outcome/window 缺失或冲突时不能显示“可执行”；用户能看出哪个字段需要处理 |
| 真实六库与谐和概念层 | 来源始终显示 database、真实/官方 Demo、分母、模块/coverage，并深链到 Patient/Cohort/Cross-DB owner | Demo 不得显示为用户真实数据；结构无源、测量缺失、all-null、未物化不得合并为一个“缺失” |
| EvidenceStore 与 fail-closed gate | 运行状态与科学状态分开；Review 显示 gate checks、blocker、artifact/evidence | `done` 不等于可报告；缺失/未知/不完整 gate 一律 `Review blocked` |
| Local-first 与隐私 | 常驻显示 Local only / Network off；外部 AI 仍逐次 opt-in；路径默认折叠 | 不新增外发；Guided memory 继续禁止 patient rows 和 identifier |
| Owner modules | Copilot 只组织上下文与摘要；执行和详细审阅继续由各原生 owner 负责 | 不在 Copilot 重做 scan、extract、Patient 图表、Cross-DB 分布或 Agent evidence 规则 |
| 可复现 provenance | Review 显示 run id、source、StudyContext revision、denominator、artifact/digest 可用性 | finding 能进入真实 artifact owner；source/context 变化后旧状态不能继续被当作当前 |
| 长任务控制 | 当前任务、取消、终态和下一决定明确；恢复能力只复用既有 job contract | stale SSE 不覆盖新项目；404/failed/cancelled 不静默变回未运行 |
| 科学诚实性 | Idea feasibility、analysis result、scientific review、reportable conclusion 使用不同状态 | `analysis_only`、hypothesis-generating 或 human-signoff pending 绝不显示成正式结论 |

## 目标界面：保留三栏，只改变信息层级

### 左栏：项目恢复信息

现状中的“运行研究项目 + metadata_only + 原始路径”改为：

```text
SOFA-2 与 ICU 死亡
Planning · 2 items need review
Next: confirm outcome window
Updated 06:03
```

设计规则：

- 第一行：研究标题；
- 第二行：`Planning / Running / Scientific review / Blocked`；
- 第三行：下一项决定；
- 时间保留；
- 本地路径放到 title/details，不占主视觉；
- 不新增新的项目模型，继续读取现有 draft/session metadata。

### 中栏：保持聊天和 inline workflow

第一轮不把聊天改成 artifact canvas，原因是：

- 当前 Prepare Data、Review Data、Idea Mining、Agent preflight 已能在聊天内完成；
- 重做中央画布会同时触碰多个 sub-flow；
- 现有“一次做一个决定”的 inline card 是可保留的优点。

中央区域只增加两个轻量行为：

- 从右栏点击某字段，会回到已有 Source/Cohort/Design/Modules/Export 卡片；
- 右栏状态变化后，焦点返回相应卡片或状态标题。

不在第一轮再造第二套表单。

### 右栏：Study Brief / Run / Scientific Review

右栏仍保持当前约 322px 宽度，但根据研究阶段显示三种状态。

#### A. Planning

```text
Study Brief                 5 / 8 reviewed

Applied when run starts
✓ Question       Lactate and mortality
✓ Data source    MIMIC-IV export

Saved research context
! Cohort         Adult first ICU stay
! Outcome        28-day mortality
! Time window    First 24 hours
○ Modules        Missing
! Comparator     None
✓ Export         Parquet

Next decision: choose feature modules
[Continue configuration]
```

字段状态不是笼统的“已完成/未完成”，而是：

- `Applied when run starts`
- `Saved research context`
- `Missing`
- `Conflict`

这样不会让 UI 比后端承诺更多。

#### B. Running

```text
Agent preflight · running
Owner: Agent Projects
Current: audit feature quality
4 / 6 checks observed

[Open Agent Projects] [Cancel]
```

只投影已有 job/gate 状态，不新建任务编排器。

#### C. Scientific review

```text
Ready for scientific review
Not reportable · human sign-off required

Source and question applied
Cohort/outcome/window remain context
Denominator: 94,458
Gate checks: 5 passed · 1 pending

[Review evidence] [Revise question] [Open Agent Project]
```

任一必需 gate 缺失、unknown 或 failed：

```text
Review blocked
Reason: denominator unresolved
```

不能用绿色完成状态代替 fail-closed。

### 阶段标签，不先做“模式切换器”

竞品研究支持 `Ask / Plan / Run`，但第一轮只显示由现有状态推导的：

- `Planning`
- `Running`
- `Scientific review`
- `Blocked`

暂不增加可点击的 `咨询 / 设计研究 / 执行` 模式。原因是 UI-only 模式无法保证消息是否会修改配置或触发动作，会制造新的承诺落差。只有服务端具备明确的 read-only/proposal/execute 权限合同后，才把它做成用户控制。

## 状态来源与单一真相

| 右栏内容 | 唯一来源 | 首版是否新增持久状态 |
|---|---|---|
| 研究问题、数据源、队列、modules、outcome、window、comparator、export | `guidedSlotSnapshot()` + active StudyContext | 否 |
| 项目标题、step、goal、updated_at | Guided draft/session metadata | 否 |
| 当前运行、进度、取消、终态 | 现有 Agent/Extraction job state | 否 |
| Review ready / blocked | 当前 `guidedGateState()` 与 owner 返回的 gate | 否 |
| artifact、evidence、denominator、digest | Agent review/science-workbench/artifact payload | 否 |
| “字段已真正用于执行” | StudyContext execution binding receipt | 否；没有 receipt 就显示 context-only |
| `legacy / brief / cockpit` 布局预览 | Guided route-local 枚举；localStorage 只保存这个枚举 | 只保存界面选择，不保存科研状态 |

首版不创建 canonical StudyBrief 数据库对象。它先作为一个 **read model** 验证交互价值，避免 Guided slots、StudyContext 和新对象三套状态互相漂移。

## 实施切片

### Patch 1 — `WEB-COPILOT-COCKPIT-01`

**目标：持久 Study Brief + 更有信息量的项目栏；不改变执行行为。**

#### 用户可见变化

1. 右栏把单纯 `0/8` 总进度替换为 Study Brief、阶段、下一决定；
2. 每个字段显示 applied/context-only/missing/conflict；
3. 点击字段回到现有 inline 配置步骤；
4. 左栏显示项目标题、阶段、下一决定，路径降为二级；
5. Demo/Real、Local only/Network 状态常驻。

#### 文件 owner 与合同

新增：

- `static/js/screens-guided-study-workspace.js`
  - owner：Guided 右栏的 Planning/Running/Review 投影与纯渲染；
  - public contract：`buildViewModel(snapshot) -> immutable view`、`render(view, helpers) -> HTML`；
  - allowed dependencies：只接收显式 snapshot/helpers，不调用 API、不读取 `screens-guided.js` 私有闭包、不持有 job；
  - stable reason codes：`brief_missing_required_slot`、`brief_context_only`、`brief_conflict`、`source_provenance_missing`、`review_gate_contract_invalid`、`review_gate_blocked`；
  - tests：纯 Node view-model 与 fail-closed tests。
- `static/css/guided-study-workspace.css`
  - owner：Guided 右栏 Study/Run/Review；
  - class prefix：`.gdsw-`；
  - 从 2,375 行 `guided.css` 迁出当前 aside/pipeline 的相邻样式，再添加新状态样式；
  - 禁止 `!important`、`:has(...)` 和 unrelated route selector。

调整：

- `screens-guided.js`
  - 组装只读 snapshot、传给新 owner、分发现有 edit/open/cancel action；
  - 迁出 `renderAside` / pipeline renderer，使文件净缩小，不继续增长；
  - 不把闭包 state 复制进 sibling。
- `screens-guided-projects.js`
  - 接管左栏 project/session 行渲染；
  - 继续只负责项目呈现，不负责 session persistence。
- `index.html`
  - 显式加载新 JS/CSS，顺序锁定在 `screens-guided.js` 前。
- tests
  - 新增 `tests/js/guided_study_workspace.test.js`；
  - 扩展 `tests/test_webserver_static_routes.py` 的 wiring 与 owner presence/absence；
  - 保留当前 gate、StudyContext、project memory tests。

灰度：

- 只在 Guided owner 内增加单一布局枚举：`legacy / brief / cockpit`；
- 内测先用 URL 参数开启，并只把这个枚举写入 localStorage；
- 默认先保持 `legacy`，任务验收通过后再把新项目默认切到 `brief/cockpit`；
- 不建立全局 feature-flag 框架，localStorage 不得保存 Study Brief、gate、job 或任何科研状态。

#### 明确不碰

- `guided_sessions.py`
- `study_contexts.py`
- Agent runner / research-agent
- Jobs backend
- `api.js`
- `app.js`
- Extraction、Patient、Cohort、Cross-DB、Ideas 页面
- 当前并发 Claude 修改的 research-agent 文件

#### 验收

- 同一现有 slot 在聊天卡与右栏显示一致；
- 缺 outcome/window/modules 时下一决定准确；
- field edit 只回到现有步骤，不出现第二套表单；
- source/question 显示 applied，其余字段显示 context-only；
- 刷新/语言切换后 brief 不丢、composer draft 不丢；
- Guided 主文件和 CSS owner 体积净下降；
- 现有 Guided API 请求数不增加。
- 关闭 route-local flag 后立即恢复 legacy 右栏，canonical slots/StudyContext 不变化。

#### 预计改动

- 1 个新 JS owner、1 个新 CSS owner；
- 3 个现有前端文件的小型 wiring/迁移；
- 2 类测试；
- 无后端行为变更；
- 目标为一个可独立 review/revert 的 commit。

### Patch 2 — `WEB-COPILOT-COCKPIT-02`

**目标：把现有 Agent 终态升级为科学审阅摘要；仍不新增后端。**

#### 用户可见变化

1. 区分 `job done`、`Ready for scientific review`、`Review blocked`；
2. 显示实际执行绑定与 context-only 字段；
3. 显示 denominator、gate checks、artifact/evidence/digest 可用性；
4. 深链到现有 Agent Projects、science workbench 或 artifact viewer；
5. 允许修改研究问题后使用当前流程重跑；
6. human sign-off 继续由 Agent owner 管理，不复制到 Copilot。

#### 实现边界

- 继续扩展 `screens-guided-study-workspace.js` 的 Review view；
- `screens-guided.js` 只负责复用已有 API、保存有界结果状态、传入 renderer；
- 不在 Copilot 重写 evidence validator、artifact parser 或 sign-off；
- 初版不做持久 DomainDiff，只显示 `current brief vs execution binding` 的计划差异；
- denominator/result 变化只能来自 owner receipt，不能由前端推算。

#### 验收

- terminal `done` 不会自动显示 scientific-ready；
- 缺 gate、未知 gate、字段不全、hard fail 一律 blocked；
- `analysis_only` 始终显示 not reportable；
- 每个 review 数值能定位到 owner payload/artifact；
- Copilot 不能解锁 manuscript；
- Review 深链回原 owner，返回后仍保持同一 StudyContext。

#### 预计改动

- 不新增第二个 Review JS owner，保持 change surface 小；
- 主要修改新 Study Workspace owner和少量 Guided adapter；
- 增加 review-ready/blocked 的 Node contract tests；
- 目标为第二个可独立 review/revert 的 commit。

### Patch 3 — `WEB-COPILOT-COCKPIT-03`（条件式）

**只有 Patch 1/2 的可用性验证通过后才做。**

范围仅为同一 FastAPI 进程内的 Guided job 恢复：

- 打开项目时用已保存 `job_id` 查询现有 job snapshot；
- running 时恢复进度并重新连接 SSE；
- done 时恢复 Review；
- failed/cancelled 显示明确终态；
- 404 明确显示服务重启后任务已中断；
- stale event、重复订阅和旧 project callback 必须被 operation token 拦截。

不做跨进程持久任务系统，不修改 `jobs.py`，不复制 Extraction/Cross-DB 私有闭包。

## 第一轮明确不做

- 不重写三栏 shell；
- 不把聊天替换成完整 artifact canvas；
- 不新建 typed DAG 或 scheduler；
- 不做多 Agent 人设选择器；
- 不让用户学习模型/provider 术语；
- 不在 Copilot 重做 Patient/Cohort/Cross-DB/Extraction/Idea Mining；
- 不做 persistent branch/checkpoint；
- 不做复杂 DomainDiff；
- 不自动运行、自动接受结果或自动写入论文；
- 不做 cloud sync；
- 不把 row-level 数据写入对话或 StudyContext；
- 不新增全局 feature-flag 框架；只允许一个不承载科研状态的 Guided route-local 布局枚举；
- 不做手机/平板 QA；
- 不把新 CSS/JS 塞进 `redesign.css`、`tweaks.*`、`app.js` 或 `guided.css` 文件尾。

## QA 与发布门

### 纯逻辑合同

- Study Brief 字段映射、完整度和下一决定；
- applied/context-only/missing/conflict 四态；
- gate missing/unknown/invalid fail closed；
- analysis-only 与 reportable 分离；
- Demo/Real provenance；
- stale source/context revision。

### 现有回归

- `tests/test_webserver_static_routes.py`
- `tests/test_webserver_ux_reliability.py`
- `tests/test_webserver_study_context_frontend.py`
- `tests/test_webserver_workspace_summary.py` 的 Guided slot/project memory 子集
- `tests/js/guided_gate_state.test.js`
- `tests/js/study_context_lifecycle.test.js`
- 所有受影响 JS syntax checks

### Ownership

- 新 JS/CSS expected marker 在 owner 中；
- Guided Study Workspace marker 不出现在非 owner route；
- Patient/Cohort/Cross-DB/Extraction/Settings marker 不进入新 owner；
- CSS brace/comment scan；
- 无 `!important`、`:has(...)`、catch-all override。

### 桌面浏览器

固定任务：

1. 新建研究文件夹；
2. 选择数据源；
3. 配置 cohort、outcome、window、modules、export；
4. 从右栏修改 outcome/time window；
5. 进入 Agent preflight；
6. 验证 ready 与 blocked 两种 gate；
7. 打开 Agent artifact，再返回 Guided；
8. 刷新并确认项目/brief 状态。

视口：

- 1180×800
- 1024×768
- 1280×720
- 1440×900

必须满足：

- document 0 overflow；
- 无裁切；
- 宽内容只在 owner 容器内滚动；
- focus visible；
- 状态切换有明确焦点或 live announcement；
- console 0 error / 0 warning。

## 成功指标与停止条件

先在当前界面记录基线，再用 8 名目标用户做 paired legacy/cockpit 对比；两个等价官方 Demo/确定性 fixture 的顺序对半互换。

### 继续条件

- 8/8 不把 `analysis_only`、blocked 或 unknown 理解为可报告/已接受结果；
- 至少 7/8 能在 30 秒内说出当前研究定义、缺失项和下一步；
- 至少 7/8 能正确区分 applied-to-run 与 context-only；
- 至少 7/8 能在 90 秒内修正预置的错误 outcome/time anchor；
- 至少 6/8 能在 60 秒内进入真实 evidence owner；
- 相比 legacy，中位“到有效 Study Brief”时间至少下降 25%，澄清轮次至少下降 30%；
- 修改 outcome/window 后无需重开项目或重新回答无关问题；
- clarification turns 不增加；
- 现有 backend/API/owner 行为无变化；
- Patch 1/2 各自能单独 revert。

### 停止或重新设计

- 需要修改三个以上无关 owner 才能完成一个 UI 状态；
- 需要复制 Extraction、Agent、Patient 或 Cross-DB 私有逻辑；
- 需要引入第二套 canonical StudyBrief 状态；
- UI 无法诚实区分 execution-bound 与 informational context；
- 新模式控制无法由服务器强制；
- gate/receipt 缺失却只能靠前端猜测；
- 真实用户的下一步判断没有改善；
- 8 人中有 2 人把 context-only 当成已用于执行，或有 2 人无法判断下一步；
- Guided 主文件或 CSS 继续净增长。

命中任一结构性停止条件时，不进入 Patch 3，先修 owner contract 或放弃该交互。

## 回滚策略

- Patch 1、Patch 2 分成两个独立 commit；
- route-local `legacy / brief / cockpit` 只控制展示，关闭即可立即回旧版；
- 不新增全局 feature-flag 系统；
- 不同时修改后端协议；
- Patch 1 只改信息投影，revert 后回到当前 `0/8` 右栏；
- Patch 2 只改 review 摘要，revert 不影响 Agent run、artifact 或 gate；
- 每个 patch 合并前保留真实浏览器基线截图与对应测试结果。

## 建议执行顺序

1. 先做 Patch 1，只验证“用户能否不翻聊天就理解研究定义和下一步”；
2. 通过后做 Patch 2，把 EasyICU 最强的 evidence-bound 差异变成可见交互；
3. 两轮任务测试均通过后，再决定是否投入 job restore；
4. typed RunPlan、branch/checkpoint、真正的 Ask/Plan/Run 权限模式留到后端合同能兑现时。

首轮真正的目标不是做一个完整的新 Copilot，而是证明：

> **只把已有研究真相上提，就能让 EasyICU 显得更可信、更可控、更像科研工具，而不是聊天机器人。**
