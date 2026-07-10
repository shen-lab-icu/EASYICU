# Cross-DB setup owner 拆分与桌面动线 QA

- 日期：2026-07-10
- 模块 / phase：`web` / `WEBAPP-FASTAPI-NATIVE-QA` · `PATIENT-CROSSDB-VISUAL-PARITY`
- 分支：`fix/easyicu-concept-bounds-enforcement`
- 代码提交：`b32c331`（`refactor(web): extract crossdb setup owner`）
- 基线：`e220715`

## 目标

把 Cross-DB 的来源选择、raw root、folder scan、数据库选择、sample budget、loading/reset 编排从共享 `screens-viz.js` 拆入显式 sibling owner；同时关闭拆分暴露出的异步回填、取消遗弃、焦点丢失和来源入口首屏不可见问题。保持 raw 首次运行 12 个核心概念、local-only、bounded-read 和 fail-closed 边界不变。

## 已实现

### 结构边界

- 新增 `src/easyicu/webserver/static/js/screens-viz-crossdb-setup.js`，唯一持有 database selection、view、raw root、scan、sample profile、registered-loading 和 operation sequence。
- 脚本依赖顺序固定为 `raw → progress → setup → shared viz → continuity → source`；continuity/source 在运行时动态解析，不复制旧 IIFE 闭包。
- `screens-viz.js` 从 5,826 行降到 5,359 行；保留 API adapter、Job/Source host 与结果 density renderer，不再持有 setup DOM selector 或双份状态。
- setup owner 904 行，低于 1,500 行 route owner 软预算；`crossdb.css` 293 行，低于 600 行 CSS owner 软预算。

### 状态与并发

- raw、demo、registered 三类请求共享 operation fence；reset、data-mode 和相关 registry 变化后，迟到 promise 不会回填旧 workspace。
- raw start 在 operation 失效后会取消刚创建的后端 job；demo late completion 在 reset 后不会重新变成 loaded。
- raw job 运行时，Edit setup 不再显示；任何残余 reset 入口会转成协作取消，不会 disconnect + forget 后让后端孤儿运行。
- registry 更新不会隐藏独立运行中的 raw job；registered operation 会正确失效。
- continuity restore 移到 bind 最后，避免 `onProbe → repaint` 后外层 bind 对新 DOM 重复绑定。
- scan 同时检查 request sequence 与 root；扫描进行中不复用旧 scan，也不允许重复点击。

### 降低用户门槛与可访问交互

- Source A registered exports 改成可保持展开状态的原生 `<details>`；默认折叠后，Source B raw root 在 1280×720 首屏可见。
- folder scan 使用稳定 `role=status` / `aria-live=polite`；失败 banner 使用 `role=alert`。
- sample、database、scan 与“一键采用已识别数据库”重绘后恢复到语义相同的控件；六库识别后可一键选中，不再手动点四张卡。
- shared folder picker 增加 `aria-modal`、labelled dialog、关闭按钮 accessible name、初始焦点、Tab 焦点环、Escape 关闭和返回触发按钮。
- Source A 展开状态跨 sample/database repaint 保留；registered loading 与 raw progress 仍为明确隔离的两条路径。

## 自动化验证

- 原结构/静态/Patient/Cohort cache-key 修改域：`86 passed, 1 warning`。
- 最终 setup/progress/continuity/source owner 门：`15 passed`。
- 扩展依赖门首次为 `242 passed, 1 failed, 1 warning`；唯一失败是既有取消测试未接受后端已明确实现的 `cancelled_at="aggregating"`。更新测试契约后，失败用例 + Cross-DB progress 全组 `6 passed, 1 warning`。
- 新 Node owner 合约覆盖 stale scan、selection revalidation、scan reuse、300/1500 budget、canonical resume、缺 API 可见错误、服务端文本转义、raw completion loaded、active-raw registry guard、raw reset cancel 和 demo late-completion fence。
- Ruff 通过；setup/source/shared viz 与 Node test syntax 全过；CSS/JS owner presence/absence、brace/comment scan 和 `git diff --check` 通过。
- 未重复全仓 3,000+ 测试；按本次依赖图执行 Cross-DB、security、static routes、workspace summary、Patient/Cohort cache 合约。

## 1280×720 真实桌面 QA

### 来源与 setup

- Source A 折叠高度约 107 px；Source B raw input 位于 viewport `488–503 px`，首屏直接可操作。
- document/content 横向 overflow 均为 0；最终 console 为 0 error / 0 warning。
- `/Volumes/外置硬盘/databases` scan 识别 6 个支持数据库；Check folders 完成后焦点仍在 scan 按钮，live status 为 `polite`。
- Source A 展开后切换 Standard/Quick sample，展开状态仍保留，焦点分别回到当前 sample 控件。
- `Use detected databases` 一次把选择从 2 库扩为 6/6，焦点转到已启用的 raw run CTA。
- folder picker 验证 `aria-modal=true`、labelled dialog、Close 初始焦点、Shift+Tab 环回 Use this folder、Escape 后焦点返回 Browse。

### 真实运行与取消

- 两库 Quick preview 成功返回：MIMIC-IV 7,200 records、eICU 6,792 records、4 shared modules、12 curated features；document/content/density panel overflow 均为 0。
- 六库 Quick preview 在 active progress 时发出协作取消：Cancel 保持 active element，文本变为 `Cancel requested`、`aria-disabled=true`；MIMIC-IV complete、eICU stopping，其余 pending。
- 约 1.5 秒后进入明确 cancelled 终态；progress 消失、alert 为 `Raw Cross-DB density job cancelled before completion.`，未写入成功 workspace。
- registered Source A 点击后显示专属 aggregate-only loading，且 `.crossdb-progress-card` 不存在，证明不会误走 raw scan/job；4 个真实 registered exports 在本轮交互窗口内未完成，因此不宣称 registered terminal/result 已验收。

## 截图证据

- 首屏双来源：`task_logs/2026-07-10_web_crossdb_setup_owner_qa/assets/01-source-choice-setup.jpg`
- 一键采用 6 个已识别数据库：`task_logs/2026-07-10_web_crossdb_setup_owner_qa/assets/06-use-detected-databases.jpg`
- 六库 cancel requested + focus-preserving 状态：`task_logs/2026-07-10_web_crossdb_setup_owner_qa/assets/07-cancel-requested-focus.jpg`

## 边界与下一步

- 本轮不把 registered exports 的长请求称为完成验收；其专属来源隔离已通过，但 4-export terminal 仍需要单独做异步 job/timeout/cancel 产品决策。
- 当前共享 `screens-viz.js` 仍为 5,359 行；下一结构 P1 应拆 Cross-DB loaded-result density renderer，或转向 Cohort 内部大 seam，不能把新功能回填共享 shell。
- 浏览器验收只覆盖桌面/笔记本 1280×720，不声称手机/平板或正式 WCAG 合规。
- 本轮未 push，未触碰并行 research-agent benchmark 或 `research_output/`。
