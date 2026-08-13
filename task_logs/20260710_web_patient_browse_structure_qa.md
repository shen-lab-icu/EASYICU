# Web Patient Review 导航与大表懒加载结构 QA

- 日期：2026-07-10
- 模块：Web
- 阶段：`WEBAPP-FASTAPI-NATIVE-QA` / `PATIENT-CROSSDB-VISUAL-PARITY`
- 分支：`fix/easyicu-concept-bounds-enforcement`
- 提交：`06b5b02`（有界 Patient browse 后端）+ `f5e37dd`（导航/表格前端 owner 拆分）
- 数据：`/Volumes/外置硬盘/easyicu_fullexport_miiv_20260622`
- 范围：FastAPI native Patient Review；桌面/笔记本 1180×800，不包含手机/平板。

## 结论

Patient Review 的三个结构 P1 已完成：实体入口从 5 个固定选项升级为全 94,458 例的有界分页/随机导航；切换单病例不再重建完整 workspace；模块表切换和翻页只读取一个模块的一页，并使用绑定 source revision 的 12 页有界缓存。

首次 drilldown 仍保留 500-entity bounded clinical/quality review，单病例比较仍最多 5 例；新增导航不会把直接临床标识符返回浏览器，也不会把全量患者行写入本地状态。

## 后端边界

- 保留 `/api/patient-review/drilldown` 作为首次 bootstrap。
- 新增 `/api/patient-review/entities`：默认 12、最大 24 个伪匿名实体；返回 ordinal、page/page_count、row range，不返回 `stay_id`。
- 新增 `/api/patient-review/entity`：必须同时提供 `entity_ref + entity_ordinal`，服务端重算 token；篡改或错配 fail closed。
- 新增 `/api/patient-review/table-preview`：严格校验 manifest module，只读取一个模块的一页；页大小最大 100、显示列最大 14。
- 未知 module 不再回退首模块；读取异常只返回稳定错误码，不暴露本地绝对路径。
- bootstrap 只预读一个默认模块，不再为 19 个模块逐表生成预览。

## 前端拆分

- `screens-viz-patient-navigation.js`：实体分页、随机组、单病例窄请求、stale-response guard、焦点恢复。
- `screens-viz-patient-tables.js`：单模块懒加载、局部 loading/error、12-entry source-bound cache、stale-response guard、焦点恢复。
- `screens-viz.js` 只保留 Patient shell/composition；不再拥有实体/表格异步请求状态。
- Patient navigation/table CSS 分别迁入 `patient-navigation.css` 与 `patient-tables.css`；旧 `pages.css` / `patient.css` 及 Cohort/Cross-DB owner 均无这些 selector。

## 用户动线与可访问性

1. 进入真实 Patient Review 后，先看到 `full cohort 94,458` 与 `bounded browser review 500` 的清晰范围说明。
2. Patient Overview 显示 `1-12 / 94,458 · page 1 / 7,872`，可上一组、下一组、随机组；翻组期间只在 navigator 显示状态，旧病例卡保持可见。
3. 跨到第 2 组后可选择 Entity 13；详情窄请求完成后显示 `13 / 94,458`，没有整页 skeleton。
4. Data Tables 可从 Blood Gas 切到 1,377,675 行的 Chemistry；加载期间旧表保留，完成后状态从 `available · load` 变为 `loaded`。
5. Chemistry 第 2 页为 `25-48 / 1,377,675`；返回第 1 页命中缓存，不重复请求。
6. 当前病例/模块使用 `aria-pressed`；实体组和分页组有可访问名称，loading 使用 polite status、失败使用 alert；异步重绘后键盘焦点回到操作控件。
7. 缺失 ICU LOS 显示 `unknown`，不再出现 `unknown d`。

## 真实浏览器结果

- 视口：1180×800。
- Patient Overview：document `clientWidth=1180`、`scrollWidth=1180`；navigator `clientWidth=874`、`scrollWidth=874`。
- Data Tables：页面 `scrollWidth=1180`；宽表 `clientWidth=874`、`scrollWidth=1844`、`overflow-x:auto`，横向溢出被正确限制在表格容器。
- Console：0 error、0 warning。
- Entity 13 选择后 active element 为对应 `aria-pressed=true` button；Chemistry 加载后 active element 为对应 module button。
- 未声称完整 WCAG 合规；截图之外的读屏器组合与高倍缩放仍需专门测试。

## 自动化验证

- 静态路由/安全/owner/新行为：`123 passed, 5 warnings`。
- Patient 旧契约聚焦集：`10 passed, 114 deselected, 5 warnings`。
- 合计本轮相关门：`133 passed`。
- Ruff：通过。
- 3 个变更 JS owner + Node 行为合约 syntax：通过。
- CSS owner presence/absence、brace/comment 平衡：通过。
- Node 行为回归覆盖：表格失败保留旧页、跨 source/并发 stale response 不覆盖新状态、12-entry cache、Entity 失败保留旧病例。
- `git diff --check`：通过。

## 截图证据

- `task_logs/2026-07-10_web_patient_browse_qa/assets/01-real-data-tables-loaded.png`
- `task_logs/2026-07-10_web_patient_browse_qa/assets/05-entity-13-accessible-selection.png`
- `task_logs/2026-07-10_web_patient_browse_qa/assets/06-final-chemistry-module.png`

## 下一步

- Patient：inventory-only 模块当前只完成表格 preview 懒加载；若要生成 module-level coverage/quality，需要独立、硬行数上限的 module review endpoint，不能塞回首次 drilldown。
- Cross-DB：继续把整库级 progress/cancel 下沉到逐库/逐 chunk；保持 12 核心概念首次运行硬边界。
- `screens-viz.js` 仍远超 owner 软预算；后续结构改动继续拆 sibling owner，不向共享 IIFE 追加 route workflow。
