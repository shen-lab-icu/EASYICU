# Patient Review Sepsis-3 判定证据时间轴

> 日期：2026-07-29
> 模块：web / Patient Review
> 任务：PATIENT-SEPSIS-EXPLANATION-TIMELINE
> 分支：`codex/web-copilot-cockpit-lite-20260729`（隔离 worktree，未合并、未 push）

## 用户问题

Patient Review 的轨迹画廊选择 `Sepsis-3（基于 SOFA-2）` 时只有一个事件特征，用户无法看到事件发生时间，也无法理解疑似感染与 SOFA-2 急性变化如何共同形成该时点的研究判定。

## 实现

- 新增后端 owner `src/easyicu/webserver/patient_drilldown/sepsis_explanation.py`：
  - 使用 prepared export 中真实 `sepsis_shared.susp_inf`、`sofa2_score.sofa2` 与 `sepsis3_sofa2.sep3_sofa2`；
  - 按当前导出的 `first suspected infection`、`-48h/+24h`、`delta_cummin`、阈值 `2` 重建有界证据窗口；
  - 返回稳定 reason code、三步证据链、研究定义和导出边界；
  - 最多返回 73 个 SOFA-2 点，不返回原始实体标识或整表。
- 新增前端 owner `screens-viz-patient-sepsis.js` 与路由纯 CSS owner `patient-sepsis.css`：
  - 在 Sepsis-3 模块内直接解释“疑似感染锚点 → SOFA-2 急性变化 → 算法事件”；
  - 明确当前导出未保留抗菌药物/标本采集的组成时点；
  - 明确这是 EasyICU SOFA-2 研究推导，不替代床旁诊断。
- 扩展共享 Patient ECharts owner：
  - x 轴显示真实 ICU 相对小时；
  - SOFA-2 y 轴固定 0–24；
  - 画出 -48h/+24h 窗口、疑似感染竖线、累计最低线、Δ≥2 门槛与判定菱形；
  - 保留 tooltip、ARIA、SVG fallback 和 resize/dispose 合同。
- 画廊选择器对该模块显示“判定时间轴 / 1 特征”，不再把事件模块当成“0 条轨迹”的空状态。

## 真实来源核验

Official MIMIC-IV Demo / Entity 1：

- 疑似感染锚点：ICU `0h`
- SOFA-2 累计最低：`2`
- 首个达标值：`4`
- 急性变化：`Δ+2`
- Sepsis-3 研究事件：ICU `0h`
- 时间轴源点：`73`（API 上限同为 73）
- `direct_identifiers_returned=false`

该解释沿用本地 EasyICU 当前 SOFA-2 research derivation；Sepsis-3 共识所述核心临床语义是“疑似感染背景下器官功能急性恶化”，但本界面不把 SOFA-2 研究实现冒充原始 SOFA-1 定义或床旁诊断。

## 验证

- Node syntax：所有改动 JS 通过。
- focused backend/frontend/owner：`18 passed`。
- affected Patient/static/route suite：`148 passed, 1 known worktree-name failure`。
- 最新 Patient/static targeted suite：`89 passed, 1 known worktree-name failure`。
- 唯一失败为隔离 worktree 目录名触发既有 callback-provenance hint 断言：期望路径含 `EASYICU`，实际目录为 `easyicu-copilot-cockpit-lite`；与本任务无关。
- `git diff --check`：通过。
- Browser DOM：
  - document overflow `0`
  - body overflow `0`
  - explanation owner `1`
  - ECharts owner slot `1`
  - SVG `1`
  - chart clipping `0`
  - raw identifier text `false`

## 浏览器证据

- 修改前：`output/ui-qa/20260729_patient_sepsis_explanation/01-before.png`
- 修改后：`output/ui-qa/20260729_patient_sepsis_explanation/02-after.png`
- 前后对照：`output/ui-qa/20260729_patient_sepsis_explanation/03-before-after.png`

修改后截图使用同源 QA harness 调用真实 `/api/patient-review/drilldown`，并复用生产 CSS、JS owner 与 ECharts；临时 harness 已删除，不进入生产静态资源。

## 边界

- 不伪造当前导出未保存的抗菌药物/培养组成时点。
- 不把研究推导写成确认性临床诊断。
- 不新增第二套 Sepsis-3 执行逻辑；UI 只解释 prepared export 已生成的事件，并按同一参数重建有界证据窗口。
- 不合并、不 push；等待用户审阅隔离分支。
