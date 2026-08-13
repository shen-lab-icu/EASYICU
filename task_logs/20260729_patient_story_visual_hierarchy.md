# Patient Review 轨迹画廊视觉层级修正

- 日期：2026-07-29
- 模块 / Task ID：`web` / `PATIENT-STORY-VISUAL-HIERARCHY`
- 分支：`codex/web-copilot-cockpit-lite-20260729`
- 数据状态：官方 MIMIC-IV Demo prepared export；Entity 1；临床重点（跨模块）

## 问题复现

在 1669×1354 的真实浏览器状态下，对齐主图同时呈现 6 条临床轨迹。原图存在以下问题：

1. 单信号轨道仍重复显示顶部滚动图例；
2. 每条轨道使用过密的纵轴刻度和网格线，视觉噪声压过数据；
3. 多条信号颜色相近，左侧轨道标题没有直接给出当前值；
4. 640px 高度仍未形成清晰层级，悬浮提示为大块浅色卡；
5. 正确 SOFA 量程已经建立，但画面没有帮助用户理解“当前值在完整临床量程中的位置”。

修改前截图：

`output/ui-qa/20260729_patient_story_visual_hierarchy/01-before.png`

## 设计与实现

Owner 保持在 `src/easyicu/webserver/static/js/screens-viz-patient-story-charts.js`，宿主高度只在既有 Patient series owner `screens-viz-patient-series.js` 调整；没有修改 Patient API、prepared export、实体选择或数据计算。

- 为 SOFA、灌注、心率、血氧、呼吸、体温建立稳定的临床语义色；
- 单信号轨道隐藏冗余全局图例；只有真正的多信号轨道才显示图例；
- 左侧轨道标题直接显示最新值和单位，并与折线同色；
- 六条轨道使用轻微交替底色、稀疏网格和更少刻度；
- SOFA/SOFA-2 总分继续固定 `0–24`，子项继续固定 `0–4`；
- 单信号轨道使用低透明度面积提示，阈值标签收敛为小型带底色标注；
- 悬浮提示改为深色紧凑卡；缩放条移除无信息量的数据阴影；
- 聚焦图高度由 640px 收敛为 560px。

修改后截图与并排对照：

- `output/ui-qa/20260729_patient_story_visual_hierarchy/02-after.png`
- `output/ui-qa/20260729_patient_story_visual_hierarchy/03-before-after.png`

## 浏览器验证

- 浏览器：Codex in-app Browser；
- 视口：1669×1354，DPR 1；
- 状态：官方演示 → Patient Review → 时间序列 → 轨迹画廊 → Entity 1 → 临床重点（跨模块）；
- 主图外壳：1224×560；
- document 横向 overflow：0；
- chart shell 裁切 / 横向 overflow：0；
- ECharts 可访问名称仍为“重点患者轨迹对齐视图，6 条共享时间轴的临床轨迹”；
- SVG 文本确认轨道标题同时呈现当前值：SOFA-2 7、MAP 69 mmHg、HR 112 bpm、SpO2 94%、呼吸 22/min、体温 37.3°C。

## 回归门

- `node --check`：Patient story chart owner、Patient series host 均通过；
- Node owner contract：通过，覆盖单信号图例、多信号图例、最新值、语义色、SOFA 量程、稀疏网格、深色 tooltip、dataZoom；
- `tests/test_webserver_patient_browse_frontend.py`：6 passed；
- `tests/test_webserver_patient_demo_data.py`：7 passed；
- `tests/test_webserver_static_routes.py` 受影响域：71/72 passed；唯一失败是隔离 worktree 名称 `easyicu-copilot-cockpit-lite` 与既有 callback project hint 断言期望 `EASYICU`，与本次 Patient 图表修改无关；
- 合计：84 passed、1 个已知 worktree-name failure；
- `git diff --check`：通过；
- CSS owner/brace/comment scan 由 Patient browse frontend 门覆盖，本次未修改 CSS。

## 边界

- 这是可视层修正，不改变来源数据、数值、单位或事件语义；
- 没有引入新的共享 CSS/JS catch-all，也没有复制 Patient 路由逻辑；
- 分支未合并、未 push。
