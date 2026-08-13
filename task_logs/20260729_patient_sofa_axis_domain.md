# Patient Review SOFA 轨迹固定量程修复

日期：2026-07-29  
分支：`codex/web-copilot-cockpit-lite-20260729`  
范围：Patient Review → 时间序列 → 轨迹画廊

## 问题

对齐轨迹图的每个 Y 轴都设置了 `scale: true`，ECharts 因此按当前患者的
最小/最大观测自动缩放。Entity 1 的 SOFA-2 总分最高只有 7、子项最高只有
2，但曲线会占满各自轨道，容易被误读为接近量表上限。

## 修复

量程策略由 Patient clinical-story chart owner
`screens-viz-patient-story-charts.js` 持有：

- `sofa` / `sofa2` 总分：固定 `0–24`，主刻度间隔 4；
- `sofa*_resp/coag/liver/cardio/cns/renal` 子项：固定 `0–4`，主刻度间隔 1；
- 其他生命体征、实验室和治疗信号仍保留数据驱动量程；
- 数据点、时间轴、tooltip、阈值线和缩放交互均未改变。

这次只修正视觉参照系，不修改后端源数据或 SOFA 计算。

## 验证

- Node owner contract 新增固定量程断言：
  - SOFA 总分 `min=0, max=24, interval=4, scale=false`；
  - SOFA 子项 `min=0, max=4, interval=1, scale=false`；
  - MAP 等非 SOFA 信号继续 `scale=true`。
- Patient 聚焦门：`15 passed, 1 warning`。
- JS syntax 与 `git diff --check` 通过。
- 1134×994 official MIMIC-IV Demo 浏览器实测：
  - SOFA-2 总分轴显示 `0, 4, 8, 12, 16, 20, 24`；
  - 五个已加载子项轴均显示 `0, 1, 2, 3, 4`；
  - 当前总分 7 不再视觉顶格；
  - document/main 横向 overflow 均为 0；
  - console warning/error 为 0。
