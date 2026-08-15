# PATIENT-CATALOG-GALLERY-AUTOLOAD — 特征目录直达轨迹画廊

> 时间：2026-07-29 22:41 EDT
> 分支：`codex/web-copilot-cockpit-lite-20260729`
> 状态：已实现、已验证，未合并、未 push

## 用户问题

Patient Review 的特征目录虽然能展示 19 个模块和 281 个特征，但与轨迹画廊没有形成连续任务流：

- 已加载轨迹在目录中不可点击；
- 未加载轨迹需要先点“加载”，再自己切换到画廊寻找；
- 画廊切换模块后还需要第二次手动点击“加载模块”；
- 目标特征没有当前患者观测时，缺少直接、诚实的反馈。

## 实现

- 目录中所有可绘图特征统一为 `data-patient-feature-open` 动作：
  - 已加载：直接进入轨迹画廊并定位该特征；
  - 未加载：先进入画廊，再复用既有 `/api/patient-review/feature` 有界请求自动加载；
  - 无多时间点观测：明确显示“已检查，当前患者没有可绘制的多时间点观测”，不伪造折线。
- 画廊 owner 新增 `selectedFeature`，目录跳转后：
  - 所属模块自动选中；
  - 目标特征优先进入对齐主图；
  - 单项图卡加 `aria-current="true"` 和视觉高亮。
- 画廊模块下拉携带真实可加载 feature key，切换模块即调用既有 `loadMany()`：
  - 并发上限仍为 4；
  - 缓存、实体/source scope 与 stale-response guard 不变；
  - 原目录“加载 N 项”和画廊“加载 N 个可用特征”按钮删除。
- CSS 只进入 `patient-catalog.css` / `patient-gallery.css` owner；没有向 broad `screens-viz.js`、`app.js`、`redesign.css` 或 `tweaks.css` 增加 route 逻辑。
- 没有新增后端 endpoint，也没有复制 feature loader。

## 浏览器验收

数据：官方去标识化 MIMIC-IV Clinical Database Demo 2.2，Entity 1。

- 目录 DOM：276 个可交互特征，手动 feature/module load 控件为 0。
- 已加载路径：点击 `sofa2` 后自动进入 `sofa2_score`，目标图卡 1 个、`aria-current` 1 个、SVG 正常。
- 未加载且有轨迹：点击 blood gas `lact`：
  - 即时状态为“正在自动加载 1 条可用轨迹…”；
  - 完成后 loaded trajectories 15→16；
  - Lactate 被置顶并显示 2 个 SVG 视图。
- 未加载且无轨迹：点击 `sofa2_liver`：
  - 自动请求；
  - 完成后明确显示“SOFA-2 肝脏评分已检查：当前患者没有可绘制的多时间点观测”。
- 模块下拉：切到 `sepsis3_sofa1` 自动发起 1 个有界 feature 请求；无轨迹后进入诚实 empty state，无手动加载按钮。
- 布局：document/body 横向 overflow 为 0；目录、画廊、主图、单图网格 clipping 为 0。
- 控制台：只有 EasyICU hydration info，warning/error 为 0。

截图：

- `output/ui-qa/20260729_patient_catalog_to_gallery/01-before.png`
- `output/ui-qa/20260729_patient_catalog_to_gallery/02-after-catalog.jpg`
- `output/ui-qa/20260729_patient_catalog_to_gallery/03-after-gallery.jpg`
- `output/ui-qa/20260729_patient_catalog_to_gallery/04-before-after-catalog.jpg`

## 自动化验证

```text
9 passed
  tests/test_webserver_patient_browse_frontend.py
  tests/test_webserver_patient_demo_data.py::test_patient_series_owner_contract_executes
  tests/test_webserver_static_routes.py::{patient time-series, patient feature-catalog}

85 passed, 1 unrelated known worktree-name failure
  tests/test_webserver_static_routes.py
  tests/test_webserver_patient_browse_frontend.py
  tests/test_webserver_patient_demo_data.py
```

扩展门唯一失败为既有断言把 callback project hint 固定为 `EASYICU`，而隔离 worktree 的目录名为 `easyicu-copilot-cockpit-lite`；与本任务文件和行为无关。三个 JS owner `node --check`、CSS owner presence/absence、brace/comment/foreign-marker scan、`git diff --check` 均通过。
