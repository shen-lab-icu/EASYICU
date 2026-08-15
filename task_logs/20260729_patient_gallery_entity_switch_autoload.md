# PATIENT-GALLERY-ENTITY-SWITCH-AUTOLOAD — 切换患者后续接当前轨迹范围

> 时间：2026-07-29 23:00 EDT
> 分支：`codex/web-copilot-cockpit-lite-20260729`
> 状态：已实现、已验证，未合并、未 push

## 用户问题

Patient Review 的轨迹画廊会保留用户所选模块，但切换病例后没有为新患者重新加载该范围。典型复现：

1. 在 Entity 1 画廊选择 `SOFA-1 评分`；
2. 切换到 Entity 5；
3. 下拉仍显示 `SOFA-1 评分`，画廊却停在 `0 轨迹 / 7 特征`；
4. 用户只能再次手动切换模块，才会触发加载。

## 根因

- lazy feature cache 正确按 `source + entity + feature` 隔离，旧患者轨迹没有串到新患者；
- 画廊 owner 也正确保留了 `selectedModule` / `selectedFeature`；
- 但自动加载只绑定在模块 `<select>` 的 `change` 事件；
- 实体导航完成后只是重新渲染画廊，不会产生新的 `<select>` change，因此新患者没有收到同一范围的 feature 请求。

## 修复合同

- 画廊 owner 使用 `source + entity + selected module/feature` 形成一次性自动加载键；
- 每次画廊 bind 时主动对齐当前实体与当前浏览意图：
  - 同一实体、同一范围重复渲染不重复请求；
  - 切换实体后保留模块/特征选择，为新实体重新调用既有 `loadMany()`；
  - 从目录点入单一特征时，只续接该特征，不扩成整个模块；
  - feature loader 既有按实体缓存、最多 4 路并发和 stale-response guard 不变；
  - 新患者没有多时间点观测时继续返回诚实空态，不复用上一位患者数据。
- 画廊说明文案明确告诉用户：切换患者会保留当前范围并为新患者重新加载。
- 未新增 endpoint、未修改后端取数、未向 broad `screens-viz.js` 或 catch-all CSS/JS 增加逻辑。

## 浏览器验收

数据：官方去标识化 MIMIC-IV Clinical Database Demo 2.2。

1. Entity 1：
   - 选择 `SOFA-1 评分`；
   - 即时显示“正在自动加载 7 条可用轨迹…”；
   - 完成后显示 `6 轨迹 / 7 特征`，主图为 6 条 SOFA-1 轨迹。
2. 切换 Entity 5：
   - 下拉仍保持 `SOFA-1 评分`；
   - 不需要再次操作，立即显示“正在自动加载 7 条可用轨迹…”；
   - 完成后显示该患者自己的 `5 轨迹 / 7 特征`，主图为 5 条轨迹；
   - Entity 1 的 6 条结果没有泄漏给 Entity 5。
3. 桌面视口 `1521 × 1354`：
   - document `scrollWidth == clientWidth == 1506`；
   - main `1258 == 1258`；
   - gallery `1188 == 1188`；
   - 主图与单图均无横向 overflow/clipping。

截图：

- `output/ui-qa/20260729_patient_switch_gallery_autoload/01-before-entity5-empty.jpg`
- `output/ui-qa/20260729_patient_switch_gallery_autoload/02-after-entity5-sofa1-loaded.jpg`

## 自动化验证

```text
8 passed
  tests/test_webserver_patient_browse_frontend.py
  tests/test_webserver_static_routes.py::{patient time-series, patient source wiring}

77 passed, 1 unrelated known worktree-name failure
  tests/test_webserver_patient_browse_frontend.py
  tests/test_webserver_static_routes.py
```

扩展门唯一失败仍为既有 callback provenance 断言把项目目录名固定为 `EASYICU`，而隔离 worktree 名为 `easyicu-copilot-cockpit-lite`；与本任务文件和行为无关。`git diff --check`、Patient CSS owner presence/absence、brace/comment scan和浏览器 overflow/clipping 检查均通过。
