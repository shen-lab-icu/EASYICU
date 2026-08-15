# Cross-DB full catalog scope and honest feature count

> 时间：2026-07-28 05:08 EDT  
> 任务：`WEBAPP-FASTAPI-NATIVE-QA · PATIENT-CROSSDB-VISUAL-PARITY`

## 用户反馈与根因

用户指出 Cross-DB 的模块和特征看起来仍然偏少。浏览器核对发现当前页面处于“真实数据 · 六库快速预览”，该运行按既有安全边界只请求 12 个核心特征，因此结果确实只有 4 个模块和 12 个 feature profiles；这不是前端把全目录藏起来，而是本次 raw job 没有计算其余目录。

同时发现官方双 Demo 结果此前显示 300 个 profiles，其中 19 个是 eICU `patientunitstayid` 在 19 个模块中的重复实体标识，不是临床特征。后端非特征列边界现已排除 `patientunitstayid`、`patienthealthsystemstayid` 和 `uniquepid`，官方结果回到真实目录口径：19 模块、281 个临床特征。

## 产品调整

- 原始数据库路径新增显式范围选择：
  - 快速核心范围：4 / 19 模块、12 / 281 特征，仍为默认首次运行。
  - 完整映射目录：19 模块、281 特征，标记为长时高级运行。
- 完整目录请求不在前端复制 281 个 feature ids；前端只发送 `feature_scope=all_catalog`，由后端 catalog owner 解析当前完整目录。
- 结果页默认特征筛选由“核心概念”改为“全部映射特征”，官方双 Demo 进入分布页即可直接看到全部 281 项。
- raw 快速结果状态栏和三个结果区明确显示 `4 / 19`、`12 / 281`，并注明“不是完整临床目录”。
- 快速结果提供“设置完整目录对比”动作；点击后保留 raw root、数据库识别结果和抽样配置，直接选择完整目录，但不会未经用户确认自动启动长任务。
- raw job reconnect metadata 新增 `feature_scope`，切换核心/完整范围会清理不匹配的旧任务指针，避免恢复错误范围的任务。

## 数据与隐私边界

- 官方 MIMIC-IV + eICU Demo 仍走 prepared export 和真实聚合 API。
- raw 完整目录仍保留每库实体数和每特征值数的有界抽样、聚合返回、协作取消和无患者行返回。
- 本轮没有实际启动六库 281 特征的长任务；请求合同、范围持久化和 UI handoff 已自动化验证。官方双 Demo 的 281 特征则完成了真实浏览器端到端验证。

## 自动化验证

```text
pytest -q \
  tests/test_webserver_static_routes.py \
  tests/test_webserver_workspace_summary.py \
  tests/test_webserver_patient_feature_coverage.py \
  tests/test_webserver_patient_demo_data.py \
  tests/test_webserver_demo_sources.py \
  tests/test_webserver_route_contracts.py \
  tests/test_webserver_crossdb_setup_frontend.py \
  tests/test_webserver_crossdb_raw_scope.py \
  tests/test_webserver_crossdb_job_continuity.py \
  tests/test_webserver_cohort_profile_ui.py \
  tests/test_repository_contract.py

284 passed, 1 warning in 16.41s
```

- 5 个受影响 JS owner 的 `node --check` 全过。
- raw scope、setup、results、job continuity 四个 Node 行为合同全过。
- `ruff check` 与 `git diff --check` 全过。

## 浏览器验收

- 官方双 Demo：2 sources / 19 modules / 281 feature profiles；分布页默认“全部映射特征”，281 个 feature buttons、20 个 module options（含“全部模块”）、1 张主图。
- 页面不存在 `patientunitstayid` 特征项。
- 六库 raw 快速运行完成后显示 `4 / 19 modules`、`12 / 281 feature profiles`，并出现唯一的完整目录 handoff。
- handoff 后 `/Volumes/外置硬盘/databases`、6/6 数据库识别和快速抽样设置保持，完整目录范围为 selected，运行条显示“281 个映射目录特征”。
- 1280 px 桌面视口的官方结果、raw 范围设置和快速结果均无 document 横向 overflow。
- 截图：`output/ui-audit/20260728_crossdb_clarity_refactor/04-full-catalog-scope.jpg`。

