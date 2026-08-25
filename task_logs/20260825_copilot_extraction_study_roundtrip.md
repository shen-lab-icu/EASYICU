# Copilot ↔ Data Extraction StudyContext 双向串联

日期：2026-08-25  
任务：`WEBAPP-FASTAPI-NATIVE-QA`  
分支：`codex/easyicu-unified-product-20260823`（隔离 worktree；未合并、未推送、未触碰 main）

## 问题

Copilot 已能打开原生数据提取预览，但此前对话中的数据库、队列、时间窗、特征模块和导出格式没有完整投影到右侧 owner；右侧修改后也缺少清楚、可验证的回写收据。这样用户在对话中提出 MIMIC-IV / ICD A41 问题后，仍可能被迫从头配置，且无法判断“同步回 Copilot”到底保存了什么。

## 实现边界

- Copilot 继续拥有自然语言需求、确认流程和项目级 StudyContext。
- Data Extraction 继续拥有本机目录选择、结构扫描、真实抽取、产物和执行状态。
- 两端只通过 path-free StudyContext 与治理回执串联；本机路径和患者行不进入模型上下文。
- 裸 `MIMIC` 必须先确认 MIMIC-III 1.4 或 MIMIC-IV 3.1。模型自行补出的数据库参数不构成用户授权。
- 只有用户当前消息同时明确“本机/本地”和精确数据库时，host 才可恢复模型遗漏的 local extraction 参数；裸 MIMIC 继续 fail closed。

## 完成内容

1. Copilot native-workspace 资源携带规范化、无路径的 `expected_database`，Python 投影和 Node sidecar 都保留同一字段。
2. 原生 Extraction 首帧从 StudyContext hydrate：数据库、ICD 纳入/排除、年龄、观察窗、模块、execution concepts 和导出格式均自动带入。
3. 当前扫描数据库与 Copilot 预期数据库不一致时阻断，不允许跨库静默继续。
4. Extraction 回写区分两类事实：
   - “抽取配置已保存”：保存结构化设置，不声称已经抽取；
   - “抽取结果已同步”：仅真实完成后才携带结果/输出状态。
5. 回写保留原始科学问题和目的，不用界面快照覆盖；未完成抽取时 `current_stage=study_setup`、`extraction_completed=false`。
6. 会话 authority 重绑后，“继续打开本机 MIMIC-IV 3.1 数据提取工作区”只在 extraction scope 下作为续接确认，否定语仍优先。

## 真实浏览器验收

服务从隔离 worktree 的 `src/` 启动，避免 editable install 误导入 canonical checkout；URL 为 `http://127.0.0.1:8897/#guided`。

同一真实 Luna 对话完成：

1. 用户询问裸 MIMIC / ICD-10 A41 / 成人 / 乳酸覆盖率。
2. Copilot 先列出 MIMIC-III 1.4、MIMIC-IV 3.1 和其他支持数据库，而非直接演示 Demo。
3. 用户选择 MIMIC-IV 3.1、本机完整库，并确认成人 ≥18、ICD A41、入科后 24 小时 lact、Parquet。
4. StudyContext 保存并经 authority rebind 后打开原生 Data Extraction。
5. 右侧首屏显示“已从 Copilot 带入”：`MIMIC-IV 3.1 · ICD A41 · 年龄 ≥ 18 · 前 24h · demographics, chemistry, outcome, blood_gas · PARQUET`；路径输入保持空白。
6. 点击“保存配置到 Copilot”后，对话出现“抽取配置已保存”，并显示同一 StudyContext 第 4 版；右侧显示“已同步到 Copilot”。

API 终态核验：

- StudyContext：`study_82d9a8892ad657bc` revision 4
- 原始问题保留为本机 MIMIC-IV 3.1 / A41 / 成人 / 24h lact 覆盖率
- database=`miiv`，path 为空
- modules=`demographics, chemistry, outcome, blood_gas`
- cohort preset=`icd`，age_min=18，include_diagnoses=`A41`，observation_window_hours=24
- export_format=`parquet`
- current_stage=`study_setup`，extraction_completed=`false`

本轮没有替用户选择任何本机目录，也没有声称完成真实数据抽取。

## 验证

- JS owner syntax：通过。
- canonical JS contracts：31/31 通过，含新增 `extraction_study_roundtrip.test.js`。
- 聚焦 Python/静态合同：84 passed。
- Ruff：通过。
- `git diff --check`：通过。
- 浏览器：Copilot 收据、右侧预填、反向同步、空路径和无横向溢出均已验证；console 无 warning/error。

补充：完整 `test_webserver_static_routes.py` 仍含两条与本轮无关、彼此已陈旧矛盾的历史断言；本轮相关的 4 条静态合同均通过，未为追平陈旧断言修改产品行为。

## 下一步

用户在右侧自行选择真实 MIMIC-IV 目录并扫描后，继续验证数据库匹配门、实际队列计算、动态进度、取消及完成结果回写。未获明确确认前不合并、不推送，也不重建 App/DMG。
