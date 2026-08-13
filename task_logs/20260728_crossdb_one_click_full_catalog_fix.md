# Cross-DB 原始六库一键完整对比修复与验收

- 时间：2026-07-28
- 模块：web
- Task ID：`PATIENT-CROSSDB-VISUAL-PARITY`
- 用户问题：真实模式暴露内部错误码 `raw_database_concept_load_failed`，并要求用户依次完成目录检查、范围、采样和数据库选择；默认流程不够直接。

## 产品与架构收口

- 默认真实路径改为一个主操作：选择原始 ICU 数据根目录后点击“开始完整对比”。
- 默认范围为完整目录：19 个模块、281 个临床特征；仍采用每库最多 200 个实体、每特征最多 600 个值的有界抽样，避免把一次预览变成无界全表扫描。
- 目录检查、数据库识别和任务提交自动串联；快速 12 项、采样强度和数据库勾选保留在折叠的“高级设置”。
- 后端由 Cross-DB owner 调用中央 `concept_output_sources` 合同，将公共输出名编译为可执行源概念，再将结果物化回公共特征名；未观测特征保留为显式 missing，不能被静默删除或伪造。
- 终态 job pointer 必定清除；内部错误码映射为可读消息，不再直接显示给用户。

## 真实数据问题修复

- 修复 MIMIC-IV 复合输出（如 `sep3_sofa1`）被误当成可执行源概念的问题。
- 修复 MIMIC-III `dex` 与 HiRID ventilation 回调链中 `dur_var` 单位声明丢失的问题：来源投影、回调和时间对齐边界现在显式保留或规范化分钟/小时合同，严格校验仍为 fail closed。

## 真实六库验收

在本地六库根目录上完成全目录任务（job `49d6eb27727f`）：

- 终态：`done`
- 数据库：MIMIC-IV、eICU-CRD、AmsterdamUMCdb、HiRID、MIMIC-III、SICdb
- 分块：72/72
- 目录：19 模块、281 个唯一临床特征
- source-feature 单元：1,686 = 1,258 present + 428 explicit missing
- 10 个在六库中均未观测的目录特征仍显式保留
- 各库存在特征数：MIMIC-IV 235、eICU 222、AmsterdamUMCdb 221、HiRID 179、MIMIC-III 213、SICdb 188

以上是聚合分布与覆盖状态，不包含患者级原始行；“281 个目录定义”不等于“每库都有 281 个非空观测”。

## 浏览器验收

- 真实 Cross-DB 页首屏只保留一个主目录卡和一个主按钮。
- 输入目录后按钮立即启用，无需失焦；一次点击已真实触发扫描、六库识别和分块任务。
- 高级设置默认折叠；取消任务返回可读提示。
- 页面不再出现 `raw_database_concept_load_failed`；控制台 0 error/warning，桌面视口横向 overflow 为 0。

## 自动化门

- 受影响后端、前端、静态资源、Cross-DB、duration/callback/medication 回归：`636 passed, 1 warning in 28.65s`
- Ruff：通过
- 变更 JS `node --check`：通过
- `crossdb.css` owner/brace/comment scan：通过（347 行）
- `screens-viz-crossdb-setup.js` owner/brace/comment scan：通过（1,078 行）
- `git diff --check`：通过

## 本地提交

- commit：`d52b05a feat(web): unify ICU demo review visualizations`
- 范围：Web 三审阅页、官方 Demo 下载/转换、ECharts vendor、Patient 全特征懒加载、Cross-DB 一键完整目录、必要概念语义与回归测试。
- 提交前暂存集复验：`443 passed, 1 warning`；Ruff、逐文件 JS syntax、cached diff-check 全过。
- 未推送远端；未把并行 Agent 任务日志或未跟踪 `uv.lock` 纳入提交。
