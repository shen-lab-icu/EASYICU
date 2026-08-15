# Patient Review 数据表真实性与密度修复

日期：2026-07-29  
分支：`codex/web-copilot-cockpit-lite-20260729`  
范围：Patient Review → 数据表

## 用户反馈

1. 模块表格概览中的特征、行数、实体覆盖与“静态/动态”是否真实且可解释。
2. 聚合摘要是否来自真实数据。
3. 中文界面出现英文 `Bounded local table previews`。
4. 19 个模块以 chip 铺开，堆叠、冗长，期望下拉选择。

## 数据核验

对当前 active official MIMIC-IV Demo prepared export 调用
`POST /api/patient-review/drilldown`，返回：

- 140 entities、19 modules、151,373 rows；
- 281 EasyICU catalog definitions、257 observed features；
- mean age 62.01、female 45.0%、mortality 5.4%、median SOFA-2 6.0、
  Sepsis-3 positive 47.1%；
- provenance 为 `source_registry`、`export_manifest`、
  `bounded_column_reads` 与 `pseudonymous_entity_reference`。

模块计数来自 catalog + export manifest/schema + Parquet row-group null
statistics。以 blood gas 为例：9 个目录特征、8 个有观测、7 个轨迹候选、
954 行；实体覆盖未计算，不能显示为 0，也不能把内部状态显示成
`unknown`。

## 实现

- 后端 `detail_gate` 保留 canonical English 字段，同时新增
  `title_i18n` / `reason_i18n`；合成兜底合同同步补齐双语。
- Patient table owner 新增明确的聚合摘要来源说明：source-backed 模式标识
  “后端从当前已准备导出计算，不是前端写死”；合成模式明确非临床观测。
- 聚合摘要行改为完整中英文映射，中文模式不再混入 English basis。
- 19 个模块 chip 改为单一 `<select>`；选项保留模块名、行数和加载状态，
  仍复用既有 bounded lazy-page API、cache 与 stale-response guard。
- 原“模块表格概览 + 模块速览”改为默认收起的“模块数据口径”：
  `目录特征 / 已观测 / 轨迹候选 / 表行数 / 实体覆盖`。`未计算`
  明确显示，不再用 `—` 或 `unknown` 暗示零值。
- 样例特征再下沉一层 `<details>`，减少右栏 chip 密度。
- CSS 只进入 `patient-tables.css` owner；未向共享/其他 route CSS 添加选择器。

## 验证

- Node owner 合同：
  `node tests/js/patient_browse_owners.test.js ...` → pass，
  覆盖中文摘要、双语 gate、select render 与未计算覆盖。
- Python 聚焦/扩展门：87 项中 86 passed；唯一失败是隔离 worktree 名称
  `easyicu-copilot-cockpit-lite` 触发现有 extraction callback hint
  `EASYICU` exact assertion，与本次 Patient 改动无关。
- JS syntax、CSS brace/comment、owner presence/absence 与 `git diff --check`
  通过。
- 浏览器 1134×994 official MIMIC-IV Demo：
  - select 19 options；
  - blood gas 口径为 `9 / 8 / 7 / 954 / 未计算`；
  - 切换 vitals 后加载 `1-24 / 12,020`，无 loading/error；
  - 中文摘要六行、中文 privacy gate 与来源标识均可见；
  - module audit 默认关闭；
  - document/main 横向 overflow 均为 0；
  - 旧 `状态 unknown` 与英文 privacy note 均不可见。

## 未改变的边界

- 未扩大 table preview 行/列上限。
- 未返回直接标识符或完整患者表。
- 未修改 Patient、Cohort、Cross-DB 后端执行 owner。
- 未把“未计算实体覆盖”伪造成 0% 或 100%。
