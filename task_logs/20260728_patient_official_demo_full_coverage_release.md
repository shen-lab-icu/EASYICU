# PATIENT-CROSSDB-VISUAL-PARITY — official Demo full-coverage release

> 时间：2026-07-28 03:10 EDT  
> 模块：web  
> 阶段：WEBAPP-FASTAPI-NATIVE-QA  
> 任务：Patient Review 的 19 模块/281 特征 coverage、按特征懒加载、MIMIC/eICU 官方 Demo E2E、GitHub 快速镜像与公开再分发包

## 结果

- Patient Review 不再用 5 个 bootstrap 模块或 24 条 signal 代表完整数据。后端生成 281-feature coverage index，前端按 19 个临床模块展示 `observed`、`all_null`、`materialized_unknown`、`not_materialized`、`structurally_unavailable`，并只在用户请求时投影读取一个实体的一个特征。
- 数值时序由本地 ECharts renderer 绘制；分类值不画伪折线；无观测或结构不可用概念保留明确状态，不生成随机值。
- MIMIC-IV Demo 2.2 与 eICU Demo 2.0.1 都完成官方 ZIP → 正常转换 → 19 模块导出 → Patient Review 浏览器验收。
- 下载默认走 SHA-256 固定的 GitHub Release asset，失败时回退 PhysioNet；复用缓存、续传文件和新下载文件都必须通过最终 SHA 校验。
- GitHub Release 已公开：<https://github.com/shen-lab-icu/EASYICU/releases/tag/official-demo-data-v1>。

## Owner 与边界

- Patient coverage owner：`src/easyicu/webserver/patient_drilldown/coverage.py`
- 单实体单特征 owner：`src/easyicu/webserver/patient_drilldown/feature_detail.py`
- 跨库实体键边界：`src/easyicu/webserver/patient_drilldown/entity_ids.py`
- Demo 合同/存储/准备：`demo_source_contracts.py`、`demo_source_storage.py`、`demo_source_prepare.py`、`demo_sources.py`
- Release pack owner：`src/easyicu/demo_release_pack.py`；CLI：`tools/build_demo_release_packs.py`
- Patient 前端 owner：`screens-viz-patient-features.js`、`screens-viz-patient-series.js`、`screens-viz-patient-feature-loader.js`、`screens-viz-patient-demo-sources.js`
- API transport 留在 `static/js/api.js`，shell routing 留在 `static/js/app.js`；未把 Patient workflow 追加进 catch-all。

eICU 保留来源列 `patientunitstayid`。浏览器验收首次暴露 `no_entity_denominator` 后，修复放在 Patient Review 实体键 owner：投影读取仍使用来源键，进入审阅内存边界后才 canonicalize 为 `stay_id`。该修复同时覆盖 bootstrap、导航、模块帧和 lazy feature，不在前端伪造分母。

## 真实数据口径

| 数据源 | 实体 | 模块 | 导出行 | 目录定义 | materialized | observed | all-null | 结构不可用 | metadata 未能证明非空 | 可按需加载 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MIMIC-IV Demo 2.2 | 140 | 19 | 151,373 | 281 | 280 | 257 | 4 | 1 | 19 | 276 |
| eICU Demo 2.0.1 | 2,520 | 19 | 1,782,280 | 281 | 276 | 255 | 0 | 5 | 21 | 276 |

`materialized_unknown` 表示 Parquet metadata 不能在不扫患者行的前提下证明非空；它不是假值，用户可通过 lazy feature 请求对所选实体核验。`structurally_unavailable` 表示该数据库缺少结构来源，不是 UI 隐藏。

eICU 原始转换为 31/31 表、0 失败。`sep3_sofa2` 的全链验证曾暴露 pandas 3 singleton `groupby.apply(include_groups=False)` 丢失 `patientunitstayid`；`eicu_rate_kg_callback` 已恢复分组键，`_callback_vaso60` 对缺失来源键 fail-local 重读。最终 eICU `sepsis3_sofa2` 导出 147 行。

## GitHub Release 与校验

公开 Release 为非 draft、非 prerelease，六个 asset 均为 `uploaded`：

| Asset | 字节 | SHA-256 |
|---|---:|---|
| `official-mimic-iv-clinical-database-demo-2.2.zip` | 16,189,661 | `97301a03820e8f41af211cf3462ddc19aefe75bbed05f11753859affaafeb8ec` |
| `official-eicu-collaborative-research-database-demo-2.0.1.zip` | 136,773,541 | `8e33a1094945d6ba07cf613b15b2fe4d98f6b3324601d026e80d445bd5b8b865` |
| `easyicu-mimic_iv_demo_v2_2-prepared-v1.zip` | 1,008,174 | `4b9cfdfc6eedad317813110e92e52e0c63eb466128dd51723e61610408ac37cc` |
| `easyicu-eicu_demo_v2_0_1-prepared-v1.zip` | 5,733,139 | `4f6c981afdbeb1da45bfbc6f7afadec8d054acecb5979462650aa9e71e03a38b` |
| `mimic_iv_demo_v2_2-release-receipt.json` | 879 | `bda936de7e2ff362014c4fef91183738930940658165bdc53b07ed7abbb83b6e` |
| `eicu_demo_v2_0_1-release-receipt.json` | 890 | `6be6503877608f130b5451068aa104077c5303fd2ba06463634fe7935ef1f093` |

项目下载 transport 实测两个镜像最终响应均为 200，MIMIC `Content-Length=16189661`，eICU `Content-Length=136773541`。prepared pack 含 ODbL 全文、`NOTICE.md`、`SOURCE.json`、receipt、`SHA256SUMS` 和 19 模块 Parquet 导出，路径已清洗。

## 许可边界

- MIMIC-IV Demo 2.2 官方许可：<https://physionet.org/content/mimic-iv-demo/view-license/2.2/>
- eICU Demo 2.0.1 官方许可：<https://physionet.org/content/eicu-crd-demo/view-license/2.0.1/>
- ODbL 说明：<https://opendatacommons.org/licenses/odbl/summary/>

本 Release 只再分发上述两个官方公开 Demo，并保留 attribution、license notice、来源与 share-alike 信息。credentialed/full MIMIC 或 eICU 数据不在这个授权范围内，绝不能上传。此处是工程许可审计，不是法律意见。

## 验证

- 修改域集成门：`262 passed, 1 warning in 14.65s`。
- Ruff：Patient/dataio/coverage/identifier/lazy-feature 修改域全过。
- `git diff --check`：修改域全过。
- MIMIC 浏览器：
  - 140 实体、19 模块、151,373 行、281 review features、257 observed；
  - 19 个模块卡；肌酐 lazy request 后新增 ECharts SVG，按钮转为“已加载轨迹”；
  - 1440×900 与 1024×768 均无 document/main 横向 overflow。
- eICU 浏览器：
  - 2,520 实体、浏览器有界审阅 500、19 模块、1,782,280 行、281 review features、255 observed；
  - 19 个模块卡；Entity 2 的肌酐 lazy request 返回 3 个真实时点并新增 ECharts SVG；
  - 1024×768 时 `body 1009/1009`、`main 789/789`；宽审计表只在 `.pt-matrix-details .table-scroll` 内部横向滚动；
  - 控制台 0 error。
- Demo source browser card对两个来源都显示“GitHub 快速镜像 · PhysioNet 回退”。

## 后续边界

- Patient 官方 Demo、281-feature coverage 和按特征 lazy load 已完成；PATIENT-CROSSDB-VISUAL-PARITY 整体仍为 `in_progress`，因为 Cross-DB legacy demo 分类连续值真实性与 loaded-result density renderer 属于 Cross-DB owner 的剩余工作。
- 本轮创建了公开 GitHub Release，但没有提交或 push 当前工作树代码；共享工作树仍包含其他并行任务的未提交修改。

## 终态轮询竞态修复

2026-07-28 03:20 EDT 浏览器复验发现：MIMIC prepare 后端已在约 0.4 秒内完成并注册
19 个模块、151,373 行，但 Patient source owner 在处理 `done` snapshot 时先 repaint、后清理
local reconnect pointer。repaint 会立即重新 bind owner，因此同一终态 job 被恢复成 `running`，
形成持续 `/api/jobs/{id}` 轮询，界面表现为“已就绪”同时仍显示“正在准备数据源”。

修复归属 `static/js/screens-viz-patient-demo-sources.js`：

- 终态先清理 poll timer 与 reconnect pointer，再允许 repaint；
- 非终态调度前清理旧 timer，catch/新 prepare 同样统一清理；
- job status 规范化为小写，并以完整 terminal state 集判定；
- Node owner 合约加入“refresh 立即 rebind”的真实 repaint 竞态回归，锁定终态只请求一次 snapshot。

验证：

- JS syntax + owner contract 通过；
- Patient demo/static 聚焦门 `12 passed, 63 deselected`；
- 新浏览器会话中两个来源均稳定为“已就绪”，等待 3 秒无后续 job poll；
- MIMIC 可打开为真实模式：140 个实体、19 个模块、281 个特征、151,373 行、257 个已观测特征；
- 1280×720：document/body `1265/1265`，无全局横向 overflow。

## Patient 模块级可视化与视图命名修复

2026-07-28 03:30 EDT 用户复核指出：页面虽然声明 281 个特征，但除 bootstrap
预载的 SOFA-2/生命体征外，其余模块默认显示 0 条轨迹；真实的逐特征加载按钮又藏在折叠
清单内，视觉上等同于“没有”。同时“临床泳道”和“单患者”都在查看同一个患者，命名无法
解释两者差异。

修复：

- 三个视图重命名为“模块总览 / 轨迹画廊 / 跨患者对比”：
  - 模块总览：19 个模块、281 个特征、导出观测与当前患者加载状态；
  - 轨迹画廊：当前患者已经加载的逐特征图表；
  - 跨患者对比：同一特征在多个伪匿名患者之间的比较。
- 每个模块新增“加载本模块数据（N）”，仍调用已有的单实体/单特征有界投影接口，
  前端 owner 最多 4 路并发，不增加宽表或直接标识符载荷。
- 特征 cache 扩为 320 个有界 payload，以容纳一个患者的完整 281-feature 审阅；
  source/entity 作用域与 stale response fence 保持不变。
- 新增“展开全部特征清单 / 全部收起”；实际 DOM 为 19 个清单、281 个 feature item。
- 模块加载后严格区分：多时点数值轨迹、单点数值、分类值、该患者无观测、来源不支持；
  非时序 Sepsis 等不生成伪折线。

真实 MIMIC Entity 1 浏览器验收：

- SOFA-1 模块一键加载 7 个特征，得到 6 条真实轨迹；SOFA 肝脏评分明确为“该实体无观测”；
- Sepsis-3 (SOFA-2) 加载后显示真实单点 `值：1 boolean`，并明确说明无多时点轨迹；
- 19/19 特征清单可同时展开，281/281 feature item 可见；
- 1280×720 document/body `1265/1265`，无全局横向 overflow；
- JS owner/语法与 Patient/static 聚焦门 `24 passed, 61 deselected`。

## 官方 Demo 模式与真实取数链路分离

2026-07-28 03:38 EDT 用户复核指出：MIMIC 官方 Demo 打开后，虽然数据来源说明正确，
顶部模式按钮和侧栏“当前配置”仍显示“真实”。根因是 UI 直接把内部 `EU_DATA=real`
当成用户产品模式；而官方 Demo 必须复用本地导出的 source-backed API，不能简单切回
synthetic 的 `EU_DATA=demo`。

修复：

- `i18n.js` 成为全局模式显示合同 owner，新增不可变 mode context，分别记录
  `display_mode=demo`、`processing_mode=real`、`kind=official_demo` 与来源标识；
- Patient 官方来源 owner 在打开已准备数据时写入 provenance context，但不再把
  `easyicu_home_data` 持久化成 `real`；
- app shell、可视化 rail 和 Patient setup 统一消费 `getDataMode()`，不再各自从
  `EU_DATA` 推断产品模式；
- “编辑设置”仍进入官方 Demo 卡片；显式切到“真实”会清除 context、标记旧工作区
  stale，并进入本地导出选择；
- synthetic fallback 会显式恢复 processing mode 为 `demo`，避免官方来源 context
  泄漏到离线兜底。

浏览器验收（MIMIC-IV Demo 2.2）：

- source-backed workspace 仍为 140 个实体、19 个模块、281 个特征、151,373 行；
- 顶部选中“官方演示”，侧栏 pill 为“官方演示”；
- 点击“编辑设置”后“演示数据”仍选中并显示 MIMIC/eICU 官方来源卡；
- 从官方 Demo 显式切到真实模式时，stale-confirm 正常出现并进入本地导出 UI；
- 1280×720 document `1265/1265`，无横向 overflow。

验证：

- JS syntax 与官方来源 Node owner contract 通过；
- WebApp static/Patient/accessibility/UX/Cohort/continuity 回归 `87 passed, 1 warning`；
- `git diff --check` 通过。
