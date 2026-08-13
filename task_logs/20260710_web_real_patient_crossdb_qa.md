# Web 真实 Patient Review + 六库 Cross-DB 桌面验收

- 日期：2026-07-10
- 模块：Web
- 阶段：`WEBAPP-FASTAPI-NATIVE-QA` / `PATIENT-CROSSDB-VISUAL-PARITY`
- 分支：`fix/easyicu-concept-bounds-enforcement`
- 代码提交：`892f75c`、`132bbf4`
- 范围：FastAPI native WebApp，桌面/笔记本视口；不包含手机/平板验收。

## 结论

Patient Review 已用 94,458-stay 真实 MIMIC-IV 导出完成浏览器验收；Cross-DB 已用本机六个真实数据库完成 12 个核心概念的 quick density 验收。两条动线现在都明确展示真实数据来源、完整队列与浏览器有界审阅范围，且 raw Cross-DB 在任何请求库未加载或患者抽样失败时 fail closed。

本轮不把页面称为数据库两两 `N×N` 统计矩阵：当前真实产物是“每个标准概念一张小图、叠加六库密度曲线”的多库分布网格。

## 真实数据源

### Patient Review

- 导出：`/Volumes/外置硬盘/easyicu_fullexport_miiv_20260622`
- 标签：`MIMIC-IV 94,458-stay full export`
- 完整队列：94,458 stays
- 模块：19
- 行数：98,596,746
- 磁盘体积：约 444 MiB
- 浏览器审阅：500 个有界实体；完整队列分母同时保留。

### Cross-DB

- raw root：`/Volumes/外置硬盘/databases`
- 检测：6/6 数据库文件夹
- 数据库：MIMIC-IV、eICU、AmsterdamUMCdb、HiRID、MIMIC-III、SICdb
- 首次真实运行范围：12 个精选核心概念
- quick 上限：每库最多 200 个实体、每特征最多 600 个值

## 真实浏览器结果

### Patient Review

- 冷启动后的复验加载约 57 秒。
- Loaded bar 明确显示：`full cohort 94,458 entities · bounded browser review 500 entities · 19 modules · 98,596,746 rows`。
- Time Series 不再只截最早 12 点，而是确定性覆盖完整时间窗并保留首末观测：
  - vitals/HR：约 `-1h → 38h`
  - SOFA-2：约 `-1h → 127h`
  - current 值取完整序列真实末次观测，而不是有界数组的偶然末点。
- Quality 明确使用 `500 reviewed / 94,458 full`；只对实际计算覆盖率的 4 个可用性模块绘制百分比，14 个 inventory-only 模块只显示清单/行数，不伪装成 0% coverage。
- event rate / exposure rate 与 missingness 分开标注。
- 从 Cross-DB 返回 Patient 后，左侧 setup rail 正确恢复 94,458 entities / 19 modules，不再继承 Cross-DB 的 4-module 摘要。

### Cross-DB

- 六库 quick run 约 63 秒完成。
- 6/6 requested databases 全部加载；任何请求库缺失都返回 `loaded_fewer_than_requested_raw_databases`，不再用部分成功冒充六库成功。
- 结果：4 个模块、12 个共享概念，12/12 在所选数据库共享。
- 抽样记录数：
  - MIMIC-IV：7,200
  - eICU：6,792
  - AmsterdamUMCdb：7,200
  - HiRID：7,021
  - MIMIC-III：7,132
  - SICdb：7,200
- 修正核心概念键 `gluc → glu`，并把 raw 请求从全 catalog 改为显式 owner 中的 12 概念硬边界。
- 最终运行按钮与固定 Page guide launcher 不再重叠。

## 代码修复

- `easyicu.api.load_concepts` 新增 opt-in `require_bounded_sample`，同时拒绝 `None` 与空列表采样；参数追加在旧 positional tail 之后，保持公共 API 位置参数兼容。
- raw Cross-DB 所有调用启用严格有界采样；请求的每个数据库都必须真实加载。
- Patient Parquet preview 的 bounded batch read 失败时不再退化为整表 pandas 读取。
- Patient 信号最多 12 点，但保留首末端和完整时间窗；latest/current 基于完整有效序列。
- Patient coverage 的 `null` 不再被 `Number(null)` 误转成 0。
- Patient Review scope 元数据贯通 StudyContext；Patient ↔ Cohort 往返会清理另一 route 的 cohort 字段、confirmation 与 UI comparator，同时保持同一项目身份。
- 新增 `screens-viz-crossdb-raw.js` 作为 raw scope owner；Cross-DB CTA 样式留在 `crossdb.css`。

## 桌面浏览器检查

- 视口：1180×800（另核对 1440 宽 source picker）。
- Patient Quality：`html/body/main/content` 均无横向 overflow。
- Cross-DB result：`html/body/main/content/.xdb-density-panel` 均无横向 overflow。
- 控制台：3 条普通日志，0 error，0 warning。
- 未声称 WCAG 或正式 accessibility 合规；本轮只验证主动线、布局、overflow/clipping 与控制台错误。

## 验证

- 修改相关测试文件：`269 passed, 5 warnings in 23.08s`。
- Ruff：通过。
- 5 个变更 JS 文件 `node --check`：通过。
- Patient scope 与 StudyContext lifecycle 两个 Node 行为合约：通过。
- `crossdb.css`：109/109 braces、8/8 comments；owner presence/absence 回归通过。
- `git diff --check`：通过。
- 独立只读终审发现并复核关闭 3 个 P1：空采样 fail-open、Patient/Cohort 状态串线、公共 API positional 兼容；终审结论无剩余阻断问题。

## 截图证据

- `task_logs/assets/20260710_web_real_qa/patient-real-source.png`
- `task_logs/assets/20260710_web_real_qa/patient-real-timeseries-fixed.png`
- `task_logs/assets/20260710_web_real_qa/patient-real-quality-scope.png`
- `task_logs/assets/20260710_web_real_qa/crossdb-real-six-source.png`
- `task_logs/assets/20260710_web_real_qa/crossdb-real-six-result-main.png`

## 下一轮结构项

以下不影响本轮真实验收结论，但仍是 Web P1 backlog：

1. Patient entity navigator 当前只暴露 5 个伪名实体，尚无分页/随机定位。
2. Patient 动态 clinical lanes 只读取部分模块；labs/support 等 inventory-only 模块应按需懒加载，而不是扩大首次 payload。
3. Patient table 翻页仍会重算完整 drilldown，需要拆出模块级分页 endpoint/cache。
4. Cross-DB cancellation/progress 仍以整库加载为粗粒度；需要逐库/逐 chunk 取消点和可见进度。
5. `screens-viz.js` 仍显著超过 JS owner 软预算；下一次结构性改动必须继续把 Patient/Cohort/Cross-DB 子流拆入显式 sibling owner，不再向该共享 IIFE 增长。
6. 现有六库 EasyICU 导出是 long schema，不能伪称 registered aggregate 已验收；本轮验证的是 raw-root 六库路径。
