# Patient Review：ECharts、281 特征分层与官方 MIMIC/eICU Demo

> 日期：2026-07-28
> 任务：`WEBAPP-FASTAPI-NATIVE-QA` · `PATIENT-CROSSDB-VISUAL-PARITY`
> 状态：Patient 子里程碑完成；Cross-DB 后续工作仍在进行

## 结果

Patient Review 已接入仓库内置的 ECharts 6.1.0，并把完整概念目录按 19 个临床模块展示。界面不再把“目录里有概念”“有分类观测”和“有足够数值点可画轨迹”混为一谈：

- 281 个目录特征；
- 83 条有至少两个有限数值点的可绘轨迹；
- 3 个有实际分类观测的字段；
- 195 个仅有定义/元数据、当前病例没有可绘记录的字段；
- 每个模块默认最多显示 8 张图，完整 281 特征仍可在模块清单中展开检查。

ECharts 采用本地 vendor、SVG renderer、真实/相对时间轴、阈值线、tooltip、data zoom、ARIA、step line 和显式 dispose；库加载或渲染失败时保留可读降级，不依赖公网 CDN。

## 数据真实性

### 官方 Demo

后端只允许两个固定来源，不接受任意下载 URL：

| 来源 | 固定版本 | 官方范围 | 下载体量 | 许可证 |
|---|---:|---:|---:|---|
| PhysioNet MIMIC-IV Clinical Database Demo | 2.2 | 100 patients | 15.5 MB | ODbL 1.0 |
| PhysioNet eICU Collaborative Research Database Demo | 2.0.1 | 2,500+ ICU stays | 130.6 MB | ODbL 1.0 |

来源卡片公开 landing page、citation、license、版本和准备状态。用户触发后才执行有界下载、安全解压、CSV→Parquet 转换、19 模块导出和本地注册；下载 host、归档路径、symlink/special file、解压体量、缓存 marker 和并发均有 fail-closed 边界。

前端准备任务采用 single-flight：一个来源正在准备时其他来源按钮进入等待态，避免两个重任务互相覆盖轮询；job id/source id 只以有界 pointer 写入 localStorage，页面刷新后会重新连接，终态或错误会清除 pointer。

MIMIC-IV Demo 在隔离缓存中完成真实端到端验证：

- 32/32 表转换成功，0 失败，转换 1,398,600 行；
- 19/19 模块导出，151,373 行；
- 281 个 feature definition，280 个 typed materialization；
- 唯一结构性不可用字段为 MIMIC 的 `vent_free_days_28`，owner receipt 明确它只由 eICU/eICU Demo 支持；
- 第二次 prepare 复用 download/extract/convert/export 四个阶段；
- 默认用户 registry 与当前 active export 均未被测试切换。

持久证据：`output/ui-qa/20260727_patient_echarts/mimic_iv_demo_e2e_receipt.json`。

eICU Demo 的目录、白名单、许可证、安全准备链和测试契约已就绪，但本轮没有声称完成 130.6 MB 数据集的完整下载/转换 E2E。

### 标注的合成兜底

离线兜底仍是明确标注的 deterministic synthetic fallback，不冒充官方 Demo 或真实导出。它使用同一病例严重度状态生成相关联的生命体征、化验、治疗和结局，并保留未建模字段为 `null`。本轮校正：

- `SaFi = 100 × SpO2 / FiO2`；
- `supp_o2 = vent_ind OR FiO2 > 21`；
- 机械通气、advanced respiratory support 和通气窗口一致；
- `mech_vent`、`vent_mode`、`vent_breath_seq` 使用正式分类词表；
- `driving_pres_controlled` 只在 controlled breath sequence 下出现；
- `norepi60`、`epi60`、`dopa60`、`dobu60` 保持药物速率，不再错误生成 Boolean；
- 分类值不转成 0，也不进入数值折线图。

## 模块边界

前端：

- `screens-viz-patient-charts.js`：ECharts option、生命周期和 fallback；
- `screens-viz-patient-series.js`：模块图组、全特征清单和多病例比较；
- `screens-viz-patient-features.js`：281 特征的 availability 分类；
- `screens-viz-patient-demo-sources.js`：官方 Demo 来源卡、准备/激活/打开状态；
- `screens-viz-demo.js`：合成值 owner；
- `screens-viz-demo-drilldown.js`：合成 Patient payload 组装；
- `patient-series.css`、`patient-demo-sources.css`：对应组件样式 owner。

后端：

- `demo_source_contracts.py`：不可变来源目录、路径和诊断契约；
- `demo_source_storage.py`：下载、ZIP 防护、缓存和 marker；
- `demo_source_prepare.py`：转换、导出、注册和阶段编排；
- `demo_sources.py`：118 行兼容 facade；
- `routes/demo_sources.py`：HTTP owner。

所有准备失败都归属到稳定的 `demo_source_<phase>_failed` reason code，并保留底层 cause。

## 验证

- 修改域整组回归：`319 passed, 1 warning`；
- Ruff lint：通过；
- 新 owner 模块 Ruff format check：通过；
- Python compile、10 个 JS syntax、CSS owner presence/absence、brace/comment scan、`git diff --check`：通过；
- API：`GET /api/demo-sources` 返回 2 个固定来源，不暴露本地绝对缓存路径；
- 1180×800 浏览器：
  - 19 个模块、281 个 inventory item；
  - 83 numeric trajectory、3 categorical observed、195 metadata-only；
  - 73 个可见 ECharts SVG，0 fallback；
  - 多病例比较 1 个 ECharts、5 条 legend；
  - 切换 Data Table 后 ECharts instance 归零，返回 Clinical lanes 后正常重建；
  - document 与目标组件均无横向 overflow/clipping；
  - console error/warning 为 0。

截图：

- `output/ui-qa/20260727_patient_echarts/07-echarts-final.png`
- `output/ui-qa/20260727_patient_echarts/08-tooltip-final.png`
- `output/ui-qa/20260727_patient_echarts/09-multi-patient-comparison.png`

## 已知后续

- Cross-DB 的 legacy demo profile 仍可能把分类概念生成为 0–100 连续高斯值；这是 Cross-DB owner 的独立技术债，应改用官方 Demo 聚合或 metadata-only，不能塞回 Patient owner。
- eICU Demo 完整 E2E 可在需要时单独执行并生成同构 receipt。
- `screens-viz.js` 仍是既有超预算集成文件；本轮新功能已进入 sibling owners，后续结构性触碰继续拆，不把功能回填到 catch-all。
