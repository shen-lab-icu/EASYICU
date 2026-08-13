# Patient / Cohort / Cross-DB 共享官方 Demo 来源

日期：2026-07-28 EDT  
任务：WEBAPP-FASTAPI-NATIVE-QA · PATIENT-CROSSDB-VISUAL-PARITY  
状态：done（本子任务）；Web 模块整体仍 in_progress

## 结果

- Patient、Cohort、Cross-DB 均提供“此前导出的数据 / 演示数据”来源选择。
- Cohort 一次选择一个 prepared export；Cross-DB 要求两个来源，并提供 MIMIC-IV Demo 2.2 + eICU Demo 2.0.1 一键官方组合。
- 官方 Demo 保持 `display_mode=demo`，内部以 `processing_mode=real` 调用现有 source-backed review API。
- 原有 6 库 seeded Cross-DB 示例不再是默认结果，仅保留为明确标注、折叠的 Offline fallback。

## 模块边界

- `static/js/screens-viz-patient-demo-sources.js` 保留兼容文件名，但公共合同为 `window.EU_OFFICIAL_DEMO_SOURCES`，由三个 review route 共用。
- `static/css/official-demo-sources.css` 持有共享来源卡样式；Cohort/Cross-DB 特有布局仍在各自 owner。
- `static/js/screens-viz-crossdb-source.js` 持有双源选择与官方 pair 执行；`screens-viz-crossdb-setup.js` 只装配 setup state。
- `webserver/entity_ids.py` 是 Patient/Cohort/Cross-DB 共用实体标识边界，统一 canonicalize `stay_id` / `patientunitstayid` 等数据库键。

## 真实数据证据

- MIMIC Cohort：140 entities，19 modules，279 comparable features。
- eICU Cohort：2,520 ICU stays，19 modules，275 available features。
- 官方 Cross-DB pair：MIMIC 151,373 records、eICU 1,782,280 records；19 shared modules、300 comparable features。
- loaded Cross-DB 结果显示 source-backed density/KDE、单位、样本量与数据质量信息；没有把 synthetic row payload 当作官方数据。

## 验证

- Python affected suite：`273 passed, 1 warning in 15.73s`。
- JS：4 个修改 owner 均通过 `node --check`。
- Node 行为合同：官方来源共享 owner、双源 pair 解析与执行均通过。
- 静态扫描：无旧 `patient-demo-sources.css`、`.pt-demo`、`patient_drilldown.entity_ids` 或旧 cache-version owner 引用。
- 桌面浏览器已验证 Cohort source/loaded、Cross-DB source/loaded；最终 1280×720 Cross-DB source 页 `document/body clientWidth == scrollWidth == 1265`，官方组合按钮可用，离线合成兜底默认折叠。

截图：

- `output/ui-audit/20260728_shared_source_flow/05-cohort-source-shared.jpg`
- `output/ui-audit/20260728_shared_source_flow/06-crossdb-official-pair.jpg`
- `output/ui-audit/20260728_shared_source_flow/07-crossdb-loaded.jpg`
- `output/ui-audit/20260728_shared_source_flow/08-crossdb-final-source.jpg`

## 后续

- Cross-DB 下一次结构性修改应拆出 loaded-result density renderer。
- registered 4-export 长请求仍需独立 async job / timeout / cancel 产品决策。
