# EasyICU Copilot 对话式数据工作台

> 日期：2026-08-23
> 分支：`codex/easyicu-desktop-app-v1`
> worktree：`/Users/haibo/Documents/GitHub/EASYICU-desktop-app-v1`
> 起始基线：`912ed0b`
> 状态：功能、聚焦回归、正式 Copilot 浏览器链路和 Apple Silicon 冻结包 smoke 均通过；未推送、未合并 `main`

## 结论

正式 EasyICU Copilot 对话页现可在同一会话中完成受治理的数据发现与审阅：准备官方 demo 数据源、查看队列汇总和筛选漏斗、查看有界特征分布、按匿名实体序号查看患者时间序列，以及比较 2–6 个已注册数据库。结果由既有 Data Extraction、Cohort、Patient 与 Cross-DB owner 产生，Copilot 只负责会话参数、工具调用和右侧结果投影，不复制科学执行逻辑。

Web 与桌面 App 使用同一套 FastAPI 原生前端和 API，因此本次改动同时进入 Web 源码和重新构建的 `.app` / DMG。`#agent` 仍是 Project Monitor，没有被改造成第二个聊天页。

## 对话工具与责任边界

| Copilot 工具 | 用户能力 | 复用 owner / 安全边界 |
|---|---|---|
| `easyicu_prepare_demo_source` | 在对话中准备官方 MIMIC-IV/eICU demo 来源 | 复用 demo source 与 extraction grant；不绕过下载/转换权限门 |
| `easyicu_review_cohort` | 队列人数、筛选漏斗、模块完整性、最多 8 个特征的有界分布 | 复用 `cohort_review` 与患者 eligibility；只返回聚合结果 |
| `easyicu_review_patient_timeline` | 按匿名 Entity 序号查看单例时间序列 | 不接受原始患者 ID；模型不可见患者原始行、时间戳和主机路径 |
| `easyicu_compare_data_sources` | 2–6 个精确注册来源的模块/特征可用性与描述性比较 | 复用 Cross-DB review；最多 8 个特征，不声称匹配或推断性比较 |

工具结果保存为项目范围内、摘要绑定、只读、原子写入的浏览器快照，文件权限为 `0600`，单快照上限 768 KB。公开 API 只返回经 path-free 校验的投影，不暴露内部 artifact 路径。

## 前端实现

- `screens-guided-pi-data-preview.js` 和 `guided-pi-data-preview.css` 是正式 Copilot 数据预览的 JS/CSS owner。
- 从超预算的 `screens-guided-pi.js` 抽出 `screens-guided-pi-resources.js`，共享状态通过显式 namespace 传递，没有把新逻辑继续堆入 broad shell。
- 右侧预览支持队列 KPI、筛选漏斗、直方分布、匿名患者时间序列和跨库矩阵；宽表只在组件内部横向滚动，不制造页面级溢出。

## 自动化验证

```text
Python focused matrix
274 passed, 4 warnings

Ruff
All checks passed

python tools/run_js_contracts.py
24/24 passed

Desktop boundary
14 passed, 1 warning

cargo test
3 passed, 0 failed

git diff --check
passed
```

Python 聚焦范围覆盖新 Data Workbench、数据包快照、Pi 静态/合同/研究工作流和 Web 路由合同。JS 合同包含新结果投影与资源 owner，桌面边界和 Tauri Rust 测试保持通过。

## 正式 Copilot 浏览器验收

在隔离本地服务、合成导出和支持流式 tool-call 的本地 mock provider 上，从正式 EasyICU Copilot 对话页完成三条真实产品链：

1. “查看队列并展示 Heart Rate 分布”触发 `easyicu_review_cohort`，返回 `easyicu_feature_distribution_ready`，owner 为 `cohort_review`。
2. “查看 Entity 3 的时间序列”触发 `easyicu_review_patient_timeline`，返回 `easyicu_patient_timeline_ready`，owner 为 `patient_drilldown`。
3. “比较 MIMIC-IV 与 eICU”触发 `easyicu_compare_data_sources`，返回 `easyicu_crossdb_comparison_ready`，owner 为 `crossdb_review`。

1280×720 下三类右侧结果均可审阅，console 为 0 error / 0 warning；页面无意外横向溢出。截图：

- `task_logs/browser_qa_20260823/copilot_feature_distribution.png`
- `task_logs/browser_qa_20260823/copilot_patient_timeline.png`
- `task_logs/browser_qa_20260823/copilot_crossdb_comparison.png`

## 桌面发行物复验

重新执行 `desktop/scripts/build_macos.py` 后得到：

- App：`desktop/src-tauri/target/release/bundle/macos/EasyICU.app`
- DMG：`desktop/src-tauri/target/release/bundle/dmg/EasyICU_1.0.0_aarch64.dmg`
- DMG 大小：450,548,074 bytes
- SHA-256：`5aec7eee2c7e214d4c8cc924e0d806a94f8d3af1a59f40e0a5a6be85a398c4f6`

`codesign --verify --deep --strict` 和 `hdiutil verify` 均通过。冻结后端在隔离端口启动后：无 token 的 `/api/catalog` 返回 403，正确 token 返回 200，Copilot 状态和新 CSS/JS 返回成功，新快照路由对未初始化项目返回预期 409，证明路由已打包且保持 fail closed。

## 没有声称的内容

- 浏览器验收使用合成导出和本地 mock provider，不是 MIMIC/eICU 真实数据验证、真实供应商模型验证或临床验证。
- 本轮没有运行完整 exact-head CI、Web E1、完整 Research Agent 科学任务或论文产出。
- ICD 条件仍由既有队列/extraction owner 解析；本轮没有另造任意 ICD 查询语言，也不把现有聚合队列审阅夸大为所有 ICD 查询均已验证。
- 下载、转换和格式导出继续由既有受控 job 执行；本轮验证了对话入口与授权复用，没有重新下载全部官方数据。
- 跨库结果是描述性数据可用性/分布审阅，不代表患者级匹配、因果或推断性比较。
- 安装包仍是 Apple Silicon、ad-hoc 签名的内测包；未做 Developer ID/notarization、Intel Mac 或 Windows 验证。
- 分支未推送、未合并；`main` 未被改写。
