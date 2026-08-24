# Copilot MIMIC 数据源确认与本地目录工作区验收

- 日期：2026-08-24
- 分支：`codex/easyicu-unified-product-20260823`
- 基线：`189d0acfd7c7d26fefe797357337d775fb98c503`
- 实现提交：`e54c21c`（提交内容随后仅补充本日志中的提交坐标）
- 状态：隔离 worktree 内完成并本地提交，未推送、未合并，local main 未触碰。

## 问题

用户只说 “MIMIC” 时，Copilot 可能沿用已绑定的官方 Demo，直接返回 Demo 队列与特征分布，造成用户以为结果来自完整数据库。数据查询还允许省略 `source_id`，因此对话模型有机会依赖活动源回退。

## 修复

1. 为六个公开 ICU 数据库投影参考版本；裸 `MIMIC` 首轮只返回数据库家族与版本，不返回可执行注册源。
2. 用户选择精确数据库后，第二轮才列出本机完整库、同库已注册导出和官方 Demo，并明确 Demo 范围。
3. 单源队列、分布、时间线和下载工具强制要求精确 `source_id`，不再回退到 bound/active source。
4. 本机完整库选择在对话内先请求一次性 Extraction 授权；中文“确认授权本轮打开本地数据选择与扫描流程”可被 host 识别，但否定或模糊语句仍 fail closed。
5. 授权后复用 Data Extraction owner，生成 path-private `native_workspace`；右侧显示本机目录输入、浏览和“识别数据目录”，路径和患者行不进入模型。

## 验证

- 静态：`git diff --check`、Python compile、Node syntax、Ruff 均通过。
- 聚焦回归：`156 passed in 4.56s`，覆盖数据库 profile、Copilot contract/data workbench/ICD preview/extraction workspace/gateway/turn authority。
- 真实浏览器 + 本机 `gpt-5.6-luna`：
  - 裸 MIMIC 查询停在 MIMIC-III 1.4 / MIMIC-IV 3.1 确认；没有执行 A41 或乳酸查询。
  - 选择 MIMIC-IV 3.1 后才询问本机完整库、已注册导出或官方 Demo。
  - 选择本机完整库时先等待一次性授权，不产生错误 tool call；确认后生成 `easyicu_local_source_workspace_ready`。
  - 用户点击会话内 `Connect local MIMIC-IV 3.1` 资源后，右侧原生工作区显示目录选择和扫描控件。
  - 控制台 error 0；`document.body` 和 `documentElement` 均为 `clientWidth == scrollWidth == 1814`。

## 截图

- `output/browser_qa/20260824_copilot_source_gate/01_database_version_confirmation.png`
- `output/browser_qa/20260824_copilot_source_gate/02_source_mode_confirmation.png`
- `output/browser_qa/20260824_copilot_source_gate/03_local_folder_workspace.png`

## 边界

本轮停在本地文件夹选择器，没有选择真实 MIMIC-IV 目录、没有扫描患者数据，也没有生成 A41 队列人数或乳酸覆盖率。只有目录扫描、注册和导出完成后，Copilot 才能用精确 `source_id` 继续受治理查询。
