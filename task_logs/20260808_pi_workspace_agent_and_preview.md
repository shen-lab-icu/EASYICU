# Pi Workspace Agent 与右侧产物预览

日期：2026-08-08

任务：`PI-COPILOT-WORKSPACE-AGENT`

分支：`feat/pi-copilot-shell`

## 目标

把 Guided Pi 从只会调用科研配置工具的受限对话壳扩展为仍受项目边界约束、但能真实创建和检查项目产物的 Workspace Agent；把真实工具生命周期投影成 Codex 式对话内轨迹，并让文件/网页步骤可在右侧就地预览。

## 实现边界

- 新 owner `src/easyicu/webserver/pi_copilot/workspace.py` 只拥有 project-scoped UTF-8 文本产物：路径规范化、原子写入、精确编辑、静态检查和预览 receipt。
- Workspace mode 注册 7 个显式 EasyICU 工具：加载 packaged skill、列文件、读、写、精确编辑、静态检查、网页预览。
- Pi 内置文件系统、Shell 和网络工具继续关闭；研究模式仍只有原有 15 个科研工具。Workspace 写权限只对当前消息授权，不能扩张 scientific authority。
- 文件和预览路由要求 project scope；预览带 no-store、nosniff、严格 CSP，并放在不含 `allow-same-origin` 的 sandbox iframe 中。
- 浏览器只接收安全的相对文件 resource metadata 和稳定回执，不接收原始工具参数、文件正文或模型私有思维链。
- 前端 owner `screens-guided-pi-preview.js` / `guided-pi-preview.css` 把右侧 study panel 切换为代码/网页预览；关闭后恢复原 study panel。

## 真实端到端证据

本地真实 `gpt-5.6-luna` Workspace turn 在项目 `draft_ad51a1cdb0f2` 中依次完成：

1. `easyicu_load_skill` → `pi_workspace_skill_loaded`
2. `easyicu_list_project_files` → `pi_workspace_files_listed`
3. `easyicu_write_project_file` → 创建 `icu-risk-demo.html`
4. `easyicu_read_project_file` → 回读同一文件
5. `easyicu_check_project_file` → `html.parser` 通过
6. `easyicu_preview_project_file` → 生成网页预览入口

Job `db0ab082c9c3` 为 `done`，6 个真实工具步骤、最终回答和 1 分 28 秒耗时均进入同一对话 turn。点击“已读取项目文件”打开代码；点击“已准备网页预览”打开 iframe。预览计算器中把年龄从 65 改成 80，结果从 39.4% 变为 42.4%，证明不是静态截图或演示卡。

## 视觉与布局 QA

- 代码预览：`task_logs/screenshots/20260808_pi_workspace_code_preview.png`
- 网页预览：`task_logs/screenshots/20260808_pi_workspace_web_preview.png`
- Codex 参考并排对比：`task_logs/screenshots/20260808_pi_workspace_codex_comparison.png`
- 详细设计判定：`design-qa.md`，`final result: passed`
- 桌面/笔记本视口：`1226 × 994`；document `scrollWidth/clientWidth = 1226/1226`。唯一宽度差异是项目副标题的有意隐藏截断，不是页面溢出。

## 回归门

- Pi workspace/gateway/contracts/static/routes/install + Web route/static owner：`159 passed`。
- Ruff：通过。
- Node syntax：sidecar、event projection、Guided Pi 和 preview owner 全通过。
- CSS 所有权、括号/注释、catch-all 污染、脚本装配顺序和 cache-buster 合同均通过。
- `uv build --wheel --sdist` 成功，wheel 与 sdist 均实际包含 packaged `web-prototype/SKILL.md`。
- `git diff --check`：通过。

## 仍然明确不做

- 不开放任意 host 文件、仓库根目录、Shell、凭据、网络或 Pi 内置 coding tools。
- 不把 Pi session/transcript 升格为科研计划、科学证据或执行 owner。
- 不展示私有 chain-of-thought；用户看到的是生命周期事实、稳定 owner/code 和项目产物。
- 当前 Workspace mode 先覆盖受控文本/单文件网页产物；复杂多文件构建、包管理和浏览器自主导航需要独立权限与 owner 设计，不在本任务中暗中放开。
