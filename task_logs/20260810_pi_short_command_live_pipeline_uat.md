# Pi Copilot 单句运行与动态科研时间线 UAT（2026-08-10）

## 任务

- Task ID：`PI-COPILOT-LIVE-PIPELINE-UAT`
- 分支：`fix/pi-workspace-review-20260809`
- 实现提交：`7922951`
- CI 预算修复：`edffda4`
- 范围：非 Canonical9、非论文 authority 的通用科研 UAT；未修改 Canonical9、shared prompt 科学内容或 frozen paper rubric，未启动正式 Provider batch。

## 目标

1. 用户不需要编写长提示词；当前研究已经配置时，一句“运行/重新运行这个研究”即可进入受治理的 EasyICU 执行链。
2. Web 对话不再只显示静态 `job_id` / `running`，而是实时显示真实运行阶段、工具回执、步骤开始/完成、验证门和终态。
3. 任务结束后，用户可在同一对话读取验证状态、证据与产物，并从消息中的资源按钮打开右侧预览。
4. 动态轨迹只展示宿主可证明的生命周期事件，不展示或伪造模型私有思维链。

## 实现边界

- Pi 继续负责自然语言交互、意图识别和工具编排；EasyICU 的 StudyContext、Extraction、ResearchAgentPipeline、EvidenceStore、科学 gate 和稿件 readiness 仍由原 owner 持有。
- 工具回执只暴露经过格式约束的 `job_id`；浏览器据此订阅 `/api/jobs/{job_id}/events`，把 child job 的真实 SSE 事件投影到当前对话。
- 外层 Pi 回合结束不会关闭仍在运行的 child timeline；child job 自己到达 terminal event 后才结束。
- 如果用户对当前已配置研究明确说“运行/重跑/分析”，Pi 被要求执行下一受治理动作，而不是只检查旧运行。
- 一条消息的权限仍然一次性；当用户只勾选“完整分析”而模型保守地请求 preflight 时，宿主依据用户已经授予的更强权限推进 full run，避免要求第二次重复授权。没有 Provider grant 时仍默认本地 preflight。
- 为本次通用描述性 UAT 补齐了 typed universe 选择、严格计划投影及 5 个确定性 owner：分组描述分布、标量 Spearman 描述关联、digest-bound 结果图、cohort flow 图和审计 panel。叙事步骤、无输出图形步骤及漂移 digest 继续 fail closed。

## 真实浏览器 UAT

### 单句启动

在既有项目记忆和新 Pi 会话中输入：

> 重新运行这个研究，并把图表、验证状态和结果解读给我。

只授予一次“完整分析”。系统提交：

- Web job：`c1c4fecf25ba`
- Pipeline run：`run_20260810T194444_35a21a`
- 计划：8 steps
- 终态：8/8 execution steps complete，child job `done`

对话内实际出现的真实事件包括：provider authorized、typed universe materialization、cohort materialised、trajectory authority staged、runtime validation、pre-plan literature、8-step plan、每步 start/executor/complete、figure audit、publication bundle、literature bundle、manuscript scaffold、LaTeX/BibTeX 和 terminal gate。

### 结果读取与预览

第二句只输入：

> 读取这个新运行的图表、验证状态和结果解读。

Pi 自动读取最新 run、`quality_gate.json`、`evidence_ledger.json` 和白名单产物。运行投影为：

- `execution_complete=true`
- `evidence_complete=true`
- `numeric_verified=true`
- `analysis_validated=false`
- `manuscript_ready=false`
- gate：`research_agent_pipeline_failed_closed`
- `reportable=false`
- `draft_unlocked=false`

因此 Pi 正确拒绝复述未验证数值或生成正式科学结论，只说明“执行已经结束，但科学闸门仍锁定”。这证明动态进度没有绕过 fail-closed 科学边界。

对话列出 9 个白名单产物。点击“图件画廊”后，右侧受控预览打开 `figure_gallery.json`，可见 4 张真实图件：1 张 primary publication figure、LOS 按性别分布图、年龄–LOS Spearman 描述关联图和 audit panel；同时显示治理状态、文件身份和可读审计表。

桌面浏览器检查：项目栏、对话栏和右侧预览三栏同时可见；输入框保持在对话底部；对话/预览各自纵向滚动；未观察到横向溢出、遮挡或裁切。

## 验证

- 聚焦/邻接回归：`156 passed, 3 warnings`。
- 推送后的 Research Agent CI 在测试前正确拦截两项本轮回归：`agents/core.py` 比架构基线增长 30 行、固定 Planner 提示达到 53,826 bytes（预算 51,600）。`edffda4` 未更新基线或抬高预算，而是把新增指导归还 `agents/plan_payload.py`，并只向 `descriptive_epidemiology` 投影描述性执行合同；`agents/core.py` 回到 4,933 行基线。
- CI 修复后的联合回归：`312 passed, 3 warnings`；其中 architecture + fixed-prompt budget 42/42，计划投影/类型/预算/owner 与 Web/Pi 组合均通过。
- 精确 HEAD 主 CI 随后只在离线资源上下文基线守卫失败：三种 Python 版本均为同一项 `test_checked_in_resource_context_baseline_has_no_drift`，其余完整套件为 `12653 passed, 196 skipped`。差异来自 `edffda4` 将通用描述性 Planner 指引移入 typed plan-payload owner：E1 获得条件投影，其余八题各减少 2 bytes，全局最大上下文也减少 2 bytes。已用零 Provider、零患者数据的基线生成器追加审议原因和 source digest 历史；未抬高任何预算或修改 Canonical9 科学协议。资源基线、静态架构及全部提示词预算聚焦回归 `194 passed`。
- Ruff：全部本轮 Python 生产文件和测试 `All checks passed!`。
- Node syntax：`event-projection.mjs`、`main.mjs`、`screens-guided-pi.js` 全部通过。
- `git diff --check`：通过。
- 前端 owner：运行时间线逻辑位于 `screens-guided-pi.js`；右侧产物预览继续由既有 `screens-guided-pi-preview.js` / `guided-pi-preview.css` 持有；没有向 catch-all CSS/JS 追加 route-specific 代码。

## 结论

本轮完成了用户提出的两个产品缺口：短命令可以启动当前研究；真实 running 细节会动态进入聊天并保留到 child job 终态。分析完成后，用户能在同一对话查看验证、证据和产物，并打开右侧预览。是否允许科学解读和稿件解锁仍由 EasyICU gate 决定，而不是由 Pi 的文本表现决定。
