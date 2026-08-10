# Pi Copilot 端到端科研工作流闭环（2026-08-10）

## 结论

分支 `fix/pi-workspace-review-20260809` 已用两个提交完成通用产品闭环：

- `7867adf feat(web): unify Pi research workflow`
- `2b2bb93 feat(agent): run Web Copilot through research pipeline`

Pi 的角色固定为对话、项目记忆和编排；EasyICU 继续拥有 StudyContext、Idea Mining、Extraction、ResearchAgentPipeline、EvidenceStore、科学 gate、结果解读和稿件证据。正式 full run 不再把旧的 native summary / manuscript scaffold 误标为完成分析，而是进入真实 `ResearchAgentPipeline` 的 Plan → Execute → Validate → Write 路径。

## 本轮闭合的用户路径

1. 用户只需在 Copilot 对话提出科学问题。
2. Pi 读取绑定 StudyContext，并在同一对话补齐 purpose、source/export、cohort、modules、outcome、time window、comparison、format、analysis goal 和 confirmations。
3. 可选 Idea Mining 与正式 handoff 继续委托给原 owner；Pi 不生成 novelty 或科学结果。
4. 数据提取由既有 extraction job owner 执行，并把 lifecycle 回写 StudyContext。
5. 本地 preflight 与完整分析使用不同的一次性授权。完整分析调用真实 ResearchAgentPipeline，不再调用旧 summary scaffold。
6. Pipeline 要求计划人工确认时，Web 返回 digest-bound pause；用户明确批准/拒绝后，以新的 provider-run grant 在同一进程续跑。拒绝形成明确 blocked 结果。
7. Web 只投影 Pipeline 自己的 analysis plan、aggregate tables、figures、claim ledger、quality gate、EvidenceStore index 和 bound manuscript scaffold；不自行计算科学数字。
8. 结果解读卡只复用 evidence-bound claims，`generated_numbers=false`；稿件继续保持 analysis-only / human-review 边界。
9. 对话中的文件、图表、表格或网页资源可点击到右侧受治理预览；进度区与预览区复用同一右栏，不割裂流程。

## 安全与治理边界

- Pi 模型配置与科学 Provider 授权仍分离；Pi 的连接成功不自动授权科研 full run。
- full run 和 plan resume 都要求外部模型 opt-in 与一次性 provider-run grant。
- 本机 OpenAI-compatible proxy 使用 server-owned endpoint；Research Agent provider factory 只在显式授权后将该 proxy 的认证 token 交给被验证的 loopback endpoint。
- 公共 metadata 不返回 API key、base URL 或 host path。
- result table 只允许有界 aggregate CSV；带 `stay_id`/`subject_id`/`patient_id` 等列的表被跳过。
- 浏览器投影若发现 host absolute path、credential 或 row identifier value，整组高风险产物被封存并 fail closed。
- same-process human-review pause 不承诺跨服务器重启恢复；重启后必须重新运行，不能伪装 durable resume。
- 本轮没有启动 Canonical9、正式 Provider batch、患者数据实验或 paper authority 解冻。

## 验证

- `.venv/bin/python -m pytest`：Pi/Web/StudyContext/provider factory 相关组合 **250 passed**。
- Web agent/provider/artifact/sign-off 邻接选择集：**33 passed / 104 deselected**。
- 新增科研工作流 owner/fail-closed 测试：**12 passed**，覆盖真实 Pipeline 委托、legacy scaffold 不晋升、aggregate table 投影、identifier table 隔离、host-path fail closed、loopback key 不外泄与 plan-approval 新授权。
- Ruff、Node syntax、`git diff --check` 通过。
- 1366×768 本地浏览器：`scrollWidth == clientWidth == 1366`，对话区独立滚动，composer 底边与 panel/viewport 底边一致；工作流 8 阶段可见；6 个一次性权限可见；历史 timeline 4 组、10 个资源链接可用；点击 `quality_gate.json` 后右栏切换为治理预览；console 0 error/warning。

## 尚未宣称的内容

- 没有用真实 Provider 做一次通用非 Canonical9 科研问题的 UAT，因此不能把本轮写成“任意科学问题已实跑成功”。
- Canonical9 仍按独立 benchmark governance 推进，不因 Web 产品接线自动变成 9/9。
- 文章结果仍必须来自 ResearchAgentPipeline / EvidenceStore；Pi 的自然语言不能替代科学执行或真人签署。

## 下一步

在不触碰 Canonical9 frozen protocol 的前提下，用一个非 Canonical9、非论文 authority 的本地 aggregate demo 做一次显式 opt-in UAT，核对：计划暂停 → 用户批准 → 真实执行 → 表/图预览 → evidence-bound 解读 → manuscript scaffold。任何科学或数据问题应修 owner contract，不向 shared prompt 写 case literal。
