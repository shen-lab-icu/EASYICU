# Pi Copilot 最后三项 P1 收口与真实工具 canary

日期：2026-08-08
分支：`feat/pi-copilot-shell`
实现提交：`ef5ac9d` (`fix(web): close Pi shell authority and runtime gaps`)
审阅输入：`/Users/haibo/.codex/attachments/b3d81ffa-9994-4785-8360-ce6a4f5c77e2/pasted-text.txt`

## 结论

审阅列出的三项 release-blocking P1 已全部在 owner 边界关闭：shell 预算现在逐 provider call 授权；ResearchProject 与 StudyContext 由 Host 持久绑定且所有 session 操作校验 project scope；安装后的 Pi runtime 由源码/锁文件 SHA256 与精确 Pi 包版本共同形成内容身份，完整性不符时 fail closed。Pi V1 的真实工具循环也已通过本地精确模型 canary。

## 1. Provider-call 级预算

- owner：`src/easyicu/webserver/pi_copilot/node_app/src/shell-budget.mjs`
- public contract：每次 `agent.streamFunction` 真正发出 provider request 前，检查累计已消费 token、当前 request 输入的保守上界、预留最大输出，以及 message/session provider-call ceiling。
- 默认调用上限：每条消息 8 次、每个 session 128 次；可由独立 `EASYICU_PI_*` 环境键配置。
- Pi 隐式 retry 已关闭，wrapper 固定 `maxRetries=0`，避免隐藏请求绕过计数。
- 稳定码：`pi_shell_token_budget_exhausted`、`pi_shell_message_provider_call_budget_exhausted`、`pi_shell_session_provider_call_budget_exhausted`。
- 未提供可信费率表时，公开 usage 返回 `cost=null`、`pricing_available=false`，不再显示假 `$0`。

## 2. Project scientific namespace

- owner：`src/easyicu/webserver/pi_copilot/project_authority.py`
- public contract：一个 `project_id` 只能绑定一个 `StudyContext.id`，一个 StudyContext 也只能属于一个 project；映射以 0600 原子 JSON 持久化。
- 新项目初始化自己的 typed StudyContext，不再回退到“当前全局 active context”。
- create/list/get/message/rebind/abort 全部同时校验 `project_id + session_id + mapped StudyContext`；浏览器不能再提交 arbitrary `study_context_id`。
- 旧的未绑定 session 在第一次 project-scoped 访问时迁移；错项目或错 StudyContext 使用稳定 mismatch code fail closed。

## 3. Content-addressed runtime

- owner：`src/easyicu/webserver/pi_copilot/install.py`
- manifest 覆盖 `package.json`、`package-lock.json`、README/notices、`main.mjs`、event projection、shell budget 的 SHA256，以及 lockfile 中 9 个精确 `@earendil-works/pi-*` 版本。
- runtime 目录为 `0.84.1-<manifest-sha12>`；安装使用 `npm ci --ignore-scripts`、临时目录和原子 rename。
- 启动前重新比对 manifest、文件 bytes 与安装包版本；已存在但不匹配的目录不再静默回退，返回 `pi_runtime_integrity_mismatch`。
- 实际临时安装：239 个 npm 包审计、0 vulnerability；manifest `41054a71de52b6c8e1ac656db980196eeaeb0c13c297c96621e500c2190bb616`，`runtime_install_smoke=ok`。

## 4. 同轮 P2 收口

- busy session 的 JSONL eviction 改为跳过或进入 pending retirement，message 结束后 dispose/unlink，不再形成 orphan。
- 文档明确：私有 CWD 是 Pi AgentSession 的逻辑 workspace，不是 Node 进程的 OS filesystem sandbox。
- blocked/failed 宿主工具回执在实时事件与恢复 transcript 中都投影为 `is_error=true`；稳定 owner code 不再丢失。
- 一次授权仍定义为一次 action attempt；owner 拒绝后 grant 被消费是刻意的保守安全语义。

## 5. 真实工具 canary

使用已验证的本地 provider 配置和精确模型，在临时 StudyContext、project authority、session JSONL 与 workspace 中运行；未打印或写入凭据。

真实模型工具序列：

1. `easyicu_inspect_context` → `study_context_projected / ok`
2. `easyicu_update_study_context`（唯一 Configure grant）→ `study_context_updated / ok`
3. 同一 turn 再次 `easyicu_inspect_context` → `pi_session_authority_stale / blocked / is_error=true`
4. Host 显式 rebind 后 stale=false
5. `easyicu_inspect_capability` → `capability_policy_projected / ok`
6. `easyicu_run`（唯一 Run grant）→ `study_context_source_required / blocked`，证明 preflight 由现有 StudyContext owner fail closed，而不是 Pi 自己启动科学运行。

该完整 canary 共 7 次 provider call；公开 cost 为 unavailable；投影扫描未发现凭据、绝对路径或患者标识符。

## 验证证据

- Pi/Web 相邻门：`158 passed, 1 warning`
- 最后 authority/gateway 复验：`35 passed`
- Ruff：通过
- Node syntax / npm check：通过
- runtime 真实临时安装：通过，0 vulnerability
- wheel + sdist：均包含 `main.mjs`、`event-projection.mjs`、`shell-budget.mjs`、package.json 与 lockfile
- `git diff --check`：通过

## 非阻塞后续

- custom hostname 的 DNS validation/request 仍有理论 TOCTOU；当前 local-first、用户手工配置和 loopback 默认下记为 P2。正确修复需要统一 provider transport 的 DNS pin/connection boundary，不能用“再解析一次”制造伪安全。
- Anthropic Messages / Google Generative AI 的真实外部 key canary 尚未运行；协议和验证路径有 contract test，但不声称已有真实 provider 证据。
- 本轮按变更面运行 158 条 Pi/Web 邻接门，没有重复跑整个仓库；Canonical9、论文与其他科学模块不在本任务范围。
