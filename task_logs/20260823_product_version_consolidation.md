# EasyICU Web / Desktop / Copilot 版本收口

日期：2026-08-23

## 结论

在不修改 `main`、不覆盖旧 worktree、不删除任何分支的前提下，建立统一产品候选：

- worktree：`/Users/haibo/Documents/GitHub/EASYICU-unified-product-20260823`
- branch：`codex/easyicu-unified-product-20260823`
- base：`codex/final-ci-web-reconcile-20260823@e6d5f43`，其 merge base 为 `origin/main@8115f93`
- 纳入：Web 收口、Project Monitor 恢复、macOS 桌面壳、Copilot 原生数据工作台、原生数据提取、ICD 队列预览
- 未纳入：任何旧 worktree 的未提交改动、Dev9/Scorer 科研分支、现有 App/DMG 二进制晋升
- 推送/合并：均未执行

本候选将 Web 与 App 共用的 `src/easyicu/webserver` 收到同一提交链。桌面 App 仍是薄 Tauri 壳；后续 Web 功能继续只实现一次，但发布 App 时必须从候选 exact HEAD 重新冻结后端并重打包。

## 提交来源与顺序

1. `e6d5f43`：远端 main 之上的 Final-CI Web 对齐与 Project Monitor 恢复基线。
2. `6f6be54` → `d4f7990`：8 个桌面/Copilot 已提交改动，按原提交顺序重放。
3. `77460a9`：对话内 ICD 队列预览。
4. 本轮收口修复：更新合并后的静态资源合同；抽出 `screens-agent-run-history.js` owner，保持 Project Monitor 行数 ratchet，不放宽护栏。

整合只出现两处职责可分离的冲突：

- `app.py`：同时保留 GZip middleware 与 desktop session middleware。
- `index.html`：同时保留 Guided project owner 与新版 extraction/Copilot 资源版本。

没有使用整文件覆盖解决冲突。

## 当前版本分层

| 层级 | ref / worktree | 状态 | 用途 |
|---|---|---:|---|
| 上游基线 | `origin/main@8115f93` | clean | 远端权威基线 |
| 统一产品候选 | `codex/easyicu-unified-product-20260823` | 本轮提交后 clean | 唯一 Web + Desktop + Copilot 整合候选 |
| 原 Web 收口 | `codex/final-ci-web-reconcile-20260823@e6d5f43` | clean | 只读来源/回溯 |
| 原 Desktop | `codex/easyicu-desktop-app-v1@d4f7990` | 28 paths dirty | 隔离待审，不作为发布源 |
| 原 ICD 预览 | `codex/copilot-bounded-data-preview-20260823@77460a9` | clean | 已纳入候选，保留回溯 |
| 本地 main | `main@1e5cda1` | 46 paths dirty，behind 3 | 不动、不发布、不作为整合源 |
| Demo capture | detached `1e5cda1` | 12 paths dirty | 只保留演示现场 |
| True Copilot demo | detached `1e5cda1` | 34 paths dirty | 只保留演示现场 |
| Scorer | `codex/scorer-reseal-record-20260823@2c5cded` | clean | 科研独立线，不混入产品候选 |
| Dev9 | `codex/dev9-quality-remediation@3cdb0a4` | clean，且有运行中 benchmark | 科研独立线，本轮完全未触碰 |

已有 recovery refs 继续保留：

- `backup/final-ci-full-wip-20260823T132020Z@570adfd`
- `backup/final-ci-web-wip-20260823T132020Z@3ee30a2`

## 验证范围

- Python 聚焦合同：Desktop、Copilot workbench、native extraction、ICD preview、workspace/research workflow、静态路由、CSS/JS ownership 等。
- JS contracts：27/27。
- Ruff：Web 与本轮相关测试全通过。
- Node syntax：`screens-agent.js` 与新 `screens-agent-run-history.js` 通过。
- owner ratchet：`screens-agent.js` 由 2048 行降至 2014 行，未提高既有 1974 + 40 slack 护栏。
- 浏览器：隔离服务 `127.0.0.1:8898` 实测 `#guided`、`#extraction`、`#agent`；三页横向溢出均为 0，console error/warning 为 0；run-history owner 资源在 `screens-agent.js` 之前成功返回 HTTP 200。
- 未运行 full exact-head CI：当前仍是产品收口迭代，不是 merge/release freeze。

## 发行边界

现有 App/DMG 仍绑定 exact source `905d0b8`，不包含本候选后续的 ICD 预览、Web 收口和 run-history owner 修复。只有在候选 HEAD 冻结并重打包、codesign/hdiutil、App 路由 smoke 与干净退出验证完成后，才能称为新桌面候选包。

## 后续门

1. 对统一候选做一次用户可见 UAT，确认 Copilot → 数据预览/提取 → Project Monitor 的产品流程。
2. 决定是否把原 Desktop 的 28-path dirty 全流程实验逐块审入；禁止整包覆盖候选。
3. 用户确认后再归档旧 worktree；本轮不删除、不 prune、不 stash、不移动。
4. 准备 merge/release 时再跑一次 full exact-head CI，并从同一 HEAD 重打包 macOS App/DMG。
