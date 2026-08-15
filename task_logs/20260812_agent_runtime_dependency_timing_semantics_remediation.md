# Agent A1–A3 runtime、依赖方向与时间语义整改

日期：2026-08-12
模块：agent
任务：FIG2-CANONICAL9-GATE（工程整改，不改变 paper authority）
工作树：`fix/pi-workspace-review-20260809`，完成复核时 HEAD `45deba3`；共享 dirty snapshot，未提交

## 范围与结论

本轮仅修复用户指定的三条 Agent 缺陷，未启动 Provider、正式 Canonical9 run 或 full exact-head CI；Canonical9 仍为 4/9，frozen rubric 与 paper authority 均未改变。

### A1 — 隔离后端失败与生成代码失败分离

- `execution/runner.py` 新增窄化判定：只有 `unshare` 自身失败、macOS sandbox profile/target exec 被拒、Python stdio 在 sandbox 内启动失败，或已知 OpenMP SHM 启动失败，才写入 isolation-backend fail-closed marker。
- `NameError` 等已经进入隔离环境运行的普通代码异常保留原 traceback，不再被 `runtime_isolation_backend_unavailable` 截走，继续进入 Coder repair 路径。
- 真正没有可用隔离后端的情况仍禁止降级到 host subprocess，保持 fail closed。

### A2 — 移除 planning → execution 反向依赖

- `planning/replan_gate.py` 不再导入 `execution.owner_declaration`。
- execution owner 仍通过真实 selector 编译 owner-declaration findings；`execution/phase.py` 将这组 typed findings 注入 replan gate。Planning 只消费诊断 receipt，不读取运行时 selector。
- owner 声明缺口仍可拒绝非法 replan，包依赖方向门恢复为绿色。

### A3 — 宽表 first_time 恢复为 observation 语义

- Materializer 合同明确：`<c>_first_time/_last_time` 是 materialization window 内首次/末次非空观测坐标；显式 0/event-negative state 也是观测，缺失只代表窗口内没有 qualifying observation。
- `window_first_time` 的 ResearchContext representation 增加 caveat 与 forbidden transformation：不得在缺少 typed state-transition authority 时解释为 clinical onset 或 treatment initiation。
- 全局 Coder prompt 删除“first recorded = initiation/onset”与从多个宽表 `_first_time` 拼 onset 的指导；真正的 onset/time-zero 必须来自 owner-authorized event time，或从绑定 long trajectory 的 qualifying positive transition 推导。

## 验证

- A1 runtime/fallback focused：`16 passed`。
- A2 replan/owner/dependency direction：`53 passed`。
- A3 materializer/metadata/representation/temporal：`134 passed`。
- Coder prompt anchor/budget/output-scope/pandas contracts：`167 passed`。
- Ruff（本轮触及的 Python 文件）：通过。
- `git diff --check`：通过。

初始红测已分别复现 A1 marker 误归类、A2 缺少 receipt 注入边界、A3 prompt/representation 错误语义，修复后均转绿。运行中曾捕获 prompt 稳定 anchor 被改名导致的相邻红测，已恢复稳定 anchor 并保留新的安全语义，随后 167 项 prompt 合同全绿。

## 未做与交接

- 未提交、未推送；共享工作树中存在大量其他 Agent/Web/data 在途改动，本轮未归因、未回滚。
- A1 还依赖本轮开始前工作树已存在的 `execution/failure_classification.py` 与对应测试改动；本轮没有覆盖其作者修改，但原子集成时必须与 runner marker 窄化一起纳入，否则环境失败分类合同不完整。
- 未跑 full CI；按 E1 开发策略，完整 exact-head CI 留到 freeze/merge/formal checkpoint。
- 后续集成时应把 A1–A3 源码、测试与本日志作为一个原子变更复核，避免只纳入调用方而遗漏新合同测试。
