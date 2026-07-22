# Research Agent Framework v2 离线发布记录

日期：2026-07-22
分支：`refactor/agent-control-plane`
状态：`offline_release_passed / online_experiments_paused`

## 交付

- 测量基线：`2c77d49`
- Host-owned Resource Scheduler：`4fe8fea`
- bounded context assembly：`c91b027`
- permissioned memory：`447049b`
- CapabilityRequest：`a34f7f7`
- LangGraph 默认 phase runtime：`404710b`
- digest-bound HITL 与旧 dispatch 退役：`01cd41e`
- 离线发布门：`00c298e`
- 旧 ExperienceBank 写回隔离：`dfcfcdd`

## 权威发布门

命令：

```bash
.venv/bin/python tools/research_agent_framework_release.py \
  --report task_logs/20260722_framework_v2_offline_release.json
```

结果：

- status：`passed`
- resource/context baseline：pass
- architecture baseline：pass
- module graph / zero-SCC：pass
- framework tests：`85 passed in 22.85s`
- provider calls：`0`
- patient-data reads：`0`

JSON 报告：`task_logs/20260722_framework_v2_offline_release.json`。

## 安全与权威边界

- Resource Scheduler 只能在 Host allowlist 内选择，允许零匹配，不用 LLM 扩权。
- 自动经验只进入 `run_lessons/quarantine`；canonical Planner 不读取旧 RunMemory/ExperienceBank 自由经验。
- reviewed/promoted memory 必须绑定 profile、内容 SHA 与审核/晋升凭证。
- CapabilityRequest 只生成可审核申请；分析容器无网络，Coder 不得临时安装包。
- LangGraph 是默认 phase 编排面，但 EvidenceStore、receipt、capsule 与 EasyICU checkpoint 仍是科学和持久运行权威。
- 本轮没有读取 `/Volumes/外置硬盘/easyicu_data/full6_20260717`，没有重抽六库，也没有启动 E2/E3/H2/SOFA-2 在线实验。

## 诚实边界

本记录只证明 Framework v2 离线发布候选成立。它不证明 ProtocolCard 已经临床/方法学签署，不证明 Canonical9 在线 A/B/C 已通过，不证明产品 UI 的所有人工审核交互已完成，也不授予任何论文结果 authority。下一步是独立审阅本发布候选；用户明确解冻后，才使用已有 `full6_20260717` 提取运行 fresh A 题。
