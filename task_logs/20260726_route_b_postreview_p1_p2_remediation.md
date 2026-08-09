# Route B 外部复审 P1/P2 修复与验证

- 日期：2026-07-26
- 模块：agent
- 阶段：Route B post-review remediation
- Task ID：`AGENT-ROUTE-B-POSTREVIEW-P1-P2`
- 分支：`fix/external-review-20260724-p0-p1`
- 复审基线：`6fd65bd3133359de555d3e5ee026ec7a64f356fa`
- 首轮代码修复终点：`b0ff93606d6ef8a113d7a29bbaf756fb26b3868b`
- 低优先级闭环终点：`3f799d8e32804183b631ed758f54dcd1542e3f18`

## 状态核实

`git ls-remote` 与 remote-tracking ref 均确认此前 18 个 Route B 提交已经到达
origin，远端终点为 `6fd65bd`。旧完成记录中的“未 push”只反映记录生成时的
状态，不再代表当前远端状态。

2026-07-26 07:40 EDT 再次核验本地 tracking ref 与 `git ls-remote`：首轮 6
个代码提交及证据提交均已推送，远端终点为
`d5d187d812ac223c2b922f63ca720f04701b1924`。此前“本轮新增修复仍仅在本地”
已经过时。外审随后提出的两个低优先级闭环与 MCP 权威投影分别提交为
`fdc46be`、`3f799d8`；截至本记录更新时，这两个新提交仅在本地，本证据更新
另形成一个 docs-only 提交，因此分支相对 origin ahead 3。本会话未执行 push。

## 分批修复

| commit | 修复 |
|---|---|
| `8fc9740` | 人工审核拒绝成为显式 `rejected` 终态；拒绝记录先落证据，再清理 live handoff；不能随后改为批准。决定记录固定按服务端 request 顺序生成，审核摘要统一使用共享 canonical SHA-256。 |
| `c8a8f6c` | 新增版本化 `StepRecoverySignature`，恢复门绑定 step、role、method、inputs、outputs、ICU rules、model requirements、input-consumption contracts、Table 1 与 trajectory stability；自由文本 intent 不进入签名；新格式异常 fail closed，旧 finding 保留兼容路径。 |
| `6275fc0` | MCP 同步 dispatcher 使用独立 bounded supervisor；请求超时或取消后，后台线程直到真实返回前继续占用并发槽，防止连续 timeout 累积无限后台任务。timeout 结果显式返回 `dispatch_started` 与 `execution_may_continue`。 |
| `8a51df7` | 将内容层结果改名暴露为 `publication_artifacts_ready`，最终 `paper_authorized` 同时要求该内容门与 `execution_identity.paper_eligible`；未提供执行身份的直接调用默认失败关闭。 |
| `03cb725` | 加强取消回归：容量为 1 时，取消客户端等待不会提前释放后台 dispatcher 的槽位。 |
| `b0ff936` | 新增 authority→reporting、orchestration→webserver、MCP transport→application 三条导入边界；把 `RunResult` 下沉至依赖中性的 contract owner，并移除 authority 对 mock/reporting 的传递依赖。CI 注释明确 Deptry 的 `DEP002` 例外。 |
| `fdc46be` | 拒绝审核除 `human_review_decisions.json` 外，新增 evidence-registered `run_status.json` 终态（`human_review_rejected` / `operator_rejected` / rejected IDs / fail-closed gates）；批准后 execute/write/finalise 失败时清除失效 pending 引用，输入或 digest 错误仍保持 paused 供修正。缓存明确拒绝该终态。 |
| `3f799d8` | MCP 外部 manifest 投影只公开 `publication_artifacts_ready`、`execution_paper_eligible`、`paper_authorized` 三轴；缺字段失败关闭，并不公开可被误当授权的 `publication_ready`。 |

## 验证

最终在固定代码终点 `b0ff936` 运行组合邻接矩阵：

```text
297 passed, 19 warnings in 12.91s
pytest-randomly seed = 2426599873
```

覆盖 workflow/review resume、截断恢复、MCP SDK/timeout/cancellation、readiness
与 execution identity、finalization boundary、runtime capability context、runner、
reporting boundary 和 module graph。

静态与架构门：

```text
Ruff: all checks passed (src/easyicu, tests, tools)
Deptry: success; 489 files scanned
Import Linter: 7 kept, 0 broken
research-agent module graph baseline diff: exit 0
git diff --check: passed
```

此前分批聚焦测试也分别通过：

- workflow：16 passed
- review-resume/runtime 邻接：63 passed
- step completion/recovery：25 passed
- completion/plan/module 邻接：25 passed
- MCP transport：17 passed
- readiness/execution identity/finalization：26 passed
- runner + module graph：52 passed

外审确认 `d5d187d` 后的低优先级闭环采用快速分层测试，不重复约 38 分钟全套：

```text
review/workflow/MCP projection 邻接：121 passed, 1 skipped
completion/MCP transport/rejected-cache 邻接：29 passed
合计不重复：150 passed, 1 skipped
Ruff（5 个改动文件）：passed
Deptry：489 files, success
Import Linter：7 kept, 0 broken
research-agent module graph baseline diff：exit 0
git diff --check：passed
```

真实 workflow 回归同时覆盖正反两面：拒绝会先持久化决定与 run-level 终态且不进入
execute；批准后 execute 抛错使 workflow=`failed` 并清除 pipeline pending；digest
错误保持 workflow=`paused` 且 pending 可再次提交。MCP 契约用
`publication_artifacts_ready=true`、`execution_paper_eligible=false`、
`paper_authorized=false` 的反例锁定客户端不能把内容就绪误判为论文权威。

## 仍然诚实保留的边界

- Route B 仍只支持 `same_process` resume；没有声称支持服务重启后的恢复。
- Python 同步线程不能被安全强杀。MCP request timeout 只停止等待；若
  `execution_may_continue=true`，任务可能继续到自身返回，但不会突破并发上限。
  长 pipeline 的 job ID、独立 supervisor 与进程级终止仍属后续产品化工作。
- Deptry 全局忽略 `DEP002`，所以本轮通过证明缺失、传递和依赖位置检查通过，
  不证明不存在多余声明依赖。
- 旧完整功能套件报告过 1306 条 warning；本轮没有运行约 38 分钟的全套，
  因而没有伪造一个不跨 Python/环境稳定的 warning 数字基线。后续应在正式 CI
  上按 warning fingerprint 建立“不净增”门。
- 拒绝终态现在可由另一个进程从磁盘辨识，但这不是 durable resume：服务重启后
  仍不能恢复 live plan/evidence handoff；只能确定该 run 已被操作员拒绝。
- 未刷新 Figure 2 scorer、resource 或 architecture 的冻结权威摘要；未启动
  Canonical9 paper-facing batch。
- 本轮没有读取患者数据，也没有访问 `/Volumes/外置硬盘/databases`。此前 Route B
  完成烟测使用的是 `/Volumes/外置硬盘/databases/mimiciv`，本记录不重复该 I/O。
