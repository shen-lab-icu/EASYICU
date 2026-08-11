# Pi Copilot 一体化科研流程交接与 Web UAT

日期：2026-08-11

实现提交：`25a0fce`（`feat(web): unify Pi research workflow handoffs`）

任务范围：只闭合 Idea Mining → 项目记忆 → 研究配置 → 提取 → Research Agent 的对话交接和用户可见状态；不修改 Canonical9、shared prompt、paper rubric 或专业后端的科学逻辑。

## 结论

Pi/Copilot 现在是同一研究项目的会话编排与交接层，不是 Idea Mining、Data Extraction 或 Research Agent 的替代实现：

- Idea Mining 候选必须来自真实 `easyicu_mine_ideas` 工具回执，不能由模型自由编造后冒充已运行。
- 用户选择候选后，Copilot 通过新的 `easyicu_accept_idea_handoff` 把 canonical handoff SHA-256、候选身份和可行性结论绑定到 typed StudyContext；后续配置在同一项目内继续。
- 当前研究只认可与 StudyContext 数据源路径精确一致的 active export，不能把另一个项目的全局数据包当成本项目已提取。
- `hold` 或尚未获得 `recommend` 的候选会阻止 Plan 和 Analysis，并显示“需要按当前数据源重新核验可行性”；不会越过 Idea Mining owner 的科学结论。
- Pi 模型连接失败时，消息 job 以稳定的 `pi_model_*` 代码失败；Web 显示可行动的本地化提示，不再把空回答误记成 done，也不暴露内部地址或上游错误文本。
- 对话顶部和右侧进度现在消费同一份项目 workflow projection，均显示 7 个必需阶段；移除了旧右栏 `0/8` 与新流程 `3/7` 并存的割裂状态。

## 真实 Web UAT

项目：`Pi 一体化科研整链 UAT 20260811`（project id `draft_178743345eee`）。

1. 首次提示暴露模型直接撰写研究方案、没有调用 Idea Mining 的问题；加入 case-neutral tool-first 规则后重新开始新 Pi 会话。
2. 对话真实调用 Idea Mining，生成 `run_id=idea_20260811_003637_844633000_21080f9e`、`idea_id=idea_21080f9e6be2`。
3. 候选在现有 MOCK/demo 数据包上得到 `hold / reportable=false`；该状态被如实保留。
4. 用户选择候选后，真实调用 digest-bound 接受工具，绑定 handoff digest `e70a222b22291095188ff189fe9d4958571969f33e91f2de0dfe5104bb40cc8e`，StudyContext revision 从 1 升至 2。
5. 对话继续通过配置工具保存 MIMIC-IV、成人首次 ICU/Sepsis-3/SOFA-2、乳酸暴露、院内死亡、24 h 时间窗、协变量、Parquet 和非因果调整关联目标；revision 升至 3。
6. 明确请求提取时，本地模型代理的上游连接中断，因而没有发生 extraction tool call。修复后同一失败被记录为 message job `failed / pi_model_provider_unavailable`，Web 明确提示“本轮没有执行 EasyICU 操作”。
7. 页面桌面视口显示三栏布局、底部输入、统一 `3/7` 进度和可行性 hold；未见横向溢出或裁切。

这是一轮真实的 fail-closed Web UAT，不是完整正向整链验收。由于候选在 demo 数据上为 `hold`，且本地模型服务连接中断，本轮没有启动数据提取、preflight、Plan、正式分析或稿件生成。既有 Web E1 canary 已单独证明 Research Agent 后半链可完成 11/11 和结果解读，但不能与本轮前半链拼接后冒充“同一项目全链已通过”。

## 定点验证

按 E1 开发迭代策略，没有运行全套 CI。

- 7 个直接/邻接测试文件：`125 passed`。
- 修改域 Ruff：通过。
- 4 个修改的 Node/浏览器 JavaScript 文件语法检查：通过。
- `git diff --check`：通过。

测试覆盖了 handoff digest/CAS、非法 digest fail-closed、hold 阻止 Plan/Analysis、active export 项目归属、工具清单与 tool-first 提示、provider error job 终态、安全错误投影、StudyContext 前后端持久化及 Web owner 资产合同。

## 下一次整链验收的最小前置

1. 恢复本地模型代理的稳定连接。
2. 将当前项目绑定到真实、非 demo 的 MIMIC-IV EasyICU export。
3. 在该数据源上重新运行 Idea Mining，并获得 owner 给出的 `recommend` 或完成所要求的可行性审阅。
4. 在同一 Web 项目继续：明确提取确认 → extraction receipt/质量报告 → Research Agent preflight → Plan 人工确认 → full analysis → EvidenceStore → 结果解读 → 稿件草稿。

在这些前置满足前，不应通过前端文案、手工改状态或复用其他项目 export 绕过门禁，也不启动 Canonical9 正式 Provider batch。

## 第二阶段：真实 full6 数据基础到 Plan 人工门

实现提交：`14c19ac`（`fix(web): enforce reviewed research plans`）

同一 Web 项目后续绑定真实 MIMIC-IV full6 export，并使用已验证的本地 provider 进行了非 Canonical9 工程 UAT。本阶段不作为论文证据。

### 实现收口

- legacy/native export 的 concept 角色能够传递到数据目录，positive-only event 不再被普通数值特征误分类。
- 数据基础把 owner 给出的 event-status outcome 投影为 typed binary endpoint，把重复测量暴露投影为已物化的运行列；新运行实际绑定 `death` 与 `sep3_sofa2_max`。
- 修复 nullable Boolean event 在 positive-only 汇总中使用 float `fillna` 导致的 materialization 崩溃。
- structured Planner 重试发出有界生命周期事件：只投影轮次、通过/拒绝和错误类，不投影私有思维或原始响应。Web bridge 保留真实 stage，不再把所有事件写成 `research_pipeline`。
- Guided Web 运行启用 `require_human_plan_review`；即使自动科学检查没有产生 error finding，完整 Plan 也必须产生与 Plan、Evidence、run capsule 和 pipeline config 绑定的审核请求，审批前不运行任何分析步骤。
- 重选 Idea 导致 primary exposure 改变、而新 handoff 没有 comparator 时，Copilot 会清空旧暴露的 comparator，避免把“每 1 mmol/L”之类连续对比污染新的二元 phenotype。

### 真实运行证据

- job `91422ead38c6` 在 nullable Boolean event 汇总处失败，直接导出上述 materializer 修复。
- job `e8ef4ca668c1` 首次证明 data foundation、cohort/audit、typed endpoint/exposure 和 Planner 动态重试可运行；同时暴露出“无 error finding 时会自动越过 Plan 人工确认”的产品门禁缺口。该次运行已硬停，未当作验收成功。
- 修复后 job `c8aeb2ea9cd1` / run `run_20260811T091733_4f7d45` 完成数据基础和 Planner。Plan 第 1–3 版被合同拒绝，第 4 版通过，每次状态均进入 Web 事件流。终态为 `human_review_pending`，gate reason 为 `human_plan_review_required`，审核请求类型为 `scientific_stop`。
- 该运行的事件流没有 `Step ... started`，运行目录也没有 execution step 结果；这是“先 Plan 确认、后分析”的正向证据。

### 当前科学裁决

现有 Plan 保持未批准。其主暴露为二元 `sep3_sofa2_max`，但 StudyContext 仍携带旧乳酸暴露的“每升高 1 mmol/L” comparator，且主模型同时调整与暴露定义高度重叠的 SOFA 总分。这两点需要回到 Copilot 配置/重规划，不应直接批准后执行。

### 聚焦验证

按 E1/Web 开发策略，本轮没有运行全套 CI。

- Plan gate / Web workflow / typed config 邻接组合：`50 passed`。
- Copilot research workflow（含暴露切换时 comparator 清理）：`25 passed`。
- 本轮累计的 data catalog/foundation、structured retry、materializer 定点组合均通过；Ruff、Python 语法、Node 语法与 `git diff --check` 通过。

## 下一步

1. 在 Copilot 中对当前 Plan 选择“拒绝/要求重规划”，确认 `sep3_sofa2` 是否真是主暴露，并选择与其类型相符的 comparator。
2. 预先决定 SOFA 总分是否属于暴露定义/中介组成；若是，从主调整集移除，不在看到结果后临时处理。
3. 用新的 digest-bound Plan 再跑 Web E1；仅在计划本身科学合理且用户明确批准后，才执行分析、解读与稿件。
