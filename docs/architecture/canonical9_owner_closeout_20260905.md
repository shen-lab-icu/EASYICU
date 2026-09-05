# Canonical9 开发分支的 owner 收尾审阅

复核起点：主线 `439f5c9138aded968f5e575e61b968d81d80513a`，待收尾快照
`dd57ff5e5b1997ae113e673449b2cbd405a61e4f`。该快照保留了整理前 87 个路径的
开发工作；本页记录其进入最终整合候选前的修复与 module 准入理由。

## 科学选择与执行合同

- `planning/ordinal_multi_outcome.py` 只解析带类型的变量与值域，说明一个
  ordered-trend action 是否适用。`cohort.outcome_columns` 可以来自数据目录推断，
  因此不能据此要求新增次要结局，也不能强制把主模型改成 categorical coding。
  Planner 明确选择该 action 后，既有 outline/compiler 仍核对唯一 owner、三个
  变量、primary lineage、输出和事件编码。额外可用结局列不改变主模型合同。
- `CohortDescriptor.requested_outcome_columns` 单独保存显式分析端点。Builder、
  outbound、科学审查与旧运行 adoption 共用该声明；猜测到的 LOS 列仍可见，
  不构成额外分析要求。旧运行恢复不能通过省略输入清单抹掉封存声明。
  缺少新字段的历史 context 保持 None，不把可用列推断成研究者的选择。
- `execution/runners/landmark_categorical_association_executor.py` 是已签署
  categorical-landmark authority 的 adapter；它消费完整计划和封存的 runtime
  projection，并复用调整关联执行器。没有该 authority 就不认领步骤。事件时间或
  观察时长出现正负无穷时拒绝执行，不能把它们解释为已证明的随访。两个 renderer
  按 prologue、审计 receipt、执行块组装，非空 flag-only scope 的脚本可编译。
- 计划要求的临床协变量理由继续强制存在。旧测试样本补齐该字段，使负向回归真正
  到达“结局泄漏／动态变量缺少时间权限／标识列误入模型”等目标校验。
- 文献筛选样本明确写出暴露与结局的研究关系，继续证明即使 P/E/O 匹配，综述和
  随机试验也不能充当当前观察性研究的直接比较文献。没有放宽筛选规则。

## 两个已有 module 的深度与 locality

`PlannedModelRequirement`、允许的模型族和精确模型项校验统一由现有
`contracts/model_terms.py` 拥有。`schema.py` 保留同一 Python 类型的兼容导出，
没有平行 schema；迁移前后的完整 `AnalysisPlan` JSON schema 已逐对象比较一致。

Progressive Planner 的 outline 坐标规范化、foundation 解析和 model 解析归入现有
`agents/progressive_payload.py`，与 step 解析共用 transport seam。Planner 消费
解析后的 typed value，继续拥有编排；旧内部导入保持 identity alias。

| ratchet | 原主线预算 | 收尾后 |
| --- | ---: | ---: |
| `schema.py` LOC | 3,277 | 3,036 |
| `agents/progressive_planner.py` LOC | 4,160 | 4,119 |
| research-agent module 数 | 641 | 643 |
| 顶层 module 数 | 34 | 34 |
| 循环依赖涉及 module 数 | 0 | 0 |

新增两个 module 分别是已签署执行 adapter 和跨 Planner/compiler 复用的适用性
owner，具有独立输入合同与负向回归；不把它们塞进原有大型 renderer 来掩盖数量。
Run receipt 的文件摘要复用既有 canonical_json owner，移除重复 helper，保持摘要字节不变。
module graph 显式记录这次准入及共享依赖并保留历史，两个大文件的 LOC 上限下调。结构工具和
零循环规则保持原样。

## 传输与 Web 投影

一次性 skeleton schema 移除了已不被引用的 `CandidateLiteratureDesignDecision`；
该对象仍由 outline 的设计选择合同管理，避免重复传输。12,000 字节预算保持不变。

完整 Planner 的协变量 rationale/temporal maps 使用闭合 key/value 行传输；
`plan_payload` 统一解码完整计划与 runtime suffix，拒绝重复键和畸形行，继续
通过原科学合同。复用等价 scalar schema 和相同 consumption 分支后，十来源请求
为 **29,616 bytes**，仍低于原 **30,000 bytes**。展开后的 schema 逐对象等价，
没有删科学字段或 figure presentation。Suffix owner 从 273 缩到 270 LOC。

计划确认卡片的一处条件表达式吞掉了非空 decision buttons。修复后“确认整份计划”
与“提出修改”均可显示；系统负责的方法问题仍隐藏，普通执行按钮不能替代该确认。
新增测试直接执行 owner JavaScript 并检查实际输出按钮。

## 验证范围与权限

- combined：改动测试及模型合同、schema、值域相邻检查，**1,346 passed**。
- executor adjacent：两阶段 registry、selection report 和 landmark，**30 passed**。
- 五项架构门通过，包含 **142 项**结构/预算检查及 **7 个** import contracts。
- 修复前的 11 项失败和原始快照保留在工作区 branch-reconciliation 证据中。
- 追加端点/传输及 v1/v2 context 相邻检查 **160 passed**；旧图件合同测试 **22 passed**。
- Run receipt 与 repository hygiene **25 passed**；复用共享 hash 后的 module graph 补验通过。
- 浏览器合成计划卡片在 1440×1000 和 1024×768 下无横向溢出，确认/修改按钮可见，
  console 无 error；详情展开后也无按钮裁切。
- 本页的工程检查不授予 Provider、正式实验或论文权限；最终候选仍需自己的
  full exact-head CI 及版本化启动合同。浏览器验收使用合成卡片，不生成研究结果。

工作区复现记录：`task_logs/20260905_canonical9_closeout.md`，机器日志位于
`task_logs/artifacts/20260905_canonical9_closeout/`。二者位于多项目工作区根目录。
