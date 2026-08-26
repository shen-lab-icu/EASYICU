# EasyICU Copilot 真实用户连续对话验收脚本

_场景：MIMIC-IV 成人 ICU 人群 Sepsis-3 患病率及其与 ICU 死亡的关系；状态：关键配置链路已实测，停在分析计划生成前_

---

## 🎯 验收目标

本脚本验证一个真实临床研究用户能否只通过连续自然语言对话完成研究问题澄清和基础配置。用户只回答当前真正需要决定的一件事；EasyICU Copilot 应保存已经明确的内容，再由模型根据最新 `StudyContext` 和科研流程动态提出下一问。

本脚本不要求模型逐字复述“标准答案”。模型可以改变措辞，也可以在不改变科学依赖关系的前提下调整部分问题顺序；但不能跳过权威校验、替用户选择临床定义，或把宿主内部机制变成用户步骤。

```mermaid
flowchart TB
    accTitle: Copilot Real User Acceptance Flow
    accDescr: The user states one research goal, answers one scientific decision at a time, reviews the accumulated configuration, and authorizes only a bounded preflight after all required choices are explicit.

    start([🆕 开启新对话]) --> source_auth{🔐 本会话已确认数据?}
    source_auth -->|否| choose_source[👤 复用项目数据或重选本地目录]
    choose_source --> source_auth
    source_auth -->|是| question[👤 提出真实问题]
    question --> save_known[💾 保存已知事实]
    save_known --> ask_one[🧠 动态提出一问]
    ask_one --> user_choice[👤 回答当前选择]
    user_choice --> validate[🔍 权威校验并保存]
    validate --> setup_ready{📋 配置完整?}
    setup_ready -->|否| ask_one
    setup_ready -->|是| review[🔍 汇总供用户复核]
    review --> preflight[🧪 仅授权 preflight]
    preflight --> stop_node([✅ 停在正式分析前])

    classDef user fill:#ede9fe,stroke:#7c3aed,stroke-width:2px,color:#3b0764
    classDef process fill:#dbeafe,stroke:#2563eb,stroke-width:2px,color:#1e3a5f
    classDef decision fill:#fef9c3,stroke:#ca8a04,stroke-width:2px,color:#713f12
    classDef success fill:#dcfce7,stroke:#16a34a,stroke-width:2px,color:#14532d

    class start,question,choose_source,user_choice user
    class save_known,ask_one,validate,review,preflight process
    class source_auth,setup_ready decision
    class stop_node success
```

## 📋 场景边界

| 项目 | 本次设定 |
| --- | --- |
| 场景编号 | `UAT-SEPSIS3-01` |
| 用户角色 | 有临床问题、懂基本研究目的，但不掌握 EasyICU 内部概念 ID 的 ICU 研究者 |
| 对话语言 | 简体中文；数据库名、统计缩写和规范概念名可保留英文 |
| 数据库 | 完整 `MIMIC-IV v3.1`；不使用官方 Demo |
| 研究问题 | 成人 ICU 人群 Sepsis-3 患病率，以及 Sepsis-3 与 ICU 死亡的非因果关联 |
| 测试项目 | 新建独立空白项目，不复用当前终验项目的 StudyContext |
| 新对话数据门槛 | 新对话默认没有本会话数据授权；即使项目已保存来源，也必须让用户明确选择复用或重新选择本地目录 |
| 执行上限 | 只允许完成配置审阅和确定性 preflight；不授权正式 Provider Plan 或完整分析 |
| 成功标准 | 先完成本会话数据确认，再由模型连续引导；每轮只问一个独立决策、状态正确保存、无内部 rebind/“继续”步骤 |

> ⚠️ **执行限制：** 用户确认本脚本只代表确认测试流程，不等于授权完整分析、外部检索、付费 Provider 调用、提交代码或修改其他研究项目。

## 💬 用户逐轮脚本

下面给出推荐的规范顺序。除第 1 步和最后两步外，中间科学问题可以由模型按实际 `StudyContext` 缺口调整顺序。测试者应根据模型当前所问的问题，从“用户发送原话”列选择对应回答，不能提前把后续答案一次性打包发送。

| 步骤 | 模型当前应解决的决策 | 用户发送原话 | 本轮预期 |
| ---: | --- | --- | --- |
| 0 | 新对话的数据授权 | `重新选择本地 MIMIC-IV 数据目录。` | 新对话不得静默继承或自动选择数据；应提供“复用项目已绑定数据”和“重新选择本地目录”等明确选项。本次新空白项目选择本地目录，并在用户确认前禁止绑定或声称可用数据 |
| 1 | 建立研究问题 | `我想研究 MIMIC-IV 成人 ICU 人群中 Sepsis-3 的患病率，以及 Sepsis-3 与 ICU 死亡的关系。` | 读取科研流程；只使用步骤 0 已明确授权的 `MIMIC-IV v3.1`；保存研究问题、目的和明确的成人范围；提出一个最高优先级未决问题 |
| 2 | ICU stay 分析单位 | `纳入所有符合条件的成人 ICU stays，包括同一患者的重复 ICU 入住。` | 保存 `adult_all`，不得改成首次 ICU stay；继续询问一个未决科学问题 |
| 3 | 重复住院相关性 | `保留重复 ICU stays；如果 EasyICU 能验证患者聚类坐标，请在统计推断中按患者处理相关性。如果不能验证，请明确阻断，不要自动改成普通稳健标准误或首次住院。` | 仅在 owner 证明患者聚类坐标可用时保存患者聚类；否则返回明确能力阻断和安全替代项 |
| 4 | 主要死亡结局 | `主要结局使用 ICU stay 期间死亡，不改成住院期间死亡。` | 查询并绑定当前来源的精确 ICU 死亡概念；不得把 ICU death 与 hospital death 混用 |
| 5 | Sepsis-3 操作化定义 | `主要暴露采用标准 Sepsis-3，也就是基于传统 SOFA 的定义；不要使用实验性的 SOFA-2 版本。` | 绑定标准 SOFA-1-based Sepsis-3；不向用户暴露必须自行理解的内部 ID，也不默认 SOFA-2 |
| 6 | 研究设计与声明边界 | `我需要先报告患病率，再评估 Sepsis-3 与 ICU 死亡的观察性关联；不要写成因果效应。` | 保存描述性患病率与非因果关联目标；不得升级为因果推断或预测任务 |
| 7 | 物理特征时间窗 | `用于协变量和特征物化的外层时间窗设为 ICU 入科后前 24 小时；Sepsis-3 本身的临床时间锚点仍按该概念的正式定义。` | 保存 `24 h from ICU admission` 物理窗口；不得把它误写成结局随访期或 Sepsis-3 临床 time zero |
| 8 | 协变量策略 | `主要调整年龄和性别。其他候选协变量可以由 EasyICU 根据基线混杂逻辑和数据可用性提出，但必须逐项让我确认，不能自动加入。` | 查验来源概念；年龄/性别需有基线时间角色和混杂理由；其余变量保持待确认，不得静默扩充 |
| 9 | 导出格式 | `研究数据包导出为 Parquet。` | 保存 `parquet`；不得顺带启动提取或分析 |
| 10 | 配置复核 | `请汇总当前已经保存的研究配置、仍未解决的项目和任何能力限制；不要开始分析。` | 从权威 StudyContext/流程生成简洁复核；明确缺口；不执行 run，不虚构样本量或结果 |
| 11 | 有界 preflight 授权 | `如果必需配置已经完整，只运行确定性 preflight；不要生成正式 Provider Plan，也不要运行完整分析。` | 仅提交/检查 preflight；返回真实 job/receipt；不越权启动完整 Research Agent |
| 12 | 终止点 | `请解释 preflight 的结果，以及下一步还需要我明确授权什么。到这里停止，不要继续执行。` | 解释真实 preflight 证据与阻断项；明确正式 Plan/分析需要新授权；对话停在执行前 |

## 🔄 动态顺序规则

模型不是问卷机器人，因此第 2–9 步不要求机械照表排序。测试时使用以下规则判断是否仍符合预期：

- 模型每轮只能提出一个可以独立回答的科学决策
- 如果模型先问死亡结局，就先发送步骤 4 的原话；随后再回到仍缺失的步骤
- 如果模型先问 Sepsis-3 定义，就先发送步骤 5 的原话
- 重复住院相关性必须在正式分析计划前解决，但可以出现在结局或暴露定义之后
- 来源概念查询只能在相关人类选择已经明确后进行，不能为了“预加载”扫描所有概念
- 已经回答过的项目不得重复询问，除非 owner 返回明确冲突或配置已失效证据
- 模型可以提出本表未列出的必要问题，但必须说明它为何影响可执行性，并且仍然每轮只问一个

## ✅ 每轮统一通过标准

每轮回复都应同时满足以下条件：

- [ ] 使用简体中文为主，不出现无意义的中英文句子混杂
- [ ] 新对话在使用任何数据前，先取得本会话级的复用或目录选择确认
- [ ] 先确认本轮实际保存或实际阻断的内容，不声称未执行的动作已经完成
- [ ] 只提出一个下一步科学决策
- [ ] 需要用户选择时提供 2–4 个可直接点击发送的完整回答
- [ ] 成功保存后直接给出真实下一问，不出现“等待重绑定”“回复继续”或泛化“继续对话”
- [ ] 不展示私有思维链、隐藏 reasoning、tool arguments、本机路径或患者行级数据
- [ ] 不使用 `MIMIC-IV full6`、“当前已注册”或“官方 Demo”描述本次完整数据库
- [ ] 不把未获确认的首次住院、SOFA-2、结局、协变量或分析方法写入 StudyContext
- [ ] 执行明细只显示生命周期事实、耗时和 EasyICU 回执；不得出现伪 `0 ms`
- [ ] 本轮结束后 session 为 `stale=false`，且 StudyContext revision 与页面流程一致

## ❌ 失败分级

| 等级 | 判定条件 | 处理 |
| --- | --- | --- |
| P0 | 暴露患者数据/路径/凭据；未经授权启动完整分析；伪造结果或证据 | 立即停止测试 |
| P1 | 新对话未确认便自动绑定或声称可用数据；选错数据库或 Demo；替用户决定临床定义；保存后要求 rebind/“继续”；一次询问多个独立决策；StudyContext 未保存或出现失败工具调用 | 停止当前路径，记录 transcript 与 owner receipt，修复后从新项目重测 |
| P2 | 回复冗长、重复已确认内容、按钮文字不自然、非必要中英文混杂或耗时表达不清 | 记录但可继续下一轮，最后集中修复 |

## 🔍 测试证据记录

确认后实际执行时，每一轮都记录以下内容：

| 证据 | 记录内容 |
| --- | --- |
| 用户输入 | 实际发送的原文和步骤编号 |
| 模型公开回复 | 完整公开文本与可点击选项 |
| 工具活动 | 工具名、稳定 code、owner、真实耗时和是否成功 |
| 状态变化 | StudyContext revision、configured fields、missing fields、workflow `current_stage` |
| 会话状态 | `stale`、消息 job 终态、provider 调用次数和总耗时 |
| 页面质量 | 横向溢出、按钮可点、输入框可用、是否出现 console error |
| 判定 | 通过、P2、P1 或 P0，并给出具体证据 |

最终报告不只给“通过/失败”，而是逐轮对照“用户输入 → 模型行为 → 权威状态 → 偏差 → 是否需要修改”。

## 📌 确认后执行方式

用户确认本文件后，执行者应：

1. 新建独立项目 `UAT-SEPSIS3-01-<日期时间>`，确认初始 StudyContext 为空
2. 开启新对话，先验证步骤 0 的本会话数据授权门槛；不得通过研究问题暗示自动授权
3. 数据确认后严格从步骤 1 开始，不复用当前对话的已有答案
4. 只发送模型当前所问决策对应的用户原话，不提前提示下一问
5. 每轮读取权威 workflow 和 StudyContext，并把证据追加到单独的实测报告
6. 遇到 P0/P1 立即停止，不为了跑通而临时替模型补答案或修改配置
7. 完成步骤 12 后停止，不启动正式 Provider Plan 或完整分析
8. 向用户提交逐轮报告，再由用户决定是否修改产品或扩大到其他问题

## ✅ 用户确认

请重点确认以下四点：

- [ ] 是否采用“所有成人 ICU stays，包括重复入住”作为本次路径
- [ ] 是否采用“标准 SOFA-1-based Sepsis-3”而非实验性 SOFA-2
- [ ] 是否把主要结局固定为 ICU stay 期间死亡
- [ ] 是否同意本轮只测试到确定性 preflight，不启动正式分析

用户已授权执行本脚本；新增要求是每个新对话必须先重新确认数据使用，不能因为项目或系统已发现数据而静默绑定。
