# EasyICU Copilot 真实用户连续对话实测报告

_场景：`UAT-SEPSIS3-01`；日期：2026-08-25；当前状态：新会话数据授权与用户关键决策负担已修复，真实模型复测停在分析计划生成前_

## 测试边界

- 使用全新空白研究项目和新对话。
- 首先验证“新对话默认没有本会话数据授权”的门槛。
- 项目级 StudyContext 可保存数据来源以支持可复现性，但不能等价于新对话已获准使用该来源。
- 遇到 P0/P1 立即停止；本次不授权正式 Provider Plan 或完整分析。

## 逐轮证据

### 步骤 0：新对话数据授权

| 证据 | 实测结果 |
| --- | --- |
| 隔离项目 | `UAT-SEPSIS3-BINDING-20260825-0725`；project id `draft_9a6d929f6e66` |
| 初始 StudyContext | revision 1；`data_source={}`；workflow 缺失字段包含 `question`、`data_source` 等 8 项；完成阶段 `0/7` |
| 新会话 | session `pi_e7c4054d1f596112c596`；message count 0；transcript 为空；`stale=false` |
| 页面行为 | 点击“开始研究对话”后直接出现空白输入框和“发送”按钮；没有“复用项目已绑定数据”或“重新选择本地目录”选择，也没有目录选择器 |
| 会话授权状态 | 会话 API 仅有 StudyContext revision binding；没有独立的数据来源确认或本会话授权字段 |
| 空白项目污染 | `guided_draft.json` 和旧 Guided 会话上下文预填了 SICDB 示例来源及预测问题；虽然 Pi StudyContext 的数据源仍为空，但“空白项目”并非真正空白，左侧还显示“本地数据” |
| 模型与工具调用 | 0 条消息；未发送步骤 1；未触发 Provider、EasyICU 数据工具、目录扫描或正式分析 |
| 判定 | **P1：新对话在首轮前没有数据确认门槛；空白项目还带有示例数据元数据。** 按脚本立即停止 |

### 未执行步骤

步骤 1–12 均未执行。原因不是模型卡住，而是产品在模型收到第一条消息前已经缺少必要的数据授权交互；继续发送问题会让现有自动选择策略掩盖该缺口。

## 当前结论

本次失败发生在 UI/会话初始化层，而不是大模型回答质量层。当前系统存在三种不同状态，但只显式建模了前两种：

1. 项目绑定到哪个 `StudyContext`；
2. `StudyContext` 保存了哪个数据源；
3. **当前新对话是否得到用户授权使用或复用该数据源。**

第 3 种状态目前缺失。建议的修复契约是：

- 创建真正空白项目时，不写入 SICDB、示例问题或虚假的“本地数据”标签；
- 每次新会话初始化 `data_source_confirmed_for_session=false`；
- 首次可能涉及数据的研究消息前，Copilot 提供两个明确动作：“继续使用当前项目已绑定的数据”和“重新选择本地数据目录”；空白项目只提供后者及可选的受控 Demo；
- 目录通过宿主原生选择器完成，路径不进入模型上下文；
- 只有宿主确认成功后，才把本会话授权与项目级 StudyContext 的来源 revision 绑定；项目中的研究配置继续保留，不因新会话而清空；
- 如果项目来源 revision 后续变化，本会话授权立即失效并重新确认。

在该门槛修复前，不应继续本场景的模型引导验收。

## 修复后复测（同日）

### 修复契约

- 空白项目不再预填示例问题或来源；新会话创建为独立的 `pending` 数据授权状态。
- 新会话在用户确认前只显示“继续使用项目数据 / 选择本地数据目录”，隐藏 composer，并禁止模型和科研工具访问数据。
- 会话授权绑定项目来源 identity 与 StudyContext revision；来源变化后授权失效。
- 项目列表从权威 StudyContext 投影是否已有本地数据，不再使用创建时的 `unbound` 元数据误报。
- 工作流新增显式 `confirmations.clinical_definition_<phenotype>` 与 `confirmations.feature_time_window` 门槛；默认值或模型顺手写入不能替代用户确认。

### 真实浏览器证据

| 轮次 | 实测结果 | 判定 |
| --- | --- | --- |
| 全新空白项目选择目录 | 初始卡片显示“未选择数据”；目录在宿主原生界面选择，路径未进入模型；保存后会话可用来源显示为 `MIMIC-IV v3.1` | 通过 |
| 新会话复用门槛 | 点击“新会话 → 开始研究对话”后只出现数据来源确认；旧 transcript 未回流，composer 不存在，模型调用为 0 | 通过 |
| 会话选择竞态 | 新会话不再被异步历史恢复覆盖；后端产生独立 pending session，旧会话没有收到新消息 | 通过 |
| 临床表型优先级 | workflow 明确报告缺失 `confirmations.clinical_definition_sepsis`；模型先询问推荐锁定 Sepsis-3 定义，没有先问外层 24/48/72 h 窗口 | 通过 |
| 表型保存 | 用户选择推荐锁定定义后，22 秒内得到“已保存研究配置”回执；随后单独询问 ICU 内死亡/住院死亡 | 通过 |
| ICU 内死亡保存 | 旧项目因缺少可复用 `source_id`，本轮需重新定位来源并查询概念，耗时 1 分 13 秒；真实 update receipt 成功，但模型同时写入默认窗口锚点，触发新的 P1 修复 | 修复后通过门禁 |
| 时间窗门禁 | 修复后 workflow 从 3/7 退回 2/7，并要求独立确认 `feature_time_window`；模型提供 720/24/48/其他小时数选择 | 通过 |
| 24 h 保存 | 点击 24 小时后仅调用 workflow 与 StudyContext update，15 秒完成；回执保存 `24 h from ICU admission` 及显式确认，流程才恢复 3/7 | 通过 |
| 项目列表 | 当前项目从“研究已配置 · 未选择数据”修正为“研究已配置 · 本地数据”；真正空白项目仍显示“未选择数据” | 通过 |
| 下一步与 Markdown | 每轮有下一步；需选择时显示可点击按钮；提示末尾不再残留 `**` | 通过 |

### 当前停止点与证据边界

- 当前 StudyContext 已保存成人全 ICU stays、重复入住策略、锁定的 Sepsis-3 表型、ICU 内死亡、所需执行模块和 `24 h from ICU admission` 外层特征窗口。
- 页面为 3/7，明确等待用户选择是否生成 Research Agent 分析计划。
- 本次没有点击“生成计划”，没有运行 preflight、Provider Plan 或完整分析，也没有产生患病率、关联估计或可投稿结论。
- 最终直接相关合同矩阵 297 项通过；Ruff、Node 语法、`git diff --check` 和进度 lint 通过。浏览器最终页无控制台 warning/error，桌面视口未见横向溢出或裁切。

## 用户关键决策负担修复（同日后续复测）

前一版把 `confirmations.clinical_definition_sepsis` 作为必经用户门槛，导致模型把抗生素/采样窗口、SOFA 计算窗口和阈值等 EasyICU 实现细节抛给普通用户。该设计已被后续修复取代：

- 只有未锁定、临床上不等价且确实会改变研究问题的表型变体才需要用户选择；唯一 owner-locked 标准定义由 EasyICU 应用并在高级配置中供复核。
- 普通照护阶段死亡采用该阶段全程语义；例如没有固定时限限定的 ICU 死亡按同一次 ICU stay 内死亡处理，不再凭空生成 24/48/72 小时死亡选项。
- 用户回答上一轮关键选择后走快速保存路径：至多读取一次 workflow、必要时读取一次 context、执行一次最小 StudyContext 更新，然后立即给出下一项关键科学选择；本轮不得顺带查询数据源、概念目录或补 execution concepts。
- 下一步选项必须互斥且不重复；推荐项在原选项上标注，不另造重复按钮，按钮标签使用纯文本。

### 真实浏览器复测证据

| 检查 | 真实结果 | 判定 |
| --- | --- | --- |
| 新会话数据门 | 每个新会话先出现“继续使用项目数据 / 选择本地数据目录”，确认前 composer 不可用 | 通过 |
| owner-locked Sepsis-3 | 首轮没有再询问抗生素、采样、SOFA 窗口、阈值或组件规则 | 通过 |
| 普通 ICU 死亡语义 | 最终 fresh 模型回合直接使用 ICU stay 全程死亡语义，没有再列 24/48/72 小时死亡 | 通过 |
| 第一项用户决策 | 模型转而询问外层特征观察窗口，解释短窗与长窗的研究权衡，并推荐 24 小时；明确该窗口不改变 Sepsis-3 定义或 ICU 死亡语义 | 通过 |
| 简单选择快速保存 | 中间复测从原先 1 分 31 秒、6 工具/7 模型回合和一次保存错误，收敛为 21 秒、一次 workflow 读取、一次 StudyContext 更新和 3 个模型回合；无来源/概念目录调用 | 通过 |
| 合同矩阵 | 相关 5 文件矩阵 `298 passed`；Ruff、Node 语法和 `git diff --check` 通过 | 通过 |

最终模型回合发生在选项纯文本/去重提示加入前，因此截图仍保留一个重复的“推荐 24 小时”按钮和 Markdown 星号；该显示瑕疵已由 prompt 合同修复并有静态回归覆盖，但未再消耗一个 Provider 回合复拍。它不影响 owner-locked 表型、普通死亡语义或简单决策快速保存的已验证结论。

## 严格脚本逐句仿真（R4–R16）

本轮改用 `docs/qa/copilot_real_user_sepsis3_conversation_zh_cn.md` 的 0–12 句作为唯一输入来源。每次出现 P1 都立即停止、修复并新建空白项目；不在受污染的对话中续跑。

| 轮次 | 停止点 / 真实发现 | 修复或判定 |
| --- | --- | --- |
| R4–R7 | 空白项目水合竞态、跨项目提取回执残留、分析单位与聚类被捆绑、默认分析目标跳过用户确认、标准 Sepsis-3 概念因未选技术模块而阻断 | 新项目强制水合并清空项目级回执；StudyContext owner 分离关键决定；默认研究文字不再充当确认；精确概念获用户授权后由 EasyICU 自动加入目录证明的 owning module |
| R8–R10 | 模型在一次更新中夹带下一字段，导致已确认结局与未确认暴露一起回滚；主要暴露未列为关联研究必需项；后续分析目标覆盖已确认患者聚类 | 保存本轮已确认字段并丢弃越界字段；关联意图显式要求 `primary_exposure`；未获当前用户授权时，后续设计更新不得改变已有 `variance_estimator` / `cluster_unit` |
| R11–R15 | 内部 module/concept 缺口产生“继续对话”；年龄/性别被再次要求确认是否为基线；界面默认 Parquet 被误作用户确认；协变量与导出格式顺序反复偏离 | 用户项排在技术就绪项之前；显式年龄/性别调整由 owner 保存为唯一 `baseline_static` roster；导出格式增加独立确认回执；关联研究的调整集决定排在导出格式之前；模型必须服从 `missing_setup_fields` owner 顺序且用户项未完时禁止 generic continue |
| R16 步骤 0–8 | 全新项目初始 `0/7` 且未选择数据；重新绑定 MIMIC-IV 后，依次完成 all stays、患者聚类、ICU stay 死亡、传统 SOFA Sepsis-3、非因果观察性关联、24 h 外层窗口、年龄与性别；第 8 句后直接询问导出格式 | **步骤 0–8 通过**；没有重复 ICU 死亡、没有覆盖患者聚类、没有自动加入第三个协变量、没有“继续对话”占位 |
| R16 步骤 9 | Parquet 保存成功，但模型提交的部分 `confirmations` 清除了既有 `feature_time_window` 回执，页面错误地要求重新确认 24 h | **P1，按脚本停止**。已修成导出确认只追加 `export_format=true`，不得清除已有 owner 回执；合同测试证明 `feature_time_window=true` 被保留。步骤 10–12 未执行，仍需新空白项目 fresh 浏览器复验 |

### 当前验证边界

- 相关 Copilot contract、静态 owner 与 research workflow：`274 passed`。
- `node --check` 与 `git diff --check`：通过。
- R16 真实浏览器已验证步骤 0–8；步骤 9 的失败已代码修复并由合同覆盖，但尚未用 R17 fresh 浏览器复验。
- 步骤 10–12 未执行；没有运行 preflight、Provider Plan 或完整分析，没有产生科学结果。

## 数据来源优先的新对话复测（R18）

本轮针对“用户尚未说明研究问题时不应看到抽取配置，也不应被要求自行猜测模块/队列/窗口”的反馈，新增独立的数据来源绑定入口，并从 fresh 空白项目重新验证。

| 检查 | 真实结果 | 判定 |
| --- | --- | --- |
| 新项目初始状态 | 项目从 `0/7`、未选择数据开始；确认来源前不开放研究消息 | 通过 |
| 数据来源入口 | 仅显示本地目录选择、来源识别和“确认数据来源并继续”；不显示“当前抽取设置”“开始抽取”、模块、队列、时间窗或导出格式 | 通过 |
| 来源绑定内容 | 只保存 MIMIC-IV 来源；StudyContext revision 2 的 cohort、modules、outcome、exposure、time window、export format 和 confirmations 均未被预设 | 通过 |
| 首句真实脚本 | 输入“我想要研究 MIMIC-IV 成人 ICU 人群 Sepsis-3 的患病率是多少，以及 Sepsis-3 与 ICU 死亡的关系。”后，模型保存问题和成人 ICU 意图，并只询问一个关键分析单位选择 | 通过 |
| 模型引导 | 页面给出“所有 ICU 住院单元”与“每位患者首次 ICU 住院单元（推荐，分析更易解释）”两个可点击选项；没有把技术模块或 Sepsis-3 内部窗口抛给用户 | 通过 |
| 来源权威 | 首轮即使模型进行了冗余的来源可用性读取，host 仍保留用户确认的精确目录，不允许研究问题中的数据库名称触发静默换源 | 通过（仍可优化一次冗余读取） |
| 抽取门禁 | 首句后科研流程为 `1/7`，特征提取仍是 `study_setup_incomplete`；没有生成抽取完成回执或分析结果 | 通过 |
| 页面布局 | 1662×1329 桌面视口中对话、下一步按钮、右侧流程状态和底部输入框均可见，未见横向溢出或裁切 | 通过 |

### 验证边界

- 相关 Copilot contract、gateway、workflow、static、extraction workspace、StudyContext frontend 和静态路由矩阵：`404 passed`。
- Ruff、Node 语法和 `git diff --check`：通过。
- 浏览器停在 R18 的第一个关键决定，未代替用户点击；没有开始抽取、preflight、Provider Plan 或分析，也没有产生患病率或关联估计。

## 刷新恢复稳定性复测（R18）

用户报告刷新后依次闪过“请选择研究项目”“模型连接”“开始研究对话”和最终历史对话，视觉上像连续崩溃。运行日志显示服务进程未重启，相关 HTTP 请求均为 200；浏览器时间线证明这是前端在项目列表、模型连接和历史会话异步恢复完成前反复渲染正式空态。

修复后，Guided shell 与 Copilot conversation owner 共用一个恢复屏障：刷新期间只显示“正在恢复当前研究”，项目列表、项目打开、模型状态、workflow、session list 和 transcript 全部完成后才切换到最终对话。项目切换也使用同一屏障。

| 真实浏览器检查 | 修复前 | 修复后 |
| --- | --- | --- |
| 约 0.18–0.45 秒 | 闪现“只选择一套提供方与模型” | 仅“正在恢复当前研究” |
| 约 0.7–1.4 秒 | 闪现“请先选择研究项目” | 仅“正在恢复当前研究” |
| 约 2.4 秒 | 闪现“在当前项目中开始对话” | 仅“正在恢复当前研究” |
| 约 3.2–3.8 秒 | 恢复原对话 | 一次切换到原对话 |
| 连续 3 次刷新 | 未测试 | provider/project/activate 三类中间页均 0 次；服务重启后的冷刷新 3.729 秒，后续两次暖刷新 2.822/2.827 秒，均回到同一分析单位选择 |

性能剖析同时发现 `/api/guided/drafts/list` 会先对全部 76 个历史项目执行 StudyContext、ProjectAuthority 和 run-history 投影，最后才截取侧栏可见的 20 项。现改为先按更新时间排序并截取可见项，再只投影这些项目；返回的总数仍为 76，未删除、改写或隐藏任何项目。该接口实测由 1.388 秒降至 0.610 秒（约降低 56%）。

相关 7 文件矩阵更新为 `406 passed`；Ruff、Node 语法和 `git diff --check` 通过。该验证证明刷新显示稳定、历史对话恢复和项目列表热路径改善；暖刷新仍需约 2.8 秒完成必要的模型、项目、workflow、session 与 transcript 水合，不能表述为即时恢复。

## API 连接授权交互精简

用户指出 API 配置表单中的强制授权 checkbox 与页面已有的连接授权说明重复。现删除额外勾选动作，将授权语义并入“验证并保存连接”按钮：用户主动点击该按钮时，浏览器显式向仍然 fail-closed 的后端发送 `enable_ai=true`；直接调用后端而未提供该授权时，原有 `external_llm_opt_in_required` 门禁保持不变。

隐私披露没有被缩减：页面仍明确说明对话文字、经 PHI 安全投影的摘要和工作区文件内容可能发送到所选服务，科研运行仍需另行确认。真实浏览器在一个已验证 API、尚未开始对话的新项目中重新打开连接设置，确认表单 checkbox 为 0、“验证并保存连接”按钮存在、横向溢出为 0；没有重新验证凭据、创建 session 或触发 Provider 请求。

相关 7 文件矩阵仍为 `406 passed`，授权 UI/路由聚焦矩阵 `148 passed`；Node 语法、CSS owner 扫描和 `git diff --check` 通过。

## 无数据先对话与官方 Demo 渐进授权

此前 R18 把数据确认放在对话之前：fresh 会话隐藏 composer，用户尚未说明研究问题就必须选择本地目录。这把数据执行授权误扩成了整段对话授权，也让没有本地数据的用户无路可走。本轮将边界改为“对话先行、用到数据时再确认”。

### owner 契约

- pending / selection-in-progress 会话可以讨论研究问题、研究设计、idea 与文献，也可以查看 path-free 来源目录。
- cohort、患者轨迹、数据包、Run/Evidence/解释/稿件和分析执行等数据相关工具，在来源未确认时由 `easyicu.webserver.pi_copilot.data_source_authority` 统一返回 `pi_session_data_source_confirmation_required`；前端不再靠隐藏输入框代替安全门禁。
- 新会话不会静默复用项目旧来源。用户可以在需要数据时选择项目来源、本地目录或官方 Demo；官方 Demo 选择不等于下载授权，准备/下载仍需下一次明确授权。
- 顶部原“完整演示”改为“审稿流程演示”，与官方 Demo 数据入口分离。

### 真实浏览器复测

| 检查 | 真实结果 | 判定 |
| --- | --- | --- |
| fresh 无数据入口 | 页面同时显示 composer、发送、本地目录、官方 Demo 和“先描述研究问题”；没有强制跳转数据页 | 通过 |
| 普通首轮 | 输入“先梳理研究问题，不要读取、下载、抽取或分析数据”后，15 秒完成；只读取 workflow 元数据，模型继续询问研究主题 | 通过 |
| Demo 首次探测 | 模型曾重复两次来源目录，并给出不适合无本地数据用户的本地完整库工作流；按 UAT 规则继续修复 | 已修复 |
| Demo final fresh | 新会话点击“查看官方 Demo 数据”，14 秒、1 个 `easyicu_list_data_sources` 完成；列出 MIMIC-IV Clinical Database Demo v2.2、eICU Collaborative Research Database Demo v2.0.1 及样本局限 | 通过 |
| Demo 行为边界 | 页面明确“尚未下载、绑定或使用”；最终选择仅为两个准确 Demo 和“继续无数据规划” | 通过 |
| 下一步按钮 | 模型输出的 Markdown `### 下一步：` 被 host 正确投影为 3 个可点击按钮，不再退化为 generic“继续对话” | 通过 |
| 页面布局 | 1662×1329，composer 存在，本地与 Demo 入口存在，横向溢出 0 | 通过 |

### 验证边界

- 相关 Copilot contract/static/routes 与 Web 静态/route/workflow 六文件矩阵：`161 + 232 = 393 passed`。
- Ruff、Node 语法和 `git diff --check` 通过。
- 本轮没有点击任一 Demo 选择，没有下载、导入、绑定、抽取、preflight、Provider Plan 或分析，也没有生成任何科学估计。

### 首屏常驻数据提示移除

用户继续指出，即使压缩成状态条，“尚未选择数据”仍在用户尚未提出数据需求时抢占首屏注意力。现取消普通 `pending` 会话的常驻数据 UI：顶部 workflow 后直接进入对话记录，数据来源仅由模型在研究真正需要时通过对话中的下一步选项提出；已经进入本地目录选择的 `selection_in_progress` 状态仍保留恢复入口，避免用户丢失正在进行的授权动作。

真实浏览器在 `1662×1329` 视口复核：`.gpi-data-consent` 数量为 0，页面不再出现“No data selected / 尚未选择数据”，composer 持续存在，workflow bottom 与对话区域直接衔接，横向溢出为 0。后端 tool owner 的数据读取门禁没有改变。CSS 仍由 `guided-pi.css` 单一 owner 持有，相关选择器未泄漏到 extraction、patient 或 crossdb 样式文件。

## 数据关键动作一键推进复测

用户在同一真实会话中先选择官方 Demo、后改为本地 MIMIC-IV v3.1。旧实现有两个连续断点：Demo catalog id 被错误提交给 `bind_source_id`，以及界面生成的“授权打开本地……数据提取工作区”没有转成当前轮次的一次性 Extraction 权限，因此用户反复点击仍停在 `pi_action_authorization_required`。

修复后，host 只从当前用户消息识别明确的数据动作，普通研究选择仍不获得 Extraction 权限；旧 transcript 中“选择 Demo”会显示为明确的“下载并准备 Demo”，发送内容要求直接调用 `easyicu_prepare_demo_source`，不再先绑定尚未注册的导出。对本地来源，实际界面产生的“授权打开”和“一次性 Extraction 授权”两种文案都映射为仅本轮 `extract` 权限。

真实浏览器在原失败会话中点击“一次性 Extraction 授权并打开本地 MIMIC-IV v3.1 数据提取工作区”：16 秒完成，回执为 `easyicu_local_source_workspace_ready`；没有再次出现授权阻塞或 Demo 绑定错误。修复后的前端同时自动展开右侧原生 Data Extraction owner，直接显示本地目录输入、`Browse…` 与“Choose folder and identify”，不再要求用户先点一次聊天按钮、再点一次 artifact 按钮。目录选择与扫描仍保留为必要的用户关键动作；本轮未替用户选择目录，未扫描患者数据，未下载 Demo，未运行抽取、分析或 Provider Plan。

聚焦回归 `87 passed`，覆盖 turn authority、下一步按钮投影、原生提取工作区与 replay 坐标；浏览器证据同时证明成功 tool receipt 会立即展开 `native_workspace`。

## 预览工作台持久回看与历史恢复

用户确认右侧预览除对话中的图表、证据和稿件外，还应保留抽取后的数据工作台视图，并在后续对话中随时回看。源码与聚焦合同确认该能力没有被删除：`data_workbench_snapshot` 仍支持 `cohort_summary`、`feature_distribution`、`patient_timeline`、`crossdb_comparison` 和 `icd_cohort_preview`，渲染继续委托给 Patient/Cohort/Cross-DB/Extraction 的原生 embedded owner；普通 `sendText()` 只重绘主对话，不调用 preview `close/clearProject`。

本轮补上两个回看缺口：

- 预览 owner 保存当前项目最近打开的 6 个受治理资源；当第二个资源打开后，右侧显示横向可滚动的“回看”入口，新产物替换当前视图时，旧的工作台、图表、证据或稿件仍可一键切回。主动关闭不销毁最近记录，切换项目才清空，避免跨项目串联。
- 历史消息资源 transport 现在完整携带 `entry_mode=source_binding`。修复前刷新后虽仍能看到“Connect local MIMIC-IV 3.1”，点击却退化成普通提取页；修复后重新点击会恢复数据来源设置与 `Browse…`，不再丢目录选择入口。

真实浏览器从刷新后的历史消息重新打开数据来源工作台：新 cache owner 均为 `20260825-preview-history1`，预览与对话并排可见（约 607/517 px），`Browse…` 可见，页面横向溢出为 0，未出现 preview error。随后打开只读审稿报告，右侧“回看”同时显示 `Open reviewer dossier` 与 `Connect local MIMIC-IV 3.1`；返回项目并切回本地来源后，两项仍在且当前项正确切换，目录浏览继续可见。没有选择目录、扫描患者数据、下载 Demo、执行抽取或分析。

直接相关矩阵为 `88 passed`（另有 1 项与本改动无关的既有 prompt 文案断言未纳入本轮通过数），JS 数据工作台 owner 为 `3/3`，两个修改后的 JS 文件均通过 `node --check`。
