# MIIV 标本级动脉 PO₂ 与分源 FiO₂：仅源码和合成验证

本接口是新的、显式调用的私有来源视图。没有修改既有 `source_events.py`、通用 `po2/pafi/fio2` 概念、六通道候选、sealed 导出或 current。此轮没有真实 PO₂ 事件读取、临床导出或拟合；合成通过不能代替后续获授权范围内的来源检查。

## 阅读到的原始依据与区别

固定 MIT-LCP mimic-code `303d26c623dcc9c49cc0f204468d4acc2f063797` 的 [bg.sql](https://github.com/MIT-LCP/mimic-code/blob/303d26c623dcc9c49cc0f204468d4acc2f063797/mimic-iv/concepts/measurement/bg.sql) 按 specimen_id 组合 52033/50821/50816；其前提是每标本每 item 单次测量，故用 MAX。FiO₂ 同标本优先，否则用此前含边界 4h 的最近 chart223835。它的 chart 联结实际仅 subject；本 API **并非该 SQL 的完整复刻**。

同版本 [first_day_bg_art.sql](https://github.com/MIT-LCP/mimic-code/blob/303d26c623dcc9c49cc0f204468d4acc2f063797/mimic-iv/concepts/firstday/first_day_bg_art.sql) 明确筛 `ART.`。[官方 labevents 文档](https://github.com/MIT-LCP/mimic-website/blob/81822278432bc33e101116427e05fa4f69265a01/docs/iv/modules/hosp/labevents.md) 将 specimen_id 作为同标本联结键，并区分通常对应采样的 charttime 与结果在系统可用的 storetime。源码和正文已实际阅读；不以字段名 PO₂ 或数值大小推断动脉属性。

本 API 保留重复、类型冲突和来源分支；不取最大 PO₂/FiO₂，不将无 specimen_id 的同小时事件拼为同标本。4h 是固定官方比较算法，非普适生理有效期。跨身份配对、原生时钟、可用时刻和既有字典范围另有明确合同。

## 入口与输出

```python
from easyicu.io.arterial_blood_gas import extract_miiv_arterial_blood_gas
result = extract_miiv_arterial_blood_gas(source, allowed_stay_ids=explicit_ids)
```

仅支持 MIIV；必须给非空、唯一正整数 stay allow-list。返回 `ArterialBloodGasResult`：

- `events`：获许可的物理 lab/chart 行，含 file+零起始行号、文件 SHA、原生 labevent/specimen ID、原值/单位、charttime/storetime、换算值、字典范围状态、归属及源时钟状态。52033 原文保留。
- `po2`：每条物理 50821 一行。类型为 `NO_SPECIMEN_ID`、`IDENTITY_CONFLICT`、`INCOMPLETE_ALLOWED_TYPE_EVIDENCE`、`TYPE_MISSING`、`TYPE_EMPTY`、`CONFLICTING_TYPES`、`OTHER_RECORDED_TYPE` 或 `DIRECT_ARTERIAL`。只有规范化后唯一非空类型 `ART.` 可认证。空类型行另计；ART. 加空行不人为等同于矛盾。`OTHER_RECORDED_TYPE` 不是由本接口认证的“静脉”。
- `pairs`：每个 PO₂ 分别保留 `SAME_SPECIMEN_LAB` 和 `PRIOR_CHART_4H`。无匹配也有占位行；多 PO₂、多 FiO₂ 产生全部物理 pair，不选 MAX、不去重。同标本每条 lab50816 保留，可能跨已归属 ICU stay，但必须同 subject/hadm，且标记 same_assigned_stay 及时间差。chart 必须同 subject/hadm/已归属 stay，取 `[t−4h,t]` 内最近有效 FiO₂ 时点，并保留该时点所有物理重复或无效伴随行。
- `clock_context`：v2 允许 stay 的时钟账本，含零事件者。不是所有 ICU 的临床导出。
- `receipt`：源文件/字典/既有加载器 SHA、源过滤计数和精确语义。所有逐行内容及实现计数均私有；公开需另审查。

## 身份、完整类型证据和记录过程

复用固定 v2 的 `_prepare_clocks` 与 `_read_scoped_events`。先在完整同住院时钟上归属，后按 allow-list 返回临床值；未知出口只返回在所有合法补全中均唯一归属的 lab，不改变 `legacy_outtime_forward_rollends`。原有出院外记录及 unknown 出口状态保留，不称都发生在 ICU 内。

对允许 PO₂ 锚定的非空 specimen_id，额外在数据库引擎内检查三个 lab 项的 subject/hadm 冲突、52033 总物理行与获许可返回行之差。只返回允许 specimen 的标志/数量，未许可类型值不返回 Python。这样可以识别“允许 ART.，但同标本另有未许可类型证据”的情况；不能把过滤后的局部 ART. 集合冒充完整鉴定。此完整性限于本次绑定的三个 lab 项和源文件，不声称所有临床信息完整。

标本关联与 ICU 归属分开；原始不同 ID、重复同值/异值不消失。lab FiO₂ 未返回证据数保留，已观察到的各物理 pair 仍可作为记录级算术；它不是唯一标本 FiO₂ 的认证。

## 数值、单位和两种时间

沿用当前字典 PO₂ 20–600 mmHg、FiO₂ 21–100%，FiO₂ 使用现有 `percent_as_numeric`。这是复用的范围，不是新亚型门槛，也与官方 SQL 的 20/20.0 附近条件不完全相同。无效原记录留 events/pair，比例留空；不把它们删除后称完整测量。优先非空 valuenum，否则 value；无穷数不借其他字段修复。

显式单位冲突不自动换算、不生成比例。缺失单位按该 item 的字典单位计算，但以 `missing_assumed_dictionary` 标明假设，不能称原生单位已认证。只有直接 ART. 且数值/显式单位可用的物理 pair 才计算 `100 × PO₂ / FiO₂_percent`。类型认证和数值范围判断分开，低于字典范围的 PO₂ 仍可有直接 ART. 类型，但不计算该视图比例。

`available_at` 取所需 PO₂、全部相关类型证据、该 FiO₂ 的最晚 storetime。任一所需 storetime 缺失、格式无效或早于其 charttime，则严格可用时刻为 unknown，具体状态保留。**负记录延迟不删除事件或已有比例，只使严格 available_at 未知**；没有将 chart/store 矛盾悄然改成真实时刻。这与保留两种记录时钟不是同一个判定问题。本接口是回顾性来源视图；后续若评价时点 t，需要显式以 available_at 检查当时可用性，不能直接使用所有事后类型。

## 可执行性与验证范围

标本先分组；chart 按 subject/hadm/stay 建组排序，再用二分查找定位时间，避免每个 PO₂ 重扫全体事件。保留全部物理 pair 的输出大小仍会随同标本记录乘积增长，不以去重掩盖真实重复。

定向命令：

```bash
PYTHONPATH=src python -m pytest -q --disable-warnings tests/core/test_arterial_blood_gas.py
```

相关完整 suite 只选逐文件审阅的合成/目录合同测试；不跑数据访问边界未知的广域 fast suite。测试覆盖无标本/空类型/重复类型/冲突、跨 hadm 污染、未许可证据、全部物理多对多、4h 等号/未来排除/同刻 ties、storetime 缺失/反序、未知出口、字典范围、SQL/Python 过滤、文件变更、无 PO₂ 空输出，以及多标本打乱顺序后的逐格一致性。测试数量、SHA 与独立复核见配套实施收据。
