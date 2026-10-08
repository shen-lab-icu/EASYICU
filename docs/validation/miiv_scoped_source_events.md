# MIIV 限定 stay 的逐事件候选提取（2026-10-09）

新增独立入口 `easyicu.io.source_events.extract_miiv_respiratory_events`，可供 preprocessing producer 直接调用。它不修改 `load_concepts` 默认行为，不写文件、不调用科研拟合、不改 sealed/current。这轮仅用临时合成 Parquet 开发和验证；真实候选提取由独立执行步骤完成。本实现不触碰旧 CRLCMM。

```python
from easyicu.config import load_src_cfg
from easyicu.datasource import ICUDataSource
from easyicu.io.source_events import extract_miiv_respiratory_events

source = ICUDataSource(load_src_cfg("miiv"), base_path=source_directory,
                       enable_cache=False)
result = extract_miiv_respiratory_events(
    source, allowed_stay_ids=explicit_allowed_stay_ids,
    concepts=["spo2", "o2sat", "peep_set", "peep_total", "fio2_chart", "fio2_lab"],
    interval_hours=1.0,
)
# result.hourly / result.trace / result.receipt are private research artefacts.
```

## 继承依据及科学边界

起点为 EasyICU `712531aa8fae8d994a7028978974d91bc452a4e2`，呼吸字典修复见 [上一批收据](respiratory_source_channel_repair.md)。已阅读既有 `datasource.py` 的 hospital lab forward rolling 路径及 DuckDB 多概念路径。后者按 hadm/time/item/value 分组可能折叠相同重复事件；候选保留物理行，不继承这种折叠。

[官方 MIIV lab 文档](https://mimic.mit.edu/docs/iv/modules/hosp/labevents.html) 区分患者、住院、原生事件、标本 ID，并说明部分记录没有住院 ID；charttime 通常对应采样，storetime 对应实验室结果可用。本接口不由缺 hadm 记录反推住院，也不把 charttime 当作决策时可获得结果的证明。

[官方 chart 文档](https://mimic.mit.edu/docs/iv/modules/icu/chartevents.html) 提供原生 stay 键，charttime 与录入/确认时刻 storetime 有不同含义。[官方 ICU stay 文档](https://mimic.mit.edu/docs/iv/modules/icu/icustays.html) 定义 ICU 身份与入出时钟。这些定义支持保留身份和时间证据，但不证明任意 rolling 归属是临床真实发生地点。

## 固定归属规则与隔离

Lab 规则明确命名为 `legacy_outtime_forward_rollends`：在完整相同 subject+hadm 的 stay 时钟中，选择第一个 outtime ≥ charttime 的 stay；超过最后 outtime 的记录归末 stay。然后只将精确 requested stay 的事件值返回 Python。选择时不能先删除未请求的竞争 stay。入 ICU 前、两次 ICU 间隙或出 ICU 后的 lab 可按该继承规则获得 stay ID；trace 的 `temporal_position` 分别标 `before_intime`、`within_icu`、`at_outtime`、`after_outtime`，不称它们全部发生于 ICU 内。本轮不新增严格区间替代分支。

源查询限定 item ID。Chart 在源 SQL 先限定 requested stay；lab 在源 SQL 先限定 requested stays 对应的住院域，使用相同 subject+hadm 完整时钟归属，再在同一 SQL 内检查精确 stay 及身份一致性。未授权 stay 的临床值不会被先读入 pandas 再筛掉。上下文只读 `icustays` 的 subject/hadm/stay/intime/outtime 五列，不读 admissions 或结局。临床值唯一的 `fetchdf` 查询带 `identity_status='identity_allowed'`。DuckDB 使用内存连接并禁用磁盘 spill；不声称共享 Parquet 压缩页或文件 hash 有物理患者隔离。

空/缺允许 ID、非正/非整数 ID、重复允许 ID、不存在请求 stay、非唯一 stay 元数据、同 hadm 的 subject 冲突、缺失/非法时钟、相同 subject+hadm 的重复 outtime 都明确失败。非请求竞争 stay 的坏时钟也不能忽略。时钟 schema 可为无时区 timestamp 或 naive ISO 字符串；显式时区 timestamp 拒绝，带时区/不可解析的字符串不被静默剥离。输出时钟是去标识源本地 naive 时间，**不是 UTC**。

同许可住院域中的缺事件时钟、身份错配、归到未请求 stay 的事件只输出私有原因计数，不返回其临床值。缺 hadm 记录不进入明确许可住院域，其贡献未量化；不额外扫患者其他住院或门诊数据补齐计数。报警项目不在请求 item 集合中，也不为统计其数量额外扫描。

## 转换与聚合合同

六概念仅用绑定字典的 MIIV 来源、范围、单位及 callback。`peep/fio2` 旧泛化定义不变。选取非 null 原 `valuenum`，否则原 `value`，转换为 object-string Series；即使非 null valuenum 是 ±inf，也不回退 value。FiO2 复用现有 `percent_as_numeric` 函数，能处理 `50%` 和 `0.4`；其他概念用数值解析。`numeric_value` 是 callback 前的解析值，`converted_value` 保留转换结果，包括 ±inf。

转换状态区分 `missing_input`、`unparseable`、`nonfinite`、`finite`；不能因 `50%` 的 callback 前值为 NaN 就误判失败。随后执行有限值与字典 inclusive bounds；零值仅在概念范围允许时保留。范围状态为 `not_evaluable/within/outside`。

完整顺序是 **逐事件转换 → 有限性/字典范围 → 所有保留事件的小时中位数**，不对分批中位数再取中位数。同小时 FiO2 `[0.4, 0.6, 50]` 转换后中位数为 50；“先中位数再转换”为 60 的反例说明顺序可能改变结果，不能假定所有旧路径等价。两份文件同值事件仍有两个贡献权重。

有合法时钟及归属的源事件即保留该 concept/hour 键，值全空或全超范围时小时值仍缺失；完全没有事件不制造网格。旧优化路径曾在源 WHERE 去 null，其他路径可能先聚合后 bounds，因此候选与 sealed 的差异可能包括**键集合变化**。新增缺失小时不是新测量或覆盖改善；下游必须单独解释身份、来源、顺序、重复权重和键的差异。

## 返回结构及失败

`SourceEventResult.hourly` 的列为 `stay_id`、`charttime` 与请求概念；时间为 ICU intime 起算的 float64 小时、按 interval floor，允许负数。空结果也有明确 int64/float64 schema。

`trace` 精确字段见源码 `TRACE_COLUMNS`：身份、概念、表/item、文件路径与 SHA、零起算原始 Parquet 行号及 event_key、可用时的原生 labevent_id/specimen_id、raw chart/store/value/valuenum/unit、ICU 时钟、归属规则/位置、callback 输入及来源、前后数值/状态、相对小时/桶、是否聚合、原因、该组 aggregate_value/aggregate_n。同一原事件可贡献到两个不同概念，不是互斥计数；不做原事件去重。没有进行标本动脉属性推断。

`receipt` 包括 schema `miiv_scoped_source_events_v1`、实际源文件清单与 SHA、字典 SHA、请求 ID 内容摘要（无 ID 列表）、归属/时间/转换/聚合合同、每个表/item 的源域原因计数、概念计数及与旧流程的差异。输入文件和字典在前后核验 SHA；发生变化即拒绝返回成功结果。文件路径、行级 trace 和所有计数均是私有产物，不能直接发布。

`SourceEventContractError.reason/counts` 保存不含患者值的合同失败原因。producer 应保留所有失败尝试，并执行自己的路由 SHA、输出 anti-join、独立时钟/转换/聚合和隐私核验。本库不会自动执行真实患者提取或发布。

## 验证

定向合成覆盖部分 stay 请求、同人跨住院、多 stay 竞争、边界/间隙/出口后、缺键、坏时钟/歧义 outtime、字符串时钟/时区拒绝、源 SQL 与 Python 返回边界、物理重复事件、所有新通道、混编码/百分号/零/边界/±inf/空值、all-null 键、输入文件变动。初版 21 例中两个 fixture 将 None 转为 NaN 后构造失败，修正 fixture 后全部通过；该失败未隐去。其后补强源语义测试，最终结果与独立审查见配套 JSON。不运行未经逐文件审阅的广域测试。
