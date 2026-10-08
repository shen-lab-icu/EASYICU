# MIIV 通气支持记录与决策时钟：独立候选接口

本轮是 **0 真实临床值读取、0 模型、0 实际提取** 的实现与合成验证。接口不修改现有 `load_concepts`、字典、三个已冻结 source API、sealed release 或 current。下一步真实候选 producer 尚未运行；本文件不是正式科学输入认证。

## 问题、来源及原文

M3 A172 的原资格读取 `floor(t)-1` 完整小时的 sealed `mech_vent`，并要求 mode/sequence 非 standby；该窗口目前属于回顾性 measurement-only 定义。第31动脉血气和第32非呼吸测量候选成功不等于通气支持在决策时已经可知。本接口只补 `mech_vent` 的两项原生 procedure 来源，不补模式、breath sequence、RRT、药物或液体。

已实际核对 EasyICU Git `97f508c6` 与本轮父 `af5b5d6b1c5a778c9f7d402ec422fc6a39089578` 的 `src/easyicu/data/concept-dict.json::mech_vent.sources.miiv`：均为 procedureevents 225792→invasive、225794→noninvasive，`sub_var=itemid`、`dur_var=endtime`。这证明该字典映射，不声称单一 Git 覆盖 sealed 每个模块全部执行历史。A172 映射入口为 M3 `code/config/m3_a172_input_mapping.json`，preparation `INPUT_PLAN.md`（d39000e 分支）；既有字段不会被本接口偷偷替换。

原文先行，本轮使用 nature-academic-search 方法；当前工具没有 PubMed/CrossRef MCP，采用官方源码和作者/出版社全文。没有引用计数排名，也没有用方法论文认证临床时间戳。

1. [MIMIC 官方 procedureevents](https://mimic.mit.edu/docs/iv/modules/icu/procedureevents/) 的 Table columns、starttime/endtime、storetime、statusdescription；网站正文工具失败后实际读取[官方固定 Git 源文](https://raw.githubusercontent.com/MIT-LCP/mimic-website/81822278432bc33e101116427e05fa4f69265a01/docs/iv/modules/icu/procedureevents.md)。程序记录不是必填，未记录不能证明未实施；start/end 为记录区间，store 为系统登记时刻，status 是最终状态。非有创通气记录较不完整。
2. [MIMIC 官方 inputevents 固定源文](https://raw.githubusercontent.com/MIT-LCP/mimic-website/81822278432bc33e101116427e05fa4f69265a01/docs/iv/modules/icu/inputevents.md)，start/end、store、amount/rate、status 段：速率变化可关闭旧区间并建立新行，完整区间量及最终状态不宜直接放回起始历史。本轮只作相邻来源方法对照，**不读取 inputevents**。
3. Albu et al. 2025，[JMIR 全文](https://www.jmir.org/2025/1/e73987/)，EHR Data Flow 与 Challenges and Recommendations / Feature Engineering：记录过程、提取与特征时间定义都可能产生伪影，应解释时间戳含义并做针对性验证。本文没有为“store≥end 即临床真值”提供保证。
4. Gottesman et al. 2018，[作者全文](https://arxiv.org/pdf/1805.12298)，§2、§4：观察性医疗 RL 的历史表示与数据伪影会影响评价。这里只用于说明决策前历史的意义，不能据此消除混杂、记录缺失或评价支持不足。

## 可调用接口与字段

```python
from easyicu.config import load_src_cfg
from easyicu.datasource import ICUDataSource
from easyicu.io.scoped_support_events import (
    extract_miiv_support_events, support_window_evidence,
)

# 示例不自行执行；真实 producer 需传经批准并校验的完整 TRAIN stay 名册。
source = ICUDataSource(load_src_cfg('miiv'), base_path=raw_root, enable_cache=False)
result = extract_miiv_support_events(source, allowed_stay_ids=approved_stay_ids)
```

返回 `ScopedSupportResult(events, clock_context, receipt)`，全部均为私有输出，API 自身不写盘。events 每行对应原始 procedure 物理行；重复、重叠、有创/非有创相冲突的记录均保留，不合并成无冲突状态。不提供小时补齐、carry-forward 或二元“无通气”。`clock_context` 覆盖所有许可 stay，包括零事件对象，不以有记录者重筛名册。

- 来源：subject/hadm/stay，source_item_id，source_file/SHA/source_row_number/event_key；原始 start/end/store/status/value/unit；可选 orderid/linkorderid/continueinnextdept 及缺列标记。
- 独立解析 start/end/store；保留 missing 与 invalid 区别。只接受 naive ISO 时间或原生无时区 timestamp。UTC 后缀/offset 不静默去掉。时间为源库去标识本地时间，不能称 UTC。相对小时都减同 stay ICU intime。
- 区间：positive_duration/zero_length/reversed/missing_start/missing_end/invalid_clock；连续 duration、store-start、store-end；记录值按明确 min/hour/day 单位另算 duration 差，不设修剪阈值、不填缺失值。
- 最终状态：官方 FinishedRunning/Stopped/Paused 为 `affirmative_terminal_record`；Cancelled/Canceled/Rewritten 单列；空与其他未知单列。所有行保留。最终状态不能回填为起始时刻已知的信息。
- native ICU 出口 NULL 单列 unknown，不影响 native stay 归属；非空非法 ICU 时钟、缺入科钟、重复 stay/身份冲突 fail closed。原生 stay 归属不需要其他住院或兄弟 stay 出口排序，因此本接口只返回并检查请求身份的 ICU 时钟。

## 两层时序合同

设记录开始 s、结束 e、登记 r；决策截点 T；过去窗口 [L,U)，L<U≤T。

**开始声明**：`recorded_start_claim_at=r` 仅当 s、r 可解析且 r≥s；否则未知。`recorded_start_claim_visible` 当 r<T。定义完全不读 e 或最终 status；即使改写未来 e、Cancelled 等状态，这一字段也不得改变。这只是“记录中开始声明已登记”的工作代理，**不是持续通气、实际操作完成或记录历史版本认证**。负时延原值保留，但不作为这条非矛盾声明证据。

**完成区间记录证据**：`completed_interval_record_at=r` 仅当 e>s 且 r≥e。决策可见还需严格 r<T。`completed_window_record_evidence` 再要求记录覆盖 [L,U) 且最终 status 属官方上述三类。若 r<e，不用 max(r,e) 合成可用时刻：该快照不能分辨计划结束、后续编辑或导出行为。

`retrospective_overlap_hours` 与 `retrospective_covers_window` 可以使用导出中的完整 e；它们明确属于回顾性审计层，不能混入开始声明历史。逐行窗口返回 `positive_completed_window_record` 或 `unknown`，不会把取消、未知、无行或不覆盖称“没有通气”。同时出现 IMV/NIV 证据也不自动仲裁。

即便 r≥e，单一 store 快照仍未证明最终状态/结束钟所有历史版本；本接口没有 `availability_certified=True`。需要正式实时资格时，仍须说明这项代理假设及 mode/sequence 等剩余来源。当前 M3 measurement-only QA 可独立继续，不以此新代理追加门槛。

## 源过滤、失败与执行边界

支持 flat/bucket Parquet 和 `icu/procedureevents.parquet` 单文件。所有分片 schema 先核；SQL WHERE 项目集合与精确 stay 名册、再核 subject/hadm 一致性，只有身份匹配的临床行 fetch 到 Python。身份不匹配只返回私有原因计数。无真实“全库零读取”或 Parquet 物理页隔离声明：文件哈希经过原始字节，SQL 引擎可解码页，承诺是返回研究进程的临床行边界。

源文件前后 SHA、API/共用 loader/字典 SHA、身份集合 SHA 和分源计数进入私有 receipt；源变动/未知 schema 失败，不降级到末端过滤。原事件钟异常作为质量状态保留；身份时钟异常终止。没有 admissions/death/outcome 读取，也不调用其他提取器。

建议后续 producer 复用已认证 A167/b **完整 TRAIN 原名册**，不按支持是否可得重选；独立 derived run、默认 dry-run、CONFIRM=1、精确 Git archive 新鲜子进程、失败目录不覆盖。只需这两项 procedure 与许可 icustays；另路 raw 物理 key 全集核对、逐行时钟/状态/窗口重算、全名册覆盖、公开小格联动检查。预计性能受 procedure 分片大小和文件哈希主导，未读取真实 schema/值估算，本轮不提供无依据的耗时保证。此计划不包含当前真实启动授权或科学 release 晋升。

## 本轮验证

实现与全部合成验证见伴随 `miiv_scoped_support_validation.json`。根节点独立测试文件用标量公式验证随机物理记录×窗口，并改变未来 end/status 检查开始声明不变。所有 fixtures 为临时新造数据，不读取真实身份、事件或候选 clinical parquet。旧 source API 文件不得随本次提交变化。
