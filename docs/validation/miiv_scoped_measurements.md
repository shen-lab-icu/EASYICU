# MIIV 肌酐、MAP、CVP：分源双时钟接口

本 opt-in 接口只返回明确允许 stay 的源事件与时钟账本。当前验收仅源码和人工合成 Parquet；没有真实患者扫描、导出、拟合或 release/current 指针变更。既有 `source_events.py` v2 和 `arterial_blood_gas.py` 保持原字节。

## 问题与原始依据

M3 A172需要区分决策前已有测量与系统已录入/可获得的历史，六呼吸通道接口没有覆盖肌酐、MAP、CVP。该最小完整块补足三种测量的来源与双时钟，并不认证其余支持、RRT、升压剂、液体、镇静或55个旧混杂字段。

实际阅读固定版本官方全文：[chartevents](https://github.com/MIT-LCP/mimic-website/blob/81822278432bc33e101116427e05fa4f69265a01/docs/iv/modules/icu/chartevents.md)、[labevents](https://github.com/MIT-LCP/mimic-website/blob/81822278432bc33e101116427e05fa4f69265a01/docs/iv/modules/hosp/labevents.md)。charttime通常最接近测量时间，chart storetime表示输入/验证；lab charttime通常为采样，storetime为结果可获得时间。官方同时指出部分lab会复制到chart，冲突时建议优先lab；本源接口不据此删物理记录，消费者必须显式选择，不能把两路自动池化后称独立检测。

[官方固定 vitalsign.sql](https://github.com/MIT-LCP/mimic-code/blob/303d26c623dcc9c49cc0f204468d4acc2f063797/mimic-iv/concepts/measurement/vitalsign.sql)将220052/220181/225312合成mbp，并另列220181为mbp_ni。本接口保留3来源，既不重复该聚合，也不借用其不同数值界限。

当前概念字典与封存 `97f508c6` 对 crea/map/cvp 的 MIIV sources 结构逐项相同；没有重写旧科学肌酐或重新认证其历史来源。实际非患者 `d_items.parquet / d_labitems.parquet` 标签检查及 SHA 在配套 JSON 中。

|concept|table/item|source_channel|非患者字典证据|
|---|---|---|---|
|crea|lab50912、52546|lab_blood_chemistry|Creatinine；Blood/Chemistry，**不能凭此认证serum/plasma**|
|crea|lab52024|lab_whole_blood|Creatinine, Whole Blood；Blood/Blood Gas|
|crea|chart220615|chart_serum|Creatinine (serum)|
|crea|chart229761|chart_whole_blood|Creatinine (whole blood)|
|map|chart220052|arterial_bp_mean|Arterial Blood Pressure mean|
|map|chart220181|noninvasive_bp_mean|Non Invasive Blood Pressure mean|
|map|chart225312|art_bp_mean|ART BP Mean|
|cvp|chart220074|central_venous_pressure|Central Venous Pressure|

这些是项目标签分类，不是每条记录真实采样方式/导管状态的认证。肌酐0–25 mg/dL、MAP0–250 mmHg、CVP−5–50 mmHg取自绑定字典，端点包含；它们是导出边界，非新增生理筛选或治疗阈值。三概念MIIV的概念/来源callback均为空；未来映射或callback变化不能无审查继承此实现。

## API 和字段合同

```python
from easyicu.io.scoped_measurements import extract_miiv_measurement_events
result = extract_miiv_measurement_events(
    data_source, allowed_stay_ids=explicit_stays,
    concepts=("crea", "map", "cvp"),
)
```

仅 MIIV；allowed 是非空、唯一正整数stay，不接受bool/浮点数。concepts可选非空子集，不新增HR等未审通道；只解析所需源表。返回 `ScopedMeasurementResult(events, clock_context, receipt)`，不写输出目录、不聚合、不填补/前向延续、不决定基线或风险集。所有表、身份键、计数均为**私有**，公开需后续披露审查。

`events`每物理事件一行，无chart/lab去重或MAP来源合并。file+row构成event_key，原生重复labevent_id和同值重复都保留。

|字段组|列与逻辑类型|定义|
|---|---|---|
|身份|subject_id/hadm_id/stay_id/source_item_id：整数；labevent_id/specimen_id：可空整数|只返回已允许且身份一致事件；lab原生ID缺列可空，不伪造|
|来源|source_table/source_file/source_file_sha256/source_row_number/event_key、concept/source_channel/item_label、dictionary_sha256|精确物理源与字典版本；row_number为文件内零基行号|
|原记录|raw_charttime/raw_storetime/raw_value/raw_valuenum/raw_valueuom|原字段保留，不转换时区或回填时间|
|时钟|charttime/storetime/intime/outtime：naive时间戳；measurement_hours/store_hours/store_delay_hours：浮点小时|分别相对原生intime及store−chart。缺失保持NaT/NaN，不要求ns/us底层dtype一致|
|时钟状态|charttime_status=known；storetime_status=known/missing/invalid；clock_order_status=unknown/store_before_chart/store_at_or_after_chart|缺/非法chart被v2源归属计数拒绝，不进入事件。非法非空store仍原文留痕，不伪装原生缺失|
|归属|assignment_rule/temporal_position/lab_assignment_status|v2完整同subject/hadm时钟后归属再allowed过滤；before_intime/outtime_unknown/within_icu/at_outtime/after_outtime原样区分|
|数值|numeric_source/numeric_value/converted_value/callback/conversion_status|lab明确value_var=valuenum，不从value文本救值；chart未指定时valuenum优先、缺失才value；非有限数不fallback。identity_numeric，不进行未知单位换算|
|范围/单位|bounds_min/bounds_max/bounds_status、expected_unit/unit_status|within/outside/not_evaluable；declared_match/missing_assumed_dictionary/mismatch|
|分析值标记|retained_for_analysis：bool；exclusion_reason：文本|只表示有限值、字典范围和单位兼容，**不认证可用性、记录在ICU内、风险合法或生理正常**|

显式单位冲突保留数值但 `retained_for_analysis=False`；缺单位使用明确字典假设并标记，不宣称单位已实测认证。零值肌酐在源范围内；消费者若研究正Cr，应在其冻结方案定义，而不是源接口偷偷删除。

storetime负延迟保留两时钟、值及矛盾标志，不在本接口生成单一available_at。M3若选择chart<t且store<t的操作性历史，须另固定；ABG严格最晚必要可用时间的规则不同，两者不能混称一个认证门槛。storetime不证明床旁人员首次知情，也不保证下游系统同刻可见。

`clock_context`仅allowed stay，包括零事件stay及outtime_status、同住院stay数量/未知出口数量。缺出口继承v2所有补全下可识别归属；native chart身份与真实发生时段分开。lab无法唯一归属只返回原因计数，不带出临床值；未知出口不会被补成∞或自动删人。

`receipt`绑定API/source_events/字典/源文件SHA、概念specs、allowed摘要、计数和规则。源文件与字典前后哈希不一致则拒绝。SQL过滤保障返回域，不声称物理Parquet页只读允许人的字节；trace自洽也不是原始全集完整性的独立证明。

## 合成复现与失败记录

精确相关suite/计数见配套JSON。新测试覆盖9项完整来源、重复物理事件、零事件账本、完整时钟先归属、部分出口可识别与歧义、数值/单位边界、负/缺/非法store、SQL返回域及输入变动。与冻结的呼吸/ABG及字典合同相关测试一起运行，不运行范围未知的广域fast suite。

首次新测试收集失败：fixture模块不是可直接导入包；改成显式按本仓测试文件加载后通过。失败日志摘要哈希保留；它不是临床试次，也不修改任何历史失败收据。


根节点另写的独立组合反例持久化为 `tests/core/test_scoped_measurements_root.py`，只复用合成 writer：10 个输入物理事件中返回9事件/2允许时钟；检查迟回报旧lab不变成最新测量、whole-blood/chart副本分开、同刻三MAP不合并、缺store值保留、负delay标记和非允许stay值不返回。这不是额外真实抽取或临床完整性核验。
