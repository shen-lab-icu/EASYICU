# B6 export-intake / concept-metadata freeze plan

> 状态：2026-07-18 只读审计完成；B6 是 Track B-Core freeze blocker。
> 边界：不修改当前并行数据会话持有的 `concept/callbacks.py`、
> `data/concept-dict.json` 或 `data/sofa2-dict.json`。

## 为什么 B6 不能后置

EasyICU 的概念层跨库抽象已经存在，但“原生导出包 → research-agent”尚未形成
一个可验证的端到端契约。当前 Web producer 写 `_manifest.json`，而 Agent 只明确
识别 legacy `easyicu_export_manifest.json`；CSV/XLSX 不在 export-package intake
路径，`feature_definitions.json` 也没有 Agent consumer。因此不能在 freeze 时声称
native Web export 能通用、跨库地进入 Agent。

同时，当前 metadata 投影会把 `lact_first_time` / `lact_last_time` 按前缀继承为
乳酸数值（`mmol/L`、生理范围），但它们实际是距 ICU 入科的小时数；本次真实
来源数据库与词典支持数据库也尚未分栏。这是确定的语义错标，不只是文档债。

## B6-A：manifest-authoritative export intake

新增纯 leaf intake 边界，research-agent 不反向 import Web 层：

1. native `_manifest.json` 优先，兼容 legacy manifest；无 root manifest 的目录不猜测为 export package。
2. manifest-listed files 是唯一物理数据 authority；manifest 外 stale 文件不进入 catalog。
3. 支持 Parquet / CSV / XLSX 的 schema inspection 与 projected read。
4. 路径 containment、symlink escape、缺失文件、格式/扩展名不符、重复 concept 全部 fail-closed。
5. manifest 声明 `feature_definitions` included 时，缺文件、schema/version/count 不符必须失败。
6. cohort materializer、data catalog 和 replication discovery 共用同一个 adapter。
7. native marker 不得回落到 `api.load_concepts()`。

验收至少覆盖三格式 round-trip、native-first/legacy compatibility、stale/missing/
escape/duplicate 对抗样例，以及 manifest authority 进入 cohort/catalog 的集成路径。

## B6-B：typed concept metadata projection

建立 concept-layer 单一纯投影，Web feature definitions 与 ResearchContext 共用，
并明确区分：

- `source_database`：本次真实数据来源；
- `available_databases`：词典支持范围；
- `extraction_bounds`：提取/filter 阶段边界；
- `analysis_plausibility_range`：分析期 flag-only 生理合理域；
- raw table/value/unit/time/item-ID lineage；
- companion role：count、measured、first_time、last_time、aggregated value。

伴随列规则必须是结构化、case-neutral 的：`*_n` 无生理单位；`*_measured` 为
binary `[0,1]`；`*_first_time` / `*_last_time` 为 hours from ICU admission；只有
声明为聚合数值的列才继承基底概念单位。dictionary min/max 绝不能覆盖
`icu_rules.py` 的 flag-only plausibility range。

metadata projection 与 ICU rule implementation 的 digest 必须进入 RunInputCapsule /
replay identity；规则变化触发选择性重审，而不是静默复用旧 success。

## Freeze 验收门

- native Parquet / CSV / XLSX export intake 全绿；
- export-local metadata 缺失或损坏 fail-closed；
- 时间/count/measured/value companion 反例全绿；
- actual source 与 availability 分栏；
- extraction bounds 与 analysis plausibility range 双轨不互相覆盖；
- metadata/rule digest 改变会改变 replay identity；
- meta benchmark、capability drift、capsule/resume/provider/evidence authority 常绿。

完成 B6-A、B6-B 后才进入完整 freeze shards；不以 legacy Parquet 三题“还能跑”
替代 native export 端到端通用性。
