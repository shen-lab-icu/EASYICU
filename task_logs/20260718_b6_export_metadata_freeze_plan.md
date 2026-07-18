# B6 export-intake / concept-metadata freeze plan

> 状态：2026-07-18 15:16 EDT；**B6-A 已完成，B6-B 正式里程碑仍为五步 2/5；ResearchContext v2 所需的 typed materialized cohort/trajectory authority 与 replay bridge 已由 `7cc215e` 完成**。
> 当前代码：`acc874c`/`1481f09`（B6-A）+ `7ee66fd`/`7e8c16f`（projector/source authority）+ `987bdc5`/`7674814`（sidecar/native export→Agent binding）；
> packaged concept baseline：`8e97d31`。
> 边界：B6-B 不混入新的 concept-dict 科学内容编辑；只建立共享、可摘要绑定的 metadata 投影。

## 实施状态

| 子阶段 | 状态 | 证据 | 下一步 |
|---|---|---|---|
| B6-A manifest-authoritative intake | **完成** | `acc874c`；native-first + legacy、Parquet/CSV/XLSX、containment、feature definitions、cohort/catalog/replication 共用 adapter | 保持兼容与 fail-close，不扩 case-specific intake |
| B6-A snapshot/performance | **完成** | `1481f09`；read-only mmap verified snapshot、CSV/XLSX 单会话解析缓存、Parquet 列裁剪、四 consumer 显式 close；105 项主回归 + 64 项独立复审 | 超大 CSV/XLSX 的内存阈值/临时列式缓存是后续预算优化，非 freeze blocker |
| B6-B typed metadata / replay identity | **进行中（2/5 + v2 bridge 前置）** | `7ee66fd` + `7e8c16f` projector/source authority；`987bdc5` + `7674814` sidecar/native export→Agent strict binding；`7cc215e` typed cohort/trajectory authority、RunInputCapsule v3、legacy v2 verified receipt 与 runner pre/post integrity；202 + 95 项最终回归 | 实施 ResearchContext v2 与 scoped Planner/Coder/repair 投影 |

## 为什么 B6 不能后置

EasyICU 的概念层跨库抽象已经存在。历史上“原生导出包 → research-agent”没有
可验证的端到端契约：Web producer 写 `_manifest.json`，而 Agent 只明确识别 legacy
manifest，CSV/XLSX 与 `feature_definitions.json` 也没有统一 consumer。这个缺口已由
`acc874c` + `1481f09` 关闭；B6-A 不再是 blocker。

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

实施证据：`acc874c` 完成上述 core；`1481f09` 为每个 manifest 物理文件捕获一次
匿名 verified snapshot，解析器只能看到无 writable fd 的 `mmap.ACCESS_READ` 流。
CSV/XLSX 在显式复用的 package session 内只解析一次并返回 defensive copy，Parquet
保留按列读取；materializer、trajectory、case builder、discovery 都显式管理生命周期。
打开后 live source mutation 不进入 session 结果，但末尾 reverify 必须发现 authority
drift。诚实边界：重复 path-form 调用会各自建立 session；超大 CSV/XLSX 首读同时
占用匿名文件副本与完整 DataFrame，后续可加大小阈值或临时列式缓存。

## B6-B：typed concept metadata projection

建立 concept-layer 单一纯投影，Web feature definitions 与 ResearchContext 共用，
并明确区分：

- `source_database`：本次真实数据来源；
- `available_databases`：词典声明的 database keys（可含空 source key），不得单独解读为已验证可提取；
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

### B6-B 五个独立提交

1. **共享 leaf projector（已完成：`7ee66fd` + `7e8c16f`）**：新增 `easyicu/concept/metadata_projection.py`；Web 与 Agent 后续只消费这一投影，不从 concept 层反向 import `research_agent`。已锁 actual database 与 dictionary source resolution chain、class-prefix most-specific 继承、可执行 source anchor、显式 time coordinates、双范围和 canonical digest。
2. **intake/native export v2（已完成：`987bdc5` + `7674814`）**：生成 content-addressed column-level metadata sidecar；明确 `source_database`、物理列角色、时间 origin/unit 与 sidecar digest。Web 生产者在包级别先证明每个选定 concept 有唯一 primary binding，Agent intake 对 schema/digest/coverage/source-selection 做 exact join；v1 权威不变。
3. **ResearchContext v2 bridge**：只给新 payload 增加 typed metadata；旧 v1/封存字节继续可读，旧科学身份不得因新增 `None` 键漂移。
4. **authority / coder / replay binding**：将 projector、sidecar、ICU-rule implementation digest 纳入已有 resolved-input/context/environment identity；不新增第二套 StepAuthorityCapsule schema。
5. **drift / freeze gate**：metadata/rule drift 触发 sealed artifact 选择性重审；补 companion、来源、范围双轨、旧 payload 兼容与 Web/Agent 同投影回归。

投影红线：`source_database` 是真实来源，不等于 `available_databases`；dictionary
`extraction_bounds` 不等于 ICU `analysis_plausibility_range`；`*_n` 无生理单位，
`*_measured` 为 binary `[0,1]`，`*_first_time`/`*_last_time` 为距 ICU 入科小时，
只有数值聚合列继承基概念单位。已复现的 `lact_first_time` 继承 mmol/L/[0,30]
必须由通用 companion-role 规则修复，不写乳酸或三题特例。

## Freeze 验收门

- native Parquet / CSV / XLSX export intake 全绿；
- export-local metadata 缺失或损坏 fail-closed；
- 时间/count/measured/value companion 反例全绿；
- actual source 与 availability 分栏；
- extraction bounds 与 analysis plausibility range 双轨不互相覆盖；
- metadata/rule digest 改变会改变 replay identity；
- meta benchmark、capability drift、capsule/resume/provider/evidence authority 常绿。

完成剩余 B6-B 后才进入完整 freeze shards；不以 legacy Parquet 三题“还能跑”
替代 native export 端到端通用性。
