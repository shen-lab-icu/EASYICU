# 原生未知 ICU 出口的唯一归属修订（v2，2026-10-09）

这是候选 `20261009a` 真实失败后明确追加的修订，不是原先预设规则。a 保留 FAILED、源码 `4edfb6695ba848206c76dbc34bcb1877b351031e` 及全部冻结产物；不会覆盖失败或静默重跑。该 run 停于身份/ICU 时钟检查，未执行呼吸事件查询；身份、时钟和源文件 hash 读取已经发生。后续由 preprocessing 独立诊断确认原生 `outtime` 缺失，涉及单 stay 和多 stay 住院，`intime` 有效；不是格式、CAST、逆序、重复身份或已知出口冲突。具体小计数和键仅留私有诊断，不在本文件披露。

本轮 EasyICU 修订仅阅读源码/非患者资料与诊断摘要，测试全部用临时合成数据，没有再读取真实路由或临床事件，没有执行候选 b。v1 合同与验证见[历史说明](miiv_scoped_source_events.md)，其失败过程不回写为通过。

## 医学与记录依据

[MIIV v3.1 官方说明](https://physionet.org/content/mimiciv/3.1/) 区分连续 ICU transfer 合并与保留的非连续 stay；不能任意合并住院内所有 ICU 身份。[lab 官方文档](https://mimic.mit.edu/docs/iv/modules/hosp/labevents.html) 区分住院、采样和结果可用记录：实验室表没有原生 ICU stay，归属必须明确。原生 chart 已有 stay，其身份与已知入 ICU 时钟不依赖出口是否齐全。

[EHR Safari 原文 §5.2](https://proceedings.mlr.press/v182/boag22a/boag22a.pdf) 使用 MIMIC-III 说明不同记录流程可产生时钟不一致，反对仅凭常识清洗排除。它不是对本次 MIIV 缺时钟的认证或插补依据。本次不借 hospital discharge、末次事件、死亡或新时长阈值填出口，也不假设不同 stay 的时间范围不重叠。

## 明确的新规则

身份、`intime`、已知 `outtime` 仍沿原合同核验。保留 **原始 `outtime IS NULL`** 标志；只有原生 NULL 可成为未知出口。非空但解析失败、时区格式错误、已知时间逆序、重复身份或重复已知出口仍明确报错。没有放宽时间格式正则。

Chart 继续由原生 subject/hadm/stay 确定身份，允许 stay 与 `intime` 有效即可在未知出口时返回。事件早于 intime 仍标 `before_intime`；其余未知出口记录标 `outtime_unknown`，不冒称处于 ICU 内。

Lab 的基本归属名仍为 `legacy_outtime_forward_rollends`：最早不早于事件的出口，否则末出口。已知完整时钟的结果不变。对于未知出口，设事件时刻为 t、未知 stay 入 ICU 时刻为 L，**只假设该未知出口 E ≥ L**，保留所有可能补全，包括可能产生并列出口的补全；并列视为无法唯一归属。只在所有补全下均是同一唯一赢家时返回事件值：

1. 存在已知未来出口时，令 K 为最小的已知 E ≥ t。只有每个未知 L 都严格大于 K，才唯一归 K 的 stay。
2. 不存在已知未来出口时，必须恰有一个未知 stay，且其 L 严格大于所有已知出口的最大值，才唯一归该未知 stay。没有任何已知出口时此比较成立，包含单 stay 原生出口未知的情形。
3. 其他部分时钟情形保留为 `unknown_outtime_assignment`：仅返回许可源域内的私有原因计数，不把无法归属的临床值带回 Python。不删除未知竞争 stay，也不把整名患者或整批候选排除。

严格不等号有实质作用：若某未知 L ≤ K，就能取 E=K 造成并列；若无已知未来而未知 L ≤ 已知末出口，则可改变末端赢家或造成并列。两个以上未知出口可分别成为最早未来赢家，故不能选一个。单未知且其 L 已晚于所有已知末端时，无论它在事件前还是事件后结束，legacy 的 forward 或 rollends 都归该未知 stay。此推导不使用未观察到的结局。

## 反例与解释

已知 stay A 出口 10、未知 stay B 入 ICU 20：t≤10 的 lab 唯一归 A，t>10 唯一归 B；不能因住院中存在未知出口而屏蔽所有 lab。若只允许 B，仍先确定属于 A 的早期事件，再在源 SQL 中排除，不能把它改配给 B。

已知出口 10、未知入 ICU 10：补全出口 10 可并列，不能选赢家。两个未知 stay 且无已知未来出口，或早期未知竞争 stay 可能排在已知未来出口前，都要保留不确定性。

未知出口下的 lab 返回表示**在明确补全域内身份可识别**，不表示出口被恢复，也不证明该测量发生于 ICU 内。source 查询、原始 file+row 事件权重、逐事件 callback、范围、小时中位数和 all-null 键合同保持不变。

## 接口和账本

`extract_miiv_respiratory_events` 参数不变。`SourceEventResult` 增加私有 `.clock_context`，每个允许 stay 一行，即使没有任何返回事件也保留：

- `subject_id/hadm_id/stay_id/intime/outtime`
- `outtime_status`：`known` 或 `unknown`
- `context_stay_count/context_unknown_outtime_count`：完整同 subject+hadm 的上下文数，含未请求的竞争 stay。

账本不返回非允许 stay 的逐行身份。未知 outtime 原样缺失。必须保存这个账本，避免将未归属 lab 的空白误称为真正未测。

Trace 增加 `lab_assignment_status`：chart 为 `native_stay_id`，完整时钟 lab 为 `complete_clocks`，含未知时钟但唯一可识别的 lab 为 `identified_under_unknown_outtime`。后者获配的 stay 自身可以有已知或未知出口，取决于上述规则。`temporal_position` 增加 `outtime_unknown`。

Receipt schema 更新为 `miiv_scoped_source_events_v2`，新增 `unknown_outtime_rule` 与 `clock_context_rows`；既有 `lab_assignment_rule` 名称保留。source counts 的 `unknown_outtime_assignment` 是归属不确定性，不等于记录没有发生。

## 验证边界

旧 Git `4edfb669` 的完整已知时钟实现通过真实合成调用与 v2 比较：hourly、原有 trace 字段与 source counts 逐格相同。新增回归覆盖单未知、多未知、可识别子区间、严格等号并列边界、两种部分 allow-list、零事件 stay 账本、原生 NULL 与非空错误时钟的区别。历史 Git 对照用例在浅检出缺少该 Git 对象时会明确 skip，其余合成用例不依赖 Git；本次实际对照已执行。

M2 独立数学 oracle 枚举完整补全并直接应用 forward/rollends 与并列定义，不调用生产 SQL 条件。其有限验证与真实 API 的独立反例见配套收据；有限枚举不是对实际缺失机制的认证。v2 不恢复真实出口，不认证候选或正式科学结果，b 的执行需绑定本版本并通过 producer 独立核验。
