# ICU治理挖矿：协议、证据与缺口清单

统计证据见 [`governance_mining.json`](governance_mining.json)（派生数据）。本文只记录协议、覆盖矩阵结论与新缺口；不复述数字。

## 协议（2026-09-18，可重跑）

- 抽样框：20家期刊（10家顶刊+10家专科二线）2021–2026年MIMIC/eICU原文，去重251 PMID。注：JAMA/NEJM/Lancet/Chest五年合计不足10篇——顶刊综合刊基本不发数据库挖掘，这是抽样框本身的发现。
- 全文：PMC链接135篇（54%，顶刊付费墙效应），BioC取回121篇，118篇有实质Methods段。
- Pass 1（确定性）：22项治理清单正则（锚点/纳排/去重/缺失/immortal/竞争/敏感性/验证/指南/代码等）。
- Pass 2（LLM，4路扇出）：time-zero定义、去重规则、缺失处理、immortal提及四问，verbatim逐字+五态分类，不许推断。
- 自查：20篇抽查约15篇全对；SOFA作协变量误触定义标签；MoE/DTR类新词在词表外。

## 覆盖矩阵结论（坑位 → 现状）

有主且够硬：time-zero显式化（locked cohort+target_trial检查，比文献41%隐含锚点更严）、landmark执行器、immortal检查、纳排锁定、MICE、敏感性包、验证/复现、STROBE/TRIPOD清单、PSM kernel。

确认缺口（四项，2026-09-18已全部落地为finding，零schema/签名改动）：
1. **重复入住去重审计未进计划** → `REPEATED_STAY_DEDUP_UNDECLARED`（major，`planning/scientific_review.py`）：只打以前静默的情形；首住法/已绑定dependence/已执行敏感轴/纯计数描述/已评审signed runtime五条出路。
2. **单值插补无审计** → `gates/preflight.py` AST规则（error）：抓method填充、前后向填充、插值、聚合填充、裸SimpleImputer；Pipeline包裹与标量fillna放行（修复模板兼容）。
3. **缺失策略未声明不拦** → `MISSINGNESS_UNEXAMINED_COMPLETE_CASE`（major）：全complete-case且无missing敏感轴/override才亮。
4. **样本量/EPV无门** → 行级EPV转maturity `LOW_EVENTS_PER_VARIABLE`（minor 提醒，非硬门；显式n_parameters优先，最终充分性需结合参数数目及独立科学审阅，不能由单一EPV启发式判定）。

当前边界：CIF描述子不做Gray检验；另有独立的实验性Gray检验kernel，尚无R `cmprsk`对拍。complete-case“显式声明0%”说明没人会写那句话——门应查“策略是否声明”而非“是否含complete-case字样”。

## 边界

- 顶刊样本偏向写得好的；社区真实水平只会更 sloppy——缺口只多不少。
- 全文语料不在仓库；重跑按本页检索式+种子（文献抽取20260918，治理同批）。
- U1覆盖格：本页即“数据治理”行的证据输入，权重/阈值仍待定。
