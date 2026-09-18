# ICU-EHR 方法覆盖挖矿：协议、证据与进货决策

统计证据见 [`method_coverage_mining.json`](method_coverage_mining.json)（派生数据，4KB）。本文只记录协议、决策与边界；不复述数字。

## 协议（2026-09-18，可重跑）

- 检索：PubMed E-utilities，六库关键词（MIMIC-IV/III、eICU全称、AmsterdamUMCdb、HiRID、SICdb），去重得 5427 PMID；排除 Review/Editorial/Letter/Comment 等 51 篇；216 篇无摘要；得 5159 篇标题+摘要。
- Pass 1（确定性，免费）：40标签方法词表 + 软件词表，大小写不敏感正则，全量匹配。
- Pass 2（LLM，7路扇出）：696篇零命中长尾逐篇抽取（原文逐字phrase+受控label+三态verdict），不调外部API。
- 全文校准：随机100篇（种子20260918），PMC OA 75篇，73篇有Methods段，正则抽软件。
- 20篇人工自查：约15篇全对；SOFA/KDIGO作协变量会误触定义标签；MoE/DTR类新词在词表外。

## 由此做出的进货决策（已落地）

- P0：装 `shap/xgboost/lightgbm/numba/torch(CPU)`；新建 PSM/IPTW 确定性kernel。
- P1：新建 NRI/IDI、中介、竞争风险 CIF kernel；另有独立的 Gray 检验实验性 kernel（尚无 R `cmprsk` 对拍），并非 CIF kernel 自带的检验。R共享rails（probe/fingerprint/ hardened runner）+ time-varying Cox改道。
- P2：目标试验脚手架、AIPTW、序贯MSM加权、点干预g-formula、单阶段DTR估值（纵向g-formula与多阶段优化明确不做）。
- 首批 Tool Cards：lasso、SHAP、PSM、RCS（含复现诚实声明），checked-in JSON + loader。
- RCS kernel本身就是挖矿发现：21.1%文献用RCS查非线性，原货架无owner。
- 后续补入实验性 GEE、混合效应、GAM、Gray 检验和 nomogram 表格 kernel；其中 GEE/混合效应仍停放，nomogram 尚无图件渲染。

## 已知边界（下次挖矿前必读）

- 软件包真相在全文：同一样本Methods段68.5%提软件，摘要仅8.2%——摘要的包频率是下限，不能当真相。
- 原文语料（~10MB）不在仓库；重跑按本页检索式+种子可再生，数字会有PubMed收录漂移。
- 待定后续（需用户决策）：LLM/NLP抽取kernel（5.6%且在涨，需Provider架构）、贝叶斯方法、nomogram图件呈现、R的MatchIt系安装、纵向g-formula/多阶段DTR。
- U1覆盖口径未定：本页是“文献在用什么”，不是“我们宣称覆盖多少”。
