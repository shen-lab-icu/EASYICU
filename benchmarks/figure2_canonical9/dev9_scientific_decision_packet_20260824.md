# Dev9 一次性科学决策签字包

决策编号：`DEV9-SCI-20260824-V1`

机器可读决策包：`dev9_scientific_decision_packet_20260824.json`，SHA-256 `82d6df250b5a5bfdf3254205cc6497164976c63657a82a660200f0eadedea769`。

这份签字只授权 Dev9 的 `analysis_only` 开发方案，不授权论文结论、临床应用、部署、Qualification12 或 Held-out27。已发表文章只用于核对研究设计、统计方法与呈现完整性，不作为数值标准答案。

## 建议一次批准的方案

| 题目 | 建议方案 | 不能证明什么 |
|---|---|---|
| E1 | 全队列报告 Sepsis-3 患病率与绝对死亡率；调整关联改为 24h landmark，调整 age、sex、charlson_first；保留全队列、非再入 ICU、柔性函数形式敏感性 | 不是脓毒症因果效应；无 patient identity 时不是患者级患病率 |
| E3 | 全队列报告 KDIGO 分期、死亡率和 LOS；24h landmark 后做分类型 KDIGO 调整关联；增加 ordinal-linear 与 creatinine-only 定义敏感性 | 不是 AKI 因果效应；LOS 趋势不是治疗效果 |
| M1 | 全队列单独呈现胆红素测量/未测量；24h landmark 后在有测量者中用 bili_max 样条，调整 age、sex、charlson_first；增加线性和 bili_first 敏感性 | 不能外推到未测量者；不假设 missing at random |
| M2 | 保留患者组 80/20 held-out；新增 10 次封存种子的患者组重复拆分，汇总 AUROC、Brier、校准截距/斜率；保留 complete-case 轴 | 仍不是外部或时间外验证，不能主张可迁移性或临床部署 |
| M3 | 外部复现缺失记为证据边界；只跑零 Provider 的确定性图件/报告 suffix | 不命名稳定生物学亚型，不主张外部可重复 |
| H1 | 现 Cox 只保留为 PH 诊断；新增调整协变量与 `log1p(time)` 交互的 extended Cox，通气系数保持常数；增加 27 天 RMST 差；不做无依据 IPCW | 仍是预后关联，不是通气因果效应 |
| H2 | 冻结当前 verified-non-use 不可得的 fail-closed 结果 | 不产生虚假对照组或因果估计 |
| H3 | 冻结当前无内部 BIC 最优解的 fail-closed 结果 | 不强选 K、不命名轨迹亚型 |

E2 不需要新科学方案：已执行的 RCS 与线性模型本来就是独立 functional-form sensitivity，之前只是成熟度门禁漏读。

H1 也不需要 IPCW：当前固定 28 天结局在 94,458 个 source stays 中有 94,452 个具备支持，非事件主要是第 28 天行政删失，不是一个已识别的失访机制。

## 签字方式

请在当前对话回复下面这一句即可：

> 我批准 DEV9-SCI-20260824-V1 作为 Dev9 analysis_only 开发方案；不授予论文、临床或部署权威。

收到后会把该回复绑定到本决策包摘要，并继续：通用 typed owner 修复 → 受影响题的零 Provider deterministic replay → 九题重新 shadow review → 一个 exact-HEAD image → 一次 full exact-HEAD CI。不会提前运行 Qualification12 或 Held-out27。

## 方法对照来源

- E1：[Amsterdam UMCdb Sepsis-3 epidemiology](https://pmc.ncbi.nlm.nih.gov/articles/PMC11192388/)；重点参考操作定义、分母和感染/器官功能障碍时间关系。
- E3：[KDIGO stage and outcomes](https://pubmed.ncbi.nlm.nih.gov/30819553/)；重点参考分期梯度、死亡率与 LOS 的分层呈现，不复制其 OR。
- M1：[SOFA components and mortality](https://pmc.ncbi.nlm.nih.gov/articles/PMC9322581/)；重点参考首 24h 组件定义和结论边界，不把未测胆红素视为正常。
- H1：[time-varying ventilation intensity](https://pmc.ncbi.nlm.nih.gov/articles/PMC7906666/)；重点参考时间变化暴露/效应处理；其通气强度研究与本题二元首次通气定义并不相同。
