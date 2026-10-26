# 完整示例：MIMIC-IV demo

公共来源：MIMIC-IV demo。这是拟议的教学计划，并非已记录的运行；不声称有结果数值或已成功执行。

入 ICU 满一天仍在院的患者中，早期乳酸分组是否与此后的院内死亡相关？

## 供您审阅的计划

成人；以入 ICU 为时间零点，24 小时为固定时间点；按前 24 小时最高乳酸分为低于 2、2 至低于 4、至少 4 mmol/L，以最低组为参照；结局为该时间点后至出院的死亡；调整年龄和性别，重复住院按患者计算聚类稳健区间；描述暴露不明者并将其排除于模型之外。

## 预期文件

- `table_one.csv`
- `cohort_flow.csv`
- `measurement_audit.csv`
- `absolute_risk.csv`
- `adjusted_association_estimates.csv`
- `key_metrics.csv`
- `caveat_flags.json`
- `report.md`
- `manifest.json`
这些是预期文件，并非已记录的结果；只有审阅后的运行成功才会生成。

若演示数据缺少所需观察、事件或独立患者，系统将记录限制。请审阅修订后的计划，不能补造输出或将演示当作真实数据验证。
