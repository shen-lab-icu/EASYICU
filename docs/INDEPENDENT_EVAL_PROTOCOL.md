# 独立评价协议（骨架，Track 4）

只定口径，不跑正式实验，不宣称任何通过率。权威口径见
`docs/EASYICU_PRODUCT_AGREEMENT.md`；U1–U4 未决事项不得在本文件或代码中静默补成共识。

- 预锁定：每题锁定题目 ID、版本、评价方式、分母 ID、允许干预；见
  `evaluation/independent_eval_harness.py` 的 `HeldOutItem` /
  `PreregistrationLock`（`digest` 绑定锁定集合）。
- 分母：预注册题目数即分母；系统缺陷计入失败分母，不许排除
  （`SYSTEM_DEFECT_COUNTS_IN_DENOMINATOR`，`DENOMINATOR_EXCLUSIONS == ()`）。
- 失败全保留：`FailureRecord.retained`恒为真，`FailureLedger`只追加、无删除接口。
- 三类恢复分别计数：运行内恢复 / 开发修复后恢复 / 后续复用
  （`RecoveryCounts` 三个独立计数；开发修复后恢复不得记为自主恢复）。
- U4 待用户定：期刊质量由谁判定（拟投期刊、检查表、独立角色、是否盲评）
  未决；`request_journal_quality_adjudication` 只抛 `PendingU4Decision`，
  绝不在代码里代替决定、不冒签独立评价。
