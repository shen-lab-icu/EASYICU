# Pi 文献检索、计划引用与右侧预览验收

> Task: `PI-LITERATURE-EVIDENCE-UX`
> Date: 2026-08-11
> Branch: `fix/pi-workspace-review-20260809`

## 结论

Pi Copilot 现把两类来源明确分开：

1. Research Agent 在 Planner 之前生成并保存的 `LiteratureBundle`，用于研究设计与方法依据；
2. 用户在当前对话单次授权后，由既有 Idea Mining owner 执行的 PubMed 元数据检索。

两者均不得被模型从记忆补写。文献设计依据也不等同于患者/结果 EvidenceStore 证据，不会因此让分析结果自动具备可报告性。

## 实现

- 新增 path-free `literature_evidence.json` Web 投影，显示检索状态、文章、真实来源链接及 Plan step → citation key 映射。
- 新增 `easyicu_inspect_literature` 只读工具和 `easyicu_search_literature` 单次授权工具；后者复用 Idea Mining PubMed owner，不保存全文、不读取患者行、不调用外部 LLM。
- Planner 的 `primary`、`secondary`、`sensitivity` 步骤在预先文献包非空时，必须绑定至少一个该包中的 exact citation key；未知 key 或遗漏绑定均 fail closed。`auxiliary` 步骤不强行附会文献。
- Guided Pi 对话实时显示检索工具活动；文章资源和计划引用均可点击，在右侧预览标题、期刊、年份、PMID/DOI、相关性与证据边界。
- 文献 UI 由 `screens-guided-pi-literature.js` / `guided-pi-literature.css` 独立持有，没有继续向 catch-all owner 堆叠。

## 真实 Web UAT

在 `http://127.0.0.1:8765` 的真实 Pi 会话中启用一次 PubMed 检索授权后：

- 对话实时显示“正在调用 检索 PubMed 文献”；
- owner 执行 3 条预先生成的查询、4 次网络调用，返回 12 篇 PubMed 文章；
- 回执明确为 `searched`，并声明未保存全文、未返回患者行、未使用外部 LLM；
- 点击文章后，右侧出现真实 PubMed 元数据和可打开的来源链接；
- 当前 Idea/Plan 未被此次检索静默修改，分析也没有继续运行。

截图：

- `output/playwright/20260811_pi_pubmed_search_preview_1440.png`
- `output/playwright/20260811_pi_literature_evidence_preview.png`

浏览器桌面检查：1440×900 下 document、聊天 panel、workflow、steps、preview 均无横向溢出；console 0 error / 0 warning。

## 当前 E1 的诚实状态

当前既有 E1 `run_20260811T112708_f426bc` 是在新合同之前生成的 Plan。它的投影如实显示：

- `status=curated_only`
- 9 条预置参考文献
- 12 个 Plan steps
- 0 个已绑定步骤（`mapping_status=not_bound`）

没有为旧 Plan 事后回填引用。新合同只作用于之后重新生成的 Plan。

## 定点验证

- 相关 Web/Pi/Research Agent focused suite：210 passed；
- 后续提示词与静态资源 delta：13 passed；
- 后续 Planner/方法合同 delta：84 passed；
- Ruff、Node syntax、`git diff --check` 全绿。

遵照 E1 开发测试策略，本轮未运行全套 exact-head CI，也未启动 Canonical9 正式 Provider batch。当前 E1 仍停在 3/7 人工 Plan 审阅门，未代替用户批准、拒绝或续跑。
