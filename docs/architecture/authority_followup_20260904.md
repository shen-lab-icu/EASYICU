# 五项后续复核与修复

复核起点：`codex/authority-closure-20260904@9886df96b32f7112173862cc1d1c035e580aba8b`。本轮功能候选：`ab2fb9485312d721418d26df94e083d8009904ee`。后续证据提交不改变功能代码。

用户提供的三项边界问题均可复现。新增负向探针最初为 5 failed、1 passed：补充图在步骤级和最终检查均漏检，Discovery 接受无关 JSON 或同 ID 改写候选，关联 owner 接受未实现的 table 图型。初始探针见会话输出；对应回归现已入库。两项绘图建议也成立；迁出 pipeline 本身没有使渲染参数可配置。

## 1. 补充图必须绑定实际产物

提交 `4b02118`。`figure_plan_binding` 按 `(figure_output, placement)` 分组；同一输出拆成主图和补充图时，使用 `supplementary_output_files` 指向独立图件，再读取同 stem 的合同。图件、合同必须存在且为安全的当前输出文件；图型、角色、面板 ID 和 typed 来源组合继续精确比较。整个输出均为补充图时，可以继续使用原 product slot。

landmark renderer 原来只导出源表并列出 supplementary panel IDs，现在还实际导出 supplementary PNG/SVG/PDF/TIFF 与合同。主图保留两条主要结果曲线，稳健性覆盖和测量可用性在补充图中。回归证明：合法拆分通过，补充图图型被替换、来源不符、合同缺失均拒绝；补充角色不会被强迫塞回主图。

## 2. Discovery 证明候选归属及转换关系

提交 `c45c3b7`。Discovery packet 升为 v5，新增 frozen `DiscoveryCandidateSource`：来源类别、原始候选身份（含可选 executable ID）、原始候选内容 SHA256 和可重放的转换声明。构造和使用时均验证成员关系；文件 digest 与 JSON 解析来自同一批读取字节。

来源只接受受支持的 dry-run/longitudinal/Web schema。无 schema 的历史 Discovery ledger/records 保留结构化兼容读取，仍要求候选唯一且内容一致；任意 validation schema 即使塞入同名 ledger 字段也不接受。Web 必须是 `easyicu.web_idea_mining/1`。

Web 的纯字段映射和 readiness overlay 归到 `discovery/source_provenance.py`，Web 调用同一 owner。校验先定位源文件中的原始 idea，再重放明确的映射，最后比较完整派生候选。来源 evidence、pre-experiment 必须匹配原 run；readiness 概念映射核对已存在 feasibility 文件的字节摘要、run/idea 身份及概念绑定；prior-art 映射核对已有文件。反向依赖 Web 的科学 owner 没有被引入。

这证明 proposal 的来源关系，不授予 execution readiness、文献创新性或科学计划审批。原 Web readiness、研究者确认和计划执行审批门继续独立生效。v4 旧包需从可验证来源重建 v5，不能继承旧确认 seal。

## 3. 关联绘图 owner 必须能实现声明的设计

提交 `ab2fb94`。调整关联和关联总览 owner 在认领前核对面板数量、图型、角色、来源组合及 placement。forest 结果不能被声明为 table，event-rate 与 estimates 来源不能互换。不兼容设计返回 `unsupported_planned_figure_design`，不以输出 metadata 冒充实现。

支持的面板声明传入 sandbox entrypoint，由运行时再次验证，并用于实际导出合同。负向测试覆盖 table 声明与来源互换，正向测试对实际 SVG/合同运行计划绑定检查。registry 的已有 capability/receipt/Coder 路由继续决定拒绝后的合法路径，没有新增无条件 Coder fallback。

## 4. 科学完整性与视觉建议分开

提交 `afe78e0`。图型数量和通用 bar/forest/heatmap 提示进入 `article_figure_strategy_design_advice`，不再使 strategy complete 失败；计划科学审查也不再把图型数量记为 major finding。display suite 中通用图型、统一面板数量、角色多样性及内容类别数量的经验阈值改为 `display_design_advice`，并进入 audit projection。

必要科学角色、确切来源、角色允许的图型、未知图型、声明的 Table 1、绝对风险信息、claim boundaries 和其他 evidence/publication 条件继续阻断。`publication_authorized()` 的科学条件合取没有删除。回归使用全部必需角色已覆盖、但只有两种通用图型的合同：策略可完成；删除必需角色或把图型改为未知仍失败。

## 5. 计划中的展示参数实际影响导出

提交 `ab2fb94`。沿用 `AnalysisStep.figure_panels`，增加可选 `presentation`，由 frozen `FigurePresentationSpec` 约束。已接入调整关联、关联总览和 prediction publication renderer；其他标准 executor 若未声明支持，会明确拒绝带新参数的认领，避免静默忽略。

```json
{
  "layout": "grid",
  "width_mm": 320,
  "height_mm": 230,
  "font_size": 15,
  "font_family": "sans-serif",
  "palette": "colorblind",
  "legend_location": "outside bottom"
}
```

布局支持 row/column/grid；字体族、配色和图例位置是明确的有限选项。同一图件的非空 presentation 声明必须一致。调整关联和关联总览代码生成器传递这些参数；预测 renderer 读取本计划的预测面板配置。实际图件改变 gridspec、尺寸、字体、配色和图例，并记录显示配置。预测图的 performance source CSV 也纳入数值投影；缺失指标不再以 0 冒充 AUROC。

工程验收使用同一组预计算合成表，生成论文版和汇报版：预测图横排/网格，关联图横排/纵排。图内数值投影及两个版本的源表 SHA256 完全一致；12 个 SVG/PDF/PNG 导出经过文字碰撞/裁切检查，零 findings。Agent 直接查看四种组合，发现初稿图例压线后改成图外图例，并检查了字号缩放；这不是独立人工审阅。关联 renderer 额外输出 TIFF。

- [论文版预测图](../evidence/authority_followup_20260904/figure_exports/paper/prediction.png) · [汇报版预测图](../evidence/authority_followup_20260904/figure_exports/presentation/prediction.png)
- [论文版关联图](../evidence/authority_followup_20260904/figure_exports/paper/association.png) · [汇报版关联图](../evidence/authority_followup_20260904/figure_exports/presentation/association.png)
- [源表、投影及导出 SHA256](../evidence/authority_followup_20260904/figure_exports/verification.json)

复现使用 [verify_figure_presentation.py](../../tools/verify_figure_presentation.py)，为其指定一个新的输出目录。它只创建工程 fixture，不调用 LLM、不拟合模型、不启动研究运行。这里的合成图不是研究结果或论文证据。验收覆盖上述 renderer 和版式；其他 family 的显示能力与任意自定义组合不能由这四张图推断。

## 验证与架构边界

- [combined.log](../evidence/authority_followup_20260904/combined.log)：12 个相关测试文件，284 passed、5 warnings；没有以 xfail 隐藏新失败。
- [publication_integration.log](../evidence/authority_followup_20260904/publication_integration.log)：显式启用 integration marker 后，6 passed、273 deselected。
- [adjacent.log](../evidence/authority_followup_20260904/adjacent.log)：中间相邻检查 200 passed，范围与 combined 有重叠，不能相加作为去重总数。
- [architecture.log](../evidence/authority_followup_20260904/architecture.log)：五项廉价架构门的最终结果；这不是 full exact-head CI。
- [validation.json](../evidence/authority_followup_20260904/validation.json)：检查命令、功能 SHA、文件摘要、验收范围和主工作区保护状态。

模块图从 640/2,632 增至 642 modules/2,642 edges，仅新增 source provenance 和 presentation 两个 owner。顶层模块仍为 34，循环模块和 SCC 均为 0，pipeline 仍为 7,062 LOC。基线逐项记录两个 owner 的准入原因并保留上轮审阅；没有放宽任何 god-module LOC 上限或 cycle 约束，也没有进一步拆分 pipeline。

主 checkout 仍为 `main@89cb7ed`，87 个并行 dirty paths 和 PID 32526 的 8765 服务保持原状。正式候选仍为 clean `802bcf5`。本轮没有合并、部署、真实数据研究、Provider 调用、正式实验或重复全量 CI；此前审阅中的其他长期架构边界继续见 [上一轮报告](authority_closure_review_20260904.md)。
