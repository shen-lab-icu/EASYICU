# Patient / Cohort / Cross-DB 共享 ECharts 可视化收口

> 日期：2026-07-28 EDT  
> 模块：web  
> 任务：`PATIENT-CROSSDB-VISUAL-PARITY`  
> 结论：三个审阅页已共享本地 ECharts 6.1.0 渲染内核；数据准备、临床语义和页面状态仍由各 route owner 持有。

## 用户问题与设计决定

用户提出大部分图表应统一使用 ECharts，并要求不同模块能统一的尽量统一。实现采用“共享渲染合同 + 页面语义 owner”，没有把所有内容强行改成图：

| 页面 | 数据关系 | ECharts 表达 | 保留的非图形表达 |
|---|---|---|---|
| Patient | 单实体不规则临床时间序列、干预阶梯、跨实体比较 | line / step line、阈值、精确 tooltip | 模块覆盖、特征状态、原始表预览 |
| Cohort | Kaplan-Meier 生存概率 | step line，固定 0–100% 概率轴 | 样本量、事件数、效应量和解释 |
| Cohort | SOFA-1 → SOFA-2 转移 | heatmap；25×25 精确模式带双轴 dataZoom | 精确单元格 count/% 的语义表兜底 |
| Cross-DB | 数值特征聚合分布 | 多来源 density line，不平滑、不伪造中间点 | 来源样本量、观测范围、桶数表 |
| Cross-DB | 分类特征百分比 | grouped horizontal bar，0–100% | 精确百分比和缺失状态 |

质量、覆盖、来源状态、精确查找表继续使用 DOM；这些信息用卡片或表格更准确，不为了“统一”而变成难读的图。

## Owner 与公共合同

- `src/easyicu/webserver/static/js/screens-viz-echarts.js`
  - 唯一共享 owner：本地 ECharts 6.1.0、统一色板/坐标轴/legend/富文本 tooltip、SVG renderer、`ResizeObserver`、按 route owner dispose、语义 fallback。
  - 公共合同：`window.EU_ECHARTS`。
- `src/easyicu/webserver/static/js/screens-viz-patient-charts.js`
  - Patient 保留不规则时间轴、干预阶梯、阈值和多实体比较的临床 option 语义。
- `src/easyicu/webserver/static/js/screens-viz-cohort-charts.js`
  - 新建 Cohort chart owner，持有 KM 与 SOFA 转移矩阵 option。
- `src/easyicu/webserver/static/js/screens-viz-crossdb-charts.js`
  - 新建 Cross-DB chart owner，持有数值密度与分类百分比 option。
- `src/easyicu/webserver/static/js/screens-viz-crossdb-results.js`
  - 只负责结果浏览、筛选和状态，把图表渲染委托给 chart owner。
- `src/easyicu/webserver/static/css/cohort-charts.css`
  - Cohort 图表样式 owner；`cohort.css` 已移除旧图表网格和无调用的 Patient selector。

边界结果：

- shared core 206 行；
- Cohort chart owner 300 行；
- Cross-DB chart owner 267 行；
- Cross-DB results owner 474 行；
- Cohort route CSS 497 行，chart CSS 55 行；
- 共享 `screens-viz.js` 为 4,815 行，本轮没有继续把 route chart 逻辑堆进去。

## 数据真实性

- 没有创建新的演示数字，也没有改变后端 payload。
- Patient 继续消费 Patient Review 的有界实体轨迹和 lazy feature 数据。
- Cohort 继续消费 reviewed aggregate survival / SOFA transition payload。
- Cross-DB 继续消费真实 prepared-export 的 KDE/分类聚合结果。
- ECharts 只替换绘制层；样本量、单位、范围、类别、事件数和精确值仍来自原 API。
- ECharts 不可用或渲染异常时 fail closed 到语义 SVG/表格；异常 chart 会 dispose，不留下半初始化实例。

## 自动化验证

完整相关 WebApp 修改域：

```text
286 passed, 1 warning in 15.47s
```

覆盖：

- shared renderer version / SVG lifecycle / resize / dispose；
- shared renderer 失败时 dispose + semantic fallback；
- Cross-DB density 两来源、真实单位、`smooth=false`、富文本 tooltip；
- Cohort KM step curve、0–100% y 轴；
- Cohort heatmap 与 13+ bin dataZoom；
- owner presence + foreign-route absence；
- static wiring/cache-bust/package；
- Patient/Cohort/Cross-DB source、API、job continuity 与 repository contracts。

附加门：

- 6 个相关 JS `node --check` 通过；
- `git diff --check` 通过；
- `GET /api/health` 返回 `{"status":"ok"}`。

## 真实浏览器桌面验收

运行页：`http://127.0.0.1:8765/`

- Cross-DB 分布页：选中特征主图 1 张，ECharts SVG 1 个、canvas 0、fallback 0；6 个来源 legend 与来源表保留；document overflow 0。
- Cohort MIMIC Demo：140 entities、19 modules、279 comparable features。
  - KM：ECharts SVG 1 个、fallback 0、overflow 0。
  - SOFA 转移：ECharts SVG 1 个、fallback 0；14-bin 标签模式正常。
  - `逐分 25 分`：560 px 精确矩阵与双轴缩放合同正常，overflow 0。
- Patient MIMIC Demo 时间序列：14 个 chart slot、14 个 SVG、fallback 0、overflow 0；首图 ARIA 明确为 12 个有界观测，并提示 hover 获取精确值。
- 最终浏览器 console：0 error / 0 warning。

## 未扩大的范围

- 本轮没有改 registered 多导出长请求的同步/异步产品合同；该问题仍需独立决定 timeout/cancel/job 模型。
- 没有把覆盖、质量、来源和精确表格强制迁成 ECharts。
- 没有 push、开 PR 或处理共享工作树中的其他并行修改。
