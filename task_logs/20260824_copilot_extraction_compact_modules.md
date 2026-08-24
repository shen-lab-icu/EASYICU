# Copilot 数据抽取预览精简

- 日期：2026-08-24
- 分支：`codex/easyicu-unified-product-20260823`
- 基线：`956fed47ead4c48021c97fd56643abfafe794cec`
- 提交：本日志所在 commit

## 用户反馈

Copilot 右侧原生数据抽取预览在完成数据连接后仍先显示大号“推荐抽取”卡，重复解释默认配置并挤压特征模块首屏。用户希望直接展示特征模块。

## 调整

- 只修改 `screens-extraction-embedded.js` 的 Copilot 适配层，不改 Data Extraction 核心 owner 的推荐配置或执行逻辑。
- 嵌入视图移除大号推荐卡和“需要更多控制”分隔器，直接展开配置。
- 将“特征模块”移动到队列与导出之前。
- 从既有推荐卡的真实状态投影一行“当前抽取设置”，保留队列、模块、概念摘要和同一个 `recommended` 执行按钮；按钮文案简化为“开始抽取”。
- 独立 `#extraction` 页面继续保留完整“推荐抽取”卡。
- 新样式只写入 `guided-pi-preview.css`，没有污染 `extraction.css` 或 catch-all CSS。

## 验证

- fail-first 合同先红后绿。
- `git diff --check`、Node syntax、Ruff 通过。
- 聚焦回归：`85 passed, 5 warnings`，覆盖 Copilot extraction workspace 与静态路由合同。
- 浏览器：当前 `#guided` 本地目录连接流程正常；独立 `#extraction` 演示模式仍能看到完整推荐卡与特征模块；console error 0；1814px 视口下 body/html 横向溢出均为 0。

## 边界

本次只改 Copilot 嵌入呈现顺序，不改变推荐队列、六个核心模块、概念选择、导出格式、目录选择、授权或抽取 payload。未选择或读取真实患者数据。
