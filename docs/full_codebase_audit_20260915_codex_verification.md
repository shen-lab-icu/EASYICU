# 2026-09-15 全量审阅问题：独立复验与修复回执

## 结论与候选

完整读过原审阅文档后，按源码的调用路径、发射/消费属性和异步回写边界独立复验了 **3 项 P1、19 项 P2**。3 项 P1、18 项可达 P2 成立；**P2-9 是不可达的旧链路，降级为死代码清理**，不能计作当前用户可触发的坏控件。所列修复、CSS 清理及已获用户确认的退役文件删除已完成。

- 开始基线：`c3c1a81d14ecd25b1838754c012efed8f11186fc`。
- 分支保持 `codex/dev9-web-acceptance-20260906`。
- 验收源码：`7e55a1fecd88585bed1b4c85fba4278a8dee0b50`。本任务未执行 commit、push 或 Git 配置修改；共享工作区在验收期间被并发任务推进至该提交，本任务未回退或覆盖它。最终源码检查前 tracked 工作树干净；本回执另行写入。
- 并发新增的正式计划 transition guard 释放逻辑和 Figure 5 延迟配置 matplotlib 样式，均已回读并纳入最终检查。
- 独立逐项复验先形成 **combined focused checks**；随后 Codex 在同一可执行提交上完成最终 xdist 全量，见末节。没有调用真实 Provider、提取真实数据库或改写既有研究产物。

## 检查简称

| 简称 | 实际检查 |
|---|---|
| A | `tests/js/audit_regressions_20260915.test.js`：14 组行为测试，覆盖恶意文本、未知 finding、属性消费、失效定时器、乱序成功/失败、会话 A→B→A、原 run/digest 绑定及非法 digest 拒绝。已登记入 `tools/run_js_contracts.py`。 |
| W | 静态路由、study context、键盘可达、设置、UX、onward paths、owner 清单，以及 Copilot retirement/static/message-actions/plan-reentry/retry-run-coordinate 聚焦 pytest。 |
| T | `tests/core/test_audit_tool_truthfulness.py`：skip-smoke 禁止抽取/绘图、失败和缺失数据报告、成功/部分失败/空任务退出码、六库完整性、NaN/Inf/重复/缺组/非法 CI/零分母和实算图注。 |
| C | pipeline config 和 submission profile 测试；相邻 literature-design authority、execution identity、review-resume/egress、pipeline authority、development execution sample 测试。 |
| B | Chromium + 127.0.0.1 静态合成测试页；使用实际 owner 模块和当前源函数，无真实服务/患者数据。点击控件、证据/图表定位、键盘焦点、CSS computed style 与截图。 |

## 逐项对账

下表行号对应验收源码；`static/` 均指 `src/easyicu/webserver/static/`。

| 条目 | 独立结论、修复位置与行为 | 检查 |
|---|---|---|
| P1-1 | 成立。`static/js/screens-extraction.js:718` 的 SSE 文件名经过既有 `escHtml`，磁盘文件名中的 HTML 不再进入 DOM。 | A、W |
| P1-2 | 成立。`static/js/screens-guided.js:659,1061,1063` 对 JSON、question frame、plan 键值在渲染端统一 `esc`；保留原始结局/比较文本用于研究设计和 slot。 | A、W、B |
| P1-3 | 成立。`tools/build_top_level_mechanism_qc.py:612` 在 skip 分支写完三份 dictionary/readiness/status CSV 后返回，不再访问 smoke 专属列或生成 smoke 报告。 | T |
| P2-1 | 成立。`static/js/screens-viz-cohort.js:257,285,297` 消费 `cohortFeatureScope`、`cohortSofaMatrixMode`、`cohortFeatureModule`。第三项实际发射的是 `data-cohort-feature-module`；没有在渲染端伪补 `data-state`。 | A、W、B |
| P2-2 | 成立。`static/js/screens-extraction.js:446` 用现有 continuity generation 捕获票据，回调复核票据、demo 模式和 running 状态；`screens-extraction-job-continuity.js` 仅暴露已有 generation 的 capture，不增加全局计数。原文建议中的 `exportJobSeq` 在当前文件并不存在。 | A、43 组 JS |
| P2-3 | 成立。`static/js/screens-extraction.js:202` 优先重绘已挂载的 embedded owner，使 Copilot 内嵌抽取收到 SSE 状态。 | A、43 组 JS |
| P2-4 | 成立。`static/js/screens-extraction.js:227` 注册完成回调改用可见 owner 的 backgroundRepaint；用户处于其他路由时不重建全壳。 | A、W |
| P2-5 | 成立。`static/js/screens-agent-render.js:703` 三个主 lane 互斥分配，剩余 finding 进入 remainder；`:738` 总数与 `:741` 系统/运行列表包含所有剩余项。未知 reason/route 原样保留，也避免重复计数。 | A、W、B |
| P2-6 | 成立于 run-files 的内联 fallback renderer；常规 pipeline artifact 路径本来由 preview 消费。`static/js/screens-guided-pi-run-files.js:262,278` 补 evidence/display 处理；`screens-guided-pi-preview.js:671` 复用原 preview API 与校验，携带原 project/run、artifact SHA、evidence SHA。图表只在同一 run section 查找。 | A、W、B |
| P2-7 | 成立。`static/js/screens-guided.js:3570` 将 `@folderopen` 接到既有 `showGuidedDraftSetup(label, 'open')`。 | A、W |
| P2-8 | 成立。`static/js/screens-guided.js:1095,1115,3773` 在 real 模式隐藏 demo 限制按钮、不进入 demo 空队列卡，strict/loosen 消费端也直接拒绝；不编造真实匹配数。 | A、W |
| P2-9 | **降级为不可达死链**。旧 `selectedGuidedRun` 首次为 null，仅在被它自己调用的函数内赋值；project memory 仅以 draft 调用。删除变量、`openGuidedRunReview` 与 `@reviewLocalRun` 臂，保留当前 `screens-guided-pi-run-files.js` 的真实 run review owner。删除项无新文件行号，见提交 diff。 | W |
| P2-10 | 成立。`static/js/screens-guided.js:2458,2476,2488` 复用 `gen`，打开项目时捕获 generation，在成功和失败回调都复核，旧请求不能绑新项目或报旧错误。 | A、W |
| P2-11 | 成立。`static/js/screens-guided-pi.js:1511,1556` 与 `screens-guided-pi-plan-actions.js:170,239` 捕获 project/session/selection revision，逐个 await 后及 catch 复核。正式计划在创建 job 前失效会释放 transition guard，job 创建后仍保留去重，避免重复 Provider 任务。 | A、W、C |
| P2-12 | 成立。`static/js/screens-guided-pi-preview.js:651` 所有有效 open 都递增现有 request ticket 并重置 loading；web/document 打开也能使旧 artifact 响应失效。 | A、W |
| P2-13 | 成立。`static/css/tokens.css:64` 定义 `--accent-2`，Pareto 渐变有效。 | B、W |
| P2-14 | 成立。`static/css/tokens.css:60` 根作用域提供 `--line`、`--ink-1`、`--mono` 别名，Cohort 不再依赖 Copilot 局部作用域。 | B、W |
| P2-15 | 成立。`static/css/tokens.css:63` 提供 `--good`→`--ok`，文献状态样式得到有效颜色。 | B、W |
| P2-16 | 成立。`tools/build_top_level_mechanism_qc.py:514,545` 报告按 readiness、错误/缺列、unsupported、absent、warnings 和实际 sample_size 生成；删除硬编码的全库 ready、13 列保留及源级裁决。 | T |
| P2-17 | 成立。`tools/run_openrouter_fullflow_validation.py:625` 仅非空任务且 n_failed=0 返回成功；失败任务保留在结果与分母。 | T |
| P2-18 | 成立。`scripts/r5_fig5_nature.py:49,104,124` 要求六库五组每格恰好一条、有限且合法的死亡率/CI及正参考分母；图注实算 normal-BMI 跨库 spread 与三个超重/肥胖组均 <1 的库数。标题描述观测结果，不把未调整比值写成保护效应。 | T、合成 Figure 5 四格式导出并检查 PNG |
| P2-19 | 成立。`src/easyicu/research_agent/orchestration/config.py:881` 对非 None 的 `planner_only`、`require_human_plan_review` profile pin 做一致性校验；历史未钉死的 profile 仍允许调用者选择。 | C |
| 指定 CSS 清理 | `static/css/tokens.css:57` 集中补足 shadow/font/sans/danger/teal/ink-5/hair-strong/muted 等别名；`static/css/agent.css:19` 加 title `:focus-within`。 | B、W |
| 退役文件决策 | 按用户明确回复删除 `static/js/screens-agent.js`；L1–L10 不再有可被重新加载的实现。同步移除仅依赖它的测试断言，混合测试保留共享 renderer/API/Copilot 合同；`tests/webserver/copilot/test_pi_history_retirement.py` 断言旧文件不存在。 | W、43 组 JS |

## 最终验收与局限

- 可执行提交 `7e55a1fecd88585bed1b4c85fba4278a8dee0b50` 的仓库正式 xdist 全量为 **19,395 passed / 84 skipped / 0 failed / 1,017 warnings**，耗时 36:05；独立 runner 文件 **51 passed**。回执为 `/Volumes/外置硬盘/easyicu_data/final_validation/7e55a1fec_20260915/pytest-xdist.log` 与同目录 `results.xml`。
- 同一提交的架构门禁 **5/5**，其中规模与预算守卫 **133 passed**；Ruff clean，JS owner contracts **43/43**。本回执是全量完成后新增的文档，不改变已验证的可执行树，可按仓库规则复用该父提交的 full exact-head 证据。
- 先前基线 SHA 上出现的 14 项失败均为子进程无法导入本地 `easyicu`：6 个失败簇的首项单跑和原 `loadfile` 相邻顺序均稳定复现；按 GitHub workflow 执行 `python -m pip install --no-deps -e .` 后，14 个失败节点全部通过。分类为本机 editable-install 环境差异，无共享状态污染、超时或代码回归。

- **587 passed, 1 skipped, 24 warnings**，默认 `-m "not slow"`；skip 是既有测试 `test_pipeline_authority_regressions.py:909`，注明由 review-resume 中的 host privacy audit 替代，该文件已纳入本轮检查。
- **43/43 JS harnesses 通过**，其中新增审阅回归为 **14/14**。Python wrapper 会重复执行新增 JS harness，这些数字不相加为独立测试总数。
- `pyproject.toml` 的 dev 依赖补入 `tabulate>=0.9`，供 QC Markdown 报告及实际报告测试使用；本地安装后该路径通过。
- 修改的 Python 文件 Ruff、修改的 JS 文件 `node --check`、baseline→验收提交 `git diff --check` 均通过。
- 浏览器：注入节点 0、脚本未执行；三项 cohort 状态 all/count/renal；3 个未知/runtime finding 全部保留；键盘 Tab 到 edit 按钮后 opacity=1；全部 15 个别名有 computed value，Pareto 渐变和边框有效；1280/1440 下无水平溢出。证据 API 实收 `project / original-run / summary / a×64`，图表定位为 `Table 1`。补齐测试页依赖后的路径无 pageerror。
- Figure 5 使用六库五组合成 fixture，故意令一库超重组比值 >1；图注正确显示 **5/6** 与 **1.50x**，SVG/PDF/TIFF/PNG 均导出，PNG 已目视检查。未重画正式研究图，亦未复核真实患者数据结论。
- 原报告第四节未被本次用户清单选中的 P3 观察项未宣称关闭。

验收日志、源码 SHA-256 与浏览器/图件 fixture 位于：

- [检查命令与结果回执](../output/playwright/audit20260915/validation-receipt.json)
- [Python 合并检查](../output/playwright/audit20260915/pytest-combined.log)
- [JS 契约清单](../output/playwright/audit20260915/js-contracts.log)
- [源码与静态检查回执](../output/playwright/audit20260915/source-verification.json)
- [浏览器截图](../output/playwright/audit20260915/browser-fixture.png)
- [合成 Figure 5](../output/playwright/audit20260915/figure_fixture/research_output/r5_obesity_crossdb/Figure5.png)

准确命令与逐文件测试范围已写入检查回执。进度页已同步，`tools/lint_progress.py` 检查 6 个 CURRENT.md 通过、0 warning；9 月 14 日旧结论原文移入 web/HISTORY.md。本轮隔离浏览器与静态服务器已关闭。

该目录是被 Git 忽略的本地工程验收产物，本文件保存逐项持久交接。
