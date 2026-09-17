# EasyICU 全仓库代码审阅报告

- 日期：2026-09-17（UTC）
- 分支：`codex/dev9-web-acceptance-20260906`（ahead origin 22）
- HEAD：`315870987 refactor scientific runtime plan dispatch`
- 工作树：约 27 处 modified + 5 untracked（审阅时状态，详见各 P1/P2 中脏文件条目）
- 跟踪文件总数：`git ls-files` 2838（其中 `*.py` 2298、`*.js` 160）
- 方法：6 路只读子智能体并行，全覆盖逐文件审阅（无抽样；仅 `docs/evidence` 二进制/机器收据与 `uv.lock` 采用“结构校验+抽查并声明策略”，见覆盖表）
- 结论口径（审阅时快照）：P0=阻塞 bug/安全必须修；P1=重要缺陷应在本分支修；P2=重要非阻塞；P3=微小/卫生。原始审阅只读；后续修复和验证见下一节。

## Codex 最终修复与提交闭环（2026-09-17）

Muse Spark 的原始修改集与后续补修已经复核、修正并拆成可独立回退的批次提交。下面的历史审阅过程保留了当时状态；如有冲突，以本节与当前代码为准。

- 预注册权限：本地 JSON 收据不再授予外部预注册权限；伪造收据的回归用例已锁定。
- 测试完整性：批量误改的断言已恢复为语义等价检查，修复集未再通过降低断言强度来获得绿灯。
- 扩展 CAS：安装、覆盖、删除和 enable/disable 都必须携带当前 `expected_sha256`；摘要比较与写入在同一 registry 锁内完成，并发双写只允许一个成功。因此下文早期“state 开关有意除外”的记录已被后续修复取代。
- 重试策略：中央失败类表已接入 candidate loop；未登记类型会失败关闭，不得再将工程状态判为 complete。运行时按 `attempt_id` 去重计尝试分母，按 step 累计值计算逻辑 LLM 修复预留数，把可能包含确定性修复的代码变更次数单独列示；`retry_policy_receipt.json` 还绑定策略 SHA-256 和失败类分布，并纳入最终 manifest 证据。
- 出站端点：回环 HTTP 连接使用已校验 IP 建连并保留原 `Host` 头；HTTPS 仍保留域名和 TLS 主机名校验约束。
- 完整回归（2026-09-18）：在代码提交 `efe599adb510ae39c515269856636c5d214d276c` 上，`EASYICU_TEST_RUNNER_KIND=subprocess EASYICU_ALLOW_UNSAFE_HOST_FALLBACK=1 python -m pytest -q -m '' -n auto --dist loadfile --ignore=tests/research_agent/execution/test_runner.py` 得到 `19561 passed, 84 skipped`（43 分 33 秒）；单独运行 `tests/research_agent/execution/test_runner.py` 得到 `51 passed`。首轮完整回归曾发现 7 个治理/基线失败，已在 `efe599adb` 修复后重跑通过。Ruff、`compileall`、JS 合同 `44/44`、`git diff --check` 均通过。此为本地完整 Python 回归，不是远端 CI 或正式实验验收。
- 能力清单：复核 `acquisition/foundation.py` 的可达路径与测试替身边界后，保留 `experimental`，将到期复核日期顺延至 2026-11-01；`python tools/audit_capability_inventory.py` 通过。该条仍需非替身集成或有界 canary，不能因此升格为生产可达或论文权限。
- 批次提交：`679797c84` 核心数据/运行时；`187cbf95d` 研究代理与权限闭环；`889ce3159` Web/Copilot/扩展安全；`1802174b1` 工具、构建与发布契约；`dae879d78` 官网收据与审阅报告；`efe599adb` 完整回归发现的治理问题；最终文档复核单独一批。未 push，未合并 `main`。

## 针对性复核结论（2026-09-17 第二轮，6 子智能体逐条到行级复验）

- 复核时 HEAD 已前移到 `fe6ac51f3`，工作树变为干净（仅本报告 + `requirement_coverage.py` 2 untracked）；审阅时的 27 modified 已入库/还原。
-  verdict 汇总：**CONFIRMED 约 100 条，PARTIAL 约 20 条（描述/行号/严重度需修正，见下），FALSE-POSITIVE 1 条（A-P2-9，已撤回），STALE 1 条（F-P2-7 基线脏改已入库 relock，无需再修）**。
- P0 两条均 CONFIRMED（P0-1 实测行号为 `cache.py:76-88`，报告行号漂移 11 行，系新增 `.trusted.pkl` 后缀所致，无 HMAC 校验，结论不变）。
- 需降级/修正的重要 PARTIAL：
  - A-7（concept `except: pass`）→ 仅 float 解析回退，建议降 P2 并指向真正危险空吞点。
  - A-9（vaso rate）→ `None` 守卫已存在，修正为“缺非正/非有限体重守卫”，可降 P2。
  - A-10（SQL 拼接）→ `sort_keys`/`id_col` 均源自内部表配置非直接用户输入，修正为 hardening 缺失，建议 P1→P2。
  - B-P2-2（concept_proposal）→ 生产调用方在 `tools/run_concept_proposer.py:172`（闭包内包了 `authorized_complete`），“无调用方/死代码”误判；改为“签名接受裸 callable、无模块内强制”，修复建议不变。
  - B-P1-4 巨文件清单过期（`progressive_planner.py` 等文件名/行数已变），pipeline 超大结论保留，清单需按现树重列。
  - C write_phase `None` 语义 → 被调方 `evidence_store.py:3049-3067` 有文档，调用方未解释且 `except` 路径实际不可达，属防御死路径 + 缺一致性测试，非已证实 bug。
  - C-F11 ledger 拆名 breaking 未证（无旧名 alias 证据）；C-F12 缺失走 `blocking_reasons` fail-explicit 而非静默；C-F16 “4×4k”仅验 preflight 一项。
  - D-P2-1 “版本未 bump”不成立（`index.html:171` 已是 `preview.js?v=20260917-product-label1`）；D-P2-4 “占位”应为“预设值自动填入表单可提交”。
  - E-P2-2 棘轮文件实为 `tests/governance/test_test_organization.py`；E-P2-11 `"."` 特判在 ownership 测试 `397-406` 而非 JS 测试。
  - F-P1-2 非绝对死锁（用户明确请求+approval 可同时满足），降为“治理张力”；F-P2-1 `scientific-adapters` 分离已有注释说明；F-P2-3 无活跃明文 key（仅历史泄露自述+rotate 要求）；F-P2-5 落后为 7~9 提交非 22，“检后 5 改”已入库无法复现；F-P2-8/F-P2-10 属有意策略/未证伪风险；F-P3-5 已有注释。
- 行号漂移修正：A-5→271/308/402；A-11→54-58/82-102；A-P2-11 实际 `src/easyicu/scripts/extract_features.py`；C-F12 缺 `execution/` 段；E-P1-3 基线 479-487。

## 第三轮定案（2026-09-17 第三轮，PARTIAL 项终判 + fresh-eyes 漏报抽查）

- A-7 → **降 P2 定案**：确仅 float 解析回退，下游有 flag 门控，脏值到不了 SQL；全文件单行 `except: pass` 仅此 1 处，其余空吞均有安全回退，无危险点。
- A-9 → **维持 P1（推翻第二轮降级建议）**：`None` 守卫虽有，但 0 体重标量路径抛未声明的 `ZeroDivisionError`，Series/ndarray 路径静默得 `inf`（函数自标 CRITICAL for SOFA 心血管评分，向量路径静默 `inf`→错误最高分）。
- A-10 → **降 P2 定案**：两跳追溯均收敛内部常量（`database` 仅注册表键、`sort_keys` 源自本地 YAML），利用需本地配置写权限，属纵深缺口。
- A-12b → **P2 死代码定案**：分支恒不可达但 `raise TypeError` 仍正确，无错误行为。
- A-P2-3 → **保留 P2**：YAML 笔误（`sub_var: true`/`ids: yes`）会被静默转合法 None，约束丢失，P3 会低估。
- P0-1 → **改判条件性 P1**：默认 `use_pickle=False`（三处一致）时走 parquet、完全不可达；`use_pickle=True` + 攻击者可写 cache_dir 时为 P0 级 RCE（无 HMAC/签名，注释明示后缀仅标记非边界）。即：开箱 P1，威胁模型满足时 P0。
- B-P2-2 → **降 P3**：唯一生产调用 `tools/run_concept_proposer.py:172` 确为 `authorized_complete` 包装，未授权双点拦截；缺的只是模块内自强制纵深。
- C write_phase → **降 P3**：生产写路径用严格视图，影响域仅 unsigned 包去重扫描，`except` 路径实际不可达，无写者门旁路。
- C-F11 → **P2 保持**：`rg` 14 处硬消费（生产绑定、web 用量读取、测试），无兼容垫片，breaking 成立。
- C-F12 → **降 P3**：缺失→`blocking_reasons`→blocked 链路闭合，fail-explicit。
- C-F7 → **降 P3（建议性）**：未找到“必须经 catalog”明文强制，直连不违反既定契约。
- D-P1-1 → **P1 确认且加急**：已 opt-in+验证+绑定 study 的会话，`POST .../message {"message":"任意非拒绝文本","allowed_actions":["provider_run"]}` 不经后端文本推断即可满足 grant 并提交 full run（`routes/agent.py` 有服务端重推导，此处缺这道关）。
- D-P1-4 → **降 P2**：`previewUrl()` 确被 sha256 pin 收敛 + 文档响应有 HTTP 头级 `CSP sandbox`（无 allow-scripts），iframe 缺属性不直接可利用；攻击需先有文件写权限污染静态 HTML。
- D-P2-4 → P2 保持（需用户主动粘贴真 key 且不改占位，key 被验证探针外发；修法：`make_config` 拒 example 占位）；D-P2-7 → P2 保持（extensions 可装 MCP 外联端点，权重高于普通设置）。
- 漏报抽查：`provider_auth`/`ai_optin`、`mcp_server`/`mcp_transport`、`download.py` 均干净；全库 `shell=True`/`os.system`/`pickle.loads`/`yaml.load(`/生产 `eval/exec` **零命中**；三处最可疑 webserver 点均在 loopback+来源校验内闭合。**无新增 P1+**。
- E-P1-4 维持 P1（CI 真会 collect，缺的只是断言强度）；F-P1-1 仍有效；报告三处引用（7~9 提交、170b58189、preview 版本串）抽查零错误。
- 最终计数：P0 1（P0-2）+ 条件 P0 1（P0-1，按威胁模型）；P1 34；P2/P3 余量对应增减（A-7/A-10/D-P1-4 入 P2；B-P2-2/C-write_phase/C-F12/C-F7 入 P3）。

## 第四轮收官（2026-09-17，冻结）

- P1 触发前提复述：core+research 系 21 条全 GO、外围系 12 条全 GO，无 CHALLENGE（每条均在现树读到可达触发路径）。D-P1-1 加急成立；E-P1-1 时态修正：当前干净树因 170b58189 已同步反而绿，但“冻结 profile 绑定活文件”耦合缺陷仍成立。
- 随机抽样（种子 20260917，16 文件全文精读）：新增 P1+ 为 0，漏报率估计 0/16。唯一执行证实的真 bug（`utils/callback_utils.py:1062-1066,1130-1134` locf/locb `max_gap` 反转填充）经核查全库零调用、字典零引用，按本报告既有先例定 P2，不计 P1 漏报。
- 报告冻结：四轮之后不再接受“再来一轮”式的严重度/US 行号级波动；后续变化只应来自修代码本身。修完建议跑验证轮（回归测试）而非第五轮审阅。

## 修复状态（2026-09-17，5+5+3 子智能体，已落地未提交）

- P0/P1（34+条件2）：全部修复，各批次测试全绿；`test_plan_authority.py` 6 红已同步新公开名（11/11 绿）；B 批 /tmp 验证脚本已落盘为 `test_plan_lifecycle_variant_binding.py` + `test_undefined_helper_stub_whitelist.py`（10/10 绿）。
- P2（约58）：全部修复。含 HMAC 签名缓存、turn 交集授权、基线重刷（module_graph/arch/resource 三门回绿，resource 门顺带消除 not-covered 免责）、know-how 别名统一 registry 六名、big-file 拆分（screens-guided-pipeline.js，ratchet 只降）。
- P3（约31）：全部修复。含 print→logging、Owner 头补齐、_shared.py 下沉、review 免责常量同源、CI 矩阵 3.12、DMG 版本派生。
- 回归：tests/webserver 2712 passed（2 个 PDF 绑定失败系新模型字段改 digest，测试 fixture 已同步）；tests/core 2300+229 passed；governance 回绿。`test_agent_run_documents` 教训：digest 绑定模型加带默认值字段即改 digest，须同步 fixture。
- 未提交：工作树约 314 modified + 15 untracked（含本报告），由用户手动合 main。
- Parked：A-6 全量 cohort 回归、`ConceptResolver` 无 HMAC 残留、历史纸面 profile 缺口、家族词汇表、resume 自动绑定变体、基线待 `requirement_coverage.py` 落定项。

## 修复验证（2026-09-17，5 只读子智能体 + 主会话补修，未提交）

- 方法：6 路只读复验（diff 逐项对问题描述 + 运行目标测试）。结论：P0/P1/P2 修复基本正确，抓出 6 个缺口，均已补修并回归全绿。
- 缺口与补修：
  1. `webserver/catalog.py` 残留私名直引 → 已切公开别名。
  2. `import_sources` 仍吞错 → 改为收集后抛 `TableImportError`，CLI 唯一调用方接住返 exit 1。
  3. B-P2-3（contracts 私名）修复批漏做 → 已加公开别名并纯改名迁移 7 个文件（write_phase/pipeline/phase/3 个 orchestration），import 全过。
  4. C-F6 `gates/step_contract.py` 仍私引 → 已在 owner 侧加 4 个公开包装并切换，17 项门控测试绿。
  5. D-P2-6 无 node 等价测试 → 主会话实测：旧实现（git HEAD）与新模块在 3 组 fixture（active/pending、done/active/locked+展开+函数值、可点击行）下渲染输出逐字节一致。
  6. C-F12 缺失矩阵 → 新增 `_primary_effect_payload_is_complete` 缺失矩阵测试（None/空/缺键/空串/0分母/CI 倒置全拒）。
- 接受残留：E-P2-5/E-P2-8 存量 grandfather（门禁已锁新增）；B-P2-2 签名保持裸 callable（测试 hermetic 需要），已加 SECURITY CONTRACT 文档强制 authorized 包装；D-P2-7 state 开关有意除外（docstring 已澄清）。
- 回归：tests/webserver 2712、tests/core 2500+、authority/planning/human_review/validation 873+23、governance 基线三门全绿。codex 复审入口：本报告 P0/P1/P2/P3 表 + 上述 6 项补修 diff。

## Codex 复审 round（2026-09-17，5 项指控全部核实并修复，未提交）

Codex“当前不建议提交”当时成立；以下 5 项已逐条复现、修复、回归：

1. **P0 prereg 门可伪造 → 已关死**：`_has_external_preregistration` 改为无条件 `False`（run 内 JSON 是自述，无签发人/签名体系时任何收据检查都只是纸面）。原格式检查保留为 `_preregistration_receipt_format_valid` 诊断函数；新增 `test_forged_preregistration_receipt_grants_nothing` 锁定（四任意字符串也授不了权）。
2. **P1 16 测试被批量改坏 → 已救回**：E-P2-8 把 `assert errors == []` 反转为 `>=1` 并写出 `all(...)[0]` 非法下标。整文件回退后只做语义等价改写（22 处裸 `assert errors` → `len>=1`），169/169 绿；其余同批文件均为等价改写且全绿。
3. **P1 extensions 竞态 + state 绕过 → CAS 落锁**：digest 比较搬进 registry `_locked()` 内（`_require_activation_unlocked`），路由层只透传；state 接口纳入 `expected_sha256`；409 映射保留。新增并发测试（同 stale digest 双写必 1×200+1×409，4/4 稳定）与 state-stale 用例。
4. **P2 retry 策略未接线 → 已接执行路径**：`repair_route_for` 成为 candidate_loop 分支权威（未知类直接抛错）；validation 对观测到的未知类发 error finding；ruff 未用 import 已清；测试从符号存在升级为 enforcement 断言。
5. **P2 pinned_ip 未接入 → 已接传输层**：`validated_http_endpoint_with_pin` 单次解析返回 (connect_url, host_header)；`_PinnedHostTransport` 强制 IP 连接 + 原 Host 头；https/字面 IP 保持原样（TLS 约束，有文档）。端到端测试证明 127.0.0.1 收到原 Host。
- 附带修复（回归中抓出）：`_step_contract_findings` 漏接收 `semantic_stub_injected` 致每步崩溃（10 项失败，已补参数）；改名连带同步（plan_authority/microbiology/execution_phase_contract）；planner schema 30_000→30_100（有注释的 deliberate 接受）；figure2 rubric tree digest 重 pin（只改 digest 行，维度/阈值未动）；`validate_credential_endpoint` 检查排序恢复 reason 契约；D-P2-4 测试加 DNS pin。
- 终验：webserver 2718、core+governance 2529、execution/gates/reporting/authority 4968、figures/planning/providers/authority 重跑绿、integration 695、evaluator 274、benchmarks 其余 115+、JS contracts 44/44、ruff 全过、diff-check 干净。工作树约 334 modified + 15 untracked（均未被 ignore，`git add -A` 可收）。

## 严重度统计

| 严重度 | 数量 | 说明 |
|---|---|---|
| P0 | 1（P0-2）+ 条件 1（P0-1：默认关闭，开箱不可达；use_pickle=True+可写目录时为P0级RCE，见第三轮定案） | 正式实验五件套缺失（不得升格 paper authority）；磁盘缓存 pickle 反序列化 |
| P1 | 34（37 − A-7 − A-10 − write_phase − D-P1-4，A-9 维持 P1；P0-1 计入条件项） | 跨 owner 反向依赖、私有跨包导入、静默 fallback、除零/注入、turn authority 提权、脏树耦合测试、供应链/治理矛盾等 |
| P2 | 约 58 | TOCTOU、契约漂移、flaky 测试、硬编码路径、超大文件、基线漂移等（见 §P2 表） |
| P3 | 约 31 | 日志/注释/文案/小卫生（见 §P3 表） |

## 覆盖表（证明“每个代码文件都过一遍”）

| 子智能体 | 范围 | 文件数 | 方法 |
|---|---|---|---|
| A core | `src/easyicu/*.py` 根 30、`concept/` 21、`io/` 18、`scores/` 13、`api/` 10、`runtime/` 10、`utils/` 13、`table/` 6、`databases/` 3、`extensions/` 4、`visualization/` 6、`scripts/extract_features` 1、`data/` py1+json16+md2 | 154（136 py 逐个打开/AST + 16 json + 2 md） | 高风险约 40 文件精读，其余 AST+模式 grep（bare except/fallback/注入/密钥/出站 URL/私有跨 owner/prepared-data） |
| B research_agent 上半 | `contracts/` 68、`authority/` 65、`planning/` 52、`orchestration/` 18、`agents/` 20、`providers/` 24（含 6 txt prompt）、`acquisition/` 11、`discovery/` 25、`research_context/` 13、根 py 35 | 331 | 批量扫描（opt-in/LLM/approved-bypass/私用导入/重试/重复定义/规模）+ 门控与 authority 核心文件精读 |
| C research_agent 下半 | `execution/` 40 + `runners/` 59、`gates/` 35、`figures/` 27、`reporting/` 55、`audits/` 18、`repairs/` 50、`methods/` 21，共 189702 行 | 305 | 全量模式扫描 + 约 30 高风险深读 + 5 脏文件 diff 逐行 |
| D webserver 全栈 | `routes/` 17、`pi_copilot/` 41（30 py + tool_catalog + node_app 10）、`static/js` 113（112 tracked + 1 新增）、`static/css` 58、app 入口/auth/静态托管/ideas/patient_drilldown 等 | 247（246 tracked + 1 untracked） | `innerHTML`/eval/`window.open`/fetch/TODO/console 全量 rg + 高危文件全读 + 脏 diff 逐行 + ownership 契约对照 |
| E 测试工具基准 | `tests/` 1249 tracked（research_agent 820、webserver 165、core 135、js 45、benchmarks 34、governance 24、clinical_specs 17、support 6、根 3）、`tools/` 99、`scripts/` 15、`benchmarks/` 92、`examples/` 9、`sources/` 1 | 1465 tracked + 3 untracked | 分组计数 + 全库 rg（弱断言/flaky/shell/secret/skip/bare-except）+ 脏/新增/改名精读 |
| F 根配置文档运维 | 根配置 21、安全贡献 4、文档根 5、`docs/` 15 条目、`desktop/` 21、`website/` 18、`baselines` + arch_baselines、`output/` 等确认项 | 约 95 条目（精读 62 + 结构化核查 33） | 全文精读 + SHA 重算 + pin/版本脚本化对比；`docs/evidence` 69 文件采用抽查策略（见注） |

> 注：`docs/evidence/` 69 文件（含 14 图二进制）：枚举 6 子目录→精读 2 索引 README→精读 gap 登记→抽 2 个 `validation.json`（最大且分属不同分支）→全目录密钥扫描→其余 `.log`/`.csv`/派生 JSON 视为机器收据不逐行；纸图不做像素核验。`uv.lock` 只做结构/依赖风险核查。`.env.local` 仅确认存在/0600/被 ignore，未读内容。

---

## P0（2 条，必须修）

### P0-1 磁盘缓存 pickle 反序列化 RCE
- file:line：`src/easyicu/api/cache.py:89-93`
- category：安全-反序列化
- description：`use_pickle=True` 时对缓存文件直接 `pickle.load`，文件名仅由可预测 sha256 派生，篡改即任意代码执行；且 `except Exception` 后回退重算，掩盖篡改。
- evidence：`with cache_file.open("rb") as handle: result = pickle.load(handle)`
- suggestion：默认禁用 pickle 缓存，或仅限签名可信目录 + hmac 校验后才 load。

### P0-2 正式实验五件套缺失，不得升格 paper authority
- file:line：`src/easyicu/research_agent/orchestration/experiment_spec.py:1-168`
- category：正式实验完备性
- description：`ExperimentSpec` 仅含 cohort 描述 + runtime 开关，无假设/estimand/主结果契约 hash/冻结计划 digest/签名主体；`registered_report_inputs.py` 只是 report-only repair 准入，不是 preregistration。
- evidence：`CohortInputSpec` + `RuntimeSpec` 即全部；无 plan-digest/signature 项
- suggestion：若行正式实验，需新增 versioned protocol + preregistration（含 statistical plan 与 acceptance contract）+ signed declarations 并与 `human_review_checkpoint` 绑定；否则明确声明当前 candidate 仅到 engineering-complete。

---

## P1（37 条，应在本分支修）

### 核心域边界（A-1~A-4）
- `src/easyicu/catalog_manifest.py:9`｜AGENTS.md 边界｜核心顶层直引 `research_agent.planning.capability_registry`｜`from .research_agent.planning.capability_registry import CAPABILITY_REGISTRY`｜改为 research_agent 侧只读投影/快照，核心只消费数据。
- `src/easyicu/demo_release_pack.py:22-24,137`｜AGENTS.md 边界｜核心 4 处 import webserver（storage/contracts/coverage/dataio），反向依赖｜`from easyicu.webserver import demo_source_storage` 等｜能力收敛到 `io/`/`table/` owner，由 webserver 调核心。
- `src/easyicu/hosted_llm_server.py:47`｜AGENTS.md 边界｜核心 relay 引 webserver `AllowedHostsMiddleware`｜见行｜中间件下沉共享安全模块或启动层注入。
- `src/easyicu/io/data_converter.py:3532`｜AGENTS.md 边界｜`convert_all` 懒引 `research_agent.authority.evidence_store.EvidenceStore`｜见行｜改为回调/协议注入（如 `register_manifest(dict)`）。

### 正确性/静默 fallback/除零/注入（A-5~A-12）
- `src/easyicu/concept/callbacks.py:309-318`｜正确性｜`_STAY_LIMIT_CACHE` 以 `id(data_source)` 为键，对象回收 id 复用串户 + 无界增长｜`cache_key = id(data_source)`｜改 `weakref`/`WeakKeyDictionary` 或版本化 key + 容量上限。
- `src/easyicu/concept/callbacks.py:318,329,338,668-725`｜静默 fallback｜多 loader `except Exception: return None`，把“读失败”当“无映射”｜`try: icu_tbl = data_source.load_table(...) except Exception: return None`｜区分“表不存在”与“读取失败”，后者抛 typed error 携带 source identity。
- `src/easyicu/concept/__init__.py:2759`｜静默 fallback｜`except Exception: pass` 空吞单位/参数解析失败（同文件约 49 处 broad except）｜见行｜至少记 debug 日志并计入 owner 明确的 fallback 计数。
- `src/easyicu/callbacks_missing.py:111-114`｜正确性｜`blood_cell_ratio` 以 `max_val>100` 全局启发式定单位且除 `wbc_num` 未防 0/NaN｜`return (val_num / wbc_num) * 100`｜逐行按单位元数据判定，0/缺失返 NA + ascertainment 标记。
- `src/easyicu/utils/unit_conversion.py:454-471`｜正确性｜`convert_vaso_rate` 对 `weight_kg=0/NaN` 无守卫，污染 SOFA 心血管评分｜`return rate / weight_kg`｜非正/非有限体重抛 `ValueError` 注明 concept 与单位。
- `src/easyicu/api/concepts.py:428-439`、`src/easyicu/datasource.py:2608`｜安全-注入｜`id_col`/排序键拼入 DuckDB SQL，仅引号转义无白名单｜`f'"{id_col}"'` / `f" ORDER BY {', '.join(sort_keys)}"`｜列名 `^[A-Za-z_][A-Za-z0-9_]*$` 校验或复用 `_quote_ident` + 来源白名单。
- `src/easyicu/io/setup_data.py:36-43,68-79`｜异常处理｜未知源 `return None`、下载失败吞掉继续、attach 失败仅 log，全链 fail-open｜`except KeyError: LOGGER.error(...); return`｜未知源抛 `KeyError`，各阶段失败返 typed receipt 或抛异常。
- `src/easyicu/io/data_env.py:107-109`｜死代码｜`.fst` 分支 import 不存在的 `.fst_reader`，必 `ImportError`｜见行｜删除分支或抛“不支持的遗留格式” typed error。
- `src/easyicu/io/src_utils.py:53-58`｜正确性｜回退调不存在的 `DataSourceRegistry.get_default()`，被外层 `except: pass` 吞后抛无关 `TypeError`｜见行｜删回退分支，未知类型直接抛带期望类型的 `TypeError`。
- `src/easyicu/config.py:710-717`｜静默 fallback｜import 期自动加载用户配置异常被静默吞，坏配置与无配置不可分｜`except Exception: pass  # Silently ignore`｜记 warning 携带路径，或 `EASYICU_STRICT_CONFIG` fail-closed 开关。
- `src/easyicu/io/import_data.py:234-243`｜异常处理｜多表 import 逐表吞异常仅 log，“部分成功”无结构化失败清单｜`except Exception as e: LOGGER.error(...)`｜返回/抛出含每表 error 的 typed receipt。

### 跨 owner 私有导入（A-13~A-14，B-2）
- `src/easyicu/scores/microbiology.py:24`、`outcomes.py:34`、`concept/callbacks.py:7569`、`scores/kdigo_aki.py:39,716` 等 7 组｜AGENTS.md 边界｜scores↔callbacks 互引私名（`_build_datasource`/`_lower_cols`/`_detect_id_col`/`_urine_rate_window_avg_multi` 等）及 kdigo 引 `io.ts_utils._infer_numeric_time_unit`｜见各行｜提升为各 owner 公开函数，私有不出包。
- `src/easyicu/api/extraction.py:2457`、`api/concepts.py:377`、`load_concepts.py:911`｜AGENTS.md 边界｜api 直引 `datasource._enumerate_bucket_parquet_files`、`runtime.memory_manager._ceil_div`、`concept._apply_callback`｜见各行｜为私有能力开公开包装，api 只依赖公开面。
- `src/easyicu/research_agent/authority/plan_authority.py:13-14,27` 等｜私有跨 owner｜authority↔planning↔contracts 三向私用耦合（另见 `contracts/step_families.py:10,17`、`planning/plan_graph.py:13-14`、`figure_plan_mutation.py:26`、`final_plan_shape.py:20`、`replan_gate.py:10`）｜`from ..planning.figure_step_contract import _preserve_figure_steps_after_replan`｜被复用函数去下划线公开 + 单测，或下沉中立 owner（如 `plan_utils`）。

### research_agent 门控与证据（B-1、B-3、C-F2~F5）
- `src/easyicu/research_agent/authority/plan_lifecycle.py:542,587-609,696-743`｜authority-lineage｜同一 revision 第二个不可变 public plan 写变体 id，但 typed loader 只读基线，变体成只写孤儿，resume/approve 读旧基线｜`evidence_id = f"{evidence_id}_{normalized.plan_sha256[:8]}"`｜变体加 typed loader（按 digest 显式选择）或禁止同 revision 第二个 normalized plan（fail-closed）+ 回归测试。
- `src/easyicu/research_agent/orchestration/config.py:293` + `workflow.py:421,440`｜门控｜`require_human_plan_review` 默认 False，干净 plan 可不经人审直进 Execute，paper 权威靠调用方记得开｜`require_human_plan_review: bool = False`｜非 mock/非 development 执行强制该旗，或 paper profile 校验显式报错。
- `src/easyicu/research_agent/repairs/runner_dispatch.py:1895-1928`｜静默 fallback｜Fix F 向 agent 代码注入 `def {helper}(*args,**kwargs)` 容错桩，把“引用不存在的分析函数”重写为返字符串/None，可能静默改变统计语义｜`repaired = stub + "\n\n" + code`｜限白名单（仅序列化形参位）或写 `step_record["semantic_stub_injected"]` 并由 `gates/step_result_evidence.py` fail-closed/警告 + 回归测试。
- `src/easyicu/research_agent/repairs/summary.py:56-114`｜证据绑定｜stdout 最后一个 JSON 与 out_dir 下任意 `*summary*.json` 被提升为 `step_summary.json`，未经 declared-output 校验即成证据｜`except Exception: continue/return False`｜打标 `summary_provenance: salvaged_from_*` 并强制过 `step_summary_integrity` 全量校验；限 `expected_outputs` 声明文件名。
- `src/easyicu/research_agent/reporting/publication_bundles.py:184-204,208-239`｜证据绑定｜rescue 读坏 `step_summary.json` 时 `except Exception: summary = {}` 后覆盖写回，销毁“损坏”证据；generic terminal 仍有跨语义 figure 提升残余风险｜`summary = {}` + `write_text` 覆盖｜腐坏先封存 `step_summary.corrupt.<sha>.json` 并记 finding；generic 提升要求 contract role 与 step 家族一致否则 fail-closed。
- `src/easyicu/research_agent/reporting/write_phase.py:1415-1442`（脏）｜正确性｜novelty 重注册调 `evidence.current_verified_records(None)`，`None` 语义（全量 vs 空视图）无注释；无该方法时回退 `evidence.get` 单记录可能漏历史版本｜双路径 try/except｜明确 `None` 语义 + 断言/注释 + 双路径一致性测试。
- `src/easyicu/research_agent/pipeline.py:1`（7141 行）等｜plan-dispatch 分散｜派发散在 pipeline/orchestration/agents 三层（另 `progressive_planner.py` 4867、`progressive_compiler.py` 2853、`scientific_review.py` 2743 行；67 文件 >800 行）｜行数统计｜暂不拆大文件，但 dispatch 表收敛到 `orchestration/progressive_planning.py` 单一显式路由 + 矩阵单测。

### webserver 提权/XSS/沙箱（D-P1-1~P1-5）
- `src/easyicu/webserver/pi_copilot/service.py:2074-2075` + `static/js/screens-guided-pi.js:31-35,148-151` + `pi_copilot/tools.py:3804,3899-3900,4074`｜turn authority｜`full` 模式浏览器供给 `allowed_actions` 与后端推断取并集，特权一次授权可被篡改端预授；`authorize()` 主路径不复核推断（仅一条 fresh+same_plan 分支校验）｜`frozenset(client) | infer_explicit_turn_actions(...)`｜特权动作（`provider_run`/`extract`/`report_revision`）以后端推断为必要条件（交集），`full` 运行时在 `authorize()` 复核 + 补用例（闲聊带 `provider_run` 必须 `pi_action_authorization_required`）。
- `src/easyicu/webserver/pi_copilot/turn_authority.py:96-106`（脏）｜turn authority｜`_ADVISORY_PLAN_TAIL` 只贴一条模式；新 `生成…计划` 模式只挡 `的建议` 不挡 `给出建议/提供分析/做敏感性分析`，其余 6 条中文重规划模式无 tail，英文 `revise plan` 无守卫｜负向前瞻仅 `(?!的(?:建议|…))`｜tail 应用到全部中文生成/修订模式并扩展 `给出|提供|提出…建议/分析`，英文补否定，补正反用例。
- `src/easyicu/webserver/static/js/screens-guided-pi-provider-control.js:67-70,83-89`｜出站 URL｜Codex `auth_url` 未校验即 `popup.location.href`/`window.open`，`javascript:`/钓鱼域可导航｜`popup.location.href = authUrl`｜复用 `literature.safeUrl` 语义（`https:` + hostname + 无 userinfo + 可选 allowlist）校验后导航；后端 `codex_gateway` 断言返回域。
- `src/easyicu/webserver/static/js/screens-guided-pi-preview.js:447 vs 452`｜沙箱｜`document` 预览 iframe 无 `sandbox`，`web` 预览有；demo HTML 同源执行违背 `routes/pi_copilot.py:326,351,560,687` 纵深｜document 分支无 sandbox｜document iframe 加 `sandbox="allow-scripts"` 或复用 srcdoc + CSP 路径 + 静态测试断言。
- `src/easyicu/webserver/static/js/screens-guided-pi-error-text.js:61-63`（新增）｜XSS｜新共享 `option()` 对 value/selected/label 零转义；当前调用方恰为硬编码故未爆，但服务端模型列表极易误用｜`` `<option value="${value}"...>${label}</option>` ``｜`esc(value)/esc(label)`，selected 只做 `===` 后输出字面；补 ownership 测试禁非转义插值。

### 测试与基线诚实性（E-P1-1~P1-5）
- `tests/research_agent/core/test_bench_submission_profile.py:762-769` + 脏 `src/easyicu/data/concept-dict.json`｜测试契约｜0917 profile 断言活指纹 `== compute_concept_dict_fingerprint()`，干净检出必红，脏树才绿｜`assert profile.expected_concept_dict_sha == fingerprint...`｜冻结期望 sha 常量，活指纹检查拆独立测试并标注 dirty 敏感/skip。
- `tools/materialize_e3_miiv_kdigo_gradient.py:30` vs `tests/core/test_e3_strict_kdigo_window.py:6`｜导入契约｜materializer 用顶层 `from e3_strict_kdigo_window import`，测试用 `from tools.e3_strict_kdigo_window import`，`tools/` 不在 path 即 `ModuleNotFoundError`｜见两行｜统一为包路径 + 导入契约测试。
- `tools/arch_baselines/research_agent_resource_context.json:478-486`（脏）｜基线漂移｜新增 history 自认 6 类源码变更 “not covered”，基线变免责声明｜`baseline_reason` 改写｜纳入测量重算 digest 或拆两次基线修订，不允许“承认未覆盖照样过门”。
- `tests/webserver/copilot/test_pi_study_results.py:10-24`｜覆盖造假｜零断言（仅 `subprocess.run(check=True, capture_output=True)`，输出丢弃），缺 node 静默 skip｜assert 计数 0｜失败打印 stdout/stderr；缺 node 改 fail/xfail(strict) 并在 CI 保证 node。
- `src/easyicu/research_agent/__init__.py`（脏）vs `tools/arch_baselines/research_agent_top_level_ownership.json` 等｜基线漂移｜包公有面新增 profile 重导出但 ownership/module_graph 基线未刷新｜`test ... is profiles.X` 新断言 vs 09-14 快照｜刷新另两基线或在 governance 对重导出加显式允许清单。

### 根配置/治理/供应链（F-P1-1~P1-3）
- `pyproject.toml:156-160` vs `.github/workflows/ci.yml:203`｜供应链｜CI 要求 wheel 含 `Dockerfile`/`requirements.lock`/`base-image.lock`/`README.md`，但 package-data 漏 `base-image.lock`｜三项列表无该锁｜补 package-data，重跑 packaging 门。
- `CONTRIBUTING.md:30-32` vs `AGENTS.md:18-22`｜治理｜前者强制并发 agent 每任务一 linked worktree，后者禁新建 worktree/分支（PR 模板复述前者）；两者无法同时遵守｜原文如上｜二选一收敛 + PR 模板同步。
- `.gitignore:147` + `uv.lock`（mtime 08-15，149 包）vs `pyproject.toml`（09-15）｜依赖｜唯一全量锁被 ignore 且过期一月；`anthropic>=0.119` 无 pin（`anthropic NOT IN LOCK`）；`cryptography` 49.0.0 vs runner 50.0.0｜stat/pin 提取｜跟踪通用锁或文档化替代快照，补 pin，对齐 release checklist §3。

---

## P2（重要非阻塞， condensed 表）

| # | file:line | 一句话问题 | 建议 |
|---|---|---|---|
| A-P2-1 | `outbound_url_security.py:56-93` | DNS 验证与建连 TOCTOU | 高风险部署 pin 验证 IP 建连 |
| A-P2-2 | `runtime/project_config.py:64-71` | 生产路径硬编码 `/home/1_publicData/...` | 无 env 则抛指引错误 |
| A-P2-3 | `concept/schema.py:69-98` | 布尔值静默转 None 掩盖字典错误 | 直接抛 `TypeError` |
| A-P2-4 | `scores/outcomes.py:39-64` | `_raw_table` glob 裸读无 schema 校验 | 先验 schema + source identity |
| A-P2-5 | `utils/file_utils.py` 多处 | `open()` 无 encoding | 统一 `utf-8` |
| A-P2-6 | `io/download.py:80-119,232-236` | 先裸请求再补凭证 + 吞异常 | 首请求即带凭证，失败上传播 |
| A-P2-7 | `table/utils.py:1-44`、`io/data_utils.py:66-77` | deprecated shim 随包发布 | 2.0 移除或标禁新用 |
| A-P2-8 | `concept/catalog.py:1-908` | 手写中英目录与 json 并行维护 | 由字典生成或 CI 对照测试 |
| A-P2-9 | ~~`scores/comorbidity.py:18-21` 文档失实~~ **FALSE-POSITIVE，已撤回** | 复核：`grep ^import/^from` 仅命中 `__future__/typing/pandas`，零 easyicu 导入，docstring 准确，无此问题 | — |
| A-P2-10 | `concept/__init__:9358` 等 5 文件 | >4000 行巨文件 | 按解析/执行/回调拆分，保持公开 import |
| A-P2-11 | `scripts/extract_features.py:26-38,346` | 相对 import 错误 + 懒引 webserver | 改绝对 import，移出 webserver 依赖 |
| A-P2-12 | `io/data_load.py:19-22` | 自述无调用者、低频无护栏 | 加契约测试或标 experimental |
| A-P2-13 | `datasource.py:184-185`、`databases/detection.py:90-103` | broad-except 丢区分度 | 保留 fallback 但计 degraded receipt |
| A-P2-14 | `table/__init__.py:1643-1646` | env 隐式解析数据源难追踪 | 写 lineage/sidecar |
| B-P2-1 | `acquisition/catalog.py:157,160-161` | 私引 `_load_concept_dict_cached` + 失败返 `{}` | 用公共 API + coverage 暴露降级 |
| B-P2-2 | `discovery/concept_proposal.py:191` | `LLMComplete` 可调用未强制 `authorized_complete`（现无调用方，死代码） | 签名收 `LLMClient` 内部授权 + 单测 |
| B-P2-3 | `contracts/runtime.py:46,87,99` | 私名结果类型被跨包依赖 | 去下划线公开 |
| B-P2-4 | `research_context/builder.py:170-179` | `_safe_get_concept_info` 全返 None 降级无标记 | 返降级标记，下游可见 |
| B-P2-5 | `orchestration/human_review_checkpoint.py:262-282` | `transitioned()` 无转移表 | 加允许转移表/具名方法 |
| C-F6 | `gates/step_contract.py:20-22` 等 | gates 跨 owner 私引多处 | owner 开公开薄包装 + `__all__` 锁 |
| C-F7 | `runners/table_one_executor.py:77,126` 等 | runners 直连 methods 绕过 adapter catalog | 经 catalog 解析 + 漂移测试 |
| C-F8 | `execution/runner.py:1629-1656` | `allow_unsafe_host_fallback` 开后改执行语义 | 保持默认关；降级追加显式标记 |
| C-F9 | `candidate_loop.py` 7 处 + `phase_support.py:713-717` | MockLLM fallback 方法等价性无人校验 | 过 `method_compatibility` + readiness 降级显示 |
| C-F10 | `orchestration/profiles.py:627-656`（脏） | 0917 canary 默认行为随指针变 + 三方 sha 一致性 | CI 断言 CURRENT ref == LOCK digest |
| C-F11 | `reporting/system_validation_report.py:500-529`（脏） | ledger 拆名破坏性变更 | 加 `supersedes` 备注或旧名 alias 一版 |
| C-F12 | `runners/deterministic_robustness.py:819-1333` | 15 处 `or` 缺省 + 15 except，缺失静默 None | sentinel 区分 missing/empty + 缺失矩阵测试 |
| C-F13 | `repairs/attempt_record.py:68-100`、`provider_budget_runtime.py` | receipts 散、无统一 denominator/registered retry 规则 | 新增 `retry_policy` 契约 + validation 引用 |
| D-P2-1 | `preview.js:360-363` 等 | `EU_PRODUCT_LABELS` 裸调用 + 默认标题语义漂移 + 版本未 bump | `?.` 防御 + 统一语义 + bump `?v=` |
| D-P2-2 | `error-text.js:43-46` | fallback 返原始 message，依赖调用方 esc | 头注释约束或统一 esc + 扫描测试 |
| D-P2-3 | `tweaks.js:66,126,169,175-180` | edit-mode `postMessage('*')` + 无 origin 校验 | 指定 origin + 校验 source |
| D-P2-4 | `screens-guided-pi-events.js:268` | `custom-openai` 占位 `example.com` 可提交致 key 外发 | 占位仅 placeholder + 后端拒 `example.*` |
| D-P2-5 | `error-text.js:49-59` | 预设 `includes` 误分类钓鱼域 | hostname 精确/后缀匹配 |
| D-P2-6 | `screens-guided.js:4194` 等 | 超大 JS/CSS 靠 ratchet 防增不治存量 + console 残留 | 按接缝续拆，console 换 tr |
| D-P2-7 | `routes/extensions.py` + `app.py:146-205` | 安装面仅靠 loopback，无确认令牌 | 加 `action_key`/`expected_sha256` 确认 |
| E-P2-1 | `benchmarks/.../evaluator/scoring.py:286` | 空 hazards/forbidden 除零 | 非空校验 + 单测 |
| E-P2-2 | `test_test_organization.py:100-112` | 大文件棘轮被拆分绕过，总量反增 | 加拆分总量断言 |
| E-P2-3 | `tests/test_e3_strict_kdigo_window.py`（D 未 staged） | 根测试改名一半，HEAD 仍红 | 提交改名 + 旧路径断言 |
| E-P2-4 | 约 40 处 skip | corpus/node 缺失即跳过，覆盖失真 | `requires_corpus` 标记 + skip 计未覆盖 |
| E-P2-5 | 约 30 命中绝对路径 | `/Volumes/...`、`/home/zhuhb/...` 写死 | `EASYICU_DB_ROOT` + 缺失 skip 计数 |
| E-P2-6 | 约 15 命中 sleep | 轮询等待 flaky | `wait_until` helper + 相对断言 |
| E-P2-7 | 多文件 `except: pass` | 吞失败后神秘 assert | 改 `pytest.fail`/raise |
| E-P2-8 | 约 70 处裸 assert | `assert errors` 不断内容 | 断 len/code，复用 truthfulness 风格 |
| E-P2-9 | `tools/fetch_baselines.py:158-169` | registry 驱动 clone 无 allowlist + `--force` rmtree | 主机 allowlist + 二次确认 |
| E-P2-10 | 6 处 rmtree | 发布/物化删前无越界守卫 | `is_relative_to` + 非 symlink 断言 |
| E-P2-11 | `guided_pi_study_results.test.js:87` 等 | JS ownership `"."` 特判 + CONTRACTS 过时 | 显式清单 + 更新注释 |
| E-P2-12 | `research_workflow_fixtures.py` 等 | fixture 中心化 + 测试间导入耦合 | 下沉 conftest/support + 禁 `from test_* import` 门禁 |
| E-P2-13 | workflow 测试 307 patch | 关键门禁全被桩 | 保留未桩集成用例（`requires_docker`），桩用例缀名 |
| F-P2-1 | `pyproject.toml:105-107` | `all` extra 漏 `scientific-adapters` | 纳入或显式说明 |
| F-P2-2 | `start_easyicu.sh:51` 等 | 可选 3.13 但 CI 仅到 3.12；sh/bat env 分叉 | 上限对齐 CI，统一 env |
| F-P2-3 | `AGENT_PLAN.md:3-6,685,695` | PLAN 陈旧与共识冲突 + 历史明文 key 自曝 | 加 SUPERSEDED 声明，轮换 key，改单变量读取 |
| F-P2-4 | `research_know_how/*.json` | `applicable_databases` 别名漂移（含注册表无值） | 统一 registry 六名或加 alias 表 |
| F-P2-5 | `website/verification.json:3-8` | 收据 head 落后 22 提交 + 检后 5 改 | 落定后重跑 focused 验证 |
| F-P2-6 | `baselines/LOCK.json` vs `REGISTRY.md` | 锁仅 5 项 + HealthFlow URL 自矛盾 | 统一 URL，补锁或声明范围 |
| F-P2-7 | `tools/arch_baselines/*.json`（脏） | **STALE**：复核时工作树已干净，疑似 `170b58189 relock` 已入库，无需再修 | — |
| F-P2-8 | `ci.yml:6-26` | 满矩阵不自动跑，main 可静默前进 | aggregate 定触发点落盘 exact-head CI URL |
| F-P2-9 | `desktop/backend_entry.py` 等 | `--session-token` 与 URL 传 token + DMG 名硬编码 | 仅 env 传 token，DMG 名派生版本 |
| F-P2-10 | `runner_image/Dockerfile:90-95` | `--no-build-isolation` 假设基座有构建工具 | 探针断言并写收据 |
| F-P2-11 | `pi_workspace_security_ci.yml` 等 | 安全门缺 concurrency；dependabot 仅 actions | 补 concurrency，定审计负责人 |

---

## P3（微小，condensed 表）

| # | file:line | 一句话问题 |
|---|---|---|
| A-P3-1 | `io/export.py:73` 等 | 库代码 `print` 污染 stdout，改 logging |
| A-P3-2 | `io/__init__.py`、`runtime/__init__.py` | 空 `__init__` 无 docstring |
| A-P3-3 | `concept/loader.py:69-84` | `load_dictionary(src_name)` 忽略参数 |
| A-P3-4 | `utils/callback_utils.py` 等 | 中英注释混排 + emoji 日志 |
| A-P3-5 | `runtime/memory_manager.py:719` 等 | JSON 写缺 encoding |
| B-P3-1 | `contracts/landmark_spline_validation.py:231` 等 | `except: return False` 丢诊断（fail-closed 对但难排障），附 `error_type` |
| B-P3-2 | `providers/clients.py:193` | Retry-After 解析 `except: pass` 良性，无需改 |
| B-P3-3 | `authority/pipeline_cache.py:570` | 缓存探测 `except: return None` 良性 |
| B-P3-4 | `acquisition/foundation.py:73` 等 | `_extract_json`/`TimeWindow`/`_coerce_scalar` 重复，下沉公共 owner |
| B-P3-5 | `providers/cost.py:587` 等 | wrapper 内直调系授权内部分发，非旁路；加注释防误改 |
| C-F14 | `reporting/reviewer.py:158-166`（脏） | 加 `simulated_deterministic` + claim_boundary 正面；两处文案抽常量 |
| C-F15 | `reporting/scientific_maturity.py:314-358`（脏） | novelty 收紧正确；仍缺签发主体（见 P0-2） |
| C-F16 | `gates/preflight.py` 6228 行等 | 4 个 4k+ 单体 + runners 小 helper 重复；下沉 `_shared.py` |
| D-P3-1 | `static/index.html:62-66,171`（脏） | picker 顺序修复正确；`?v=` 非内容哈希，改 owner 必 bump |
| D-P3-2 | `tests/js/guided_pi_module_harness.cjs` | 缺 `errorText`，补 harness + 单测 |
| D-P3-3 | `guided-pi-workspace.css` 等 | 窄视口模型名截断等 3 处，加 ellipsis/title |
| D-P3-4 | 约 32 js | 缺 `Owner:` 头，补注释或加门禁 |
| D-P3-5 | `error-text.js:44,70,76` | 双语全；`receipt`/`StudyContext` 术语保留现状即可 |
| D-P3-6 | `SKILL.md` vs preview | 相对路径/隐私基本守住；例外见 D-P1-3/P2-4 |
| E-P3-1 | `test_coder_output_scope.py:1042` | 字符串内 `except:` 系故意坏码样本，加 noqa 防误修 |
| E-P3-2 | `examples/*.py` | 占位 key 样式 + 真 env 透传，改 PLACEHOLDER + 注释 |
| E-P3-3 | `scripts/*.py` | 默认写相对 `research_output/` 脏树，改 `--out` 必填/tmp |
| E-P3-4 | `test_bench_submission_profile.py` 多处 | frozen SHA 手写三处同步 churn，收敛单一快照或自动派生 |
| E-P3-5 | `test_reviewer.py` 等 | 新增测试只断字符串/过滤，未覆盖装配；ledger 丢弃应留痕 |
| E-P3-6 | `safety_runner.py` 等 | transport/acceptance fail-closed 好；仅缺空 adjudication 单测（见 E-P2-1） |
| F-P3-1 | `LICENSE:3` 等 | 版权 holder 与 URL 大小写三处漂移，对齐 |
| F-P3-2 | `CONTRIBUTING.md:16-18` | `pytest -q` 默认跳 slow，与“核心门”不符，双层表述 |
| F-P3-3 | `website/scripts/build_site.py:58,71,76` | 数字硬编码与 COUNTS 并存，统一渲染 |
| F-P3-4 | `start_easyicu.sh:2,88-92` | 缺 `pipefail`，`read` 非交互一闪而过 |
| F-P3-5 | `.gitignore:158-159` | docs 忽略规则绕，加注释 |
| F-P3-6 | `ci.yml:106-110` 等 | 冗余装 openai；research 门缺 3.12 腿 |

---

## 已验证的正面项（避免误修）

- `io/data_converter.py` tar 安全解压（containment/链接/类型校验）+ 内容 receipt；`databases/detection` schema-first 身份检测 fail-closed；`extensions/mcp_client` allowlist + 字节界 + 脱敏；`registry` digest 冻结 + fcntl 锁 + 出站校验。
- kdigo strict numeric/time（坏文本抛 `KDIGOComponentSchemaError`）；outcomes follow-up 显式建模；`table/id_conversion` unmapped-policy。
- research 上半：`independent_review` 只是 remediation 路由与 finding 编码，无自批准；已接线 LLM 均走 `authorized_complete`；`cli.py` 先过 `ai_optin` 再 stamp；outbound deny-by-default；重试有界。
- research 下半：`provenance_fail_closed` 2284 行 fail-closed 引擎；`failure_classification` deterministic 门防误命中；`step_execution` 重试不预清空；`delong_auc` logit CI；`multiple_testing` 精确列名 + 家族内校正；`profiles.py` 全 `False` 隔离降级默认关。
- webserver：`literature/markdown.safeUrl`（https + 无 userinfo）、workspace srcdoc + CSP、loopback + proxy 头拒 + `AllowedHostsMiddleware`、CSRF 写拒基本面正确。
- 测试工具：`test_audit_tool_truthfulness` 精确计数、`test_death_inhospital_callback` SICdb 时钟修正、`e3 strict` 状态机与测试一致、`safety_runner` 无 shell/pickle/yaml.load/eval 均为正确实践。
- 根运维：`output/`/`outputs/`/`task_logs/` 零跟踪且被 ignore；`.env.local` 被 ignore + 0600；密钥扫描零命中；actions 全 SHA pin + 最小权限。

## 建议修复顺序

1. P0 两条（缓存反序列化；正式实验声明/实现二选一）。
2. 安全类 P1：D-P1-1/2/3/4/5、A SQL 注入与除零、F 密钥/供应链。
3. 证据与门控类 P1：C-F2/F3/F4、F5、B-1/B-3、E-P1-1。
4. 边界类 P1：A-1~4、A-13/14、B-2（公开化私有导入，消除核心↔research↔webserver 反向依赖）。
5. 脏树收敛：先提交/还原改名与基线（E-P2-3、E-P1-5、C-F10、F-P2-7），再重跑 verification（F-P2-5）与 packaging 门（F-P1-1）。
6. P2 按表分 owner 修；P3 随手修或建卫生批次。

---
*本报告由 6 子智能体只读审阅汇总而成，未修改任何源码。行号以审阅时工作树为准（含未提交改动），修复前请以对应文件最新行号复核。*
