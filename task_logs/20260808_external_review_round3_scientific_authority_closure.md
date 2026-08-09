# 2026-08-08 外审第三轮：科学权威链收口

## 范围与边界

- 任务：`AGENT-SCIENTIFIC-MODEL-CONTRACT-CONSOLIDATION-20260807`
- 分支：`fix/external-review-20260724-p0-p1`
- 起点：`513018dc3a646141107fccc6dbdca5f40796fff8`
- 本轮只处理外审列出的六项严格缺口及其必要契约测试；没有调用 Provider、没有读取患者数据、没有构建镜像、没有启动 M3，也没有改变 Canonical9 的 `4/9` 状态。
- 本日志所在提交即本轮候选提交；后续只能从经过独立复核、push/current-SHA CI 的精确提交构建新镜像。

## 裁决与实现

### 1. JSON 修复不再污染宿主解释器

旧修复通过赋值 `json.dump/json.dumps` 改写进程级标准库模块，导致生成脚本之外的宿主代码也被改写，连 `allow_nan=False` 的证据身份 fail-closed 都会失效。现在只重写待修脚本自己的 `json.dump(`/`json.dumps(` 调用点，局部 wrapper 仍能把脚本中的非有限值规范成 `null`，但不再改变进程内标准库对象。测试同时钉住正向修复、宿主隔离、队列身份拒绝和 AST 无模块属性赋值。

### 2. AnalysisStep 科学签名覆盖全部结构化权威字段

`_step_scientific_signature` 保留原有核心坐标，并对全部结构化科学字段做 canonical JSON。字段被精确分入 core scientific、structured scientific、presentation-only、runtime-only 四类；新增、漏分、未知或重叠字段均在运行时 fail-closed。嵌套模型要求中的 covariates、level/reference、primary contrast，以及所有后续 `*_spec` 都进入 seal/resume identity。

### 3. freeform association 声明和失败语义收紧

- `association_freeform_v1` 明确为 `analysis_only`，不能授权论文主结论。
- capability id 未声明、未知、跨 family、与 step shape 冲突均得到稳定裁决；矛盾不再静默重路由到宽松 freeform。
- `AnalysisPlan` 在 schema 边界用 registry 验证 capability，阻止未知或跨族声明进入后续流程。
- capability assessment 覆盖 resolver 的每一种 failure reason；缺失声明降为 `analysis_only`，矛盾声明为 `unsupported`。

### 4. readiness 先绑定当前计划权威

finalization 现在先解析并验证 current registered plan authority，再写 readiness。论文授权要求 verified plan authority 且带 64 位 SHA；解析失败发生在 durable run status 之前，避免先产出“ready”再发现计划权威无效。

### 5. 确定性 association → semantic authority → O23 E-value

O23 先通过当前成功 step ledger 和 digest-verified `EvidenceStore` 解析 `primary_association` 与 `outcome_rate`，再读取 deterministic owner 的 typed CSV：只有 `effect_scale=odds_ratio` 才使用 `estimate/ci_low/ci_high`；已声明的非 OR scale 不猜测、不转换。缺失或歧义 event rate 结构化拒绝。生成的 `e_values.csv/.md` 会重新注册为 typed evidence。冻结 oracle 只声明 RR 公式对照，不虚构 OR 转换已经获得外部 oracle 验证。

### 6. PH 外部对照使用真实三类生存夹具

规范 token 改为诚实的 `schoenfeld_per_covariate_with_bonferroni_summary`，旧 token 仅作输入兼容别名。生成器固定产生三套 n=1800 piecewise-exponential 夹具：true PH、exposure non-PH、nuisance non-PH，并冻结 R `survival::cox.zph` 逐协变量结果。测试核对 Bonferroni 下的逐协变量拒绝方向，并只对刻意违反项比较卡方量级；没有把我方 Bonferroni summary 冒充 R 的 global 联合检验。

## 验证

- 依赖/邻接套件：`505 passed`，10 warnings。
- repair、JSON 隔离、R oracle、PH 与静默丢步套件：`137 passed, 1 skipped`；唯一 skip 是故意模拟 lifelines 不可用的分支。
- 合计不重叠文件口径：`642 passed, 1 expected skip`。
- oracle 生成器复跑成功；所有改动 Python 文件 `ruff check` 通过；`git diff --check` 通过。

## 全腿边界与基线红裁决

一次 `tests/research_agent` 全腿在约 8% 时因失败提前停止，已完成部分为 `835 passed, 1 skipped`。其中静默丢步语料计数断言是本轮语料增长后的脆弱测试，已改为断言实际 silent-drop 非零并复验通过。剩余 `test_cache_hit_reuses_run_id_and_workdir` 在精确起点 `513018dc3a646141107fccc6dbdca5f40796fff8` 的 detached worktree 中独立运行，得到同样的 `run_id` 不复用失败（66.70 秒），因此是起点既有红，不是本轮回归。

该缓存红的现状是：默认 Mock 计划因 locked robustness specs 没有 owned replay，不能达到 manuscript-ready，缓存按现合同不会记录它。是否改变 Mock 计划、缓存门或 robustness ownership 是独立架构决策；本轮没有用放宽 readiness 或伪造可缓存结果去“修绿”。因此本轮可以声明外审六项严格缺口已关闭，但不能声明全仓库全绿。

## 后续门

1. 独立审阅本轮提交。
2. push 后跑 current-SHA CI/packaging/portability。
3. 单独裁决并修复既有 cache contract 红；不要把它混入本轮科学权威补丁。
4. 全部离线门和冻结 Figure 2 authority 通过且用户授权后，才从 exact SHA 构建 immutable image 并 fresh aware-only M3；不 resume verify82/83/84/85/87。
