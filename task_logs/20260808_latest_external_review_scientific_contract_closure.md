# 最新外审 scientific-contract 收口（2026-08-08）

## 范围与基线

- 分支：`fix/external-review-20260724-p0-p1`
- 精确起点：`dff4ee6a38df078f7088a8f38fddac2b8826fe1e`
- 输入审阅：`/Users/haibo/.codex/attachments/6d3a5831-8e82-4d21-8695-1277ab07ef4c/pasted-text.txt`
- 任务边界：只收口审阅列出的 O24、plan/endpoint authority、schema 分层、E-value 转换合同、JSON key collision、reachability 残留，以及 current-SHA release evidence；不启动 Provider、不读取患者数据、不构建/运行 M3。

## 裁决与实现

### 1. O24 subgroup/fairness 不再由 Host 擅自决定 secondary science

- `PipelineConfig.enable_fairness_subgroups` 默认改为 `False`。
- 新增 typed `SubgroupAnalysisSpec`，显式声明 primary model requirement、predictor/outcome、subgroup 轴、连续变量分箱、最小样本量、effect scale 与 multiplicity family。
- spec 必须绑定 exact primary `PlannedModelRequirement` 和 primary association evidence；不再从列名猜 predictor、自动挑 age/sex/race 或自动决定分箱。
- 当前 kernel 只支持未调整分析，因此非空 adjustment roster 用稳定原因 `subgroup_adjustment_unsupported` 静默拒绝，不伪称沿用 primary adjusted model。
- O24 调整到 O22 之前执行；interaction p-value 作为独立 hypothesis row 写入显式 family，再由 O22 统一校正，不再在 strata 行复制同一个 p-value。

### 2. Endpoint 只有一个科学权威源

- `ResearchContext.endpoint` 是 immutable source of truth。
- `AnalysisPlan.endpoint` 仅保留为兼容 projection；存在时必须与 context 完全一致，Planner 不能重新发明 endpoint。
- typed endpoint contract 从 `schema.py` 下沉到 dependency-neutral contracts；规划边界提供稳定原因 `endpoint_context_authority_required`、`endpoint_projection_mismatch`、`endpoint_context_missing_or_wrong_kind`。
- execution/final validation 只消费 `context.endpoint`；删除无意义的 Planner endpoint retry（Planner 无权修改 sealed context）。

### 3. AnalysisPlan authority 分类与 capability 分层

- 所有 `AnalysisPlan` public fields 被精确分为 core science、structured science、steps、presentation、runtime 五类；exact union/no-overlap drift test 防止未来字段漏分类。
- endpoint projection 纳入 scientific signature；`display_labels` 与 `revision` 不影响科学身份。
- 稳定 capability ID vocabulary 下沉至 `contracts/capability_ids.py`；schema 只校验 ID，family compatibility 留在 planning resolver，解除 schema→planning 反向依赖。
- module-load drift assertion 保证 registry 与 neutral vocabulary 精确一致。

### 4. OR E-value 转换合同完整化

- 新增 typed `EValueConversionSpec`：estimate scale、Zhang–Yu method、baseline-risk evidence ID、rate column、population column/value、point/interval transform 与 null-crossing rule。
- observed event rate 必须从声明 evidence 的唯一 population row 读取；缺 spec、缺产品、population 不存在、rate 非法或行歧义均 fail closed。
- receipt 记录完整 spec/digest，并明确 external oracle 只验证 RR E-value 公式；OR→RR 由本 receipt 的 population contract 承担，不夸大 oracle 范围。

### 5. 小型完整性与可达性残留

- generated-script JSON sanitizer 对 canonicalized key collision（例如 `1` 与 `"1"`）稳定 fail closed，不再静默覆盖。
- 删除 reachability 测试里已删除 `evalue` 模块的历史残留说明。

### 6. 架构与 release 裁决

- paper-authority completion gate 从 `reporting/readiness.py` 提取到 `reporting/completion.py`；endpoint validator 从 `plan_utils.py` 提取到 owner contract。
- `arch_measure --diff` 全部 lower-is-better 指标不回退，无 rebaseline；import-linter 7/7、deptry 均通过。
- 历史 cache 红不是生产 cache 漏洞：默认 Mock run 为 `analysis_only`，按合同本来就不能缓存为 completed。旧测试改为断言 fresh/no-index；同时建立真正满足 sealed authority、evidence、manifest/status 的 completed candidate，保留 cache 命中及 identity/evidence/status 篡改负测。完整 cache 文件 19/19 通过，生产 fail-closed gate 未放宽。
- wheel 与 sdist 构建成功，wheel 以 zip import 验证 runner image resources。额外发现 `[methods]` 的 `scikit-survival 0.28` 与项目 Python 3.10 声明冲突；现以 `python_version >= '3.11'` marker 限定并加 3.10 metadata resolution regression。

## 验证证据

- 审阅相关广域回归：`272 passed, 1 skipped`。
- cache 合同：`19 passed`。
- runner/architecture：`70 passed, 3 skipped`。
- scientific adapters / dependency floors：`16 passed`。
- Ruff（`src` + `tests`）、`git diff --check`、architecture ratchet、import-linter、deptry：全部通过。
- wheel + sdist：构建成功；wheel clean zip import 与 runner image resource 检查通过。
- 首次 Research Agent 全套给出 `18 failed, 9837 passed, 13 skipped`。逐项裁决后，1 项是本轮 contracts 包初始化回归；其余由旧 scaffold golden、已迁移 envelope owner、readiness 旧夹具、paper replication 旧断言、依赖本机 DNS 重写的 provider 测试和过期错误文案断言构成。4 个代表性历史红已在精确起点 `dff4ee6` 的 detached worktree 复现，生产 cache/fail-closed 合同没有为追绿而放宽。
- 上述 18 项对应文件修正后：`75 passed`；关联 runner/architecture/dependency 回归：`86 passed, 3 skipped`。
- **最终 Research Agent 全套**（排除单独已跑的 `test_runner.py`）：`9855 passed, 13 skipped, 0 failed`，耗时 `46:42`。
- 最终静态/release 检查：Ruff、`git diff --check`、architecture ratchet、import-linter `7 kept / 0 broken`、deptry 均通过；current-worktree wheel + sdist 构建成功，wheel clean zip import、`EndpointSpec` 与 runner image `Dockerfile`/`requirements.lock` 资源检查通过。

## 未做与后续边界

- 未调用 Provider、未读取患者数据、未运行 paid canary、未构建 exact-SHA immutable Docker image、未运行 M3；Canonical9 仍为 4/9。
- 未声称 macOS 本机完成 Python 3.10 全套运行；本轮完成的是 metadata resolver regression，current-SHA 远端 Python matrix 仍需 push 后 CI。
- 审阅中的 god-module 拆分、reportable capability breadth、`protocol_only` 产品降级是明确列在 Canonical9 之后的独立阶段，不与本批 scientific-contract 修复混提。下一步应冻结并接受独立 review/current-SHA CI，再决定是否进入 immutable image 与 fresh aware-only M3。
