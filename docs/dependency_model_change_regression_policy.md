# 依赖与模型变更的分层回归政策（A6/A12）

> 目的：复用仓库已有的版本/许可/测试字段，把依赖或模型变更的验证固定为两层必做 + 一层按授权，
> 并给出回滚坐标。不新增竞争账本，不把模型诊断当作必需前置。
> 机器可读清单：`docs/dependency_upgrade_canaries.json`（由 `tests/governance/test_dependency_upgrade_canaries.py` 校验）。

## 1. 分层

### 第 1 层：固定输入的结构与权限回归（0 Provider，必做）

适用范围：`requirements.lock`、`base-image.lock`、`contracts/method_packages.py`、模型路由配置、
Web/JS 资产发生任何变更。

- 宿主契约：`tests/research_agent/critical_regressions.txt`（CI 无过滤收集）。
- JS 合同：`python tools/run_js_contracts.py`（42 个 harness，含未注册项已清）。
- 启动/权限：`tests/webserver/test_web_execution_runtime_preflight.py`、
  `tools/check_agent_runtime.py --image <tag>`（镜像能力探针）。
- 镜像身份：重建后核对闭包身份与依赖锁摘要（`execution/kernel_identity.py`），更新部署包回执。

### 第 2 层：数值 oracle 回归（离线参考实现，必做于科学栈升级）

- 内核 oracle：`tests/research_agent/data/method_kernel_oracles.json` +
  `tests/research_agent/execution/test_method_kernel_r_oracles.py`（R: survival / pROC / EValue）。
  容差与场景由 oracle 文件与对应测试拥有，本政策不改写数值。
- 预测校验 oracle：`tests/research_agent/data/prediction_validation_r_oracle.json` 及
  `tests/research_agent/authority/test_prediction_validation_provenance.py`、
  `tests/research_agent/gates/test_prediction_validation_owner.py`。
- **已知未覆盖（不得当作已通过）**：`methods.rmst`、`methods.decision_curve` 无外部 oracle
  （见 kernel oracle 的 `_provenance.not_covered`）。二者在补 oracle 前不得进入可报告声明。
- 升级科学栈时必须同时跑底层库自身影响的每个 oracle；不能只跑 critical 清单。

### 第 3 层：模型/路由诊断（有明确授权与预算才做）

- `cross_model_panel` 等模型对照不是升级前置；仅当第 1/2 层通过、且研究需要答复模型行为问题时，
  以新的受治理运行执行（外部 LLM opt-in、预算、失败分母照常记录）。

## 2. 版本与许可字段（复用，不另建）

- 适配器声明：`scientific_adapters/runtime.py` 的 typed spec（版本范围、license、验证引用）。
- 科学包登记：`contracts/method_packages.py`；镜像直接依赖锁定于 `runner_image/requirements.lock`。
- base 镜像 digest：`runner_image/base-image.lock` + Dockerfile 双写；升级需审阅上游 manifest、跑镜像 CI 冒烟并重新生成 CycloneDX SBOM。
- 运行记录：收据含代码版本、镜像 id 与依赖快照（`authority/run_receipt.py`）。

## 3. 回滚坐标（升级前记录，失败即回退）

| 对象 | 坐标 |
|---|---|
| 镜像 | 旧 tag + immutable image id（部署包 `receipts/new-image-*.txt` 模式） |
| 依赖 | `requirements.lock` sha256（记录于本政策清单 `verified_against`） |
| 源码 | 升级前 `HEAD` 与工作树状态（提交升级需可 revert 的最小 patch） |
| 内核身份 | 升级前后 `identity_sha256`；不一致必须在部署回执中记录重建原因 |

## 4. 明确不做

- 不为升级引入新框架/SDK；不提高任何门限阈值来通过升级；不把第 3 层诊断计入九问完成度。
