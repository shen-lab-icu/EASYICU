# GPT 整仓审阅复核与 F1–F16 定点修复

- 日期：2026-08-11
- 分支 / 起始 HEAD：`fix/pi-workspace-review-20260809@e0ee97f`
- 实现提交：`3e03529`（首轮）+ `a3ff37a`（二次 N1–N4）
- 状态：两轮修复均已提交；复核时远端已由并发流程更新到 `a3ff37a`，exact-head CI 运行中；未启动 Provider、Canonical9 或真实患者分析
- 模块：`DATA-FIX1` / `WEBAPP-FASTAPI-NATIVE-QA` / `FIG2-CANONICAL9-GATE`

## 裁决

审阅中的 F1–F16 均有可复现依据；其中 F9、F10、F12、F13 会改变科学终点，F6/F7 会改变纳入队列，必须修。两点需要限定：SOFA-2 六库 `mapping_only` 是当前已披露的验证上限，不是新代码缺陷；尿量使用何种体重是应另立 protocol/golden vector 的治理建议，本轮没有证据支持顺手改变其科学政策。

F5 对 alias 漂移的判断成立，但“立即删除所有 detector”是架构建议，不宜与正确性热修混成无表征重写。本轮把实际进入公共 cohort/loader/outcome 路径的 `miii`、`sicdb`、`mimic-iv` 先经 typed registry 归一化，并用回归锁定；其余旧能力 allowlist 的整体退役继续作为独立架构债，不伪称已一口气删除。

| ID | 复核 | 修复 / 证据 |
|---|---|---|
| F1 | 正确 | 不刷新 baseline；把 Planner prompt/citation 投影、structured-retry 进度和 scientific runtime authority binding 移交小 owner。`arch_measure.py` 当前所有 lower-is-better 指标无回退：`pipeline.py -2 LOC`、`agents/core.py -13 LOC`、`execution/phase.py -2 LOC`。 |
| F2 | 正确 | demo PNG 从被 `demo/` ignore 改为明确 package data；wheel 实包包含 93,214-byte 文件，SHA-256 `34a46b54558a6f08cc02434a6958558ecb8077abd59db78713ef8f9dd4172e4b`。 |
| F3–F5 | 首轮主体正确；二次见 N3 | `detect_database_type(data_path=None)` 契约统一；MIMIC-III/IV 先看 `icustays` schema，`admissions.parquet` 不再冒充 AUMC marker；Loader、PatientFilter、outcomes、`load_src_cfg` 入口使用 registry canonical key。首轮遗留的 unknown→MIIV 已由 `a3ff37a` 关闭。 |
| F6–F7 | 首轮大部分关闭；二次见 N1/N2 | 请求 age/LOS/gender/first-stay/survival 而字段缺失或全空时抛 typed `PatientFilterCriterionError`；Sepsis-3 unknown-as-negative 已关闭。SIC first-stay 与 nullable survival 残余已由 `a3ff37a` 关闭。 |
| F8 | 正确 | eICU 年龄先去空白，再把完整 token `>89` 映射为 90。 |
| F9–F13 | 正确 | SICdb 使用 `AgeOnAdmission`；ICU LOS=`(TimeOfStay-ICUOffset)/3600`；出院生存读 `HospitalDischargeType`；free-days 使用 ICU 原点；6-month follow-up 下 `mort_365d=NA`，不当作存活。字段/代码经官方文档与本地 `cases.parquet`/`d_references` 交叉核对。 |
| F14 | 正确 | cache 默认改为 opaque full-SHA Parquet；pickle 仅显式 `use_pickle=True`，文件后缀标记 `.trusted.pkl`，不再把 source/concept 路径片段写入文件名。 |
| F15 | 正确 | AUMC 保留 `age_lower`/`age_upper`；阈值切开已发布年龄带时返回 `patient_filter_grouped_age_indeterminate`，不以 midpoint 假装精确年龄。 |
| F16 | 实现关闭；二次见 N4 | workspace CAS 与 project↔StudyContext authority 的 read-modify-write 同时受进程内 RLock 和 OS-released file lock；两进程竞争测试证明只允许一个 create/bind 成功。首轮 dedicated CI 未纳入该回归，已由 `a3ff37a` 补齐。 |

SICdb 语义来源：<https://www.sicdb.com/Documentation/Table%3A_Cases>、<https://www.sicdb.com/Documentation/Offsetable>。

## Owner / public contract

- `patient_filter.py` 继续拥有队列条件应用；新公开诊断为 `PatientFilterCriterionError(code, criterion)`，错误不可由调用方解释成空队列。
- `orchestration/scientific_runtime.py` 只编译并分发 immutable runtime authority pair，保留 trajectory/current-case owner 的精确异常。
- `orchestration/progress.py` 只把 typed structured-retry lifecycle 投影为有界 UI 进度，不携带模型响应或 validator 私有详情。
- `pi_copilot/locking.py` 只拥有跨进程 exclusive-lock primitive；workspace 与 project-authority 各自提供稳定 owner code。
- cache owner 默认只读写 DataFrame Parquet；trusted-local pickle 是调用者显式选择，不是隐含信任。

## 验证

- 规范 `.venv` Python 3.11.15 聚焦门共 `243 passed / 9 real-data skipped`：数据正确性 `59`、registry/resource/column-metadata 邻接 `51`、outcome/death 邻接 `5/9 skipped`、Research Agent owner/authority/prompt/architecture `74`、Pi workspace/provider/gateway/demo `54`。
- 独立 `arch_measure.py --diff tools/arch_baselines/execution_phase.json` 为 `OK`；Pi 组含两条 spawn 多进程竞争回归和 provider-error sanitization 真正到达目标路径。
- Node 三 owner parse-check 通过；renderer hostile-input `6/6`；Chromium hostile-preview isolation `passed=true`，1280px 无横溢出。
- wheel 构建成功并实查包含 demo PNG；`ruff check`、`git diff --check` 通过。

按 E1/Web 开发策略未运行 full exact-head CI；稳定 checkpoint、合并或正式实验冻结时再运行一次。本轮修复已以 `3e03529` 独立提交；当前工作树原有并行 Research Agent/Pi 改动均保留，未替用户提交或覆盖。

## 二次 exact-head 复核（N1–N4）

对 `185db3a` 的远端与官方字典复核确认首轮仍有四个 blocker；已由提交 `a3ff37a` 定点修复。收尾复核时远端已由并发流程更新到该 SHA，但 exact-head CI 尚在运行，暂不宣称绿色。

| ID | 裁决 | `a3ff37a` 修复 |
|---|---|---|
| N1 SICdb first ICU stay | **成立，HIGH scientific** | `CaseID` 不再等同首次入住；按 `PatientID + OffsetAfterFirstAdmission` 求患者级最早 admission，缺字段/缺值保持 unknown，并列最早 offset 不猜。 |
| N2 nullable survival | **成立，HIGH scientific** | eICU hospital/unit discharge status 只把 `Alive/Expired` 映射为 nullable boolean；NULL 保持 unknown。AUMC destination 的 NULL 同步不再编码成存活；本地类别审计另发现荷兰语 `Overleden`，现显式映射为死亡。 |
| N3 detector fail-open | **成立，HIGH** | 新增公开 `DatabaseDetectionError(code, data_path, candidates)`；未知证据、冲突 marker、多数据库环境与显式无效 prepared path 均失败关闭，不再默认 `miiv` 或原样放行路径。 |
| N4 Pi security CI | **成立，merge blocker** | 远端 run `31559641320` 在 provider-error regression 因读取 runner 全局 Settings 而失败，后续 Node/XSS/Chromium 被跳过；测试现显式隔离 `ai_enabled`，workflow 纳入 multiprocess + workflow regression，并以 `if: !cancelled()` 保证普通 pytest 失败后仍运行三项安全证明。 |

官方依据：SICdb `cases` 文档明确每次 ICU admission 生成独立 `CaseID`，readmission 由 `PatientID` 与 `OffsetAfterFirstAdmission` 识别；eICU patient 表明确 hospital/unit discharge status 均允许 `Alive / Expired / NULL`。本地 SICdb v1.0.6 `d_references.parquet` 只读核验：`2026=Survived`、`2028=Deceased`、`3076=6 Months`、`3077=1 Year`；本轮不把动态 reference 解码扩成额外重构。

二次验证：数据/API 邻接 `88 passed`；Pi dedicated 清单（含 multiprocess/workflow regression）`126 passed`；Node 三 owner parse 与 renderer hostile vectors `6/6`；Chromium hostile-preview `passed=true`、1280px 无横溢出；真实本地六库路径识别/显式验证 `6/6`；Ruff、YAML parse、`git diff --check` 通过。`a3ff37a` 的 push/PR Pi security runs `31561506020` / `31561507835` 已触发，记录收尾时为 `in_progress`，当前不宣称绿色。
