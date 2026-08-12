# GPT 整仓审阅复核与 F1–F16 / N1–N18 定点修复

- 日期：2026-08-11–2026-08-12
- 分支 / 起始 HEAD：`fix/pi-workspace-review-20260809@e0ee97f`
- 实现提交：`3e03529`（首轮）+ `a3ff37a`（二次 N1–N4）+ `35bfc23`（终点语义）+ `d92c055`（Research Agent CI 根因）+ `ab58c5e`（N8/N9 first-stay）+ `d06c095`（N10/N11 MIMIC evidence）+ `6783de6`（N12/N13 age/API contract）+ `1210efb`（N14 HiRID age bins）+ `5c075e1`（N15–N18 endpoint/weight evidence）
- 状态：八轮定点修复均已提交；N1–N18 代码层已失败关闭，新 exact-head CI 待推送后复核，未启动 Provider、Canonical9 或真实患者分析
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
| N1 SICdb first ICU stay | **成立，HIGH scientific；第三次复核纠偏** | `CaseID` 不再等同首次入住；最终按 `OffsetAfterFirstAdmission` 的绝对语义裁决：`0` 且患者内唯一才是首次，`>0` 为再次入住，负值/缺失/并列零保持 unknown，不再把当前 extract 的最小正 offset 当首次。 |
| N2 nullable survival | **成立，HIGH scientific** | eICU hospital/unit discharge status 只把 `Alive/Expired` 映射为 nullable boolean；NULL 保持 unknown。AUMC destination 的 NULL 同步不再编码成存活；本地类别审计另发现荷兰语 `Overleden`，现显式映射为死亡。 |
| N3 detector fail-open | **成立，HIGH** | 新增公开 `DatabaseDetectionError(code, data_path, candidates)`；未知证据、冲突 marker、多数据库环境与显式无效 prepared path 均失败关闭，不再默认 `miiv` 或原样放行路径。 |
| N4 Pi security CI | **成立，merge blocker** | 远端 run `31559641320` 在 provider-error regression 因读取 runner 全局 Settings 而失败，后续 Node/XSS/Chromium 被跳过；测试现显式隔离 `ai_enabled`，workflow 纳入 multiprocess + workflow regression，并以 `if: !cancelled()` 保证普通 pytest 失败后仍运行三项安全证明。 |

官方依据：SICdb `cases` 文档明确每次 ICU admission 生成独立 `CaseID`，readmission 由 `PatientID` 与 `OffsetAfterFirstAdmission` 识别；eICU patient 表明确 hospital/unit discharge status 均允许 `Alive / Expired / NULL`。本地 SICdb v1.0.6 `d_references.parquet` 只读核验：`2026=Survived`、`2028=Deceased`、`3076=6 Months`、`3077=1 Year`；本轮不把动态 reference 解码扩成额外重构。

二次验证：数据/API 邻接 `88 passed`；Pi dedicated 清单（含 multiprocess/workflow regression）`126 passed`；Node 三 owner parse 与 renderer hostile vectors `6/6`；Chromium hostile-preview `passed=true`、1280px 无横溢出；真实本地六库路径识别/显式验证 `6/6`；Ruff、YAML parse、`git diff --check` 通过。`a3ff37a` 的 push/PR Pi security runs `31561506020` / `31561507835` 后续均为 `success`。

## 第三次 exact-head 复核（N1 / N5–N7 + Research Agent CI）

再次按官方字段定义核对后，N1 的“当前 extract 内求最小 offset”仍不符合 SICdb 的绝对 offset 语义；另确认 AUMC `destination` 是 ICU/MCU 出院终点、eICU `unitVisitNumber` 只在一次 hospitalization 内编号、`unitDischargeStatus` 也不能代替 hospital discharge survival。四项均成立并由 `35bfc23` 定点失败关闭。

| ID | 裁决 | 最终合同 |
|---|---|---|
| N1 SICdb positive-min edge | **成立，HIGH scientific** | `OffsetAfterFirstAdmission > 0` 始终为再次入住；仅唯一零 offset 可判首次；负值、缺失与重复零不猜。 |
| N5 AUMC survival endpoint | **成立，HIGH scientific** | `destination` 只产出 endpoint-specific `icu_survived`；通用 `survived` 定义为 hospital-discharge survival，在 AUMC 失败关闭。 |
| N6 eICU first stay scope | **成立，HIGH scientific** | `unitVisitNumber` 只产出 `first_unit_stay_within_hospitalization`；不再冒充 patient-global `first_icu_stay`。 |
| N7 eICU unit survival fallback | **成立，HIGH scientific** | `unitDischargeStatus` 只产出 `unit_survived`；缺少 `hospitalDischargeStatus` 时通用 `survived` 不可用。 |

`a3ff37a` 的 Research Agent run `31561507838` 在 Python 3.10/3.11 均红。日志中的高扇出根因有两个：新增的 `literature_citation_keys` 未进入显式科学 authority 分类，且内置 Mock Planner 收到 run-bound literature authority 后仍生成无 citation 的 scientific steps；固定 Planner prompt 同时为 `51,989 > 51,600` bytes。`d92c055` 将 citation key 纳入 step/plan signature、让 Mock Planner 只绑定提示词发布的 exact key，并压缩重复 guidance 而不抬预算，最终 fixed cost 为 `51,595` bytes。

第三次验证：数据/API 相邻 `91 passed`；Research Agent authority/literature/prompt/example 合同 `72 passed`；原 CI 失败面的 pipeline/resume/reviewer/sidecar 端到端 `4 passed`；architecture baseline diff `OK`；两组 Ruff、`git diff --check` 与 middle-layer progress lint 均通过。遵循开发测试策略，未在本地等待整个 Research Agent exact-head 矩阵；后续推送后的远端矩阵仍是冻结/合并前的最终门。

官方依据：SICdb [Cases](https://www.sicdb.com/Documentation/Table%3A_Cases)；eICU [patient](https://eicu.mit.edu/eicutables/patient/)；AmsterdamUMCdb [Table 1 patient characteristics notebook](https://github.com/AmsterdamUMC/AmsterdamUMCdb/blob/master/paper/paper-table1-patient-data-characteristics.ipynb)。

## 第四次复核（N8–N9）

N8/N9 均成立并由 `ab58c5e` 定点修复。AmsterdamUMCdb 官方 admissions 表明确 `admissioncount` 会随同一患者每次额外 ICU/MCU admission 递增，`admittedat` 则是相对首次 admission 的毫秒数；因此 partial extract 内重排不能签发首次入住。HiRID 官方明确每次 ICU (re-)admission 都生成新的 Patient ID，且发布数据无法识别多次 admission 是否来自同一患者，因此不能把全部 stay 断言为 patient-global first stay。

| ID | 裁决 | 最终合同 |
|---|---|---|
| N8 AUMC first stay | **成立，HIGH scientific** | 只使用原生 `admissioncount`：`1=True`、`>1=False`、`0`/负值/NULL/非法值 unknown；字段缺失时通用 `first_icu_stay` unavailable，不再按当前 extract 的 `admittedat` 排序。 |
| N9 HiRID first stay | **成立，HIGH scientific** | 不生成 `first_icu_stay`；请求 patient-global first stay 时沿用 typed `patient_filter_criterion_unavailable` 失败关闭。非阻塞的 `discharge_status → icu_survived` API completeness 未混入本修复。 |

第四次验证：三条新增 regression 在旧实现上 `3 failed`、修复后 `3 passed`；数据/API 相邻 `94 passed`；Ruff 与 `git diff --check` 通过。复核时 `79f9a81` 的 Research Agent push/PR runs `31567323098` / `31567325771` 和 main CI runs `31567323110` / `31567325792` 仍为 `in_progress`，Pi security 与 runner image trust 已成功；不得把 pending matrix 写成 closed。

官方依据：AmsterdamUMCdb [admissions table](https://github.com/AmsterdamUMC/AmsterdamUMCdb/blob/master/tables/admissions.ipynb)；HiRID [data details](https://hirid.intensivecare.ai/data-details)。

## 第五次复核（N10–N11）

N10/N11 均成立并由 `d06c095` 定点修复。MIMIC-III/IV 的 `subject_id` 是患者、`icustay_id`/`stay_id` 是单次 ICU stay；prepared extract 内按 `subject_id + intime` 排序只能得到当前切片的相对顺序，不能证明 patient-global 首次 ICU。MIT-LCP 官方 `mimic-code` 中同名 `first_icu_stay` 还明确按 `hadm_id` 排名，语义是当前 hospitalization 内首次 ICU，而不是 EasyICU 通用合同采用的 patient-global 首次 ICU。

| ID | 裁决 | 最终合同 |
|---|---|---|
| N10 MIMIC first stay | **成立，HIGH scientific** | MIMIC-III/IV 不再从当前 extract 的 `subject_id + intime` 生成通用 `first_icu_stay`；缺少绝对 patient ICU ordinal 或完整历史回执时，该 criterion 走 typed `patient_filter_criterion_unavailable`。本轮不新增同样受 partial-extract 影响的 hospitalization-level 派生字段。 |
| N11 MIMIC survival | **成立，HIGH scientific** | 只接受标准 `hospital_expire_flag`：`0=True`、`1=False`，NULL、解析失败、非 0/1 及 left-merge 未匹配均为 nullable unknown；删除无法区分“确认匹配且 NULL”与“整行缺失”的 `deathtime.isna()` fallback。缺标准 flag 时通用 `survived` unavailable。 |

第五次验证：六条 MIMIC-III/IV 参数化边界在旧实现上 `6 failed / 34 deselected`，修复后 `6 passed / 34 deselected`；数据正确性、profile/resource 与 cohort API 相邻门 `100 passed`；Ruff 与 `git diff --check` 通过。`c51003a` 的 PR runs `31568665528`（Research Agent）/`31568665508`（main CI）及 push main CI `31568662145` 仍为 `in_progress`，Pi security `31568665509` 与 runner trust `31568665500` 已成功；新提交 `d06c095` 尚未推送，故不存在可宣称绿色的新 exact-head 矩阵。

官方依据：MIMIC-IV [ICU stays](https://mimic.mit.edu/docs/iv/modules/icu/icustays.html)、[admissions](https://mimic.mit.edu/docs/IV/modules/hosp/admissions.html)；MIMIC-III [ICU stays](https://mimic.mit.edu/docs/III/tables/icustays.html)、[admissions](https://mimic.mit.edu/docs/iii/tables/admissions.html)；MIT-LCP `mimic-code` 的 [MIMIC-IV icustay_detail](https://github.com/MIT-LCP/mimic-code/blob/main/mimic-iv/concepts/demographics/icustay_detail.sql) 与 [MIMIC-III icustay_detail](https://github.com/MIT-LCP/mimic-code/blob/main/mimic-iii/concepts/demographics/icustay_detail.sql)。

## 第六次复核（N12–N13）

N12/N13 均成立并由 `6783de6` 定点修复。此前只有 AUMC 保留发布年龄带；eICU `>89`、MIMIC-III shifted DOB、MIMIC-IV `anchor_age=91` 和 SICdb 5 年分组/90 顶码仍会被压成貌似精确的 scalar，随后由公共 `age_min` / `age_max` 静默改写队列。`FilterCriteria.admission_type` 同时没有对应的 `PatientFilter.filter()` 参数或执行路径，属于公开 schema 漂移。

| ID | 裁决 | 最终合同 |
|---|---|---|
| N12 cross-database age | **成立，HIGH scientific** | 六库人口学 loader 统一发布 `age`、`age_lower`、`age_upper`、`age_is_grouped`、`age_is_censored`。精确年龄上下界相等；eICU/MIMIC 高龄为有下界、无上界；AUMC 保留原生年龄带；SICdb 依据官方“±5y，over 90 set to 90”采用保守 `age±5` 区间，90 的上界开放。所有数据库走同一 interval-aware filter：整段满足才纳入、整段不满足才排除、阈值切段时抛 `patient_filter_grouped_age_indeterminate`。 |
| N13 admission type drift | **成立，MEDIUM API** | 删除未实现的 `FilterCriteria.admission_type` 和模块“支持入院类型”声明；新增 dataclass 字段与真实 filter 参数一致性回归。concept catalog / Web 中独立存在的入院类型展示或概念不属于这个悬空 PatientFilter API，未扩大删除范围。 |

第六次验证：五条新增回归在旧实现上 `5 failed / 40 deselected`，修复后连同 AUMC 既有边界为 `7 passed / 38 deselected`；`test_patient_filter_correctness.py` 全量 `45 passed`；数据正确性、profile/resource、cohort API 与 callback 邻接门 `245 passed`；Ruff、`git diff --check` 通过。本地只读核验 SICdb v1.0.6 `AgeOnAdmission` 的非空发布值恰为 15–90 的 5 年步长，未输出患者标识。`216426d` 的 Research Agent PR run `31569536019` 与 main CI PR/push runs `31569536017` / `31569533396` 仍为 `in_progress`，Pi security `31569536042` 与 runner trust `31569536071` 已成功；新提交 `6783de6` 尚未推送，故尚无它自己的 exact-head 矩阵。

官方依据：eICU [patient age](https://eicu.mit.edu/eicutables/patient/)；MIMIC-III [patients DOB de-identification](https://mimic.mit.edu/docs/III/tables/patients.html) 与 [official age tutorial](https://mimic.mit.edu/docs/III/tutorials/intro-to-mimic-iii.html)；MIMIC-IV [patients anchor age](https://mimic.mit.edu/docs/IV/modules/hosp/patients.html)；SICdb [case data](https://www.sicdb.com/Documentation/Table%3A_Cases) 与 [full documentation](https://www.sicdb.com/Documentation/SICdb_Documentation)。

## 第七次复核（N14）

N14 成立并由 `1210efb` 定点修复。HiRID 官方与 PhysioNet 均明确年龄、身高和体重按 5 年/单位分箱，年龄最大档为 90 且包含全部更高年龄；此前 HiRID loader 却发布 `age_lower == age_upper == age`、`grouped=False`、`censored=False`，会让公共年龄阈值把匿名化档位当作精确年龄。

| ID | 裁决 | 最终合同 |
|---|---|---|
| N14 HiRID age bins | **成立，HIGH scientific** | 普通 HiRID 年龄档发布保守 `age±5` 区间并标记 grouped；90 档发布 `[85, open]` 并标记 grouped+censored。官方没有公开档位标签表示左边界、右边界还是中心，因此不用未经证实的窄区间，采用覆盖三种解释的 envelope；统一 evaluator 对切穿普通档或 90 顶档的阈值抛 `patient_filter_grouped_age_indeterminate`，只有整段可判定时才纳排。 |

第七次验证：两条新增回归在旧实现上 `2 failed / 45 deselected`，修复后 `2 passed / 45 deselected`；`test_patient_filter_correctness.py` 全量 `47 passed`；数据正确性、profile/resource、cohort API 与 callback 相邻门 `209 passed`；Ruff 与 `git diff --check` 通过。远端 exact head `c1b7924` 的 Pi security run `31570935158` 与 runner trust `31570935056` 已成功，Research Agent run `31570935036`、PR/push main CI `31570934935` / `31570932350` 仍在运行；新提交 `1210efb` 尚未推送，因此不存在可宣称绿色的新 exact-head 矩阵。

官方依据：HiRID [data details](https://hirid.intensivecare.ai/data-details)；PhysioNet [HiRID v1.0 anonymization procedure](https://physionet.org/content/hirid/1.0/)。

## 第八次复核（N15–N18）

N15–N18 均成立并由 `5c075e1` 定点修复。旧实现把单次 ICU stay 的 LOS 当作完整 28 天 ICU 轨迹、用当前 extract 内排序推断再入院、以 eICU 院内死亡代替 28 天死亡，并在 KDIGO 尿量率计算中从冲突体重里取首条；这些路径都会在证据不足时静默发布临床结论。

| ID | 裁决 | 最终合同 |
|---|---|---|
| N15 ICU-free days 28 | **成立，HIGH scientific** | 六库均不再从 index stay LOS 直接发布 `icu_free_days_28`。只有具备完整 28 天 ICU 进出轨迹及终点所需生存证据后才可恢复；当前通过 outcome-availability owner 明确报告 structurally unavailable。 |
| N16 MIMIC ICU readmission | **成立，HIGH scientific** | MIMIC-III/IV 不再按当前 extract 的 `subject_id + intime` 排序生成 `icu_readmission`；缺少原生绝对 ICU ordinal 或 patient-history completeness 回执时失败关闭。 |
| N17 eICU ventilator-free days 28 | **成立，HIGH scientific** | `actualHospitalMortality` 是院内死亡而非 28 天死亡，不能与 `actualVentdays` 拼成正式 VFD28；eICU 的 `vent_free_days_28` 现明确 unavailable，既有 horizon mortality 仍按各自证据合同发布。 |
| N18 KDIGO keyed weight | **成立，HIGH scientific** | `urine_weight_linkage.py` 作为依赖中立 owner，只接受同一实体唯一一致的正体重；重复相同值可折叠，多个不同有效值抛 `kdigo_weight_values_conflict`。事件体积与 HiRID rate-source 两条 KDIGO 路径共用该合同，不再依赖行顺序，也不在缺临床选择规则时擅取首条/中位数。 |

第八次验证：七条新增/调整回归在旧实现上 `6 failed / 1 passed`（相同重复体重是负向对照），修复后 `7 passed`；outcome availability、outcomes、数据正确性、KDIGO、native export 与 Web patient feature 相邻门 `112 passed / 9 real-data skipped`；API/cohort/module ownership/concept catalog/extract/callback 邻接门 `60 passed`；Ruff 与 `git diff --check` 通过。按开发测试策略未在本地启动 full matrix，新 exact-head 以推送后 GitHub Actions 为准。

官方依据：eICU [`apachePatientResult`](https://eicu.mit.edu/eicutables/apachepatientresult/) 将 `actualHospitalMortality` 定义为院内死亡、`actualVentdays` 定义为实际通气天数且 30 天顶码；ICU-free days 试验方案说明 ICU 再入院后应从最终 ICU 出院计算、28 天前死亡计 0（[protocol](https://pmc.ncbi.nlm.nih.gov/articles/PMC6687016/)）；MIMIC-III [`icustays`](https://mimic.mit.edu/docs/iii/tables/icustays/) 区分 hospitalization `HADM_ID` 与单次 ICU stay `ICUSTAY_ID`；VFD28 的正式定义要求患者存活且脱离有创机械通气（[PubMed](https://pubmed.ncbi.nlm.nih.gov/41361939/)）。
