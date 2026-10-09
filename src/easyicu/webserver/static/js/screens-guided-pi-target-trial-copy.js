/* Owner: the wording of the target trial card and its conversation rows.

   One line per stable code the host's target trial owners can return
   (webserver/target_trial_card.TARGET_TRIAL_UI_CODES): what happened, then
   what to do next, without repeating the values the host's detail carries.  A
   line that needs that detail says the card shows it below.  The review
   session drafted the lines; a test keeps this table and the host's codes
   equal in both directions, and a code missing here at runtime falls back to
   the host's English detail.  The protocol's own text is English and inside
   the approved record's digest, so only its column names are translated. */
(function () {
  'use strict';

  const LINES = Object.freeze({
    tte_treatment_not_offered: ['The treatment\'s concept records no event status, so the hour it starts cannot be read. Restate the trial with a treatment concept that records when it starts.', '所述治疗概念不记录事件状态，读不出开始的时刻。请在对话里改用能记录开始时刻的治疗概念。'],
    tte_treatment_capture_undeclared: ['The capture registry does not state that a missing record means the treatment was not given in this database, so deferring cannot be told apart from a missing record. Restate the trial with a treatment whose capture is declared.', '采集登记没有声明在本库中“无记录即未用该治疗”，“延迟”因此与记录缺失分不开。请改用采集已声明的治疗概念。'],
    tte_treatment_class_unknown: ['The capture registry names no such treatment class. Restate the trial with one of the classes it names (listed below).', '采集登记里没有所述的治疗类别。请改用登记列出的类别（见下方原句）。'],
    tte_treatment_definition_partial: ['The treatment\'s concepts miss some drugs of its class, so a stay given only those would count as deferring. Add concepts that record them, or narrow the class.', '治疗概念漏掉了该类别中的部分药物，只用这些药物的入住会被算作“延迟”。请补上记录这些药物的概念，或缩小类别。'],
    tte_strategy_not_typed: ['A strategy states more than starting or deferring the treatment, which this version cannot emulate. Restate the strategies as starting within the grace period or not.', '策略中有“开始或延迟”之外的规定，本版本无法模拟。请把策略改述为在宽限期内开始或不开始治疗。'],
    tte_time_zero_not_offered: ['The time zero is not one the host offers. Restate it as a whole hour within the offered range (below).', '时间零点不在可选范围内。请改为范围内的整点小时（见下方原句）。'],
    tte_eligibility_after_time_zero: ['Some population criteria are decided only after time zero, so eligibility would use information from the future. Move time zero later, or restate those criteria so they are decided by time zero.', '有入选条件要到时间零点之后才能判定，入选会用到未来的信息。请推后时间零点，或改述这些条件，使其在零点前可判定。'],
    tte_grace_period_not_offered: ['The grace period is longer than the host offers. Restate it within the offered range (below).', '宽限期超过可选的最长时长。请改为范围内的小时数（见下方原句）。'],
    tte_endpoint_unsupported: ['The outcome is not a fixed-horizon death endpoint this database follows up. Restate it as one, such as death by day 28.', '结局不是本库可随访的固定时限死亡终点。请改为这类终点，例如 28 天死亡。'],
    tte_horizon_within_grace: ['The outcome\'s horizon ends within the grace period, before the strategies can differ. Shorten the grace period or choose a longer horizon.', '结局时限在宽限期内就已结束，两种策略还来不及分开。请缩短宽限期，或改用更长的时限。'],
    tte_indication_unstated: ['No indication is stated: the inclusion criterion under which either strategy is plausible for every eligible stay. Name it in the conversation.', '未指明适应证，即哪条入选条件使每个合格入住都可能采用任一策略。请在对话里指明。'],
    tte_indication_not_in_population: ['The indication names a criterion the population does not state. Name one of the population\'s criteria, or add it to the population.', '适应证引用的条件不在人群陈述中。请引用人群里已有的条件，或先把它加入人群。'],
    tte_indication_not_inclusion: ['The indication names an exclusion; an indication keeps stays. Name an inclusion criterion instead.', '适应证引用的是排除条件，而适应证应保留入住。请改为引用入选条件。'],
    tte_indication_not_clinical: ['The indication\'s criterion is not a clinical state, such as an age or a length of stay. Name a condition, measurement or diagnosis criterion.', '适应证引用的条件不是临床状态（例如年龄或住院时长）。请改为引用疾病、测量值或诊断类条件。'],
    tte_indication_not_applied: ['The population cannot apply the indication\'s criterion (reason below), so the indication does not hold. Restate that criterion so it can be applied.', '人群无法执行适应证引用的条件（原因见下方原句），适应证因此不成立。请改述该条件，使其能够执行。'],
    tte_death_time_not_hourly: ['This database records death without an hour, so a death in the grace period cannot be ordered against censoring. The database decides this; the trial cannot be emulated in it.', '本库记录的死亡时间不到小时，宽限期内的死亡无法与删失排定先后。这由数据库决定，本库无法模拟该试验。'],
    tte_icu_exit_undefined: ['Data Extraction defines no ICU length of stay for this database, so follow-up cannot be censored at ICU discharge. The database decides this; the trial cannot be emulated in it.', '数据提取在本库中没有定义 ICU 住院时长，无法在出 ICU 时删失。这由数据库决定，本库无法模拟该试验。'],
    tte_no_confounder_carried: ['None of the stated confounders can be adjusted for at time zero, so the comparison would be unadjusted. State confounders observed before time zero.', '所述混杂因素没有一个能在时间零点调整，比较将不作任何调整。请补充在零点前可观测的混杂因素。'],
    tte_treatment_onset_not_materialized: ['The data holds no treatment start read from ICU admission through the grace period. Extract data that includes the treatment, then state the trial again.', '数据中没有从入 ICU 到宽限期结束的治疗开始时刻。请重新提取包含该治疗的数据，再说明试验。'],
    tte_grace_beyond_capture: ['The treatment\'s start is read over a window that ends before the grace period does. Shorten the grace period, or extract data that covers it.', '治疗开始时刻的读取窗口在宽限期结束前就截止了。请缩短宽限期，或重新提取覆盖整个宽限期的数据。'],
    tte_endpoint_not_materialized: ['The data holds no outcome event or its follow-up time. Extract data that includes the outcome, then state the trial again.', '数据中没有结局事件或其随访时间。请重新提取包含该结局的数据，再说明试验。'],
    tte_indication_requires_extraction: ['The indication\'s criterion reads data not yet extracted (reason below). Extract it, then state the trial again.', '适应证引用的条件所需数据尚未提取（原因见下方原句）。请提取后再说明试验。'],
    tte_death_time_not_materialized: ['The data holds no death status with its time. Extract data that includes them, then state the trial again.', '数据中没有死亡状态及其时间。请重新提取包含它们的数据，再说明试验。'],
    tte_icu_exit_unavailable: ['The data holds no ICU length of stay, so follow-up cannot be censored at ICU discharge. Extract data that includes it, then state the trial again.', '数据中没有 ICU 住院时长，无法在出 ICU 时删失。请重新提取包含它的数据，再说明试验。'],
    tte_patient_identity_unavailable: ['A patient may contribute several stays and the data identifies no patient, so the bootstrap cannot resample patients. Extract data with patient identity, or keep each patient\'s first ICU stay.', '同一患者可能有多次入住，而数据中没有患者标识，自助法无法按患者重抽样。请提取带患者标识的数据，或只保留每位患者的首次 ICU 入住。'],
    tte_confounders_require_extraction: ['No confounder is observed by time zero in this data; an extraction would carry those listed below. Extract them, then state the trial again.', '当前数据中没有混杂因素在时间零点前可观测；重新提取可纳入下方所列的混杂因素。提取后再说明试验。'],
    tte_confounder_unavailable: ['This confounder is neither a column of the data nor a concept Data Extraction defines, so it is not adjusted for. Remove it, or name a defined concept.', '该混杂因素既不是数据中的列，也不是数据提取定义的概念，因此不作调整。请删去，或改用已定义的概念。'],
    tte_confounder_is_design_concept: ['This confounder is the treatment, the outcome, a death or the ICU stay itself; the strategies account for it, not the weights. Nothing to do.', '该混杂因素就是治疗、结局、死亡或 ICU 入住本身，已由策略处理，不进入权重。无需处理。'],
    tte_confounder_after_time_zero: ['The host cannot prove this confounder observed by time zero, so it is not adjusted for. Remove it, or name a measurement taken before time zero.', '宿主无法证明该混杂因素在时间零点前已观测，因此不作调整。请删去，或改用零点前测得的指标。'],
    tte_confounder_not_in_export: ['The data does not hold this confounder; an extraction summarizing it before time zero would. Extract it, then state the trial again.', '数据中没有该混杂因素；提取其零点前的汇总值后即可纳入。请提取后再说明试验。'],
    tte_confounder_window_after_time_zero: ['This confounder is summarized past time zero in the data; summarized before time zero it would be observed by then. Extract it over that window, then state the trial again.', '数据中该混杂因素的汇总窗口越过了时间零点；改为零点前汇总即可纳入。请按该窗口重新提取，再说明试验。'],
    tte_trial_not_confirmed: ['This study has no approved target trial, so no plan is generated. State the trial in the conversation and approve it on its card.', '这项研究还没有批准的目标试验，所以不生成计划。请在对话里说明试验，并在试验卡片上批准。'],
    population_inclusion_requires_extraction: ['This population criterion can be applied only to data extracted for this study\'s population (reason below). Ask in the conversation to extract that data, then state the trial again.', '这个人群条件只有按本研究人群重新提取数据后才能施加（原因见下方原句）。请在对话里要求按本研究人群提取数据，再说明试验。'],
    population_inclusion_not_applied: ['This population criterion cannot be applied as written (reason below). Revise it in the conversation, then state the trial again.', '这个人群条件按原文无法施加（原因见下方原句）。请在对话里改述这个条件，再说明试验。'],
    target_trial_statement_invalid: ['The stated trial or population falls outside the host\'s options in the fields named below. Restate those fields in the conversation.', '试验或人群陈述在下列字段上超出了可选范围。请在对话里改述这些字段。'],
    target_trial_family_mismatch: ['A target trial is stated only for a causal-inference study. If this study estimates a causal effect, declare its analysis type as causal inference in the conversation first; otherwise it needs no target trial.', '目标试验只用于因果推断研究。如果这项研究要估计因果效应，请先在对话里把分析类型改为因果推断；否则不需要目标试验。'],
    target_trial_export_required: ['The study has no prepared data package to compile the trial on. Extract the study\'s data first.', '研究还没有准备好的数据包，无法编译试验。请先提取研究数据。'],
    target_trial_compile_busy: ['Another target trial is compiling. State this one when it finishes.', '另一个目标试验正在编译。请等它完成后再说明这个试验。'],
    target_trial_database_out_of_scope: ['This version emulates target trials in MIMIC-IV only. Use a MIMIC-IV data package for this study.', '本版本只在 MIMIC-IV 中模拟目标试验。请为研究改用 MIMIC-IV 的数据包。'],
    target_trial_study_not_ready: ['A run of this study would refuse to start, so the trial was not compiled. Complete the study\'s setup, then state the trial again.', '研究设置尚不完整，运行会被拒绝启动，因此没有编译试验。请先补全研究设置，再说明试验。'],
    target_trial_data_unavailable: ['The study\'s data package does not provide concepts the trial reads (listed below). Extract data that includes them, or restate the trial.', '研究的数据包缺少试验要读的概念（见下方列表）。请提取包含它们的数据，或改述试验。'],
    target_trial_population_not_compiled: ['The stated population does not compile at the trial\'s time zero. Restate the population in the conversation.', '所述人群在试验的时间零点上无法编译。请在对话里改述人群。'],
    target_trial_study_changed: ['The study changed while the trial compiled, so this result was discarded. State the trial again.', '编译期间研究发生了改动，本次结果作废。请重新说明试验。'],
    target_trial_compile_failed: ['The compile failed unexpectedly (cause below). State the trial again to retry; if it fails again, report the cause code.', '编译意外失败（原因码见下方）。可再说明一次试验重试；若再次失败，请反馈原因码。'],
    target_trial_compile_interrupted: ['The compile job was interrupted and left no result (for example, the host restarted). State the trial again in the conversation.', '编译作业中断，没有留下结果（例如宿主重启）。请在对话里重新说明试验。'],
    easyicu_target_trial_compile_submitted: ['Compiling the target trial on the study\'s data. The card shows the result when it finishes.', '正在用研究数据编译目标试验，完成后卡片会显示结果。'],
    target_trial_approved: ['Target trial approved. The plan will be generated on the study\'s data for this trial.', '已批准目标试验。计划将按此试验在研究数据上生成。'],
    target_trial_statement_needed: ['A causal study states the target trial it emulates before its plan. Describe the treatment, strategies, time zero, grace period, outcome, indication and confounders in the conversation.', '因果研究在生成计划前要先说明所模拟的目标试验。请在对话里描述治疗、策略、时间零点、宽限期、结局、适应证和混杂因素。'],
    target_trial_review: ['The target trial is not approved yet. Check its card; the plan waits for the approval.', '目标试验尚未批准。请查看试验卡片，批准后才能生成计划。'],
    target_trial_plan_ready: ['The target trial is approved. Generate the plan on this study\'s data.', '目标试验已批准，可以按本研究的数据生成计划。'],
    target_trial_restatement_pending: ['A newer statement of the trial has not finished compiling, so this version cannot be approved. Review the new card when it finishes.', '更新的试验陈述还在编译，这一版不能批准。请等新卡片出来后再核对。'],
    target_trial_approval_record_mismatch: ['The card shows another version of the trial than the study holds. Refresh the card and review it before approving.', '卡片显示的不是研究当前这一版试验。请刷新卡片，核对后再批准。'],
    target_trial_design_invalid: ['The confirmed lines do not match the record, or the record cannot be approved. Confirm every line, then approve.', '已确认的行数与记录不符，或该记录不可批准。请逐行确认后再批准。'],
    target_trial_record_missing: ['The host no longer keeps this trial\'s compile record. State the trial again in the conversation.', '宿主已找不到这一版试验的编译记录。请在对话里重新说明试验。'],
    study_job_running: ['A job is already running for this study. Try again when it ends.', '该研究已有作业在运行。请等它结束后再试。'],
    job_capacity_exceeded: ['This machine is running as many jobs as it allows. State the trial again when one finishes.', '本机同时运行的作业已满。请等某个作业结束后再说明试验。'],
    study_context_revision_conflict: ['The study changed after it was read. Refresh and try again.', '读取之后研究被修改了。请刷新后再试。'],
    study_context_active_job_conflict: ['A job is running for this study. Approve the trial when it ends.', '研究有作业在运行。请等它结束后再批准试验。'],
    host_action_study_mismatch: ['This conversation is bound to another study. Act from that study\'s conversation.', '此对话绑定的是另一个研究。请回到对应研究的对话里操作。'],
  });
  // The protocol rows, in the order the host lists them.
  const ITEM_LABELS = Object.freeze({
    eligibility: ['Eligibility', '入选标准'],
    treatment_strategies: ['Treatment strategies', '治疗策略'],
    assignment: ['Assignment', '分配'],
    time_zero: ['Time zero', '时间零点'],
    follow_up: ['Follow-up', '随访'],
    outcome: ['Outcome', '结局'],
    causal_contrast: ['Causal contrast', '因果对比'],
    analysis_plan: ['Analysis plan', '分析计划'],
  });
  const LIMITATION_LABELS = Object.freeze({
    pre_admission_use_not_visible: ['Use before ICU admission not visible', '看不到入 ICU 前的用药'],
    baseline_only_weights: ['Baseline-only weights in the grace period', '宽限期内只按基线加权'],
    treatment_outside_icu_not_recorded: ['Starts after ICU exit not recorded', '出 ICU 后的开始不被记录'],
    grace_period_deaths_in_both_strategies: ['Grace-period deaths count in both strategies', '宽限期内死亡计入两种策略'],
    not_typed: ['Not emulated', '未模拟的陈述'],
    evidence_ceiling: ['Evidence ceiling: analysis only', '证据上限：仅为分析'],
  });

  function pick(table, code, tr) {
    const row = Object.prototype.hasOwnProperty.call(table, code) ? table[code] : null;
    return row ? tr(row[0], row[1]) : '';
  }

  window.EasyICU.guidedPi.declare('targetTrialCopy', Object.freeze({
    CODES: Object.freeze(Object.keys(LINES)),
    line: (code, tr) => pick(LINES, String(code || ''), tr),
    itemLabel: (item, tr) => pick(ITEM_LABELS, String(item || ''), tr) || String(item || ''),
    limitationLabel: (code, tr) => pick(LIMITATION_LABELS, String(code || ''), tr),
  }));
})();
