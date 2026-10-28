/* Owner: shared artifact reader vocabulary. */
/* Reader-facing names for the host codes the shared artifact renderers show:
   plan-review findings and the questions their decisions ask, result-table
   columns, their coded values and titles, method cards and the design
   elements a method source supports.  Text only -- nothing here
   derives, rounds, or reinterprets a registered value. */
(function () {
  'use strict';

  // The renderers' own i18n global decides the language, so a reader never
  // sees a table in one language and its column names in another.
  function pick(pair) {
    return typeof window.t === 'function' ? window.t(pair[0], pair[1]) : (window.EU_LANG === 'zh' ? pair[1] : pair[0]);
  }
  function zh() { return pick(['en', 'zh']) === 'zh'; }

  // [Chinese title, Chinese detail].  English readers keep the host's own
  // finding message, so only the Chinese reading is supplied here.
  const FINDINGS = {
    REQUESTED_CONCEPT_NO_DATA: ['所需变量没有数据', '问题要求汇总的变量在当前数据中没有任何记录值，这部分结果无法给出。请换用有记录的变量或数据源后重新规划。'],
    POPULATION_EMPTIES_EXPOSURE_LEVEL: ['入组条件清空了一个研究分组', '入组条件会排除某个研究分组的所有住院，当前计划不能批准，请重新生成保留各组的计划。'],
    ACCEPTED_BASELINE_CONTENT_MISSING: ['已接受的基线表内容缺失', '此前已接受的基线表有变量或数据坐标没有保留；需先恢复这些来源绑定的数据再重新规划。'],
    ACCEPTED_BASELINE_MATERIALIZATION_MISSING: ['已接受的基线表数据缺失', '此前已接受的基线表有变量或数据坐标没有保留；需先恢复这些来源绑定的数据再重新规划。'],
    ADJUSTMENT_RATIONALE_OR_TIMING_UNBOUND: ['调整变量缺少理由或时间定位', '有协变量缺少混杂理由，或没有证明它在时间零点之前。'],
    APPLICABLE_METHOD_LAYERS_NOT_BOUND: ['方法依据没有覆盖适用环节', '引用的文献没有覆盖时间对齐、相关性、缺失数据、函数形式、解读或报告中适用的环节。'],
    ARTICLE_CONTENT_ROLES_INCOMPLETE: ['文章内容模块不完整', '计划缺少队列、基线、数据质量、描述、主分析或稳健性中应有的证据模块。'],
    CLINICAL_DEFINITION_DATABASE_CONFORMANCE_NOT_ESTABLISHED: ['临床定义与数据库的一致性未确认', '这个临床定义在当前数据库上的实现是否与原定义一致，尚未确认。'],
    CLINICAL_DEFINITION_INDEPENDENT_REVIEW_PENDING: ['临床定义待独立审阅', '临床定义还需独立审阅，之后才能用于正式结论。'],
    COHORT_ELIGIBILITY_AFTER_TIME_ZERO: ['入组晚于分析时间零点', '源导出在分析时间零点之后才确定谁入组：概念人群的窗口或最短 ICU 住院时长晚于时间零点。在时间零点及之前确定的入组（例如窗口内的概念阳性）不受影响。'],
    COHORT_PREDICATE_COLUMN_WINDOW_UNRECORDED: ['入组条件所用列的物化窗口未记录', '方案的入组谓词筛选的列，宿主没有记录它是在哪个窗口内汇总的（列自身的分析窗口或研究的物化窗口），或没有记录事件时间的起点与单位，无法确认它在分析时间零点前确定。需要宿主补上记录，修改方案无法解决。'],
    COHORT_PREDICATE_DECIDED_AFTER_TIME_ZERO: ['入组选择晚于分析时间零点', '方案的入组谓词按时间零点之后才确定的信息选人：所筛选的列汇总到时间零点之后、ICU 住院时长或事件时间的阈值超过时间零点、要等事件发生才知道的事件时间、用到住院结束时才记录的结局，或用到没有确定时刻的住院级评分。这会改变研究人群或时间零点，需要在研究层面决定。'],
    COHORT_PREDICATE_OUTSIDE_COLUMN_DOMAIN: ['入组谓词与所筛选列的取值对不上', '方案的入组谓词在它所筛选列的全部取值上不起作用：纳入条件没有一个取值满足、或排除条件每个取值都满足时，有取值的行全被去掉；每个取值都满足纳入条件、或没有一个取值满足排除条件时，这条条件不按取值筛选任何行。常见原因是把概念描述里的数字当成了列的取值，例如对 0/1 标志写「≥ 2」。方案需按列的取值改写比较（只问有没有记录时写「缺失 / 非缺失」），不要删掉这条谓词，否则研究人群会被悄悄改变。'],
    COHORT_PREDICATE_WINDOW_AFTER_TIME_ZERO: ['入组谓词的窗口晚于分析时间零点', '方案的入组谓词写的窗口不从入 ICU 算起，或终点晚于时间零点；它筛选的列在时间零点前已经确定，只需把谓词窗口改为从入 ICU 到时间零点，选人结果不变。'],
    COHORT_SELECTS_ON_OUTCOME: ['入组条件按结局本身选人', '方案的纳排条件按研究结局本身的取值选人（不论阈值是多少），或在分析时间零点之后按参与定义结局的变量选人。这样选出的住院，结局在这条条件处被截断，估计只描述通过这条条件的住院。需要在研究层面决定：结局改为从时间零点起算（住院时长就改报时间零点之后的剩余时长）、换一个不受这条条件限定的结局，或去掉这条条件；如果本意就是这个条件人群，就把结局写成以这条条件为前提。'],
    COMPLETE_CASE_SENSITIVITY_REPEATS_PRIMARY: ['完整病例重拟合重复了主分析', '这些完整病例重拟合保留的行与主模型完全相同，只是重述主估计；会写明，但不算作稳健性证据。'],
    CONCEPT_POPULATION_RECORD_UNREADABLE: ['概念人群记录无法读取', '源导出的概念人群记录读不出，审阅无法确认导出何时确定了谁入组；需要重新准备导出。'],
    CONTINUOUS_COVARIATE_FUNCTIONAL_FORM_UNCHECKED: ['连续协变量的函数形式未检查', '连续协变量以线性形式进入模型，但没有可执行的非线性检查（如样条）。'],
    DESCRIPTIVE_INTERVAL_DEPENDENCE_UNRESOLVED: ['描述性区间的独立性未解决', '描述性区间缺少独立单位或患者分组依据；应只报告计数与比例。'],
    DESCRIPTIVE_INTERVALS_WITHHELD_WITHOUT_PATIENT_GROUPING: ['患者分组依据未核实，暂不报告区间', '同一位患者可能有多次 ICU 住院，但患者分组依据未经核实或未通过校验。保留全部住院，只报告计数与比例，不报告不确定性区间；完整方案审阅和报告局限中需说明这一原因。'],
    DESCRIPTIVE_INTERVAL_LIMITATION_MISSING: ['未写明描述性区间受限原因', '当前数据条件下不能报告不确定性区间，但分布步骤尚未写明原因。需由系统补全区间受限原因，并在完整方案审阅和报告局限中披露。'],
    DESCRIPTIVE_VARIANCE_DESIGN_CONFLICT: ['区间方法与研究设计不一致', '研究设计要求报告不确定性，但分布步骤只提供计数与比例。方案需要按研究设计补全区间方法，并验证所需的数据条件后再执行。'],
    DESCRIPTIVE_PRIMARY_SUMMARY_MISSING: ['缺少描述性主要汇总', '这个问题需要一个主要步骤汇总指定测量的取值（总体或按已声明的分组）；问题要求的分组汇总仍由次要步骤报告。'],
    DESCRIPTIVE_POPULATION_SCOPE_UNRESOLVED: ['描述性步骤的人群未确定', '描述性风险步骤没有可执行的人群选择，分母无法确定。'],
    DESIGN_ANALOGUE_NOT_BOUND_TO_PRIMARY_PLAN: ['可参照研究没有约束主分析', '已筛到可参照的研究，但它没有约束任何主分析步骤；需写明借鉴或有意不同的设计要素。'],
    DIRECT_COMPARATOR_NOT_BOUND_TO_PRIMARY_PLAN: ['可比研究没有约束主分析', '已筛到直接可比的研究，但它没有约束任何主分析步骤；需写明借鉴或有意不同的设计要素。'],
    DIRECT_COMPARATOR_NOT_ESTABLISHED: ['尚无直接可比研究', '还没有找到与本研究直接可比的已发表研究。'],
    DISTRIBUTION_MISSINGNESS_AUTHORITY_INVALID: ['分布或缺失分析的设定无效', '某个分布或缺失分析步骤的设定不能被执行组件接受。'],
    LANDMARK_STAY_LENGTH_OUTCOME: ['结局是时间零点时仍在 ICU 者的总住院时长', '入组条件恰好保留在分析时间零点时仍在 ICU 的住院（landmark 队列），而结局是从入 ICU 算起的 ICU 住院时长。按这个条件人群陈述结局即可，或改报时间零点之后剩余的 ICU 住院时长。'],
    LITERATURE_DESIGN_ROUTE_NOT_EXPLICIT: ['文献对设计的作用未写明', '引用了文献，但没有写明每篇文献支撑了哪个设计要素。'],
    LITERATURE_RETRIEVAL_NOT_CONDUCTED: ['未进行文献检索', '只有人工种子文献，没有完成可核查的实时检索。'],
    LITERATURE_SEARCH_PROVENANCE_INCOMPLETE: ['文献检索记录不完整', '检索的来源、日期或筛选记录不完整。'],
    LITERATURE_SEARCH_QUERY_NOT_RECORDED: ['检索式未记录', '检索回执没有保存确切的检索式。'],
    METHOD_SOURCE_DESIGN_ELEMENT_UNSUPPORTED: ['方法文献与设计要素不匹配', '有方法文献被绑定到它的方法卡并不支持的设计要素上。'],
    MISSINGNESS_UNEXAMINED_COMPLETE_CASE: ['完整病例分析的假设未检验', '所有模型都只用完整病例，也没有缺失数据敏感性分析。'],
    MISSING_CATEGORY_WITHOUT_COMPLETE_CASE_SENSITIVITY: ['未测量类别缺少完整病例敏感性分析', '主模型把协变量未测量的行保留为单独的状态，却没有预先设定的完整病例重拟合覆盖这些协变量。'],
    MODEL_TERM_CODING_CONFLICTS_WITH_DECLARED_DOMAIN: ['变量编码与取值类型冲突', '有分类或有序变量被当作连续变量编码。'],
    NOVELTY_POSITIONING_REVIEW_REQUIRED: ['创新性定位待独立审阅', '已筛到可比研究，但检索与自动筛选不能证明创新性；人群、暴露、时间零点、估计目标、分析路线和临床贡献仍需独立评估。'],
    OUTCOME_DEFINITION_UNRESOLVED: ['主要结局的定义不完整', '主要结局的含义、观察范围或汇总方式还没有确定，暂时无法分析。'],
    PHENOTYPING_COMPARISON_CONTRACT_INVALID: ['表型比较的设定无效', '聚类后的描述性比较没有绑定到确切的主聚类队列与分配结果。'],
    PHENOTYPING_FIT_ROSTER_INVALID: ['聚类拟合变量清单无效', '需要把用于聚类拟合的变量和仅供阅读的变量分开声明。'],
    PHENOTYPING_OUTCOME_COMPARISON_INCOMPLETE: ['聚类后的结局比较不完整', '没有可执行的步骤覆盖所请求结局的分群比较。'],
    PLANNER_ADJUSTMENT_PROPOSAL_INCOMPLETE: ['调整变量方案不完整', 'Agent 提出的调整变量缺少混杂理由或基线时间证明；由 Agent 修订，不需要你补填。'],
    PLAN_INPUT_STRUCTURALLY_UNAVAILABLE: ['所需变量在数据源中不可得', '计划需要的科学变量在这个数据源中结构性缺失；不会改用别的结局或暴露替代。'],
    PLAN_POPULATION_REQUIREMENT_DRIFT: ['研究人群发生了变化', '描述性结果改变了研究人群；需要在新计划中审阅这一变更。'],
    POPULATION_CRITERION_NOT_APPLIED: ['写明的人群条件没有施加', '方案写明了问题限定的这条人群条件，但没有谓词施加它：分析保留了其余入组条件选出的全部输入行，人群比写明的更宽。方案修订无法补上；需要为研究提供能施加这条条件的数据（能表达它、并按它写明的窗口记录的概念），或由研究者接受更宽的人群。结果与稿件按实际分析的人群表述。'],
    POPULATION_SCOPE_AMENDMENT_DECLARED: ['研究人群已声明变更', '描述性结果改变了研究人群；需要在新计划中审阅这一变更。'],
    POST_BASELINE_EXPOSURE_TIMING_NOT_CLOSED: ['暴露在时间零点之后才判定', '暴露在入 ICU 之后才判定，但计划没有可执行的 landmark 或时变设计来处理暴露机会和早期事件。'],
    PREDICTION_PATIENT_GROUPING_NOT_CARRIED_BY_TRAJECTORY: ['轨迹或 landmark 设计不带患者分组', '这份计划需要按患者划分或聚类住院，同一位患者可能有多次 ICU 住院。数据来源能按患者分组，但研究声明的轨迹或 landmark 设计读取的数据不带患者分组。可以去掉轨迹或 landmark 设计，或只保留每位患者的首次 ICU 住院后重新规划。'],
    PREDICTION_PATIENT_GROUPING_UNAVAILABLE: ['数据来源无法按患者分组', '这份计划需要按患者划分或聚类住院。同一位患者可能有多次 ICU 住院，但这个数据来源无法提供患者分组。可以只保留每位患者的首次 ICU 住院后重新规划，或为数据来源登记患者身份。'],
    PREDICTION_PREDICTOR_TIMING_UNPROVEN: ['预测变量的观测时间无法确认', '这份计划无法证明预测变量都在预测时点之前观测到。'],
    PREDICTION_RISK_SET_NOT_KEPT: ['队列没有限定预测时点仍在 ICU 的住院', '队列没有限定为预测时点仍在 ICU 的住院。预测时点之前已死亡或已转出 ICU 的住院可能也进了分析，它们的结局可能在预测之前就已发生。人群条件里需要加上"预测时点仍在 ICU"。'],
    PRIMARY_DESCRIPTIVE_SUMMARY_SCOPE_UNRESOLVED: ['主要描述性汇总无法执行', '主要描述性汇总只描述一个数值测量的取值，可按一个已声明的分类变量分组；当前计划不满足这一点，执行时会失败。几项测量有无记录、缺失多少，应由测量审计步骤报告；常规化验和生命体征的测量时间目前无法报告。'],
    PRIMARY_EXPOSURE_TIME_ANCHOR_MISMATCH: ['暴露定义的时间锚点不一致', '主暴露临床定义的时间锚点与研究声明的时间零点不一致。'],
    PRIMARY_EXPOSURE_TIME_ANCHOR_UNVERIFIED: ['暴露定义的时间锚点未核实', '主暴露临床定义的时间锚点是否与研究的时间零点一致，尚未核实。'],
    PRIMARY_MODEL_NOT_EVALUABLE_ON_SEALED_COHORT: ['主模型无法在封存队列上求值', '主模型无法在执行将读取的队列上求值，批准后的运行会在那里失败；需先把模型绑定到队列实际具有的列和水平。'],
    PRIMARY_MODEL_RETENTION_INSUFFICIENT: ['主模型保留的行过少', '主模型实际拟合的行只占可评估行的一小部分，拟合结果不足以回答所审阅的问题。'],
    PRIMARY_MODEL_RETENTION_PROBE_FAILED: ['主模型行数统计失败', '统计主模型将拟合的行数时意外失败，保留比例未知。'],
    PRIMARY_MODEL_RETENTION_REDUCED: ['主模型保留的行减少', '主模型只会拟合一部分可评估的行，保留比例明显下降。'],
    PRIMARY_MODEL_RETENTION_UNVERIFIED: ['主模型行数在批准前未核实', '批准前无法统计主模型将拟合的行数；执行后仍会审计实际拟合的分母。'],
    PRIMARY_POPULATION_EXECUTION_OWNER_MISSING: ['主模型人群没有执行组件', '描述性步骤要求使用主模型人群，但没有执行组件绑定这个人群。'],
    RECENT_DESIGN_ANALOGUE_NOT_ESTABLISHED: ['尚无近期可参照设计', '没有找到近期发表、方法可参照的研究。'],
    RECENT_DIRECT_COMPARATOR_NOT_ESTABLISHED: ['尚无近期直接可比研究', '没有找到近期发表的直接可比研究。'],
    REPEATED_STAY_DEDUP_UNDECLARED: ['重复入住的处理规则未声明', '可能存在同一患者多次入住，但没有步骤写明是只取首次入住，还是按患者聚类或混合模型处理。'],
    REPEATED_STAY_IDENTITY_UNAVAILABLE: ['无法识别同一患者的多次入住', '数据源没有给出患者身份，无法排除同一患者多次入住；可以继续做开发性分析，但不能按患者独立处理，也不能作论文结论。'],
    REPEATED_STAY_METHOD_NOT_DECLARED: ['重复入住没有对应的统计方法', '有患者身份，但没有可执行的方法处理同一患者的多次入住；由 Agent 选定方法。'],
    REQUESTED_DOSE_RESPONSE_NOT_ESTIMABLE: ['所问的剂量反应无法估计', '问题要求剂量反应关系，但主要暴露只有两个水平；梯度至少需要三个有序水平或连续暴露，两水平对比不能代替。'],
    REQUESTED_OCCURRENCE_COVERAGE_INCOMPLETE:['暴露发生比例没有分析步骤', '问题问到暴露的发生比例，但没有步骤在研究队列上报告它。'],
    REQUESTED_OUTCOME_COVERAGE_INCOMPLETE: ['部分结局没有分析步骤', '问题里提到的结局并非都有可执行的分析。'],
    REQUIRED_SENSITIVITY_IS_PROTOCOL_ONLY: ['要求的敏感性分析只写在文字里', '你要求的敏感性分析缺失，或只有描述、没有可执行步骤。'],
    REVIEWABLE_PLAN_SPECIFICATION_MISSING: ['缺少完整的推荐设定', '所选设计没有给出完整、可审阅的推荐设定。'],
    ROBUSTNESS_AXES_TOO_NARROW: ['稳健性分析维度不足', '可执行的稳健性分析维度少于这类研究的要求。'],
    ROBUSTNESS_DIAGNOSTIC_DISPLAY_MISMATCH: ['稳健性图混入了诊断结果', '稳健性图绑定了不是效应估计的函数形式诊断。'],
    ROBUSTNESS_OVERRIDE_EVENT_WINDOW_UNREADABLE: ['稳健性分析的事件窗口读不出', '有锁定的稳健性分析要按有限时间窗读取一个事件，但输入只记录了整段住院的状态，没有该事件的时间列；宿主不会构建这些队列，这些分析没有估计，运行会在稳健性面板处不通过。由 Agent 改写或删除这些敏感性分析，不需要你补填；主估计目标不变。'],
    ROBUSTNESS_SPECS_NOT_EXECUTABLE: ['部分稳健性分析没有执行组件', '计划声明的部分稳健性分析没有任何组件会执行。'],
    SCIENTIFIC_STEP_METHOD_SOURCE_NOT_BOUND: ['科学步骤缺少方法依据', '有科学步骤没有引用约束其方法的文献。'],
    STUDY_TIME_ZERO_MISMATCH: ['研究时间零点与观察窗起点不一致', '研究声明的时间零点不是已物化观察窗计时的起点；研究没有主暴露，无法由暴露定义承载这个时间零点。'],
    SURVIVAL_LANDMARK_OWNER_NOT_SEALED: ['生存分析组件尚未封存', '计划选用了 landmark 生存分析组件，但研究还没有声明生存设计。暴露、终点、随访、landmark 和协变量齐全时，由系统编译执行设置后重新规划；否则需修订计划。'],
    TABLE_ONE_INDEPENDENT_TESTS_IGNORE_REPEATED_UNITS: ['基线表检验忽略了重复单位', '队列保留了重复单位，基线表却要求按独立样本做检验。'],
    TIME_VARYING_RUNTIME_UNAVAILABLE: ['时变分析无法执行', '这个数据源或设计目前还不能执行时变分析。'],
    TOP_JOURNAL_LITERATURE_SEARCH_NOT_ESTABLISHED: ['顶刊可比研究检索未完成', '还没有完成针对顶刊可比研究的检索。'],
    TRAJECTORY_DESIGN_STATES_NO_POPULATION: ['轨迹设计未写明人群', '封存的轨迹设计在源队列的全部输入行上聚类：它没有写明人群谓词，所以问题限定的人群没有施加到这些类别上。'],
    TRAJECTORY_LONGITUDINAL_OWNER_NOT_SEALED: ['轨迹尚未按时间窗建模', '计划的轨迹聚类只用了每次住院的一个值；这些变量可以交给固定时间窗轨迹组件建模，由系统编译执行设置后重新规划。'],
    TRAJECTORY_REPRESENTATION_NOT_LONGITUDINAL: ['轨迹类别只基于单个时点', '计划的轨迹聚类只用了每次住院的一个值，类别概括的是住院而不是轨迹；这些变量也不能交给固定时间窗轨迹组件建模。可以继续做分析级结果，需写明这一局限。'],
    STATED_ADJUSTMENT_MISSING: ['主模型遗漏了问题要求的调整变量', '研究问题明确要求调整的变量没有进入每个主模型。由系统修订计划，保留该变量并说明理由和基线时间；若无法实现，必须明确记录具体缺口。'],
    STATED_ADJUSTMENT_UNRESOLVED: ['问题要求的调整变量未能对应到数据', '研究问题要求调整的变量没有对应到数据来源里的具体变量，系统无法核对主模型是否调整了它；批准前请核对主模型的调整变量。'],
    STATED_ADJUSTMENT_UNAVAILABLE: ['来源缺少问题要求的调整变量', '研究问题要求的调整变量在当前数据来源中不可用，缺口已经保留；解决来源缺口之前不能批准执行计划。'],
    STATED_SUBGROUP_MISSING: ['计划遗漏了问题要求的分层分析', '研究问题要求按这个变量分层分析（描述性研究则按它分组描述），计划里没有这样的分析步骤；只按它分组的基线表不算。由系统修订计划，已接受的基线表保持不变。'],
    STATED_SUBGROUP_UNAVAILABLE: ['来源缺少问题要求的分层变量', '研究问题要求的分层变量在当前数据来源中不可用，缺口已保留；解决来源缺口之前不能批准执行计划。'],
    STATED_SUBGROUP_UNRESOLVED: ['问题要求的分层变量未能对应到数据', '研究问题要求分层的变量没有对应到数据来源里的具体变量，系统无法核对计划是否按它分层；批准前请核对计划的分层分析。'],
    STATED_SUBGROUP_UNSUPPORTED: ['当前研究类型无法执行所需分层分析', '这类研究目前没有能按该变量分层分析的可执行方法，基线表分组不能代替；缺口已记录，不能批准执行计划。'],
    UNADJUSTED_ASSOCIATION_NOT_ARTICLE_GRADE: ['主关联没有调整混杂', '主要关联未做调整，只能作描述性解读。'],
  };

  // The question a review decision asks the researcher, in Chinese.  English
  // readers keep the host's own authorization question.
  const QUESTIONS = {
    ADJUSTMENT_RATIONALE_OR_TIMING_UNBOUND: '你是否批准新研究版本中每个调整协变量的临床理由与基线时间定位？',
    POPULATION_SCOPE_AMENDMENT_DECLARED: '新计划改变了描述性结果的研究人群，是否接受这项声明的科学变更？',
    PRIMARY_EXPOSURE_TIME_ANCHOR_MISMATCH: '研究应采用概念所有者发布的临床定义锚点，还是为所声明的临床锚点发布新的暴露定义（新的概念或研究版本）？',
    PRIMARY_EXPOSURE_TIME_ANCHOR_UNVERIFIED: '研究应采用概念所有者发布的临床定义锚点，还是为所声明的临床锚点发布新的暴露定义（新的概念或研究版本）？',
    REPEATED_STAY_METHOD_NOT_DECLARED: '新的研究版本应每位患者只取一次 ICU 入住，还是保留多次入住并采用聚类或混合效应估计？',
    REQUESTED_DOSE_RESPONSE_NOT_ESTIMABLE: '问题要求剂量反应梯度，但暴露只有两个水平。新的研究版本应改用分级或连续暴露，还是改问两个水平之间的对比？',
    REQUIRED_SENSITIVITY_IS_PROTOCOL_ONLY: '是否需要一个实际执行这些敏感性分析的新研究版本，还是缩减所要求的输出？',
    ROBUSTNESS_AXES_TOO_NARROW: '新的研究版本应增加预先设定、有来源支持的敏感性分析维度，还是让当前较窄的分析保持仅分析级？',
    STUDY_TIME_ZERO_MISMATCH: '研究应以观察窗的起点作为时间零点，还是修订研究，使观察窗从所声明的时间零点开始计时？',
    UNADJUSTED_ASSOCIATION_NOT_ARTICLE_GRADE: '保留描述性分析，还是授权一个采用有临床时序调整策略的新研究版本？',
  };

  const COLUMNS = {
    // Run and artifact fields.
    run_id: ['Run ID', '运行 ID'],
    run_type: ['Run type', '运行类型'],
    study_id: ['Study ID', '研究 ID'],
    status: ['Status', '状态'],
    mode: ['Mode', '模式'],
    question: ['Question', '问题'],
    local_first: ['Local-first', '本地优先'],
    cohort_size: ['Cohort size', '队列规模'],
    evidence_count: ['Evidence items', '证据项'],
    missing_evidence: ['Missing evidence', '缺失证据'],
    signed: ['Signed', '已签署'],
    provider: ['Provider', 'Provider'],
    database_scope: ['Database scope', '数据库范围'],
    stage: ['Stage', '阶段'],
    population_rule: ['Population rule', '人群规则'],
    excluded_from_previous: ['Excluded', '上一步排除'],
    exposure_value: ['Exposure', '暴露值'],
    reference_exposure_value: ['Reference', '参考值'],
    adjusted_absolute_risk: ['Adjusted risk', '校正后风险'],
    adjusted_odds_ratio: ['Adjusted OR', '校正后 OR'],
    ci_low: ['95% CI low', '95% CI 下限'],
    ci_high: ['95% CI high', '95% CI 上限'],
    standardization_n: ['Standardized N', '标准化样本量'],
    standardization_method: ['Standardization', '标准化方法'],
    estimate_type: ['Estimate', '估计类型'],
    point_estimate: ['Estimate', '估计值'],
    effect_scale: ['Scale', '效应尺度'],
    converged: ['Converged', '已收敛'],
    model_id: ['Model', '模型'],
    spec_id: ['Specification', '规格'],
    // Result-table columns.
    row_role: ['Row type', '行类型'],
    exposure_level: ['Exposure level', '暴露水平'],
    exposure_level_index: ['Level index', '水平序号'],
    n_rows: ['Records', '记录数'],
    exposure_denominator: ['Exposure denominator', '暴露分母'],
    exposure_pct: ['Share (%)', '占比 (%)'],
    exposure_ci_low_pct: ['Share 95% CI low (%)', '占比 95% CI 下限 (%)'],
    exposure_ci_high_pct: ['Share 95% CI high (%)', '占比 95% CI 上限 (%)'],
    exposure_standard_error_pct: ['Share SE (%)', '占比标准误 (%)'],
    exposure_interval_covariance: ['Interval variance', '区间方差类型'],
    exposure_interval_cluster_count: ['Clusters', '聚类数'],
    outcome_observed_n: ['Records with outcome', '有结局记录数'],
    outcome_events: ['Outcome events', '结局事件数'],
    outcome_denominator: ['Outcome denominator', '结局分母'],
    outcome_rate_pct: ['Outcome rate (%)', '结局比例 (%)'],
    n_denominator: ['Exposure denominator', '暴露分母'],
    group_type: ['Group type', '分组类型'],
    group_value: ['Group value', '分组取值'],
    outcome_n: ['Outcome denominator', '结局分母'],
    outcome_missing_n: ['Missing outcomes', '结局缺失数'],
    outcome_event_n: ['Outcome events', '结局事件数'],
    outcome_risk_pct: ['Outcome risk (%)', '结局风险 (%)'],
    outcome_risk_ci_low_pct: ['Risk 95% CI low (%)', '风险 95% CI 下限 (%)'],
    outcome_risk_ci_high_pct: ['Risk 95% CI high (%)', '风险 95% CI 上限 (%)'],
    outcome_ci_low_pct: ['Rate 95% CI low (%)', '比例 95% CI 下限 (%)'],
    outcome_ci_high_pct: ['Rate 95% CI high (%)', '比例 95% CI 上限 (%)'],
    step_order: ['Step', '步骤'],
    predicate_kind: ['Rule type', '条件类型'],
    concept_id: ['Concept', '概念'],
    resolved_column: ['Column', '列'],
    aggregation: ['Aggregation', '汇总方式'],
    op: ['Operator', '运算'],
    value: ['Value', '取值'],
    n_before: ['Before step', '本步之前'],
    n_excluded: ['Excluded', '排除'],
    n_remaining: ['Remaining', '剩余'],
    event_time_column: ['Event-time column', '事件时间列'],
    event_time_start_hours: ['Event-time start (h)', '事件时间起点（小时）'],
    variable: ['Variable', '变量'],
    fraction_missing: ['Missing fraction', '缺失比例'],
    n_missing: ['Missing', '缺失数'],
    n_unique_non_missing: ['Distinct values', '不同取值数'],
    variable_type: ['Variable type', '变量类型'],
    variable_order: ['Order', '顺序'],
    category: ['Category', '类别'],
    group: ['Group', '分组'],
    group_order: ['Group order', '分组顺序'],
    group_level: ['Group level', '分组水平'],
    denominator_n: ['Denominator', '分母'],
    nonmissing_n: ['Non-missing', '非缺失数'],
    missing_n: ['Missing', '缺失数'],
    missing_pct: ['Missing (%)', '缺失比例 (%)'],
    cohort_n: ['Cohort', '队列人数'],
    group_n: ['Group', '分组人数'],
    group_pct_of_cohort: ['Share of cohort (%)', '占队列 (%)'],
    group_missing_excluded_n: ['Missing group excluded', '分组缺失排除'],
    n_nonmissing: ['Non-missing', '非缺失数'],
    median: ['Median', '中位数'],
    q25: ['25th percentile', '下四分位数'],
    q75: ['75th percentile', '上四分位数'],
    mean: ['Mean', '均值'],
    sd: ['SD', '标准差'],
    analysis_set: ['Analysis set', '分析集'],
    required_variables: ['Required variables', '所需变量'],
    n_total: ['Total', '总数'],
    n_complete: ['Complete', '完整数'],
    n_excluded_missing: ['Excluded for missing values', '因缺失排除'],
    complete_pct: ['Complete (%)', '完整比例 (%)'],
    estimate: ['Estimate', '估计值'],
    fit_status: ['Fit status', '拟合状态'],
    exposure: ['Exposure', '暴露'],
    outcome: ['Outcome', '结局'],
    covariates: ['Covariates', '协变量'],
    estimator_kind: ['Estimator', '估计方法'],
    analysis_role: ['Analysis role', '分析角色'],
    contrast: ['Comparison', '对比'],
    is_primary_contrast: ['Primary comparison', '主要对比'],
    reference_level: ['Reference level', '参照水平'],
    standard_error: ['Standard error', '标准误'],
    variance_estimator: ['Variance estimator', '方差估计'],
    cluster_count: ['Clusters', '聚类数'],
    notes: ['Notes', '备注'],
    n: ['N', '人数'],
    n_events: ['Events', '事件数'],
    p_value: ['P value', 'P 值'],
    p_value_holm: ['P value (Holm)', 'P 值（Holm 校正）'],
    level: ['Level', '水平'],
    in_primary_model: ['In primary model', '进入主模型'],
    test_id: ['Test', '检验'],
    statistic: ['Statistic', '统计量'],
    requirement_id: ['Requirement', '分析需求'],
    eligible_n: ['Applicable', '适用数'],
    not_applicable_n: ['Not applicable', '不适用数'],
    event_present_n: ['Event present', '有事件'],
    event_absent_n: ['Event absent', '无事件'],
    before_origin_n: ['Before time zero', '早于时间零点'],
    value_missing_n: ['Value missing', '值缺失数'],
    value_missing_pct: ['Value missing (%)', '值缺失比例 (%)'],
    indicator_semantics: ['Indicator meaning', '指示含义'],
    missingness_kind: ['Missingness type', '缺失类型'],
    measured_n: ['Measured', '有测量数'],
    measured_pct: ['Measured (%)', '测量比例 (%)'],
    measured_one_n: ['Measured', '有测量数'],
    raw_indicator_one_n: ['Raw indicator = 1', '原始指示为 1'],
    n_stratum: ['Stratum', '分层人数'],
    exposure_variable: ['Exposure variable', '暴露变量'],
    exposure_category: ['Exposure category', '暴露类别'],
    value_column: ['Value column', '取值列'],
    measurement_count_column: ['Measurement-count column', '测量次数列'],
    measurement_total_n: ['Measurements', '测量总次数'],
    measurement_count_median_when_measured: ['Median measurements', '测量次数中位数'],
    measurement_count_max: ['Most measurements', '最多测量次数'],
    repeat_measured_n: ['Measured more than once', '多次测量数'],
    value_present_but_measured_zero_n: ['Value without measurement flag', '有值但测量标记为 0'],
    raw_value_missing_n: ['Raw value missing', '原始值缺失数'],
    // A prediction model's performance on its validation set
    // (prediction_performance.csv; rows are the cohort's units, here ICU stays).
    model: ['Model', '模型'],
    authority_scope: ['Evidence scope', '证据范围'],
    paper_authorization_allowed: ['Paper authorization', '可用于论文'],
    split_seed: ['Split seed', '切分种子'],
    validation_fraction: ['Validation fraction', '验证集比例'],
    predictor_n: ['Predictors', '预测变量数'],
    predictors: ['Predictor roster', '预测变量'],
    development_n: ['Development records', '开发集记录数'],
    validation_n: ['Validation records', '验证集记录数'],
    development_subject_n: ['Development patients', '开发集患者数'],
    validation_subject_n: ['Validation patients', '验证集患者数'],
    patient_overlap_n: ['Patients in both sets', '两集共有患者数'],
    validation_event_n: ['Validation events', '验证集事件数'],
    validation_event_rate: ['Validation event rate', '验证集事件率'],
    auroc: ['AUROC', 'AUROC'],
    auroc_se: ['AUROC SE', 'AUROC 标准误'],
    auroc_ci_low: ['AUROC 95% CI low', 'AUROC 95% CI 下限'],
    auroc_ci_high: ['AUROC 95% CI high', 'AUROC 95% CI 上限'],
    auroc_ci_method: ['AUROC interval method', 'AUROC 区间方法'],
    auroc_bootstrap_n: ['AUROC bootstrap resamples', 'AUROC bootstrap 重抽样次数'],
    auroc_bootstrap_skipped_n: ['AUROC resamples skipped', 'AUROC 跳过的重抽样次数'],
    average_precision: ['Average precision', '平均精确率'],
    brier_score: ['Brier score', 'Brier 分数'],
    calibration_status: ['Calibration', '校准'],
    calibration_intercept: ['Calibration intercept', '校准截距'],
    calibration_slope: ['Calibration slope', '校准斜率'],
    preprocessing_fit_scope: ['Preprocessing fitted on', '预处理拟合范围'],
    patient_group_source: ['Patient grouping column', '患者分组列'],
    patient_group_derivation: ['Patient grouping rule', '患者分组规则'],
    repeated_split_n: ['Repeated splits', '重复切分次数'],
    repeated_split_seeds: ['Repeated-split seeds', '重复切分种子'],
    repeated_split_auroc_mean: ['AUROC across splits (mean)', '各次切分 AUROC 均值'],
    repeated_split_auroc_sd: ['AUROC across splits (SD)', '各次切分 AUROC 标准差'],
    repeated_split_average_precision_mean: ['Average precision across splits (mean)', '各次切分平均精确率均值'],
    repeated_split_average_precision_sd: ['Average precision across splits (SD)', '各次切分平均精确率标准差'],
    repeated_split_brier_mean: ['Brier score across splits (mean)', '各次切分 Brier 分数均值'],
    repeated_split_brier_sd: ['Brier score across splits (SD)', '各次切分 Brier 分数标准差'],
    repeated_split_results: ['Per-split results', '各次切分结果'],
    // The model against existing scores or models on the same validation
    // rows (benchmark_comparison.csv; interval columns hold values only on
    // the AUROC row).
    comparator_column: ['Comparator column', '比较对象列'],
    comparator_concept: ['Comparator', '比较对象'],
    comparator_kind: ['Comparator kind', '比较对象类型'],
    metric: ['Metric', '指标'],
    model_value: ['Model value', '模型值'],
    model_ci_low: ['Model 95% CI low', '模型 95% CI 下限'],
    model_ci_high: ['Model 95% CI high', '模型 95% CI 上限'],
    comparator_value: ['Comparator value', '比较对象值'],
    comparator_ci_low: ['Comparator 95% CI low', '比较对象 95% CI 下限'],
    comparator_ci_high: ['Comparator 95% CI high', '比较对象 95% CI 上限'],
    difference: ['Difference (model − comparator)', '差值（模型 − 比较对象）'],
    difference_se: ['Difference SE', '差值标准误'],
    difference_ci_low: ['Difference 95% CI low', '差值 95% CI 下限'],
    difference_ci_high: ['Difference 95% CI high', '差值 95% CI 上限'],
    z: ['z', 'z 值'],
    interval_method: ['Interval method', '区间方法'],
    bootstrap_n: ['Bootstrap resamples', 'Bootstrap 重抽样次数'],
    bootstrap_skipped_n: ['Resamples skipped', '跳过的重抽样次数'],
    comparator_missing_n: ['Comparator missing', '比较对象缺失数'],
    comparison_n: ['Compared records', '参与比较记录数'],
    comparison_event_n: ['Compared events', '参与比较事件数'],
    comparison_subject_n: ['Compared patients', '参与比较患者数'],
    calibration_reason: ['Calibration not compared because', '未比较校准的原因'],
    comparator_predicts: ['Comparator predicts', '比较对象预测的结局'],
    outcome_concept: ['Study outcome', '研究结局'],
    comparator_information_window: ['Comparator information window', '比较对象信息窗口'],
    prediction_time_hours: ['Prediction time (h)', '预测时点（小时）'],
    information_window_relation: ['Information window relation', '信息窗口关系'],
    information_window_differs: ['Information window differs', '信息窗口不同'],
  };

  // Reader names for the closed value sets of result columns, by column.  A
  // value outside its column's set is shown as written, so a new code reads
  // raw rather than under a neighbour's name.
  const VALUES = {
    auroc_ci_method: {
      delong_logit_normal_95pct: ['DeLong', 'DeLong'],
      patient_stratified_bootstrap_percentile_95pct: ['resampled by patient', '按患者重抽样'],
    },
    interval_method: {
      none_counts_only: ['counts only, no interval', '仅计数，无区间'],
      wilson: ['Wilson', 'Wilson'],
      delong_paired_normal_95pct: ['paired DeLong', '配对 DeLong'],
      patient_stratified_bootstrap_percentile_95pct: ['resampled by patient', '按患者重抽样'],
    },
    row_role: {
      exposure_level: ['exposure group', '暴露组'],
      overall: ['all records', '全部记录'],
    },
    comparator_kind: {
      probability: ['probability', '概率'],
      score: ['score', '评分'],
    },
    metric: {
      auroc: ['AUROC', 'AUROC'],
      brier_score: ['Brier score', 'Brier 分数'],
      calibration_intercept: ['Calibration intercept', '校准截距'],
      calibration_slope: ['Calibration slope', '校准斜率'],
    },
    // The comparison table's two states, then the performance table's own.
    calibration_status: {
      compared: ['compared', '已比较'],
      calibration_not_compared: ['not compared', '未比较'],
      estimated: ['estimated', '已估计'],
      not_estimable_constant_probability: ['not estimable: every predicted probability is the same', '无法估计：预测概率全部相同'],
      not_estimable_perfect_separation: ['not estimable: the outcome is perfectly separated', '无法估计：结局被完全分离'],
      not_estimable_nonconvergence: ['not estimable: the calibration model did not converge', '无法估计：校准模型未收敛'],
    },
    calibration_reason: {
      score_scale: ['the score is not a probability', '评分刻度不是概率'],
      predicts_another_outcome: ['it predicts another outcome', '预测的是另一结局'],
      predicts_unstated: ['what it predicts is not stated', '未说明预测什么'],
    },
    information_window_relation: {
      same: ['comparator window ends at the prediction time', '比较对象窗口在预测时点结束'],
      comparator_ends_after_prediction_time: ['comparator window ends after the prediction time', '比较对象窗口晚于预测时点结束'],
      comparator_ends_before_prediction_time: ['comparator window ends before the prediction time', '比较对象窗口早于预测时点结束'],
      comparator_window_unstated: ['comparator window not stated', '比较对象窗口未说明'],
    },
    // A CSV cell carries the boolean as Python writes it.
    information_window_differs: {
      True: ['yes', '是'],
      False: ['no', '否'],
    },
  };

  // Tested before the renderer's own broader patterns, so a specific audit
  // table is never folded into a generic "measurement" title.
  const TABLES = [
    [/table_one/, 'Baseline characteristics (Table 1)', '基线特征表（Table 1）'],
    [/analytic_denominators/, 'Analysis denominators', '分析分母'],
    [/event_timing_audit/, 'Event-timing audit', '事件时间审计'],
    [/exposure_component_completeness/, 'Exposure component completeness', '暴露分量完整性'],
    [/measurement_process_audit/, 'Measurement-process audit', '测量过程审计'],
    [/measurement_source_audit/, 'Measurement-source audit', '测量来源审计'],
    [/missingness_audit/, 'Missingness audit', '缺失审计'],
    [/distribution_prevalence/, 'Distribution and share by group', '分组分布与占比'],
    [/probe_variable_profile/, 'Variable profile', '变量概况'],
    [/adjusted_association_estimates/, 'Adjusted association estimates', '调整后关联估计'],
    [/adjusted_association_coefficients/, 'Model coefficients', '模型系数'],
    [/key_metrics/, 'Key metrics', '关键指标'],
    [/ordinal_trend|trend_tests?/, 'Ordered trend tests', '有序趋势检验'],
    [/cohort_analysis_flow|exact sequential attrition ledger/, 'Cohort flow (sequential exclusions)', '队列流程（逐步排除）'],
    [/benchmark_comparison/, 'Comparison with existing scores and models', '与已有评分或模型的比较'],
    [/prediction_performance|model_performance/, 'Prediction performance', '预测性能'],
    [/prediction_scores/, 'Predicted probabilities', '预测概率'],
    [/analysis_cohort/, 'Analysis cohort', '分析队列'],
    [/cohort_flow|cohort_accounting|cohort_definition/, 'Cohort flow', '队列流程'],
    [/absolute_risk_context/, 'Absolute risks', '绝对风险'],
    [/measurement_audit_completeness|measurement_missingness/, 'Measurement completeness', '测量完整性'],
    [/measurement_audit_denominators/, 'Analysis denominators', '分析分母'],
    [/measurement_audit_missingness/, 'Missingness audit', '缺失审计'],
    [/measurement_audit_process/, 'Measurement process', '测量过程'],
    [/measurement_audit_source/, 'Measurement sources', '测量来源'],
    [/robustness_matrix/, 'Sensitivity matrix', '敏感性分析矩阵'],
    [/robustness_summary|robustness_replay/, 'Sensitivity summary', '敏感性分析汇总'],
    [/feature_functional_form_sensitivity/, 'Feature functional-form sensitivity', '特征函数形式敏感性'],
    [/calibration/, 'Calibration', '校准'],
    [/clinical_utility/, 'Clinical utility', '临床效用'],
    [/internal_validation|(?:table:)?validation$/, 'Internal validation', '内部验证'],
    [/feature_availability/, 'Feature availability', '特征可用性'],
    [/landmark_risk_set_flow/, 'Landmark risk-set flow', 'Landmark 风险集流程'],
    [/landmark_km_curve/, 'Kaplan–Meier curves', 'Kaplan–Meier 曲线'],
    [/landmark_cox_summary/, 'Cox estimates', 'Cox 估计'],
    [/landmark_ph_diagnostics/, 'Proportional-hazards diagnostics', '比例风险诊断'],
    [/landmark_rmst_summary/, 'Restricted mean survival time', '限制性平均生存时间'],
    [/landmark_measurement_audit/, 'Landmark measurement audit', 'Landmark 测量审计'],
    [/landmark_exposure_onset_hours/, 'Exposure onset times', '暴露起始时间'],
    [/landmark_prevalence_sensitivity/, 'Exposure-definition sensitivity', '暴露定义敏感性'],
    [/landmark_survival_receipt/, 'Survival analysis receipt', '生存分析记录'],
    [/landmark_observability/, 'Landmark follow-up observability', 'Landmark 随访可观察性'],
    [/landmark_performance/, 'Performance by landmark', '各 Landmark 预测性能'],
    [/time_varying_cox_estimates/, 'Time-varying Cox estimates', '时变 Cox 估计'],
    [/time_varying_input_audit/, 'Time-varying input audit', '时变输入审计'],
    [/time_varying_runtime_receipt/, 'Time-varying analysis receipt', '时变分析记录'],
    [/trial_protocol/, 'Trial protocol', '试验方案'],
    [/causal_effect/, 'Causal contrast', '因果对比'],
    [/covariate_balance/, 'Covariate balance', '协变量平衡'],
    [/positivity_diagnostics/, 'Positivity diagnostics', '阳性诊断'],
    [/phenotype_profiles/, 'Phenotype profiles', '表型特征'],
    [/phenotype_assignments/, 'Phenotype assignments', '表型归属'],
    [/outcome_by_cluster/, 'Outcomes by cluster', '各类结局'],
    [/trajectory_profiles/, 'Trajectory profiles', '轨迹特征'],
    [/trajectory_representation_schema/, 'Trajectory representation schema', '轨迹表示结构'],
    [/trajectory_representation/, 'Trajectory representation', '轨迹表示'],
    [/trajectory_missingness_policy/, 'Trajectory missingness policy', '轨迹缺失处理规则'],
    [/trajectory_window_manifest/, 'Trajectory window manifest', '轨迹时间窗清单'],
    [/candidate_cluster_solution_schema/, 'Candidate cluster schema', '候选聚类结构'],
    [/candidate_cluster_assignments/, 'Candidate cluster assignments', '候选聚类归属'],
    [/cluster_assignment_provenance/, 'Cluster assignment provenance', '聚类归属来源'],
    [/cluster_stability_assignments/, 'Resampled cluster assignments', '重采样聚类归属'],
    [/cluster_stability_spec/, 'Cluster stability specification', '聚类稳定性设定'],
    [/cluster_stability/, 'Cluster stability', '聚类稳定性'],
    [/cluster_assignments/, 'Frozen cluster assignments', '固定聚类归属'],
    [/cluster_sizes/, 'Cluster sizes', '各类人数'],
    [/cluster_selection|trajectory_candidate_selection/, 'Cluster selection', '聚类选择'],
    [/stability_freeze/, 'Cluster freeze record', '聚类固定记录'],
    [/trajectory_selection_diagnostics/, 'Trajectory selection diagnostics', '轨迹选择诊断'],
    [/figure:visualization/, 'Study results figure', '研究结果图'],
    [/report:report/, 'Study report', '研究报告'],
  ];

  const METHOD_CARDS = {
    reporting_observational_study: ['Observational-study reporting', '观察性研究报告规范'],
    reporting_routinely_collected_data: ['Routinely collected data reporting', '常规收集数据的报告规范'],
    time_zero_and_immortal_time: ['Time zero and immortal time', '时间零点与不朽时间偏倚'],
    landmark_analysis: ['Landmark analysis', 'Landmark 分析'],
    proportional_hazards_diagnostics: ['Proportional-hazards diagnostics', '比例风险诊断'],
    restricted_mean_survival_time: ['Restricted mean survival time', '限制性平均生存时间'],
    repeated_units_per_patient: ['Repeated units per patient', '同一患者的重复单位'],
    continuous_covariate_functional_form: ['Functional form of continuous covariates', '连续协变量的函数形式'],
    missing_data_handling: ['Missing-data handling', '缺失数据处理'],
    absolute_and_relative_effects: ['Absolute and relative effects', '绝对与相对效应'],
  };

  const DESIGN_ELEMENTS = {
    adjustment: ['adjustment', '调整变量'],
    dependence: ['dependence', '相关性结构'],
    estimand: ['estimand', '估计目标'],
    exposure: ['exposure', '暴露'],
    missing_data: ['missing data', '缺失数据'],
    outcome: ['outcome', '结局'],
    reporting: ['reporting', '报告规范'],
    robustness: ['robustness', '稳健性'],
    time_zero: ['time zero', '时间零点'],
  };

  // The capability catalog's categories qualify an automatically read
  // concept, particularly a measured level versus a medication given.
  const CATEGORIES = {
    'blood gas': ['blood gas test', '血气检验'],
    cardiovascular: ['cardiovascular measure', '心血管指标'],
    chemistry: ['laboratory test', '检验指标'],
    demographics: ['demographic measure', '人口学指标'],
    hematology: ['hematology test', '血液检验'],
    medications: ['medication', '用药'],
    microbiology: ['microbiology test', '微生物检验'],
    neurological: ['neurological measure', '神经系统指标'],
    neurology: ['neurological measure', '神经系统指标'],
    outcome: ['outcome', '结局'],
    output: ['recorded output', '排出量'],
    renal: ['renal measure', '肾脏相关指标'],
    respiratory: ['respiratory measure', '呼吸指标'],
    severity: ['severity score', '严重程度评分'],
    ventilator: ['ventilation measure', '通气指标'],
    vitals: ['vital sign', '生命体征'],
  };
  // A finding whose host record names its cause reads by that cause: the
  // same code covers a source without patient grouping and a prepared
  // package that did not carry the source's verified grouping.
  const FINDING_REASONS = {
    DESCRIPTIVE_INTERVALS_WITHHELD_WITHOUT_PATIENT_GROUPING: {
      field: 'descriptive_interval_limitation',
      reasons: {
        verified_patient_grouping_not_carried_by_prepared_package: ['准备后的数据未保留患者分组', '来源已有经核实的患者分组，但准备后的数据包未携带与其住院行对应的分组凭证。恢复这一绑定前，保留全部住院，只报告计数与比例，不报告区间。'],
      },
    },
  };
  function category(key) {
    const row = Object.prototype.hasOwnProperty.call(CATEGORIES, key) ? CATEGORIES[key] : null;
    return row ? pick(row) : '';
  }

  function finding(code, record) {
    const byReason = FINDING_REASONS[String(code || '')];
    const cause = byReason && record && record[byReason.field] && typeof record[byReason.field] === 'object'
      ? String(record[byReason.field].reason || '') : '';
    const reasons = byReason ? byReason.reasons : {};
    const row = (cause && Object.prototype.hasOwnProperty.call(reasons, cause) ? reasons[cause] : null)
      || FINDINGS[String(code || '')];
    return row && zh() ? { title: row[0], detail: row[1] } : null;
  }
  function question(code) {
    const text = QUESTIONS[String(code || '')];
    return text && zh() ? text : '';
  }
  function column(key) {
    const row = COLUMNS[String(key || '')];
    return row ? pick(row) : '';
  }
  function value(key, raw) {
    const names = VALUES[String(key || '')];
    const text = String(raw);
    return names && Object.prototype.hasOwnProperty.call(names, text) ? pick(names[text]) : '';
  }
  function tableTitle(...tokens) {
    const text = tokens.map(value => String(value || '').toLowerCase()).join(' ');
    const row = TABLES.find(([pattern]) => pattern.test(text));
    return row ? pick([row[1], row[2]]) : '';
  }
  // What a cited method source is used for, from its typed binding rather
  // than the host's English application sentence.
  function citationUse(row) {
    const support = row && row.method_card_support && typeof row.method_card_support === 'object'
      ? row.method_card_support : {};
    const cards = (Array.isArray(support.matched_card_ids) ? support.matched_card_ids : [])
      .map(id => METHOD_CARDS[String(id)]).filter(Boolean).map(pick);
    const elements = (Array.isArray(row && row.design_elements) ? row.design_elements : [])
      .map(id => DESIGN_ELEMENTS[String(id)]).filter(Boolean).map(pick);
    if (!cards.length && !elements.length) return '';
    const chinese = zh();
    const parts = [];
    if (cards.length) parts.push(chinese ? `方法：${cards.join('、')}` : `Method: ${cards.join(', ')}`);
    if (elements.length) parts.push(chinese ? `支撑：${elements.join('、')}` : `Supports: ${elements.join(', ')}`);
    return parts.join(chinese ? '；' : '; ');
  }

  window.AGENT_READER_VOCAB = Object.freeze({ finding, question, column, value, tableTitle, citationUse, category });
})();
