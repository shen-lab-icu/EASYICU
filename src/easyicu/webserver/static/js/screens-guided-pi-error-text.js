/* Owner: Pi failure codes and setup-value helpers → user-facing copy.

   The transport and the session runner report failures as machine codes; the
   shell used to carry their bilingual text, which kept screens-guided-pi.js
   over its size budget. Only the presentation lives here: the codes stay the
   contract with the gateway and the runner.

   D-P2-2 escaping contract: `errorText()` returns RAW user-facing copy — the
   final `return String(error.message || error.code || error)` fallback carries
   untrusted transport text verbatim. 返回值须经esc后插入innerHTML: callers
   MUST pass the result through `esc()` (window.EU_HTML) before inserting it
   into innerHTML. Only `option()` in this file escapes internally. A static
   test scans every `innerHTML ... errorText` interpolation and fails the
   build when `esc` is not on the same expression. */
(function () {
  'use strict';

  const { esc } = window.EU_HTML;

  function create({ tr, staticPreview }) {
    function errorText(error) {
      if (!error) return '';
      if (error.code === 'pi_session_authority_stale') {
        return tr('The study binding changed after this conversation was saved. Rebind it before continuing.', '这段对话保存后研究绑定发生了变化，请先重新绑定再继续。');
      }
      if (error.code === 'pi_provider_auth_failed') {
        return tr('The model service rejected this API credential.', '模型服务拒绝了这个 API 凭据，请检查后重试。');
      }
      if (error.code === 'pi_provider_model_unavailable') {
        return tr('The selected model was not reported by this service.', '该服务没有返回所选模型，请从下方发现的模型中选择。');
      }
      if (error.code === 'pi_provider_connection_failed') {
        return tr('EasyICU could not reach the model service.', 'EasyICU 无法连接到模型服务，请检查地址和服务状态。');
      }
      if (error.code === 'pi_session_project_mismatch') {
        return tr('That Copilot conversation belongs to another research project.', '该研究助手对话属于另一个研究项目，不能在当前项目中打开。');
      }
      if (error.code === 'pi_project_study_context_missing') {
        return tr('This project’s saved study setup no longer exists. Recreate or rebind the project before starting Copilot.', '当前项目保存的研究配置已不存在。请重新创建或绑定项目后再启动研究助手。');
      }
      if (error.code === 'pi_project_initialization_required') {
        return tr('Confirm this project’s study setup before starting Copilot.', '请先确认当前项目的研究配置，再启动研究助手。');
      }
      if (error.code === 'codex_auth_login_required') {
        return tr('Sign in with your ChatGPT account before starting this conversation.', '请先登录你的 ChatGPT 账户，再开始这段对话。');
      }
      if (error.code === 'codex_auth_model_unavailable') {
        return tr('That model is no longer available for this Codex account. Refresh the account model list.', '该 Codex 账户已无法使用这个模型，请刷新账户模型列表。');
      }
      if (error.code === 'codex_auth_url_invalid') {
        return tr('The sign-in link was blocked because it is not a valid OpenAI authorization address.', '登录链接不是有效的 OpenAI 授权地址，已被拦截。');
      }
      if (error.code === 'research_pipeline_execution_runtime_unavailable') {
        return tr('The container runtime that executes analysis code is not running. Start it (Docker Desktop, or "colima start") and run again.', '执行分析代码的容器运行环境未启动。请先启动它（Docker Desktop，或 "colima start"），然后重新运行。');
      }
      if (error.code === 'research_pipeline_execution_retry_futile') {
        // A stop the failed step's executor named repeats on the same plan
        // and data; no repair budget was involved.
        if (error.details && error.details.reason_code === 'execution_retry_repeats_typed_stop') {
          return tr('A retry would repeat the failure: the failed step stopped for a reason its data and the approved plan determine, and nothing it runs on has changed since. Generate a fresh plan with the change the stop names, or update EasyICU before retrying.', '重试只会重复同样的失败：失败步骤因数据和已批准计划本身决定的原因停止，且此后它依赖的代码与运行镜像都没有变化。请按停止原因所指的修改重新生成计划，或先更新 EasyICU 再重试。');
        }
        return tr('A retry would repeat the failure: the failed step has used its automatic repairs, and nothing it runs on has changed since. Generate a fresh plan, or update EasyICU before retrying.', '重试只会重复同样的失败：失败步骤的自动修复已用尽，且此后它依赖的代码与运行镜像都没有变化。请重新生成计划，或先更新 EasyICU 再重试。');
      }
      if (error.code === 'research_pipeline_runner_image_mismatch') {
        return tr('The analysis runner image does not match this EasyICU version. Rebuild it from the current commit, restart EasyICU, and run again.', '分析运行镜像与当前 EasyICU 版本不一致。请按当前提交重建运行镜像并重启 EasyICU，然后重新运行。');
      }
      if (error.code === 'research_pipeline_export_cohort_mismatch') {
        return tr('The bound data package was not extracted for this study’s cohort under the current rule, so it cannot serve as this study’s data. Extract this study’s cohort, or choose another data source, then generate the plan.', '绑定的数据包不是按本研究的人群和当前规则提取的，不能作为本研究的数据。请先为本研究的人群提取数据，或改用其他数据源，再生成计划。');
      }
      if (error.code === 'research_pipeline_export_cohort_invalid') {
        return tr('The study’s cohort or the bound data package’s recorded cohort cannot be executed as an extraction. Check the study’s cohort, or choose another data source.', '本研究的人群设置或绑定数据包记录的人群无法按提取规则执行。请检查人群设置，或改用其他数据源。');
      }
      // D-P3-5: keep the URL as plain text (no <a>) — errorText() returns RAW
      // copy esc'd by callers per D-P2-2, so embedded HTML would be escaped and
      // never clickable. Copy the address into the browser manually. Terminology
      // (receipt/StudyContext) intentionally unchanged.
      if (staticPreview() && String(error.message || '').includes('Failed to fetch')) {
        return tr('This is a static preview without the EasyICU backend. Start EasyICU and open http://127.0.0.1:8765/#guided.', '这是不带 EasyICU 后端的静态预览。请启动 EasyICU，再打开 http://127.0.0.1:8765/#guided。');
      }
      return String(error.message || error.code || error);
    }

    // D-P2-5: hostname-exact-or-suffix preset matching. The previous raw-URL
    // substring check classified any URL containing a first-party marker —
    // including a lookalike registrable domain or a query string — as
    // first-party. Parse the URL and compare the hostname exactly
    // (or as a true subdomain); anything unparseable is custom-openai.
    function presetHostname(base) {
      try {
        const host = new URL(String(base || '')).hostname.toLowerCase().replace(/\.+$/, '');
        return host || '';
      } catch (_) {
        return '';
      }
    }

    function hostnameMatches(host, root) {
      return !!host && (host === root || host.endsWith('.' + root));
    }

    function providerPreset(config, runtime) {
      const transport = config.api_transport || runtime.api_transport || 'openai-completions';
      if (transport === 'anthropic-messages') return 'anthropic';
      if (transport === 'google-generative-ai') return 'google';
      const raw = String(config.base_url || '');
      const host = presetHostname(raw);
      if (!host) return 'custom-openai';
      let port = '';
      try { port = new URL(raw).port || ''; } catch (_) { port = ''; }
      if (hostnameMatches(host, 'api.openai.com')) return 'openai';
      if (hostnameMatches(host, 'openrouter.ai')) return 'openrouter';
      if (hostnameMatches(host, 'api.deepseek.com')) return 'deepseek';
      // The local proxy is loopback-only: exact host plus its pinned port, so
      // `https://127.0.0.1.evil.example:8317/` cannot borrow the preset.
      if ((host === '127.0.0.1' || host === 'localhost') && port === '8317') return 'cliproxyapi';
      return 'custom-openai';
    }

    function option(value, selected, label) {
      // D-P1-5: value/label are untrusted (server model lists); escape both.
      // `selected` is only a strict-equality gate that emits a literal.
      return `<option value="${esc(value)}"${value === selected ? ' selected' : ''}>${esc(label)}</option>`;
    }

    function modelErrorText(code, completedAction) {
      const value = String(code || '');
      if (completedAction) {
        return tr(
          'An EasyICU tool action completed, but the model service could not finish the explanation. Review the completed receipt above; do not repeat the action automatically.',
          'EasyICU 工具操作已完成，但模型服务未能生成最终说明。请以上方已完成的 receipt 为准，不要自动重复执行该操作。'
        );
      }
      if (value === 'pi_shell_token_budget_exhausted' || value === 'pi_shell_session_provider_call_budget_exhausted') {
        return tr(
          'This conversation reached its bounded safety budget. Start a new conversation in the same research project; the StudyContext, literature, data source, runs, and evidence remain bound to the project.',
          '本会话已达到安全预算。请在同一研究项目中新建后续对话；StudyContext、文献、数据源、运行和证据仍保留在项目中。'
        );
      }
      if (value === 'pi_model_context_limit') return tr('The model context limit was reached. Start a new conversation or shorten the request.', '模型上下文已达到上限，请新建会话或缩短请求。');
      if (value === 'pi_model_rate_limited') return tr('The model service is temporarily rate-limited. No EasyICU action was executed; retry shortly.', '模型服务暂时限流。本轮没有执行 EasyICU 操作，请稍后重试。');
      if (value === 'pi_model_provider_unavailable') return tr('The model service connection was interrupted. No EasyICU action was executed; retry after connectivity recovers.', '模型服务连接中断。本轮没有执行 EasyICU 操作，连接恢复后可直接重试。');
      return tr('The model service could not complete this turn. No EasyICU action should be assumed.', '模型服务未能完成本轮，不能据此认为任何 EasyICU 操作已经执行。');
    }

    /* A failed Research Agent run reaches the conversation as a gate/error
       code. The code stays the contract; the researcher sees what stopped the
       run and what to change next, never the bare identifier. Returns RAW
       copy: callers esc() it before insertion (D-P2-2). */
    /* ``detail`` is the gate's lower-layer cause as codes: ``{code, missing,
       cause}`` from the run row's ``gate_detail_code`` /
       ``gate_missing_concepts`` / ``gate_detail_cause_code``. It turns "data
       preparation did not pass" into which variable the data lacks, and "the
       run did not pass a check" into which check and why, without the
       projection carrying free text. */
    /* A stop the continuous-exposure survival suite named: the interval model
       its rejected proportional-hazards check made primary could not be
       estimated. The remedy follows the cause. */
    function intervalResultNotEstimableText(cause) {
      const why = {
        interval_without_event: tr('an interval had no events', '有一个区间没有事件'),
        follow_up_ends_by_final_cutpoint: tr('follow-up ended by the last cut point', '随访在最后一个切点之前就结束了'),
        did_not_converge: tr('the interval model did not converge', '区间模型没有收敛'),
        invalid_contrast_variance: tr('an interval estimate had no valid variance', '有一个区间估计的方差无效'),
        non_finite_estimate: tr('an interval estimate was not finite', '有一个区间估计不是有限值'),
      }[cause];
      const cutPoints = cause === 'interval_without_event' || cause === 'follow_up_ends_by_final_cutpoint';
      const remedy = !why
        ? tr('Change the plan\'s follow-up intervals or its adjustment, then generate the plan again.', '请调整计划的随访区间或调整变量，再生成计划。')
        : cutPoints
          ? tr('Choose interval cut points with events in every interval and follow-up past the last cut point, then generate the plan again.', '请选择每个区间都有事件、且随访超过最后一个切点的区间切点，再生成计划。')
          : tr('Adjust for fewer variables or merge sparse categories, then generate the plan again.', '请减少调整变量或合并稀疏的类别，再生成计划。');
      return tr(
        `The proportional-hazards check rejected one hazard ratio for the whole follow-up, so the planned hazard ratios by follow-up interval are the primary result, and they could not be estimated${why ? `: ${why}` : ''}. The run has no primary result. ${remedy}`,
        `比例风险检验否定了整个随访期使用同一个风险比，因此预先设定的分区间风险比是主结果，但无法估计${why ? `：${why}` : ''}。这次运行没有主结果。${remedy}`,
      );
    }

    function runFailureDetailText(detail) {
      const code = String(detail && detail.code || '').trim();
      const cause = String(detail && detail.cause || '').trim();
      const missing = Array.isArray(detail && detail.missing) ? detail.missing.map(String).filter(Boolean) : [];
      const list = missing.length ? missing.join(', ') : '';
      const detailCopy = {
        required_concepts_unavailable: list
          ? tr(`Variables this study needs are not in this data: ${list}. Choose another data source or drop them from the question.`, `这份数据里没有研究所需的变量：${list}。请更换数据源，或从问题中去掉这些变量。`)
          : tr('A variable this study needs is not in this data.', '这份数据里缺少研究所需的变量。'),
        outcome_concept_undeclared: tr('No executable outcome variable was named; state the outcome in the question (for example in-hospital death).', '没有可执行的结局变量；请在问题中写明结局（例如住院死亡）。'),
        outcome_concept_unavailable: list
          ? tr(`The outcome variable ${list} is not in this data.`, `结局变量 ${list} 不在这份数据里。`)
          : tr('The outcome variable is not in this data.', '结局变量不在这份数据里。'),
        concept_selection_failed: tr('The model did not return a usable variable selection; retry or narrow the question.', '模型没有给出可用的变量选择；请重试或收窄问题。'),
        no_available_concepts: tr('None of the selected variables is available in this data.', '所选变量在这份数据里都不可用。'),
        progressive_family_result_contract_unwritable: tr('No executable EasyICU method can yet produce the primary result this causal or survival question needs, so planning stopped before its analysis steps were drafted. Ask an association question, or choose a design EasyICU can execute.', 'EasyICU 目前没有能给出这个因果或生存问题所需主结果的可执行方法，规划在起草分析步骤之前停止。可以改为关联性问题，或选用 EasyICU 能执行的设计。'),
        progressive_family_spec_cohort_eligibility_after_time_zero: tr('This study decides who is in its cohort after the analysis time zero (its concept window or minimum ICU stay ends later), so planning stopped before the model was called and no analysis was run. End the concept window or minimum ICU stay by time zero, or move time zero later; then prepare the export again and generate a fresh plan.', '这项研究在分析时间零点之后才确定谁入组（概念人群的窗口或最短 ICU 住院时长晚于时间零点），规划在调用模型之前停止，没有运行分析。请让概念窗口或最短住院时长在时间零点前结束，或把时间零点后移；然后重新准备导出，再生成计划。'),
        progressive_family_spec_prediction_risk_set_unavailable: tr('A prediction model is made for the stays still in the ICU when it predicts, which needs each stay\'s ICU length of stay; EasyICU prepares it for a study declared as a prediction model. Planning stopped before the model was called and no analysis was run. Declare the study\'s analysis as a prediction model, then generate the plan again.', '预测模型只针对预测时点仍在 ICU 的入住，需要每次入住的 ICU 住院时长；研究声明为预测模型时，EasyICU 才会准备这一列。规划在调用模型之前停止，没有运行分析。请把研究的分析类型声明为预测模型，再生成计划。'),
        execution_complete_not_satisfied: tr('An analysis step did not finish, so the run produced no result it can report.', '有分析步骤没有完成，这次运行没有可以报告的结果。'),
        analysis_validated_not_satisfied: tr('The analysis ran, but its automated validation did not pass, so its results cannot be reported.', '分析已运行，但没有通过自动校验，结果不能报告。'),
        evidence_complete_not_satisfied: tr('The analysis ran, but some results lack the evidence they rest on, so they cannot be reported.', '分析已运行，但有些结果缺少所依据的证据，不能报告。'),
        numeric_verified_not_satisfied: tr('The analysis ran, but some reported numbers could not be checked against the results they come from, so they cannot be reported.', '分析已运行，但有些报告的数字无法与其来源结果核对，不能报告。'),
        continuous_survival_interval_result_not_estimable: intervalResultNotEstimableText(cause),
        landmark_survival_interval_result_not_estimable: intervalResultNotEstimableText(cause),
        continuous_survival_exposure_has_one_value: tr('Every analysed stay had the same value of the continuous exposure, so no association with it can be estimated. The run has no primary result. Choose an exposure, summary or window whose values differ between stays, then generate the plan again.', '分析人群中每次入住的连续暴露取值都相同，无法估计与它的关联。这次运行没有主结果。请选用在不同入住之间取值不同的暴露、汇总方式或窗口，再生成计划。'),
        trajectory_stability_refit_failed: tr('The prespecified stability check needs every planned refit to succeed, and a refit could not be completed, so the run has no result. Revise the plan, for example to consider fewer classes, then generate the plan again.', '预设的稳定性检查要求每次计划的重拟合都成功，有重拟合没能完成，这次运行没有结果。请修订计划（例如考虑更少的类别数），再生成计划。'),
        progressive_product_has_multiple_owners: tr('Two steps of the candidate plan produce the same result, so EasyICU cannot tell which step it comes from; planning stopped and no analysis was run. Generate the plan again; the stop is recorded for diagnosis.', '候选计划里有两个步骤产出同一份结果，EasyICU 无法确定它来自哪一步；规划已停止，没有运行分析。请重新生成计划；这次停止已记录，便于排查。'),
        progressive_outline_replay_producer_absent: tr('A step of the candidate outline reuses the result of a method that no earlier step runs, so planning stopped and no analysis was run. Generate the plan again.', '候选大纲里有一步要沿用某个方法的结果，但前面没有步骤运行这个方法；规划已停止，没有运行分析。请重新生成计划。'),
        progressive_outline_trajectory_comparison_unowned: tr('The candidate outline compares outcomes between trajectory groups that a model-written step assigned, and EasyICU compares only groups its own methods assign; planning stopped and no analysis was run. Generate the plan again.', '候选大纲要比较由模型编写的步骤划出的轨迹分组之间的结局，而 EasyICU 只比较由它自己的方法划出的分组；规划已停止，没有运行分析。请重新生成计划。'),
        progressive_family_spec_icu_stay_unit_unread: tr('The prepared data records the ICU length of stay in a unit EasyICU reads as neither days nor hours, so planning stopped before the model was called and no analysis was run. Prepare the export again with the unit recorded.', '准备好的数据里，ICU 住院时长的单位既不是天也不是小时，EasyICU 无法读取；规划在调用模型之前停止，没有运行分析。请重新准备导出，并记录单位。'),
      };
      if (detailCopy[code]) return detailCopy[code];
      // Every family-template stop is an EasyICU check that ran no analysis;
      // the generic compile sentence would name a candidate plan step that a
      // refused request never drafted.
      if (code.startsWith('progressive_family_spec_')) {
        return tr(`Planning stopped at an EasyICU check of the study's template plan (code: ${code}); no analysis was run.`, `模板规划没有通过 EasyICU 的检查（代码：${code}），没有运行分析。`);
      }
      // The compile sentence names a variable or level the data cannot
      // resolve, which is true of these stops only.  Any other candidate-plan
      // stop says what is true of every such stop, with its code.
      const unresolvedDataStops = new Set([
        'progressive_unknown_variable',
        'progressive_unknown_robustness_variable',
        'progressive_level_index_out_of_range',
        'progressive_model_levels_unavailable',
        'progressive_distribution_levels_unavailable',
        'progressive_table_one_levels_unavailable',
        'progressive_table_one_group_levels_unavailable',
      ]);
      if (code.startsWith('progressive_') && !unresolvedDataStops.has(code)) {
        return tr(`Planning stopped at an EasyICU check of the candidate plan (code: ${code}); no analysis was run.`, `候选计划没有通过 EasyICU 的检查（代码：${code}），规划停止，没有运行分析。`);
      }
      return '';
    }

    function runFailureText(code, detail) {
      const value = String(code || '').trim();
      if (!value) return '';
      const suffix = runFailureDetailText(detail);
      const withDetail = text => (suffix ? `${text} ${suffix}` : text);
      // A failed-closed run did not pass a check after it started. Its detail
      // says which check and why; the fallback holds for every check.
      if (value === 'research_agent_pipeline_failed_closed') {
        return suffix || tr('The run did not pass EasyICU\'s checks, so its results cannot be reported.', '这次运行没有通过 EasyICU 的检查，结果不能报告。');
      }
      const known = {
        research_pipeline_planning_identity_unavailable: tr(
          'The selected database has no ICU-stay identity definition for planning. Choose a supported database family and generate the plan again.',
          '所选数据库缺少可用于规划的 ICU 住院身份定义。请改用受支持的数据库家族后重新生成计划。',
        ),
        research_pipeline_progressive_compile_failed: tr(
          'The candidate plan did not pass its compile check: one step referenced a variable level or model term that the current data definition cannot resolve. Regenerating the plan lets the Planner re-derive it; narrowing the exposure levels or covariates in the question also helps.',
          '候选计划未通过编译校验：某一步引用了当前数据定义无法解析的变量水平或模型项。重新生成计划会让 Planner 重新推导；也可以在问题里收窄分组水平或协变量。',
        ),
        research_pipeline_required_concept_structurally_unavailable: tr(
          'A variable the question requires has no supported source in this database. Change the variable or the database before planning.',
          '问题里要求的某个变量在该数据库没有受支持的来源。请更换变量或数据库后再规划。',
        ),
        research_pipeline_plan_contract_exhausted: tr(
          'Four plan drafts in a row did not pass the scientific checks, so analysis did not start.',
          '连续 4 版计划草案都没有通过科学检查，分析没有开始。',
        ),
        research_pipeline_planner_provider_unavailable: tr(
          'The model service was unavailable while planning. Check the connection and generate the plan again.',
          '规划期间模型服务不可用。请检查连接后重新生成计划。',
        ),
        research_pipeline_planner_efficiency_budget_exhausted: tr(
          'The Planner reached its efficiency budget; a validated checkpoint was saved.',
          'Planner 已达到效率预算；已保存验证检查点。',
        ),
        research_pipeline_execution_runtime_unavailable: tr(
          'The container runtime that executes analysis code was not running.',
          '执行分析代码的容器运行环境未启动。',
        ),
        research_pipeline_runner_image_mismatch: tr(
          'The analysis runner image did not match this EasyICU version.',
          '分析运行镜像与当前 EasyICU 版本不一致。',
        ),
        research_pipeline_approved_run_failed: tr(
          'The approved analysis stopped before it finished and cannot resume. The failure is recorded for diagnosis.',
          '已批准的分析在完成前停止，且无法恢复；失败已记录，供排查。',
        ),
        research_pipeline_review_resume_failed: tr(
          'The approved plan could not resume. Its review is still open, so it can be approved again.',
          '已批准的计划未能恢复执行；审阅仍然有效，可以再次批准。',
        ),
        data_foundation_blocked: tr(
          'Data preparation did not pass, so no plan was generated.',
          '数据准备未通过，因此没有生成计划。',
        ),
        agent_plan_patient_grouping_unavailable: tr(
          'The plan analyses every ICU stay, which needs source-owned patient grouping (linking repeated stays of one patient); this data package does not provide it. Choose an admission rule below, or request a plan change.',
          '计划按全部 ICU 住院分析，这需要数据源提供患者分组（把同一患者的多次住院归并），当前数据包没有提供。请在下方选择入住规则，或提出修改。',
        ),
        agent_plan_configuration_failed: tr(
          'EasyICU could not configure the reviewed plan for execution automatically.',
          'EasyICU 无法自动为已审阅的计划完成执行配置。',
        ),
        web_scientific_runtime_columns_missing: tr(
          'Some adjustment variables in the plan are not in the prepared data, so analysis stopped before it began. The data must be prepared again with those variables.',
          '计划要调整的变量有几列不在准备好的数据里，分析在开始前停止。需要重新准备包含这些变量的数据。',
        ),
        web_scientific_runtime_schema_unavailable: tr(
          'The prepared data table could not be read, so analysis stopped before it began. Prepare the data again.',
          '准备好的数据表无法读取，分析在开始前停止。请重新准备数据。',
        ),
        web_scientific_runtime_covariate_encoding_unsupported: tr(
          'One adjustment variable has a data type the model cannot use yet.',
          '有一个调整变量的数据类型目前无法用于建模。',
        ),
        web_scientific_runtime_projection_ambiguous: tr(
          'The plan does not map to the prepared data unambiguously, so analysis stopped before it began.',
          '计划与准备好的数据对应关系不唯一，分析在开始前停止。',
        ),
        research_pipeline_execution_failed: tr(
          'An analysis step failed while running.',
          '有一个分析步骤在运行中失败了。',
        ),
        research_pipeline_provider_timeout: tr(
          'The model service did not answer in time. Try again shortly.',
          '模型服务响应超时，请稍后重试。',
        ),
        research_pipeline_time_window_invalid: tr(
          'The study time window is not valid for this data.',
          '研究时间窗对这份数据无效。',
        ),
        research_pipeline_package_binding_changed: tr(
          'The analysis data changed while the run was in progress, so the run stopped.',
          '运行期间分析数据发生了变化，本次运行已停止。',
        ),
        research_pipeline_database_required: tr(
          'No confirmed database was available for this run.',
          '这次运行没有已确认的数据库。',
        ),
        research_pipeline_database_unknown: tr(
          'The run named a database EasyICU does not support.',
          '这次运行使用的数据库不受 EasyICU 支持。',
        ),
        research_pipeline_cancelled: tr('The run was cancelled.', '运行已取消。'),
        research_pipeline_codex_user_auth_required: tr(
          'The model account needs to sign in again.',
          '模型账户需要重新登录。',
        ),
        research_pipeline_pi_verified_credentials_required: tr(
          'The model connection must be verified before this run.',
          '需要先验证模型连接，才能运行。',
        ),
        research_pipeline_schema_validation_failed: tr(
          'A plan or result file did not pass its structure check.',
          '计划或结果文件没有通过结构校验。',
        ),
        research_pipeline_execution_retry_input_invalid: tr(
          'The inputs this retry needs are incomplete.',
          '这次重试需要的运行输入不完整。',
        ),
        WRITER_ONLY_AUTHORITY_REPAIR_FAILED_PRIOR_PRESERVED: tr(
          'The new report version did not pass the number and evidence checks; the previous version is kept.',
          '新版本报告没有通过数字与证据校验，保留原版本。',
        ),
      };
      // A typed compile reason explains the stop itself; the generic compile
      // sentence would name a cause that did not happen.
      if (value === 'research_pipeline_progressive_compile_failed' && suffix) return suffix;
      if (known[value]) return withDetail(known[value]);
      // An unlisted code stays visible for diagnosis, after a plain sentence.
      return withDetail(tr(`The run stopped before finishing (code: ${value}).`, `运行在完成前停止（代码：${value}）。`));
    }

    return Object.freeze({ errorText, modelErrorText, providerPreset, option, runFailureText });
  }

  window.EasyICU.guidedPi.declare('errorText', { create });
})();
