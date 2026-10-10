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
      // The host's answer to a repeated or stale job-starting decision
      // (webserver/host_action_jobs.py); the page reloads its projection.
      if (error.code === 'host_action_in_progress') {
        return tr('EasyICU is already preparing this step; this page follows it.', 'EasyICU 正在准备这一步，页面会自动更新。');
      }
      if (error.code === 'study_job_running') {
        return tr('Another task of this study is running or starting. Wait for it to finish, then choose again.', '本研究的另一个任务正在运行或准备中，请等它结束后再选择。');
      }
      if (error.code === 'host_action_decision_stale') {
        const status = String(error.details && error.details.job_status || '');
        if (status === 'done') {
          return tr('An earlier task already completed this step. The page now shows the current step.', '之前的任务已经完成了这一步，页面已更新为当前这一步。');
        }
        if (['failed', 'cancelled', 'interrupted'].includes(status)) {
          return tr('This step was started before and did not finish. The page now shows what the study offers next.', '这一步之前已经启动过，没有完成；页面已更新为本研究现在可做的下一步。');
        }
        return tr('The study changed after this page was loaded, so this step no longer applies. The page now shows the current step.', '页面打开后研究状态已经变化，这一步已不再适用；页面已更新为当前这一步。');
      }
      if (['host_action_invalid', 'host_action_request_mismatch', 'host_action_study_mismatch', 'host_action_state_unavailable'].includes(error.code)) {
        return tr('EasyICU could not match this step to the current study. Refresh the project and try again.', 'EasyICU 无法把这一步对应到当前研究，请刷新项目后重试。');
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
      if (error.code === 'research_pipeline_target_trial_plans_on_data') {
        return tr('This study’s target trial is approved, so its plan is generated on the study’s data, not as a metadata-only candidate. Generate the plan.', '这项研究的目标试验已批准，计划要按研究数据生成，不能只读元数据生成候选计划。请直接生成计划。');
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

    /* The stops the signed target trial emulation names: each a threshold
       its host prespecified, judged on the eligible stays. The design is a
       researcher's confirmed choice, so the remedy revises it and confirms
       it again. Dropping a confounder trades confounding for positivity, so
       the positivity and weight stops name narrower eligibility, the
       strategy's definition, or covariates that affect only the start of
       treatment; fewer covariates remedy only a model with too few events. */
    function targetTrialStopCopy(cause) {
      const revise = (en, zh) => tr(
        ` Revise the target trial's design, for example with ${en}, confirm it again, then generate the plan again.`,
        `请修订目标试验的设计（例如${zh}），重新确认后再生成计划。`,
      );
      // A cause outside a table, a prototype key included, names no reason.
      const reason = (table) => (Object.prototype.hasOwnProperty.call(table, cause) ? table[cause] : '');
      const weightWhy = reason({
        separation: tr('the model predicted almost perfectly when stays started the treatment or left the ICU', '模型几乎能完全预测入住何时开始治疗或离开 ICU'),
        not_converged: tr('the model did not converge', '模型没有收敛'),
        singular_design: tr('a covariate did not vary, a category had no stays, or covariates repeated each other\'s information, among the stays the model uses', '在模型使用的入住中，有协变量没有变化、有类别没有入住，或有协变量的信息彼此重复'),
      });
      const bootstrapWhy = reason({
        resamples_failed: tr('more than the prespecified share of bootstrap resamples could not be estimated', '超过预设比例的自助法重抽样无法估计'),
        estimate_outside_interval: tr('an estimate lay outside its own bootstrap interval, which then does not describe it', '有估计值落在它自己的自助法区间之外，这个区间不能描述它'),
      });
      return {
        target_trial_sample_insufficient: tr('Fewer stays met the target trial\'s eligibility at its time zero than the prespecified minimum, so the run has no primary result.', '在时间零点符合目标试验入组条件的入住少于预设的最低数量，这次运行没有主结果。') + revise('an earlier time zero or a wider population', '更早的时间零点或更宽的人群'),
        target_trial_events_insufficient: tr('A strategy had fewer outcome events by the horizon than the prespecified minimum, too few to estimate its risk, so the run has no primary result.', '有一个策略在随访期内的结局事件少于预设的最低数量，不足以估计它的风险，这次运行没有主结果。') + revise('a longer horizon or a wider population', '更长的随访期或更宽的人群'),
        target_trial_strategy_unobserved: tr('Too few stays started the treatment within the grace period to model starting with the chosen covariates, or too few did not start it for the other strategy to be followed, so the run has no primary result.', '在宽限期内开始治疗的入住太少，无法用所选协变量建立开始治疗的模型；或不开始治疗的入住太少，无法跟随另一种策略，这次运行没有主结果。') + revise('a different grace period or fewer covariates', '调整宽限期或减少协变量'),
        target_trial_positivity_violated: tr('For more than the prespecified share of eligible stays, the modelled chance of starting the treatment within the grace period was close to certain or close to none, so the two strategies cannot be compared for them and the run has no primary result.', '超过预设比例的合格入住，在宽限期内开始治疗的模型概率接近必然或接近为零，无法为它们比较两种策略，这次运行没有主结果。') + revise('eligibility that keeps only stays for whom either strategy is plausible, or without covariates that affect starting the treatment but not the outcome', '只纳入两种策略都可能的入住，或去掉只影响是否开始治疗、不影响结局的协变量'),
        target_trial_weights_extreme: tr('A few stays carried most of a strategy\'s weight, so its estimate would rest on them and the run has no primary result.', '有一个策略的权重集中在少数入住上，它的估计将取决于这些入住，这次运行没有主结果。') + revise('narrower eligibility, a shorter grace period, or without covariates that affect starting the treatment but not the outcome', '收窄入组条件、缩短宽限期，或去掉只影响是否开始治疗、不影响结局的协变量'),
        target_trial_icu_exit_excessive: tr('More than the prespecified share of eligible stays left the ICU within the grace period before starting the treatment, and a start after ICU exit is not recorded, so the run has no primary result.', '超过预设比例的合格入住在宽限期内、开始治疗之前离开了 ICU，而离开 ICU 后开始的治疗没有记录，这次运行没有主结果。') + revise('a shorter grace period or a later time zero', '缩短宽限期或把时间零点后移'),
        target_trial_weight_model_not_estimable: tr(`A model of the target trial's weights could not be estimated${weightWhy ? `: ${weightWhy}` : ''}, so the run has no primary result.`, `目标试验的权重模型无法估计${weightWhy ? `：${weightWhy}` : ''}，这次运行没有主结果。`) + revise('merged sparse categories or fewer covariates', '合并稀疏的类别或减少协变量'),
        target_trial_bootstrap_unstable: tr(`The target trial's bootstrap gives no interval to report${bootstrapWhy ? `: ${bootstrapWhy}` : ''}, so the run has no primary result.`, `目标试验的自助法没有可报告的区间${bootstrapWhy ? `：${bootstrapWhy}` : ''}，这次运行没有主结果。`) + revise('fewer covariates or a wider population', '减少协变量或扩大人群'),
      };
    }

    /* A Planner's capability gap: what the question needs that no executable
       family expresses, keyed by its requirement
       (progressive_contract.CapabilityGapRequirement). The host may have been
       unable to check the claim, so the copy says planning found it. */
    function capabilityGapCopy(cause) {
      const why = {
        levels_from_thresholds_unavailable: tr('it groups a variable by thresholds, and EasyICU cannot yet build those groups from this data. Ask about the variable as a continuous measure, or use a variable the data already groups.', '它需要把某个变量按阈值分组，而 EasyICU 目前还不能从这份数据构建这些分组。可以把这个变量作为连续变量来问，或改用数据中已有分组的变量。'),
        longitudinal_representation_unavailable: tr('it needs repeated measurements over time for each stay, and this study\'s data has no such time series. Extract the time series it needs, or ask about a summary over a fixed window.', '它需要每次入住随时间的重复测量，而本研究的数据里没有这样的时间序列。请提取这个问题需要的时间序列，或改为询问固定时间窗内的汇总值。'),
        multiple_sources_required: tr('it needs more than one database, and this study uses one. Ask the question within one database, or run it as a separate study in each database.', '它需要不止一个数据库，而本研究只使用一个。请在单个数据库内提出这个问题，或在每个数据库各开一项研究。'),
        estimand_unsupported: tr('it asks for an estimate no executable EasyICU method produces. Ask for an estimate EasyICU supports instead, for example an association, a prediction or a descriptive summary.', '它要求的估计量没有 EasyICU 的可执行方法能给出。可以改问 EasyICU 支持的估计，例如关联、预测或描述性汇总。'),
        design_element_unsupported: tr('it needs a design element no executable EasyICU method provides yet. Ask the question without it, or with a design EasyICU supports.', '它需要一项 EasyICU 目前没有可执行方法提供的设计要素。可以去掉这个要素再问，或改用 EasyICU 支持的设计。'),
      };
      if (!Object.prototype.hasOwnProperty.call(why, cause)) {
        return tr('Planning found that the question needs something no executable EasyICU method can yet provide, so it stopped before the analysis steps were drafted. No analysis was run.', '规划发现这个问题需要 EasyICU 目前没有可执行方法提供的能力，因此在起草分析步骤之前已停止，没有运行任何分析。');
      }
      return tr('Planning found that the question needs something no executable EasyICU method can yet provide, so it stopped and no analysis was run: ', '规划发现这个问题需要 EasyICU 目前没有可执行方法提供的能力，因此已停止，没有运行任何分析：') + why[cause];
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
        progressive_family_result_contract_unwritable: tr('No executable EasyICU method can yet produce the primary result this survival question needs, so planning stopped before its analysis steps were drafted. Ask whether the outcome occurred within a fixed time as an association question, or choose a survival design EasyICU can execute.', 'EasyICU 目前没有能给出这个生存问题所需主结果的可执行方法，规划在起草分析步骤之前停止。可以改为关联性问题（例如结局是否在固定时间内发生），或选用 EasyICU 能执行的生存分析设计。'),
        progressive_family_spec_cohort_eligibility_after_time_zero: tr('This study decides who is in its cohort after the analysis time zero (its concept window or minimum ICU stay ends later), so planning stopped before the model was called and no analysis was run. End the concept window or minimum ICU stay by time zero, or move time zero later; then prepare the export again and generate a fresh plan.', '这项研究在分析时间零点之后才确定谁入组（概念人群的窗口或最短 ICU 住院时长晚于时间零点），规划在调用模型之前停止，没有运行分析。请让概念窗口或最短住院时长在时间零点前结束，或把时间零点后移；然后重新准备导出，再生成计划。'),
        progressive_family_spec_prediction_risk_set_unavailable: tr('A prediction model is made for the stays still in the ICU when it predicts, which needs each stay\'s ICU length of stay; EasyICU prepares it for a study declared as a prediction model. Planning stopped before the model was called and no analysis was run. Declare the study\'s analysis as a prediction model, then generate the plan again.', '预测模型只针对预测时点仍在 ICU 的入住，需要每次入住的 ICU 住院时长；研究声明为预测模型时，EasyICU 才会准备这一列。规划在调用模型之前停止，没有运行分析。请把研究的分析类型声明为预测模型，再生成计划。'),
        progressive_static_prediction_requires_family_template: tr('A prediction study of this kind is planned from the prediction template, which states the prediction time, the risk set and the predictors available then. This plan did not use the template, so planning stopped and no analysis was run. If prediction is the question you care about most, make it the study\'s primary question and plan again.', '这类预测研究要用预测模板规划，模板会写明预测时点、风险集和可用的预测变量。这份计划没有用模板，规划已停止，没有运行分析。如果预测是你最关心的问题，请把它作为研究的主要问题后重新规划。'),
        execution_complete_not_satisfied: tr('An analysis step did not finish, so the run produced no result it can report.', '有分析步骤没有完成，这次运行没有可以报告的结果。'),
        analysis_validated_not_satisfied: tr('The analysis ran, but its automated validation did not pass, so its results cannot be reported.', '分析已运行，但没有通过自动校验，结果不能报告。'),
        evidence_complete_not_satisfied: tr('The analysis ran, but some results lack the evidence they rest on, so they cannot be reported.', '分析已运行，但有些结果缺少所依据的证据，不能报告。'),
        numeric_verified_not_satisfied: tr('The analysis ran, but some reported numbers could not be checked against the results they come from, so they cannot be reported.', '分析已运行，但有些报告的数字无法与其来源结果核对，不能报告。'),
        // Every analysis step finished; the Writer's model service then
        // stayed unavailable through its retries.
        writer_provider_transport_unavailable: tr('The model service was unavailable while the manuscript was being drafted; the analysis finished and its results are kept. Retry the run to draft it again; finished analysis steps are reused.', '写稿时模型服务不可用；分析已完成，结果已保留。可以重试这次运行重新写稿，已完成的分析步骤会复用。'),
        continuous_survival_interval_result_not_estimable: intervalResultNotEstimableText(cause),
        landmark_survival_interval_result_not_estimable: intervalResultNotEstimableText(cause),
        continuous_survival_exposure_has_one_value: tr('Every analysed stay had the same value of the continuous exposure, so no association with it can be estimated. The run has no primary result. Choose an exposure, summary or window whose values differ between stays, then generate the plan again.', '分析人群中每次入住的连续暴露取值都相同，无法估计与它的关联。这次运行没有主结果。请选用在不同入住之间取值不同的暴露、汇总方式或窗口，再生成计划。'),
        // A prediction step's own stops (contracts/executor_stop.py, static_prediction_model).
        prediction_predictor_unobserved: tr('A predictor in the plan has no value in any development stay, so the model would have been fitted without it. The run stopped; revise the plan\'s predictors, then generate the plan again.', '计划里有一个预测变量在开发集里一个值都没有，模型会在缺少它的情况下拟合，所以运行停止了。请修改计划的预测变量，再重新生成计划。'),
        benchmark_comparator_unobserved: tr('The score the model is compared with has no value in any validation stay, so there is nothing to compare. The run stopped; drop the comparison or use a source that records the score, then generate the plan again.', '用来比较的评分在验证集里一个值都没有，无从比较，所以运行停止了。请去掉这项比较，或改用记录了该评分的数据源，再重新生成计划。'),
        ...targetTrialStopCopy(cause),
        tte_trial_not_confirmed: tr('This study has no confirmed target trial, so planning stopped before any model was called and no analysis was run. Set up the target trial in the conversation and approve it on its confirmation card, then generate the plan.', '这项研究还没有已确认的目标试验，规划在调用模型之前已停止，没有运行分析。请先在对话里设定目标试验，并在确认卡片上批准，再生成计划。'),
        target_trial_compile_drifted: tr('EasyICU derived the approved target trial again from this run\'s data, and the result differs from the version approved on its confirmation card; the data may have been prepared again, or the study setup or EasyICU\'s rules may have changed since. EasyICU runs only the trial that was approved, so planning stopped before any model was called and no analysis was run. Review the target trial on its confirmation card and approve it again, then generate the plan.', 'EasyICU 按这次运行的数据重新生成了已批准的目标试验，结果与确认卡片上批准的版本不同；批准之后数据可能重新准备过，研究配置或 EasyICU 的规则也可能有变化。EasyICU 只运行已批准的目标试验，所以规划在调用模型之前已停止，没有运行分析。请在确认卡片上复核目标试验并重新批准，再生成计划。'),
        progressive_capability_gap: capabilityGapCopy(cause),
        trajectory_stability_refit_failed: tr('The prespecified stability check needs every planned refit to succeed, and a refit could not be completed, so the run has no result. Revise the plan, for example to consider fewer classes, then generate the plan again.', '预设的稳定性检查要求每次计划的重拟合都成功，有重拟合没能完成，这次运行没有结果。请修订计划（例如考虑更少的类别数），再生成计划。'),
        progressive_product_has_multiple_owners: tr('Two steps of the candidate plan produce the same result, so EasyICU cannot tell which step it comes from; planning stopped and no analysis was run. Generate the plan again; the stop is recorded for diagnosis.', '候选计划里有两个步骤产出同一份结果，EasyICU 无法确定它来自哪一步；规划已停止，没有运行分析。请重新生成计划；这次停止已记录，便于排查。'),
        progressive_outline_replay_producer_absent: tr('A step of the candidate outline reuses the result of a method that no earlier step runs, so planning stopped and no analysis was run. Generate the plan again.', '候选大纲里有一步要沿用某个方法的结果，但前面没有步骤运行这个方法；规划已停止，没有运行分析。请重新生成计划。'),
        progressive_outline_trajectory_comparison_unowned: tr('The candidate outline compares outcomes between trajectory groups that a model-written step assigned, and EasyICU compares only groups its own methods assign; planning stopped and no analysis was run. Generate the plan again.', '候选大纲要比较由模型编写的步骤划出的轨迹分组之间的结局，而 EasyICU 只比较由它自己的方法划出的分组；规划已停止，没有运行分析。请重新生成计划。'),
        progressive_family_spec_icu_stay_unit_unread: tr('The prepared data records the ICU length of stay in a unit EasyICU reads as neither days nor hours, so planning stopped before the model was called and no analysis was run. Prepare the export again with the unit recorded.', '准备好的数据里，ICU 住院时长的单位既不是天也不是小时，EasyICU 无法读取；规划在调用模型之前停止，没有运行分析。请重新准备导出，并记录单位。'),
        // An exposure grouping the study states, formed before the plan is
        // outlined (orchestration/exposure_grouping_phase.py) or compared
        // by its analysis template.
        exposure_group_requires_extraction: tr('The study forms its exposure groups from a value the prepared data do not hold, so EasyICU cannot form them on this input. Planning stopped before the plan was outlined and no analysis was run. Prepare data that include this value, or change how the question groups the exposure, then generate the plan again.', '研究要按一个数值分组，但准备好的数据里没有这个值，EasyICU 无法在这份数据上分组。规划在拟定计划大纲之前已停止，没有运行分析。请准备包含该数值的数据，或在对话里修改问题中的分组方式，再重新生成计划。'),
        exposure_group_not_applied: tr('EasyICU cannot apply the study\'s exposure grouping to this input as stated; the detail names the grouping and the reason. Planning stopped before the plan was outlined and no analysis was run. Change how the question groups the exposure, then generate the plan again.', '研究所述的暴露分组无法按原样用在这份数据上，详情里写明了是哪一个分组以及原因。规划在拟定计划大纲之前已停止，没有运行分析。请在对话里修改问题中的分组方式，再重新生成计划。'),
        exposure_grouping_unanswered: tr('After the allowed retries, the model did not state the study\'s exposure groupings in a form EasyICU could read. Planning stopped and no analysis was run. Generate the plan again; if this recurs, state in the question how the exposure is grouped.', '模型在允许的重试次数内，没有以 EasyICU 能读取的形式说明研究的暴露分组。规划已停止，没有运行分析。请重新生成计划；如果再次出现，请在问题里写明暴露如何分组。'),
        exposure_group_variable_unbound: tr('EasyICU staged the exposure groups, but the research context built on them does not hold the grouped variable. This is an EasyICU fault, not a problem with the study. Planning stopped and no analysis was run. Generate the plan again; if it recurs, report it with the run record.', 'EasyICU 已暂存暴露分组，但在其上构建的研究上下文里没有这个分组变量。这是 EasyICU 的内部故障，不是研究本身的问题。规划已停止，没有运行分析。请重新生成计划；如果再次出现，请连同运行记录一起反馈。'),
        exposure_group_candidate_drift: tr('EasyICU formed the approved plan\'s exposure groups again on this run\'s data, and they read other values than in the approved plan; the data may have been prepared again, or EasyICU\'s rules may have changed. EasyICU runs only the grouping that was approved, so planning stopped and no analysis was run. Generate the plan again, then review and approve it.', 'EasyICU 按这次运行的数据重新组成已批准计划里的暴露分组，读取的数值与批准时不同；数据可能重新准备过，EasyICU 的规则也可能有变化。EasyICU 只运行已批准的分组，所以规划已停止，没有运行分析。请重新生成计划，复核后再批准。'),
        exposure_group_level_empty: tr('A group the study stated holds no ICU stay on this data, so the plan would compare a group that is not there. EasyICU does not drop or merge a group on its own. Planning stopped and no analysis was run. Change the grouping\'s thresholds in the question, then generate the plan again.', '研究声明的某个分组在这份数据里一个 ICU 住院都没有，计划会去比较一个并不存在的组。EasyICU 不会自行删除或合并分组。规划已停止，没有运行分析。请在对话里修改问题中的分组阈值，再重新生成计划。'),
        progressive_family_spec_exposure_group_contrast_unavailable: tr('The analysis template cannot compare the exposure groups the study stated: its reference or comparison group is not among the levels the template can offer. Planning stopped and no analysis was run. Change the grouping\'s reference or comparison group in the question, or choose an analysis that can compare these groups, then generate the plan again.', '分析模板无法比较研究声明的暴露分组：参照组或对比组不在模板能提供的水平里。规划已停止，没有运行分析。请在对话里修改问题中分组的参照组或对比组，或改用能比较这些组的分析，再重新生成计划。'),
        // A substance the question names as given or as measured, read by the
        // plan's primary exposure in the other form (planning/question_substance_forms.py).
        progressive_question_substance_form_substituted: tr('The question names a substance in one form (given, or measured as a level), and the plan\'s primary exposure reads it in the other, so the plan would answer another question. Planning stopped and no analysis was run. Generate the plan again; to study the other form, say so in the question.', '题面指的是这种物质的一种形式（作为给药，或作为测得的水平），而计划的主要暴露读成了另一种形式，回答的就成了另一个问题。规划已停止，没有运行分析。请重新生成计划；如果要研究另一种形式，请在问题里写明。'),
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
        research_pipeline_required_concept_without_values: tr(
          'A variable the study requires is listed by its data source, but the source holds no value of it, so planning stopped before any model was called. Change the variable, or use a source that records it, then generate the plan again.',
          '研究需要的某个变量，数据源里有这一列，但一个值都没有，所以在调用模型之前就停止了规划。请更换这个变量，或改用记录了它的数据源，再重新生成计划。',
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
        // A step that cannot run without patient groups (run_patient_grouping).
        research_pipeline_patient_grouping_required: tr(
          'A step of this plan needs to know which ICU stays belong to the same patient (for example, a prediction model keeps all of a patient\'s stays on one side of its development and validation split), and the data this run uses has no verified patient grouping, so analysis stopped before any step ran. If the selected data source cannot provide patient grouping, choose one that can, or ask a question that does not need it; otherwise generate the plan again.',
          '这个计划有步骤需要知道哪些 ICU 住院属于同一位患者（例如预测模型要把同一患者的住院都放在开发集或验证集的同一边），而这次运行用的数据没有已核实的患者分组，分析在任何步骤运行前停止。如果所选数据源不能提供患者分组，请改用能提供的数据源，或改问不需要它的问题；否则请重新生成计划。',
        ),
        research_pipeline_patient_grouping_trajectory_conflict: tr(
          'A step of the accepted plan needs patient grouping, and this study also declares a design read from each stay\'s measurements over time (a trajectory or landmark design). EasyICU cannot yet prepare both in one run, so it stopped before preparing the data. Keep one of them in the conversation, then generate the plan again.',
          '已接受的计划里有步骤需要患者分组，而这项研究还声明了要读取每次住院随时间变化的测量（轨迹或 landmark 设计）。EasyICU 目前不能在一次运行里同时准备这两样，因此在准备数据之前停止。请在对话里只保留其中一项，再生成计划。',
        ),
        agent_plan_configuration_failed: tr(
          'EasyICU could not configure the reviewed plan for execution automatically.',
          'EasyICU 无法自动为已审阅的计划完成执行配置。',
        ),
        web_scientific_runtime_columns_missing: tr(
          'Some adjustment variables in the plan are not in the prepared data, so analysis stopped before it began. The data must be prepared again with those variables.',
          '计划要调整的变量有几列不在准备好的数据里，分析在开始前停止。需要重新准备包含这些变量的数据。',
        ),
        target_trial_materialization_mismatch: tr('The prepared data does not match what the target trial reads: a column it needs is missing, the treatment start was not read from ICU admission through the end of the grace period, or the data does not record the time window each column was read in. Analysis stopped before it began. Prepare the data again for the target trial\'s windows, with covariates read before time zero and the treatment start read through the end of the grace period.', '准备好的数据与目标试验要读取的内容不符：缺少它需要的列，治疗开始时间没有从入住 ICU 起一直读到宽限期结束，或者数据没有记录每一列在哪个时间窗读取。分析在开始前停止。请按目标试验的时间窗重新准备数据：协变量在时间零点之前读取，治疗开始时间一直读到宽限期结束。'),
        target_trial_family_mismatch: tr('This study has a target trial, but its analysis is not declared as causal inference, so analysis stopped before it began. In the conversation, declare the study\'s analysis as causal inference, or remove the target trial, then generate the plan again.', '这项研究带有目标试验，但分析类型没有声明为因果推断，分析在开始前停止。请在对话里把研究的分析类型声明为因果推断，或者去掉目标试验，再生成计划。'),
        target_trial_configuration_invalid: tr('The target trial saved with this study cannot be run as it is: for example, a strategy label contains digits or markup symbols, or its approval was not given for this study. Analysis stopped before it began. Set up the target trial again in the conversation and approve it on its confirmation card, then generate the plan.', '这项研究保存的目标试验无法按现状运行：例如策略标签里有数字或标记符号，或者它的批准记录不属于这项研究。分析在开始前停止。请在对话里重新设定目标试验，并在确认卡片上批准，再生成计划。'),
        research_pipeline_conflicting_sealed_suites: tr('This study combines two designs that each prespecify the whole analysis, for example an approved target trial together with a time-varying, restricted mean survival time (RMST) or landmark design, and a run can follow only one of them. Analysis stopped before it began. Keep one of the designs in the conversation, then generate the plan again.', '这项研究同时带有两个各自预先设定整套分析的设计，例如已批准的目标试验加上时变、限制平均生存时间（RMST）或 landmark 设计，而一次运行只能按其中一个执行。分析在开始前停止。请在对话里只保留其中一个设计，再生成计划。'),
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
