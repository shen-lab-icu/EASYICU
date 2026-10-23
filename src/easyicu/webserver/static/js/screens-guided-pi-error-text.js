/* Owner: Pi failure codes and setup-value helpers → user-facing copy.

   The transport and the session runner report failures as machine codes; the
   shell used to carry their bilingual text, which kept screens-guided-pi.js
   over its size budget. Only the presentation lives here: the codes stay the
   contract with the gateway and the runner.

   D-P2-2 escaping contract: `errorText()` returns RAW user-facing copy — the
   final `fallbackText(error)` keeps a readable server sentence verbatim, and
   that text is untrusted transport text. 返回值须经esc后插入innerHTML: callers
   MUST pass the result through `esc()` (window.EU_HTML) before inserting it
   into innerHTML. Only `option()` in this file escapes internally. A static
   test scans every `innerHTML ... errorText` interpolation and fails the
   build when `esc` is not on the same expression. */
(function () {
  'use strict';

  const { esc } = window.EU_HTML;

  // Folder scans return codes, including intake's manifest/layout failures.
  // Both extraction screens use these sentences; unknown codes stay readable.
  function scanErrorText(code, tr) {
    const messages = {
      no_path: ['Choose or paste a folder path first.', '请先选择或粘贴文件夹路径。'],
      not_a_directory: ['That path is not a folder EasyICU can open.', '该路径不是 EasyICU 能打开的文件夹。'],
      permission_denied: ['EasyICU does not have permission to read that folder.', 'EasyICU 没有读取该文件夹的权限。'],
      unrecognized_folder: ['This folder has no supported ICU data layout.', '该文件夹中没有可识别的 ICU 数据结构。'],
      scan_api_unavailable: ['The folder scanner is unavailable; restart EasyICU and try again.', '文件夹扫描服务暂不可用，请重启 EasyICU 后重试。'],
      database_mismatch: ['This folder belongs to a different database; choose the database this study uses.', '该文件夹属于其他数据库，请选择本研究使用的数据库文件夹。'],
      database_detection_ambiguous: ['This folder contains conflicting database identities; choose a folder for one database.', '该文件夹中的数据库标识相互冲突，请选择只包含一个数据库的文件夹。'],
      database_detection_manifest_invalid: ['The export manifest names an unsupported database.', '导出清单中的数据库不受支持。'],
      database_detection_schema_unreadable: ['The table headers could not be read; choose a readable data folder.', '无法读取数据表的字段信息，请选择可读取的数据文件夹。'],
      database_detection_unavailable: ['The database could not be identified from this folder.', '无法从该文件夹识别数据库。'],
      unavailable_when_hosted: ['The administrator prepares data on this server.', '这台服务器上的数据由管理员准备。'],
      path_outside_data_roots: ['This folder is outside the data the administrator prepared.', '该文件夹不在管理员准备的数据范围内。'],
    };
    const value = String(code || '');
    let text = messages[value];
    if (!text && /^(?:export_manifest|manifest_marker)/.test(value)) {
      text = ['The export manifest is missing or unreadable; choose a complete export.', '导出清单缺失或无法读取，请选择完整的导出文件夹。'];
    }
    if (!text && /^manifest_(?:path|file)/.test(value)) {
      text = ['An export member is missing or points outside the export; choose a complete export.', '导出文件缺失或指向导出目录之外，请选择完整的导出文件夹。'];
    }
    if (!text && /^column_metadata_/.test(value)) {
      text = ['The export column metadata is incomplete or invalid; prepare the export again.', '导出文件的字段元数据不完整或无效，请重新准备导出。'];
    }
    if (!text && /^(?:manifest_|file_|export_package|export_layout)/.test(value)) {
      text = ['The export structure does not match its manifest; choose a complete export.', '导出文件结构与清单不符，请选择完整的导出文件夹。'];
    }
    return tr(...(text || ['This folder could not be scanned; choose another folder.', '无法扫描该文件夹，请选择其他文件夹。']));
  }

  function create({ tr, staticPreview }) {
    function errorText(error) {
      if (!error) return '';
      const resumeFailure = developmentResumeFailureText(error.code)
        || developmentResumeFailureText(error.message)
        || developmentResumeFailureText(typeof error === 'string' ? error : '');
      if (resumeFailure) return resumeFailure;
      // The hosted login gateway ended the session (js/hosted-session.js
      // shows the way back to the login page).
      if (error.code === 'login_required') {
        return tr('Your sign-in has expired. Sign in again to continue.', '登录已过期，请重新登录后继续。');
      }
      // Hosted boundaries (webserver/hosted.py): the administrator's data is
      // read-only and some actions belong to the administrator.
      if (error.code === 'hosted_data_read_only') {
        return tr('The data the administrator prepared is read-only; results are saved in your own workspace.', '管理员准备的数据是只读的；结果会保存在你自己的工作区。');
      }
      if (error.code === 'path_outside_data_roots') {
        return tr('That location is outside the data the administrator prepared.', '这个位置不在管理员准备的数据范围内。');
      }
      if (error.code === 'unavailable_when_hosted') {
        return tr('The administrator handles this action. Please contact them.', '这项操作由管理员负责，请联系管理员。');
      }
      if (error.code === 'pi_session_authority_stale') {
        return tr('The study binding changed after this conversation was saved. Rebind it before continuing.', '这段对话保存后研究绑定发生了变化，请先重新绑定再继续。');
      }
      // Choosing a registered export the host offered
      // (pi_copilot/registered_source_offer.py); nothing was bound.
      if (error.code === 'pi_data_source_not_offered' || error.code === 'pi_data_source_not_registered') {
        return tr('That export is no longer offered. Reload the page and choose again.', '这份数据已不在可选范围内，请刷新页面后重新选择。');
      }
      if (error.code === 'pi_data_source_database_mismatch') {
        return tr('That export is not of the database your question names.', '这份数据不是你的问题指定的数据库。');
      }
      if (error.code === 'pi_source_id_required') {
        return tr('Choose one of the offered exports.', '请从列出的数据中选择一份。');
      }
      if (error.code === 'pi_session_data_source_authorization_not_pending') {
        return tr('This conversation’s data source was already settled. Reload the page to see it.', '本次会话的数据源已经确定，请刷新页面查看。');
      }
      if (error.code === 'study_context_active_job_conflict') {
        return tr('A data job for this study is still running; choose the source after it finishes.', '这项研究还有数据任务在运行，请等它结束后再选择数据源。');
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
      // The connect-then-choose form (pi_copilot/provider_config.py).
      if (error.code === 'pi_provider_credential_required') {
        return tr('Enter the API credential for this service. A saved credential is reused only for the same service and address.', '请输入这个服务的 API 凭据。已保存的凭据只用于同一服务和地址。');
      }
      if (error.code === 'pi_provider_request_invalid') {
        return tr('Check the service, address and credential, then try again.', '请检查服务、地址和凭据后重试。');
      }
      if (error.code === 'pi_provider_verification_failed') {
        const status = Number(error.details && error.details.status_code);
        const suffix = Number.isFinite(status) && status > 0 ? tr(` (status ${status})`, `（状态码 ${status}）`) : '';
        return tr('The model service did not accept the request. Check the address and protocol.', '模型服务没有接受这次请求，请检查地址和兼容协议。') + suffix;
      }
      if (error.code === 'pi_provider_models_response_invalid') {
        return tr('This service did not return a model list EasyICU can read. Check the protocol under Advanced settings.', '该服务返回的模型列表无法读取，请在高级设置中检查兼容协议。');
      }
      if (error.code === 'pi_provider_base_url_rejected') {
        return tr('This computer’s security policy does not allow this service address. Use another address.', '本机安全策略不允许使用这个服务地址，请换一个地址。');
      }
      if (error.code === 'pi_provider_api_transport_unsupported') {
        return tr('Choose a supported protocol under Advanced settings.', '请在高级设置中选择一个支持的兼容协议。');
      }
      if (error.code === 'external_llm_opt_in_required') {
        return tr('Confirm the use of an external model service before connecting.', '连接前需要你确认使用外部模型服务。');
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
      // The folder selection a researcher leaves (pi_copilot/service.py).
      if (error.code === 'pi_session_local_selection_job_active') {
        return tr('A data task of this study is still running. The folder selection can be left once it finishes.', '本研究的数据任务还在运行，等它结束后才能放弃本地目录选择。');
      }
      if (error.code === 'pi_session_local_selection_not_started') {
        return tr('The local folder selection has already ended. Refresh this conversation.', '本地目录选择已经结束，请刷新这段对话。');
      }
      // The Web memory admission (pi_copilot/resource_lifecycle.py): nothing was sent.
      if (error.code === 'pi_web_memory_pressure') {
        return tr('EasyICU is busy finishing other work and did not send this message. Wait a minute, then send it again.', 'EasyICU 正忙于处理其他任务，这条消息没有发送。请稍等一分钟再发送。');
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
      if (error.code === 'study_causal_trial_design_unset') {
        return tr('This question needs a target-trial design. State the trial in this conversation before generating its plan; analysis has not started.', '这道题需要按目标试验规划。请先在当前对话中说明目标试验，再生成计划；分析尚未开始。');
      }
      if (error.code === 'target_trial_treatment_not_registered') {
        const gap = error.details && error.details.causal_trial_gap || {};
        const names = (Array.isArray(gap.treatment_concepts) ? gap.treatment_concepts : []).slice(0, 6)
          .map(row => row && tr(row.label_en, row.label_zh)).filter(Boolean).map(text => String(text).slice(0, 80));
        const classes = (Array.isArray(gap.supported_classes) ? gap.supported_classes : []).slice(0, 6)
          .map(row => row && tr(row.label_en, row.label_zh)).filter(Boolean).map(text => String(text).slice(0, 80));
        return tr(
          `This target trial uses a treatment not registered for emulation${names.length ? `: ${names.join(', ')}` : ''}. No analysis started.${classes.length ? ` Registered treatment classes: ${classes.join(', ')}.` : ''} Revise the research design in the conversation if appropriate.`,
          `这道目标试验题所用治疗尚未登记为可模拟治疗${names.length ? `：${names.join('、')}` : ''}，分析没有开始。${classes.length ? `当前登记的治疗类别：${classes.join('、')}。` : ''}如需修改研究设计，请在对话中说明。`,
        );
      }
      if (error.code === 'study_family_reading_disagreement') {
        const gap = studyFamilyGapText(error.details && error.details.study_family_gap);
        return tr(`${gap.refusal}; no analysis started.`, `${gap.refusal}，分析没有开始。`);
      }
      if (error.code === 'study_family_database_out_of_scope') {
        const gap = studyFamilyGapText(error.details && error.details.study_family_gap);
        return tr(
          `${gap.refusal}, so no analysis started. ${gap.instead}`,
          `${gap.refusal}，分析没有开始。${gap.instead}`,
        );
      }
      if (error.code === 'target_trial_database_out_of_scope') {
        return tr('Target-trial emulation is currently supported only for MIMIC-IV. This study’s source is outside that scope, so analysis did not start. Revise the study in the conversation if appropriate.', '当前目标试验模拟只支持 MIMIC-IV。本研究的数据源不在该范围内，分析没有开始；如需修改研究，请在对话中说明。');
      }
      // D-P3-5: keep the URL as plain text (no <a>) — errorText() returns RAW
      // copy esc'd by callers per D-P2-2, so embedded HTML would be escaped and
      // never clickable. Copy the address into the browser manually. Terminology
      // (receipt/StudyContext) intentionally unchanged.
      if (staticPreview() && String(error.message || '').includes('Failed to fetch')) {
        return tr('This is a static preview without the EasyICU backend. Start EasyICU and open http://127.0.0.1:8765/#guided.', '这是不带 EasyICU 后端的静态预览。请启动 EasyICU，再打开 http://127.0.0.1:8765/#guided。');
      }
      return fallbackText(error);
    }

    // A job's end event carries its failure as text, usually `code: detail`;
    // the code chooses the copy and the detail stays for the fallback.
    function jobErrorText(raw) {
      const text = String(raw || '').trim();
      const match = text.match(/^([a-z][a-z0-9_]{2,79})(?:\s*:\s*|$)/);
      if (!match) return errorText({ message: text });
      return errorText({ code: match[1], message: text.slice(match[0].length) || match[1] });
    }

    // Whom a clinician asks: the administrator a hosted instance names.
    function askAdministrator() {
      const registry = window.EasyICU && window.EasyICU.guidedPi;
      const hosted = registry && typeof registry.optional === 'function' ? registry.optional('hosted') : null;
      const contact = hosted && hosted.active() ? hosted.supportContact() : '';
      return contact
        ? { en: `contact the administrator (${contact})`, zh: `请联系管理员（${contact}）` }
        : { en: 'contact the administrator', zh: '请联系管理员' };
    }

    // A failure no copy above names. A server sentence in the reader's
    // language is kept; a machine code, a transport string or a sentence in
    // another language becomes a plain sentence, with the code (or the
    // original text) in brackets for the administrator.
    function fallbackText(error) {
      const value = typeof error === 'object' ? error : { message: String(error) };
      const message = String(value.message || '').trim();
      const code = String(value.code || '').trim().slice(0, 80);
      const status = Number(value.status || 0);
      // Request-validation arrays have no typed code. Reuse the generic
      // fallback's first sentence without exposing validation internals.
      if (status === 422 && !code) {
        return tr('This step did not complete (HTTP 422).', '这一步没有完成（HTTP 422）。');
      }
      const ask = askAdministrator();
      if (!status && /Failed to fetch|NetworkError|Load failed|ERR_NETWORK/i.test(message)) {
        return tr('EasyICU could not reach its server. Check the network, then try again.', '连不上 EasyICU 服务器。请检查网络后重试。');
      }
      if (status >= 500) {
        const ref = code || `HTTP ${status}`;
        return tr(`The EasyICU server hit an error on this step (${ref}). Try again shortly; if it keeps happening, ${ask.en}.`, `EasyICU 服务器处理这一步时出错了（${ref}）。请稍后重试；如果反复出现，${ask.zh}。`);
      }
      const zh = tr('', 'zh') === 'zh';
      const machine = !message || message === code || /^[A-Za-z][A-Za-z0-9_.:-]*$/.test(message) || /->\s*HTTP\s+\d+/.test(message);
      const foreign = zh && !/[\u3400-\u9fff]/.test(message);
      if (!machine && !foreign) return message;
      const ref = code || (status ? `HTTP ${status}` : '') || message.slice(0, 120);
      return tr(`This step did not complete${ref ? ` (${ref})` : ''}. If it keeps happening, ${ask.en}.`, `这一步没有完成${ref ? `（${ref}）` : ''}。如果反复出现，${ask.zh}。`);
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
          'EasyICU 工具操作已完成，但模型服务未能生成最终说明。请以上方已完成的记录为准，不要自动重复执行该操作。'
        );
      }
      if (value === 'pi_shell_message_provider_call_budget_exhausted' || value === 'pi_shell_message_cost_budget_exhausted') {
        return tr(
          'This reply reached its model budget before it finished. Send "continue" to carry on; completed EasyICU actions are kept.',
          '本轮回复已用完单次模型预算，尚未说完。发送“继续”即可接着完成，已完成的 EasyICU 操作会保留。'
        );
      }
      if (value === 'pi_shell_token_budget_exhausted' || value === 'pi_shell_session_provider_call_budget_exhausted'
          || value === 'pi_shell_session_cost_budget_exhausted') {
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
    function capabilityGapCopy(cause, list) {
      const why = {
        levels_from_thresholds_unavailable: tr('it groups a variable by thresholds, and EasyICU cannot yet build those groups from this data. Ask about the variable as a continuous measure, or use a variable the data already groups.', '规划发现这个问题需要按阈值分组，而 EasyICU 目前无法从这份数据构建这些分组，已停止规划且未运行分析；可以改问连续变量，或使用已有分组的变量。'),
        longitudinal_representation_unavailable: tr('it needs repeated measurements over time for each stay, and this study\'s data has no such time series. Extract the time series it needs, or ask about a summary over a fixed window.', '规划发现这个问题需要每次入住随时间的重复测量，但本研究的数据里没有这样的时间序列，已停止规划且未运行分析；请提取所需时间序列，或改问固定时间窗的汇总值。'),
        multiple_sources_required: tr('it needs more than one database, and this study uses one. Ask the question within one database, or run it as a separate study in each database.', '规划发现这个问题需要多个数据库，而本研究只使用一个，已停止规划且未运行分析；请在单个数据库内提问，或在每个数据库各开一项研究。'),
        estimand_unsupported: tr('it asks for an estimate no executable EasyICU method produces. Ask for an estimate EasyICU supports instead, for example an association, a prediction or a descriptive summary.', '规划发现这个问题需要当前方法无法给出的估计量，已停止规划且未运行分析；可以改问支持的估计，例如关联、预测或描述性汇总。'),
        design_element_unsupported: list
          ? tr(`it needs a design element involving ${list} that no executable EasyICU method provides yet. Ask the question without it, or with a design EasyICU supports.`, `规划发现这个问题需要一项涉及 ${list} 的设计要素，EasyICU 目前没有可执行方法提供，已停止规划且未运行分析。`)
          : tr('it needs a design element no executable EasyICU method provides yet. Ask the question without it, or with a design EasyICU supports.', '规划发现这个问题需要一项 EasyICU 目前没有可执行方法提供的设计要素，已停止规划且未运行分析。'),
      };
      if (!Object.prototype.hasOwnProperty.call(why, cause)) {
        return tr('Planning found that the question needs something no executable EasyICU method can yet provide, so it stopped before the analysis steps were drafted. No analysis was run.', '规划发现这个问题需要 EasyICU 目前没有可执行方法提供的能力，因此在起草分析步骤之前已停止，没有运行任何分析。');
      }
      return tr('Planning found that the question needs something no executable EasyICU method can yet provide, so it stopped and no analysis was run: ', '') + why[cause];
    }

    function runFailureDetailText(detail) {
      const code = String(detail && detail.code || '').trim();
      const cause = String(detail && detail.cause || '').trim();
      const missing = Array.isArray(detail && detail.missing) ? detail.missing.map(String).filter(Boolean) : [];
      const list = missing.length ? missing.join(', ') : '';
      const detailCopy = {
        declared_exposure_level_empty: tr(
          'A declared group has no analysable stays, so the run stopped; generate a new plan that checks eligibility and preserves every group.',
          '计划中的一个分组没有可分析的住院，因此运行已停止，请重新生成计划，检查入组条件并保留所有分组。',
        ),
        landmark_spline_binary_sensitivity_unsupported: tr(
          'A sensitivity analysis in the plan reused the continuous exposure result for a two-group comparison, which needs its own grouped model, so EasyICU stopped before any analysis ran. Generate the plan again.',
          '计划里有一项敏感性分析沿用了连续暴露的结果做两组比较，这需要单独按分组建模，因此在分析前停止。请重新生成计划。',
        ),
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
        // Every evidence error was a manuscript check; no result lacks its evidence.
        manuscript_checks_not_passed: tr('The analysis finished and its results can be reviewed, but the automatically drafted manuscript did not pass its checks (for example, a section was incomplete), so it is not a reportable manuscript. Retry the run to draft it again; finished analysis steps are reused.', '分析已完成，结果可以在分析审阅中查看；但自动起草的稿件没有通过检查（例如某一部分不完整），还不能作为可报告的稿件。可以重试这次运行重新写稿，已完成的分析步骤会复用。'),
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
        progressive_capability_gap: capabilityGapCopy(cause, list),
        progressive_exposure_window_required: tr('The exposure is a measurement with no time window, so EasyICU cannot tell whether it was already known at ICU admission or measured later. Planning stopped and no analysis was run. State the window in the question, for example the first 24 hours after admission, then generate the plan again.', '暴露是一项测量值，但没有设定时间窗，EasyICU 无法确定它在入 ICU 时就已知，还是之后才测得。规划已停止，没有运行分析。请在问题里写明时间窗，例如入 ICU 后最初 24 小时，然后重新生成计划。'),
        progressive_continuous_outcome_invalid: tr('The grouped summary could not be tied to a categorical grouping and a numeric measure EasyICU can verify, so planning stopped and no analysis was run. State in the question which groups to compare and which measure to summarize, then generate the plan again.', '分组汇总没能对应到 EasyICU 可核验的分类分组和数值指标，规划已停止，没有运行分析。请在问题里写明按什么分组、汇总哪个指标，再重新生成计划。'),
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
        exposure_grouping_unanswered: tr('After the allowed retries, the model did not state the study\'s exposure groupings in a form EasyICU could read. Planning stopped and no analysis was run. Generate the plan again; if this recurs, keep the run record for diagnosis.', '模型在允许的重试次数内，没有以 EasyICU 能读取的形式说明研究的暴露分组。规划已停止，没有运行分析。可重新生成计划；若仍失败，请保留运行记录供排查。'),
        exposure_group_variable_unbound: tr('EasyICU staged the exposure groups, but the research context built on them does not hold the grouped variable. This is an EasyICU fault, not a problem with the study. Planning stopped and no analysis was run. Generate the plan again; if it recurs, report it with the run record.', 'EasyICU 已暂存暴露分组，但在其上构建的研究上下文里没有这个分组变量。这是 EasyICU 的内部故障，不是研究本身的问题。规划已停止，没有运行分析。请重新生成计划；如果再次出现，请连同运行记录一起反馈。'),
        exposure_group_candidate_drift: tr('EasyICU formed the approved plan\'s exposure groups again on this run\'s data, and they read other values than in the approved plan; the data may have been prepared again, or EasyICU\'s rules may have changed. EasyICU runs only the grouping that was approved, so planning stopped and no analysis was run. Generate the plan again, then review and approve it.', 'EasyICU 按这次运行的数据重新组成已批准计划里的暴露分组，读取的数值与批准时不同；数据可能重新准备过，EasyICU 的规则也可能有变化。EasyICU 只运行已批准的分组，所以规划已停止，没有运行分析。请重新生成计划，复核后再批准。'),
        exposure_group_level_empty: tr('A group the study stated holds no ICU stay on this data, so the plan would compare a group that is not there. EasyICU does not drop or merge a group on its own. Planning stopped and no analysis was run. Change the grouping\'s thresholds in the question, then generate the plan again.', '研究声明的某个分组在这份数据里一个 ICU 住院都没有，计划会去比较一个并不存在的组。EasyICU 不会自行删除或合并分组。规划已停止，没有运行分析。请在对话里修改问题中的分组阈值，再重新生成计划。'),
        progressive_family_spec_exposure_group_contrast_unavailable: tr('The analysis template cannot compare the exposure groups the study stated: its reference or comparison group is not among the levels the template can offer. Planning stopped and no analysis was run. Change the grouping\'s reference or comparison group in the question, or choose an analysis that can compare these groups, then generate the plan again.', '分析模板无法比较研究声明的暴露分组：参照组或对比组不在模板能提供的水平里。规划已停止，没有运行分析。请在对话里修改问题中分组的参照组或对比组，或改用能比较这些组的分析，再重新生成计划。'),
        progressive_exposure_group_contrast_mismatch: tr('The proposed plan changed the reference or comparison group the study stated, or used conflicting contrasts across its steps. Planning stopped and no analysis was run. Generate the plan again.', '拟定的计划改动了研究声明的参照组或比较组，或在不同步骤用了不一致的比较方向。规划已停止，没有运行分析。请重新生成计划。'),
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

    function developmentResumeFailureText(code) {
      const value = String(code || '').trim();
      if (!value.startsWith('research_pipeline_development_resume_')) return '';
      const reason = value === 'research_pipeline_development_resume_phase_not_replayed'
        ? tr('The exposure groups or research context saved with the planning checkpoint could not be restored exactly.', '无法原样恢复规划检查点保存的暴露分组或对应的研究上下文。')
        : /scope_mismatch|plan_contract_mismatch|acquisition_authority_mismatch/.test(value)
          ? tr('The saved planning checkpoint belongs to different study inputs or requirements.', '已保存的规划检查点与当前研究输入或要求不一致。')
          : /literature_/.test(value)
            ? tr('The literature bound to the saved planning checkpoint could not be verified.', '无法校验规划检查点绑定的文献依据。')
            : tr('The saved planning checkpoint could not be verified for continuation.', '已保存的规划检查点未通过继续生成计划所需的校验。');
      return reason.replace(/[。.]+$/, '') + tr(
        '; use the study card to prepare the same reviewed candidate again.',
        '，请在研究卡片上按原已审阅的候选方案重新准备数据。',
      );
    }

    // Why v1 does not offer a study family on a database (host:
    // study_family_scope): the refusal, and what to ask instead.  The gap is
    // codes only; a name it does not know reads generically.
    const STUDY_FAMILY_NAMES = {
      descriptive_epidemiology: ['a descriptive study', '描述性研究'],
      association_study: ['an association study', '关联研究'],
      prediction_model: ['a prediction model', '预测模型'],
      survival: ['a survival analysis (time to event)', '生存分析（时间到事件）'],
      causal_inference: ['a causal effect estimate', '因果效应估计'],
      trajectory_clustering: ['subtype or trajectory clustering', '亚型或轨迹聚类'],
    };

    function studyFamilyGapText(gap) {
      const value = gap || {};
      // The question reads as two families; the researcher chooses in the
      // conversation (host: question_family_check).
      if (value.code === 'study_family_reading_disagreement') {
        const name = key => STUDY_FAMILY_NAMES[key] || ['another study type', '另一种研究'];
        const first = name(value.model_family);
        const second = name(value.host_family);
        return {
          refusal: tr(`Choose the study type first: ${first[0]} or ${second[0]}`, `请先选研究类型：${first[1]}，或${second[1]}`),
          instead: '',
        };
      }
      const family = {
        survival: ['survival (time-to-event) analysis', '生存分析（时间到事件）'],
        trajectory_clustering: ['subtype or trajectory clustering', '亚型或轨迹聚类分析'],
      }[value.family] || ['this kind of study', '这类研究'];
      const databases = { eicu: 'eICU', miiv: 'MIMIC-IV' };
      const database = databases[value.database] || tr('this database', '这个数据库');
      const offered = (Array.isArray(value.supported_databases) ? value.supported_databases : ['miiv'])
        .map(key => databases[key]).filter(Boolean).join(tr(' or ', '或')) || 'MIMIC-IV';
      return {
        refusal: tr(`This version does not offer ${family[0]} on ${database}`, `本版本在 ${database} 上不支持${family[1]}`),
        instead: tr(`Ask it on ${offered}, or ask with another study design.`, `可以改用 ${offered} 数据，或换一种研究设计后再问。`),
      };
    }

    function runFailureText(code, detail) {
      const value = String(code || '').trim();
      if (!value) return '';
      const resumeFailure = developmentResumeFailureText(value);
      if (resumeFailure) return resumeFailure;
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
        // Only follow-up outcomes have this stop (outcome_availability:
        // mortality at a fixed horizon, follow-up days, ICU- or
        // ventilator-free days, ICU readmission).
        research_pipeline_required_concept_structurally_unavailable: tr(
          'The outcome the question asks for (such as death within 28 or 90 days, ICU-free days or ICU readmission) is not recorded in this database: it needs follow-up this database does not have. No analysis was run. Ask about in-hospital death instead, or use a database that records this outcome.',
          '问题要求的结局（如 28 天或 90 天死亡、无 ICU 天数、再入 ICU）在这个数据库里没有记录：它需要该库没有的随访信息。没有运行分析。可以改问院内死亡，或改用记录了这个结局的数据库。',
        ),
        research_pipeline_required_concept_without_values: tr(
          'A variable the study requires is listed by its data source, but the source holds no value of it, so planning stopped before any model was called. Change the variable, or use a source that records it, then generate the plan again.',
          '研究需要的某个变量，数据源里有这一列，但一个值都没有，所以在调用模型之前就停止了规划。请更换这个变量，或改用记录了它的数据源，再重新生成计划。',
        ),
        research_pipeline_plan_contract_exhausted: tr(
          'Planning exhausted its allowed attempts without a valid plan. Analysis did not start.',
          '规划在允许的尝试次数内未能生成有效计划，分析没有开始。',
        ),
        research_pipeline_planner_provider_unavailable: tr(
          'The model service was unavailable while planning, and no analysis started. Generate the plan again in a moment.',
          '规划期间模型服务暂时不可用，没有开始任何分析。请稍后重新生成计划。',
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
        research_pipeline_manifest_required: tr(
          'The bound folder is a converted database, not a data package prepared for this study. Extract this study\'s data first, then generate the plan.',
          '绑定的是转换好的数据库，还不是为本研究准备的数据包。请先为本研究提取数据，再生成计划。',
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
        research_pipeline_scientific_authority_refused: tr(
          'The plan does not match the analysis design EasyICU set up for this study (for example its 24-hour landmark, the coding of the exposure, or the set of models), so EasyICU stopped before any analysis ran. Generate the plan again; if it stops here again, keep the run record for review.',
          '计划与 EasyICU 为这项研究设定的分析设计不一致（例如 24 小时 landmark、暴露的编码方式或模型组合），因此在任何分析运行前停止。请重新生成计划；如果再次停在这里，请保留运行记录供排查。',
        ),
        research_pipeline_execution_failed: tr(
          'An analysis step failed while running.',
          '有一个分析步骤在运行中失败了。',
        ),
        research_pipeline_provider_budget_exhausted: tr(
          'The run used up its model budget before it finished, and repeated connection failures count against that budget; try again once the connection is stable.',
          '本次运行在完成前用完了模型预算，反复连接失败也会计入预算，请在连接稳定后重试。',
        ),
        research_pipeline_provider_recovery_exhausted: tr(
          'Automatic provider recovery reached its limit; the verified checkpoint is preserved.',
          '模型服务自动恢复已达到上限，已保留验证通过的检查点。',
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

    return Object.freeze({ errorText, jobErrorText, modelErrorText, providerPreset, option, runFailureText, studyFamilyGapText });
  }

  window.EasyICU.guidedPi.declare('errorText', { create, scanErrorText });
})();
