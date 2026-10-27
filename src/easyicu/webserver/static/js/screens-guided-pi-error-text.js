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
      if (/^result_append_(source_changed|execution_failed|unavailable)$/.test(error.code || '')) return runFailureText(error.code);
      if (error.code === 'study_stated_variable_unrecorded') return runFailureText(error.code);
      if (error.code === 'job_interrupted_by_server_restart') return tr('The server restarted before this task finished. Review its saved state before continuing.', '服务器在任务完成前重启了，请核对保存的状态后再继续。');
      if (error.code === 'pi_followup_queue_full') return tr('Up to four follow-ups can wait. Remove one or wait for the current reply.', '最多排队四条消息，请移除一条或等待当前回复。');
      if (error.code === 'pi_followup_action_requires_idle' || error.code === 'pi_followup_plain_message_required') return tr('Queue a question or comment here. Send approvals, action requests, and source changes after the reply finishes.', '这里可排队提问或补充说明；请在当前回复结束后发送批准、执行请求或数据源变更。');
      if (error.code === 'pi_followup_store_unavailable') return tr('The saved message queue could not be read. Wait for the current reply before sending.', '无法读取已保存的消息队列，请等待当前回复结束后再发送。');
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
      if (error.code === 'agentic_coder_credentialed_source_forbidden') {
        return tr('The desktop app does not run model-written code on credentialed data such as MIMIC-IV or eICU. No analysis was run.', '桌面版不会在 MIMIC-IV、eICU 等需授权的数据上运行模型临时写的代码。尚未运行任何分析。');
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
      // The one test message sent before saving (provider_config.verify_provider_connection).
      if (error.code === 'pi_provider_inference_verification_failed') {
        const status = Number(error.details && error.details.status_code);
        const known = Number.isFinite(status) && status > 0;
        if (known && (status >= 500 || status === 429)) {
          return tr(`The model service is temporarily unavailable (status ${status}). Try again in a moment.`, `模型服务暂时不可用（状态码 ${status}），请稍后重试。`);
        }
        const suffix = known ? tr(` (status ${status})`, `（状态码 ${status}）`) : '';
        return tr('The selected model did not answer a test message', '所选模型没有回应测试消息') + suffix
          + tr('. Check the model name or choose another model.', '，请检查模型名或换一个模型。');
      }
      if (error.code === 'pi_provider_inference_response_invalid') {
        return tr('The model replied in a format EasyICU cannot read. Check the protocol under Advanced settings.', '模型的回复格式无法读取，请在高级设置中检查兼容协议。');
      }
      if (error.code === 'pi_provider_response_too_large') {
        return tr('The model service sent a reply too large to check. Check the address.', '模型服务返回的内容过大，无法核对，请检查服务地址。');
      }
      if (error.code === 'pi_provider_models_response_invalid') {
        return tr('This service did not return a model list EasyICU can read. Check the protocol under Advanced settings.', '该服务返回的模型列表无法读取，请在高级设置中检查兼容协议。');
      }
      if (error.code === 'pi_provider_model_list_unavailable') {
        return tr('This service does not list its models. Enter the model name below.', '该服务不提供模型列表，请在下面填写模型名。');
      }
      if (error.code === 'openrouter_login_not_started' || error.code === 'openrouter_login_not_signed_in') {
        return tr('The OpenRouter sign-in expired. Sign in again.', 'OpenRouter 登录已过期，请重新登录。');
      }
      if (error.code === 'openrouter_login_denied') {
        return tr('OpenRouter did not complete the sign-in.', 'OpenRouter 没有完成登录。');
      }
      if (error.code === 'openrouter_login_exchange_failed' || error.code === 'openrouter_login_unreachable') {
        return tr('OpenRouter did not issue a key for this sign-in. Try again, or paste an API key.', 'OpenRouter 没有为这次登录签发密钥。请重试，或直接粘贴 API 凭据。');
      }
      if (error.code === 'openrouter_login_wrong_service' || error.code === 'openrouter_auth_url_invalid') {
        return tr('An OpenRouter sign-in can only connect OpenRouter.', 'OpenRouter 登录只能用于连接 OpenRouter。');
      }
      if (error.code === 'pi_provider_base_url_rejected') {
        return tr('This computer’s security policy does not allow this service address. Use another address.', '本机安全策略不允许使用这个服务地址，请换一个地址。');
      }
      if (error.code === 'pi_provider_api_transport_unsupported') {
        return tr('Choose a supported protocol under Advanced settings.', '请在高级设置中选择一个支持的兼容协议。');
      }
      if (error.code === 'external_llm_opt_in_required') {
        // Connecting a model turns this on (settings.py, one opt-in); it is
        // off only when the Settings switch was turned off.
        return tr('Outbound model calls are off. Turn on “Allow outbound model calls” in Settings, or connect the model again.', '外部模型调用已关闭。请在设置中打开「允许外部模型调用」，或重新连接模型。');
      }
      if (error.code === 'pi_session_busy') {
        return tr('Wait for the current reply or study run to finish, then try again.', '请等当前回复或研究运行结束后再试。');
      }
      if (error.code === 'pi_session_pinned') {
        return tr('Unpin this conversation from the presentation before deleting it.', '这个对话已固定用于演示，取消固定后才能删除。');
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
      if (error.code === 'research_pipeline_plan_step_needs_sandbox' || error.code === 'revision_added_unrequested_analysis') {
        return runFailureText(error.code);
      }
      if (error.code === 'requested_concept_no_data') {
        return tr('The data contain no recorded values for a variable the question asks to summarize, so this part of the result cannot be produced.', '问题要求汇总的变量在当前数据中没有任何记录值，这部分结果无法给出。');
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
        const gap = studyFamilyGapText({ family: 'causal_inference', ...((error.details && error.details.study_family_gap) || {}) });
        return tr(
          `${gap.refusal}, so analysis did not start. ${gap.instead}`,
          `${gap.refusal}，分析没有开始。${gap.instead}`,
        );
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
      if (hostnameMatches(host, 'dashscope.aliyuncs.com') || /\.maas\.aliyuncs\.com$/.test(host)) return 'qwen';
      if (hostnameMatches(host, 'open.bigmodel.cn')) return 'zhipu';
      if (hostnameMatches(host, 'api.moonshot.cn')) return 'moonshot';
      if (/^ark\.[a-z0-9-]+\.volces\.com$/.test(host)) return 'doubao';
      if (hostnameMatches(host, 'api.siliconflow.cn')) return 'siliconflow';
      if (hostnameMatches(host, 'api.minimaxi.com')) return 'minimax';
      if (hostnameMatches(host, 'openai.azure.com') || hostnameMatches(host, 'services.ai.azure.com')) return 'azure';
      if (/^bedrock-(mantle\.[a-z0-9-]+\.api\.aws|runtime\.[a-z0-9-]+\.amazonaws\.com)$/.test(host)) return 'bedrock';
      // Local services are loopback-only: exact host plus their pinned port,
      // so `https://127.0.0.1.evil.example:8317/` cannot borrow a preset.
      const LOOPBACK_PRESETS = { 8317: 'cliproxyapi', 11434: 'ollama', 1234: 'lmstudio' };
      if ((host === '127.0.0.1' || host === 'localhost') && LOOPBACK_PRESETS[port]) return LOOPBACK_PRESETS[port];
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
      const statedVariable = String(detail && detail.statedVariable || '').trim();
      if (code === 'progressive_context_bindings_invalid') {
        const step = String(detail && detail.step || '').trim();
        const title = String(detail && detail.stepTitle || '').trim()
          || (step === 'adjusted_association' ? tr('adjusted association analysis', '调整后的关联分析') : step);
        const where = title ? tr(`The “${title}” step`, `「${title}」步骤`) : tr('A candidate plan step', '候选计划中的一个步骤');
        return tr(`${where} could not be matched to columns in the prepared data. Planning stopped; analysis has not started. Generate the plan again; keep the failure record if it stops again.`,
          `${where}无法对应到准备好的数据中的字段。规划已停止，分析尚未开始。请重新生成计划；如再次停止，请保留失败记录供排查。`);
      }
      const baselineDesignMismatch = code === 'baseline_association_design_mismatch'
        || (code === 'progressive_scientific_authority_repairable' && cause === 'baseline_association_design_mismatch');
      if (baselineDesignMismatch && Array.isArray(detail.mismatchFields)) {
        const labels = {
          dependence: tr('handling of repeated ICU stays within a patient', '患者内多次 ICU 入住的相关性处理'),
          exposure_source: tr('exposure', '暴露'),
          outcome: tr('outcome', '结局'),
          outcome_type: tr('outcome type', '结局类型'),
          exposure_levels: tr('exposure groups', '暴露分组'),
          exposure_reference_level: tr('reference group', '参照组'),
          primary_contrast_level: tr('comparison group', '比较组'),
          exposure_encoding: tr('exposure coding', '暴露编码'),
        };
        const fields = Object.keys(labels).filter(field => detail.mismatchFields.includes(field));
        if (fields.length) {
          const names = fields.map(field => labels[field]).join(tr(', ', '、'));
          return tr(`The plan's ${names} differs from the confirmed design. Planning stopped; analysis has not started. Generate the plan again; keep the failure record if it stops again.`,
            `计划中的「${names}」与已确认的设计不一致。规划已停止，分析尚未开始。请重新生成计划；如再次停止，请保留失败记录供排查。`);
        }
      }
      if (code === 'progressive_family_spec_stated_adjustment_unavailable' && statedVariable) {
        const why = cause === 'stated_adjustment_name_unresolved'
          ? tr(`EasyICU could not identify the stated adjustment “${statedVariable}”. Give its exact name, or remove it from the question.`,
            `EasyICU 无法识别问题中的调整变量「${statedVariable}」。请写明它的准确名称，或在问题中去掉它。`)
          : cause === 'stated_adjustment_column_ambiguous'
            ? tr(`The adjustment “${statedVariable}” matches several columns. State which value and time window to use, or remove it from the question.`,
              `调整变量「${statedVariable}」在数据中对应多列。请写明使用哪个数值和时间窗，或在问题中去掉它。`)
            : tr(`This data has no usable column for the adjustment “${statedVariable}”. Use data that includes it, or remove it from the question.`,
              `这份数据里没有可用的调整变量「${statedVariable}」。请改用包含它的数据，或在问题中去掉它。`);
        return why + tr(' Planning stopped and no analysis was run. Generate the plan again after that change.',
          ' 规划已停止，没有运行分析。修改后请重新生成计划。');
      }
      if (code === 'progressive_family_spec_exact_roster_mismatch' && statedVariable
          && cause === 'stated_adjustment_outside_exact_roster') {
        return tr(`The study's fixed adjustment variables leave out “${statedVariable}”, which the question asks to adjust for. Add it to the adjustment variables, or remove it from the question. Planning stopped and no analysis was run. Generate the plan again after that change.`,
          `研究设置中固定的调整变量没有包含问题要求调整的「${statedVariable}」。请把它加入调整变量，或在问题中去掉它。规划已停止，没有运行分析。修改后请重新生成计划。`);
      }
      const missing = Array.isArray(detail && detail.missing) ? detail.missing.map(String).filter(Boolean) : [];
      const list = missing.length ? missing.join(', ') : '';
      if (code === 'execution_step_raised') {
        const step = String(detail && detail.step || '').trim();
        const name = {
          cohort_definition: tr('cohort selection', '研究人群筛选'),
          table_one: tr('baseline characteristics', '基线特征汇总'),
          measurement_audit: tr('data completeness checks', '数据完整性检查'),
          adjusted_association: tr('adjusted association analysis', '调整后的关联分析'),
          absolute_risk_context: tr('absolute risk comparison', '绝对风险比较'),
          visualization: tr('result figures', '结果绘图'),
          report: tr('report preparation', '报告生成'),
        }[step] || String(detail && detail.stepTitle || '').trim() || step || tr('analysis', '分析');
        return tr(`The “${name}” step stopped on a calculation error. Send the failure receipt to support, then retry the approved plan once it is fixed.`,
          `「${name}」步骤计算出错，已停止。请把失败回执发给支持人员，修复后按已批准的计划重试。`);
      }
      if (code === 'deterministic_standard_executor_failed_closed') {
        const title = String(detail && detail.stepTitle || '').trim();
        const step = title || String(detail && detail.step || '').trim();
        const why = {
          executor_runtime_failure: tr('the built-in calculation failed', '内置计算执行失败'),
          executor_output_contract_failed: tr('its output did not pass validation', '输出未通过校验'),
          missing_executor_outputs: tr('it did not produce the required output', '未生成必需的输出'),
        }[cause] || tr('the built-in calculation stopped', '内置计算停止');
        return tr(
          `${step ? `Step “${step}” stopped` : 'An analysis step stopped'}: ${why}. The run is incomplete. Send the failure receipt to support; retry the approved plan after EasyICU is updated.`,
          `${step ? `步骤「${step}」停止` : '有分析步骤停止'}：${why}。本次运行未完成。请把失败回执发给支持人员，EasyICU 更新后再按已批准计划重试。`,
        );
      }
      const detailCopy = {
        progressive_action_module_mismatch: tr("The candidate plan assigns a scientific method to an analysis step that cannot execute it, so planning stopped and no analysis was run. Generate the plan again.", "候选计划把科学方法分配给了无法执行它的分析步骤，因此规划已停止，没有运行分析。请重新生成计划。"),
        progressive_outline_action_module_mismatch: tr("The candidate plan assigns a scientific method to an analysis step that cannot execute it, so planning stopped and no analysis was run. Generate the plan again.", "候选计划把科学方法分配给了无法执行它的分析步骤，因此规划已停止，没有运行分析。请重新生成计划。"),
        progressive_outline_primary_module_family_mismatch: tr("The candidate primary analysis method does not apply to the selected study type, so planning stopped and no analysis was run. Generate the plan again.", "候选主要分析方法不适用于所选研究类型，因此规划已停止，没有运行分析。请重新生成计划。"),
        progressive_primary_module_family_mismatch: tr("The candidate primary analysis method does not apply to the selected study type, so planning stopped and no analysis was run. Generate the plan again.", "候选主要分析方法不适用于所选研究类型，因此规划已停止，没有运行分析。请重新生成计划。"),
        progressive_phenotype_comparison_levels_unavailable: tr("The candidate cluster comparison needs category levels not established in the data description, so planning stopped and no analysis was run. Generate the plan again.", "候选聚类比较需要的分类水平尚未在数据说明中确定，因此规划已停止，没有运行分析。请重新生成计划。"),
        progressive_phenotype_comparison_outcome_missing: tr("The candidate cluster comparison has not included a declared study outcome, so planning stopped and no analysis was run. Generate the plan again.", "候选聚类比较没有包含已声明的研究结局，因此规划已停止，没有运行分析。请重新生成计划。"),
        progressive_phenotype_comparison_roster_missing: tr("The candidate cluster comparison has not declared which variables to summarize, so planning stopped and no analysis was run. Generate the plan again.", "候选聚类比较没有声明要汇总的变量，因此规划已停止，没有运行分析。请重新生成计划。"),
        progressive_phenotype_comparison_spec_invalid: tr("The candidate cluster comparison does not meet its execution requirements, so planning stopped and no analysis was run. Generate the plan again.", "候选聚类比较不满足执行要求，因此规划已停止，没有运行分析。请重新生成计划。"),
        progressive_phenotype_comparison_summary_incompatible: tr("A summary in the candidate cluster comparison is incompatible with the variable type, so planning stopped and no analysis was run. Generate the plan again.", "候选聚类比较中的汇总方式与变量类型不兼容，因此规划已停止，没有运行分析。请重新生成计划。"),
        progressive_phenotyping_feature_input_mismatch: tr("A fitting feature in the candidate clustering plan is absent from its declared inputs, so planning stopped and no analysis was run. Generate the plan again.", "候选聚类计划的某个拟合特征不在已声明输入中，因此规划已停止，没有运行分析。请重新生成计划。"),
        progressive_phenotyping_feature_role_forbidden: tr("The candidate clustering plan uses an outcome or an identifier as a fitting feature, so planning stopped and no analysis was run. Generate the plan again.", "候选聚类计划把结局或标识字段用作了拟合特征，因此规划已停止，没有运行分析。请重新生成计划。"),
        progressive_phenotyping_feature_roster_invalid: tr("The candidate clustering plan does not declare at least two distinct fitting features, so planning stopped and no analysis was run. Generate the plan again.", "候选聚类计划没有声明至少两个不同的拟合特征，因此规划已停止，没有运行分析。请重新生成计划。"),
        progressive_phenotyping_feature_roster_missing: tr("The candidate clustering plan has not separated fitting features from descriptions and outcomes, so planning stopped and no analysis was run. Generate the plan again.", "候选聚类计划没有区分拟合特征与描述变量和结局，因此规划已停止，没有运行分析。请重新生成计划。"),
        progressive_adjusted_model_contract_invalid: tr("The candidate plan's adjusted model does not meet its execution requirements, so planning stopped and no analysis was run. Generate the plan again.", "候选计划中的调整模型不满足执行要求，因此规划已停止，没有运行分析。请重新生成计划。"),
        progressive_adjustment_roster_mismatch: tr("The candidate model adjusts for different variables than this study's adjustment set, so planning stopped and no analysis was run. Generate the plan again.", "候选模型的调整变量与本研究确定的调整变量不一致，因此规划已停止，没有运行分析。请重新生成计划。"),
        progressive_adjusted_model_uses_navigation_coordinate: tr("The candidate model uses a row identifier or navigation field as a statistical variable, so planning stopped and no analysis was run. Generate the plan again.", "候选模型把行标识或导航字段用作了统计变量，因此规划已停止，没有运行分析。请重新生成计划。"),
        progressive_analysis_module_unavailable: tr("The candidate plan used an analysis step that EasyICU cannot run for this kind of study, so planning stopped and no analysis was run. Generate the plan again.", "候选计划中有一个分析步骤不适用于这类研究，因此规划已停止，没有运行分析。请重新生成计划。"),
        progressive_analysis_plan_invalid: tr("The assembled candidate plan has inconsistent analysis requirements, so planning stopped and no analysis was run. Generate the plan again.", "汇总后的候选计划存在不一致的分析要求，因此规划已停止，没有运行分析。请重新生成计划。"),
        progressive_analysis_step_invalid: tr("An analysis step in the candidate plan does not meet its execution requirements, so planning stopped and no analysis was run. Generate the plan again.", "候选计划中的一个分析步骤不满足执行要求，因此规划已停止，没有运行分析。请重新生成计划。"),
        progressive_analysis_type_unauthorized: tr("The candidate plan selected a study type outside the study's confirmed scope, so planning stopped and no analysis was run. Generate the plan again.", "候选计划选择的研究类型超出了已确认的研究范围，因此规划已停止，没有运行分析。请重新生成计划。"),
        progressive_association_sensitivity_contract_invalid: tr("A proposed association sensitivity analysis does not meet its execution requirements, so planning stopped and no analysis was run. Generate the plan again.", "拟定的关联敏感性分析不满足执行要求，因此规划已停止，没有运行分析。请重新生成计划。"),
        progressive_association_sensitivity_parent_invalid: tr("A proposed sensitivity analysis cannot be tied to the primary association result, so planning stopped and no analysis was run. Generate the plan again.", "拟定的敏感性分析无法对应到主要关联结果，因此规划已停止，没有运行分析。请重新生成计划。"),
        progressive_baseline_summary_semantic_role_ineligible: tr("The candidate baseline summary uses identifiers or measurement metadata as patient characteristics, so planning stopped and no analysis was run. Generate the plan again.", "候选基线汇总把标识字段或测量元数据用作了患者特征，因此规划已停止，没有运行分析。请重新生成计划。"),
        progressive_cohort_predicate_on_identity: tr("The candidate plan uses an identifier as a clinical inclusion or exclusion criterion, so planning stopped and no analysis was run. Generate the plan again.", "候选计划把标识字段用作了临床纳入或排除标准，因此规划已停止，没有运行分析。请重新生成计划。"),
        progressive_compile_batch_invalid: tr("Several steps in the candidate plan failed their execution checks, so planning stopped and no analysis was run. Generate the plan again.", "候选计划中有多个步骤未通过执行检查，因此规划已停止，没有运行分析。请重新生成计划。"),
        progressive_complete_case_includes_not_applicable_time: tr("The candidate complete-case analysis treats an inapplicable time value as a missing observation, so planning stopped and no analysis was run. Generate the plan again.", "候选完整病例分析把不适用的时间值当作了缺失观测，因此规划已停止，没有运行分析。请重新生成计划。"),
        progressive_conflicting_output_role: tr("The candidate plan assigns conflicting purposes to the same result, so planning stopped and no analysis was run. Generate the plan again.", "候选计划给同一份结果指定了相互冲突的用途，因此规划已停止，没有运行分析。请重新生成计划。"),
        progressive_covariate_semantic_role_ineligible: tr("The candidate model uses an outcome, identifier or metadata field as an adjustment variable, so planning stopped and no analysis was run. Generate the plan again.", "候选模型把结局、标识或元数据字段用作了调整变量，因此规划已停止，没有运行分析。请重新生成计划。"),
        progressive_covariate_time_zero_authority_missing: tr("The candidate plan has not established that an adjustment variable is known by the analysis start, so planning stopped and no analysis was run. Generate the plan again.", "候选计划未能确认某个调整变量在分析起点前已知，因此规划已停止，没有运行分析。请重新生成计划。"),
        progressive_descriptive_robustness_unavailable: tr("The candidate descriptive plan requests a robustness analysis that requires a fitted effect estimate, so planning stopped and no analysis was run. Generate the plan again.", "候选描述性计划要求运行依赖已拟合效应估计的稳健性分析，因此规划已停止，没有运行分析。请重新生成计划。"),
        progressive_design_input_structurally_unavailable: tr("The selected design requires a variable the chosen data source cannot supply, so planning stopped and no analysis was run. Generate the plan again.", "所选设计需要一个当前数据源无法提供的变量，因此规划已停止，没有运行分析。请重新生成计划。"),
        progressive_distribution_contrast_not_distinct: tr("The candidate distribution comparison uses the same group as both reference and comparison, so planning stopped and no analysis was run. Generate the plan again.", "候选分布比较把同一组同时用作参照组和比较组，因此规划已停止，没有运行分析。请重新生成计划。"),
        progressive_distribution_counts_only_interval_conflict: tr("The candidate plan requests uncertainty intervals for a description restricted to counts, so planning stopped and no analysis was run. Generate the plan again.", "候选计划为仅允许报告计数的描述要求了不确定性区间，因此规划已停止，没有运行分析。请重新生成计划。"),
        progressive_distribution_levels_unavailable: tr("The candidate distribution comparison needs category levels that the data description has not established, so planning stopped and no analysis was run. Generate the plan again.", "候选分布比较需要的分类水平尚未在数据说明中确定，因此规划已停止，没有运行分析。请重新生成计划。"),
        progressive_distribution_missingness_authority_invalid: tr("The proposed distribution summary uses a missing-data policy unsupported by the data description, so planning stopped and no analysis was run. Generate the plan again.", "拟定的分布汇总使用了数据说明不支持的缺失处理方式，因此规划已停止，没有运行分析。请重新生成计划。"),
        progressive_distribution_spec_invalid: tr("The candidate distribution summary does not meet its execution requirements, so planning stopped and no analysis was run. Generate the plan again.", "候选分布汇总不满足执行要求，因此规划已停止，没有运行分析。请重新生成计划。"),
        progressive_figure_product_kind: tr("A proposed figure step declares an output that is not a figure, so planning stopped and no analysis was run. Generate the plan again.", "拟定的绘图步骤声明的输出并非图形，因此规划已停止，没有运行分析。请重新生成计划。"),
        progressive_foundation_cohort_invalid: tr("The candidate plan's cohort definition does not meet the study requirements, so planning stopped and no analysis was run. Generate the plan again.", "候选计划的队列定义不满足研究要求，因此规划已停止，没有运行分析。请重新生成计划。"),
        progressive_foundation_robustness_intent_missing: tr("The candidate plan has not declared the robustness choices its steps require, so planning stopped and no analysis was run. Generate the plan again.", "候选计划没有声明其步骤所需的稳健性分析选择，因此规划已停止，没有运行分析。请重新生成计划。"),
        progressive_functional_form_authority_mismatch: tr("The candidate sensitivity analysis changes the confirmed functional-form comparison, so planning stopped and no analysis was run. Generate the plan again.", "候选敏感性分析改变了已确认的函数形式比较，因此规划已停止，没有运行分析。请重新生成计划。"),
        progressive_functional_form_family_mismatch: tr("The candidate functional-form sensitivity analysis does not apply to this study type, so planning stopped and no analysis was run. Generate the plan again.", "候选函数形式敏感性分析不适用于这类研究，因此规划已停止，没有运行分析。请重新生成计划。"),
        progressive_functional_form_method_unsupported: tr("The candidate functional-form sensitivity analysis selected a method EasyICU cannot execute, so planning stopped and no analysis was run. Generate the plan again.", "候选函数形式敏感性分析选择了 EasyICU 无法执行的方法，因此规划已停止，没有运行分析。请重新生成计划。"),
        progressive_functional_form_target_invalid: tr("The candidate functional-form comparison selected an unsuitable variable, so planning stopped and no analysis was run. Generate the plan again.", "候选函数形式比较选择了不适用的变量，因此规划已停止，没有运行分析。请重新生成计划。"),
        progressive_functional_form_target_missing: tr("The candidate functional-form comparison has not identified its target variable, so planning stopped and no analysis was run. Generate the plan again.", "候选函数形式比较没有指定目标变量，因此规划已停止，没有运行分析。请重新生成计划。"),
        progressive_host_reporting_source_invalid: tr("The candidate plan's reporting-method source does not meet the evidence requirements, so planning stopped and no analysis was run. Generate the plan again.", "候选计划的报告方法来源不满足证据要求，因此规划已停止，没有运行分析。请重新生成计划。"),
        progressive_host_reporting_source_unavailable: tr("The reporting-method source required by the candidate plan is unavailable, so planning stopped and no analysis was run. Generate the plan again.", "候选计划所需的报告方法来源不可用，因此规划已停止，没有运行分析。请重新生成计划。"),
        progressive_identity_column_ambiguous: tr("The candidate plan cannot identify exactly one column for its analysis rows, so planning stopped and no analysis was run. Generate the plan again.", "候选计划无法确定唯一的分析行标识列，因此规划已停止，没有运行分析。请重新生成计划。"),
        progressive_know_how_authority_absent: tr("The candidate plan cites a protocol decision without its approved source, so planning stopped and no analysis was run. Generate the plan again.", "候选计划引用的方案决策缺少已批准的依据，因此规划已停止，没有运行分析。请重新生成计划。"),
        progressive_level_index_out_of_range: tr("A category selected by the candidate plan is outside the declared category list, so planning stopped and no analysis was run. Generate the plan again.", "候选计划选择的类别不在已声明的类别列表中，因此规划已停止，没有运行分析。请重新生成计划。"),
        progressive_literature_merge_invalid: tr("The candidate plan's combined literature bindings are inconsistent, so planning stopped and no analysis was run. Generate the plan again.", "候选计划合并后的文献依据不一致，因此规划已停止，没有运行分析。请重新生成计划。"),
        progressive_literature_merge_overflow: tr("The candidate plan requires more literature bindings than its supported limit, so planning stopped and no analysis was run. Generate the plan again.", "候选计划所需的文献依据数量超出了支持上限，因此规划已停止，没有运行分析。请重新生成计划。"),
        progressive_locked_prefix_changed: tr("A plan repair changed an earlier step that had already been accepted, so planning stopped and no analysis was run. Generate the plan again.", "计划修复改动了前面已经接受的步骤，因此规划已停止，没有运行分析。请重新生成计划。"),
        progressive_locked_prefix_length_invalid: tr("A plan repair removed steps from the already accepted part of the plan, so planning stopped and no analysis was run. Generate the plan again.", "计划修复删除了前面已经接受的部分中的步骤，因此规划已停止，没有运行分析。请重新生成计划。"),
        progressive_measurement_audit_spec_invalid: tr("The proposed measurement audit does not meet its execution requirements, so planning stopped and no analysis was run. Generate the plan again.", "拟定的测量审计不满足执行要求，因此规划已停止，没有运行分析。请重新生成计划。"),
        progressive_measurement_product_kind: tr("The proposed measurement audit declares an output in an unsupported form, so planning stopped and no analysis was run. Generate the plan again.", "拟定的测量审计声明了不受支持的输出形式，因此规划已停止，没有运行分析。请重新生成计划。"),
        progressive_model_continuous_coding_declared_domain: tr("The candidate model treats a declared categorical variable as continuous, so planning stopped and no analysis was run. Generate the plan again.", "候选模型把已声明的分类变量当作了连续变量，因此规划已停止，没有运行分析。请重新生成计划。"),
        progressive_model_levels_unavailable: tr("The candidate model needs category levels that the data description has not established, so planning stopped and no analysis was run. Generate the plan again.", "候选模型需要的分类水平尚未在数据说明中确定，因此规划已停止，没有运行分析。请重新生成计划。"),
        progressive_model_outcome_is_covariate: tr("The candidate model also uses its outcome as an adjustment variable, so planning stopped and no analysis was run. Generate the plan again.", "候选模型同时把结局用作了调整变量，因此规划已停止，没有运行分析。请重新生成计划。"),
        progressive_model_term_duplicate: tr("The candidate model lists the same variable more than once, so planning stopped and no analysis was run. Generate the plan again.", "候选模型重复列出了同一个变量，因此规划已停止，没有运行分析。请重新生成计划。"),
        progressive_ordered_trend_continuous_outcome_ambiguous: tr("The candidate ordered-trend analysis does not identify a unique continuous outcome, so planning stopped and no analysis was run. Generate the plan again.", "候选有序趋势分析未确定唯一的连续结局，因此规划已停止，没有运行分析。请重新生成计划。"),
        progressive_ordered_trend_continuous_outcome_invalid: tr("The continuous outcome selected for the candidate ordered-trend analysis is unsuitable, so planning stopped and no analysis was run. Generate the plan again.", "候选有序趋势分析选择的连续结局不适用，因此规划已停止，没有运行分析。请重新生成计划。"),
        progressive_ordered_trend_risk_ratio_unsupported: tr("The candidate ordered-trend analysis tests an odds-ratio slope, so it cannot follow a risk-ratio primary model. Planning stopped and no analysis was run. Generate the plan again.", "候选有序趋势分析检验的是比值比斜率，不能接在风险比主模型之后，因此规划已停止，没有运行分析。请重新生成计划。"),
        progressive_effect_measure_unrequested: tr("The candidate plan reports a risk ratio, but the question does not ask for one; a binary outcome is reported as an odds ratio unless the question asks for a risk ratio. Planning stopped and no analysis was run. Generate the plan again, or ask for a risk ratio in the question.", "候选计划报告风险比，但问题没有要求风险比；二分类结局默认报告比值比，除非问题明确要求风险比。因此规划已停止，没有运行分析。请重新生成计划，或在问题中写明要求风险比。"),
        progressive_effect_measure_not_declared: tr("The question asks for a risk ratio, but the candidate primary model reports an odds ratio, so planning stopped and no analysis was run. Generate the plan again.", "问题要求风险比，但候选主模型报告的是比值比，因此规划已停止，没有运行分析。请重新生成计划。"),
        progressive_effect_measure_requires_binary_outcome: tr("The candidate plan asked for a risk ratio or odds ratio for an outcome that is not binary, so planning stopped and no analysis was run. Generate the plan again.", "候选计划为非二分类结局指定了风险比或比值比，因此规划已停止，没有运行分析。请重新生成计划。"),
        progressive_ordered_trend_domain_unsupported: tr("The candidate ordered-trend analysis needs an ordered category domain that is unavailable, so planning stopped and no analysis was run. Generate the plan again.", "候选有序趋势分析需要的有序分类范围不可用，因此规划已停止，没有运行分析。请重新生成计划。"),
        progressive_ordered_trend_primary_binding_invalid: tr("The candidate ordered-trend analysis cannot be tied to the primary result, so planning stopped and no analysis was run. Generate the plan again.", "候选有序趋势分析无法对应到主要结果，因此规划已停止，没有运行分析。请重新生成计划。"),
        progressive_ordered_trend_primary_columns_missing: tr("The candidate ordered-trend analysis lacks columns required by its primary result, so planning stopped and no analysis was run. Generate the plan again.", "候选有序趋势分析缺少其主要结果所需的列，因此规划已停止，没有运行分析。请重新生成计划。"),
        progressive_ordered_trend_product_invalid: tr("The candidate ordered-trend analysis declares an unsupported result, so planning stopped and no analysis was run. Generate the plan again.", "候选有序趋势分析声明了不受支持的结果，因此规划已停止，没有运行分析。请重新生成计划。"),
        progressive_ordered_trend_shape_invalid: tr("The candidate ordered-trend analysis does not meet its execution requirements, so planning stopped and no analysis was run. Generate the plan again.", "候选有序趋势分析不满足执行要求，因此规划已停止，没有运行分析。请重新生成计划。"),
        progressive_outline_module_unavailable: tr("The candidate outline selected an analysis step that EasyICU cannot run for this kind of study, so planning stopped and no analysis was run. Generate the plan again.", "候选大纲选择了一个不适用于这类研究的分析步骤，因此规划已停止，没有运行分析。请重新生成计划。"),
        progressive_output_role_mismatch: tr("A candidate step assigns an output a purpose its method cannot provide, so planning stopped and no analysis was run. Generate the plan again.", "候选步骤给输出指定了该方法无法提供的用途，因此规划已停止，没有运行分析。请重新生成计划。"),
        progressive_planner_covariate_baseline_authority_missing: tr("The candidate model has not established baseline timing for its adjustment variables, so planning stopped and no analysis was run. Generate the plan again.", "候选模型未能确认调整变量的基线时间依据，因此规划已停止，没有运行分析。请重新生成计划。"),
        progressive_population_concept_unavailable: tr("The candidate population definition requires a clinical concept unavailable in the data, so planning stopped and no analysis was run. Generate the plan again.", "候选人群定义需要一个数据中不可用的临床概念，因此规划已停止，没有运行分析。请重新生成计划。"),
        progressive_population_criterion_unapplied: tr("The candidate plan leaves a stated population criterion unapplied, so planning stopped and no analysis was run. Generate the plan again.", "候选计划没有落实一项已声明的人群条件，因此规划已停止，没有运行分析。请重新生成计划。"),
        progressive_population_quote_not_verbatim: tr("The candidate population definition does not preserve the study's stated criterion, so planning stopped and no analysis was run. Generate the plan again.", "候选人群定义未保留研究原文中的条件，因此规划已停止，没有运行分析。请重新生成计划。"),
        progressive_population_requirement_drift: tr("The candidate plan changed the study's stated population requirements, so planning stopped and no analysis was run. Generate the plan again.", "候选计划改变了研究已声明的人群要求，因此规划已停止，没有运行分析。请重新生成计划。"),
        progressive_population_scope_conflict: tr("The candidate plan makes conflicting choices about the population to summarize, so planning stopped and no analysis was run. Generate the plan again.", "候选计划对要汇总的人群作出了相互冲突的选择，因此规划已停止，没有运行分析。请重新生成计划。"),
        progressive_population_spec_invalid: tr("The candidate population definition is incomplete or inconsistent, so planning stopped and no analysis was run. Generate the plan again.", "候选人群定义不完整或不一致，因此规划已停止，没有运行分析。请重新生成计划。"),
        progressive_population_spec_uncompilable: tr("EasyICU cannot turn the candidate population definition into executable cohort criteria, so planning stopped and no analysis was run. Generate the plan again.", "EasyICU 无法将候选人群定义转换为可执行的队列条件，因此规划已停止，没有运行分析。请重新生成计划。"),
        progressive_primary_contrast_is_reference: tr("The candidate primary comparison selects the reference group as its comparison group, so planning stopped and no analysis was run. Generate the plan again.", "候选主要比较把参照组本身选作了比较组，因此规划已停止，没有运行分析。请重新生成计划。"),
        progressive_primary_exposure_term_mismatch: tr("The candidate model's exposure term differs from the declared primary exposure, so planning stopped and no analysis was run. Generate the plan again.", "候选模型中的暴露项与已声明的主要暴露不一致，因此规划已停止，没有运行分析。请重新生成计划。"),
        progressive_primary_population_owner_missing: tr("The candidate plan has no step that defines the required primary analysis population, so planning stopped and no analysis was run. Generate the plan again.", "候选计划没有步骤定义所需的主要分析人群，因此规划已停止，没有运行分析。请重新生成计划。"),
        progressive_primary_population_variables_mismatch: tr("The candidate plan's primary population uses a different variable set from the primary analysis, so planning stopped and no analysis was run. Generate the plan again.", "候选计划的主要人群使用的变量集合与主要分析不一致，因此规划已停止，没有运行分析。请重新生成计划。"),
        progressive_product_dependency_mismatch: tr("A candidate step consumes a result without depending on the step that produces it, so planning stopped and no analysis was run. Generate the plan again.", "候选步骤使用了一份结果，却未依赖生成它的步骤，因此规划已停止，没有运行分析。请重新生成计划。"),
        progressive_product_reference_mismatch: tr("A candidate step attributes an input result to the wrong producing step, so planning stopped and no analysis was run. Generate the plan again.", "候选步骤把输入结果归到了错误的生成步骤，因此规划已停止，没有运行分析。请重新生成计划。"),
        progressive_raw_input_structurally_unavailable: tr("A candidate step requires an input that the chosen data source cannot supply, so planning stopped and no analysis was run. Generate the plan again.", "候选步骤需要一个当前数据源无法提供的输入，因此规划已停止，没有运行分析。请重新生成计划。"),
        progressive_report_product_kind: tr("A proposed report step declares an output that is not a report, so planning stopped and no analysis was run. Generate the plan again.", "拟定的报告步骤声明的输出并非报告，因此规划已停止，没有运行分析。请重新生成计划。"),
        progressive_required_binary_display_labels_missing: tr("The candidate plan is missing readable names for its binary comparison groups, so planning stopped and no analysis was run. Generate the plan again.", "候选计划缺少二分类比较组的可读名称，因此规划已停止，没有运行分析。请重新生成计划。"),
        progressive_required_reader_display_labels_missing: tr("The candidate plan is missing readable names needed to explain its results, so planning stopped and no analysis was run. Generate the plan again.", "候选计划缺少解释结果所需的可读名称，因此规划已停止，没有运行分析。请重新生成计划。"),
        progressive_required_robustness_intent_missing: tr("A candidate robustness step refers to a choice the plan has not declared, so planning stopped and no analysis was run. Generate the plan again.", "候选稳健性步骤引用了计划未声明的分析选择，因此规划已停止，没有运行分析。请重新生成计划。"),
        progressive_robustness_diagnostic_display_mismatch: tr("The proposed robustness display does not match its diagnostic analysis, so planning stopped and no analysis was run. Generate the plan again.", "拟定的稳健性展示与其诊断分析不一致，因此规划已停止，没有运行分析。请重新生成计划。"),
        progressive_robustness_intent_not_replayable: tr("A proposed robustness choice cannot be applied to the declared primary method, so planning stopped and no analysis was run. Generate the plan again.", "拟定的稳健性分析选择无法应用于已声明的主要方法，因此规划已停止，没有运行分析。请重新生成计划。"),
        progressive_robustness_replay_spec_invalid: tr("The candidate robustness analysis does not meet its execution requirements, so planning stopped and no analysis was run. Generate the plan again.", "候选稳健性分析不满足执行要求，因此规划已停止，没有运行分析。请重新生成计划。"),
        progressive_scientific_action_dependency_mismatch: tr("A candidate scientific method is missing a required result from an earlier step, so planning stopped and no analysis was run. Generate the plan again.", "候选科学方法缺少前面步骤应提供的必需结果，因此规划已停止，没有运行分析。请重新生成计划。"),
        progressive_scientific_action_inputs_mismatch: tr("The inputs declared for a candidate scientific method do not match its requirements, so planning stopped and no analysis was run. Generate the plan again.", "候选科学方法声明的输入与其要求不一致，因此规划已停止，没有运行分析。请重新生成计划。"),
        progressive_scientific_action_invalid: tr("A candidate scientific method is unavailable or does not match its declared analysis, so planning stopped and no analysis was run. Generate the plan again.", "候选科学方法不可用或与其声明的分析不匹配，因此规划已停止，没有运行分析。请重新生成计划。"),
        progressive_scientific_action_outputs_mismatch: tr("A candidate scientific method declares results that differ from its supported outputs, so planning stopped and no analysis was run. Generate the plan again.", "候选科学方法声明的结果与其支持的输出不一致，因此规划已停止，没有运行分析。请重新生成计划。"),
        progressive_study_population_binding_invalid: tr("The candidate study-population step does not meet its cohort binding requirements, so planning stopped and no analysis was run. Generate the plan again.", "候选研究人群步骤不满足队列绑定要求，因此规划已停止，没有运行分析。请重新生成计划。"),
        progressive_study_population_product_is_locked_cohort: tr("The candidate plan confuses the broader study population with the fixed primary analysis cohort, so planning stopped and no analysis was run. Generate the plan again.", "候选计划混淆了较广的研究人群与固定的主要分析队列，因此规划已停止，没有运行分析。请重新生成计划。"),
        progressive_study_population_scope_mismatch: tr("The candidate plan's study-population scope differs from its declared scope, so planning stopped and no analysis was run. Generate the plan again.", "候选计划的研究人群范围与已声明范围不一致，因此规划已停止，没有运行分析。请重新生成计划。"),
        progressive_table_one_contract_invalid: tr("The candidate baseline table does not meet its execution requirements, so planning stopped and no analysis was run. Generate the plan again.", "候选基线表不满足执行要求，因此规划已停止，没有运行分析。请重新生成计划。"),
        progressive_table_one_group_levels_unavailable: tr("The candidate baseline table needs grouping levels that the data description has not established, so planning stopped and no analysis was run. Generate the plan again.", "候选基线表需要的分组水平尚未在数据说明中确定，因此规划已停止，没有运行分析。请重新生成计划。"),
        progressive_table_one_levels_unavailable: tr("The candidate baseline table needs category levels that the data description has not established, so planning stopped and no analysis was run. Generate the plan again.", "候选基线表需要的分类水平尚未在数据说明中确定，因此规划已停止，没有运行分析。请重新生成计划。"),
        progressive_table_one_rows_missing: tr("The candidate baseline table has no variables to summarize, so planning stopped and no analysis was run. Generate the plan again.", "候选基线表没有需要汇总的变量，因此规划已停止，没有运行分析。请重新生成计划。"),
        progressive_table_one_semantic_role_ineligible: tr("The candidate baseline table uses identifiers or measurement metadata as clinical characteristics, so planning stopped and no analysis was run. Generate the plan again.", "候选基线表把标识字段或测量元数据用作了临床特征，因此规划已停止，没有运行分析。请重新生成计划。"),
        progressive_table_one_summary_incompatible: tr("A summary chosen for the candidate baseline table is incompatible with the variable's type, so planning stopped and no analysis was run. Generate the plan again.", "候选基线表选择的汇总方式与变量类型不兼容，因此规划已停止，没有运行分析。请重新生成计划。"),
        progressive_unknown_analysis_type: tr("EasyICU does not recognize the study type selected by the candidate plan, so planning stopped and no analysis was run. Generate the plan again.", "EasyICU 无法识别候选计划选择的研究类型，因此规划已停止，没有运行分析。请重新生成计划。"),
        progressive_unknown_literature_source: tr("The candidate plan cites a literature source outside the available evidence set, so planning stopped and no analysis was run. Generate the plan again.", "候选计划引用了可用证据集合之外的文献来源，因此规划已停止，没有运行分析。请重新生成计划。"),
        progressive_unknown_robustness_variable: tr("The candidate robustness analysis uses a variable outside the study's available inputs, so planning stopped and no analysis was run. Generate the plan again.", "候选稳健性分析使用了研究可用输入之外的变量，因此规划已停止，没有运行分析。请重新生成计划。"),
        progressive_unknown_variable: tr("The candidate plan uses a variable outside the study's available inputs, so planning stopped and no analysis was run. Generate the plan again.", "候选计划使用了研究可用输入之外的变量，因此规划已停止，没有运行分析。请重新生成计划。"),
        progressive_unsupported_product: tr("The candidate plan declares a result type EasyICU cannot materialize, so planning stopped and no analysis was run. Generate the plan again.", "候选计划声明了 EasyICU 无法生成的结果类型，因此规划已停止，没有运行分析。请重新生成计划。"),
        progressive_visualization_raw_input_forbidden: tr("A candidate figure reads raw variables where it must use verified analysis results, so planning stopped and no analysis was run. Generate the plan again.", "候选图形在应使用已核验分析结果的位置读取了原始变量，因此规划已停止，没有运行分析。请重新生成计划。"),
        progressive_visualization_source_budget_exceeded: tr("A candidate figure combines more result sources than its renderer supports, so planning stopped and no analysis was run. Generate the plan again.", "候选图形组合的结果来源数量超出了绘图方法支持的上限，因此规划已停止，没有运行分析。请重新生成计划。"),
        progressive_visualization_source_kind_invalid: tr("A candidate figure uses an unsupported kind of result source, so planning stopped and no analysis was run. Generate the plan again.", "候选图形使用了不受支持的结果来源类型，因此规划已停止，没有运行分析。请重新生成计划。"),
        association_adjustment_exposure_conflict: tr(
          'The requested adjustment includes the exposure itself or a variable used to form its groups. Planning stopped; analysis has not started. Specify a distinct adjustment variable, or revise the exposure grouping, then generate the plan again.',
          '要求调整的变量包含暴露本身，或用于形成暴露分组的原始变量。规划已停止，分析尚未开始。请改用与暴露不同的调整变量，或修改暴露分组，再重新生成计划。',
        ),
        baseline_association_roster_mismatch: tr(
          'The plan added, removed or reordered adjustment variables from the confirmed set. Planning stopped; analysis has not started. Check the adjustment variables in the study settings, then generate the plan again.',
          '计划中的调整变量与已确认的列表不一致，存在增加、遗漏或顺序变化。规划已停止，分析尚未开始。请核对研究设置中的调整变量，再重新生成计划。',
        ),
        baseline_association_design_mismatch: tr(
          'The plan does not match the confirmed exposure, outcome, group comparison or patient-clustering design. Planning stopped; analysis has not started. Check these study settings and generate the plan again; keep the failure record if it stops again.',
          '计划中的暴露、结局、分组比较或患者聚类设置与已确认的设计不一致。规划已停止，分析尚未开始。请核对这些研究设置后重新生成计划；如再次停止，请保留失败记录供排查。',
        ),
        baseline_association_step_roster_mismatch: tr(
          'Analysis-plan validation failed and analysis has not started: the plan EasyICU assembled has steps that differ from the confirmed association design. This is an EasyICU error. Send the failure receipt to support; after the fix, generate and review a new plan.',
          '分析计划校验未通过，分析尚未开始：EasyICU 组装的计划步骤与已确认的关联分析设计不一致，这是 EasyICU 的错误。请把失败回执发给支持人员，修复后重新生成并审阅计划。',
        ),
        pipeline_operation_raised: tr(
          'The run stopped on an internal error. The failure receipt records where it failed and the error type; send it to support and retry after the fix.',
          '运行因内部错误停止。失败回执记录了出错位置和异常类型，请发给支持人员，修复后再重试。',
        ),
        declared_exposure_level_empty: tr(
          'A declared group has no analysable stays, so the run stopped; generate a new plan that checks eligibility and preserves every group.',
          '计划中的一个分组没有可分析的住院，因此运行已停止，请重新生成计划，检查入组条件并保留所有分组。',
        ),
        landmark_spline_risk_ratio_unsupported: tr(
          'The question asks for a risk ratio, but the landmark analysis of a continuous exposure fits a logistic spline that reports odds ratios only, so EasyICU stopped before any analysis ran instead of reporting an odds ratio. Ask for an odds ratio, or group the exposure, then generate the plan again.',
          '问题要求风险比，但连续暴露的 landmark 分析使用 logistic 样条，只能报告比值比；为避免以比值比代替，EasyICU 在分析前停止。请改为比值比，或将暴露分组后重新生成计划。',
        ),
        association_model_grid_risk_ratio_unsupported: tr(
          'The study\'s prespecified model grid refits odds ratios and cannot accompany a risk-ratio primary model, so EasyICU stopped before any analysis ran instead of mixing the two. Ask for an odds ratio, then generate the plan again.',
          '本研究预设的模型网格重拟合的是比值比，不能与风险比主模型并列；为避免混用两种指标，EasyICU 在分析前停止。请改为比值比后重新生成计划。',
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
        manuscript_checks_not_passed: tr('The manuscript draft did not pass its automated checks.', '论文稿没有通过自动检查。'),
        evidence_complete_not_satisfied: tr('Some statements in the manuscript draft are not linked to a result from this analysis.', '论文稿中有部分表述没有对应到本次分析的结果。'),
        numeric_verified_not_satisfied: tr('Some numbers in the manuscript draft do not match the analysis results.', '论文稿中有部分数字与分析结果不一致。'),
        // Every analysis step finished; the Writer's model service then
        // stayed unavailable through its retries.
        writer_provider_transport_unavailable: tr('Every analysis step finished, but the model service could not be reached while the manuscript draft was being written. Try again later.', '所有分析步骤都已完成，但撰写论文稿时一直无法连接模型服务。请稍后重试。'),
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
        progressive_family_spec_stated_adjustment_unavailable: tr('The question asks to adjust for a variable that the extracted data does not provide, so planning stopped and no analysis was run. Choose data that includes it, or remove it from the adjustment variables, then generate the plan again.', '问题要求调整的某个变量在提取的数据里没有对应的列，因此规划已停止，没有运行分析。请选择包含该变量的数据，或把它从调整变量中去掉，然后重新生成计划。'),
        progressive_family_spec_exact_roster_mismatch: tr("The plan's adjustment variables differ from the ones named in the question, so planning stopped and no analysis was run. Generate the plan again.", '计划中的调整变量与问题里写明的不一致，因此规划已停止，没有运行分析。请重新生成计划。'),
        progressive_family_spec_covariate_timing_unproven: tr('An adjustment variable is not known by the landmark time, so it cannot be used to adjust this analysis. Planning stopped and no analysis was run. Give that variable a time window that ends before the landmark, or remove it, then generate the plan again.', '有一个调整变量在 landmark 时间点之前还无法确定，不能用于这项分析的调整。规划已停止，没有运行分析。请为该变量设定在 landmark 之前结束的时间窗，或将其去掉，然后重新生成计划。'),
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
      // The host compares the study's requirements with the database's
      // capability profile; the missing capabilities are the reason.
      const family = STUDY_FAMILY_NAMES[value.family] || ['this kind of study', '这类研究'];
      // Database names and missing-capability wording: owner sourceAdjustments.
      const adjustments = window.EasyICU.guidedPi.optional('sourceAdjustments');
      const nameOf = key => {
        const name = adjustments && typeof key === 'string' ? adjustments.databaseName(key) : '';
        return name && name !== key ? name : '';
      };
      const database = nameOf(value.database);
      const reasons = adjustments && Array.isArray(value.missing_capabilities)
        ? [...new Set(value.missing_capabilities.slice(0, 20).map(adjustments.missingText).filter(Boolean))].slice(0, 4) : [];
      // A demo is named only when its full database is not.
      const offeredKeys = Array.isArray(value.supported_databases) ? value.supported_databases.slice(0, 12) : null;
      const offered = offeredKeys && offeredKeys
        .filter(key => !(String(key).endsWith('_demo') && offeredKeys.includes(String(key).slice(0, -5))))
        .map(nameOf).filter(Boolean).join(tr(', ', '、'));
      // Where it is offered is a fact, not advice: the researcher may not hold that database.
      const where = offered ? tr(`This version supports it on ${offered}. `, `本版本可在 ${offered} 上做这类研究。`)
        : offeredKeys && !offeredKeys.length ? tr('No database supports it in this version. ', '本版本还没有能做这类研究的数据库。') : '';
      return {
        refusal: database
          ? tr(`${database} lacks data that ${family[0]} needs`, `${database} 缺少${family[1]}所需的数据`)
          : tr(`This database lacks data that ${family[0]} needs`, `这个数据库缺少${family[1]}所需的数据`),
        instead: `${reasons.length ? tr(`Specifically, ${reasons.join('; ')}. `, `具体是：${reasons.join('；')}。`) : ''}${where}${tr(
          'You can also ask with another study design.', '也可以换一种研究设计再问。')}`,
      };
    }

    function runFailureText(code, detail) {
      if (code === 'research_pipeline_provider_timeout' && detail
        && Number.isInteger(detail.transportRetries) && detail.transportRetries >= 0 && detail.transportRetries <= 32) {
        return tr(
          `The model service did not return a result in time. EasyICU retried ${detail.transportRetries} times, but this attempt still stopped. You can retry when the service recovers.`,
          `模型服务未及时返回结果。系统已自动重试 ${detail.transportRetries} 次，这次尝试仍未完成。可在服务恢复后重试。`,
        );
      }
      const value = String(code || '').trim();
      if (!value) return '';
      const resumeFailure = developmentResumeFailureText(value);
      if (resumeFailure) return resumeFailure;
      const suffix = runFailureDetailText(detail);
      const withDetail = text => (suffix ? `${text} ${suffix}` : text);
      if (value === 'agentic_coder_credentialed_source_forbidden' || value === 'requested_concept_no_data') {
        return withDetail(errorText({ code: value }));
      }
      // A failed-closed run did not pass a check after it started. Its detail
      // says which check and why; the fallback holds for every check.
      if (value === 'research_agent_pipeline_failed_closed') {
        return suffix || tr('The run did not pass EasyICU\'s checks, so its results cannot be reported.', '这次运行没有通过 EasyICU 的检查，结果不能报告。');
      }
      const known = {
        extraction_database_filter_unsupported: tr(
          'The data filter could not process this official Demo; update EasyICU before retrying extraction.',
          '数据筛选器无法处理这份官方 Demo，请更新 EasyICU 后重新提取。',
        ),
        windowed_summary_statistic_unresolved: tr(
          'The question names a measurement but not which summary to report, such as its highest, lowest or mean value over a time window. State it, for example the highest lactate in the first 24 hours after ICU admission, then generate the plan again.',
          '问题提到了一项测量，但没有说明要报告哪种汇总值，例如某个时间窗内的最高值、最低值或平均值。请写明，例如入 ICU 后最初 24 小时内的最高乳酸，然后重新生成计划。',
        ),
        windowed_summary_statistic_unsupported: tr(
          'EasyICU can summarize a measurement over a time window only as its highest, lowest, mean, first or last value. Choose one of these, then generate the plan again.',
          'EasyICU 只能把时间窗内的测量值汇总为最高值、最低值、平均值、首次值或末次值。请从中选择一种，然后重新生成计划。',
        ),
        windowed_summary_window_required: tr(
          'A summary of a measurement needs a time window, for example the first 24 hours after ICU admission. State the window in the question, then generate the plan again.',
          '对测量值做汇总需要时间窗，例如入 ICU 后最初 24 小时。请在问题里写明时间窗，然后重新生成计划。',
        ),
        windowed_summary_window_unresolved: tr(
          'EasyICU could not read a single time window for this measurement from the question. State one window, for example the first 24 hours after ICU admission, then generate the plan again.',
          'EasyICU 无法从问题中读出这项测量的唯一时间窗。请写明一个时间窗，例如入 ICU 后最初 24 小时，然后重新生成计划。',
        ),
        windowed_summary_window_unsupported: tr(
          'EasyICU can summarize a measurement only over a window that starts at ICU admission, such as the first 24 hours. Change the window, then generate the plan again.',
          'EasyICU 目前只能对从入 ICU 时开始的时间窗做测量汇总，例如最初 24 小时。请修改时间窗，然后重新生成计划。',
        ),
        windowed_summary_concept_not_continuous: tr(
          'This variable is not a continuous measurement, so a highest, lowest or mean value over a time window does not apply. Ask about it another way, for example whether it occurred, then generate the plan again.',
          '这个变量不是连续测量值，不能取时间窗内的最高、最低或平均值。请换一种问法，例如是否发生，然后重新生成计划。',
        ),
        windowed_summary_materialization_window_mismatch: tr(
          'The prepared data covers a different time window from the one the question states, so the summary would not match the question. No analysis was run. Prepare the data again for the stated window.',
          '已准备的数据覆盖的时间窗与问题写明的不一致，汇总结果会与问题不符。尚未运行任何分析。请按问题写明的时间窗重新准备数据。',
        ),
        question_treatment_use_unrecorded: tr(
          'The question asks whether a treatment was given, but the prepared data record only its dose or rate, not whether it was given. A plan would answer a question about the dose, so planning stopped and no analysis was run. Ask about its dose, or about a treatment class whose use is recorded, for example “vasopressors”.',
          '问题问的是是否用了某种治疗，但准备好的数据只记录它的剂量或速率，不记录是否使用；按这些数据做的计划会回答剂量的问题，因此规划已停止，没有运行分析。请改问它的剂量，或改问记录了使用情况的药物类别，例如“血管活性药”。',
        ),
        question_mean_survival_unplanned: tr(
          'The question asks how long patients lived on average (a restricted mean survival time). A plan would answer it with the share who died by that time or with a hazard ratio, which is a different result, and a mean survival time is not yet planned as a study’s main result. Planning stopped and no analysis was run. Ask about death by that time, for example “28-day mortality”, or for a survival analysis.',
          '问题问的是平均存活多长时间（限制性平均生存时间）。按现有方法做的计划只能回答到该时间的死亡比例或风险比，结论不同；平均生存时间还不能作为研究的主要结果来规划，因此规划已停止，没有运行分析。请改问到该时间的死亡，例如“28 天死亡”，或改做生存分析。',
        ),
        question_named_level_undistinguished: tr(
          'The question names one kind of a clinical measure, such as invasive or non-invasive ventilation, but the prepared data record only whether it happened, not which kind. A plan would answer a broader question, so planning stopped and no analysis was run. Ask again without naming the kind, for example “mechanical ventilation”; the result then covers every kind.',
          '问题指定了某项临床指标的一种类型（例如有创或无创通气），但准备好的数据只记录是否发生，不区分类型；按这些数据做的计划会回答一个更宽的问题，因此规划已停止，没有运行分析。请去掉类型再问，例如写“机械通气”；结果将包含所有类型。',
        ),
        study_stated_variable_unrecorded: tr(
          'The question asks to adjust for or analyse by subgroups of a variable that was not recorded from it, so planning stopped before any model was called and no analysis was run. Ask in the conversation to record it again in the question\'s own words, or remove it from the question.',
          '问题要求调整或按其分组分析的一个变量没有记录下来，因此规划在调用模型前已停止，没有运行分析。请在对话里让它按问题原文重新记录这个变量，或在问题中去掉它。',
        ),
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
        research_pipeline_plan_step_needs_sandbox: tr(
          'One step in this plan needs code written by the model during the run, and the desktop app does not run such code. Remove or change that step, then approve the plan. No analysis has been run.',
          '计划里有一步要靠模型临时写代码才能完成，桌面版不运行这类代码。请删除或修改这一步，再批准计划。尚未运行任何分析。',
        ),
        // Database capability gate (host: databases/research_approval.py).
        study_capability_refused: tr(
          'The selected database lacks data this study needs, so no analysis has been run. Change the study design or data source in the conversation.',
          '所选数据库缺少这项研究需要的数据，尚未运行任何分析。请在对话里修改研究设计或数据来源。',
        ),
        capability_substitution_approval_required: tr(
          'On this database the study runs with adjusted definitions, which need your approval in plan review. No analysis has been run.',
          '在这个数据库上，研究要按调整后的口径进行，需要先在计划审阅里批准。尚未运行任何分析。',
        ),
        capability_substitution_not_applied: tr(
          'The plan does not use the adjusted definitions this database requires, so no analysis has been run. Generate the plan again.',
          '计划没有按这个数据库要求的口径修改，尚未运行任何分析。请重新生成计划。',
        ),
        capability_substitution_review_mismatch: tr(
          'The approved definitions no longer match this plan, so no analysis has been run. Review and approve the plan again.',
          '已批准的口径与当前计划不一致，尚未运行任何分析。请重新审阅并批准计划。',
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
        research_pipeline_recorded_approval_required: tr(
          'The analysis did not continue on its own: no saved approval of this plan was found. Review the plan and approve it again.',
          '分析没有自动继续：没有找到这份计划已保存的批准记录。请重新审阅并批准计划。',
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
        revision_added_unrequested_analysis: tr(
          'The revised plan added an analysis that neither your question nor the review asked for, so the previous plan was kept. Review it, then generate a revised candidate if that button is shown; if not, or if it stops again, send the failure receipt to support.',
          '修订后的计划多加了一项你的问题和审查都没有要求的分析，已保留修订前的计划。请先审阅这份计划；如有「生成修订版候选计划」按钮可点击重试，若没有或再次停止，请把失败回执发给支持人员。',
        ),
        research_pipeline_execution_failed: tr(
          'The run stopped before finishing. Send the failure receipt to support and retry once the cause is found.',
          '本次运行未完成。请把失败回执发给支持人员，查明原因后再重试。',
        ),
        research_pipeline_projection_privacy_blocked: tr(
          'For privacy, this result page is withheld: the page data contained content that must not be shown. The analysis output is kept. Send the failure receipt to support.',
          '为保护隐私，本次结果页面暂不显示：页面数据中检测到不应展示的内容，分析结果已保留。请把失败回执发给支持人员。',
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
          'The model service did not return a result in time. See the run record for automatic retries and recovery.',
          '模型服务未及时返回结果。自动重试和恢复进展可在运行记录中查看。',
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
        result_append_source_changed: tr('The data or results this addition builds on have changed, so it stopped. Check the source in data preparation.', '这次结果所用的数据或结果文件已被改动，追加已停止。请到数据准备里核对数据来源。'),
        result_append_execution_failed: tr('The addition did not finish; earlier results are unchanged. Check its analysis settings.', '追加分析没有完成，之前的结果不受影响。请检查分析设置。'),
        result_append_unavailable: tr('This result cannot take an addition yet. Open analysis settings to check its source and required variables.', '这次结果还不能追加分析。请到分析设置核对数据来源和所需变量。'),
        plan_append_scope_changed: tr('The addition changes more than the selected analysis. Prepare it again from the result page.', '追加方案改动了所选分析以外的内容，请回到结果页重新准备。'),
        plan_append_source_changed: tr('The study plan has changed. Check the earlier result, then prepare the addition again.', '原研究方案已经改变。请先核对之前的结果，再重新准备追加分析。'),
        plan_append_source_not_approvable: tr('The study plan has not passed review yet. Finish its review before adding an analysis.', '原研究方案还没有通过审阅，请先完成审阅再追加分析。'),
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
      // A typed reason already explains these stops and supplies recovery copy.
      if (['research_pipeline_progressive_compile_failed', 'research_pipeline_scientific_authority_refused', 'research_pipeline_execution_failed', 'research_pipeline_approved_run_failed'].includes(value) && suffix) return suffix;
      if (known[value]) return withDetail(known[value]);
      // An unlisted code stays visible for diagnosis, after a plain sentence.
      return withDetail(tr(`The run stopped before finishing (code: ${value}).`, `运行在完成前停止（代码：${value}）。`));
    }

    function manuscriptNotPassedText(reasonCode) {
      const code = String(reasonCode || '').trim();
      const lead = tr('Analysis complete. The manuscript draft still needs revision.', '分析已完成。论文草稿仍需修改。');
      // Append only a distinct manuscript reason, never the generic failure again.
      const reason = code === 'writer_provider_transport_unavailable'
        ? tr('The model service was unavailable while writing the draft. Try again later.', '撰写草稿时无法连接模型服务，请稍后重试。')
        : ['evidence_complete_not_satisfied', 'numeric_verified_not_satisfied'].includes(code)
          ? runFailureDetailText({ code }) : '';
      return lead + (reason ? ' ' + reason : '');
    }

    return Object.freeze({ errorText, jobErrorText, modelErrorText, providerPreset, option, runFailureText, studyFamilyGapText, manuscriptNotPassedText });
  }

  window.EasyICU.guidedPi.declare('errorText', { create, scanErrorText });
})();
