/* Owner: readiness at the Copilot governed-run action. */
(function () {
  'use strict';

  const RUN_ACTIONS = new Set([
    'provider_ready_to_generate_plan', 'plan_execution_upgrade_required',
    'failed_pipeline_execution_retry_available', 'failed_pipeline_execution_retry_futile',
    'plan_scientific_changes_required', 'failed_pipeline_requires_fresh_plan',
    'plan_configuration_superseded', 'plan_review_not_resumable',
    'scientific_plan_review_policy_stale', 'planner_checkpoint_resume_available',
    'agent_plan_revision_nonconvergent', 'operator_plan_approval_required', 'target_trial_plan_ready',
    'question_requirement_not_covered',
  ]);

  function create({ state, tr }) {
    function forAction(code) {
      if (!RUN_ACTIONS.has(code)) return null;
      const provider = (state.session || {}).research_provider || {};
      const runtime = state.runtime || {};
      const configuration = runtime.configuration || {};
      const connect = reason => {
        const hosted = window.EasyICU.guidedPi.optional('hosted');
        return hosted && hosted.active() ? {
          reason: tr('The model connection needs the administrator’s attention.', '模型连接需要管理员处理。'),
          connect: false,
        } : { reason, connect: true };
      };
      if (!provider.provider || !provider.model) return connect(tr(
        'Connect a model for this conversation first.', '请先为此对话连接模型。',
      ));
      if (provider.provider === 'codex') {
        if (provider.credential_source !== 'codex_user_auth'
          || !(state.codexAuth || {}).authentication_verified) return connect(tr(
          'Sign in to the account in this browser first.', '请先在此浏览器登录账户。',
        ));
        if (runtime.shell_ready !== true && runtime.status !== 'ready') return connect(tr(
          'The account connection is not ready.', '账户连接尚未就绪。',
        ));
      } else if (provider.credential_source !== 'pi_verified' || runtime.status !== 'ready'
        || configuration.connection_verified === false || configuration.credential_present === false
        || configuration.model_available === false || configuration.inference_verified === false
        || configuration.connection_error
        || (configuration.api_transport || runtime.api_transport) !== 'openai-completions') {
        return connect(tr('Check the model connection first.', '请先检查模型连接。'));
      }
      const workflow = state.workflow || {};
      const decisions = workflow.host_decisions;
      const decision = decisions && decisions.plan_transition;
      const reviewedRun = String((workflow.plan_review_summary || {}).run_id || '');
      if (code === 'plan_execution_upgrade_required' && decisions
        && (!decision || decision.next_action_code !== code || !decision.source_run_id
          || (reviewedRun && reviewedRun !== decision.source_run_id))) return {
        reason: tr('The displayed plan changed. Refresh this project.', '显示的计划已变更，请刷新项目。'),
        connect: false,
      };
      return null;
    }

    function noteRefusal(error) {
      const code = String(error && error.code || '');
      const { provider } = (state.session || {}).research_provider || {};
      if (provider === 'codex') {
        if (!['codex_auth_login_required', 'codex_auth_session_invalid'].includes(code)) return false;
        state.codexAuth = null;
        return true;
      }
      if (provider && (['pi_provider_research_configuration_unverified', 'pi_provider_research_inference_unverified',
        'pi_provider_research_transport_unsupported'].includes(code) || code.startsWith('pi_credential_'))) {
        state.runtime = { ...state.runtime, configuration: {
          ...(state.runtime || {}).configuration, connection_verified: false,
        } };
        return true;
      }
      return false;
    }

    function isVisible() {
      return Boolean(state.host?.querySelector('[data-gpi-run-readiness]'));
    }

    return { forAction, noteRefusal, isVisible };
  }

  window.EasyICU.guidedPi.declare('runReadiness', { create });
})();
