/* Owner: Pi failure codes and setup-value helpers → user-facing copy.

   The transport and the session runner report failures as machine codes; the
   shell used to carry their bilingual text, which kept screens-guided-pi.js
   over its size budget. Only the presentation lives here: the codes stay the
   contract with the gateway and the runner. */
(function () {
  'use strict';

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
      if (error.code === 'research_pipeline_execution_runtime_unavailable') {
        return tr('The container runtime that executes analysis code is not running. Start it (Docker Desktop, or "colima start") and run again.', '执行分析代码的容器运行环境未启动。请先启动它（Docker Desktop，或 "colima start"），然后重新运行。');
      }
      if (staticPreview() && String(error.message || '').includes('Failed to fetch')) {
        return tr('This is a static preview without the EasyICU backend. Start EasyICU and open http://127.0.0.1:8765/#guided.', '这是不带 EasyICU 后端的静态预览。请启动 EasyICU，再打开 http://127.0.0.1:8765/#guided。');
      }
      return String(error.message || error.code || error);
    }

    function providerPreset(config, runtime) {
      const transport = config.api_transport || runtime.api_transport || 'openai-completions';
      const base = String(config.base_url || '').toLowerCase();
      if (transport === 'anthropic-messages') return 'anthropic';
      if (transport === 'google-generative-ai') return 'google';
      if (base.includes('api.openai.com')) return 'openai';
      if (base.includes('openrouter.ai')) return 'openrouter';
      if (base.includes('api.deepseek.com')) return 'deepseek';
      if (base.includes('127.0.0.1:8317') || base.includes('localhost:8317')) return 'cliproxyapi';
      return 'custom-openai';
    }

    function option(value, selected, label) {
      return `<option value="${value}" ${value === selected ? 'selected' : ''}>${label}</option>`;
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

    return Object.freeze({ errorText, modelErrorText, providerPreset, option });
  }

  window.EasyICU.guidedPi.declare('errorText', { create });
})();
