/* Owner: the public wording of assistant text in the Guided Copilot
   conversation. Internal status codes, owner module paths and file digests
   become plain words; files and digests are shown beside the conversation. */
(function () {
  'use strict';

  function create({ tr }) {
    function publicAssistantText(value) {
      return String(value || '')
        .replace(/\b(?:one[-\s]time\s+)?extraction grant\b/gi, tr(
          'one-time data preparation authorization',
          '本轮一次性数据准备授权',
        ))
        .replace(/\beasyicu_prepare_demo_source\b/gi, tr(
          'the official demo preparation workflow',
          '官方 Demo 准备流程',
        ))
        .replace(/\bpi_action_authorization_required\b/gi, tr(
          'the required EasyICU data authorization is not active for this turn',
          '本轮数据准备授权尚未生效',
        ))
        .replace(/\bpi_session_data_source_confirmation_required\b/gi, tr(
          'this conversation still needs the selected data source to be confirmed',
          '本会话仍需确认所选数据源',
        ))
        .replace(/\bpi_[a-z0-9_]+\b/gi, tr('an EasyICU internal status', 'EasyICU 内部状态'))
        .replace(/\beasyicu\.webserver\.pi_copilot(?:\.[a-z0-9_.]+)?\b/gi, 'EasyICU Copilot')
        // Earlier replies quoted readiness codes and file digests; the host
        // shows files and digests beside the conversation.
        .replace(/`?report_revision_quality_pass_analysis_only`?/gi, tr(
          'a report revision that passed its quality checks for analysis review',
          '已通过质量检查、仅供分析审阅的报告修订',
        ))
        .replace(/[，,;；]?\s*SHA-?256\s*[:：]\s*`?[a-f0-9]{64}`?/gi, '')
        .replace(/`[a-f0-9]{64}`/gi, '');
    }
    return { publicAssistantText };
  }

  window.EasyICU.guidedPi.declare('assistantText', { create });
})();
