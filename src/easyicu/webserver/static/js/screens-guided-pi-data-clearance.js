/* Owner: Guided Pi credentialed-data clearance on model-step cards. */
/* The host projects session.data_clearance (pi_copilot/service.py): whether
   the clearance gate would refuse the next model step for this conversation's
   data. A card that would call the model then offers the way to clear it
   instead of a button that can only fail, and a refusal opens the connection
   settings rather than an error banner. The gate stays at every model call. */
(function () {
  'use strict';

  const REFUSAL = 'provider_not_cleared_for_credentialed_data';

  function blocked(session) {
    const clearance = session && session.data_clearance;
    return !!(clearance && clearance.status === 'blocked' && clearance.reason_code === REFUSAL);
  }

  function blocksCard(confirmation, session) {
    const grants = confirmation && Array.isArray(confirmation.grants) ? confirmation.grants : [];
    return grants.includes('provider_run') && blocked(session);
  }

  // An account sign-in carries no declaration; only an API connection does.
  function declarable(session) {
    return !(session && session.data_clearance && session.data_clearance.declarable === false);
  }

  function cardReason(session, tr) {
    return declarable(session)
      ? tr('The connected model has no declaration for credentialed data such as MIMIC-IV or eICU.', '当前模型还没有声明能否处理 MIMIC-IV、eICU 等需授权的数据。')
      : tr('A signed-in account cannot be used with credentialed data such as MIMIC-IV or eICU.', '登录账号的模型不能用于 MIMIC-IV、eICU 等需授权的数据。');
  }

  function cardAction(session, tr, esc) {
    const label = declarable(session)
      ? tr('Declare data handling', '声明数据处理方式')
      : tr('Change model connection', '更换模型连接');
    return `<button class="btn primary sm" type="button" data-gpi-config>${esc(label)}</button>`;
  }

  function isRefusal(error) {
    return !!(error && error.code === REFUSAL);
  }

  window.EasyICU.guidedPi.declare('dataClearance', { blocked, blocksCard, cardReason, cardAction, isRefusal });
}());
