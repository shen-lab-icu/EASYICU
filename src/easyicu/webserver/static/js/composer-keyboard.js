/* Shared chat-composer keyboard contract for Guided Copilot and its fallback. */
(function () {
  'use strict';

  function enterShouldSend(event) {
    return event.key === 'Enter'
      && !event.shiftKey
      && !event.isComposing
      && event.keyCode !== 229;
  }

  window.EU_COMPOSER_KEYBOARD = Object.freeze({ enterShouldSend });
})();
