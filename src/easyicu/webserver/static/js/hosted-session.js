/* Owner: the hosted login session in the browser.

   On a hosted instance the login gateway (src/easyicu/hosted_gateway) answers
   an /api/ call whose session has ended with 401 and the code
   `login_required`. api.js reports that here. One notice offers the way back
   to the login page instead of leaving the page at once, so a question the
   clinician is still typing can be copied first. */
(function () {
  'use strict';

  const LOGIN_REQUIRED = 'login_required';
  let notice = null;

  function tr(en, zh) {
    return typeof window.t === 'function' ? window.t(en, zh) : zh;
  }

  function expired() {
    if (notice && notice.isConnected) return;
    notice = document.createElement('div');
    notice.className = 'eu-session-expired';
    notice.setAttribute('role', 'alert');
    const text = document.createElement('span');
    text.textContent = tr(
      'Your sign-in has expired. Copy anything you have not sent yet.',
      '登录已过期。还没发送的内容请先复制。'
    );
    const link = document.createElement('a');
    link.href = '/login';
    link.textContent = tr('Sign in again', '重新登录');
    notice.append(text, link);
    document.body.appendChild(notice);
  }

  window.EU_HOSTED_SESSION = Object.freeze({ LOGIN_REQUIRED, expired });
})();
