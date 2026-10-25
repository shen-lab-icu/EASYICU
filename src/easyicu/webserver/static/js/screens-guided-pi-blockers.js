/* Owner: turning Pi runtime readiness codes into something a user can act on.

   The gateway reports readiness as a set of positively-named boolean flags
   (`node_available`, `dependency_installed`, …) and the setup panel used to
   render the failing keys verbatim:

       "The Pi runtime also needs attention before chat can open:
        dependency_installed, node_available"

   Two problems. The codes are machine identifiers in a page whose other copy
   is carefully written bilingual prose; and because the flags are named for
   the *satisfied* state, listing them under "needs attention" reads as the
   exact opposite of what is wrong — a user sees "dependency_installed" and
   reasonably concludes the dependency is installed.

   This file owns the code → human text mapping. The codes stay the contract
   between gateway.py and the UI; only the presentation lives here. It is a
   separate owner file rather than more lines in screens-guided-pi.js, which
   is already over its size budget.

   Ordering note: `describe()` returns items in the order the user should act
   on them — Node before the runtime that needs Node, install before integrity
   — not in the order the gateway happened to list them. */
(function () {
  const t = (en, zh) => (window.t ? window.t(en, zh) : en);

  const incomplete = () => t('Copilot components are incomplete. Contact support.', '研究助手组件不完整，请联系技术支持。');
  const supportFix = () => t(
    'Open “Having a problem?” and copy the problem reference for support; include “Diagnostic details” from this notice.',
    '请打开「遇到问题」复制问题编号，并向技术支持提供本提示的「诊断详情」。',
  );

  /* Lower number = fix this first. */
  const CATALOG = {
    node_available: {
      order: 10,
      title: incomplete,
      fix: supportFix,
    },
    node_version_supported: {
      order: 20,
      title: incomplete,
      fix: supportFix,
    },
    entrypoint_available: {
      order: 30,
      title: incomplete,
      fix: supportFix,
    },
    dependency_installed: {
      order: 40,
      title: incomplete,
      fix: supportFix,
    },
    lockfile_present: {
      order: 50,
      title: incomplete,
      fix: supportFix,
    },
    runtime_integrity_verified: {
      order: 60,
      title: () => t('Copilot component checks failed. Contact support.', '研究助手组件校验未通过，请联系技术支持。'),
      fix: supportFix,
    },
    base_url_configured: {
      order: 70,
      title: () => t('No service address is configured', '尚未配置服务地址'),
      fix: () => t(
        'Fill in the service address field above and verify the connection.',
        '请在上方填写服务地址并完成连接验证。',
      ),
    },
  };

  /* Codes this owner deliberately does not describe: the setup form itself
     already surfaces them (the credential field, the verify button, the AI
     opt-in checkbox), so repeating them as runtime blockers would report one
     problem twice. The seven codes in CATALOG above are exactly the
     `runtime_blockers` set in pi_copilot/service.py — the ones that make the
     runtime "unavailable" rather than merely "setup_required". */
  const HANDLED_BY_FORM = [
    'api_key_configured',
    'provider_connection_verified',
    'provider_connection_unverified',
    'easyicu_ai_opt_in_disabled',
  ];

  /* codes: string[] from runtime.blockers.
     Returns [{ code, title, fix }] ordered by fix-this-first, plus any
     unrecognised codes passed through so a new gateway code is visible
     rather than silently dropped. */
  function describe(codes, runtime) {
    const list = Array.isArray(codes) ? codes : [];
    const known = [];
    const unknown = [];
    list.forEach((code) => {
      if (HANDLED_BY_FORM.indexOf(code) !== -1) return;
      const entry = CATALOG[code];
      if (entry) known.push({ code, order: entry.order, title: entry.title(), fix: entry.fix(runtime || {}) });
      else unknown.push({ code, order: 999, title: code, fix: '' });
    });
    known.sort((a, b) => a.order - b.order);
    return known.concat(unknown).map(({ code, title, fix }) => ({ code, title, fix }));
  }

  window.EU_PI_BLOCKERS = { describe, codes: () => Object.keys(CATALOG) };
})();
