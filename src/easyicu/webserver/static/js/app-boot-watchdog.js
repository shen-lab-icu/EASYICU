/* Owner: first app shell deadline and redacted boot diagnostics. */
(function () {
  'use strict';

  // The shell renders locally before API hydration. Allow a cold asset load
  // 45 seconds from navigation, independently of any screen's startup timer.
  const BOOT_DEADLINE_MS = 45000;
  const CODE = 'app_boot_stalled';
  const MARK_NAMES = new Set([
    'app_script', 'first_render_start', 'first_render_done',
    'boot_start', 'boot_hydrated', 'boot_rerender',
    'view_transition_start', 'view_transition_finished', 'view_transition_skipped',
  ]);
  const ROUTE_HASHES = new Set([
    '', '#guided', '#extraction', '#patient', '#cohort', '#crossdb',
    '#dictionary', '#skills', '#settings', '#tutorial', '#agent', '#assistant',
    '#entry', '#ideas', '#states', '#audit', '#sofareclass', '#icd', '#datamap', '#help',
  ]);
  const ERROR_NAMES = new Set([
    'Error', 'TypeError', 'ReferenceError', 'SyntaxError', 'RangeError', 'URIError',
    'EvalError', 'AggregateError', 'AbortError', 'NetworkError', 'SecurityError',
    'InvalidStateError', 'NotAllowedError',
  ]);
  const marks = {};
  const errors = [];
  let stop = null;
  let bodyObserver = null;
  const boot = window.EU_BOOT = {
    mark(name) {
      if (MARK_NAMES.has(name)) marks[name] = Math.min(3600000, Math.floor(performance.now()));
    },
    get marks() { return { ...marks }; },
    failure: null,
    shellMounted() {
      bodyObserver?.disconnect();
      bodyObserver = null;
      stop?.remove();
      stop = null;
    },
  };

  function recordError(type, value) {
    if (errors.length >= 8) return;
    let name = 'UnknownError';
    try {
      const candidate = value && value.name;
      if (ERROR_NAMES.has(candidate)) name = candidate;
    } catch (_) { /* An arbitrary thrown object can have a throwing name getter. */ }
    errors.push({ type, name });
  }

  window.addEventListener('error', (event) => recordError('error', event.error), true);
  window.addEventListener('unhandledrejection', (event) => recordError('unhandledrejection', event.reason));

  function showStop() {
    stop = document.createElement('section');
    stop.className = 'eu-app-boot-stop';
    stop.setAttribute('data-app-boot-failed', CODE);
    stop.setAttribute('role', 'alert');
    const card = document.createElement('div');
    card.className = 'gd-startup-card';
    const english = document.createElement('span');
    english.lang = 'en';
    english.textContent = 'EasyICU took too long to start. Reload the page to try again.';
    const chinese = document.createElement('span');
    chinese.lang = 'zh';
    chinese.textContent = 'EasyICU 启动超时。请重新加载页面再试。';
    const code = document.createElement('code');
    code.textContent = CODE;
    const reload = document.createElement('button');
    reload.className = 'gd-startup-reload';
    reload.type = 'button';
    reload.textContent = 'Reload / 重新加载';
    reload.addEventListener('click', () => window.location.reload());
    card.append(english, chinese, code, reload);
    stop.append(card);
    document.body.append(stop);
  }

  function checkShell() {
    const app = document.getElementById('app');
    if (boot.failure || app?.querySelector(':scope > [data-eu-app-shell]')) return;
    const route = window.location.hash.split(/[/?]/, 1)[0];
    const width = window.innerWidth;
    const payload = boot.failure = {
      code: CODE,
      readyState: document.readyState,
      marks: boot.marks,
      app_child_count: Math.min(2147483647, app?.childElementCount ?? 0),
      app_text_length: Math.min(2147483647, app?.textContent.length ?? 0),
      presence: {
        SCREENS: !!window.SCREENS,
        __euRender: !!window.__euRender,
        EU_API: !!window.EU_API,
        EU_CATALOG: !!window.EU_CATALOG,
        EU_CAPABILITIES: !!window.EU_CAPABILITIES,
        EU_GUIDED_STARTUP: !!window.EU_GUIDED_STARTUP,
      },
      route_transitioning: document.documentElement.classList.contains('eu-route-transitioning'),
      view_transition_available: typeof document.startViewTransition === 'function',
      route_hash: ROUTE_HASHES.has(route) ? route : '#other',
      viewport_width_bucket: width < 768 ? 'under_768' : width < 1181 ? '768_to_1180' : '1181_plus',
      errors: errors.slice(),
    };
    if (document.body) {
      showStop();
    } else {
      bodyObserver = new MutationObserver(() => {
        if (!document.body) return;
        bodyObserver.disconnect();
        bodyObserver = null;
        if (!document.getElementById('app')?.querySelector(':scope > [data-eu-app-shell]')) {
          showStop();
        }
      });
      bodyObserver.observe(document.documentElement, { childList: true });
    }
    window.fetch('/api/client/boot-diagnostics', {
      method: 'POST',
      credentials: 'same-origin',
      keepalive: true,
      redirect: 'error',
      // Keep the Origin header on this same-origin POST and omit Referer path/query.
      referrerPolicy: 'strict-origin',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    }).catch(() => {});
  }

  setTimeout(checkShell, Math.max(0, BOOT_DEADLINE_MS - performance.now()));
})();
