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

  /* Keep in sync with MIN_NODE_VERSION in pi_copilot/gateway.py. */
  const MIN_NODE = '22.19.0';
  const INSTALL_CMD = 'easyicu-copilot-install';

  /* Lower number = fix this first. */
  const CATALOG = {
    node_available: {
      order: 10,
      title: () => t('Node.js was not found', '未找到 Node.js'),
      fix: () => t(
        `The Copilot runtime requires Node.js. Install Node ${MIN_NODE} or newer, then reopen this page.`,
        `研究助手运行环境依赖 Node.js。请安装 Node ${MIN_NODE} 或更高版本，然后重新打开本页。`,
      ),
    },
    node_version_supported: {
      order: 20,
      title: () => t('The installed Node.js is too old', '已安装的 Node.js 版本过低'),
      fix: (runtime) => {
        const found = runtime && runtime.node_version;
        return found
          ? t(
            `Found Node ${found}; the Copilot runtime needs ${MIN_NODE} or newer. Upgrade Node, then reopen this page.`,
            `检测到 Node ${found}，研究助手运行环境需要 ${MIN_NODE} 或更高版本。请升级 Node 后重新打开本页。`,
          )
          : t(
            `The Copilot runtime needs Node ${MIN_NODE} or newer. Upgrade Node, then reopen this page.`,
            `研究助手运行环境需要 Node ${MIN_NODE} 或更高版本。请升级 Node 后重新打开本页。`,
          );
      },
    },
    entrypoint_available: {
      order: 30,
      title: () => t('The Copilot runtime is not installed yet', '研究助手运行环境尚未安装'),
      fix: () => t(
        `Run \`${INSTALL_CMD}\` in your EasyICU environment to install the pinned runtime.`,
        `在 EasyICU 环境中运行 \`${INSTALL_CMD}\` 安装已固定版本的运行环境。`,
      ),
    },
    dependency_installed: {
      order: 40,
      title: () => t('The Copilot runtime files are incomplete', '研究助手运行环境文件不完整'),
      fix: () => t(
        `The runtime folder exists but its packages are missing. Run \`${INSTALL_CMD}\` to reinstall.`,
        `运行环境目录存在但依赖包缺失。请运行 \`${INSTALL_CMD}\` 重新安装。`,
      ),
    },
    lockfile_present: {
      order: 50,
      title: () => t('The runtime lockfile is missing', '运行环境 lockfile 缺失'),
      fix: () => t(
        `EasyICU pins the Copilot runtime to an exact version set, and that lock is absent. Run \`${INSTALL_CMD}\` to restore it.`,
        `EasyICU 会把研究助手运行环境固定到确切版本集合，当前缺少该锁定文件。请运行 \`${INSTALL_CMD}\` 恢复。`,
      ),
    },
    runtime_integrity_verified: {
      order: 60,
      title: () => t('The runtime files do not match their pinned digests', '运行环境文件与固定摘要不一致'),
      fix: () => t(
        `The installed files differ from the version EasyICU pinned. Reinstall with \`${INSTALL_CMD}\` rather than continuing.`,
        `已安装文件与 EasyICU 固定的版本不一致。请用 \`${INSTALL_CMD}\` 重新安装，不要继续使用。`,
      ),
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
