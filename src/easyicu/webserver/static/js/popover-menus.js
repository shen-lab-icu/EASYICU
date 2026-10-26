/* Owner: floating popover menus — the shell-wide dismissal contract.
   A `<details data-popover-menu>` is a menu that floats over the page: the
   composer's + / access / effort / model menus, the header's layout and
   overflow menus, a conversation's ••• menu, the project switcher, the skills
   page's menus. Native <details> only toggles from its own summary; a menu
   is expected to do more, and every one of them gets it here instead of
   each owner wiring its own: opening one closes the others, a press or a
   focus move anywhere outside closes it, and Escape closes it and returns
   focus to its summary. Inline disclosures (run record, traces, evidence,
   plan sections) are not menus and never carry the attribute. Owners still
   close their own menu after an action chosen inside it. A menu whose open
   state lives in its owner rather than in a <details> registers
   { isOpen, contains, close } and gets the same press and Escape rules. An
   owner that rebuilds its markup wraps the rebuild in keepOpen so an open
   menu survives the repaint. An open menu also keeps to the room it has:
   a panel that would run past the window or the scrolling region holding
   it takes that room and scrolls inside it. Owners' CSS still picks the
   side a menu opens on (composer menus: guided-pi-composer-menus.css). */
(function () {
  'use strict';

  const ATTRIBUTE = 'data-popover-menu';
  const SELECTOR = `details[${ATTRIBUTE}]`;
  const stateMenus = [];

  function register(menu) {
    stateMenus.push(menu);
    return () => { const at = stateMenus.indexOf(menu); if (at >= 0) stateMenus.splice(at, 1); };
  }

  function closeStateMenus(node, focus) {
    return stateMenus.filter(menu => menu.isOpen() && !menu.contains(node))
      .reduce((count, menu) => { menu.close({ focus }); return count + 1; }, 0);
  }

  function openMenus(root) {
    const scope = root || document;
    return scope && typeof scope.querySelectorAll === 'function'
      ? Array.from(scope.querySelectorAll(`${SELECTOR}[open]`)) : [];
  }

  function close(menu, focusSummary) {
    if (!menu || !menu.open) return false;
    menu.open = false;
    if (focusSummary) {
      const summary = menu.querySelector(':scope > summary');
      if (summary && typeof summary.focus === 'function') summary.focus();
    }
    return true;
  }

  function contains(menu, node) {
    return !!node && typeof menu.contains === 'function' && menu.contains(node);
  }

  // Close every open menu the node is not inside. Returns how many closed.
  function closeOutside(node, options) {
    const focusSummary = !!(options && options.focusSummary);
    return openMenus().reduce((count, menu) => (
      contains(menu, node) ? count : count + (close(menu, focusSummary) ? 1 : 0)
    ), 0) + closeStateMenus(node, focusSummary);
  }

  function closeAll(options) { return closeOutside(null, options); }

  // An owner that rebuilds its markup (a running job repaints the conversation
  // on every progress event) keeps the menu the researcher has open: open
  // menus are remembered by their data-popover-key when they belong to a row
  // that can move (a conversation's •••), otherwise by class and position
  // among menus of that class, then reopened in the rebuilt markup, with
  // focus back on the summary when it was inside the menu. An owner closes
  // its menu before it rebuilds.
  function keepOpen(root, rebuild) {
    if (!root || typeof root.querySelectorAll !== 'function') return rebuild();
    const key = menu => {
      const own = typeof menu.getAttribute === 'function' ? menu.getAttribute('data-popover-key') : '';
      if (own) return `key:${own}`;
      const kind = String(menu.className || '');
      return `${kind}#${Array.from(root.querySelectorAll(SELECTOR)).filter(other => String(other.className || '') === kind).indexOf(menu)}`;
    };
    const active = typeof document !== 'undefined' ? document.activeElement : null;
    const open = openMenus(root).map(menu => ({ key: key(menu), focused: contains(menu, active) }));
    const result = rebuild();
    if (open.length) {
      Array.from(root.querySelectorAll(SELECTOR)).forEach(menu => {
        const kept = open.find(row => row.key === key(menu));
        if (!kept) return;
        menu.open = true;
        const summary = kept.focused ? menu.querySelector(':scope > summary') : null;
        if (summary && typeof summary.focus === 'function') summary.focus({ preventScroll: true });
      });
    }
    return result;
  }

  // The room is the window narrowed by every ancestor of the panel that
  // scrolls or clips; a fixed panel answers to the window alone.
  const EDGE = 8;
  function room(panel) {
    const style = node => window.getComputedStyle(node);
    let top = 0;
    let bottom = window.innerHeight;
    if (style(panel).position !== 'fixed') {
      for (let node = panel.parentElement; node && node !== document.body && node !== document.documentElement;
        node = node.parentElement) {
        if (style(node).overflowY === 'visible') continue;
        const box = node.getBoundingClientRect();
        top = Math.max(top, box.top);
        bottom = Math.min(bottom, box.bottom);
      }
    }
    return { top, bottom };
  }

  const panelOf = menu => Array.from(menu.children || []).find(child => child.tagName !== 'SUMMARY');

  // A panel that opens below its summary may reach the room's bottom edge, one
  // that opens above it the top edge; past that, it scrolls inside.
  function fit(menu) {
    if (typeof window.getComputedStyle !== 'function') return;
    const panel = panelOf(menu);
    const summary = menu.querySelector(':scope > summary');
    if (!panel || !summary || typeof panel.getBoundingClientRect !== 'function') return;
    panel.style.removeProperty('max-height');
    const box = panel.getBoundingClientRect();
    const { top, bottom } = room(panel);
    const below = box.top >= summary.getBoundingClientRect().top;
    const space = Math.floor(below ? bottom - box.top - EDGE : box.bottom - top - EDGE);
    if (space > 0 && box.height > space) {
      panel.style.maxHeight = `${space}px`;
      panel.style.overflowY = 'auto';
    }
  }

  // A panel can grow after it opens (its list arrives from the host), so an
  // open panel is fitted again whenever its size changes.
  let resizes = null;
  function watch(menu) {
    const panel = panelOf(menu);
    if (!panel || typeof window.ResizeObserver !== 'function') return;
    resizes = resizes || new window.ResizeObserver(entries => entries.forEach(entry => {
      const owner = entry.target.parentElement;
      if (owner && owner.open) fit(owner);
      else resizes.unobserve(entry.target);
    }));
    resizes.observe(panel);
  }

  // `toggle` does not bubble, so this listens in the capture phase.
  function onToggle(event) {
    const menu = event.target;
    if (!menu || typeof menu.matches !== 'function' || !menu.matches(SELECTOR) || !menu.open) return;
    openMenus().forEach(other => {
      if (other !== menu && !contains(other, menu) && !contains(menu, other)) close(other);
    });
    fit(menu);
    watch(menu);
  }

  // A press outside closes on the way down, like a native menu; the press
  // still reaches whatever was pressed. Focus leaving (Tab) closes as well.
  function onPointerDown(event) { closeOutside(event.target); }
  function onFocusIn(event) { closeOutside(event.target); }

  function onKeyDown(event) {
    if (event.key !== 'Escape' || event.defaultPrevented) return;
    if (closeAll({ focusSummary: true })) event.preventDefault();
  }

  function mount(target) {
    const doc = target || (typeof document !== 'undefined' ? document : null);
    if (!doc || typeof doc.addEventListener !== 'function') return false;
    doc.addEventListener('toggle', onToggle, true);
    doc.addEventListener('pointerdown', onPointerDown, true);
    doc.addEventListener('focusin', onFocusIn, true);
    doc.addEventListener('keydown', onKeyDown);
    return true;
  }

  mount();
  window.EU_POPOVER_MENUS = Object.freeze({
    ATTRIBUTE, SELECTOR, mount, register, closeAll, closeOutside, keepOpen, fit, onToggle, onPointerDown, onFocusIn, onKeyDown,
  });
})();
