/* Owner: drawing the theme colours in browsers older than Chrome 111.

   The stylesheets write the theme as oklch() tokens and derive tints with
   color-mix(). Chrome and Edge draw neither before version 111 (March 2023);
   Windows 7 stops at 109, and hospital intranet PCs rarely update. Such a
   browser keeps the raw text of a declaration that holds var(), but the
   value resolves to nothing, so the property is unset: the accent, the
   tints and many borders disappear.

   This owner runs once the stylesheets in <head> are parsed. It computes
   each oklch()/color-mix() value from the root tokens and writes rgb()/rgba()
   back into the same declaration; it does the same for inline styles that
   pages render, and again when the appearance editor (tweaks.js) changes a
   root token. A browser that draws both functions returns at once.
   Colours without var() are written as hex in the sources, because an old
   browser drops those declarations while parsing
   (tests/webserver/test_legacy_browser_colours.py). */
(function () {
  'use strict';

  const FUNCTIONS = /(color-mix|oklch)\(/;

  function clamp(x, low, high) { return Math.min(high, Math.max(low, x)); }

  function oklch(l, c, h, alpha) {
    const angle = h * Math.PI / 180;
    const a = c * Math.cos(angle);
    const b = c * Math.sin(angle);
    const l_ = Math.pow(l + 0.3963377774 * a + 0.2158037573 * b, 3);
    const m_ = Math.pow(l - 0.1055613458 * a - 0.0638541728 * b, 3);
    const s_ = Math.pow(l - 0.0894841775 * a - 1.2914855480 * b, 3);
    const linear = [
      4.0767416621 * l_ - 3.3077115913 * m_ + 0.2309699292 * s_,
      -1.2684380046 * l_ + 2.6097574011 * m_ - 0.3413193965 * s_,
      -0.0041960863 * l_ - 0.7034186147 * m_ + 1.7076147010 * s_,
    ];
    const encode = x => {
      const v = clamp(x, 0, 1);
      return 255 * (v <= 0.0031308 ? 12.92 * v : 1.055 * Math.pow(v, 1 / 2.4) - 0.055);
    };
    return [encode(linear[0]), encode(linear[1]), encode(linear[2]), alpha];
  }

  function number(text, percentScale) {
    const value = parseFloat(text);
    return /%$/.test(text.trim()) ? value / 100 * percentScale : value;
  }

  let probe = null;
  // [r, g, b, a] with channels 0-255 and alpha 0-1, or null.
  function parseColor(text) {
    const value = String(text || '').trim().toLowerCase();
    if (!value) return null;
    if (value === 'transparent') return [0, 0, 0, 0];
    if (value === 'white') return [255, 255, 255, 1];
    if (value === 'black') return [0, 0, 0, 1];
    let m = /^#([0-9a-f]{3,8})$/.exec(value);
    if (m) {
      let hex = m[1];
      if (hex.length === 3 || hex.length === 4) hex = hex.split('').map(ch => ch + ch).join('');
      if (hex.length !== 6 && hex.length !== 8) return null;
      const channel = i => parseInt(hex.slice(i, i + 2), 16);
      return [channel(0), channel(2), channel(4), hex.length === 8 ? channel(6) / 255 : 1];
    }
    m = /^oklch\(\s*([\d.]+%?)\s+([\d.]+%?)\s+([\d.]+)(?:deg)?\s*(?:\/\s*([\d.]+%?))?\s*\)$/.exec(value);
    if (m) {
      return oklch(number(m[1], 1), number(m[2], 0.4), parseFloat(m[3]), m[4] == null ? 1 : number(m[4], 1));
    }
    m = /^rgba?\(\s*([^)]*)\)$/.exec(value);
    if (m) {
      const parts = m[1].split(/[\s,/]+/).filter(Boolean);
      if (parts.length < 3) return null;
      return [number(parts[0], 255), number(parts[1], 255), number(parts[2], 255), parts[3] == null ? 1 : number(parts[3], 1)];
    }
    // Named and hsl() colours: let the browser read them.
    if (!probe && (document.head || document.documentElement)) {
      probe = document.createElement('span');
      (document.head || document.documentElement).appendChild(probe);
    }
    if (!probe) return null;
    probe.style.color = '';
    probe.style.color = value;
    if (!probe.style.color) return null;
    const computed = getComputedStyle(probe).color;
    return /^rgba?\(/.test(computed) ? parseColor(computed) : null;
  }

  // color-mix(in srgb, A p1, B p2): CSS Color 5, premultiplied alpha.
  function mix(argText) {
    const parts = splitTop(argText, ',');
    if (parts.length !== 3 || !/^in\s+srgb$/i.test(parts[0].trim())) return null;
    const side = text => {
      const m = /^(.*?)(?:\s+([\d.]+)%)?$/.exec(text.trim());
      return { color: parseColor(m[1]), pct: m[2] == null ? null : parseFloat(m[2]) };
    };
    const one = side(parts[1]);
    const two = side(parts[2]);
    if (!one.color || !two.color) return null;
    let p1 = one.pct;
    let p2 = two.pct;
    if (p1 == null && p2 == null) { p1 = 50; p2 = 50; }
    else if (p1 == null) p1 = 100 - p2;
    else if (p2 == null) p2 = 100 - p1;
    const sum = p1 + p2;
    if (sum <= 0) return null;
    const w1 = p1 / sum;
    const w2 = p2 / sum;
    const alpha = one.color[3] * w1 + two.color[3] * w2;
    const rgb = [0, 1, 2].map(i => alpha === 0 ? 0
      : (one.color[i] * one.color[3] * w1 + two.color[i] * two.color[3] * w2) / alpha);
    return rgb.concat(alpha * Math.min(1, sum / 100));
  }

  function format(color) {
    const rgb = color.slice(0, 3).map(x => Math.round(clamp(x, 0, 255)));
    const alpha = Math.round(clamp(color[3], 0, 1) * 1000) / 1000;
    return alpha === 1 ? `rgb(${rgb.join(', ')})` : `rgba(${rgb.join(', ')}, ${alpha})`;
  }

  // Splits on a separator outside parentheses and quotes.
  function splitTop(text, separator) {
    const out = [];
    let depth = 0;
    let quote = '';
    let start = 0;
    for (let i = 0; i < text.length; i += 1) {
      const ch = text[i];
      if (quote) { if (ch === quote && text[i - 1] !== '\\') quote = ''; continue; }
      if (ch === '"' || ch === "'") quote = ch;
      else if (ch === '(') depth += 1;
      else if (ch === ')') depth -= 1;
      else if (ch === separator && depth === 0) { out.push(text.slice(start, i)); start = i + 1; }
    }
    out.push(text.slice(start));
    return out;
  }

  function closing(text, open) {
    let depth = 0;
    for (let i = open; i < text.length; i += 1) {
      if (text[i] === '(') depth += 1;
      else if (text[i] === ')') { depth -= 1; if (depth === 0) return i; }
    }
    return -1;
  }

  // Replaces var(--name, fallback) with the root token, recursively.
  function substitute(value, tokens, depth) {
    let out = value;
    for (let guard = 0; guard < 64; guard += 1) {
      const at = out.lastIndexOf('var(');
      if (at < 0) return out;
      const end = closing(out, at + 3);
      if (end < 0) return null;
      const args = splitTop(out.slice(at + 4, end), ',');
      const name = args[0].trim();
      let replacement = tokens(name);
      if (!replacement && args.length > 1) replacement = args.slice(1).join(',').trim();
      if (!replacement || depth > 8) return null;
      replacement = substitute(replacement, tokens, depth + 1);
      if (replacement == null) return null;
      out = out.slice(0, at) + replacement + out.slice(end + 1);
    }
    return null;
  }

  // The value with every oklch()/color-mix() computed, or null.
  function evaluate(value, tokens) {
    let out = substitute(value, tokens, 0);
    if (out == null) return null;
    for (let guard = 0; guard < 64; guard += 1) {
      const mixAt = out.lastIndexOf('color-mix(');
      const okAt = out.lastIndexOf('oklch(');
      const at = Math.max(mixAt, okAt);
      if (at < 0) return out;
      const isMix = at === mixAt;
      const open = at + (isMix ? 9 : 5);
      const end = closing(out, open);
      if (end < 0) return null;
      const color = isMix ? mix(out.slice(open + 1, end)) : parseColor(out.slice(at, end + 1));
      if (!color) return null;
      out = out.slice(0, at) + format(color) + out.slice(end + 1);
    }
    return null;
  }

  // "prop: value !important; ..." -> [{ name, value, important }]
  function declarations(cssText) {
    return splitTop(cssText || '', ';').map(part => {
      const colon = part.indexOf(':');
      if (colon < 0) return null;
      let value = part.slice(colon + 1).trim();
      const important = /!\s*important$/i.test(value);
      if (important) value = value.replace(/!\s*important$/i, '').trim();
      return { name: part.slice(0, colon).trim(), value, important };
    }).filter(d => d && d.name);
  }

  // A rewritten declaration keeps its raw value, so a changed root token can
  // recompute it. Inline styles are dropped once their element leaves.
  const written = [];
  const rootTokens = {};
  const scopedTokens = {};
  const seenSheets = new WeakSet();
  const seenStyles = new WeakSet();
  const root = document.documentElement;
  let lastRootStyle = null;

  function tokens(name) {
    return root.style.getPropertyValue(name).trim()
      || rootTokens[name] || scopedTokens[name]
      || getComputedStyle(root).getPropertyValue(name).trim();
  }

  function adopt(style, owner) {
    if (!style || seenStyles.has(style)) return false;
    seenStyles.add(style);
    let found = false;
    declarations(style.cssText).forEach(d => {
      if (owner && owner.rule && d.name.indexOf('--') === 0) {
        (owner.rule === ':root' ? rootTokens : scopedTokens)[d.name] = d.value;
      }
      if (!FUNCTIONS.test(d.value)) return;
      for (let i = written.length - 1; i >= 0; i -= 1) {
        if (written[i].style === style && written[i].name === d.name) written.splice(i, 1);
      }
      written.push({ style, el: owner && owner.el, name: d.name, raw: d.value, important: d.important });
      found = true;
    });
    return found;
  }

  function walk(rules) {
    Array.prototype.forEach.call(rules || [], rule => {
      if (rule.style) adopt(rule.style, { rule: rule.selectorText === ':root' ? ':root' : 'scoped' });
      if (rule.cssRules) walk(rule.cssRules);
    });
  }

  function adoptSheets() {
    Array.prototype.forEach.call(document.styleSheets, sheet => {
      if (seenSheets.has(sheet)) return;
      let rules = null;
      try { rules = sheet.cssRules; } catch (e) { return; }
      seenSheets.add(sheet);
      walk(rules);
    });
  }

  function apply() {
    for (let i = written.length - 1; i >= 0; i -= 1) {
      if (written[i].el && !written[i].el.isConnected) written.splice(i, 1);
    }
    written.forEach(item => {
      const value = evaluate(item.raw, tokens);
      if (value != null && item.style.getPropertyValue(item.name).trim() !== value) {
        item.style.setProperty(item.name, value, item.important ? 'important' : '');
      }
    });
    lastRootStyle = root.getAttribute('style');
  }

  function adoptInline(node) {
    if (!node || node.nodeType !== 1) return false;
    const found = [];
    if (node.hasAttribute('style') && FUNCTIONS.test(node.getAttribute('style'))) found.push(node);
    if (node.querySelectorAll) found.push(...node.querySelectorAll('[style*="color-mix("],[style*="oklch("]'));
    return found.map(el => adopt(el.style, { el })).some(Boolean);
  }

  function readRootStyle() {
    // tweaks.js wrote root tokens: their raw values replace the old ones.
    for (let i = written.length - 1; i >= 0; i -= 1) {
      if (written[i].style === root.style) written.splice(i, 1);
    }
    seenStyles.delete(root.style);
    adopt(root.style, { el: root });
  }

  function start() {
    adoptSheets();
    readRootStyle();
    apply();
    const observer = new MutationObserver(records => {
      let changed = false;
      records.forEach(record => {
        if (record.type === 'attributes' && record.target === root) {
          if (root.getAttribute('style') === lastRootStyle) return;
          readRootStyle();
          changed = true;
        } else if (record.type === 'attributes') {
          const el = record.target;
          if (!FUNCTIONS.test(el.getAttribute('style') || '')) return;
          seenStyles.delete(el.style);
          changed = adopt(el.style, { el }) || changed;
        } else {
          record.addedNodes.forEach(node => { changed = adoptInline(node) || changed; });
        }
      });
      if (changed) apply();
    });
    observer.observe(root, { attributes: true, attributeFilter: ['style'], childList: true, subtree: true });
    window.addEventListener('load', () => { adoptSheets(); apply(); });
  }

  const supports = window.CSS && typeof CSS.supports === 'function'
    && CSS.supports('color', 'color-mix(in srgb, red, blue)')
    && CSS.supports('color', 'oklch(50% 0.1 200)');

  window.EU_LEGACY_COLORS = Object.freeze({ parseColor, mix, evaluate, declarations, active: !supports });
  if (!supports) start();
})();
