/* Owner: column identifiers quoted in plan prose, named for a Chinese reader.

   A plan step states its intent in the planner's words, and those words quote
   column identifiers ("估计 age_group_x1 与院内 death 的 sex 调整观察性关联").
   On a Chinese page, in a sentence that is already Chinese, each whole
   identifier is shown by its Chinese name: a grouped variable by its
   grouping's name (the exposure-levels reader's `variableName`), any other column
   by the plan's registered label when that label is Chinese, else by the
   catalog's governed Chinese name for that exact concept id, without a
   trailing parenthetical definition.  An identifier with no Chinese name
   keeps its text, so nothing is translated or guessed.  A name that repeats
   the words just before it ("院内 death" → "院内死亡") is not repeated, and
   the spaces an identifier needed between Chinese words are dropped.  An
   English page, or a sentence with no Chinese, keeps the plan's words with
   only its grouped variables named. */
(function () {
  'use strict';

  const CJK = /[\u3400-\u9fff]/;
  const IDENTIFIER = /(?<![A-Za-z0-9_])[a-z][a-z0-9_]*(?![A-Za-z0-9_])/g;

  function shortName(value) {
    return String(value || '').replace(/\s*[（(][^）)]*[）)]\s*$/, '').trim();
  }

  function chineseName(labels, column) {
    const registered = labels && typeof labels[column] === 'string' ? labels[column] : '';
    if (CJK.test(registered)) return shortName(registered);
    const dict = window.EU_CATALOG && window.EU_CATALOG.dict ? window.EU_CATALOG.dict : {};
    const row = Object.prototype.hasOwnProperty.call(dict, column) ? dict[column] : null;
    const name = Array.isArray(row) && typeof row[1] === 'string' ? shortName(row[1]) : '';
    return CJK.test(name) ? name : '';
  }

  function namedZh(text, options) {
    const { labels, levels } = options || {};
    const stated = String(text == null ? '' : text);
    if (window.EU_LANG !== 'zh' || !CJK.test(stated)) {
      return levels && typeof levels.inText === 'function' ? levels.inText(stated) : stated;
    }
    let out = '';
    let last = 0;
    for (const match of stated.matchAll(IDENTIFIER)) {
      const name = levels && typeof levels.has === 'function' && levels.has(match[0])
        ? levels.variableName(match[0]) : chineseName(labels, match[0]);
      if (!name) continue;
      let head = out + stated.slice(last, match.index);
      if (CJK.test(head.trimEnd().slice(-1))) head = head.trimEnd();
      let overlap = Math.min(name.length - 1, head.length);
      while (overlap >= 2 && !head.endsWith(name.slice(0, overlap))) overlap -= 1;
      out = head + (overlap >= 2 ? name.slice(overlap) : name);
      last = match.index + match[0].length;
      const space = /^\s+(?=[\u3400-\u9fff])/.exec(stated.slice(last));
      if (space) last += space[0].length;
    }
    return out + stated.slice(last);
  }

  window.AGENT_PLAN_PROSE = Object.freeze({ chineseName, namedZh });
})();
