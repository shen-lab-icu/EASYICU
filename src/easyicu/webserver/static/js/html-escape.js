/* Owner: HTML escaping for every frontend module.

   This file exists because escaping had been re-rolled in 19 separate IIFEs,
   and the copies had drifted into four incompatible contracts: `[&<>]` only,
   `[&<>"]`, `[&<>"']`, and a chained-replace variant. Escaping is a security
   primitive; four contracts means the weakest one decides what actually ships.

   The concrete defect that motivated the consolidation: screens-agent.js
   escaped with the `[&<>]` variant and then interpolated the result into ten
   HTML *attribute* positions (run labels, artifact names, project paths). A
   single `"` in any of those values truncates the attribute. The correct
   attribute escaper already existed next door in screens-agent-render.js, but
   it was trapped in that file's closure with no export — so the duplication
   was forced by the module pattern, not by carelessness.

   Contract:
     esc(value)      — safe for HTML *text* nodes AND attribute values.
     escAttr(value)  — alias of esc, kept as an explicit signal at attribute
                       call sites that the quoting matters there.

   Both escape `& < > " '` and both are null-safe: null/undefined render as
   empty string, never the literal "null". Escaping quotes inside text nodes is
   harmless — the parser decodes `&quot;`/`&#39;` back to `"`/`'` — so one
   function can safely serve both positions rather than asking every call site
   to pick correctly.

   Attribute values still round-trip through `dataset`: the HTML parser decodes
   entities, so `data-x="${esc(path)}"` read back as `el.dataset.x` returns the
   original string. Call sites that compare a dataset value against raw data
   keep working unchanged.

   Consumers must destructure from this owner at the top of their IIFE:
     const { esc } = window.EU_HTML;
   Do not add another local `function esc(` — tests/webserver/test_static_frontend_ownership.py
   fails the build if one reappears. */
(function () {
  const ENTITIES = {
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#39;',
  };

  function esc(value) {
    return String(value == null ? '' : value).replace(/[&<>"']/g, ch => ENTITIES[ch]);
  }

  window.EU_HTML = {
    esc,
    /* Same function. The distinct name documents intent at attribute call
       sites so a future reader does not have to re-derive whether quotes are
       covered there. */
    escAttr: esc,
  };
})();
