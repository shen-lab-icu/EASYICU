'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

function renderer() {
  const window = { EU_HTML: { esc: value => String(value)
    .replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;').replaceAll("'", '&#39;') } };
  const context = vm.createContext({ window, URL });
  for (const name of ['modules', 'markdown']) {
    vm.runInContext(fs.readFileSync(path.resolve(__dirname,
      `../../src/easyicu/webserver/static/js/screens-guided-pi-${name}.js`), 'utf8'), context);
  }
  return window.EasyICU.guidedPi.require('markdown').render;
}

test('display fractions become readable text without changing the saved reply', () => {
  const render = renderer();
  const source = String.raw`\[
\frac{\text{基线乳酸}-\text{后续乳酸}}{\text{基线乳酸}}\times100\%
\]`;
  const html = render(source);
  assert.match(html, /role="math"/);
  assert.match(html, /\(基线乳酸-后续乳酸\) \/ \(基线乳酸\) × 100%/);
  assert.doesNotMatch(html, /\\(?:frac|text|times|\[|\])/);
  assert.equal(source.includes(String.raw`\frac`), true, 'copy still reads the untouched message');
});

test('inline formulas and nested fractions retain arithmetic grouping', () => {
  const render = renderer();
  const html = render(String.raw`模型为 \(\beta_1 \times x + \alpha\)，比值：
$$\frac{a+b}{\frac{c}{d}}$$`);
  assert.match(html, /β_1 × x \+ α/);
  assert.match(html, /\(a\+b\) \/ \(\(c\) \/ \(d\)\)/);
});

test('formula text stays escaped, and unsupported commands do not become links or HTML', () => {
  const render = renderer();
  const html = render(String.raw`\[\text{<img src=x onerror=alert(1)>}\]

\[\href{javascript:alert(1)}{x}\]`);
  assert.match(html, /&lt;img/);
  assert.match(html, /gpi-math-source/);
  assert.doesNotMatch(html, /<img|<script|<a\b|onerror="/);
});

test('malformed and excessive formulas are shown as escaped source rather than partly reinterpreted', () => {
  const render = renderer();
  for (const formula of [String.raw`\frac{a}{b`, String.raw`\unknown{x}`, '{'.repeat(40) + 'a' + '}'.repeat(40)]) {
    const html = render(String.raw`\[` + formula + String.raw`\]`);
    assert.match(html, /gpi-math-source/);
    assert.doesNotMatch(html, /role="math"/);
  }
});

test('code spans and fenced code preserve formula syntax verbatim', () => {
  const render = renderer();
  const html = render('`\\(\\frac{a}{b}\\)`\n\n```tex\n\\[\n\\frac{a}{b}\n\\]\n<script>x</script>\n```');
  assert.match(html, /<code>\\\(\\frac\{a\}\{b\}\\\)<\/code>/);
  assert.match(html, /<pre><code>\\\[\n\\frac\{a\}\{b\}\n\\\]\n&lt;script&gt;x&lt;\/script&gt;<\/code><\/pre>/);
  assert.doesNotMatch(html, /role="math"|<script>/);
});

test('ordinary headings, tables, links, and currency remain readable', () => {
  const render = renderer();
  const html = render('# 结果\n\n价格 $20；**结果**与[来源](https://example.org/a)\n\n|分组|值|\n|---|---|\n|A|2|');
  assert.match(html, /gpi-md-heading/);
  assert.match(html, /价格 \$20；<strong>结果<\/strong>/);
  assert.match(html, /href="https:\/\/example.org\/a"/);
  assert.match(html, /<table class="gpi-md-table">/);
});
