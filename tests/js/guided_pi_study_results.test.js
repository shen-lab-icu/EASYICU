/* Current-result identity, historical isolation, and non-destructive reader layout. */
'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
// E-P2-11: argv[2:] are the explicit owner files from tools/run_js_contracts.py
// (no "." directory marker).  Resolve the js/ root from the first file's
// directory so a directory argument keeps working during migration.
const _first = path.resolve(process.argv[2]);
const jsRoot = fs.existsSync(_first) && fs.statSync(_first).isDirectory()
  ? _first
  : path.dirname(_first);
global.window = global;
global.EU_HTML = { esc: value => String(value ?? '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/"/g, '&quot;') };
require(path.join(jsRoot, 'screens-guided-pi-modules.js'));
require(path.join(jsRoot, 'screens-guided-pi-resources.js'));
require(path.join(jsRoot, 'screens-guided-pi-study-workspace.js'));
require(path.join(jsRoot, 'screens-guided-pi-run-outcome.js'));
const modules = global.EasyICU.guidedPi;
const resources = modules.require('resources').create(EU_HTML);
const outcome = modules.require('runOutcome').create({
  tr: en => en, esc: EU_HTML.esc, iconHtml: () => '', resourceButton: resources.button,
});
const ledgerSha = 'a'.repeat(64);
const artifactSha = 'd'.repeat(64);
const pdfSha = 'b'.repeat(64);
const ref = (artifact, kind = 'research_artifact') => ({
  kind, artifact, run_id: 'run_current', sha256: artifactSha, label: artifact,
});
const latest = {
  present: true, analysis_results_available: true, run_id: 'run_current', figure_count: 2,
  analysis_validated: true, manuscript_ready: true, report_revision_ready: true,
  report_revision_pdf_ready: true,
  artifact_refs: [{ ...ref('evidence_ledger.json'), sha256: ledgerSha },
    ref('result_tables.json'), ref('figure_gallery.json'),
    ref('literature_evidence.json'), ref('scientific_readiness.json'),
    ref('manuscript_scaffold.pdf', 'research_document'),
    { ...ref('manuscript_revision.pdf', 'research_document'), sha256: pdfSha }],
};
const workflow = { current_stage: 'interpretation', stages: [
  { id: 'analysis', status: 'complete' }, { id: 'interpretation', status: 'review_required' },
] };
const original = JSON.stringify({ latest, workflow });
const collection = outcome.collection(latest, workflow);
assert.equal(collection.length, 7);
assert.equal(collection.find(row => row.artifact === 'manuscript_revision.pdf').sha256, pdfSha);
assert.equal(collection.find(row => row.artifact === 'article_report.json').sha256, ledgerSha);
assert.equal(collection.find(row => row.artifact === 'full_analysis_report.json').sha256, ledgerSha);
assert.ok(!collection.some(row => row.artifact === 'manuscript_scaffold.pdf'));
assert.match(outcome.renderReviewAction(latest, workflow), /full_analysis_report.json/);
assert.equal(outcome.renderReviewAction(latest, { ...workflow, stages: [{ id: 'interpretation', status: 'blocked' }] }), '');
for (const unready of [null, { ...latest, present: false }, { ...latest, analysis_results_available: false }]) {
  assert.deepEqual(outcome.collection(unready, workflow), []);
  assert.equal(outcome.renderShelf(unready, workflow), '');
}
assert.deepEqual(outcome.collection(latest, { stages: [{ id: 'analysis', status: 'running' }] }), []);
const noPdf = outcome.collection({ ...latest, report_revision_pdf_ready: false }, workflow);
assert.ok(!noPdf.some(row => row.artifact.endsWith('.pdf')), 'Do not replace a missing revision with an old PDF');
const missingRefs = outcome.collection({ ...latest, artifact_refs: latest.artifact_refs.filter(row => row.artifact !== 'result_tables.json') }, workflow);
assert.ok(!missingRefs.some(row => row.artifact === 'result_tables.json'));
const staleLedger = outcome.collection({ ...latest, artifact_refs: latest.artifact_refs.map(row =>
  row.artifact === 'evidence_ledger.json' ? { ...row, run_id: 'run_old' } : row) }, workflow);
assert.ok(!staleLedger.some(row => row.kind === 'research_report'));
const noFigure = outcome.collection({ ...latest, figure_count: 0 }, workflow);
assert.ok(!noFigure.some(row => row.artifact === 'figure_gallery.json'));
assert.deepEqual(outcome.collection({ ...latest, artifact_refs: latest.artifact_refs.map(row => ({ ...row, run_id: 'run_old' })) }, workflow), []);
assert.equal(JSON.stringify({ latest, workflow }), original, 'Rendering must not mutate scientific authority');

// The reader toggles layout in place: PDF DOM and scroll position survive.
const classes = new Set();
const classList = { contains(name) { return classes.has(name); }, toggle(name, on) { if (on) classes.add(name); else classes.delete(name); } };
const log = {
  clientHeight: 400, top: 600,
  get scrollHeight() { return classes.has('gpi-preview-open') ? 1600 : 1000; },
  get scrollTop() { return this.top; },
  set scrollTop(value) { this.top = Math.max(0, Math.min(value, this.scrollHeight - this.clientHeight)); },
};
const main = { classList, querySelector: () => log };
const aside = { classList, closest: () => main };
const study = {};
const nav = { innerHTML: '', hidden: true };
let click;
let replacements = 0;
let inspector = null;
const host = {
  innerHTML: '', scrollTop: 283,
  addEventListener(name, fn) { if (name === 'click') click = fn; },
  querySelector(selector) { return selector === '[data-gpi-study-resources]' ? nav : selector === '.gpi-preview-inspector' ? inspector : null; },
  querySelectorAll() { return []; },
  replaceChildren() { replacements++; this.innerHTML = ''; },
};
global.document = { getElementById(id) { return id === 'gdContextAside' ? aside : id === 'gdStudyAside' ? study : null; } };
global.EU_API = { piCopilotResearchDocumentUrl: (project, run, artifact, digest) => `/preview/${project}/${run}/${artifact}?sha=${digest}` };
require(path.join(jsRoot, 'product-labels.js'));
require(path.join(jsRoot, 'screens-guided-pi-preview.js'));
const preview = modules.require('preview');
preview.mount(host);
const opened = [];
preview.setStudyResources(collection, 'project_current', row => opened.push(row));
const pdf = collection.find(row => row.artifact === 'manuscript_revision.pdf');
assert.equal(preview.open(pdf, 'project_current', { currentRunId: 'run_current' }), true);
assert.match(host.innerHTML, /This run’s results/);
assert.match(host.innerHTML, /data-gpi-study-resource="0"/);
assert.match(host.innerHTML, new RegExp(pdfSha));
assert.ok(classes.has('gpi-preview-focus'));
assert.equal(log.scrollTop + log.clientHeight, log.scrollHeight, 'Opening results must follow the bottom through message reflow');
log.scrollTop = 200;
const htmlBeforeToggle = host.innerHTML;
const focusButton = { setAttribute(name, value) { this[name] = value; } };
const clickTarget = (selector, target) => click({ target: { closest: query => query === selector ? target : null } });
clickTarget('[data-gpi-preview-focus]', focusButton);
assert.ok(!classes.has('gpi-preview-focus'));
assert.equal(focusButton['aria-pressed'], 'false');
clickTarget('[data-gpi-preview-focus]', focusButton);
assert.ok(classes.has('gpi-preview-focus'));
assert.equal(host.innerHTML, htmlBeforeToggle);
assert.equal(host.scrollTop, 283);
assert.equal(log.scrollTop, 200, 'Reading history must not be pulled to the bottom');
assert.equal(replacements, 0);
clickTarget('[data-gpi-study-resource]', { dataset: { gpiStudyResource: String(collection.indexOf(pdf)) } });
assert.equal(opened[0].sha256, pdfSha);
preview.setStudyResources(collection, 'project_current', row => opened.push(row));
assert.equal(host.innerHTML, htmlBeforeToggle, 'Shelf updates must not recreate an open report');
preview.open({ ...pdf, run_id: 'run_old' }, 'project_current');
assert.doesNotMatch(host.innerHTML, /data-gpi-study-resource="/);
clickTarget('[data-gpi-study-resource]', { dataset: { gpiStudyResource: '0' } });
assert.equal(opened.length, 1, 'Historical preview cannot use the current-run collection');
preview.open(pdf, 'project_other');
assert.doesNotMatch(host.innerHTML, /data-gpi-study-resource="/);
preview.clearProject();
assert.ok(!classes.has('gpi-preview-open'));
preview.open(pdf, 'project_current');
assert.doesNotMatch(host.innerHTML, /data-gpi-study-resource="/);
preview.close();
assert.equal(study.hidden, false);

// Both the conversation and persistent shelf use the same review receipt path.
require(path.join(jsRoot, 'screens-guided-pi-events.js'));
const receipts = [];
const events = modules.require('events').create({
  state: {}, RESOURCE_OWNER: resources, projectId: () => 'project_current',
  previewWorkflowContext: () => ({ currentRunId: 'run_current' }),
  recordHostAction: (...args) => receipts.push(args),
});
events.openResource(pdf);
assert.deepEqual(receipts, [['review_manuscript', 'run_current:manuscript_revision.pdf']]);
events.openResource({ ...pdf, run_id: '../bad' });
assert.equal(receipts.length, 1, 'Invalid preview must not record a review action');
preview.close();
// The pending-decision link only reveals the existing review card.
let focused = 0;
let scrolled = 0;
const pendingCard = {
  scrollIntoView() { scrolled++; }, setAttribute() {}, focus() { focused++; },
};
const pendingEvents = modules.require('events').create({
  state: { host: { querySelector: () => pendingCard } },
});
pendingEvents.revealPendingReview();
assert.equal(focused, 1);
assert.equal(scrolled, 1);
const panelHead = { innerHTML: '' };
const panelBody = { innerHTML: '', querySelector: () => null };
global.document = { getElementById(id) {
  return id === 'gdStudyAside' ? { querySelector: () => panelHead }
    : id === 'gdAsideBody' ? panelBody : null;
} };
require(path.join(jsRoot, 'screens-guided-pi-aside.js'));
let reveals = 0;
let selections = 0;
const panel = modules.require('aside').create({
  tr: en => en, esc: EU_HTML.esc, iconHtml: () => '',
  projectId: () => 'project_current', displayProjectTitle: value => value,
  demoMode: () => false, shell: () => 'pi', project: () => ({}), workflow: () => workflow,
  resultsHtml: () => outcome.renderShelf(latest, workflow), hasPendingReview: () => true,
  openResource: () => { selections++; }, revealPendingReview: () => { reveals++; },
});
panel.syncProjectWorkflowAside();
assert.equal(reveals, 0, 'Rendering is not approval or navigation');
assert.match(panelBody.innerHTML, /View pending decision/);
panelBody.onclick({ target: { closest: query => query === '[data-gpi-aside-pending]' ? {} : null } });
assert.equal(reveals, 1);
panelBody.onclick({ target: { closest: query => query === '[data-gpi-resource-kind]' ? {} : null } });
assert.equal(selections, 1);
assert.equal(JSON.stringify({ latest, workflow }), original);
process.stdout.write('Current results, revision identity, historical isolation, in-place layout and review receipts passed.\n');

// Compact title chrome leaves model selection in the composer, preserving mode controls.
require(path.join(jsRoot, 'screens-guided-pi-header.js'));
const header = modules.require('header');
const options = { tr: en => en, esc: EU_HTML.esc, icon: () => '', projectTitle: 'Study', sessionTitle: 'Study', connectionLabel: 'provider · model' };
assert.doesNotMatch(header.render(options), /data-gpi-config/);
assert.match(header.render(options), /data-gpi-mode-switch="workspace"/);
assert.match(header.renderModelControl(options), /data-gpi-config/);
assert.match(header.renderModelControl(options), /provider · model/);

// Collapsing history is only a view preference and cannot lose failed rows.
const workspace = modules.require('studyWorkspace').create({ tr: en => en, esc: EU_HTML.esc });
const transcript = '<div>failed attempt</div><div>new response</div>';
assert.ok(workspace.history(transcript, 'p:s', false).includes(transcript));
assert.doesNotMatch(workspace.history(transcript, 'p:s', false), / open>/);
assert.match(workspace.history(transcript, 'p:s', true), / open>/);
workspace.capture({ querySelector: () => ({ dataset: { gpiStudyHistory: 'p:s' }, open: true }) });
assert.match(workspace.history(transcript, 'p:s', false), / open>/);
assert.doesNotMatch(workspace.history(transcript, 'p:other', false), / open>/);
// References are pinned to exact versions and cannot cross projects or sessions.
assert.equal(workspace.setReference(pdf, 'project_current', 'session_a'), true);
assert.match(workspace.renderReference('project_current', 'session_a'), /Current PDF/);
assert.equal(workspace.renderReference('project_current', 'session_b'), '');
assert.equal(workspace.renderReference('project_other', 'session_a'), '');
const question = 'Explain this result';
const message = workspace.decorateMessage(question, 'project_current', 'session_a');
assert.ok(message.startsWith(question));
assert.ok(message.includes(pdfSha));
assert.ok(message.includes('run_current'));
assert.equal(workspace.decorateMessage(question, 'project_other', 'session_a'), question);
assert.equal(workspace.decorateMessage('', 'project_current', 'session_a'), '');
assert.equal(workspace.setReference({ ...pdf, sha256: '' }, 'project_current', 'session_a'), false);
assert.equal(workspace.setReference({ ...pdf, artifact: '../private' }, 'project_current', 'session_a'), false);
workspace.consume('project_other', 'session_a');
assert.notEqual(workspace.renderReference('project_current', 'session_a'), '');
workspace.consume('project_current', 'session_a');
assert.equal(workspace.decorateMessage(question, 'project_current', 'session_a'), question);

// Preview temporarily reveals the context panel and restores the user's choice.
global.document = { getElementById(id) { return id === 'gdContextAside' ? aside : id === 'gdStudyAside' ? study : null; } };
let collapsed = true;
const panelChanges = [];
global.EU_GUIDED_PANELS = {
  isContextAsideCollapsed: () => collapsed,
  setContextAsideCollapsed(value) { collapsed = value; panelChanges.push(value); },
};
preview.open(pdf, 'project_current');
assert.equal(collapsed, false);
preview.open({ ...pdf, artifact: 'manuscript_scaffold.pdf' }, 'project_current');
preview.close();
assert.equal(collapsed, true, 'Switching reports must not overwrite the pre-preview panel preference');
const referenceCalls = [];
preview.setStudyResources(collection, 'project_current', () => {}, { title: 'Current study', reference: (...args) => referenceCalls.push(args) });
preview.open(pdf, 'project_current');
clickTarget('[data-gpi-preview-reference]', {});
assert.equal(referenceCalls[0][0].sha256, pdfSha);
assert.equal(referenceCalls[0][1], 'project_current');
preview.open(pdf, 'project_other');
clickTarget('[data-gpi-preview-reference]', {});
assert.equal(referenceCalls.length, 1, 'An old preview cannot quote into a different project');
preview.close();

// Referencing keeps the user's draft, does not submit, and rejects a stale project.
global.requestAnimationFrame = fn => fn();
const draftInput = { value: 'My unfinished question', focus() {}, scrollIntoView() {} };
const quoteState = { session: { session_id: 'session_a' }, host: { querySelector: () => draftInput }, draft: 'old draft' };
let rendered = 0;
const quoteEvents = modules.require('events').create({ state: quoteState, STUDY_WORKSPACE: workspace,
  projectId: () => 'project_current', render() { rendered++; } });
assert.equal(quoteEvents.referenceResource(pdf, 'project_other'), false);
assert.equal(rendered, 0);
assert.equal(quoteEvents.referenceResource(pdf, 'project_current'), true);
assert.equal(quoteState.draft, 'My unfinished question');
assert.equal(rendered, 1);
assert.match(workspace.decorateMessage('Why?', 'project_current', 'session_a'), new RegExp(pdfSha));
// Startup has no workflow yet; it must render a loading state without result access.
global.document = { getElementById(id) { return id === 'gdStudyAside' ? { querySelector: () => panelHead } : id === 'gdAsideBody' ? panelBody : null; } };
modules.require('aside').create({ tr: en => en, esc: EU_HTML.esc, iconHtml: () => '',
  projectId: () => 'p', displayProjectTitle: value => value, demoMode: () => false,
  shell: () => 'pi', project: () => ({ title: 'Study' }), workflow: () => null,
}).syncProjectWorkflowAside();
assert.match(panelBody.innerHTML, /data-gpi-project-workflow-loading/);
process.stdout.write('Study history, draft retention, reference scope and layout restoration passed.\n');

const displayReference = workspace.messageView({ role: 'user', text: message }, 'project_current');
assert.equal(displayReference.text, question);
assert.equal(displayReference.resources[0].sha256, pdfSha);
assert.equal(workspace.messageView({ role: 'user', text: message }, 'other').text, message);
assert.equal(workspace.messageView({ role: 'assistant', text: message }, 'project_current').text, message);
const badReference = message.replace(pdfSha, 'invalid');
assert.equal(workspace.messageView({ role: 'user', text: badReference }, 'project_current').text, badReference);

// A newer digest of the same artifact must not be labelled as the open version.
preview.setStudyResources(collection, 'project_current', () => {});
preview.open({ ...pdf, sha256: 'c'.repeat(64) }, 'project_current');
const currentPdfTab = host.innerHTML.match(/<button[^>]*data-gpi-study-resource="5"[^>]*>/);
assert.ok(currentPdfTab);
assert.match(currentPdfTab[0], /aria-current="false"/);
preview.close();

inspector = { dataset: { gpiPreviewResource: resources.key(pdf) + ':' + pdfSha }, open: true };
preview.open(pdf, 'project_current');
assert.match(host.innerHTML, /<details class="gpi-preview-inspector"[^>]* open>/);
preview.open({ ...pdf, sha256: 'c'.repeat(64) }, 'project_current');
assert.doesNotMatch(host.innerHTML, /<details class="gpi-preview-inspector"[^>]* open>/);
preview.close();
