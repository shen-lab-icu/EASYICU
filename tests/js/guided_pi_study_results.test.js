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
    ref('result_tables.json'), ref('figure_gallery.json'), ref('quality_gate.json'),
    ref('literature_evidence.json'), ref('scientific_readiness.json'),
    ref('manuscript_scaffold.pdf', 'research_document'),
    { ...ref('manuscript_revision.pdf', 'research_document'), sha256: pdfSha }],
  review_evidence_refs: [{
    reference: 'reviewer_report.json', evidence_id: 'reviewer_report_json',
    sha256: 'e'.repeat(64), kind: 'log', description: 'Registered reviewer report.',
  }],
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
const resultShelf = outcome.renderShelf({ ...latest, artifact_refs: latest.artifact_refs.map(row =>
  row.artifact === 'result_tables.json' ? { ...row, size: 2048 } : row) }, workflow);
assert.match(resultShelf, /data-gpi-results-search/);
assert.match(resultShelf, /result_tables\.json/);
assert.match(resultShelf, /2\.0 KB/);
assert.match(outcome.renderShelf(latest, workflow, 'no-such-file'), /data-gpi-result-file="result_tables\.json" hidden/);
assert.match(outcome.renderShelf(latest, workflow, 'no-such-file'), /No matching files/);
assert.ok(outcome.followUps(latest, workflow).length >= 3);
assert.match(outcome.render(latest, workflow), /data-gpi-followup="0"/);
assert.ok(outcome.selectReviewTab(2));
assert.match(outcome.render(latest, workflow), /data-gpi-review-tab="2" aria-selected="true"/);
assert.equal(outcome.selectReviewTab(8), false);
outcome.selectReviewTab(0);
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
global.location = new URL('http://127.0.0.1:8770/?pi_project=project_current#guided');
global.history = { state: null, replaceState(_state, _title, url) { global.location = new URL(url); } };
global.EU_API = { piCopilotResearchDocumentUrl: (project, run, artifact, digest) => `/preview/${project}/${run}/${artifact}?sha=${digest}` };
require(path.join(jsRoot, 'product-labels.js'));
modules.declare('evidencePreview', {
  kindLabel: () => 'File',
  render: () => '<div>registered evidence</div>',
});
modules.declare('literature', { renderArtifact: () => '' });
require(path.join(jsRoot, 'screens-guided-pi-preview.js'));
const preview = modules.require('preview');
preview.mount(host);
const opened = [];
preview.setStudyResources(collection, 'project_current', row => opened.push(row));
const pdf = collection.find(row => row.artifact === 'manuscript_revision.pdf');
assert.equal(preview.open(pdf, 'project_current', { currentRunId: 'run_current' }), true);
assert.equal(global.location.searchParams.get('pi_view_artifact'), 'manuscript_revision.pdf');
assert.equal(global.location.searchParams.get('pi_view_sha'), pdfSha);
assert.match(host.innerHTML, /This run’s results/);
assert.match(host.innerHTML, /data-gpi-study-resource="0"/);
assert.match(host.innerHTML, new RegExp(pdfSha));
assert.match(host.innerHTML, new RegExp(`href="/preview/project_current/run_current/manuscript_revision.pdf\\?sha=${pdfSha}" download="manuscript_revision.pdf"`));
assert.doesNotMatch(host.innerHTML, /<iframe/);
assert.match(host.innerHTML, /Read the matching article online/);
assert.ok(!classes.has('gpi-preview-focus'), 'Opening a result preserves the conversation by default');
assert.equal(log.scrollTop + log.clientHeight, log.scrollHeight, 'Opening results must follow the bottom through message reflow');
log.scrollTop = 200;
const htmlBeforeToggle = host.innerHTML;
const focusButton = { setAttribute(name, value) { this[name] = value; } };
const clickTarget = (selector, target) => click({ target: { closest: query => query === selector ? target : null } });
clickTarget('[data-gpi-preview-focus]', focusButton);
assert.ok(classes.has('gpi-preview-focus'));
assert.equal(focusButton['aria-pressed'], 'true');
clickTarget('[data-gpi-preview-focus]', focusButton);
assert.ok(!classes.has('gpi-preview-focus'));
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
assert.doesNotMatch(host.innerHTML, /Read the matching article online/);
clickTarget('[data-gpi-study-resource]', { dataset: { gpiStudyResource: '0' } });
assert.equal(opened.length, 1, 'Historical preview cannot use the current-run collection');
preview.open(pdf, 'project_other');
assert.doesNotMatch(host.innerHTML, /data-gpi-study-resource="/);
preview.clearProject();
assert.ok(!classes.has('gpi-preview-open'));
preview.open(pdf, 'project_current');
assert.doesNotMatch(host.innerHTML, /data-gpi-study-resource="/);
preview.close();
assert.equal(global.location.searchParams.has('pi_view'), false);
global.location = new URL(`http://127.0.0.1:8770/?pi_project=project_current&pi_view=artifact&pi_view_kind=research_document&pi_view_run=run_current&pi_view_artifact=manuscript_revision.pdf&pi_view_sha=${pdfSha}#guided`);
assert.equal(preview.restoreFromLocation('project_current', { currentRunId: 'run_current' }), true);
assert.match(host.innerHTML, /manuscript_revision\.pdf/);
preview.close();
global.EU_API.loadPiCopilotResearchArtifact = async () => ({ payload: {}, governance: {} });
global.EU_API.loadPiCopilotResearchEvidence = async () => ({ payload: { renderer: 'metadata' } });
const evidenceSha = 'e'.repeat(64);
global.location = new URL(`http://127.0.0.1:8770/?pi_project=project_current&pi_view=evidence&pi_view_kind=research_artifact&pi_view_run=run_current&pi_view_artifact=scientific_readiness.json&pi_view_sha=${artifactSha}&pi_view_evidence=reviewer_report_json&pi_view_evidence_sha=${evidenceSha}&pi_view_evidence_kind=log#guided`);
assert.equal(preview.restoreFromLocation('project_current', { currentRunId: 'run_current' }), true);
assert.equal(global.location.searchParams.get('pi_view'), 'evidence');
assert.equal(global.location.searchParams.get('pi_view_evidence'), 'reviewer_report_json');
assert.match(host.innerHTML, /data-gpi-evidence-tab="reviewer_report_json"/);
assert.match(host.innerHTML, /aria-label="Close evidence tab"/);
preview.clearProject({ preserveLocation: true });
assert.equal(global.location.searchParams.get('pi_view'), 'evidence');
preview.close();

// Material selection resolves against the current host collection, never a stale row index.
const materialWorkspace = modules.require('studyWorkspace').create({ tr: en => en, esc: EU_HTML.esc, iconHtml: () => '' });
const materialContext = JSON.stringify(['project_current', 'session_a']);
const materialRow = { dataset: { gpiMaterialKey: JSON.stringify([pdf.kind, pdf.run_id, pdf.artifact, pdf.sha256]) } };
const materialMenu = { dataset: { gpiMaterialPicker: materialContext } };
const materialButton = { closest(selector) { return selector === '[data-gpi-material-picker]' ? materialMenu : materialRow; } };
assert.equal(materialWorkspace.selectedMaterial(materialButton, collection, 'project_current', 'session_a').sha256, pdfSha);
assert.equal(materialWorkspace.selectedMaterial(materialButton, collection, 'project_other', 'session_a'), null);
assert.equal(materialWorkspace.selectedMaterial(materialButton, collection, 'project_current', 'session_b'), null);
assert.equal(materialWorkspace.selectedMaterial(materialButton, [{ ...pdf, sha256: 'f'.repeat(64) }], 'project_current', 'session_a'), null);
assert.equal(materialWorkspace.selectedMaterial(materialButton, [], 'project_current', 'session_a'), null);
materialWorkspace.openMaterials('project_current', 'session_a');
assert.doesNotMatch(materialWorkspace.renderMaterials([{ ...pdf, sha256: '' }], 'project_current', 'session_a', false), /data-gpi-material-row/);
assert.match(materialWorkspace.renderMaterials([], 'project_current', 'session_a', false), /No results yet/);
assert.doesNotMatch(materialWorkspace.renderMaterials(collection, 'project_current', 'session_a', true), /data-gpi-material-reference/);
const materialSearch = { value: 'PDF' };
const materialRows = [{ dataset: { gpiMaterialSearchText: 'current pdf manuscript_revision.pdf' } }, { dataset: { gpiMaterialSearchText: 'result tables result_tables.json' } }];
const noMatches = {};
Object.assign(materialMenu, { open: true, querySelector(selector) { return selector === '[data-gpi-material-search]' ? materialSearch : noMatches; }, querySelectorAll() { return materialRows; } });
const materialHost = { querySelector(selector) { return selector === '[data-gpi-material-picker]' ? materialMenu : null; } };
materialWorkspace.filterMaterials(materialHost, 'PDF');
assert.equal(materialRows[0].hidden, false);
assert.equal(materialRows[1].hidden, true);
assert.equal(noMatches.hidden, true);
materialWorkspace.filterMaterials(materialHost, 'not present');
assert.equal(noMatches.hidden, false);
materialWorkspace.openMaterials('project_current', 'session_a');
materialWorkspace.capture(materialHost);
assert.match(materialWorkspace.renderMaterials(collection, 'project_current', 'session_a', false), /value="PDF"/);
assert.doesNotMatch(materialWorkspace.renderMaterials(collection, 'project_current', 'session_b', false), /value="PDF"| open>/);

// Project tasks expose maintenance without making the whole row an ambiguous menu target.
const originalGetElementById = global.document.getElementById;
const taskRail = { hidden: true, innerHTML: '' };
global.document.getElementById = id => id === 'gdConversationRail' ? taskRail : originalGetElementById(id);
materialWorkspace.syncNavigation({
  visible: true, projectId: 'project_current', loading: false, disabled: false,
  sessions: [{ session_id: 'pi_empty', title: 'New task', has_history: false, created_at: '2026-09-21T00:00:00Z' }],
  selectedId: 'pi_empty', title: row => row.title, status: () => 'Not started', time: () => '',
  open: () => {}, create: () => {}, rename: () => {}, remove: () => {}, resources: [],
});
assert.match(taskRail.innerHTML, /data-gpi-rail-rename="pi_empty"/);
assert.match(taskRail.innerHTML, /data-gpi-rail-remove="pi_empty"/);
materialWorkspace.syncNavigation({
  visible: true, projectId: 'project_current', loading: false, disabled: false,
  sessions: [{ session_id: 'pi_history', title: 'Lactate review', has_history: true, last_turn_status: 'done' }],
  selectedId: '', title: row => row.title, status: () => 'Completed', time: () => '',
  open: () => {}, create: () => {}, rename: () => {}, remove: () => {}, resources: [],
});
assert.match(taskRail.innerHTML, /data-gpi-rail-rename="pi_history"/);
assert.doesNotMatch(taskRail.innerHTML, /data-gpi-rail-remove="pi_history"/);
assert.match(materialWorkspace.renderAccessMode('assist', key => key, () => ''), /title="Auto-approve low-risk setup and inspection/);
global.document.getElementById = originalGetElementById;

// Missing/failed checks cannot acquire a positive scientific status through presentation.
const pendingRun = { ...latest, analysis_validated: false, numeric_verified: false };
const pendingHtml = outcome.render(pendingRun, workflow);
assert.match(pendingHtml, /data-gpi-review-tab="0"/);
assert.match(pendingHtml, /Open/);
assert.doesNotMatch(pendingHtml, /is-checked|Execution, evidence binding, and numeric checks completed/);
assert.match(outcome.render({ ...latest, analysis_validated: undefined, numeric_verified: undefined }, workflow), /Unreported/);
assert.match(outcome.render({ ...latest, numeric_verified: true }, workflow), /is-checked/);
const noRecords = outcome.render({ ...pendingRun, artifact_refs: latest.artifact_refs.map(row => ({ ...row, run_id: 'run_old' })) }, workflow);
assert.match(noRecords, /Unreported/);
assert.doesNotMatch(noRecords.match(/<section class="gpi-review-summary"[\s\S]*?<\/section>/)[0], /data-gpi-resource-run/);
assert.equal(JSON.stringify({ latest, workflow }), original);
process.stdout.write('Materials filtering, context and digest isolation, exact PDF download and review status contracts passed.\n');
preview.open({ ...pdf, sha256: '' }, 'project_current');
assert.doesNotMatch(host.innerHTML, / download=|Version pinned|Read the matching article online/);
assert.match(host.innerHTML, /registered document digest is missing/);
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

const workspace = modules.require('studyWorkspace').create({ tr: en => en, esc: EU_HTML.esc, iconHtml: () => '' });
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
assert.doesNotMatch(host.innerHTML, /Read the matching article online/);
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

// Project navigation exposes only this project's sessions and clears during restoration.
const sessionRail = { innerHTML: '', hidden: false };
global.document = { getElementById: id => id === 'gdConversationRail' ? sessionRail : null };
const navigationCalls = [];
const navigation = { projectId: 'p1', visible: true, sessions: [{ session_id: 's1', title: 'Current session',
  last_turn_status: 'done', last_activity_at: '2026-09-18T06:20:20Z' }], selectedId: 's1',
  title: row => row.title, status: () => 'Completed', time: () => '1d',
  open: id => navigationCalls.push(id), create: () => navigationCalls.push('new'),
  resources: [ref('result_tables.json')], openResource: row => navigationCalls.push(row.artifact) };
workspace.syncNavigation(navigation);
assert.match(sessionRail.innerHTML, /aria-current="page"/);
assert.match(sessionRail.innerHTML, /is-complete/);
assert.match(sessionRail.innerHTML, /Completed/);
assert.match(sessionRail.innerHTML, /<time[^>]*>1d<\/time>/);
assert.match(sessionRail.innerHTML, /data-gpi-rail-material="0"/);
workspace.syncNavigation({ ...navigation, materialsInteractive: false });
assert.match(sessionRail.innerHTML, /class="gpi-project-material-row"/);
assert.doesNotMatch(sessionRail.innerHTML, /data-gpi-rail-material="0"/);
workspace.syncNavigation(navigation);
const sessionClick = id => ({ target: { closest: query => query === '[data-gpi-rail-session]' ? { dataset: { gpiRailSession: id } } : null } });
sessionRail.onclick(sessionClick('other-session'));
assert.deepEqual(navigationCalls, []);
sessionRail.onclick(sessionClick('s1'));
assert.deepEqual(navigationCalls, ['s1']);
workspace.syncNavigation({ ...navigation, projectId: 'p2', loading: true });
assert.doesNotMatch(sessionRail.innerHTML, /data-gpi-rail-session/);
sessionRail.onclick(sessionClick('s1'));
assert.deepEqual(navigationCalls, ['s1']);
workspace.syncNavigation({ ...navigation, projectId: 'p2', sessions: [] });
assert.doesNotMatch(sessionRail.innerHTML, /Current session/);

// Empty drafts collapse behind one count so the rail stays readable, and the
// selected conversation is never hidden inside a closed group.
const emptyDraft = id => ({ session_id: id, title: 'New research task' });
const mixedSessions = [emptyDraft('e1'), { ...navigation.sessions[0] }, emptyDraft('e2'), emptyDraft('e3')];
workspace.syncNavigation({ ...navigation, sessions: mixedSessions });
assert.match(sessionRail.innerHTML, /class="gpi-conversation-empty-group" data-gpi-rail-empty-group>/);
assert.match(sessionRail.innerHTML, /class="gpi-conversation-empty-count">3<\/span>/);
assert.ok(sessionRail.innerHTML.indexOf('Current session')
  < sessionRail.innerHTML.indexOf('gpi-conversation-empty-group'), 'completed work stays outside the group');
workspace.syncNavigation({ ...navigation, sessions: mixedSessions, selectedId: 'e2' });
assert.match(sessionRail.innerHTML, /class="gpi-conversation-empty-count">2<\/span>/);
assert.ok(sessionRail.innerHTML.indexOf('data-gpi-rail-session="e2"')
  < sessionRail.innerHTML.indexOf('gpi-conversation-empty-group'), 'the selected draft is pinned, not buried');
workspace.syncNavigation({ ...navigation, sessions: [...navigation.sessions, emptyDraft('e1')] });
assert.doesNotMatch(sessionRail.innerHTML, /gpi-conversation-empty-group/);
assert.match(outcome.render(latest, workflow), /<table class="gpi-deliverables">/);
assert.doesNotMatch(outcome.render(latest, workflow), /gpi-result-shortcut/);

// Progress and results remain independently present on loading, error and ready screens.
global.document = { getElementById: id => id === 'gdStudyAside' ? { querySelector: () => panelHead } : id === 'gdAsideBody' ? panelBody : null };
modules.require('aside').create({ tr: en => en, esc: EU_HTML.esc, iconHtml: () => '', projectId: () => 'p2',
  demoMode: () => false, shell: () => 'pi', workflow: () => null, workflowError: () => 'Progress unavailable',
}).syncProjectWorkflowAside();
assert.match(panelBody.innerHTML, /data-gpi-aside-section="progress"/);
assert.match(panelBody.innerHTML, /data-gpi-aside-section="results"/);
assert.match(panelBody.innerHTML, /Progress unavailable/);
assert.doesNotMatch(panelBody.innerHTML, /Loading project progress/);
assert.equal(panelBody.onclick, null);
process.stdout.write('Workspace restoration, project navigation and response structure passed.\n');

// A composer Skill is selectable only from the immutable conversation snapshot.
const skillDigest = 'e'.repeat(64);
const frozenSession = { session_id: 'session_current', extension_activation: { skills: [
  { name: 'clear-writing', description: 'Keep reports concise.', digest: skillDigest, stages: ['conversation'] },
  { name: 'report-only', description: 'Writing stage.', digest: 'f'.repeat(64), stages: ['writing'] },
] } };
workspace.openSkills('project_current', frozenSession);
assert.match(workspace.renderSkillPicker('project_current', frozenSession, false), /clear-writing/);
assert.doesNotMatch(workspace.renderSkillPicker('project_current', frozenSession, false), /report-only/);
const skillButton = { closest: selector => selector === '[data-gpi-skill-row]'
  ? { dataset: { gpiSkillName: 'clear-writing', gpiSkillDigest: skillDigest } } : null };
assert.equal(workspace.selectSkill(skillButton, 'project_current', frozenSession), true);
assert.match(workspace.renderSkillReference('project_current', frozenSession), /clear-writing/);
assert.match(workspace.decorateSkillMessage('Explain the result.', 'project_current', frozenSession), /easyicu_load_skill/);
// A Skill hint projects back to a readable user question and a distinct chip.
const skillProjection = workspace.messageView({ role: 'user', text: workspace.decorateSkillMessage(
  'Explain the result.', 'project_current', frozenSession) }, 'project_current');
assert.equal(skillProjection.text, 'Explain the result.');
assert.equal(skillProjection.requestedSkill, 'clear-writing');
assert.equal(workspace.decorateSkillMessage('Explain the result.', 'other_project', frozenSession), 'Explain the result.');
assert.equal(workspace.decorateSkillMessage('Explain the result.', 'project_current', { ...frozenSession,
  extension_activation: { skills: [] } }), 'Explain the result.');
workspace.consumeSkill('project_current', frozenSession);
assert.doesNotMatch(workspace.renderSkillReference('project_current', frozenSession), /clear-writing/);
global.EU_CAPABILITIES = { capabilities: { method_skills: {
  items: [{ id: 'table-one', title: 'Table 1', title_zh: '基线表', category: 'Descriptive', category_zh: '描述性分析',
    description: 'Describe a cohort.', description_zh: '描述队列。', prompt: 'Describe the cohort.', prompt_zh: '请描述队列。',
    capability_id: 'descriptive_v1', action_ids: ['table.one'], claim_ceiling: 'analysis_only' }],
  components: [{ id: 'cox', title: 'Cox model', title_zh: 'Cox 模型', category: 'Survival', category_zh: '生存分析',
    description: 'Estimate a hazard ratio.', description_zh: '估计风险比。', prompt: 'Fit a Cox model.', prompt_zh: '请拟合 Cox 模型。',
    method_family: 'time_to_event', method_key: 'cox_hr', claim_ceiling: 'reportable' }],
} } };
workspace.openSkills('project_current', frozenSession);
assert.match(workspace.renderSkillPicker('project_current', frozenSession, false), /data-gpi-catalog-select/);
const methodAction = workspace.selectCatalog({ dataset: { gpiSkillKind: 'method', gpiSkillId: 'table-one' } });
assert.equal(methodAction.text, 'Describe the cohort.');
assert.equal(methodAction.method.capabilityId, 'descriptive_v1');
workspace.openSkills('project_current', frozenSession);
workspace.setSkillCatalog('methods');
assert.match(workspace.renderSkillPicker('project_current', frozenSession, false), /Cox model/);
const componentAction = workspace.selectCatalog({ dataset: { gpiSkillKind: 'method_component', gpiSkillId: 'cox' } });
assert.equal(componentAction.method.methodFamily, 'time_to_event');
const pendingSkillHubIntent = new Map([['easyicu.skillHub.use', JSON.stringify({ name: 'clear-writing', digest: skillDigest })]]);
global.sessionStorage = { getItem: key => pendingSkillHubIntent.get(key) || null,
  removeItem: key => pendingSkillHubIntent.delete(key) };
assert.equal(workspace.hasHubIntent(), true);
workspace.applyHubIntent('project_current', frozenSession);
assert.match(workspace.renderSkillReference('project_current', frozenSession), /clear-writing/);
assert.equal(workspace.hasHubIntent(), false);
workspace.consumeSkill('project_current', frozenSession);
pendingSkillHubIntent.set('easyicu.skillHub.use', JSON.stringify({ name: 'clear-writing', digest: skillDigest }));
workspace.applyHubIntent('project_current', { ...frozenSession, extension_activation: { skills: [] } });
assert.equal(workspace.hasHubIntent(), true, 'A snapshot without this Skill must not consume the intent');
assert.doesNotMatch(workspace.renderSkillReference('project_current', frozenSession), /clear-writing/);
pendingSkillHubIntent.clear();
pendingSkillHubIntent.set('easyicu.skillHub.question', 'How can this skill help?');
assert.equal(workspace.hasHubIntent(), true, 'A built-in skill question also starts a new task');
pendingSkillHubIntent.clear();
pendingSkillHubIntent.set('easyicu.skillHub.builder', JSON.stringify({ mode: 'create', title: 'Skill Builder' }));
pendingSkillHubIntent.set('easyicu.skillHub.question', 'Create a reusable Skill.');
assert.equal(workspace.hasHubIntent(), true, 'A Skill Builder request starts a new task');
workspace.applyHubIntent('project_current', frozenSession);
assert.match(workspace.renderSkillReference('project_current', frozenSession), /Skill Builder/);
assert.match(workspace.renderSkillReference('project_current', frozenSession), /data-gpi-builder-hub/);
assert.equal(pendingSkillHubIntent.has('easyicu.skillHub.builder'), false);
assert.equal(pendingSkillHubIntent.has('easyicu.skillHub.question'), true, 'The editable draft is consumed after task creation');
workspace.removeBuilder();
assert.doesNotMatch(workspace.renderSkillReference('project_current', frozenSession), /Skill Builder/);
pendingSkillHubIntent.clear();
pendingSkillHubIntent.set('easyicu.skillHub.method', JSON.stringify({ id: 'survival-time-to-event',
  title: 'Survival and time-to-event analysis', capability_id: 'survival_time_to_event_v1',
  action_ids: ['time_to_event.cox_hr', 'time_to_event.ph_check'], claim_ceiling: 'reportable' }));
pendingSkillHubIntent.set('easyicu.skillHub.question', 'Use the survival workflow.');
assert.equal(workspace.hasHubIntent(), true, 'A method Skill starts a new task');
workspace.applyHubIntent('project_current', frozenSession);
assert.match(workspace.renderSkillReference('project_current', frozenSession), /Survival and time-to-event analysis/);
assert.doesNotMatch(workspace.renderSkillReference('project_current', frozenSession), /reportable contract|analysis only/);
const methodMessage = workspace.decorateSkillMessage('Analyze time to event.', 'project_current', frozenSession);
assert.match(methodMessage, /EasyICU method workflow request/);
assert.match(methodMessage, /survival_time_to_event_v1/);
const methodProjection = workspace.messageView({ role: 'user', text: methodMessage }, 'project_current');
assert.equal(methodProjection.text, 'Analyze time to event.');
assert.equal(methodProjection.requestedMethod.method_skill_id, 'survival-time-to-event');
assert.equal(workspace.selectFrozenSkill('project_current', frozenSession, 'clear-writing', skillDigest), true);
const combinedProjection = workspace.messageView({ role: 'user', text: workspace.decorateSkillMessage(
  'Analyze and explain.', 'project_current', frozenSession) }, 'project_current');
assert.equal(combinedProjection.text, 'Analyze and explain.');
assert.equal(combinedProjection.requestedSkill, 'clear-writing');
assert.equal(combinedProjection.requestedMethod.method_skill_id, 'survival-time-to-event');
workspace.removeSkill();
workspace.removeMethod();
assert.doesNotMatch(workspace.renderSkillReference('project_current', frozenSession), /Survival and time-to-event analysis/);
pendingSkillHubIntent.delete('easyicu.skillHub.question');

// A digest-bound scientific record supplies the actual review findings; it is
// never synthesized from the presence of a file or a successful code check.
const reviewNode = { dataset: { gpiReviewRun: 'run_current' }, outerHTML: '' };
const detailedOutcome = modules.require('runOutcome').create({
  tr: en => en, esc: EU_HTML.esc, iconHtml: () => '', resourceButton: resources.button,
  projectId: () => 'project_current', host: () => ({ querySelector: () => reviewNode }),
  api: () => ({
    piCopilotResearchDocumentUrl: (_project, _run, artifact, digest) => `/document/${artifact}?sha=${digest}`,
    piCopilotResearchArtifactDownloadUrl: (_project, _run, artifact, digest) => `/artifact/${artifact}?sha=${digest}`,
    loadPiCopilotResearchArtifact: async (_project, run, artifact, digest) => {
      assert.equal(run, 'run_current');
      assert.equal(artifact, 'scientific_readiness.json');
      assert.equal(digest, artifactSha);
      return { ok: true, payload: { run_id: run, domains: [
        { domain: 'analysis', status: 'blocked', summary: 'Prespecified model is incomplete.', evidence_refs: ['reviewer_report.json'] },
      ], findings: [
        { domain: 'analysis', severity: 'blocker', code: 'MODEL_INCOMPLETE', message: 'A required model is missing.', remediation: 'Complete the model.', evidence_refs: ['quality_gate.json'] },
      ] } };
    },
  }),
});
assert.match(detailedOutcome.renderShelf(latest, workflow), /download="manuscript_revision\.pdf"/);
assert.match(detailedOutcome.renderShelf(latest, workflow), /download="result_tables\.review\.json"/);
assert.match(detailedOutcome.renderShelf(latest, workflow), new RegExp(`/artifact/result_tables\\.json\\?sha=${artifactSha}`));
detailedOutcome.loadScientificReview(latest, workflow).then(async () => {
  assert.match(reviewNode.outerHTML, /Prespecified model is incomplete/);
  assert.match(reviewNode.outerHTML, /A required model is missing/);
  assert.match(reviewNode.outerHTML, /Complete the model/);
  assert.match(reviewNode.outerHTML, /MODEL_INCOMPLETE/);
  assert.match(reviewNode.outerHTML, /data-gpi-resource-artifact="quality_gate\.json"/);
  assert.match(reviewNode.outerHTML, /data-gpi-resource-digest="[a-f0-9]{64}"/);
  assert.match(reviewNode.outerHTML, /data-gpi-evidence-open/);
  assert.match(reviewNode.outerHTML, /data-evidence-id="reviewer_report_json"/);
  assert.match(reviewNode.outerHTML, new RegExp(`data-evidence-sha256="${'e'.repeat(64)}"`));
  assert.doesNotMatch(detailedOutcome.render({ ...latest, run_id: 'run_other' }, workflow), /Prespecified model is incomplete/);
  const zhReviewNode = { dataset: { gpiReviewRun: 'run_current' }, outerHTML: '' };
  const zhOutcome = modules.require('runOutcome').create({
    tr: (_en, zh) => zh, esc: EU_HTML.esc, iconHtml: () => '', resourceButton: resources.button,
    projectId: () => 'project_current', host: () => ({ querySelector: () => zhReviewNode }),
    api: () => ({ loadPiCopilotResearchArtifact: async () => ({ ok: true, payload: {
      run_id: 'run_current',
      domains: [{ domain: 'analysis', status: 'blocked', summary: 'Prespecified model is incomplete.', evidence_refs: ['quality_gate.json'] }],
      findings: [{ domain: 'analysis', severity: 'blocker', code: 'MODEL_INCOMPLETE', message: 'A required model is missing.', remediation: 'Complete the model.', evidence_refs: ['quality_gate.json'] }],
    } }) }),
  });
  await zhOutcome.loadScientificReview(latest, workflow);
  assert.match(zhReviewNode.outerHTML, /预先设定的模型尚未完成/);
  assert.match(zhReviewNode.outerHTML, /预设模型尚未完成/);
  assert.match(zhReviewNode.outerHTML, /缺少一项必需模型/);
  assert.match(zhReviewNode.outerHTML, /完成该模型并重新审阅/);
  assert.match(zhReviewNode.outerHTML, /data-review-code="MODEL_INCOMPLETE"/);
  assert.doesNotMatch(zhReviewNode.outerHTML, /记录代码/);
  process.stdout.write('Digest-bound scientific review and PDF download passed.\n');
}).catch(error => { console.error(error); process.exitCode = 1; });
