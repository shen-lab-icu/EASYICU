'use strict';

/* Node-only compatibility harness for owner files tested outside index.html.
   Production loads screens-guided-pi-modules.js first. These isolated tests
   replace `window` repeatedly, so the setter installs a fresh equivalent
   registry and exposes the former names only to their legacy assertions. */

const LEGACY_NAMES = {
  EU_GUIDED_PI: 'shell',
  EU_GUIDED_PI_ACTIVITY: 'activity',
  EU_GUIDED_PI_ANALYSIS_REPORT: 'analysisReport',
  EU_GUIDED_PI_ARTICLE_REPORT: 'articleReport',
  EU_GUIDED_PI_ASIDE: 'aside',
  EU_GUIDED_PI_CHILDJOB: 'childJob',
  EU_GUIDED_PI_COHORT_ELIGIBILITY: 'cohortEligibility',
  EU_GUIDED_PI_CONFIRMATION: 'confirmation',
  EU_GUIDED_PI_DATA_BINDING: 'dataBinding',
  EU_GUIDED_PI_DATA_CONSENT: 'dataConsent',
  EU_GUIDED_PI_DATA_PREVIEW: 'dataPreview',
  EU_GUIDED_PI_DEMO: 'demo',
  EU_GUIDED_PI_EVENTS: 'events',
  EU_GUIDED_PI_EVIDENCE_PREVIEW: 'evidencePreview',
  EU_GUIDED_PI_HEADER: 'header',
  EU_GUIDED_PI_IDEA_SOURCE: 'ideaSource',
  EU_GUIDED_PI_LITERATURE: 'literature',
  EU_GUIDED_PI_MARKDOWN: 'markdown',
  EU_GUIDED_PI_MESSAGE_ACTIONS: 'messageActions',
  EU_GUIDED_PI_NEXT_ACTIONS: 'nextActions',
  EU_GUIDED_PI_PLAN_ACTIONS: 'planActions',
  EU_GUIDED_PI_PREVIEW: 'preview',
  EU_GUIDED_PI_PROJECT: 'project',
  EU_GUIDED_PI_PROVIDER: 'provider',
  EU_GUIDED_PI_PROVIDER_CONTROL: 'providerControl',
  EU_GUIDED_PI_REGENERATION: 'regeneration',
  EU_GUIDED_PI_REPORT_ARTIFACTS: 'reportArtifacts',
  EU_GUIDED_PI_REPLAY: 'replay',
  EU_GUIDED_PI_RESULT_SUMMARY: 'resultSummary',
  EU_GUIDED_PI_RESOURCES: 'resources',
  EU_GUIDED_PI_RUN_OUTCOME: 'runOutcome',
  EU_GUIDED_PI_STARTERS: 'starters',
  EU_GUIDED_PI_TECHNICAL_REPORT: 'technicalReport',
  EU_GUIDED_PI_TRANSCRIPT: 'transcript',
  EU_GUIDED_PI_WORKBENCH_PREVIEW: 'workbenchPreview',
};

function augment(target) {
  if (!target || (typeof target !== 'object' && typeof target !== 'function')) return target;
  const declared = Object.create(null);
  target.EasyICU = target.EasyICU || {};
  target.EasyICU.guidedPi = {
    declare(name, api) {
      if (Object.prototype.hasOwnProperty.call(declared, name)) throw new Error(`duplicate ${name}`);
      declared[name] = api;
      return api;
    },
    require(name) {
      if (!Object.prototype.hasOwnProperty.call(declared, name)) throw new Error(`missing ${name}`);
      return declared[name];
    },
    optional(name) {
      return Object.prototype.hasOwnProperty.call(declared, name) ? declared[name] : null;
    },
  };
  Object.entries(LEGACY_NAMES).forEach(([legacy, name]) => {
    const existing = target[legacy];
    if (existing != null) declared[name] = existing;
    Object.defineProperty(target, legacy, {
      configurable: true,
      get: () => target.EasyICU.guidedPi.optional(name),
      set: value => { declared[name] = value; },
    });
  });
  return target;
}

let currentWindow;
Object.defineProperty(global, 'window', {
  configurable: true,
  get: () => currentWindow,
  set: value => { currentWindow = augment(value); },
});
