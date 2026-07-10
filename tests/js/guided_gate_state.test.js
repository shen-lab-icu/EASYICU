'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

function extractFunction(source, name) {
  const start = source.indexOf(`function ${name}(`);
  assert.notEqual(start, -1, `${name} must exist`);
  const bodyStart = source.indexOf('{', start);
  let depth = 0;
  for (let i = bodyStart; i < source.length; i += 1) {
    if (source[i] === '{') depth += 1;
    if (source[i] === '}') {
      depth -= 1;
      if (depth === 0) return source.slice(start, i + 1);
    }
  }
  throw new Error(`Could not extract ${name}`);
}

const source = fs.readFileSync(path.resolve(process.argv[2]), 'utf8');
const sandbox = {};
vm.runInNewContext(`${extractFunction(source, 'guidedGateState')}; this.gateState = guidedGateState;`, sandbox);
const gateState = sandbox.gateState;

const validChecks = [
  'source_valid',
  'denominator_resolved',
  'quality_audited',
  'no_bad_non_event_coverage',
  'no_patient_rows_persisted',
].map(id => ({ id, label: id, passed: true }));
validChecks.push({ id: 'human_signoff', label: 'Human sign-off', passed: false });

function validResult() {
  return {
    run_type: 'preflight',
    gate: {
      status: 'analysis_only',
      reportable: false,
      draft_unlocked: false,
      reason: 'preflight_complete_human_signoff_required',
      checks: validChecks.map(check => ({ ...check })),
    },
  };
}

assert.equal(gateState(null).blocked, true);
assert.equal(gateState({}).blocked, true);
assert.equal(gateState({ gate: { status: 'unknown' } }).blocked, true);
assert.equal(gateState({ run_type: 'preflight', gate: { status: 'analysis_only' } }).blocked, true);
assert.equal(gateState(validResult()).blocked, false, 'only the complete analysis-only preflight contract may pass');

const failedCheck = validResult();
failedCheck.gate.checks[1].passed = false;
assert.equal(gateState(failedCheck).blocked, true);

const wrongReason = validResult();
wrongReason.gate.reason = 'unexpected';
assert.equal(gateState(wrongReason).blocked, true);

const blocked = validResult();
blocked.gate.status = 'blocked';
assert.equal(gateState(blocked).blocked, true);

process.stdout.write(JSON.stringify({ ok: true, cases: 8 }));
