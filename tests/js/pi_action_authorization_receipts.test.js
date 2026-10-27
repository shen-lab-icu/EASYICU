const assert = require('node:assert/strict');
const { test } = require('node:test');
const { readFileSync } = require('node:fs');
const { resolve } = require('node:path');

const projection = import('../../src/easyicu/webserver/pi_copilot/node_app/src/event-projection.mjs');
const actions = [
  'configure', 'idea', 'literature', 'extract', 'run', 'provider_run',
  'report_revision', 'cancel', 'workspace_write', 'mcp_read',
];

function receipts(ownerDetails, envelope = {}) {
  const result = { details: {
    status: 'blocked', code: 'pi_action_authorization_required',
    owner: 'easyicu.webserver.pi_copilot', ...envelope, details: ownerDetails,
  } };
  return projection.then(({ normalizePiEvent, projectTranscriptMessage }) => [
    normalizePiEvent({ type: 'tool_execution_end', toolCallId: 'call-synthetic', result }),
    projectTranscriptMessage({ role: 'toolResult', toolCallId: 'call-synthetic', ...result }).content[0],
  ]);
}

test('bridge action names match the host allowlist', () => {
  const service = readFileSync(resolve(__dirname, '../../src/easyicu/webserver/pi_copilot/service.py'), 'utf8');
  const declaration = service.match(/ALLOWED_TURN_ACTIONS = frozenset\(([\s\S]*?)\n\)/)[1];
  assert.deepEqual([...declaration.matchAll(/"([a-z_]+)"/g)].map(match => match[1]).sort(), [...actions].sort());
});

for (const action of actions) {
  for (const consumed of [false, true]) {
    test(`live and transcript ${consumed ? 'consumed' : 'missing'} receipts retain ${action}`, async () => {
      const grantable = !['extract', 'provider_run', 'report_revision'].includes(action);
      for (const receipt of await receipts(
        { required_action: action, grantable_by_turn: grantable },
        consumed ? { code: 'pi_action_grant_consumed' } : {},
      )) {
        assert.equal(receipt.required_action, action);
        assert.equal(receipt.grantable_by_turn, grantable);
        assert.equal(receipt.code, consumed ? 'pi_action_grant_consumed' : 'pi_action_authorization_required');
        assert.equal(receipt.status, 'blocked');
        assert.equal(receipt.is_error, true);
      }
    });
  }
}

test('unknown action names and non-boolean grantability are dropped', async () => {
  for (const action of ['unknown_action', ' configure', 'CONFIGURE', {}, ['run'], null]) {
    for (const receipt of await receipts({ required_action: action, grantable_by_turn: true })) {
      assert.equal(Object.hasOwn(receipt, 'required_action'), false);
      assert.equal(receipt.grantable_by_turn, true);
    }
  }
  for (const value of ['false', 'true', 0, 1, null, {}]) {
    for (const receipt of await receipts({ required_action: 'run', grantable_by_turn: value })) {
      assert.equal(receipt.required_action, 'run');
      assert.equal(Object.hasOwn(receipt, 'grantable_by_turn'), false);
    }
  }
});

test('legacy receipts omit both fields and ignore model-authored envelope fields', async () => {
  for (const receipt of await receipts({}, { required_action: 'run', grantable_by_turn: true })) {
    assert.equal(Object.hasOwn(receipt, 'required_action'), false);
    assert.equal(Object.hasOwn(receipt, 'grantable_by_turn'), false);
  }
});
