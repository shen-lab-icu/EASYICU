/* A job the server has forgotten must be reported as gone, not as a dropped
   connection the user can reconnect to.

   Both continuity modules detected that case with /HTTP\s+404\b/ against
   error.message. api.js only puts "path -> HTTP 404" in .message when there is
   no human reason; /api/jobs/... answers 404 with detail="unknown job", which
   api.js promotes to the message. So the regex never matched and the missing
   branch was dead. This test reproduces apiError's real behaviour rather than
   asserting on source text. */
const assert = require('assert');
const fs = require('fs');
const path = require('path');

const JS_DIR = path.join(__dirname, '..', '..', 'src', 'easyicu', 'webserver', 'static', 'js');

// Verbatim from api.js — kept in sync by the assertion below.
function apiError(reqPath, res, d) {
  const technical = reqPath + ' -> HTTP ' + res.status;
  const reason = d && typeof d === 'object' ? (d.reason || '') : '';
  const code = d && typeof d === 'object' ? (d.error || '') : (typeof d === 'string' ? d : '');
  const human = reason || code;
  const err = new Error(res.status < 500 && human ? human : technical + (human ? ' · ' + human : ''));
  err.technical = technical; err.status = res.status; err.code = code || null;
  return err;
}

const apiSrc = fs.readFileSync(path.join(JS_DIR, 'api.js'), 'utf8');
assert.ok(apiSrc.includes('err.technical = technical; err.status = res.status;'),
  'api.js changed how it builds errors; update this test copy of apiError');

// What /api/jobs/<id> actually returns for a job it no longer has.
const err = apiError('/api/jobs/abc', { status: 404 }, 'unknown job');

assert.strictEqual(err.message, 'unknown job');
assert.strictEqual(err.status, 404);
assert.ok(!/HTTP\s+404\b/.test(String(err.message)),
  'the old detection could not have matched — that was the bug');
assert.ok(/HTTP\s+404\b/.test(err.technical), 'the transport string keeps the code');

for (const file of ['screens-extraction-job-continuity.js', 'screens-viz-crossdb-job-continuity.js']) {
  const src = fs.readFileSync(path.join(JS_DIR, file), 'utf8');
  assert.ok(!/\/HTTP\\s\+404\\b\/\.test/.test(src),
    `${file} still matches the transport string instead of error.status`);
  assert.ok(src.includes('error.status === 404'),
    `${file} must key the missing-job branch on error.status`);
}

console.log('ok - job continuity keys the missing-job branch on error.status');
