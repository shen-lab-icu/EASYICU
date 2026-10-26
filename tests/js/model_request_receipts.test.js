const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const crypto = require("node:crypto");

test("conversation attempts persist only closed metadata, including retries", async () => {
  const { appendModelRequestReceipt } = await import("../../src/easyicu/webserver/pi_copilot/node_app/src/shell-budget.mjs");
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "easyicu-egress-"));
  try {
    const context = { systemPrompt: "SYNTHETIC_PRIVATE", messages: [{ content: "文" }], tools: [] };
    for (let i = 0; i < 2; i++) {
      appendModelRequestReceipt(root, "session", { provider: "openai", id: "SYNTHETIC_PRIVATE" }, context);
    }
    const file = path.join(root, ".egress", crypto.createHash("sha256").update("session").digest("hex") + ".jsonl");
    const raw = fs.readFileSync(file, "utf8");
    const rows = raw.trim().split("\n").map(JSON.parse);
    assert.equal(rows.length, 2);
    assert.equal(rows[0].provider, "openai");
    assert.equal(rows[0].byte_size, Buffer.byteLength(JSON.stringify(context), "utf8"));
    assert.equal(rows[0].redaction_status, "not_recorded");
    assert.deepEqual(rows[0].data_classes, ["text"]);
    assert.ok(!raw.includes("SYNTHETIC_PRIVATE"));
    assert.ok(!raw.includes("文"));
    appendModelRequestReceipt(root, "session", { provider: "SYNTHETIC_PRIVATE" }, { tools: [{ name: "tool" }] });
    const last = JSON.parse(fs.readFileSync(file, "utf8").trim().split("\n").at(-1));
    assert.equal(last.provider, "unknown");
    assert.deepEqual(last.data_classes, ["text", "schema"]);
  } finally { fs.rmSync(root, { recursive: true, force: true }); }
});

test("receipt storage refuses symlinks before any append", async () => {
  const { appendModelRequestReceipt } = await import("../../src/easyicu/webserver/pi_copilot/node_app/src/shell-budget.mjs");
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "easyicu-egress-"));
  try {
    fs.mkdirSync(path.join(root, "other"));
    fs.symlinkSync(path.join(root, "other"), path.join(root, ".egress"));
    assert.throws(() => appendModelRequestReceipt(root, "session", {}, {}));
    assert.equal(fs.readdirSync(path.join(root, "other")).length, 0);
  } finally { fs.rmSync(root, { recursive: true, force: true }); }
});

test("receipt files cannot alias another file or grow without a bound", async () => {
  const { appendModelRequestReceipt } = await import("../../src/easyicu/webserver/pi_copilot/node_app/src/shell-budget.mjs");
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "easyicu-egress-"));
  try {
    fs.mkdirSync(path.join(root, ".egress"));
    const file = path.join(root, ".egress", crypto.createHash("sha256").update("session").digest("hex") + ".jsonl");
    const other = path.join(root, "other");
    fs.writeFileSync(other, "unchanged");
    fs.symlinkSync(other, file);
    assert.throws(() => appendModelRequestReceipt(root, "session", {}, {}));
    assert.equal(fs.readFileSync(other, "utf8"), "unchanged");
    fs.unlinkSync(file);
    fs.writeFileSync(file, "");
    fs.truncateSync(file, 8 * 1024 * 1024);
    assert.throws(() => appendModelRequestReceipt(root, "session", {}, {}));
    assert.equal(fs.statSync(file).size, 8 * 1024 * 1024);
  } finally { fs.rmSync(root, { recursive: true, force: true }); }
});
