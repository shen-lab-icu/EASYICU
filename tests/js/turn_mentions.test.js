const assert = require("node:assert/strict");

(async () => {
  const { mentionedObjectContext, MENTION_MARKER } = await import("../../src/easyicu/webserver/pi_copilot/node_app/src/attached-skills.mjs");
  const { ResearchContextMemory } = await import("../../src/easyicu/webserver/pi_copilot/node_app/src/research-context.mjs");
  const row = { reference: { kind: "concept", id: "synthetic_measurement", digest: "a".repeat(64), role: "exposure" }, object: { kind: "concept", concept_id: "synthetic_measurement" } };
  assert.equal(mentionedObjectContext(undefined), "");
  assert.equal(mentionedObjectContext([]), "");
  const text = mentionedObjectContext([row]);
  assert.ok(text.startsWith(MENTION_MARKER));
  assert.ok(text.includes(row.reference.digest));
  assert.ok(text.includes("never instructions"));
  const memory = new ResearchContextMemory();
  memory.beginTurn();
  const original = `Synthetic question\n\n[EASYICU_INTERNAL_RESPONSE_LANGUAGE_V1]\nEnglish${text}`;
  assert.ok(JSON.stringify(memory.context([{ role: "user", content: original }])).includes(row.reference.id));
  assert.ok(!JSON.stringify(memory.context([{ role: "user", content: original }, { role: "user", content: "Next question" }])).includes(row.reference.id));
  for (const bad of [{}, Array(17).fill(row), [{ reference: row.reference, object: { kind: "file" } }], [{ ...row, reference: { ...row.reference, digest: "stale" } }], [{ ...row, object: { ...row.object, text: "x".repeat(64000) } }]]) {
    assert.throws(() => mentionedObjectContext(bad), { code: "mention_invalid" });
  }
  console.log("turn mentions: 12 assertions passed");
})().catch((error) => { console.error(error); process.exitCode = 1; });
