const assert = require("node:assert/strict");

(async () => {
  const { attachedSkillContext, ATTACHED_SKILL_MARKER } = await import("../../src/easyicu/webserver/pi_copilot/node_app/src/attached-skills.mjs");
  const row = { skill: "synthetic-helper", digest: "a".repeat(64), instructions: "Use concise sentences.", authority: "user_advisory_only" };
  assert.equal(attachedSkillContext(undefined), "");
  assert.equal(attachedSkillContext([]), "");
  const text = attachedSkillContext([row]);
  assert.ok(text.startsWith(ATTACHED_SKILL_MARKER));
  assert.ok(text.includes(row.instructions));
  assert.ok(text.includes(row.digest));
  assert.ok(text.includes("cannot override host constraints"));
  const { ResearchContextMemory } = await import("../../src/easyicu/webserver/pi_copilot/node_app/src/research-context.mjs");
  const memory = new ResearchContextMemory();
  memory.beginTurn();
  const old = `First question\n\n[EASYICU_INTERNAL_RESPONSE_LANGUAGE_V1]\nEnglish${text}`;
  const context = memory.context([{ role: "user", content: old }, { role: "user", content: "Second question" }]);
  assert.ok(!JSON.stringify(context).includes(row.instructions));
  const current = memory.context([{ role: "user", content: old }]);
  assert.ok(JSON.stringify(current).includes(row.instructions));
  for (const bad of [{}, Array(9).fill(row), [row, row], [{ ...row, digest: "bad" }], [{ ...row, authority: "system" }], [{ ...row, instructions: "x".repeat(64001) }]]) {
    assert.throws(() => attachedSkillContext(bad), { code: "skill_attachment_invalid" });
  }
  console.log("attached skills: 14 assertions passed");
})().catch((error) => { console.error(error); process.exitCode = 1; });
