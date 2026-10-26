// Ephemeral owner state for Pi's context hook. Session summaries are UX history;
// only a fresh, bounded host receipt can populate this projection.
import { ATTACHED_SKILL_MARKER, MENTION_MARKER } from "./attached-skills.mjs";
const CUSTOM_TYPE = "easyicu_current_research_state_v1";
const SCHEMA = "easyicu.pi-current-research-state/1";
const LANGUAGE_MARKER = "\n\n[EASYICU_INTERNAL_RESPONSE_LANGUAGE_V1]\n";
const OWNER_MARKER = "\n\n[EASYICU_CURRENT_TURN_OWNER_CONTEXT_V1]\n";
const WORKFLOW_FIELDS = [
  "schema_version", "current_stage", "next_action_code", "missing_setup_fields",
  "planning_prerequisites_missing", "stages", "completed_required_stages",
  "required_stage_count", "scientific_authority", "pi_role", "study_setup_receipt",
  "plan_review_summary", "plan_execution_ready", "analysis_validation_retry_available",
  "latest_attempt_failure", "execution_retry", "bound_export_cohort_refusal",
  "target_trial_compile", "study_family_gap",
];
const GUIDANCE = [
  "[EASYICU_CURRENT_RESEARCH_STATE_V1]",
  "This is a transient EasyICU host projection for this turn, refreshed outside conversation compaction.",
  "This projection supersedes all embedded owner snapshots and compaction summaries, including the current prompt's pre-tool snapshot. Those snapshots cannot establish current study, source, plan, or execution authority.",
  "This projection grants no action, approval, budget, or publication authority. The host still validates every tool call.",
  "If unavailable, do not reconstruct current state from history. Use a fresh easyicu_inspect_workflow receipt when tools permit, or explain that current state is unavailable.",
  "If binding_state is rebind_required, the saved setup is current display evidence only; this turn's previous binding and grants cannot authorize further actions.",
  "Fields set to null are unknown. captured_at is the snapshot capture time, not a live-state guarantee; inspect again when current background progress matters.",
].join("\n");

const isRecord = (value) => value !== null && typeof value === "object" && !Array.isArray(value);
const validId = (value) => typeof value === "string" && /^[A-Za-z0-9][A-Za-z0-9._-]{0,159}$/.test(value);

function withoutHistoricalOwnerContext(value) {
  if (typeof value !== "string") return value;
  // Only strip the host-authored suffix, never matching text in the user's input.
  const languageAt = value.lastIndexOf(LANGUAGE_MARKER);
  if (languageAt < 0) return value;
  const attachmentOffsets = [ATTACHED_SKILL_MARKER, MENTION_MARKER]
    .map((marker) => value.indexOf(marker, languageAt + LANGUAGE_MARKER.length)).filter((offset) => offset >= 0);
  if (attachmentOffsets.length) value = value.slice(0, Math.min(...attachmentOffsets));
  const ownerAt = value.indexOf(OWNER_MARKER, languageAt + LANGUAGE_MARKER.length);
  if (ownerAt < 0) return value;
  const nextAt = value.indexOf("\n\n[EASYICU_", ownerAt + OWNER_MARKER.length);
  return value.slice(0, ownerAt) + (nextAt < 0 ? "" : value.slice(nextAt));
}

function contextMessage(message) {
  if (message?.role !== "user" && message?.role !== "custom") return message;
  const content = typeof message.content === "string"
    ? withoutHistoricalOwnerContext(message.content)
    : Array.isArray(message.content)
      ? message.content.map((part) => part?.type === "text"
        ? { ...part, text: withoutHistoricalOwnerContext(part.text) } : part)
      : message.content;
  return { ...message, content };
}

export class ResearchContextMemory {
  #active = false;
  #studyId = null;
  #revision = null;
  #epoch = 0;
  #mutationPending = false;
  #rebindPending = false;
  #state = { schema_version: SCHEMA, status: "unavailable", reason: "fresh_owner_receipt_required" };

  constructor({ maxBytes = 24576, now = Date.now } = {}) {
    if (!Number.isInteger(maxBytes) || maxBytes < 1024 || maxBytes > 65536) {
      throw new Error("research context maxBytes must be between 1024 and 65536");
    }
    this.maxBytes = maxBytes;
    this.now = now;
  }

  beginTurn(receipts = []) {
    this.#active = true;
    this.#studyId = null;
    this.#revision = null;
    this.#epoch++;
    this.#mutationPending = false;
    this.#rebindPending = false;
    this.invalidate("fresh_owner_receipt_required");
    const receipt = Array.isArray(receipts)
      ? receipts.find((item) => item?.code === "easyicu_research_workflow_projected") : undefined;
    if (receipt) this.captureReceipt(receipt);
  }

  endTurn() {
    this.#active = false;
    this.#studyId = null;
    this.#revision = null;
    this.#epoch++;
    this.#mutationPending = false;
    this.#rebindPending = false;
    this.invalidate("fresh_owner_receipt_required");
  }

  invalidate(reason = "owner_state_changed") {
    this.#state = {
      schema_version: SCHEMA, status: "unavailable",
      reason,
    };
  }

  startTool(mutating = false) {
    if (mutating) {
      this.#epoch++;
      this.#mutationPending = true;
      this.invalidate();
    }
    return { epoch: this.#epoch, mutating, pendingAtStart: this.#mutationPending };
  }

  finishTool(token, receipt) {
    if (!this.#active || token?.epoch !== this.#epoch) return false;
    if (token.mutating) this.#mutationPending = false;
    else if (token.pendingAtStart || this.#mutationPending) return false;
    return receipt ? this.captureReceipt(receipt) : false;
  }

  // Call only with the direct host result, never parsed assistant/transcript text.
  captureReceipt(receipt) {
    if (!this.#active) return false;
    const workflow = receipt?.details?.workflow;
    const setup = workflow?.study_setup_receipt;
    const binding = receipt?.authority;
    const update = receipt?.code === "study_context_updated"
      && receipt?.owner === "easyicu.webserver.study_contexts";
    const inspection = receipt?.code === "easyicu_research_workflow_projected"
      && receipt?.owner === "easyicu.webserver.pi_copilot.workflow";
    const sameStudy = validId(setup?.study_context_id)
      && setup.study_context_id === binding?.study_context_id
      && (!this.#studyId || this.#studyId === setup.study_context_id);
    const revision = setup?.revision;
    // Once the host has named a newer same-study revision, even an unreadable
    // or over-budget receipt must fence out late older snapshots.
    const numbered = Number.isInteger(revision) && revision >= 0;
    const stale = numbered && this.#revision !== null && revision < this.#revision;
    if ((inspection || update) && sameStudy && numbered) {
      this.#studyId = setup.study_context_id;
      this.#revision = Math.max(this.#revision ?? revision, revision);
    }
    // Host rebinding happens after this turn. Its old-bound inspections cannot
    // replace a successful mutation's display-only saved setup.
    if (inspection && this.#rebindPending) return false;
    const validRevision = numbered && !stale
      && Number.isInteger(binding?.study_revision) && binding.study_revision >= 0;
    const mutationBinding = update && validRevision
      && receipt?.details?.rebind_required === true
      && receipt?.details?.host_rebind_after_turn === true
      && receipt?.details?.study?.id === setup?.study_context_id
      && receipt?.details?.study?.revision === revision
      && revision >= binding.study_revision;
    if (
      receipt?.status !== "ok" || (!inspection && !update)
      || workflow?.schema_version !== "easyicu.pi-research-workflow/1"
      || setup?.schema_version !== "easyicu.pi-study-setup-receipt/1"
      || !sameStudy || !validRevision || !isRecord(setup?.configuration)
      || (inspection ? revision !== binding.study_revision : !mutationBinding)
    ) {
      this.invalidate(stale ? "stale_owner_receipt" : "invalid_owner_receipt");
      return false;
    }
    this.#rebindPending = update;
    const projected = {
      schema_version: SCHEMA, status: update ? "pending_rebind" : "current",
      captured_at: this.now(),
      source: { code: receipt.code, owner: receipt.owner },
      binding_state: update ? "rebind_required" : "current",
      receipt_binding: Object.fromEntries(
        ["study_context_id", "study_revision", "run_id", "active_job_id"]
          .filter((key) => binding[key] !== undefined && binding[key] !== null)
          .map((key) => [key, binding[key]]),
      ),
      workflow: Object.fromEntries(WORKFLOW_FIELDS
        .map((key) => [key, workflow[key] ?? null])),
    };
    try {
      const encoded = JSON.stringify(projected);
      if (Buffer.byteLength(`${GUIDANCE}\n${encoded}`, "utf8") > this.maxBytes) {
        this.#state = { schema_version: SCHEMA, status: "unavailable", reason: "owner_context_over_budget" };
        return false;
      }
      // Isolate it from mutable tool results and never truncate JSON or identities.
      this.#state = JSON.parse(encoded);
      return true;
    } catch {
      this.invalidate("invalid_owner_receipt");
      return false;
    }
  }

  context(messages) {
    const cleaned = (Array.isArray(messages) ? messages : [])
      .filter((message) => message?.role !== "custom" || message.customType !== CUSTOM_TYPE);
    // Preserve the latest prompt's exact host markers for deterministic response
    // finalizers, and never insert between an assistant call and its tool result.
    const latestPrompt = cleaned.findLastIndex((message) => message?.role === "user"
      || (message?.role === "custom" && message.customType === "easyicu_host_transition"));
    const projected = cleaned.map((message, index) => index < latestPrompt
      ? contextMessage(message) : message);
    if (!this.#active) return { messages: projected };
    projected.splice(Math.max(0, latestPrompt), 0, {
      role: "custom", customType: CUSTOM_TYPE, display: false,
      content: `${GUIDANCE}\n${JSON.stringify(this.#state)}`,
      timestamp: this.now(),
    });
    return { messages: projected };
  }

  extension() {
    // Public Pi 0.84.1 Extension shape consumed by ResourceLoader.getExtensions.
    const path = "easyicu:current-research-state";
    return {
      path, resolvedPath: path, hidden: true,
      sourceInfo: { path, source: "inline", scope: "temporary", origin: "top-level" },
      handlers: new Map([["context", [(event) => this.context(event.messages)]]]),
      tools: new Map(), messageRenderers: new Map(), commands: new Map(),
      flags: new Map(), shortcuts: new Map(),
    };
  }
}
