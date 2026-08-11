import { randomUUID } from "node:crypto";
import { mkdirSync, mkdtempSync, realpathSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, isAbsolute, join, relative, resolve, sep } from "node:path";
import { createInterface } from "node:readline";

import { InMemoryCredentialStore, lazyStream } from "@earendil-works/pi-ai";
import {
  createAgentSession,
  createExtensionRuntime,
  defineTool,
  ModelRuntime,
  SessionManager,
  SettingsManager,
} from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";

import { normalizePiEvent, projectTranscriptMessage } from "./event-projection.mjs";
import {
  providerCallReceipt,
  restoredProviderCallCount,
  SHELL_BUDGET_RECEIPT,
  ShellBudgetGuard,
} from "./shell-budget.mjs";

const PROTOCOL_VERSION = "easyicu.pi-copilot/1";
const MAX_LINE_BYTES = 1024 * 1024;
const MAX_TEXT_CHARS = 12000;
const SESSION_DIR = resolve(
  process.env.EASYICU_PI_SESSION_DIR || join(process.cwd(), ".easyicu-pi-sessions"),
);
const CWD = resolve(process.env.EASYICU_PI_CWD || process.cwd());
const RESEARCH_TOOL_NAMES = Object.freeze([
  "easyicu_workspace_status",
  "easyicu_inspect_workflow",
  "easyicu_inspect_context",
  "easyicu_inspect_plan",
  "easyicu_inspect_literature",
  "easyicu_inspect_capability",
  "easyicu_inspect_run",
  "easyicu_inspect_step",
  "easyicu_inspect_validation",
  "easyicu_list_artifacts",
  "easyicu_inspect_evidence",
  "easyicu_explain_blocker",
  "easyicu_inspect_interpretation",
  "easyicu_inspect_manuscript",
  "easyicu_update_study_context",
  "easyicu_mine_ideas",
  "easyicu_search_literature",
  "easyicu_prepare_idea_handoff",
  "easyicu_accept_idea_handoff",
  "easyicu_start_extraction",
  "easyicu_run",
  "easyicu_resume",
  "easyicu_cancel",
  "easyicu_request_replan",
]);
const WORKSPACE_TOOL_NAMES = Object.freeze([
  "easyicu_load_skill",
  "easyicu_list_project_files",
  "easyicu_read_project_file",
  "easyicu_write_project_file",
  "easyicu_edit_project_file",
  "easyicu_check_project_file",
  "easyicu_preview_project_file",
]);
const ALL_TOOL_NAMES = Object.freeze([...RESEARCH_TOOL_NAMES, ...WORKSPACE_TOOL_NAMES]);

const sessions = new Map();
const activeRequestBySession = new Map();
const pendingToolResponses = new Map();
let modelRuntimePromise;
let temporaryModelDir;

mkdirSync(SESSION_DIR, { recursive: true, mode: 0o700 });

function emit(payload) {
  process.stdout.write(`${JSON.stringify({ protocol_version: PROTOCOL_VERSION, ...payload })}\n`);
}

function errorPayload(code, message, details = undefined) {
  return { code, message: String(message || code).slice(0, 1000), ...(details ? { details } : {}) };
}

function fail(requestId, code, message, details = undefined) {
  emit({ kind: "response", request_id: requestId, ok: false, error: errorPayload(code, message, details) });
}

function ok(requestId, result) {
  emit({ kind: "response", request_id: requestId, ok: true, result });
}

function boundedText(value, limit = MAX_TEXT_CHARS) {
  return String(value ?? "").slice(0, limit);
}

function assertExactKeys(value, allowed, code) {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw Object.assign(new Error("request must be an object"), { code });
  }
  const unknown = Object.keys(value).filter((key) => !allowed.has(key));
  if (unknown.length) {
    throw Object.assign(new Error(`unknown fields: ${unknown.join(", ")}`), {
      code,
      details: { fields: unknown },
    });
  }
}

function safeSessionFile(rawPath) {
  const candidate = String(rawPath || "").trim();
  if (!candidate || !isAbsolute(candidate)) {
    throw Object.assign(new Error("session_file must be an absolute path"), {
      code: "pi_session_file_invalid",
    });
  }
  const root = realpathSync(SESSION_DIR);
  const file = realpathSync(candidate);
  const rel = relative(root, file);
  if (!rel || rel === ".." || rel.startsWith(`..${sep}`) || isAbsolute(rel)) {
    throw Object.assign(new Error("session_file is outside the Pi session directory"), {
      code: "pi_session_file_outside_root",
    });
  }
  return file;
}

function integerEnv(name, fallback, minimum, maximum) {
  const parsed = Number.parseInt(process.env[name] || "", 10);
  if (!Number.isFinite(parsed)) return fallback;
  return Math.max(minimum, Math.min(maximum, parsed));
}

function modelConfig() {
  const apiKey = String(process.env.EASYICU_PI_API_KEY || "").trim();
  const baseUrl = String(process.env.EASYICU_PI_BASE_URL || "http://127.0.0.1:8317/v1").trim();
  const model = String(process.env.EASYICU_PI_MODEL || "gpt-5.6-luna").trim();
  const provider = String(process.env.EASYICU_PI_PROVIDER || "easyicu-local").trim();
  const api = String(process.env.EASYICU_PI_API || "openai-completions").trim();
  if (!apiKey) throw Object.assign(new Error("EASYICU_PI_API_KEY is not configured"), { code: "pi_api_key_missing" });
  if (!baseUrl || !model || !provider) {
    throw Object.assign(new Error("Pi provider configuration is incomplete"), { code: "pi_model_config_invalid" });
  }
  if (!new Set([
    "anthropic-messages",
    "google-generative-ai",
    "openai-completions",
    "openai-responses",
  ]).has(api)) {
    throw Object.assign(new Error(`unsupported Pi API transport: ${api}`), { code: "pi_api_transport_unsupported" });
  }
  const maxTokens = integerEnv("EASYICU_PI_MAX_TOKENS", 16384, 256, 131072);
  return {
    apiKey,
    baseUrl,
    model,
    provider,
    api,
    contextWindow: integerEnv("EASYICU_PI_CONTEXT_WINDOW", 200000, 8192, 2000000),
    maxTokens,
    sessionTokenBudget: Math.max(
      maxTokens,
      integerEnv("EASYICU_PI_SESSION_TOKEN_BUDGET", 1000000, 16384, 100000000),
    ),
    maxProviderCallsPerMessage: integerEnv(
      "EASYICU_PI_MAX_PROVIDER_CALLS_PER_MESSAGE", 8, 1, 64,
    ),
    maxProviderCallsPerSession: integerEnv(
      "EASYICU_PI_MAX_PROVIDER_CALLS_PER_SESSION", 128, 1, 10000,
    ),
  };
}

async function getModelRuntime() {
  if (modelRuntimePromise) return modelRuntimePromise;
  modelRuntimePromise = (async () => {
    const config = modelConfig();
    temporaryModelDir = mkdtempSync(join(tmpdir(), "easyicu-pi-model-"));
    const modelsPath = join(temporaryModelDir, "models.json");
    const payload = {
      providers: {
        [config.provider]: {
          baseUrl: config.baseUrl,
          api: config.api,
          authHeader: config.api === "openai-completions" || config.api === "openai-responses",
          models: [
            {
              id: config.model,
              name: config.model,
              reasoning: true,
              input: ["text"],
              contextWindow: config.contextWindow,
              maxTokens: config.maxTokens,
              cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
            },
          ],
        },
      },
    };
    writeFileSync(modelsPath, JSON.stringify(payload), { encoding: "utf8", mode: 0o600 });
    const credentials = new InMemoryCredentialStore();
    const runtime = await ModelRuntime.create({ credentials, modelsPath });
    await runtime.setRuntimeApiKey(config.provider, config.apiKey);
    const selected = runtime.getModel(config.provider, config.model);
    if (!selected) throw Object.assign(new Error("configured Pi model did not load"), { code: "pi_model_not_found" });
    return { runtime, selected, config };
  })();
  try {
    return await modelRuntimePromise;
  } catch (error) {
    modelRuntimePromise = undefined;
    throw error;
  }
}

function resourceLoader(agentMode) {
  const workspaceMode = agentMode === "workspace";
  return {
    getExtensions: () => ({ extensions: [], errors: [], runtime: createExtensionRuntime() }),
    getSkills: () => ({ skills: [], diagnostics: [] }),
    getPrompts: () => ({ prompts: [], diagnostics: [] }),
    getThemes: () => ({ themes: [], diagnostics: [] }),
    getAgentsFiles: () => ({ agentsFiles: [] }),
    getSystemPrompt: () => [
      "You are the conversational shell for EasyICU, a local ICU research workspace.",
      "EasyICU, not this conversation, owns study configuration, scientific plans, execution, validation, evidence, and readiness.",
      workspaceMode
        ? "You are in workspace mode. Use the registered project artifact tools to inspect, create, edit, check, and preview files inside this project's isolated workspace. Never substitute a large code block in chat when an authorized tool can create the requested artifact."
        : "You are in research mode. Use only the registered EasyICU research tools. You have no project-file authoring capability in this mode.",
      "You have no arbitrary host filesystem, shell, raw-data, network, credential, or direct EvidenceStore access.",
      "Never ask for or reproduce patient rows, identifiers, timestamps, free-text notes, credentials, or raw files.",
      "Inspection results are bounded host projections. Explain their stable codes and owner; do not invent missing details.",
      "A tool request to save study setup, run, cancel, or replan can be blocked unless the user granted that action for this turn. Do not claim an action happened unless the tool confirms it.",
      "A Pi session is UX history only. EasyICU study revision, run id, plan receipts, and evidence artefacts remain authoritative.",
      "Match the user's language and brevity. Answer the current request first. Unless the user asks for a report or explanation, use at most two short sentences around tool calls and let the UI timeline carry job ids, owners, status codes, and execution detail.",
      "At a governed confirmation gate, ask one direct question and stop. Do not repeat the full workflow, handoff report, or permission inventory unless the user asks for it. Never hide a blocker or weaken its exact stable code.",
      "Keep shared guidance case-neutral. Ask for unread study slots rather than filling defaults.",
      "Start each research request by inspecting the project workflow. Idea Mining is an optional EasyICU stage; when the user selects a mined idea, accept its digest-bound handoff before continuing setup. Study setup, extraction, plan, execution, validation, interpretation, and manuscript review remain receipt-driven stages.",
      "Tool-first Idea Mining rule: when the user asks to discover, mine, compare, or propose research ideas, do not author a candidate from general model knowledge. If the one-turn idea grant is present, call easyicu_mine_ideas before writing the answer and ground the answer only in that EasyICU receipt. If the grant is absent, ask the user to enable it. Never imply that Idea Mining ran when no Idea Mining receipt exists.",
      "Tool-first literature rule: when the user asks to search papers, prior art, or supporting literature, call easyicu_search_literature and let the host-held one-turn gate authoritatively allow or block it. The grant is intentionally not exposed in the conversation, so never infer or claim that it is absent before the tool returns a typed gate receipt. Report the exact receipt status and never call curated references a completed search. For an existing Research Agent plan, call easyicu_inspect_literature and distinguish literature design support from patient/result evidence. Never invent a paper or a plan-to-paper mapping.",
      "When the user explicitly selects a mined candidate, use easyicu_accept_idea_handoff with its exact run_id and idea_id if the one-turn idea grant is present. Do not manually copy or silently reinterpret the selected idea, and do not claim it is bound until the digest-bound acceptance receipt succeeds.",
      "When the user explicitly asks to run, rerun, execute, or analyze the already-configured current study, treat that as execution intent rather than a request to inspect an older run. Inspect the workflow only as needed to choose the next governed action, then call easyicu_run: use a full run when the provider-run grant is present and the required setup, export, and preflight are ready; otherwise start the required preflight or state the exact missing authority. Use easyicu_inspect_run only when the user asks for status, prior results, or failure diagnosis.",
      workspaceMode
        ? "For a webpage, calculator, dashboard, or interactive artifact: load the web-prototype skill, list files, write or edit the artifact, read it back, run the static check, and prepare its web preview. Label simulated formulas and values as unvalidated demo content."
        : "Treat a scientific question as the start of a governed research workflow. Inspect the typed StudyContext first; map only facts the user actually supplied, ask concise follow-up questions for required missing slots, and save them only with the one-turn configure grant.",
      workspaceMode
        ? "Research tools remain available in workspace mode, but project files never replace EasyICU-owned plans, runs, gates, evidence, or scientific results."
        : "When the typed setup is ready, use the authorized EasyICU extraction tool if no active export exists. Start the deterministic preflight before any full run. A full Research Agent provider run requires its separate one-turn provider-run grant and its existing scientific-provider gate; Pi model credentials alone do not authorize scientific execution. If the pipeline pauses for a digest-bound plan review, summarize the exact pending request and wait for the user's explicit decision; only then call easyicu_resume with approved or rejected and a fresh provider-run grant. On later turns inspect validation, evidence, evidence-bound interpretation, and the Research Agent manuscript. The UI can open projected artifact references in its governed right-side preview.",
      workspaceMode
        ? "If the requested result needs EasyICU scientific execution, use the research tools instead of fabricating a file-based result."
        : "If the user asks to create code or files, explain that they must open workspace mode; do not paste a pretend substitute artifact.",
    ].join("\n"),
    getSystemPromptSource: () => undefined,
    getAppendSystemPrompt: () => [],
    getAppendSystemPromptSources: () => [],
    extendResources: () => {},
    reload: async () => {},
  };
}

function hostTool(sessionId, definition) {
  return defineTool({
    ...definition,
    execute: async (_toolCallId, params) => {
      let result;
      try {
        result = await requestHostTool(sessionId, definition.name, params);
      } catch (error) {
        const code = boundedText(error?.code || "pi_host_tool_rejected", 160);
        const modelVisible = {
          status: "blocked",
          code,
          summary: `EasyICU blocked this tool request (${code}).`,
          owner: "easyicu.webserver.pi_copilot",
          details: {},
          authority: {},
        };
        return {
          content: [{ type: "text", text: JSON.stringify(modelVisible) }],
          details: modelVisible,
          isError: true,
        };
      }
      const code = boundedText(result?.code || "pi_tool_result", 160);
      const summary = boundedText(result?.summary || code, 2000);
      const modelVisible = {
        status: boundedText(result?.status || "error", 40),
        code,
        summary,
        owner: boundedText(result?.owner || "easyicu.webserver.pi_copilot", 240),
        details: result?.details || {},
        authority: result?.authority || {},
      };
      return {
        content: [{ type: "text", text: JSON.stringify(modelVisible) }],
        details: modelVisible,
      };
    },
  });
}

function customTools(sessionId, agentMode) {
  const optionalRunId = Type.Optional(Type.String({ maxLength: 160 }));
  const empty = Type.Object({}, { additionalProperties: false });
  const optionalText = (maxLength) => Type.Optional(Type.String({ maxLength }));
  const studyCohort = Type.Object({
    preset: optionalText(500), label: optionalText(500), review: optionalText(500),
    review_scope: optionalText(500), comparison: optionalText(500), source_type: optionalText(500),
    comparison_mode: optionalText(500), age_min: Type.Optional(Type.Number()),
    age_max: Type.Optional(Type.Number()), min_icu_los_hours: Type.Optional(Type.Number()),
    observation_window_hours: Type.Optional(Type.Number()), max_patients: Type.Optional(Type.Number()),
    exclude_readmissions: Type.Optional(Type.Boolean()),
    include_diagnoses: Type.Optional(Type.Array(Type.String({ maxLength: 160 }), { maxItems: 64 })),
    exclude_diagnoses: Type.Optional(Type.Array(Type.String({ maxLength: 160 }), { maxItems: 64 })),
  }, { additionalProperties: false });
  const studyWindow = Type.Object({
    hours: Type.Optional(Type.Number()), observation_hours: Type.Optional(Type.Number()),
    anchor: optionalText(500), preset: optionalText(500), label: optionalText(500),
  }, { additionalProperties: false });
  const tools = [
    hostTool(sessionId, { name: "easyicu_workspace_status", label: "EasyICU workspace status", description: "Inspect the current EasyICU workspace and authoritative study/run binding without reading patient rows or source paths.", parameters: empty }),
    hostTool(sessionId, { name: "easyicu_inspect_workflow", label: "Inspect research workflow", description: "Inspect the typed project workflow from scientific question through Idea Mining, setup, extraction, Research Agent analysis, interpretation, and manuscript review.", parameters: empty }),
    hostTool(sessionId, { name: "easyicu_inspect_context", label: "Inspect study context", description: "Read the PHI-safe projection of the bound typed StudyContext and its revision.", parameters: empty }),
    hostTool(sessionId, { name: "easyicu_inspect_plan", label: "Inspect scientific plan", description: "Read a bounded projection of the current or selected EasyICU plan artefact.", parameters: Type.Object({ run_id: optionalRunId }, { additionalProperties: false }) }),
    hostTool(sessionId, { name: "easyicu_inspect_literature", label: "Inspect literature evidence", description: "Read the bounded literature bundle and plan-step citation mapping projected by EasyICU. Curated-only bundles are explicitly distinguished from completed searches.", parameters: Type.Object({ run_id: optionalRunId }, { additionalProperties: false }) }),
    hostTool(sessionId, { name: "easyicu_inspect_capability", label: "Inspect capabilities", description: "Inspect the current EasyICU capability policy and availability without credentials or private paths.", parameters: empty }),
    hostTool(sessionId, { name: "easyicu_inspect_run", label: "Inspect run", description: "Inspect a bounded run/job status owned by EasyICU.", parameters: Type.Object({ run_id: optionalRunId, job_id: Type.Optional(Type.String({ maxLength: 160 })) }, { additionalProperties: false }) }),
    hostTool(sessionId, { name: "easyicu_inspect_step", label: "Inspect plan step", description: "Inspect one step from the bound plan without executing or editing it.", parameters: Type.Object({ run_id: optionalRunId, step_id: Type.String({ minLength: 1, maxLength: 160 }) }, { additionalProperties: false }) }),
    hostTool(sessionId, { name: "easyicu_inspect_validation", label: "Inspect validation", description: "Inspect quality/readiness gate status from an existing EasyICU run.", parameters: Type.Object({ run_id: optionalRunId }, { additionalProperties: false }) }),
    hostTool(sessionId, { name: "easyicu_list_artifacts", label: "List run artefacts", description: "List whitelisted EasyICU artefact names and digests; never return file contents or paths.", parameters: Type.Object({ run_id: optionalRunId }, { additionalProperties: false }) }),
    hostTool(sessionId, { name: "easyicu_inspect_evidence", label: "Inspect evidence", description: "Inspect bounded evidence-ledger and audit status for an EasyICU run.", parameters: Type.Object({ run_id: optionalRunId }, { additionalProperties: false }) }),
    hostTool(sessionId, { name: "easyicu_explain_blocker", label: "Explain blocker", description: "Explain the current stable EasyICU blocker code and its owning boundary.", parameters: Type.Object({ run_id: optionalRunId, job_id: Type.Optional(Type.String({ maxLength: 160 })) }, { additionalProperties: false }) }),
    hostTool(sessionId, { name: "easyicu_inspect_interpretation", label: "Interpret validated results", description: "Organize existing Research Agent claims, gates, limitations, and artifact references into an evidence-bound human-review card. This tool never calculates a new number or invents an explanation.", parameters: Type.Object({ run_id: optionalRunId }, { additionalProperties: false }) }),
    hostTool(sessionId, { name: "easyicu_inspect_manuscript", label: "Inspect manuscript draft", description: "Open the Research Agent-produced, evidence-bound manuscript draft and its governance status. Pi does not author or unlock it.", parameters: Type.Object({ run_id: optionalRunId }, { additionalProperties: false }) }),
    hostTool(sessionId, { name: "easyicu_update_study_context", executionMode: "sequential", label: "Save study setup", description: "Persist typed conversational study slots through the existing StudyContext owner. Requires a host-held one-turn Configure authorization; the conversation host rebinds the session after the turn settles.", parameters: Type.Object({
      title: optionalText(160), question: optionalText(1200), purpose: optionalText(800),
      cohort: Type.Optional(studyCohort), modules: Type.Optional(Type.Array(Type.String({ maxLength: 80 }), { maxItems: 64 })),
      outcome: optionalText(500), primary_exposure: optionalText(160),
      covariates: Type.Optional(Type.Array(Type.String({ maxLength: 160 }), { maxItems: 64 })),
      time_window: Type.Optional(studyWindow), comparator: optionalText(500),
      export_format: optionalText(40), analysis_goal: optionalText(1200),
      confirmations: Type.Optional(Type.Record(Type.String({ maxLength: 80 }), Type.Boolean())),
      bind_active_export: Type.Optional(Type.Boolean()),
    }, { additionalProperties: false }) }),
    hostTool(sessionId, { name: "easyicu_mine_ideas", executionMode: "sequential", label: "Mine research ideas", description: "Create one local, metadata-only Idea Mining candidate from the bound question or a bounded source seed. Requires the one-use Idea Mining grant and never produces a novelty or scientific result claim.", parameters: Type.Object({ topic: optionalText(1200), title: optionalText(220), excerpt: optionalText(1200), journal: optionalText(160), year: Type.Optional(Type.Integer({ minimum: 1800, maximum: 2200 })), doi: optionalText(240), pmid: optionalText(80) }, { additionalProperties: false }) }),
    hostTool(sessionId, { name: "easyicu_search_literature", executionMode: "sequential", label: "Search PubMed literature", description: "Run the existing Idea Mining PubMed metadata/abstract discovery owner. Requires the separate one-turn literature-network grant; no full text, patient rows, or external LLM is used.", parameters: Type.Object({ topic: optionalText(1200), journal: optionalText(160), limit: Type.Optional(Type.Integer({ minimum: 1, maximum: 20 })) }, { additionalProperties: false }) }),
    hostTool(sessionId, { name: "easyicu_prepare_idea_handoff", executionMode: "sequential", label: "Prepare idea plan", description: "Create the canonical metadata-only Idea Mining plan/handoff for conversational review. Requires the one-use Idea Mining grant; it does not start analysis or make reportable claims.", parameters: Type.Object({ run_id: Type.String({ minLength: 1, maxLength: 160 }), idea_id: optionalText(160), plan_edits: optionalText(1200) }, { additionalProperties: false }) }),
    hostTool(sessionId, { name: "easyicu_accept_idea_handoff", executionMode: "sequential", label: "Accept selected idea", description: "After the user explicitly selects an idea, bind its canonical digest and agreed fields to the current StudyContext. Requires the one-use Idea Mining grant and stops the turn for an authority rebind.", parameters: Type.Object({ run_id: Type.String({ minLength: 1, maxLength: 160 }), idea_id: Type.String({ minLength: 1, maxLength: 160 }), plan_edits: optionalText(1200) }, { additionalProperties: false }) }),
    hostTool(sessionId, { name: "easyicu_start_extraction", executionMode: "sequential", label: "Start feature extraction", description: "Submit the existing EasyICU feature extraction owner using only the bound typed StudyContext. Requires the one-use Extraction grant; raw paths never come from the model.", parameters: empty }),
    hostTool(sessionId, { name: "easyicu_run", executionMode: "sequential", label: "Start EasyICU run", description: "Submit an EasyICU preflight or the real ResearchAgentPipeline Plan -> Execute -> Validate -> Write workflow. Preflight requires the local-run grant. Full analysis requires the separate provider-run grant and the existing scientific provider gates. The host, not the model, selects the already verified provider configuration. Submission invalidates this turn's authority: report the receipt and stop until host rebind.", parameters: Type.Object({ run_type: Type.Optional(Type.Union([Type.Literal("preflight"), Type.Literal("full")])) }, { additionalProperties: false }) }),
    hostTool(sessionId, { name: "easyicu_resume", executionMode: "sequential", label: "Resume EasyICU work", description: "Reattach to an active job, or submit an explicit approved/rejected decision for a same-process digest-bound Research Agent plan review. A review decision needs a fresh provider-run grant.", parameters: Type.Object({ job_id: Type.Optional(Type.String({ maxLength: 160 })), run_id: optionalRunId, decision: Type.Optional(Type.Union([Type.Literal("approved"), Type.Literal("rejected")])), reviewer: Type.Optional(Type.String({ maxLength: 200 })), note: Type.Optional(Type.String({ maxLength: 1000 })) }, { additionalProperties: false }) }),
    hostTool(sessionId, { name: "easyicu_cancel", executionMode: "sequential", label: "Cancel EasyICU job", description: "Request cooperative cancellation of the specifically bound EasyICU job. Requires a host-held one-turn user authorization.", parameters: Type.Object({ job_id: Type.Optional(Type.String({ maxLength: 160 })) }, { additionalProperties: false }) }),
    hostTool(sessionId, { name: "easyicu_request_replan", executionMode: "sequential", label: "Request replan", description: "Request re-planning through EasyICU authority. Version 1 returns a typed blocked result until a public replan owner exists.", parameters: Type.Object({ reason: Type.String({ minLength: 1, maxLength: 1200 }) }, { additionalProperties: false }) }),
  ];
  if (agentMode !== "workspace") return tools;
  const projectFile = Type.String({ minLength: 1, maxLength: 240 });
  const fileSha256 = Type.String({ pattern: "^[A-Fa-f0-9]{64}$" });
  return tools.concat([
    hostTool(sessionId, { name: "easyicu_load_skill", executionMode: "sequential", label: "Load workspace skill", description: "Load one reviewed EasyICU project-workspace skill before authoring an artifact.", parameters: Type.Object({ name: Type.Literal("web-prototype") }, { additionalProperties: false }) }),
    hostTool(sessionId, { name: "easyicu_list_project_files", executionMode: "sequential", label: "List project files", description: "List bounded text and web artifacts in this project's isolated workspace.", parameters: empty }),
    hostTool(sessionId, { name: "easyicu_read_project_file", executionMode: "sequential", label: "Read project file", description: "Read a bounded UTF-8 file from this project's isolated workspace.", parameters: Type.Object({ file: projectFile, start_line: Type.Optional(Type.Integer({ minimum: 1, maximum: 100000 })), end_line: Type.Optional(Type.Integer({ minimum: 1, maximum: 100000 })) }, { additionalProperties: false }) }),
    hostTool(sessionId, { name: "easyicu_write_project_file", executionMode: "sequential", label: "Write project file", description: "Create a new bounded artifact. Existing files must be changed with the exact-edit tool. Requires the reusable host-held workspace-write capability for this message.", parameters: Type.Object({ file: projectFile, content: Type.String({ minLength: 1, maxLength: 262144 }) }, { additionalProperties: false }) }),
    hostTool(sessionId, { name: "easyicu_edit_project_file", executionMode: "sequential", label: "Edit project file", description: "Apply one exact replacement after reading the artifact and passing its current SHA-256 digest. Requires the reusable host-held workspace-write capability for this message.", parameters: Type.Object({ file: projectFile, old_text: Type.String({ minLength: 1, maxLength: 120000 }), new_text: Type.String({ maxLength: 120000 }), expected_sha256: fileSha256 }, { additionalProperties: false }) }),
    hostTool(sessionId, { name: "easyicu_check_project_file", executionMode: "sequential", label: "Check project file", description: "Run a bounded non-executing syntax or structure check on a project artifact.", parameters: Type.Object({ file: projectFile }, { additionalProperties: false }) }),
    hostTool(sessionId, { name: "easyicu_preview_project_file", executionMode: "sequential", label: "Prepare web preview", description: "Prepare a governed browser preview for a bounded HTML artifact in this project workspace.", parameters: Type.Object({ file: projectFile }, { additionalProperties: false }) }),
  ]);
}

async function requestHostTool(sessionId, name, args) {
  const parentRequestId = activeRequestBySession.get(sessionId);
  if (!parentRequestId) {
    throw Object.assign(new Error("tool call has no active Pi prompt"), { code: "pi_tool_without_active_prompt" });
  }
  const requestId = randomUUID();
  emit({
    kind: "tool_request",
    request_id: requestId,
    parent_request_id: parentRequestId,
    session_id: sessionId,
    method: "tool.execute",
    params: { name, arguments: args || {} },
  });
  return await new Promise((resolvePromise, rejectPromise) => {
    const timer = setTimeout(() => {
      pendingToolResponses.delete(requestId);
      rejectPromise(Object.assign(new Error("EasyICU host tool timed out"), { code: "pi_host_tool_timeout" }));
    }, 10 * 60 * 1000);
    pendingToolResponses.set(requestId, {
      resolve: (result) => { clearTimeout(timer); resolvePromise(result); },
      reject: (error) => { clearTimeout(timer); rejectPromise(error); },
    });
  });
}

function sessionState(record) {
  const { session } = record;
  const stats = session.getSessionStats();
  return {
    session_id: record.externalId,
    agent_mode: record.agentMode,
    pi_session_id: session.sessionId,
    session_file: session.sessionFile,
    model: session.model ? { provider: session.model.provider, id: session.model.id } : null,
    thinking_level: session.thinkingLevel,
    message_count: session.messages.length,
    streaming: session.isStreaming,
    enabled_tools: session.getActiveToolNames().filter((name) => ALL_TOOL_NAMES.includes(name)),
    transcript: session.messages.slice(-100).map(projectTranscriptMessage).filter(Boolean),
    shell_usage: {
      tokens: stats.tokens,
      cost: null,
      pricing_available: false,
      token_budget: record.sessionTokenBudget,
      tokens_remaining: Math.max(0, record.sessionTokenBudget - stats.tokens.total),
      ...record.budgetGuard.state(),
    },
  };
}

async function createSession(params) {
  assertExactKeys(params, new Set(["session_id", "session_file", "thinking_level", "agent_mode"]), "pi_session_create_invalid");
  const externalId = boundedText(params.session_id, 160).trim();
  if (!/^[A-Za-z0-9][A-Za-z0-9._-]{0,159}$/.test(externalId)) {
    throw Object.assign(new Error("invalid external session id"), { code: "pi_session_id_invalid" });
  }
  if (sessions.has(externalId)) return sessionState(sessions.get(externalId));
  const agentMode = params.agent_mode === "workspace" ? "workspace" : "research";
  const { runtime, selected, config } = await getModelRuntime();
  const manager = params.session_file
    ? SessionManager.open(safeSessionFile(params.session_file), SESSION_DIR, CWD)
    : SessionManager.create(CWD, SESSION_DIR);
  const thinkingLevel = "off";
  const settingsManager = SettingsManager.inMemory({
    compaction: { enabled: true },
    // Hidden provider retries bypass host call accounting. EasyICU therefore
    // disables them and lets every outbound call cross ShellBudgetGuard once.
    retry: { enabled: false, maxRetries: 0 },
  });
  const { session } = await createAgentSession({
    cwd: CWD,
    agentDir: dirname(SESSION_DIR),
    model: selected,
    thinkingLevel,
    modelRuntime: runtime,
    resourceLoader: resourceLoader(agentMode),
    sessionManager: manager,
    settingsManager,
    noTools: "builtin",
    tools: agentMode === "workspace" ? ALL_TOOL_NAMES : RESEARCH_TOOL_NAMES,
    customTools: customTools(externalId, agentMode),
  });
  const unsubscribe = session.subscribe((event) => {
    const requestId = activeRequestBySession.get(externalId);
    if (!requestId) return;
    const normalized = normalizePiEvent(event);
    if (normalized) emit({ kind: "event", request_id: requestId, session_id: externalId, event: normalized });
  });
  const record = {
    externalId,
    agentMode,
    session,
    unsubscribe,
    sessionTokenBudget: config.sessionTokenBudget,
    maxTokens: config.maxTokens,
  };
  const originalStreamFunction = session.agent.streamFunction;
  record.budgetGuard = new ShellBudgetGuard({
    tokenBudget: config.sessionTokenBudget,
    maxOutputTokens: config.maxTokens,
    maxProviderCallsPerMessage: config.maxProviderCallsPerMessage,
    maxProviderCallsPerSession: config.maxProviderCallsPerSession,
    consumedTokens: () => session.getSessionStats().tokens.total,
    initialProviderCalls: restoredProviderCallCount(
      typeof manager?.getEntries === "function" ? manager.getEntries() : [],
      session.getSessionStats().assistantMessages,
    ),
  });
  session.agent.streamFunction = (model, context, options = {}) => lazyStream(
    model,
    async () => {
      const authorization = record.budgetGuard.authorize(context, options);
      manager.appendCustomEntry(
        SHELL_BUDGET_RECEIPT,
        providerCallReceipt(authorization.session_provider_call),
      );
      return await originalStreamFunction(model, context, {
        ...options,
        maxRetries: 0,
      });
    },
  );
  sessions.set(externalId, record);
  return sessionState(record);
}

async function promptSession(requestId, params) {
  assertExactKeys(params, new Set(["session_id", "message", "streaming_behavior"]), "pi_prompt_invalid");
  const sessionId = boundedText(params.session_id, 160).trim();
  const message = boundedText(params.message, MAX_TEXT_CHARS).trim();
  const record = sessions.get(sessionId);
  if (!record) throw Object.assign(new Error("Pi session is not open"), { code: "pi_session_not_open" });
  if (!message) throw Object.assign(new Error("message is required"), { code: "pi_message_required" });
  if (activeRequestBySession.has(sessionId)) {
    throw Object.assign(new Error("Pi session already has an active prompt"), { code: "pi_session_busy" });
  }
  record.budgetGuard.beginMessage();
  activeRequestBySession.set(sessionId, requestId);
  try {
    await record.session.prompt(message);
    return sessionState(record);
  } finally {
    record.budgetGuard.endMessage();
    activeRequestBySession.delete(sessionId);
  }
}

async function handleRequest(request) {
  assertExactKeys(request, new Set(["protocol_version", "kind", "request_id", "method", "params"]), "pi_protocol_unknown_fields");
  const requestId = boundedText(request.request_id, 160).trim();
  if (!requestId) throw Object.assign(new Error("request_id is required"), { code: "pi_request_id_required" });
  if (request.protocol_version !== PROTOCOL_VERSION) {
    throw Object.assign(new Error("unsupported protocol version"), { code: "pi_protocol_version_unsupported", requestId });
  }
  const params = request.params || {};
  switch (request.method) {
    case "runtime.status":
      assertExactKeys(params, new Set(), "pi_runtime_status_invalid");
      return {
        gateway: "ready",
        pi_package_version: "0.84.1",
        pi_source_commit: "9dd90a49711d088b86fdd9b4aea575913a8328a8",
        model_configured: Boolean(String(process.env.EASYICU_PI_API_KEY || "").trim()),
        model: String(process.env.EASYICU_PI_MODEL || "gpt-5.6-luna"),
        provider: String(process.env.EASYICU_PI_PROVIDER || "easyicu-local"),
        built_in_tools_enabled: [],
        custom_tools: [...RESEARCH_TOOL_NAMES],
        custom_tools_by_mode: {
          research: [...RESEARCH_TOOL_NAMES],
          workspace: [...ALL_TOOL_NAMES],
        },
      };
    case "session.create":
      return await createSession(params);
    case "session.prompt":
      return await promptSession(requestId, params);
    case "session.state": {
      assertExactKeys(params, new Set(["session_id"]), "pi_session_state_invalid");
      const record = sessions.get(boundedText(params.session_id, 160).trim());
      if (!record) throw Object.assign(new Error("Pi session is not open"), { code: "pi_session_not_open" });
      return sessionState(record);
    }
    case "session.abort": {
      assertExactKeys(params, new Set(["session_id"]), "pi_session_abort_invalid");
      const record = sessions.get(boundedText(params.session_id, 160).trim());
      if (!record) throw Object.assign(new Error("Pi session is not open"), { code: "pi_session_not_open" });
      await record.session.abort();
      return { aborted: true, ...sessionState(record) };
    }
    case "session.dispose": {
      assertExactKeys(params, new Set(["session_id"]), "pi_session_dispose_invalid");
      const sessionId = boundedText(params.session_id, 160).trim();
      const record = sessions.get(sessionId);
      if (record) {
        record.unsubscribe?.();
        record.session.dispose();
        sessions.delete(sessionId);
      }
      return { disposed: Boolean(record), session_id: sessionId };
    }
    default:
      throw Object.assign(new Error(`unknown method: ${boundedText(request.method, 160)}`), { code: "pi_method_unknown", requestId });
  }
}

function handleToolResponse(payload) {
  assertExactKeys(payload, new Set(["protocol_version", "kind", "request_id", "ok", "result", "error"]), "pi_tool_response_unknown_fields");
  if (payload.protocol_version !== PROTOCOL_VERSION) return;
  const pending = pendingToolResponses.get(String(payload.request_id || ""));
  if (!pending) return;
  pendingToolResponses.delete(String(payload.request_id));
  if (payload.ok) pending.resolve(payload.result || {});
  else pending.reject(Object.assign(new Error(payload.error?.message || "EasyICU host tool failed"), { code: payload.error?.code || "pi_host_tool_failed" }));
}

async function handleLine(line) {
  if (Buffer.byteLength(line, "utf8") > MAX_LINE_BYTES) {
    fail("unknown", "pi_protocol_line_too_large", "protocol line exceeds the size limit");
    return;
  }
  let payload;
  try {
    payload = JSON.parse(line);
  } catch {
    fail("unknown", "pi_protocol_invalid_json", "invalid JSON line");
    return;
  }
  if (payload?.kind === "tool_response") {
    try { handleToolResponse(payload); } catch (error) { process.stderr.write(`[easyicu-pi] ${error?.code || "tool_response_error"}\n`); }
    return;
  }
  const requestId = boundedText(payload?.request_id || "unknown", 160);
  if (payload?.kind !== "request") {
    fail(requestId, "pi_protocol_kind_invalid", "expected a request or tool_response envelope");
    return;
  }
  try {
    const result = await handleRequest(payload);
    ok(requestId, result);
  } catch (error) {
    fail(requestId, error?.code || "pi_gateway_error", error?.message || "Pi gateway failed", error?.details);
  }
}

async function shutdown() {
  for (const record of sessions.values()) {
    try { record.unsubscribe?.(); } catch {}
    try { record.session.dispose(); } catch {}
  }
  sessions.clear();
  try { rmSync(temporaryModelDir, { recursive: true, force: true }); } catch {}
}

const reader = createInterface({ input: process.stdin, crlfDelay: Infinity });
reader.on("line", (line) => { void handleLine(line); });
reader.on("close", () => { void shutdown().finally(() => process.exit(0)); });
process.on("SIGTERM", () => { void shutdown().finally(() => process.exit(0)); });
process.on("SIGINT", () => { void shutdown().finally(() => process.exit(0)); });
