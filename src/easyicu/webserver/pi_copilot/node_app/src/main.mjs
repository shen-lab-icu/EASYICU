import { randomUUID } from "node:crypto";
import { mkdirSync, mkdtempSync, realpathSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, isAbsolute, join, relative, resolve, sep } from "node:path";
import { createInterface } from "node:readline";

import { InMemoryCredentialStore } from "@earendil-works/pi-ai";
import {
  createAgentSession,
  createExtensionRuntime,
  defineTool,
  ModelRuntime,
  SessionManager,
  SettingsManager,
} from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";

const PROTOCOL_VERSION = "easyicu.pi-copilot/1";
const MAX_LINE_BYTES = 1024 * 1024;
const MAX_TEXT_CHARS = 12000;
const SESSION_DIR = resolve(
  process.env.EASYICU_PI_SESSION_DIR || join(process.cwd(), ".easyicu-pi-sessions"),
);
const CWD = resolve(process.env.EASYICU_PI_CWD || process.cwd());
const TOOL_NAMES = Object.freeze([
  "easyicu_workspace_status",
  "easyicu_inspect_context",
  "easyicu_inspect_plan",
  "easyicu_inspect_capability",
  "easyicu_inspect_run",
  "easyicu_inspect_step",
  "easyicu_inspect_validation",
  "easyicu_list_artifacts",
  "easyicu_inspect_evidence",
  "easyicu_explain_blocker",
  "easyicu_update_study_context",
  "easyicu_run",
  "easyicu_resume",
  "easyicu_cancel",
  "easyicu_request_replan",
]);

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
  const model = String(process.env.EASYICU_PI_MODEL || "gpt5.6 luna").trim();
  const provider = String(process.env.EASYICU_PI_PROVIDER || "easyicu-local").trim();
  const api = String(process.env.EASYICU_PI_API || "openai-completions").trim();
  if (!apiKey) throw Object.assign(new Error("EASYICU_PI_API_KEY is not configured"), { code: "pi_api_key_missing" });
  if (!baseUrl || !model || !provider) {
    throw Object.assign(new Error("Pi provider configuration is incomplete"), { code: "pi_model_config_invalid" });
  }
  if (!new Set(["openai-completions", "openai-responses"]).has(api)) {
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
          authHeader: true,
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

function resourceLoader() {
  return {
    getExtensions: () => ({ extensions: [], errors: [], runtime: createExtensionRuntime() }),
    getSkills: () => ({ skills: [], diagnostics: [] }),
    getPrompts: () => ({ prompts: [], diagnostics: [] }),
    getThemes: () => ({ themes: [], diagnostics: [] }),
    getAgentsFiles: () => ({ agentsFiles: [] }),
    getSystemPrompt: () => [
      "You are the conversational shell for EasyICU, a local ICU research workspace.",
      "EasyICU, not this conversation, owns study configuration, scientific plans, execution, validation, evidence, and readiness.",
      "Use only the registered EasyICU tools. You have no filesystem, shell, code-editing, raw-data, network, credential, or direct EvidenceStore access.",
      "Never ask for or reproduce patient rows, identifiers, timestamps, free-text notes, credentials, or raw files.",
      "Inspection results are bounded host projections. Explain their stable codes and owner; do not invent missing details.",
      "A tool request to save study setup, run, cancel, or replan can be blocked unless the user granted that action for this turn. Do not claim an action happened unless the tool confirms it.",
      "A Pi session is UX history only. EasyICU study revision, run id, plan receipts, and evidence artefacts remain authoritative.",
      "Keep shared guidance case-neutral. Ask for unread study slots rather than filling defaults.",
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
      const result = await requestHostTool(sessionId, definition.name, params);
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

function customTools(sessionId) {
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
  return [
    hostTool(sessionId, { name: "easyicu_workspace_status", label: "EasyICU workspace status", description: "Inspect the current EasyICU workspace and authoritative study/run binding without reading patient rows or source paths.", parameters: empty }),
    hostTool(sessionId, { name: "easyicu_inspect_context", label: "Inspect study context", description: "Read the PHI-safe projection of the bound typed StudyContext and its revision.", parameters: empty }),
    hostTool(sessionId, { name: "easyicu_inspect_plan", label: "Inspect scientific plan", description: "Read a bounded projection of the current or selected EasyICU plan artefact.", parameters: Type.Object({ run_id: optionalRunId }, { additionalProperties: false }) }),
    hostTool(sessionId, { name: "easyicu_inspect_capability", label: "Inspect capabilities", description: "Inspect the current EasyICU capability policy and availability without credentials or private paths.", parameters: empty }),
    hostTool(sessionId, { name: "easyicu_inspect_run", label: "Inspect run", description: "Inspect a bounded run/job status owned by EasyICU.", parameters: Type.Object({ run_id: optionalRunId, job_id: Type.Optional(Type.String({ maxLength: 160 })) }, { additionalProperties: false }) }),
    hostTool(sessionId, { name: "easyicu_inspect_step", label: "Inspect plan step", description: "Inspect one step from the bound plan without executing or editing it.", parameters: Type.Object({ run_id: optionalRunId, step_id: Type.String({ minLength: 1, maxLength: 160 }) }, { additionalProperties: false }) }),
    hostTool(sessionId, { name: "easyicu_inspect_validation", label: "Inspect validation", description: "Inspect quality/readiness gate status from an existing EasyICU run.", parameters: Type.Object({ run_id: optionalRunId }, { additionalProperties: false }) }),
    hostTool(sessionId, { name: "easyicu_list_artifacts", label: "List run artefacts", description: "List whitelisted EasyICU artefact names and digests; never return file contents or paths.", parameters: Type.Object({ run_id: optionalRunId }, { additionalProperties: false }) }),
    hostTool(sessionId, { name: "easyicu_inspect_evidence", label: "Inspect evidence", description: "Inspect bounded evidence-ledger and audit status for an EasyICU run.", parameters: Type.Object({ run_id: optionalRunId }, { additionalProperties: false }) }),
    hostTool(sessionId, { name: "easyicu_explain_blocker", label: "Explain blocker", description: "Explain the current stable EasyICU blocker code and its owning boundary.", parameters: Type.Object({ run_id: optionalRunId, job_id: Type.Optional(Type.String({ maxLength: 160 })) }, { additionalProperties: false }) }),
    hostTool(sessionId, { name: "easyicu_update_study_context", label: "Save study setup", description: "Persist typed conversational study slots through the existing StudyContext owner. Requires a host-held one-turn Configure authorization and makes the session stale until explicit rebind.", parameters: Type.Object({
      title: optionalText(160), question: optionalText(1200), purpose: optionalText(800),
      cohort: Type.Optional(studyCohort), modules: Type.Optional(Type.Array(Type.String({ maxLength: 80 }), { maxItems: 64 })),
      outcome: optionalText(500), time_window: Type.Optional(studyWindow), comparator: optionalText(500),
      export_format: optionalText(40), analysis_goal: optionalText(1200),
      confirmations: Type.Optional(Type.Record(Type.String({ maxLength: 80 }), Type.Boolean())),
      bind_active_export: Type.Optional(Type.Boolean()),
    }, { additionalProperties: false }) }),
    hostTool(sessionId, { name: "easyicu_run", label: "Start EasyICU run", description: "Submit an existing EasyICU Research Agent run. Requires a host-held one-turn user authorization.", parameters: Type.Object({ run_type: Type.Optional(Type.Union([Type.Literal("preflight"), Type.Literal("full")])), llm_provider: Type.Optional(Type.String({ maxLength: 80 })) }, { additionalProperties: false }) }),
    hostTool(sessionId, { name: "easyicu_resume", label: "Resume EasyICU work", description: "Reattach to an existing active EasyICU job. Scientific crash-resume fails closed until an owner contract exists.", parameters: Type.Object({ job_id: Type.Optional(Type.String({ maxLength: 160 })), run_id: optionalRunId }, { additionalProperties: false }) }),
    hostTool(sessionId, { name: "easyicu_cancel", label: "Cancel EasyICU job", description: "Request cooperative cancellation of the specifically bound EasyICU job. Requires a host-held one-turn user authorization.", parameters: Type.Object({ job_id: Type.Optional(Type.String({ maxLength: 160 })) }, { additionalProperties: false }) }),
    hostTool(sessionId, { name: "easyicu_request_replan", label: "Request replan", description: "Request re-planning through EasyICU authority. Version 1 returns a typed blocked result until a public replan owner exists.", parameters: Type.Object({ reason: Type.String({ minLength: 1, maxLength: 1200 }) }, { additionalProperties: false }) }),
  ];
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

function normalizeEvent(event) {
  if (!event || typeof event !== "object") return undefined;
  if (event.type === "message_update") {
    const update = event.assistantMessageEvent || {};
    if (update.type === "text_delta") return { type: "text_delta", delta: boundedText(update.delta, 8000) };
    return undefined;
  }
  if (event.type === "tool_execution_start") {
    return { type: "tool_start", tool_call_id: boundedText(event.toolCallId, 160), tool_name: boundedText(event.toolName, 160) };
  }
  if (event.type === "tool_execution_end") {
    const content = Array.isArray(event.result?.content) ? event.result.content : [];
    const summary = event.result?.details?.summary
      || content.find((item) => item && item.type === "text")?.text
      || "";
    return { type: "tool_end", tool_call_id: boundedText(event.toolCallId, 160), tool_name: boundedText(event.toolName, 160), is_error: Boolean(event.isError), summary: boundedText(summary, 2000) };
  }
  if (event.type === "message_end" && event.message?.role === "assistant") {
    return { type: "message_end", stop_reason: boundedText(event.message.stopReason || "complete", 80) };
  }
  if (event.type === "compaction_start") return { type: "compaction_start", reason: boundedText(event.reason, 80) };
  if (event.type === "compaction_end") return { type: "compaction_end", reason: boundedText(event.reason, 80), aborted: Boolean(event.aborted) };
  if (event.type === "auto_retry_start") return { type: "retry", attempt: Number(event.attempt || 0), max_attempts: Number(event.maxAttempts || 0) };
  if (event.type === "agent_start" || event.type === "agent_settled") return { type: event.type };
  return undefined;
}

function transcriptMessage(message) {
  if (!message || typeof message !== "object") return undefined;
  const role = boundedText(message.role, 40);
  if (!new Set(["user", "assistant", "toolResult"]).has(role)) return undefined;
  const content = Array.isArray(message.content) ? message.content : [{ type: "text", text: message.content }];
  const parts = [];
  for (const item of content.slice(0, 80)) {
    if (!item || typeof item !== "object") continue;
    if (item.type === "text") {
      parts.push({ type: "text", text: boundedText(item.text, 12000) });
    } else if (item.type === "toolCall") {
      parts.push({ type: "tool_call", tool_call_id: boundedText(item.id, 160), tool_name: boundedText(item.name, 160) });
    }
  }
  return {
    role: role === "toolResult" ? "tool" : role,
    content: parts,
    stop_reason: role === "assistant" ? boundedText(message.stopReason || "", 80) : undefined,
  };
}

function sessionState(record) {
  const { session } = record;
  const stats = session.getSessionStats();
  return {
    session_id: record.externalId,
    pi_session_id: session.sessionId,
    session_file: session.sessionFile,
    model: session.model ? { provider: session.model.provider, id: session.model.id } : null,
    thinking_level: session.thinkingLevel,
    message_count: session.messages.length,
    streaming: session.isStreaming,
    enabled_tools: session.getActiveToolNames().filter((name) => TOOL_NAMES.includes(name)),
    transcript: session.messages.slice(-100).map(transcriptMessage).filter(Boolean),
    shell_usage: {
      tokens: stats.tokens,
      cost: stats.cost,
      token_budget: record.sessionTokenBudget,
      tokens_remaining: Math.max(0, record.sessionTokenBudget - stats.tokens.total),
    },
  };
}

async function createSession(params) {
  assertExactKeys(params, new Set(["session_id", "session_file", "thinking_level"]), "pi_session_create_invalid");
  const externalId = boundedText(params.session_id, 160).trim();
  if (!/^[A-Za-z0-9][A-Za-z0-9._-]{0,159}$/.test(externalId)) {
    throw Object.assign(new Error("invalid external session id"), { code: "pi_session_id_invalid" });
  }
  if (sessions.has(externalId)) return sessionState(sessions.get(externalId));
  const { runtime, selected, config } = await getModelRuntime();
  const manager = params.session_file
    ? SessionManager.open(safeSessionFile(params.session_file), SESSION_DIR, CWD)
    : SessionManager.create(CWD, SESSION_DIR);
  const thinkingLevel = "off";
  const settingsManager = SettingsManager.inMemory({
    compaction: { enabled: true },
    retry: { enabled: true, maxRetries: 2 },
  });
  const { session } = await createAgentSession({
    cwd: CWD,
    agentDir: dirname(SESSION_DIR),
    model: selected,
    thinkingLevel,
    modelRuntime: runtime,
    resourceLoader: resourceLoader(),
    sessionManager: manager,
    settingsManager,
    noTools: "builtin",
    tools: TOOL_NAMES,
    customTools: customTools(externalId),
  });
  const unsubscribe = session.subscribe((event) => {
    const requestId = activeRequestBySession.get(externalId);
    if (!requestId) return;
    const normalized = normalizeEvent(event);
    if (normalized) emit({ kind: "event", request_id: requestId, session_id: externalId, event: normalized });
  });
  const record = {
    externalId,
    session,
    unsubscribe,
    sessionTokenBudget: config.sessionTokenBudget,
    maxTokens: config.maxTokens,
  };
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
  const stats = record.session.getSessionStats();
  if (stats.tokens.total + record.maxTokens > record.sessionTokenBudget) {
    throw Object.assign(
      new Error("Pi shell session token budget is exhausted"),
      {
        code: "pi_shell_token_budget_exhausted",
        details: {
          consumed_tokens: stats.tokens.total,
          reserved_output_tokens: record.maxTokens,
          token_budget: record.sessionTokenBudget,
        },
      },
    );
  }
  activeRequestBySession.set(sessionId, requestId);
  try {
    await record.session.prompt(message);
    return sessionState(record);
  } finally {
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
        model: String(process.env.EASYICU_PI_MODEL || "gpt5.6 luna"),
        provider: String(process.env.EASYICU_PI_PROVIDER || "easyicu-local"),
        built_in_tools_enabled: [],
        custom_tools: [...TOOL_NAMES],
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
