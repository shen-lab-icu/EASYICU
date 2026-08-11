/* Safe, dependency-neutral projection from Pi SDK events/session messages to
   EasyICU's browser contract. Raw reasoning, tool arguments, and partial tool
   output intentionally never cross this boundary. */

const MAX_TEXT_CHARS = 12000;
const WORKSPACE_FILE_TOOLS = new Set([
  "easyicu_read_project_file",
  "easyicu_write_project_file",
  "easyicu_edit_project_file",
  "easyicu_check_project_file",
  "easyicu_preview_project_file",
]);

function boundedText(value, limit = MAX_TEXT_CHARS) {
  return String(value ?? "").slice(0, limit);
}

function modelErrorCode(message) {
  if (!message || message.stopReason !== "error") return "";
  const detail = String(message.errorMessage || "").toLowerCase();
  if (/context|token.*limit|maximum.*length/.test(detail)) return "pi_model_context_limit";
  if (/rate.?limit|too many requests|quota/.test(detail)) return "pi_model_rate_limited";
  if (/timeout|timed out|connection reset|server_error|internal_server_error|\b5\d\d\b/.test(detail)) {
    return "pi_model_provider_unavailable";
  }
  return "pi_model_provider_error";
}

function eventTimestamp(event) {
  const value = Number(event?.timestamp ?? event?.message?.timestamp);
  return Number.isFinite(value) && value > 0
    ? new Date(value).toISOString()
    : new Date().toISOString();
}

function safeRelativeFile(value) {
  const file = boundedText(value, 240).trim().replaceAll("\\", "/");
  if (!file || file.startsWith("/") || file.includes("\0")) return "";
  const parts = file.split("/");
  if (parts.some((part) => !part || part === "." || part === "..")) return "";
  return file;
}

function safeStableId(value, limit = 160) {
  const text = boundedText(value, limit).trim();
  return /^[A-Za-z][A-Za-z0-9_.-]{0,159}$/.test(text) ? text : "";
}

function safeJobId(value) {
  const text = boundedText(value, 160).trim();
  return /^[A-Za-z0-9][A-Za-z0-9_.-]{0,159}$/.test(text) ? text : "";
}

function safeArtifactName(value) {
  const name = boundedText(value, 160).trim();
  if (!name || name.includes("/") || name.includes("\\") || !name.endsWith(".json")) return "";
  return /^[A-Za-z0-9_.-]+$/.test(name) ? name : "";
}

function projectedResource(value) {
  if (!value || typeof value !== "object" || Array.isArray(value)) return undefined;
  if (value.kind === "research_artifact") {
    const runId = safeStableId(value.run_id);
    const artifact = safeArtifactName(value.artifact);
    if (!runId || !artifact) return undefined;
    return {
      kind: "research_artifact",
      run_id: runId,
      artifact,
      label: boundedText(value.label || artifact, 160),
      media_type: "application/json",
    };
  }
  const file = safeRelativeFile(value.file);
  if (!file) return undefined;
  const kind = value.kind === "webpage" ? "webpage" : "file";
  const fallbackLabel = file.split("/").at(-1) || file;
  return {
    kind,
    file,
    label: boundedText(value.label || fallbackLabel, 160),
    media_type: boundedText(value.media_type || "text/plain", 120),
  };
}

function projectedResources(value) {
  if (!Array.isArray(value)) return [];
  return value.slice(0, 80).map(projectedResource).filter(Boolean);
}

function toolResource(toolName, args) {
  const name = boundedText(toolName, 160);
  if (!WORKSPACE_FILE_TOOLS.has(name)) return undefined;
  let params = args;
  if (typeof params === "string") {
    try { params = JSON.parse(params); } catch { return undefined; }
  }
  return projectedResource({
    kind: name === "easyicu_preview_project_file" ? "webpage" : "file",
    file: params?.file,
    media_type: name === "easyicu_preview_project_file" ? "text/html" : "text/plain",
  });
}

function toolReceipt(result) {
  const details = result?.details && typeof result.details === "object"
    ? result.details
    : {};
  const ownerDetails = details.details && typeof details.details === "object"
    ? details.details
    : {};
  const content = Array.isArray(result?.content) ? result.content : [];
  const fallback = content.find((item) => item && item.type === "text")?.text || "";
  const resource = projectedResource(details.resource || ownerDetails.resource);
  const resources = projectedResources(details.resources || ownerDetails.resources);
  const jobId = safeJobId(ownerDetails.job_id || details.job_id);
  return {
    status: boundedText(details.status || "", 40),
    code: boundedText(details.code || "", 160),
    summary: boundedText(details.summary || fallback, 2000),
    owner: boundedText(details.owner || "", 240),
    ...(jobId ? { job_id: jobId } : {}),
    ...(resource ? { resource } : {}),
    ...(resources.length ? { resources } : {}),
  };
}

export function normalizePiEvent(event) {
  if (!event || typeof event !== "object") return undefined;
  const at = eventTimestamp(event);
  if (event.type === "agent_start") return { type: "run_start", at };
  if (event.type === "turn_start") {
    return { type: "turn_start", at, turn_index: Number(event.turnIndex || 0) };
  }
  if (event.type === "message_start" && event.message?.role === "assistant") {
    return { type: "assistant_start", at };
  }
  if (event.type === "message_update") {
    const update = event.assistantMessageEvent || {};
    if (update.type === "text_delta") {
      return { type: "text_delta", at, delta: boundedText(update.delta, 8000) };
    }
    return undefined;
  }
  if (event.type === "tool_execution_start") {
    const resource = toolResource(event.toolName, event.args);
    return {
      type: "tool_start",
      at,
      tool_call_id: boundedText(event.toolCallId, 160),
      tool_name: boundedText(event.toolName, 160),
      ...(resource ? { resource } : {}),
    };
  }
  if (event.type === "tool_execution_update") {
    return {
      type: "tool_progress",
      at,
      tool_call_id: boundedText(event.toolCallId, 160),
      tool_name: boundedText(event.toolName, 160),
    };
  }
  if (event.type === "tool_execution_end") {
    const receipt = toolReceipt(event.result);
    return {
      type: "tool_end",
      at,
      tool_call_id: boundedText(event.toolCallId, 160),
      tool_name: boundedText(event.toolName, 160),
      is_error: Boolean(
        event.isError || receipt.status === "blocked" || receipt.status === "failed",
      ),
      ...receipt,
    };
  }
  if (event.type === "message_end" && event.message?.role === "assistant") {
    const errorCode = modelErrorCode(event.message);
    return {
      type: "message_end",
      at,
      stop_reason: boundedText(event.message.stopReason || "complete", 80),
      ...(errorCode ? { error_code: errorCode } : {}),
    };
  }
  if (event.type === "turn_end") {
    return { type: "turn_end", at, turn_index: Number(event.turnIndex || 0) };
  }
  if (event.type === "agent_end") {
    return { type: "agent_cycle_end", at, will_retry: Boolean(event.willRetry) };
  }
  if (event.type === "agent_settled") return { type: "run_end", at };
  if (event.type === "compaction_start") {
    return { type: "compaction_start", at, reason: boundedText(event.reason, 80) };
  }
  if (event.type === "compaction_end") {
    return {
      type: "compaction_end",
      at,
      reason: boundedText(event.reason, 80),
      aborted: Boolean(event.aborted),
    };
  }
  if (event.type === "auto_retry_start") {
    return {
      type: "retry",
      at,
      attempt: Number(event.attempt || 0),
      max_attempts: Number(event.maxAttempts || 0),
    };
  }
  return undefined;
}

export function projectTranscriptMessage(message) {
  if (!message || typeof message !== "object") return undefined;
  const role = boundedText(message.role, 40);
  if (!new Set(["user", "assistant", "toolResult"]).has(role)) return undefined;
  const content = Array.isArray(message.content)
    ? message.content
    : [{ type: "text", text: message.content }];
  const timestamp = eventTimestamp({ message });

  if (role === "toolResult") {
    const receipt = toolReceipt(message);
    return {
      role: "tool",
      timestamp,
      content: [{
        type: "tool_result",
        tool_call_id: boundedText(message.toolCallId, 160),
        tool_name: boundedText(message.toolName, 160),
        is_error: Boolean(
          message.isError || receipt.status === "blocked" || receipt.status === "failed",
        ),
        ...receipt,
      }],
    };
  }

  const parts = [];
  for (const item of content.slice(0, 80)) {
    if (!item || typeof item !== "object") continue;
    if (item.type === "text") {
      parts.push({ type: "text", text: boundedText(item.text, MAX_TEXT_CHARS) });
    } else if (item.type === "toolCall") {
      const resource = toolResource(item.name, item.arguments);
      parts.push({
        type: "tool_call",
        tool_call_id: boundedText(item.id, 160),
        tool_name: boundedText(item.name, 160),
        ...(resource ? { resource } : {}),
      });
    }
  }
  return {
    role,
    timestamp,
    content: parts,
    stop_reason: role === "assistant"
      ? boundedText(message.stopReason || "", 80)
      : undefined,
    ...(role === "assistant" && modelErrorCode(message)
      ? { error_code: modelErrorCode(message) }
      : {}),
  };
}
