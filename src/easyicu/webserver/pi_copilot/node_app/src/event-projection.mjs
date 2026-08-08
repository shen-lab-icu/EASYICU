/* Safe, dependency-neutral projection from Pi SDK events/session messages to
   EasyICU's browser contract. Raw reasoning, tool arguments, and partial tool
   output intentionally never cross this boundary. */

const MAX_TEXT_CHARS = 12000;

function boundedText(value, limit = MAX_TEXT_CHARS) {
  return String(value ?? "").slice(0, limit);
}

function eventTimestamp(event) {
  const value = Number(event?.timestamp ?? event?.message?.timestamp);
  return Number.isFinite(value) && value > 0
    ? new Date(value).toISOString()
    : new Date().toISOString();
}

function toolReceipt(result) {
  const details = result?.details && typeof result.details === "object"
    ? result.details
    : {};
  const content = Array.isArray(result?.content) ? result.content : [];
  const fallback = content.find((item) => item && item.type === "text")?.text || "";
  return {
    status: boundedText(details.status || "", 40),
    code: boundedText(details.code || "", 160),
    summary: boundedText(details.summary || fallback, 2000),
    owner: boundedText(details.owner || "", 240),
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
    return {
      type: "tool_start",
      at,
      tool_call_id: boundedText(event.toolCallId, 160),
      tool_name: boundedText(event.toolName, 160),
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
      is_error: Boolean(event.isError),
      ...receipt,
    };
  }
  if (event.type === "message_end" && event.message?.role === "assistant") {
    return {
      type: "message_end",
      at,
      stop_reason: boundedText(event.message.stopReason || "complete", 80),
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
        is_error: Boolean(message.isError),
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
      parts.push({
        type: "tool_call",
        tool_call_id: boundedText(item.id, 160),
        tool_name: boundedText(item.name, 160),
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
  };
}
