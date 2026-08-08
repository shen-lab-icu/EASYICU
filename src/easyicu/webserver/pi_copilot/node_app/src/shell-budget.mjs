import { Buffer } from "node:buffer";

const STRUCTURAL_TOKEN_RESERVE = 4096;

function integer(value, fallback, minimum = 0) {
  const parsed = Number.parseInt(String(value ?? ""), 10);
  return Number.isFinite(parsed) ? Math.max(minimum, parsed) : fallback;
}
export function estimatedInputTokenUpperBound(context) {
  let encoded;
  try {
    encoded = JSON.stringify(context ?? {});
  } catch {
    encoded = String(context ?? "");
  }
  // Pi is text-only in EasyICU. UTF-8 bytes are a conservative ceiling for
  // byte-tokenized text; the fixed reserve covers message/tool framing.
  return Buffer.byteLength(encoded, "utf8") + STRUCTURAL_TOKEN_RESERVE;
}

function budgetError(code, message, details) {
  return Object.assign(new Error(`${code}: ${message}`), { code, details });
}

export class ShellBudgetGuard {
  constructor({
    tokenBudget,
    maxOutputTokens,
    maxProviderCallsPerMessage,
    maxProviderCallsPerSession,
    consumedTokens,
    initialProviderCalls = 0,
  }) {
    this.tokenBudget = integer(tokenBudget, 1_000_000, 1);
    this.maxOutputTokens = integer(maxOutputTokens, 16_384, 1);
    this.maxProviderCallsPerMessage = integer(maxProviderCallsPerMessage, 8, 1);
    this.maxProviderCallsPerSession = integer(maxProviderCallsPerSession, 128, 1);
    this.consumedTokens = consumedTokens;
    this.sessionProviderCalls = integer(initialProviderCalls, 0, 0);
    this.messageProviderCalls = 0;
    this.messageActive = false;
  }

  beginMessage() {
    this.messageProviderCalls = 0;
    this.messageActive = true;
  }

  endMessage() {
    this.messageActive = false;
  }

  authorize(context, options = {}) {
    if (!this.messageActive) {
      throw budgetError(
        "pi_shell_provider_call_without_message",
        "Provider calls require an active host-owned Pi message.",
      );
    }
    if (this.messageProviderCalls >= this.maxProviderCallsPerMessage) {
      throw budgetError(
        "pi_shell_message_provider_call_budget_exhausted",
        "The provider-call ceiling for this Pi message is exhausted.",
        {
          provider_calls: this.messageProviderCalls,
          provider_call_budget: this.maxProviderCallsPerMessage,
        },
      );
    }
    if (this.sessionProviderCalls >= this.maxProviderCallsPerSession) {
      throw budgetError(
        "pi_shell_session_provider_call_budget_exhausted",
        "The provider-call ceiling for this Pi session is exhausted.",
        {
          provider_calls: this.sessionProviderCalls,
          provider_call_budget: this.maxProviderCallsPerSession,
        },
      );
    }

    const consumed = integer(this.consumedTokens?.(), 0, 0);
    const inputUpperBound = estimatedInputTokenUpperBound(context);
    const requestedOutput = integer(options?.maxTokens, this.maxOutputTokens, 1);
    const reservedOutput = Math.min(this.maxOutputTokens, requestedOutput);
    const reservedTotal = consumed + inputUpperBound + reservedOutput;
    if (reservedTotal > this.tokenBudget) {
      throw budgetError(
        "pi_shell_token_budget_exhausted",
        "The Pi shell session token budget cannot authorize another provider call.",
        {
          consumed_tokens: consumed,
          input_token_upper_bound: inputUpperBound,
          reserved_output_tokens: reservedOutput,
          reserved_total_tokens: reservedTotal,
          token_budget: this.tokenBudget,
        },
      );
    }

    this.messageProviderCalls += 1;
    this.sessionProviderCalls += 1;
    return {
      message_provider_call: this.messageProviderCalls,
      session_provider_call: this.sessionProviderCalls,
      input_token_upper_bound: inputUpperBound,
      reserved_output_tokens: reservedOutput,
    };
  }

  state() {
    return {
      provider_calls: this.sessionProviderCalls,
      provider_call_budget: this.maxProviderCallsPerSession,
      message_provider_calls: this.messageProviderCalls,
      message_provider_call_budget: this.maxProviderCallsPerMessage,
    };
  }
}
