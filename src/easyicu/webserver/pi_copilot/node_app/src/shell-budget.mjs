import { Buffer } from "node:buffer";
import { createHash } from "node:crypto";

const STRUCTURAL_TOKEN_RESERVE = 4096;
const MIN_SESSION_TOKEN_BUDGET = 1_000_000;
const DEFAULT_SESSION_CONTEXT_MULTIPLIER = 20;
export const SHELL_BUDGET_RECEIPT = "easyicu.shell-budget/1";
export const SHELL_BUDGET_RECEIPT_V2 = "easyicu.shell-budget/2";
const MICRO_USD_PER_USD = 1_000_000;

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

export function defaultShellSessionTokenBudget(contextWindow) {
  const boundedContext = integer(contextWindow, 200_000, 8192);
  return Math.max(
    MIN_SESSION_TOKEN_BUDGET,
    boundedContext * DEFAULT_SESSION_CONTEXT_MULTIPLIER,
  );
}

function budgetError(code, message, details) {
  return Object.assign(new Error(`${code}: ${message}`), { code, details });
}

function finiteNumber(value, label, { minimum = 0, positive = false } = {}) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || parsed < minimum || (positive && parsed <= 0)) {
    throw budgetError(
      "pi_shell_pricing_invalid",
      `${label} must be a finite ${positive ? "positive" : "non-negative"} number.`,
    );
  }
  return parsed;
}

function microUsd(value, label, { positive = false } = {}) {
  const parsed = finiteNumber(value, label, { positive });
  const converted = Math.floor((parsed * MICRO_USD_PER_USD) + 1e-9);
  if (!Number.isSafeInteger(converted)) {
    throw budgetError("pi_shell_pricing_invalid", `${label} exceeds the supported range.`);
  }
  return converted;
}

function normalizePricing(pricing) {
  if (pricing === null || pricing === undefined) return null;
  const normalized = {
    input_price_usd_per_million_tokens: finiteNumber(
      pricing.inputPriceUsdPerMillionTokens,
      "inputPriceUsdPerMillionTokens",
    ),
    output_price_usd_per_million_tokens: finiteNumber(
      pricing.outputPriceUsdPerMillionTokens,
      "outputPriceUsdPerMillionTokens",
    ),
    max_cost_usd_per_message: finiteNumber(
      pricing.maxCostUsdPerMessage,
      "maxCostUsdPerMessage",
      { positive: true },
    ),
    max_cost_usd_per_session: finiteNumber(
      pricing.maxCostUsdPerSession,
      "maxCostUsdPerSession",
      { positive: true },
    ),
  };
  if (
    normalized.input_price_usd_per_million_tokens === 0
    && normalized.output_price_usd_per_million_tokens === 0
  ) {
    throw budgetError(
      "pi_shell_pricing_invalid",
      "At least one provider token price must be positive.",
    );
  }
  if (normalized.max_cost_usd_per_message > normalized.max_cost_usd_per_session) {
    throw budgetError(
      "pi_shell_pricing_invalid",
      "The per-message cost ceiling cannot exceed the session cost ceiling.",
    );
  }
  return Object.freeze({
    ...normalized,
    input_price: normalized.input_price_usd_per_million_tokens,
    output_price: normalized.output_price_usd_per_million_tokens,
    max_message_micro_usd: microUsd(
      normalized.max_cost_usd_per_message,
      "maxCostUsdPerMessage",
      { positive: true },
    ),
    max_session_micro_usd: microUsd(
      normalized.max_cost_usd_per_session,
      "maxCostUsdPerSession",
      { positive: true },
    ),
  });
}

function pricingBindingSha256(pricing) {
  if (!pricing) return null;
  const canonical = JSON.stringify({
    input_price_usd_per_million_tokens:
      pricing.input_price_usd_per_million_tokens.toFixed(6),
    output_price_usd_per_million_tokens:
      pricing.output_price_usd_per_million_tokens.toFixed(6),
    max_cost_usd_per_message: pricing.max_cost_usd_per_message.toFixed(6),
    max_cost_usd_per_session: pricing.max_cost_usd_per_session.toFixed(6),
  });
  return createHash("sha256").update(canonical, "utf8").digest("hex");
}

export function restoredShellBudgetState(
  entries,
  { fallbackProviderCalls = 0, pricingBinding = undefined } = {},
) {
  const rows = Array.isArray(entries) ? entries : [];
  for (let index = rows.length - 1; index >= 0; index -= 1) {
    const entry = rows[index];
    if (
      entry?.type !== "custom"
      || !new Set([SHELL_BUDGET_RECEIPT, SHELL_BUDGET_RECEIPT_V2]).has(entry.customType)
    ) continue;
    const calls = Number.parseInt(String(entry.data?.provider_calls ?? ""), 10);
    if (!Number.isFinite(calls) || calls < 0) continue;
    if (entry.customType === SHELL_BUDGET_RECEIPT_V2) {
      const reservedCost = Number.parseInt(
        String(entry.data?.reserved_cost_micro_usd ?? ""),
        10,
      );
      const receiptBinding = String(entry.data?.pricing_binding_sha256 || "").trim();
      if (!Number.isSafeInteger(reservedCost) || reservedCost < 0 || !receiptBinding) {
        throw budgetError(
          "pi_shell_budget_receipt_invalid",
          "The persisted Pi shell budget receipt is incomplete.",
        );
      }
      if (pricingBinding === null || (pricingBinding && receiptBinding !== pricingBinding)) {
        throw budgetError(
          "pi_shell_pricing_binding_mismatch",
          "The current provider pricing no longer matches the persisted session budget.",
        );
      }
      return {
        providerCalls: calls,
        reservedCostMicroUsd: typeof pricingBinding === "string" ? reservedCost : 0,
      };
    }
    if (pricingBinding && calls > 0) {
      throw budgetError(
        "pi_shell_cost_history_unavailable",
        "This existing Copilot session predates cost receipts and cannot adopt a priced budget.",
      );
    }
    return { providerCalls: calls, reservedCostMicroUsd: 0 };
  }
  return {
    providerCalls: integer(fallbackProviderCalls, 0, 0),
    reservedCostMicroUsd: 0,
  };
}

export function restoredProviderCallCount(entries, fallback = 0) {
  return restoredShellBudgetState(entries, { fallbackProviderCalls: fallback }).providerCalls;
}

export function providerCallReceipt(providerCalls) {
  return {
    schema_version: SHELL_BUDGET_RECEIPT,
    provider_calls: integer(providerCalls, 0, 0),
  };
}

export class ShellBudgetGuard {
  constructor({
    tokenBudget,
    maxOutputTokens,
    maxProviderCallsPerMessage,
    maxProviderCallsPerSession,
    consumedTokens,
    initialProviderCalls = 0,
    persistedEntries = [],
    pricing = null,
  }) {
    this.tokenBudget = integer(tokenBudget, 1_000_000, 1);
    this.maxOutputTokens = integer(maxOutputTokens, 16_384, 1);
    this.maxProviderCallsPerMessage = integer(maxProviderCallsPerMessage, 8, 1);
    this.maxProviderCallsPerSession = integer(maxProviderCallsPerSession, 128, 1);
    this.consumedTokens = consumedTokens;
    this.pricing = normalizePricing(pricing);
    this.pricingBindingSha256 = pricingBindingSha256(this.pricing);
    const restored = restoredShellBudgetState(persistedEntries, {
      fallbackProviderCalls: initialProviderCalls,
      pricingBinding: this.pricingBindingSha256,
    });
    this.sessionProviderCalls = restored.providerCalls;
    this.sessionReservedCostMicroUsd = restored.reservedCostMicroUsd;
    this.messageProviderCalls = 0;
    this.messageReservedCostMicroUsd = 0;
    this.messageActive = false;
  }

  beginMessage() {
    this.messageProviderCalls = 0;
    this.messageReservedCostMicroUsd = 0;
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
        "The provider-call ceiling for this Copilot session is exhausted.",
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

    let reservedCostMicroUsd = 0;
    if (this.pricing) {
      reservedCostMicroUsd = Math.ceil(
        (inputUpperBound * this.pricing.input_price)
        + (reservedOutput * this.pricing.output_price),
      );
      if (
        this.messageReservedCostMicroUsd + reservedCostMicroUsd
        > this.pricing.max_message_micro_usd
      ) {
        throw budgetError(
          "pi_shell_message_cost_budget_exhausted",
          "The conservative provider-cost ceiling for this Pi message is exhausted.",
          {
            reserved_cost_micro_usd: this.messageReservedCostMicroUsd,
            next_call_reserved_cost_micro_usd: reservedCostMicroUsd,
            cost_budget_micro_usd: this.pricing.max_message_micro_usd,
          },
        );
      }
      if (
        this.sessionReservedCostMicroUsd + reservedCostMicroUsd
        > this.pricing.max_session_micro_usd
      ) {
        throw budgetError(
          "pi_shell_session_cost_budget_exhausted",
          "The conservative provider-cost ceiling for this Copilot session is exhausted.",
          {
            reserved_cost_micro_usd: this.sessionReservedCostMicroUsd,
            next_call_reserved_cost_micro_usd: reservedCostMicroUsd,
            cost_budget_micro_usd: this.pricing.max_session_micro_usd,
          },
        );
      }
    }

    this.messageProviderCalls += 1;
    this.sessionProviderCalls += 1;
    this.messageReservedCostMicroUsd += reservedCostMicroUsd;
    this.sessionReservedCostMicroUsd += reservedCostMicroUsd;
    return {
      message_provider_call: this.messageProviderCalls,
      session_provider_call: this.sessionProviderCalls,
      input_token_upper_bound: inputUpperBound,
      reserved_output_tokens: reservedOutput,
      call_reserved_cost_micro_usd: reservedCostMicroUsd,
      message_reserved_cost_micro_usd: this.messageReservedCostMicroUsd,
      session_reserved_cost_micro_usd: this.sessionReservedCostMicroUsd,
    };
  }

  receipt() {
    if (!this.pricing || !this.pricingBindingSha256) {
      return providerCallReceipt(this.sessionProviderCalls);
    }
    return {
      schema_version: SHELL_BUDGET_RECEIPT_V2,
      provider_calls: this.sessionProviderCalls,
      reserved_cost_micro_usd: this.sessionReservedCostMicroUsd,
      pricing_binding_sha256: this.pricingBindingSha256,
    };
  }

  state() {
    const state = {
      provider_calls: this.sessionProviderCalls,
      provider_call_budget: this.maxProviderCallsPerSession,
      message_provider_calls: this.messageProviderCalls,
      message_provider_call_budget: this.maxProviderCallsPerMessage,
    };
    if (!this.pricing) {
      return {
        ...state,
        cost: null,
        pricing_available: false,
      };
    }
    return {
      ...state,
      cost: this.sessionReservedCostMicroUsd / MICRO_USD_PER_USD,
      cost_is_conservative_reservation: true,
      pricing_available: true,
      cost_budget_usd: this.pricing.max_cost_usd_per_session,
      cost_remaining_usd: Math.max(
        0,
        this.pricing.max_session_micro_usd - this.sessionReservedCostMicroUsd,
      ) / MICRO_USD_PER_USD,
      message_cost_usd: this.messageReservedCostMicroUsd / MICRO_USD_PER_USD,
      message_cost_budget_usd: this.pricing.max_cost_usd_per_message,
    };
  }
}
