// Accounting receipts share the existing Pi session journal and budget call IDs.
// This projection never releases or changes a ShellBudgetGuard reservation.
const SCHEMA = "easyicu.shell-usage/1";
const MAX_CALLS = 10000;
const validNumber = value => typeof value === "number" && Number.isFinite(value) && value >= 0;
const validCount = value => Number.isSafeInteger(value) && value >= 0;
const round = value => Math.round(value * 1e8) / 1e8;

function totals(rows) {
  const reported = rows.filter(row => row.input_tokens !== null && row.output_tokens !== null);
  return {
    calls: rows.length,
    accounted_tokens: rows.reduce((sum, row) => sum + row.accounted_tokens, 0),
    estimated_cost_usd: rows.some(row => row.estimated_cost_usd === null) ? null : round(rows.reduce((sum, row) => sum + row.estimated_cost_usd, 0)),
    provider_reported_calls: reported.length,
    provider_reported_tokens: reported.reduce((sum, row) => sum + row.input_tokens + row.output_tokens, 0),
    provider_reported_prompt_tokens: reported.reduce((sum, row) => sum + row.input_tokens, 0),
    provider_reported_completion_tokens: reported.reduce((sum, row) => sum + row.output_tokens, 0),
    usage_unknown_calls: rows.length - reported.length,
    failed_calls: rows.filter(row => row.state === "failed").length,
    in_flight_calls: rows.filter(row => row.state === "in_flight").length,
  };
}

export class ShellUsageLedger {
  constructor({ entries = [], persist }) {
    this.rows = new Map();
    this.persist = persist;
    this.valid = true;
    let count = 0;
    for (const entry of entries) {
      if (entry?.type !== "custom" || entry.customType !== SCHEMA) continue;
      if (++count > MAX_CALLS * 2 || !this.accept(entry.data)) { this.valid = false; break; }
    }
  }

  accept(row) {
    if (!row || !validCount(row.call_id) || row.call_id < 1 || row.call_id > MAX_CALLS
        || !["in_flight", "completed", "failed"].includes(row.state)
        || !validCount(row.accounted_tokens)
        || !(row.estimated_cost_usd === null || validNumber(row.estimated_cost_usd))
        || ![row.input_tokens, row.output_tokens].every(value => value === null || validCount(value))) return false;
    const previous = this.rows.get(row.call_id);
    if (previous ? previous.state !== "in_flight" || row.state === "in_flight" : row.state !== "in_flight") return false;
    this.rows.set(row.call_id, { ...row });
    return true;
  }

  write(row) {
    this.persist(row);
    if (!this.accept(row)) this.valid = false;
  }

  async track(open, reservation, pricing) {
    const row = {
      call_id: reservation.session_provider_call, state: "in_flight",
      input_tokens: null, output_tokens: null,
      accounted_tokens: reservation.input_token_upper_bound + reservation.reserved_output_tokens,
      estimated_cost_usd: pricing ? reservation.call_reserved_cost_micro_usd / 1e6 : null,
    };
    this.write(row); // Durable before the provider opens, just like the budget receipt.
    let finished = false;
    const finish = (state, usage) => {
      if (finished) return;
      finished = true;
      const components = [usage?.input, usage?.output, usage?.cacheRead ?? 0, usage?.cacheWrite ?? 0];
      const measured = state === "completed" && components.every(validCount) && components.some(value => value > 0);
      const input = measured ? components[0] + components[2] + components[3] : null;
      const output = measured ? components[1] : null;
      this.write({ ...row, state, input_tokens: input, output_tokens: output,
        accounted_tokens: measured ? input + output : row.accounted_tokens,
        estimated_cost_usd: measured && pricing ? (input * pricing.input_price + output * pricing.output_price) / 1e6 : row.estimated_cost_usd,
      });
    };
    let stream;
    try { stream = await open(); } catch (error) { finish("failed"); throw error; }
    return {
      async *[Symbol.asyncIterator]() {
        try {
          for await (const event of stream) {
            if (event.type === "done") {
              finish(["error", "aborted"].includes(event.message?.stopReason) ? "failed" : "completed", event.message?.usage);
            } else if (event.type === "error") finish("failed");
            yield event;
          }
        } finally { if (!finished) finish("failed"); }
      },
      result: () => stream.result(),
    };
  }

  summary(providerCalls) {
    if (!this.valid || !validCount(providerCalls) || providerCalls !== this.rows.size
        || [...this.rows.keys()].some(id => id > providerCalls)) return {
      schema_version: SCHEMA, accounting_complete: false,
      reason_code: "conversation_usage_history_unavailable", calls: null, estimated_cost_usd: null,
    };
    const rows = [...this.rows.values()];
    return {
      schema_version: SCHEMA, accounting_complete: true,
      cost_kind: "ledger_estimate_with_unreported_reservations", ...totals(rows),
      by_state: Object.fromEntries(["completed", "failed", "in_flight", "unknown"].map(state => [state, totals(rows.filter(row => row.state === state))])),
    };
  }
}
