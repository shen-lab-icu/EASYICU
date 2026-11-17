import { setTimeout as sleep } from "node:timers/promises";

import { modelErrorCode } from "./event-projection.mjs";

const RETRY_DELAYS_MS = [2000, 4000, 8000, 16000, 30000, 30000, 30000];
const RETRY_WINDOW_MS = 120000;

function hasContent(event) {
  // Opening a content block commits this attempt even before its first delta.
  if (/^(text|thinking|toolcall)_(start|delta|end)$/.test(event?.type || "")) return true;
  return [event?.partial, event?.message, event?.error].some((message) => (
    Array.isArray(message?.content) && message.content.some((part) => (
      (part.type === "text" && Boolean(part.text))
      || (part.type === "thinking" && Boolean(part.thinking))
      || part.type === "toolCall"
    ))
  ));
}

function isTransient(error) {
  if (error?.stopReason && error.stopReason !== "error") return false;
  const detail = `${error?.errorMessage || error?.message || error} ${error?.code || ""}`;
  const status = Number(error?.status ?? error?.statusCode ?? error?.response?.status ?? detail.match(
    /(?:^\s*|\bhttp(?:\/[\d.]+)?(?: error)?\s+|\bstatus(?: code)?\s*[:=]?\s*)([45]\d\d)\b/i,
  )?.[1]);
  if (status >= 400 && status < 500 && status !== 408 && status !== 429) return false;
  // A rate limit may recover; exhausted credits and host budgets cannot.
  if (/quota|budget|authenticat|authoriz|credential|api[ _-]?key|permission|forbidden/i.test(detail)) return false;
  if (status === 408 || status === 429) return true;
  if (status && ![500, 502, 503, 504].includes(status)) return false;
  return modelErrorCode({
    stopReason: error?.stopReason ?? "error",
    errorMessage: status ? `HTTP ${status}: ${detail}` : detail,
  }) === "pi_model_provider_unavailable"
    || (!status && /fetch failed|network error|econnrefused|econnaborted|enetunreach|ehostunreach|enotfound|eai_again|socket hang up|connection (?:error|closed|refused)/i.test(detail));
}

function replayPrefix(stream, iterator, prefix) {
  return {
    async *[Symbol.asyncIterator]() {
      try {
        yield* prefix;
        while (true) {
          const next = await iterator.next();
          if (next.done) return;
          yield next.value;
        }
      } finally {
        await iterator.return?.();
      }
    },
    result: () => stream.result(),
  };
}

/** Retry only empty attempts; every open is host-guarded, every wait abortable. */
export async function retryTransientProviderStream(
  openAttempt,
  { signal, sleep: wait = sleep, now = () => performance.now(), onRetry,
    retryAllowed = () => true, onCommit } = {},
) {
  const deadline = now() + RETRY_WINDOW_MS;
  // Leave a margin for the final probe instead of sleeping to the deadline
  // and losing that probe to an ordinary late timer wakeup.
  const lastRetryStart = deadline - 1000;
  for (let attempt = 0; ; attempt += 1) {
    let iterator;
    let stream;
    let failure;
    const prefix = [];
    const canRetry = () => attempt < RETRY_DELAYS_MS.length && now() < lastRetryStart
      && !signal?.aborted && retryAllowed();
    try {
      stream = await openAttempt();
      iterator = stream[Symbol.asyncIterator]();
      while (true) {
        const next = await iterator.next();
        if (next.done) return replayPrefix(stream, iterator, prefix);
        const event = next.value;
        prefix.push(event);
        if (hasContent(event) || event.type === "done") {
          onCommit?.();
          return replayPrefix(stream, iterator, prefix);
        }
        if (event.type === "error") {
          if (canRetry() && isTransient(event.error) && !prefix.some(hasContent)) break;
          return replayPrefix(stream, iterator, prefix);
        }
      }
    } catch (error) {
      if (!canRetry() || !isTransient(error) || prefix.some(hasContent)) throw error;
      failure = error;
    }
    // Discard this failed prefix before announcing a retry. Counts describe
    // retries, not total calls; no provider error detail crosses here.
    await iterator?.return?.();
    signal?.throwIfAborted();
    onRetry?.({ type: "auto_retry_start", attempt: attempt + 1, maxAttempts: RETRY_DELAYS_MS.length });
    await wait(Math.min(RETRY_DELAYS_MS[attempt], Math.max(0, lastRetryStart - now())), undefined, { signal });
    signal?.throwIfAborted();
    if (now() > deadline || !retryAllowed()) {
      if (failure) throw failure;
      return replayPrefix(stream, iterator, prefix);
    }
  }
}
