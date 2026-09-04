import { totalmem } from "node:os";

const MIB = 1024 * 1024;

function boundedInteger(value, fallback, minimum, maximum) {
  const parsed = Number.parseInt(String(value ?? ""), 10);
  const selected = Number.isFinite(parsed) ? parsed : fallback;
  return Math.max(minimum, Math.min(maximum, selected));
}

export function sessionLifecycleConfig(
  environ = process.env,
  { totalMemoryBytes = totalmem() } = {},
) {
  const totalMemoryMb = Math.max(1024, Math.floor(Number(totalMemoryBytes) / MIB));
  const defaultSoftRssMb = Math.max(512, Math.min(2048, Math.floor(totalMemoryMb * 0.10)));
  const defaultEmergencyRssMb = Math.max(
    defaultSoftRssMb + 256,
    Math.min(3072, Math.floor(totalMemoryMb * 0.15)),
  );
  const softRssMb = boundedInteger(
    environ.EASYICU_PI_SOFT_RSS_MB,
    defaultSoftRssMb,
    256,
    1_000_000,
  );
  return Object.freeze({
    maxOpenSessions: boundedInteger(
      environ.EASYICU_PI_MAX_OPEN_SESSIONS,
      8,
      1,
      100,
    ),
    idleMs: boundedInteger(
      environ.EASYICU_PI_SESSION_IDLE_SECONDS,
      30 * 60,
      60,
      7 * 24 * 60 * 60,
    ) * 1000,
    softRssBytes: softRssMb * MIB,
    emergencyRssBytes: boundedInteger(
      environ.EASYICU_PI_EMERGENCY_RSS_MB,
      defaultEmergencyRssMb,
      softRssMb + 64,
      1_000_000,
    ) * MIB,
  });
}

function memoryPressureError(status) {
  return Object.assign(
    new Error("The Copilot runtime is under memory pressure; retry after current work finishes."),
    {
      code: "pi_shell_memory_pressure",
      details: status,
    },
  );
}

function sessionCapacityError(status) {
  return Object.assign(
    new Error("All Copilot runtime session slots are active; retry after a turn finishes."),
    {
      code: "pi_shell_session_capacity",
      details: status,
    },
  );
}

export class HotSessionLifecycle {
  constructor({
    sessions,
    activeRequests,
    config,
    now = () => Date.now(),
    rssBytes = () => process.memoryUsage().rss,
  }) {
    this.sessions = sessions;
    this.activeRequests = activeRequests;
    this.config = config;
    this.now = now;
    this.rssBytes = rssBytes;
    this.suspendedSessions = 0;
  }

  touch(sessionId) {
    const record = this.sessions.get(sessionId);
    if (record) record.lastAccessedAt = this.now();
    return record;
  }

  _candidates(excludeSessionId = "") {
    return [...this.sessions.values()]
      .filter((record) => (
        record.externalId !== excludeSessionId
        && !this.activeRequests.has(record.externalId)
        && !record.session?.isStreaming
      ))
      .sort((left, right) => (
        Number(left.lastAccessedAt || 0) - Number(right.lastAccessedAt || 0)
      ));
  }

  dispose(sessionId, { force = false, suspension = true } = {}) {
    const record = this.sessions.get(sessionId);
    if (!record) return false;
    if (!force && (
      this.activeRequests.has(sessionId)
      || record.session?.isStreaming
    )) return false;
    try { record.unsubscribe?.(); } catch {}
    try { record.session?.dispose(); } catch {}
    this.sessions.delete(sessionId);
    if (suspension) this.suspendedSessions += 1;
    return true;
  }

  suspendIdle({ excludeSessionId = "" } = {}) {
    const cutoff = this.now() - this.config.idleMs;
    let suspended = 0;
    for (const record of this._candidates(excludeSessionId)) {
      if (Number(record.lastAccessedAt || 0) > cutoff) continue;
      if (this.dispose(record.externalId)) suspended += 1;
    }
    return suspended;
  }

  enforceHotLimit({ excludeSessionId = "", incoming = 0 } = {}) {
    let suspended = 0;
    for (const record of this._candidates(excludeSessionId)) {
      if (this.sessions.size + incoming <= this.config.maxOpenSessions) break;
      if (this.dispose(record.externalId)) suspended += 1;
    }
    return suspended;
  }

  relieveMemoryPressure({ excludeSessionId = "" } = {}) {
    let suspended = 0;
    if (this.rssBytes() < this.config.softRssBytes) return suspended;
    for (const record of this._candidates(excludeSessionId)) {
      if (this.dispose(record.externalId)) suspended += 1;
      if (this.rssBytes() < this.config.softRssBytes) break;
    }
    return suspended;
  }

  admit({ excludeSessionId = "", incoming = 0 } = {}) {
    this.suspendIdle({ excludeSessionId });
    this.enforceHotLimit({ excludeSessionId, incoming });
    this.relieveMemoryPressure({ excludeSessionId });
    const status = this.status();
    if (this.sessions.size + incoming > this.config.maxOpenSessions) {
      throw sessionCapacityError(status);
    }
    if (status.rss_bytes >= this.config.emergencyRssBytes) {
      throw memoryPressureError(status);
    }
    return status;
  }

  status() {
    const rss = Math.max(0, Number(this.rssBytes()) || 0);
    return {
      hot_sessions: this.sessions.size,
      active_sessions: this.activeRequests.size,
      suspended_sessions: this.suspendedSessions,
      max_open_sessions: this.config.maxOpenSessions,
      idle_seconds: Math.floor(this.config.idleMs / 1000),
      rss_bytes: rss,
      rss_mb: Math.round((rss / MIB) * 10) / 10,
      soft_rss_mb: Math.round((this.config.softRssBytes / MIB) * 10) / 10,
      emergency_rss_mb: Math.round((this.config.emergencyRssBytes / MIB) * 10) / 10,
      pressure: rss >= this.config.emergencyRssBytes
        ? "emergency"
        : (rss >= this.config.softRssBytes ? "soft" : "normal"),
    };
  }

  shutdown() {
    for (const sessionId of [...this.sessions.keys()]) {
      this.dispose(sessionId, { force: true, suspension: false });
    }
  }
}
