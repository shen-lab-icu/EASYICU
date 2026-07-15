"""Evidence store with content hashing and provenance index.

Every analytical artefact — table, figure, statistic, log, code —
is registered here with a SHA-256 hash of its bytes and a record of
which step produced it. The manuscript writer is only allowed to
cite ``evidence_id`` strings that resolve to a record in this store;
unbound citations are stripped during post-processing.

Why this exists:

* It is the single mechanism that makes the agent's claims
  *auditable*. A reviewer can recompute the hash of a figure and
  confirm it matches the manifest.
* It is what lets the manuscript scaffold be "honestly thin": the
  sentences are templates with placeholders, and the placeholders
  are guaranteed to point at real artefacts.

Alias resolution
----------------
Registration assigns a hash-suffixed evidence_id (e.g.
``table_table_one_8f3c19a4``) so that two artefacts with the same
filename stay distinct. The writer agent, however, prefers stable
semantic names like ``table_one`` or ``outcome_rate``. The store
therefore maintains an *alias* table — first-write-wins — so that
``{evidence:table_one}`` placeholders in the manuscript resolve to
the first registered evidence with that filename stem (or to an
explicit alias passed by the pipeline). Aliases are persisted
alongside the index so binding is reproducible.

The on-disk layout under ``<workdir>/evidence/`` is::

    evidence/
        <evidence_id>__<basename>.<ext>     # the artefact
        evidence_index.json                 # serialised list of EvidenceRecord
        evidence_aliases.json               # alias → evidence_id map
"""

from __future__ import annotations

import enum
import hashlib
import json
import logging
import os
import re
import secrets
import shutil
import stat
import tempfile
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .schema import EvidenceRecord


def _validated_directory_fd(path: Path, *, expected_root: Path) -> int:
    """Open ``path`` without following symlinks and anchor it under root."""

    if path.is_symlink():
        raise ValueError("evidence directory must not be a symbolic link")
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(expected_root.resolve(strict=True))
    except (FileNotFoundError, ValueError) as exc:
        raise ValueError("evidence directory escapes the store root") from exc
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        path_stat = os.stat(path, follow_symlinks=False)
        fd_stat = os.fstat(fd)
        if (
            not stat.S_ISDIR(path_stat.st_mode)
            or path_stat.st_dev != fd_stat.st_dev
            or path_stat.st_ino != fd_stat.st_ino
        ):
            raise ValueError("evidence directory changed during validation")
    except BaseException:
        os.close(fd)
        raise
    return fd


def _atomic_write_bytes(
    path: Path,
    payload: bytes,
    *,
    expected_root: Optional[Path] = None,
) -> None:
    """Write ``payload`` to ``path`` atomically (temp file + fsync + os.replace).

    The evidence index and content-addressed blobs are the manuscript's
    provenance backbone; a crash during a bare write would truncate them and
    fail the SHA-256 verification / leave the index unreadable. os.replace is
    atomic on POSIX, so a reader sees either the old or the new complete file.
    """
    if expected_root is not None and os.name == "posix":
        parent_fd = _validated_directory_fd(path.parent, expected_root=expected_root)
        target_name = _path_component(path.name, label="evidence filename")
        tmp_name = f".{target_name}.{secrets.token_hex(8)}.tmp"
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        fd: Optional[int] = None
        try:
            fd = os.open(tmp_name, flags, 0o600, dir_fd=parent_fd)
            with os.fdopen(fd, "wb") as handle:
                fd = None
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(
                tmp_name,
                target_name,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
            )
            os.fsync(parent_fd)
            current = os.stat(path.parent, follow_symlinks=False)
            opened = os.fstat(parent_fd)
            if current.st_dev != opened.st_dev or current.st_ino != opened.st_ino:
                raise ValueError("evidence directory changed during atomic write")
        except BaseException:
            if fd is not None:
                os.close(fd)
            try:
                os.unlink(tmp_name, dir_fd=parent_fd)
            except OSError:
                pass
            raise
        finally:
            os.close(parent_fd)
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _atomic_write_text(
    path: Path,
    text: str,
    *,
    encoding: str = "utf-8",
    expected_root: Optional[Path] = None,
) -> None:
    """Atomic text write; see :func:`_atomic_write_bytes`."""
    _atomic_write_bytes(path, text.encode(encoding), expected_root=expected_root)


def _atomic_copy_file(source: Path, target: Path, *, expected_root: Path) -> None:
    """Stream-copy into an anchored directory and atomically publish it."""

    if os.name != "posix":
        _atomic_write_bytes(target, source.read_bytes(), expected_root=expected_root)
        return
    parent_fd = _validated_directory_fd(target.parent, expected_root=expected_root)
    target_name = _path_component(target.name, label="evidence filename")
    tmp_name = f".{target_name}.{secrets.token_hex(8)}.tmp"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    fd: Optional[int] = None
    try:
        fd = os.open(tmp_name, flags, 0o600, dir_fd=parent_fd)
        with source.open("rb") as source_handle, os.fdopen(fd, "wb") as target_handle:
            fd = None
            shutil.copyfileobj(source_handle, target_handle, length=1024 * 1024)
            target_handle.flush()
            os.fsync(target_handle.fileno())
        os.replace(
            tmp_name,
            target_name,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
        )
        os.fsync(parent_fd)
        current = os.stat(target.parent, follow_symlinks=False)
        opened = os.fstat(parent_fd)
        if current.st_dev != opened.st_dev or current.st_ino != opened.st_ino:
            raise ValueError("evidence directory changed during atomic copy")
    except BaseException:
        if fd is not None:
            os.close(fd)
        try:
            os.unlink(tmp_name, dir_fd=parent_fd)
        except OSError:
            pass
        raise
    finally:
        os.close(parent_fd)


def _path_component(value: str, *, label: str) -> str:
    """Return one safe filename component or fail closed.

    Evidence ids and caller-supplied filenames are combined into an on-disk
    filename.  They must therefore never carry an absolute path, a separator,
    or a dot-directory component.  Keeping this check here protects every
    EvidenceStore caller, including MCP and future integrations.
    """

    text = str(value or "")
    if (
        not text
        or text in {".", ".."}
        or "\x00" in text
        or "/" in text
        or "\\" in text
        or Path(text).is_absolute()
        or Path(text).name != text
    ):
        raise ValueError(f"{label} must be a single safe path component")
    return text


logger = logging.getLogger(__name__)


class EvidenceEnforcementMode(str, enum.Enum):
    """How strictly the EvidenceStore polices manuscript-bound output.

    ``SOFT`` (default) — the long-standing behaviour: unsupported
    result-like sentences are silently filtered, unresolved
    ``{evidence:<id>}`` placeholders are rendered as
    ``[evidence missing: <id>]`` and reported as warnings. Suitable for
    interactive runs where the writer is being iterated on.

    ``STRICT`` — every guard raises :class:`EvidenceEnforcementError`
    instead of repairing the manuscript in place. Use this for CI
    gates and final submission packaging where a silent demotion would
    let an unverified claim slip into the bound manuscript.
    """

    SOFT = "soft"
    STRICT = "strict"


class EvidenceEnforcementError(RuntimeError):
    """Raised by an :class:`EvidenceStore` in ``STRICT`` mode when the
    manuscript would otherwise need to be silently repaired.

    The ``detail`` mapping carries the offending items (removed
    sentences, missing evidence ids, ...) so callers can include them
    in audit logs without re-parsing the message string.
    """

    def __init__(
        self, message: str, *, detail: Optional[Dict[str, Any]] = None
    ) -> None:
        super().__init__(message)
        self.detail: Dict[str, Any] = dict(detail or {})


def _coerce_enforcement_mode(
    value: Optional[str | EvidenceEnforcementMode],
) -> EvidenceEnforcementMode:
    if value is None:
        return EvidenceEnforcementMode.SOFT
    if isinstance(value, EvidenceEnforcementMode):
        return value
    try:
        return EvidenceEnforcementMode(str(value).lower())
    except ValueError as exc:
        raise ValueError(
            f"Unknown evidence enforcement mode: {value!r}; "
            f"expected one of {[m.value for m in EvidenceEnforcementMode]}"
        ) from exc


def _quarantine_corrupt_index(path: Path, exc: Exception, kind: str) -> None:
    """Rename a corrupted index file aside and log a warning.

    Silently returning empty would erase the audit trail the writer agent
    relies on; instead we keep the broken bytes for forensic inspection
    and emit a warning so the operator knows evidence was lost.
    """
    if not path.exists():
        logger.warning("evidence %s missing; starting fresh: %s", kind, exc)
        return
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = path.with_suffix(path.suffix + f".broken-{timestamp}")
    try:
        path.replace(backup)
    except OSError as rename_err:
        logger.warning(
            "evidence %s at %s is corrupt (%s); also failed to back up: %s",
            kind,
            path,
            exc,
            rename_err,
        )
        return
    logger.warning(
        "evidence %s at %s is corrupt (%s); moved to %s and starting fresh",
        kind,
        path,
        exc,
        backup,
    )


# ---------------------------------------------------------------------------
# Numeric claim registry (value-level provenance, A-track)
# ---------------------------------------------------------------------------
#
# Sentence-level evidence binding (``{evidence:<id>}`` placeholders) tells
# a reviewer which artefact a sentence is grounded in, but a sentence
# typically embeds multiple numeric values — odds ratios, p-values,
# AUC, cohort counts — each of which the reviewer may want to verify
# *individually*. The :class:`NumericClaim` registry closes that gap:
# every numeric leaf the runner emits in ``step_summary.json`` is
# captured with (value, source step, source field, owning evidence id),
# so the manuscript post-processor can scan rendered prose and bind
# each number to the exact field of the exact step output that
# produced it.
#
# Inspired by ``data-to-paper`` (NEJM AI 2024) which uses
# ``\hypertarget`` / ``\hyperlink`` to make every number in the
# manuscript click-traceable to its producing code line.


_NUMERIC_LEAF_RE = re.compile(r"^[-+]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][-+]?\d+)?$")
# Numbers embedded in manuscript prose. The manuscript layer only
# binds numbers that *look like* result quantities: decimal-bearing
# (1.42 / 0.003 / 12.5%), comma-grouped thousands (1,234), exponent
# form (1.2e-5), or bare integers with ≥3 digits (999, 1234). Two-digit
# integers are deliberately rejected because they collide with hyphenated
# identifiers (SOFA-2), CI labels (95% CI), and chapter-section refs
# (Section 4). When a manuscript truly needs to cite a two-digit count
# it should embed it inside an explicit evidence placeholder.
_NUMERIC_IN_PROSE_RE = re.compile(
    r"(?<![A-Za-z_\d.])"  # avoid mid-identifier digits
    r"(?P<value>"
    r"(?:"  # --- general numeric form ---
    r"[-+]?"
    r"(?:"
    r"\d{1,3}(?:,\d{3})+(?:\.\d+)?"  # comma-grouped (with optional fraction)
    r"|"
    r"\d+\.\d+(?:[eE][-+]?\d+)?"  # decimal (with optional exponent)
    r"|"
    r"\d+[eE][-+]?\d+"  # bare exponent
    r"|"
    r"\d{3,}"  # ≥3-digit integer
    r")"
    r"%?"  # optional percent suffix
    r")"
    r"|"  # --- short percent form ---
    # A bare 1-2 digit integer is normally rejected (it collides with
    # SOFA-2, "Section 4", "n=42", ...), but a trailing percent sign
    # disambiguates it as a data value (mortality 23%, 8% decline), so it
    # is bound. The lookahead still excludes confidence / credible-interval
    # *levels* ("95% CI", "90% confidence", "99% credible interval"), which
    # are labels, not claims.
    r"\d{1,2}%(?!\s*(?:CI\b|confidence|credible))"
    r")"
    r"(?![A-Za-z_\d]|\.\d)"  # not followed by identifier / decimal continuation
)


@dataclass
class NumericClaim:
    """One numeric leaf the manuscript may cite, tied back to its source.

    Fields:

    * ``value`` — literal string form (preserves precision/formatting)
    * ``canonical`` — float for tolerance-based matching
    * ``evidence_id`` — owning evidence record (e.g. the
      ``step_summary.json`` for the step)
    * ``step_id`` — step that emitted this value
    * ``source_field`` — dotted path inside step_summary
      (e.g. ``primary_or`` or ``stratified.male.auc``)
    * ``tolerance`` — relative tolerance for fuzzy matching when the
      manuscript prints a rounded version of the canonical value

    Phase-1 derived-claim fields (Commit 2, May 2026). All optional;
    when ``formula is None`` the claim is a regular step_summary leaf
    and behaviour is byte-identical to pre-derived versions.

    * ``formula`` — the source expression as a string (e.g.
      ``exp(log(primary_or) - 1.96 * primary_or_se)``). When set, the
      claim was computed at register-time from one or more source
      claims via the restricted-AST evaluator
      (``_evaluate_derived_formula``).
    * ``explanation`` — short human-readable rationale for the
      formula (e.g. ``"low 95% CI for primary OR, log-normal
      approximation"``). Surfaces in audit reports and the writer
      digest's derived block.
    * ``derived_from`` — list of ``(source_step_id, source_field)``
      pairs identifying which source claims the formula references.
      Captured at register-time so audit can replay derivation
      without re-parsing the formula string.

    Inspired by data-to-paper's ``\\num{<formula>, "<explanation>"}``
    macro (NEJM AI 2024); evaluated at register-time rather than
    compile-time so the result is persisted as a regular claim and
    can be matched by the existing reverse-binder.
    """

    value: str
    canonical: float
    evidence_id: str
    step_id: str
    source_field: str
    tolerance: float = 1e-3
    # Derived-claim metadata. Default `None` / empty list so existing
    # step_summary leaves serialise to the same JSON as before.
    formula: Optional[str] = None
    explanation: Optional[str] = None
    derived_from: List[Tuple[str, str]] = field(default_factory=list)

    @property
    def is_derived(self) -> bool:
        return self.formula is not None

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        if not self.is_derived:
            # Preserve the pre-derived JSON shape for ordinary numeric
            # leaves so old audit tooling does not suddenly see empty
            # formula/provenance fields on every claim.
            payload.pop("formula", None)
            payload.pop("explanation", None)
            payload.pop("derived_from", None)
        return payload

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NumericClaim":
        known = {f for f in cls.__dataclass_fields__}
        clean = {k: v for k, v in data.items() if k in known}
        # ``derived_from`` may come back from JSON as list-of-lists;
        # coerce inner pairs to tuples for hashability + .__eq__ parity
        # with the dataclass-default empty list.
        if "derived_from" in clean and clean["derived_from"]:
            clean["derived_from"] = [
                tuple(pair) if not isinstance(pair, tuple) else pair
                for pair in clean["derived_from"]
            ]
        return cls(**clean)


# ---------------------------------------------------------------------------
# Derived-claim formula evaluator (restricted AST sandbox)
# ---------------------------------------------------------------------------
#
# Coder-generated step_summary entries can request that a derived
# numeric value (an OR confidence-interval bound, a between-cohort
# difference, an AUC delta) be computed from other registered claims
# and registered as its own NumericClaim. The formula is evaluated
# here in a restricted-AST sandbox that accepts ONLY:
#
#   * Numeric constants
#   * Names resolving to entries in ``sources`` (or to a math whitelist
#     constant — pi, e)
#   * Arithmetic: + - * / **
#   * Unary +/-
#   * Calls to a small math whitelist: exp, log, log10, sqrt, abs,
#     min, max
#   * Parentheses
#
# Rejected: attribute access, subscripts, comprehensions, comparisons,
# bool ops, lambda, named expressions, anything else. Names starting
# with ``_`` are rejected outright. The output is a finite float; nan
# / inf raise. Errors raise ``DerivedFormulaError`` with the offending
# AST node type for the audit trail.
#
# Inspired by data-to-paper's ``\num{<formula>, "<explanation>"}``;
# our variant evaluates at register-time and persists the result as a
# regular NumericClaim with ``derived_from`` provenance — so the
# existing reverse-binder picks it up with no extra plumbing.


class DerivedFormulaError(ValueError):
    """Raised when a derived-claim formula contains a disallowed
    expression, references an unknown source, or evaluates to a
    non-finite value. The message is safe to surface in audit findings.
    """


# Math whitelist. ``min`` / ``max`` accept any positive number of
# args; the rest are single-arg. Anything outside this dict is rejected
# at call-time.
import math as _math  # noqa: E402  (deliberate post-typing local-only import)

_DERIVED_FORMULA_FUNCS: Dict[str, Any] = {
    "exp": _math.exp,
    "log": _math.log,
    "log10": _math.log10,
    "sqrt": _math.sqrt,
    "abs": abs,
    "min": min,
    "max": max,
}
_DERIVED_FORMULA_CONSTS: Dict[str, float] = {
    "pi": _math.pi,
    "e": _math.e,
}


def _evaluate_derived_formula(
    formula: str,
    *,
    sources: Dict[str, float],
) -> float:
    """Evaluate ``formula`` in the restricted sandbox.

    ``sources`` maps Python identifier → canonical float of a
    registered claim. Names referenced in the formula that aren't in
    ``sources`` and aren't in the math constants whitelist raise
    ``DerivedFormulaError``. Returns a finite float.
    """
    import ast  # local import — only needed when a derived claim arrives

    try:
        tree = ast.parse(formula, mode="eval")
    except SyntaxError as exc:
        raise DerivedFormulaError(f"derived-formula syntax error: {exc.msg}") from exc

    _ALLOWED_BINOPS = (
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Pow,
    )
    _ALLOWED_UNARYOPS = (ast.UAdd, ast.USub)

    def _walk(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return _walk(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)) and not isinstance(
                node.value, bool
            ):
                return float(node.value)
            raise DerivedFormulaError(
                f"derived-formula rejected constant of type {type(node.value).__name__}"
            )
        if isinstance(node, ast.Name):
            name = node.id
            if name.startswith("_"):
                raise DerivedFormulaError(
                    f"derived-formula rejected dunder/private name {name!r}"
                )
            if name in sources:
                return float(sources[name])
            if name in _DERIVED_FORMULA_CONSTS:
                return _DERIVED_FORMULA_CONSTS[name]
            raise DerivedFormulaError(
                f"derived-formula references unknown source {name!r}; "
                f"available sources: {sorted(sources)}"
            )
        if isinstance(node, ast.BinOp):
            if not isinstance(node.op, _ALLOWED_BINOPS):
                raise DerivedFormulaError(
                    f"derived-formula rejected binary operator "
                    f"{type(node.op).__name__}"
                )
            lhs = _walk(node.left)
            rhs = _walk(node.right)
            try:
                if isinstance(node.op, ast.Add):
                    return lhs + rhs
                if isinstance(node.op, ast.Sub):
                    return lhs - rhs
                if isinstance(node.op, ast.Mult):
                    return lhs * rhs
                if isinstance(node.op, ast.Div):
                    if rhs == 0.0:
                        raise DerivedFormulaError("derived-formula division by zero")
                    return lhs / rhs
                if isinstance(node.op, ast.Pow):
                    return lhs**rhs
            except OverflowError as exc:
                # Python raises OverflowError on float ** float when the
                # result would exceed float range; we surface this as a
                # non-finite result so callers see one consistent error
                # type regardless of whether overflow happens silently
                # (→ inf, caught below) or eagerly (→ OverflowError).
                raise DerivedFormulaError(
                    f"derived-formula evaluated to non-finite value "
                    f"(arithmetic overflow): {exc}"
                ) from exc
        if isinstance(node, ast.UnaryOp):
            if not isinstance(node.op, _ALLOWED_UNARYOPS):
                raise DerivedFormulaError(
                    f"derived-formula rejected unary operator "
                    f"{type(node.op).__name__}"
                )
            operand = _walk(node.operand)
            return +operand if isinstance(node.op, ast.UAdd) else -operand
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise DerivedFormulaError(
                    "derived-formula rejected call with non-name callable"
                )
            fname = node.func.id
            if fname.startswith("_") or fname not in _DERIVED_FORMULA_FUNCS:
                raise DerivedFormulaError(
                    f"derived-formula rejected call to {fname!r}; "
                    f"allowed: {sorted(_DERIVED_FORMULA_FUNCS)}"
                )
            if node.keywords:
                raise DerivedFormulaError(
                    f"derived-formula rejected keyword arguments to {fname!r}"
                )
            args = [_walk(a) for a in node.args]
            try:
                return float(_DERIVED_FORMULA_FUNCS[fname](*args))
            except (ArithmeticError, ValueError, TypeError) as exc:
                raise DerivedFormulaError(
                    f"derived-formula call {fname!r} failed: {exc}"
                ) from exc
        raise DerivedFormulaError(
            f"derived-formula rejected AST node {type(node).__name__}"
        )

    result = _walk(tree)
    if not isinstance(result, (int, float)):
        raise DerivedFormulaError(
            f"derived-formula evaluated to non-numeric {type(result).__name__}"
        )
    fresult = float(result)
    if fresult != fresult or fresult in (float("inf"), float("-inf")):
        raise DerivedFormulaError(
            f"derived-formula evaluated to non-finite value {fresult!r}"
        )
    return fresult


def _coerce_numeric_literal(value: Any) -> Optional[Tuple[str, float]]:
    """Return ``(literal_str, canonical_float)`` for a numeric leaf.

    Booleans are rejected (they are not numeric *values* for our
    purposes even though ``isinstance(True, int)`` is ``True``).
    Non-finite floats (``inf`` / ``nan``) are rejected because the
    manuscript binder cannot resolve them.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        f = float(value)
        if not (f == f) or f in (float("inf"), float("-inf")):
            return None
        if isinstance(value, int):
            return (str(value), f)
        return (f"{value:.6g}", f)
    if isinstance(value, str):
        stripped = value.strip().rstrip("%").replace(",", "")
        if _NUMERIC_LEAF_RE.match(stripped):
            try:
                return (value.strip(), float(stripped))
            except ValueError:
                return None
    return None


def _decimal_places(value_str: str) -> int:
    """Return the number of decimal places in a numeric display string.

    The manuscript binder uses this to allow rounded prose values such
    as ``1.22`` to match a more precise canonical claim like
    ``1.224779``. Scientific notation is treated as zero-decimal for
    the purposes of display rounding because the plain-text manuscript
    binder only sees the rendered literal.
    """
    text = (value_str or "").strip()
    if not text or "e" in text.lower() or "." not in text:
        return 0
    frac = text.split(".", 1)[1]
    frac = re.sub(r"[^0-9].*$", "", frac)
    return len(frac)


def _walk_numeric_leaves(obj: Any, prefix: str = "") -> List[Tuple[str, str, float]]:
    """Yield ``(dotted_path, literal_str, canonical_float)`` for every
    numeric leaf in a nested dict/list. Strings that happen to parse as
    numbers are also captured because runners frequently emit
    ``"0.18"`` rather than ``0.18``."""
    out: List[Tuple[str, str, float]] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            out.extend(_walk_numeric_leaves(v, key))
    elif isinstance(obj, (list, tuple)):
        for idx, v in enumerate(obj):
            key = f"{prefix}[{idx}]"
            out.extend(_walk_numeric_leaves(v, key))
    else:
        coerced = _coerce_numeric_literal(obj)
        if coerced is not None:
            literal, canonical = coerced
            out.append((prefix or "<root>", literal, canonical))
    return out


def sha256_of_file(path: Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            buf = f.read(chunk)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def sha256_of_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class EvidenceStore:
    """A directory-backed store of hashed artefacts.

    Each call to :meth:`register_file` copies the file into
    ``evidence/`` under a deterministic name (``<id>__<basename>``)
    and writes an :class:`EvidenceRecord` to the in-memory index. The
    store is persisted to ``evidence_index.json`` on every call so
    crashes don't lose evidence.
    """

    def __init__(
        self,
        root: Path,
        *,
        enforcement_mode: Optional[str | EvidenceEnforcementMode] = None,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.dir = self.root / "evidence"
        self.dir.mkdir(parents=True, exist_ok=True)
        self._validated_evidence_dir()
        self.index_path = self.dir / "evidence_index.json"
        self.aliases_path = self.dir / "evidence_aliases.json"
        self.numeric_claims_path = self.dir / "numeric_claims.json"
        self.enforcement_mode: EvidenceEnforcementMode = _coerce_enforcement_mode(
            enforcement_mode
        )
        self._records: List[EvidenceRecord] = self._load_records()
        self._aliases: Dict[str, str] = self._load_aliases()
        self._numeric_claims: List[NumericClaim] = self._load_numeric_claims()
        # T3.3 — concurrent step execution: every register / get / save
        # path runs under this lock so two worker threads can safely
        # call ``register_file`` simultaneously. Reentrant so that
        # methods that internally call ``register_file`` (e.g.
        # ``register_text`` → ``register_file``) don't self-deadlock.
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _validated_evidence_dir(self) -> Path:
        """Re-prove the mutable on-disk directory before every write."""

        if self.dir.is_symlink():
            raise ValueError("evidence directory must not be a symbolic link")
        try:
            resolved = self.dir.resolve(strict=True)
            resolved.relative_to(self.root)
        except (FileNotFoundError, ValueError) as exc:
            raise ValueError("evidence directory escapes the store root") from exc
        if resolved != self.dir:
            raise ValueError("evidence directory must use its canonical path")
        if not self.dir.is_dir():
            raise ValueError("evidence path must remain a directory")
        return self.dir

    def _load_records(self) -> List[EvidenceRecord]:
        if not self.index_path.exists():
            return []
        try:
            data = json.loads(self.index_path.read_text(encoding="utf-8"))
            return [EvidenceRecord.model_validate(r) for r in data]
        except Exception as exc:
            _quarantine_corrupt_index(self.index_path, exc, kind="index")
            return []

    def _load_aliases(self) -> Dict[str, str]:
        if not self.aliases_path.exists():
            return {}
        try:
            data = json.loads(self.aliases_path.read_text(encoding="utf-8"))
        except Exception as exc:
            _quarantine_corrupt_index(self.aliases_path, exc, kind="aliases")
            return {}
        if isinstance(data, dict):
            return {str(k): str(v) for k, v in data.items()}
        logger.warning(
            "evidence aliases at %s is not a JSON object (got %s); ignoring",
            self.aliases_path,
            type(data).__name__,
        )
        _quarantine_corrupt_index(
            self.aliases_path,
            ValueError(f"unexpected type {type(data).__name__}"),
            kind="aliases",
        )
        return {}

    def _load_numeric_claims(self) -> List[NumericClaim]:
        if not self.numeric_claims_path.exists():
            return []
        try:
            data = json.loads(self.numeric_claims_path.read_text(encoding="utf-8"))
            return [NumericClaim.from_dict(c) for c in data]
        except Exception as exc:
            _quarantine_corrupt_index(
                self.numeric_claims_path, exc, kind="numeric_claims"
            )
            return []

    def _save(self) -> None:
        self._validated_evidence_dir()
        _atomic_write_text(
            self.index_path,
            json.dumps(
                [r.model_dump(mode="json") for r in self._records],
                indent=2,
                ensure_ascii=False,
                default=str,
            ),
            expected_root=self.root,
        )
        _atomic_write_text(
            self.aliases_path,
            json.dumps(self._aliases, indent=2, ensure_ascii=False, sort_keys=True),
            expected_root=self.root,
        )
        _atomic_write_text(
            self.numeric_claims_path,
            json.dumps(
                [c.to_dict() for c in self._numeric_claims],
                indent=2,
                ensure_ascii=False,
            ),
            expected_root=self.root,
        )

    def _add_alias(self, alias: str, evidence_id: str) -> None:
        """First-write-wins. We do not overwrite an existing alias because
        the manuscript scaffold is generated against the first artefact of
        each kind; later registrations with the same filename should not
        retroactively change what an earlier sentence pointed to."""
        if not alias:
            return
        if alias in self._aliases:
            return
        self._aliases[alias] = evidence_id

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def _record_by_id(self, evidence_id: str) -> Optional[EvidenceRecord]:
        for record in self._records:
            if record.evidence_id == evidence_id:
                return record
        return None

    def _make_id(self, prefix: str, digest: str) -> str:
        return _path_component(f"{prefix}_{digest[:8]}", label="evidence_id")

    def _target_path(self, *, evidence_id: str, filename: str) -> Path:
        """Build an evidence target and prove it remains under ``evidence/``."""

        safe_id = _path_component(evidence_id, label="evidence_id")
        safe_filename = _path_component(filename, label="filename")
        base = self._validated_evidence_dir()
        target = base / f"{safe_id}__{safe_filename}"
        if target.is_symlink():
            raise ValueError("evidence target must not be a symbolic link")
        resolved = target.resolve(strict=False)
        try:
            resolved.relative_to(base)
        except ValueError as exc:
            raise ValueError("evidence target escapes the evidence directory") from exc
        return target

    def _next_versioned_id(self, evidence_id: str) -> str:
        evidence_id = _path_component(evidence_id, label="evidence_id")
        suffix_n = 2
        while self._record_by_id(f"{evidence_id}_v{suffix_n}") is not None:
            suffix_n += 1
        return f"{evidence_id}_v{suffix_n}"

    def _register_target(
        self,
        *,
        evidence_id: str,
        kind: str,
        description: str,
        target: Path,
        sha256: str,
        produced_by_step: Optional[str],
        inputs: Optional[Sequence[str]],
        script_evidence_id: Optional[str],
        aliases: Optional[Sequence[str]],
        producer: Optional[str],
        generation_mode: Optional[str],
        prompt_pack_version: Optional[str],
        metadata: Optional[Dict[str, Any]],
        publish_aliases: bool,
        on_sha_change: str = "raise",
    ) -> EvidenceRecord:
        """Register an evidence record on disk.

        ``on_sha_change`` controls what happens when the supplied
        ``evidence_id`` already exists with a *different* sha256:

        * ``"raise"`` (default) — keep the existing strict behavior;
          collisions raise ``ValueError``. This protects against
          accidental overwrites in normal pipeline operation.
        * ``"new_id"`` — register the new content under a derived id
          (``{evidence_id}_v{n}`` where ``n`` is the smallest integer
          that yields a free id, starting at 2). The original record
          keeps its id and its alias bindings; the new record is added
          to the store so its content remains auditable. Used by the
          resume path for transient envelopes / cost summaries that
          legitimately differ between the original run and the
          resumption (different per-call timestamps, model versions,
          etc.).
        * ``"keep_existing"`` — the new file is dropped (left on disk
          as-is) and the existing record is returned unchanged. Use
          this when the first-registered content is authoritative.
        """
        evidence_id = _path_component(evidence_id, label="evidence_id")
        base = self._validated_evidence_dir()
        try:
            target.resolve(strict=False).relative_to(base)
        except ValueError as exc:
            raise ValueError("evidence target escapes the evidence directory") from exc
        if on_sha_change not in {"raise", "new_id", "keep_existing"}:
            raise ValueError(
                f"Unknown on_sha_change mode: {on_sha_change!r}. "
                "Expected one of: raise, new_id, keep_existing."
            )
        existing = self._record_by_id(evidence_id)
        if existing is not None:
            if existing.sha256 != sha256:
                if on_sha_change == "keep_existing":
                    if publish_aliases:
                        for alias in aliases or []:
                            self._add_alias(alias, evidence_id)
                        self._add_alias(
                            _target_basename_stem(target, evidence_id), evidence_id
                        )
                        self._add_alias(evidence_id, evidence_id)
                        existing.metadata = {
                            **dict(existing.metadata or {}),
                            "aliases_published": True,
                        }
                    self._save()
                    return existing
                if on_sha_change == "new_id":
                    # Find the next free suffix so multiple resumes
                    # accumulate without ever colliding.
                    suffix_n = 2
                    while self._record_by_id(f"{evidence_id}_v{suffix_n}") is not None:
                        suffix_n += 1
                    new_id = f"{evidence_id}_v{suffix_n}"
                    record = EvidenceRecord(
                        evidence_id=new_id,
                        kind=kind,  # type: ignore[arg-type]
                        description=description,
                        relative_path=str(target.relative_to(self.root)),
                        sha256=sha256,
                        produced_by_step=produced_by_step,
                        inputs=list(inputs or []),
                        script_evidence_id=script_evidence_id,
                        producer=producer,
                        generation_mode=generation_mode,
                        prompt_pack_version=prompt_pack_version,
                        metadata={
                            **dict(metadata or {}),
                            "resume_supersedes": evidence_id,
                        },
                        created_at=datetime.now(timezone.utc),
                    )
                    self._records.append(record)
                    # Bind the basename alias to the NEW id so the
                    # second-write file is still discoverable on disk;
                    # the original evidence_id alias keeps pointing at
                    # the original record (it is the canonical citation
                    # target for the run).
                    if publish_aliases:
                        for alias in aliases or []:
                            self._add_alias(alias, new_id)
                        self._add_alias(_target_basename_stem(target, new_id), new_id)
                        self._add_alias(new_id, new_id)
                    self._save()
                    return record
                raise ValueError(
                    f"Evidence id collision for {evidence_id}: "
                    f"existing sha256={existing.sha256[:8]} new sha256={sha256[:8]}"
                )
            if publish_aliases:
                for alias in aliases or []:
                    self._add_alias(alias, evidence_id)
                self._add_alias(_target_basename_stem(target, evidence_id), evidence_id)
                self._add_alias(evidence_id, evidence_id)
                existing.metadata = {
                    **dict(existing.metadata or {}),
                    "aliases_published": True,
                }
            self._save()
            return existing

        record = EvidenceRecord(
            evidence_id=evidence_id,
            kind=kind,  # type: ignore[arg-type]
            description=description,
            relative_path=str(target.relative_to(self.root)),
            sha256=sha256,
            produced_by_step=produced_by_step,
            inputs=list(inputs or []),
            script_evidence_id=script_evidence_id,
            producer=producer,
            generation_mode=generation_mode,
            prompt_pack_version=prompt_pack_version,
            metadata=dict(metadata or {}),
            created_at=datetime.now(timezone.utc),
        )
        self._records.append(record)

        if publish_aliases:
            for alias in aliases or []:
                self._add_alias(alias, evidence_id)
            self._add_alias(_target_basename_stem(target, evidence_id), evidence_id)
            self._add_alias(evidence_id, evidence_id)

        self._save()
        return record

    def register_file(
        self,
        *,
        kind: str,
        description: str,
        source_path: Path,
        produced_by_step: Optional[str] = None,
        inputs: Optional[Sequence[str]] = None,
        script_evidence_id: Optional[str] = None,
        evidence_id: Optional[str] = None,
        aliases: Optional[Sequence[str]] = None,
        producer: Optional[str] = None,
        generation_mode: Optional[str] = None,
        prompt_pack_version: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        publish_aliases: bool = True,
        on_sha_change: str = "raise",
    ) -> EvidenceRecord:
        if not source_path.exists():
            raise FileNotFoundError(f"Cannot register missing file: {source_path}")
        # T3.3 — guard the entire critical section (id allocation, file
        # copy into evidence/, record append, alias update, persist).
        # The lock is reentrant so register_text → register_file
        # composition stays deadlock-free.
        with self._lock:
            source_digest = sha256_of_file(source_path)
            eid = evidence_id or self._make_id(
                _id_prefix(kind, source_path.stem), source_digest
            )
            target_eid = eid
            target_metadata = dict(metadata or {})
            target_metadata["aliases_published"] = bool(publish_aliases)
            target_on_sha_change = on_sha_change
            existing = self._record_by_id(eid)
            if (
                existing is not None
                and existing.sha256 != source_digest
                and on_sha_change == "new_id"
            ):
                target_eid = self._next_versioned_id(eid)
                target_metadata.setdefault("resume_supersedes", eid)
                target_on_sha_change = "raise"
            target = self._target_path(
                evidence_id=target_eid,
                filename=source_path.name,
            )
            if target.resolve() != source_path.resolve():
                _atomic_copy_file(source_path, target, expected_root=self.root)
            digest = sha256_of_file(target)
            return self._register_target(
                evidence_id=target_eid,
                kind=kind,
                description=description,
                target=target,
                sha256=digest,
                produced_by_step=produced_by_step,
                inputs=inputs,
                script_evidence_id=script_evidence_id,
                aliases=aliases,
                producer=producer,
                generation_mode=generation_mode,
                prompt_pack_version=prompt_pack_version,
                metadata=target_metadata,
                publish_aliases=publish_aliases,
                on_sha_change=target_on_sha_change,
            )

    def register_text(
        self,
        *,
        kind: str,
        description: str,
        text: str,
        filename: str,
        produced_by_step: Optional[str] = None,
        inputs: Optional[Sequence[str]] = None,
        script_evidence_id: Optional[str] = None,
        evidence_id: Optional[str] = None,
        aliases: Optional[Sequence[str]] = None,
        producer: Optional[str] = None,
        generation_mode: Optional[str] = None,
        prompt_pack_version: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        publish_aliases: bool = True,
        on_sha_change: str = "raise",
    ) -> EvidenceRecord:
        payload = text.encode("utf-8")
        digest = sha256_of_bytes(payload)
        eid = evidence_id or self._make_id(
            _id_prefix(kind, Path(filename).stem), digest
        )
        with self._lock:
            target_eid = eid
            target_metadata = dict(metadata or {})
            target_metadata["aliases_published"] = bool(publish_aliases)
            target_on_sha_change = on_sha_change
            existing = self._record_by_id(eid)
            if (
                existing is not None
                and existing.sha256 != digest
                and on_sha_change == "new_id"
            ):
                target_eid = self._next_versioned_id(eid)
                target_metadata.setdefault("resume_supersedes", eid)
                target_on_sha_change = "raise"
            target = self._target_path(
                evidence_id=target_eid,
                filename=filename,
            )
            _atomic_write_bytes(target, payload, expected_root=self.root)
            return self._register_target(
                evidence_id=target_eid,
                kind=kind,
                description=description,
                target=target,
                sha256=digest,
                produced_by_step=produced_by_step,
                inputs=inputs,
                script_evidence_id=script_evidence_id,
                aliases=aliases,
                producer=producer,
                generation_mode=generation_mode,
                prompt_pack_version=prompt_pack_version,
                metadata=target_metadata,
                publish_aliases=publish_aliases,
                on_sha_change=target_on_sha_change,
            )

    def register_json(
        self,
        *,
        kind: str,
        description: str,
        payload: Any,
        filename: str,
        produced_by_step: Optional[str] = None,
        inputs: Optional[Sequence[str]] = None,
        evidence_id: Optional[str] = None,
        aliases: Optional[Sequence[str]] = None,
        producer: Optional[str] = None,
        generation_mode: Optional[str] = None,
        prompt_pack_version: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        publish_aliases: bool = True,
        on_sha_change: str = "raise",
    ) -> EvidenceRecord:
        text = json.dumps(payload, indent=2, ensure_ascii=False, default=str)
        return self.register_text(
            kind=kind,
            description=description,
            text=text,
            filename=filename,
            produced_by_step=produced_by_step,
            inputs=inputs,
            evidence_id=evidence_id,
            aliases=aliases,
            producer=producer,
            generation_mode=generation_mode,
            prompt_pack_version=prompt_pack_version,
            metadata=metadata,
            publish_aliases=publish_aliases,
            on_sha_change=on_sha_change,
        )

    def update_record(
        self,
        evidence_id: str,
        *,
        finding_severity: Optional[str] = None,
        finding_messages: Optional[Sequence[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        producer: Optional[str] = None,
        generation_mode: Optional[str] = None,
        prompt_pack_version: Optional[str] = None,
    ) -> Optional[EvidenceRecord]:
        with self._lock:
            record = self.get(evidence_id)
            if record is None:
                return None
            if finding_severity is not None:
                record.finding_severity = finding_severity
            if finding_messages is not None:
                record.finding_messages = list(finding_messages)
            if metadata:
                merged = dict(record.metadata)
                merged.update(metadata)
                record.metadata = merged
            if producer is not None:
                record.producer = producer
            if generation_mode is not None:
                record.generation_mode = generation_mode
            if prompt_pack_version is not None:
                record.prompt_pack_version = prompt_pack_version
            self._save()
            return record

    def publish_success_aliases(
        self,
        evidence_id: str,
        *,
        aliases: Optional[Sequence[str]] = None,
    ) -> Dict[str, str]:
        """Publish aliases after a step has passed every success gate.

        Callers can register draft/repair candidates with
        ``publish_aliases=False`` so no explicit alias, basename alias, or id
        alias becomes visible before validation and sealing. Once the owning
        step succeeds, this method publishes that complete alias surface.

        An existing alias may move only between records owned by the exact same
        non-empty ``produced_by_step``. Cross-step and run-level collisions
        retain their prior binding; the returned mapping contains only aliases
        that now resolve to ``evidence_id``.
        """

        from .runtime_artifacts import verified_run_evidence_path

        with self._lock:
            record = self._record_by_id(evidence_id)
            if record is None:
                raise KeyError(f"Unknown evidence id: {evidence_id}")
            owner = str(record.produced_by_step or "").strip()
            if not owner:
                raise ValueError(
                    "success-only alias publication requires produced_by_step"
                )
            if verified_run_evidence_path(self.root, record) is None:
                raise ValueError(
                    f"cannot publish aliases for unverified evidence: {evidence_id}"
                )

            target = self.root / record.relative_path
            requested = [
                *(str(alias).strip() for alias in aliases or []),
                _target_basename_stem(target, record.evidence_id),
                record.evidence_id,
            ]
            published: Dict[str, str] = {}
            for alias in requested:
                if not alias:
                    continue
                existing_id = self._aliases.get(alias)
                if existing_id not in {None, record.evidence_id}:
                    prior_record = self._record_by_id(existing_id)
                    prior_owner = str(
                        getattr(prior_record, "produced_by_step", None) or ""
                    ).strip()
                    if prior_owner != owner:
                        continue
                self._aliases[alias] = record.evidence_id
                published[alias] = record.evidence_id
            record.metadata = {
                **dict(record.metadata or {}),
                "aliases_published": True,
            }
            self._save()
            return published

    def publish_step_success_aliases(
        self,
        bindings: Mapping[str, Sequence[str]],
        *,
        step_id: str,
        suppressed_basename_evidence_ids: Sequence[str] = (),
    ) -> Dict[str, Dict[str, str]]:
        """Atomically promote one successful step's deferred alias surface.

        Unlike :meth:`publish_success_aliases`, this strict pipeline boundary
        rejects a stale alias owned by another step instead of silently
        retaining it. Every record and alias is validated before any in-memory
        authority is changed, so a failed attempt cannot publish only part of
        its result bundle.
        """

        from .runtime_artifacts import verified_run_evidence_path

        with self._lock:
            expected_owner = str(step_id or "").strip()
            if not expected_owner:
                raise ValueError("success promotion requires a non-empty step_id")
            prepared: List[Tuple[EvidenceRecord, List[str]]] = []
            suppressed_basename_ids = {
                str(evidence_id).strip()
                for evidence_id in suppressed_basename_evidence_ids
                if str(evidence_id).strip()
            }
            unknown_suppressions = suppressed_basename_ids.difference(
                str(evidence_id) for evidence_id in bindings
            )
            if unknown_suppressions:
                raise ValueError(
                    "basename suppression references evidence outside the success "
                    f"batch: {sorted(unknown_suppressions)}"
                )
            owners: set[str] = set()
            batch_alias_owners: Dict[str, str] = {}

            def _claim_batch_alias(alias: str, evidence_id: str) -> None:
                """Prove one alias has exactly one owner before any mutation."""

                if not alias:
                    return
                prior_evidence_id = batch_alias_owners.get(alias)
                if (
                    prior_evidence_id is not None
                    and prior_evidence_id != evidence_id
                ):
                    raise ValueError(
                        f"success promotion batch alias {alias!r} is claimed by "
                        f"both {prior_evidence_id!r} and {evidence_id!r}"
                    )
                batch_alias_owners[alias] = evidence_id

            for evidence_id, aliases in bindings.items():
                record = self._record_by_id(str(evidence_id))
                if record is None:
                    raise KeyError(f"Unknown evidence id: {evidence_id}")
                owner = str(record.produced_by_step or "").strip()
                if not owner:
                    raise ValueError(
                        "success-only alias publication requires produced_by_step"
                    )
                if owner != expected_owner:
                    raise ValueError(
                        f"evidence {evidence_id!r} is owned by step {owner!r}, "
                        f"not the successful step {expected_owner!r}"
                    )
                owners.add(owner)
                if verified_run_evidence_path(self.root, record) is None:
                    raise ValueError(
                        "cannot publish aliases for unverified evidence: "
                        f"{evidence_id}"
                    )
                target = self.root / record.relative_path
                explicit_aliases = list(
                    dict.fromkeys(str(alias).strip() for alias in aliases)
                )
                strict_aliases = list(
                    dict.fromkeys([*explicit_aliases, record.evidence_id])
                )
                for alias in strict_aliases:
                    if not alias:
                        continue
                    _claim_batch_alias(alias, record.evidence_id)
                    existing_id = self._aliases.get(alias)
                    if existing_id in {None, record.evidence_id}:
                        continue
                    prior_record = self._record_by_id(existing_id)
                    prior_owner = str(
                        getattr(prior_record, "produced_by_step", None) or ""
                    ).strip()
                    if prior_owner != owner:
                        raise ValueError(
                            f"success authority alias {alias!r} is already owned "
                            f"by step {prior_owner or '<run-level>'!r}"
                        )
                requested = list(strict_aliases)
                basename_alias = _target_basename_stem(target, record.evidence_id)
                if record.evidence_id not in suppressed_basename_ids:
                    _claim_batch_alias(basename_alias, record.evidence_id)
                    basename_owner_id = self._aliases.get(basename_alias)
                    if basename_owner_id not in {None, record.evidence_id}:
                        basename_record = self._record_by_id(basename_owner_id)
                        basename_owner = str(
                            getattr(basename_record, "produced_by_step", None) or ""
                        ).strip()
                        # Generic runner filenames such as ``step_summary`` and
                        # ``critique_report`` legitimately repeat across steps.
                        # Keep those ambiguous basename aliases first-write-only;
                        # explicit semantic aliases above remain strict.
                        if basename_owner == owner:
                            requested.append(basename_alias)
                    else:
                        requested.append(basename_alias)
                requested = list(dict.fromkeys(requested))
                prepared.append((record, requested))

            if len(owners) > 1:
                raise ValueError(
                    "one success promotion batch cannot mix evidence from "
                    f"multiple steps: {sorted(owners)}"
                )

            previous_aliases = dict(self._aliases)
            previous_metadata = {
                record.evidence_id: dict(record.metadata or {})
                for record, _ in prepared
            }
            published: Dict[str, Dict[str, str]] = {}
            try:
                for record, requested in prepared:
                    record_published: Dict[str, str] = {}
                    for alias in requested:
                        if not alias:
                            continue
                        self._aliases[alias] = record.evidence_id
                        record_published[alias] = record.evidence_id
                    record.metadata = {
                        **dict(record.metadata or {}),
                        "aliases_published": True,
                    }
                    published[record.evidence_id] = record_published
                self._save()
            except BaseException:
                self._aliases = previous_aliases
                for record, _ in prepared:
                    record.metadata = previous_metadata[record.evidence_id]
                raise
            return published

    def retire_step_current_aliases(
        self,
        evidence_ids: Sequence[str],
        *,
        step_id: str,
    ) -> Dict[str, str]:
        """Atomically retire current aliases after a step fails revalidation.

        Retirement is deliberately narrower than deleting a step's historical
        evidence.  The caller must name every candidate evidence id and prove
        the exact non-empty producing step.  Only aliases that *currently*
        target one of those indexed records are removed; aliases owned by a
        different step or by the run remain untouched.

        All records are validated before mutation.  If persistence fails, the
        in-memory alias table and record metadata are restored to their prior
        state so callers cannot observe a partially retired authority bundle.
        """

        return self.retire_steps_current_aliases({step_id: evidence_ids}).get(
            str(step_id), {}
        )

    def retire_steps_current_aliases(
        self,
        evidence_ids_by_step: Mapping[str, Sequence[str]],
    ) -> Dict[str, Dict[str, str]]:
        """Atomically retire the current alias surfaces of several steps.

        Cross-step deterministic revalidation is one authority transition.  A
        separate save per step could retire the first invalid step and fail on
        the second, splitting the alias ledger.  This batch validates every
        indexed owner before one mutation and one durable save.
        """

        with self._lock:
            requested_by_step: Dict[str, set[str]] = {}
            prepared: Dict[str, EvidenceRecord] = {}
            for raw_step_id, evidence_ids in evidence_ids_by_step.items():
                step_id = str(raw_step_id or "").strip()
                if not step_id:
                    raise ValueError("alias retirement requires a non-empty step_id")
                requested: set[str] = set()
                for raw_evidence_id in evidence_ids:
                    evidence_id = str(raw_evidence_id or "").strip()
                    if not evidence_id:
                        raise ValueError(
                            "alias retirement requires non-empty evidence ids"
                        )
                    record = self._record_by_id(evidence_id)
                    if record is None:
                        raise KeyError(f"Unknown evidence id: {evidence_id}")
                    owner = str(record.produced_by_step or "")
                    if owner != step_id:
                        raise ValueError(
                            f"evidence {evidence_id!r} is owned by step "
                            f"{owner or '<run-level>'!r}, not the revalidated "
                            f"step {step_id!r}"
                        )
                    requested.add(evidence_id)
                    prepared[evidence_id] = record
                if requested:
                    requested_by_step[step_id] = requested

            retired_by_step: Dict[str, Dict[str, str]] = {
                step_id: {
                    alias: evidence_id
                    for alias, evidence_id in self._aliases.items()
                    if evidence_id in requested
                }
                for step_id, requested in requested_by_step.items()
            }
            if not prepared:
                return retired_by_step

            previous_aliases = dict(self._aliases)
            previous_metadata = {
                evidence_id: dict(record.metadata or {})
                for evidence_id, record in prepared.items()
            }
            try:
                for retired in retired_by_step.values():
                    for alias in retired:
                        self._aliases.pop(alias, None)
                for record in prepared.values():
                    record.metadata = {
                        **dict(record.metadata or {}),
                        "aliases_published": False,
                    }
                self._save()
            except BaseException:
                self._aliases = previous_aliases
                for evidence_id, record in prepared.items():
                    record.metadata = previous_metadata[evidence_id]
                raise
            return retired_by_step

    # ------------------------------------------------------------------
    # Numeric claim registry (value-level provenance)
    # ------------------------------------------------------------------

    def register_numeric_claim(
        self,
        *,
        value: str,
        canonical: float,
        evidence_id: str,
        step_id: str,
        source_field: str,
        tolerance: float = 1e-3,
    ) -> NumericClaim:
        """Register a single numeric leaf for later manuscript binding.

        Idempotent on ``(evidence_id, step_id, source_field, canonical)`` —
        replaying the exact sealed evidence does not duplicate claims, while a
        later successful attempt receives its own digest-bound authority. The
        literal ``value`` is preserved with the most precise form seen so far.
        """
        with self._lock:
            for claim in self._numeric_claims:
                if (
                    claim.evidence_id == evidence_id
                    and claim.step_id == step_id
                    and claim.source_field == source_field
                    and abs(claim.canonical - canonical) <= claim.tolerance
                ):
                    if len(value) > len(claim.value):
                        claim.value = value
                    self._save()
                    return claim
            claim = NumericClaim(
                value=value,
                canonical=canonical,
                evidence_id=evidence_id,
                step_id=step_id,
                source_field=source_field,
                tolerance=tolerance,
            )
            self._numeric_claims.append(claim)
            self._save()
            return claim

    def register_step_summary_numerics(
        self,
        *,
        step_id: str,
        evidence_id: str,
        summary: Any,
        tolerance: float = 1e-3,
        max_leaves: Optional[int] = None,
    ) -> List[NumericClaim]:
        """Walk a ``step_summary`` payload and register every numeric leaf.

        This is the bulk registration hook invoked from the pipeline
        after a step's ``step_summary.json`` is loaded. Non-numeric
        leaves and structural fields are silently skipped.

        ``max_leaves`` caps the number of claims registered per call to
        prevent a single step that dumps a full interaction matrix into
        ``step_summary`` from drowning the registry. When the cap is
        exceeded the **first** ``max_leaves`` leaves (typically the most
        salient summary-level fields like ``primary_or``) are kept and
        the remainder is silently skipped; a single info-level marker
        claim records that truncation happened. ``None`` (the default
        when called directly) disables the cap. Pipelines pass the
        ``PipelineConfig.max_numeric_claims_per_step`` value.
        """
        registered: List[NumericClaim] = []
        leaves = _walk_numeric_leaves(summary)
        truncated = False
        if max_leaves is not None and max_leaves > 0 and len(leaves) > max_leaves:
            truncated_count = len(leaves) - max_leaves
            leaves = leaves[:max_leaves]
            truncated = True
        else:
            truncated_count = 0
        for path, literal, canonical in leaves:
            registered.append(
                self.register_numeric_claim(
                    value=literal,
                    canonical=canonical,
                    evidence_id=evidence_id,
                    step_id=step_id,
                    source_field=path,
                    tolerance=tolerance,
                )
            )
        if truncated:
            # A sentinel claim so reviewers see that the step exceeded
            # the cap without scrolling validator findings. The value is
            # the cap itself so an audit query can correlate cap value
            # → truncated step. The float is the count of *dropped*
            # leaves; matching it back to the cap is just (literal -
            # registered_count).
            self.register_numeric_claim(
                value=str(truncated_count),
                canonical=float(truncated_count),
                evidence_id=evidence_id,
                step_id=step_id,
                source_field="__easyicu_numeric_claim_overflow__",
                tolerance=tolerance,
            )
        return registered

    # ------------------------------------------------------------------
    # Derived numeric claims (Commit 2, Phase-1 widening)
    # ------------------------------------------------------------------

    def _resolve_derived_sources(
        self,
        *,
        sources: Dict[str, Tuple[str, str]],
    ) -> Tuple[Dict[str, float], List[Tuple[str, str]]]:
        """Look up canonical floats for the sources a formula names.

        ``sources`` is ``{formula_name: (source_step_id, source_field)}``.
        For every source the most recently registered claim whose
        ``(step_id, source_field)`` matches wins (matches the same
        latest-wins semantics as ``register_numeric_claim`` dedup).

        Returns ``(values, provenance)`` where ``provenance`` is a
        sorted list of ``(step_id, source_field)`` pairs for the
        ``derived_from`` field. Raises ``DerivedFormulaError`` when
        any source is unresolved — caller surfaces this to validator
        findings rather than ignoring silently.
        """
        values: Dict[str, float] = {}
        provenance: List[Tuple[str, str]] = []
        with self._lock:
            for name, (src_step, src_field) in sources.items():
                if name.startswith("_") or not name.isidentifier():
                    raise DerivedFormulaError(
                        f"derived-claim source name {name!r} is not a "
                        f"valid Python identifier (or starts with underscore)"
                    )
                match: Optional[NumericClaim] = None
                for claim in reversed(self._numeric_claims):
                    if claim.step_id == src_step and claim.source_field == src_field:
                        match = claim
                        break
                if match is None:
                    raise DerivedFormulaError(
                        f"derived-claim source {name!r} → "
                        f"({src_step}, {src_field}) not found in registry"
                    )
                values[name] = match.canonical
                provenance.append((src_step, src_field))
        provenance.sort()
        return values, provenance

    def register_derived_claim(
        self,
        *,
        name: str,
        formula: str,
        explanation: str,
        sources: Dict[str, Tuple[str, str]],
        evidence_id: str,
        step_id: str,
        tolerance: float = 1e-3,
    ) -> NumericClaim:
        """Evaluate ``formula`` in the restricted sandbox and register it.

        ``formula`` is a string in the grammar accepted by
        :func:`_evaluate_derived_formula`. ``sources`` maps formula
        identifiers to ``(step_id, source_field)`` pairs that must
        resolve to existing claims; the resolved canonical floats are
        substituted in. The result is registered as a NumericClaim
        with ``source_field=name``, ``formula``, ``explanation`` and
        ``derived_from`` populated — so the existing reverse-binder
        in ``manuscript_post.bind_numeric_values`` recognises it like
        any other claim.

        Raises ``DerivedFormulaError`` on syntax, unresolved sources,
        disallowed operators, division by zero, or non-finite result.
        """
        if not name or not name.isidentifier() or name.startswith("_"):
            raise DerivedFormulaError(
                f"derived-claim name {name!r} must be a non-empty Python "
                f"identifier not starting with underscore"
            )
        if not isinstance(formula, str) or not formula.strip():
            raise DerivedFormulaError(
                "derived-claim formula must be a non-empty string"
            )
        if not isinstance(explanation, str) or not explanation.strip():
            raise DerivedFormulaError(
                "derived-claim explanation must be a non-empty string "
                "(this surfaces in audit findings and the writer digest)"
            )
        source_values, provenance = self._resolve_derived_sources(sources=sources)
        result = _evaluate_derived_formula(formula, sources=source_values)
        # Use the same value/canonical pair shape as register_numeric_claim
        # so downstream tooling does not need to special-case derived.
        literal = (
            f"{result:.6g}" if not float(result).is_integer() else str(int(result))
        )
        with self._lock:
            for claim in self._numeric_claims:
                if (
                    claim.step_id == step_id
                    and claim.source_field == name
                    and abs(claim.canonical - result) <= claim.tolerance
                ):
                    # Idempotent re-registration (e.g. resume of the same
                    # step). Refresh formula/explanation in case the
                    # coder updated them between runs.
                    if len(literal) > len(claim.value):
                        claim.value = literal
                    claim.formula = formula.strip()
                    claim.explanation = explanation.strip()
                    claim.derived_from = list(provenance)
                    self._save()
                    return claim
            claim = NumericClaim(
                value=literal,
                canonical=float(result),
                evidence_id=evidence_id,
                step_id=step_id,
                source_field=name,
                tolerance=tolerance,
                formula=formula.strip(),
                explanation=explanation.strip(),
                derived_from=list(provenance),
            )
            self._numeric_claims.append(claim)
            self._save()
            return claim

    def register_step_derived_claims(
        self,
        *,
        step_id: str,
        evidence_id: str,
        summary: Any,
        tolerance: float = 1e-3,
    ) -> Tuple[List[NumericClaim], List[Dict[str, Any]]]:
        """Bulk-register all ``derived_claims`` entries from a step_summary.

        The coder declares derived numbers under a top-level
        ``derived_claims`` list in ``step_summary.json``::

            {
              "primary_or": 1.42,
              "primary_or_se": 0.13,
              "derived_claims": [
                {
                  "name": "primary_or_ci_low",
                  "formula": "exp(log(primary_or) - 1.96 * primary_or_se)",
                  "explanation": "Lower 95% CI for primary OR, log-normal approx",
                  "sources": {
                    "primary_or":   {"step_id": "<this-step>", "field": "primary_or"},
                    "primary_or_se": {"step_id": "<this-step>", "field": "primary_or_se"}
                  }
                }
              ]
            }

        Returns ``(registered_claims, errors)`` where ``errors`` is a
        list of ``{name, message}`` dicts for each entry that failed
        (bad formula, unresolved source, non-finite result, etc.). The
        pipeline turns each error into a ``derived_claim_error``
        validator finding; *registered_claims* is what enters the
        binding registry.

        Default ``step_id`` for omitted source ``step_id`` is the
        current step — most derived numbers are intra-step (build a
        CI from the same step's OR and SE). Cross-step derivations
        are supported by giving an explicit ``step_id``.
        """
        registered: List[NumericClaim] = []
        errors: List[Dict[str, Any]] = []
        entries: Any = None
        if isinstance(summary, dict):
            entries = summary.get("derived_claims")
        if not isinstance(entries, list):
            return registered, errors
        for idx, entry in enumerate(entries):
            try:
                if not isinstance(entry, dict):
                    raise DerivedFormulaError(f"derived_claims[{idx}] is not a dict")
                name = entry.get("name")
                formula = entry.get("formula")
                explanation = entry.get("explanation", "")
                raw_sources = entry.get("sources", {})
                if not isinstance(raw_sources, dict):
                    raise DerivedFormulaError(
                        f"derived_claims[{idx}].sources must be a dict"
                    )
                sources: Dict[str, Tuple[str, str]] = {}
                for src_name, ref in raw_sources.items():
                    if isinstance(ref, dict):
                        src_step = ref.get("step_id") or step_id
                        src_field = ref.get("field") or ref.get("source_field")
                    elif isinstance(ref, str):
                        # Shorthand: "field" inside the same step.
                        src_step = step_id
                        src_field = ref
                    else:
                        raise DerivedFormulaError(
                            f"derived_claims[{idx}].sources[{src_name!r}] "
                            f"must be a dict or a string, got "
                            f"{type(ref).__name__}"
                        )
                    if not src_field:
                        raise DerivedFormulaError(
                            f"derived_claims[{idx}].sources[{src_name!r}] "
                            f"missing 'field'"
                        )
                    sources[src_name] = (str(src_step), str(src_field))
                claim = self.register_derived_claim(
                    name=str(name) if name is not None else "",
                    formula=str(formula) if formula is not None else "",
                    explanation=str(explanation),
                    sources=sources,
                    evidence_id=evidence_id,
                    step_id=step_id,
                    tolerance=tolerance,
                )
                registered.append(claim)
            except DerivedFormulaError as exc:
                errors.append(
                    {
                        "name": str(
                            entry.get("name") if isinstance(entry, dict) else f"#{idx}"
                        ),
                        "message": str(exc),
                    }
                )
        return registered, errors

    def numeric_claims(self) -> List[NumericClaim]:
        with self._lock:
            return list(self._numeric_claims)

    def find_claim_for_value(
        self,
        value_str: str,
        *,
        tolerance: Optional[float] = None,
    ) -> Optional[NumericClaim]:
        """Look up the claim that best matches a numeric literal.

        Matching strategy (first hit wins):

        1. Exact literal equality (preserves precision/formatting).
        2. Canonical float equality.
        3. Relative-tolerance match against canonical (defaults to the
           claim's own tolerance; override with ``tolerance`` for a
           caller-specific window).

        Returns ``None`` if no claim matches — callers in STRICT mode
        should treat that as a binding failure.
        """
        raw = value_str.strip()
        has_percent = raw.endswith("%")
        stripped = raw.rstrip("%").replace(",", "")
        try:
            canonical = float(stripped)
        except ValueError:
            return None
        display_places = _decimal_places(stripped)
        display_abs_tol = 0.0
        if display_places > 0:
            display_abs_tol = 0.5 * (10 ** (-display_places))
        with self._lock:
            for claim in self._numeric_claims:
                if claim.value == raw:
                    return claim
            for claim in self._numeric_claims:
                if claim.canonical == canonical:
                    return claim
            if has_percent:
                for claim in self._numeric_claims:
                    if claim.canonical * 100.0 == canonical:
                        return claim
            for claim in self._numeric_claims:
                window = tolerance if tolerance is not None else claim.tolerance
                candidate = claim.canonical * 100.0 if has_percent else claim.canonical
                if abs(candidate) > 1e-9:
                    rel = abs(candidate - canonical) / abs(candidate)
                else:
                    rel = 0.0 if abs(canonical) <= 1e-12 else float("inf")
                abs_window = max(
                    display_abs_tol, window * max(abs(candidate), abs(canonical))
                )
                if rel <= window or abs(candidate - canonical) <= abs_window:
                    return claim
        return None

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def records(self) -> List[EvidenceRecord]:
        with self._lock:
            return list(self._records)

    def ids(self) -> List[str]:
        with self._lock:
            return [r.evidence_id for r in self._records]

    def aliases(self) -> Dict[str, str]:
        with self._lock:
            return dict(self._aliases)

    def get(self, evidence_id_or_alias: str) -> Optional[EvidenceRecord]:
        with self._lock:
            # Direct id match first.
            for r in self._records:
                if r.evidence_id == evidence_id_or_alias:
                    return r
            # Then alias lookup. Resolve under the same lock so a
            # concurrent register_file can never invalidate the alias
            # → record path mid-lookup.
            eid = self._aliases.get(evidence_id_or_alias)
            if eid is not None:
                for r in self._records:
                    if r.evidence_id == eid:
                        return r
            # Hosted writers sometimes cite the stable basename of a
            # hash-suffixed artefact, e.g. ``figure_mortality`` for
            # ``figure_mortality_ab12cd34``. Accept this only when the
            # prefix is unique so we do not silently bind an ambiguous claim.
            prefix = f"{evidence_id_or_alias}_"
            candidate_ids = {
                r.evidence_id
                for r in self._records
                if r.evidence_id.startswith(prefix)
                and bool((r.metadata or {}).get("aliases_published", True))
            }
            candidate_ids.update(
                eid for alias, eid in self._aliases.items() if alias.startswith(prefix)
            )
            if len(candidate_ids) == 1:
                only = next(iter(candidate_ids))
                for r in self._records:
                    if r.evidence_id == only:
                        return r
        return None

    def resolvable_names(self) -> List[str]:
        """Every name the binder will accept (evidence_ids + aliases)."""
        with self._lock:
            return sorted(
                set(r.evidence_id for r in self._records) | set(self._aliases)
            )

    def current_verified_records(
        self,
        per_step_records: Optional[Sequence[Mapping[str, Any]]],
    ) -> List[EvidenceRecord]:
        """Return current evidence whose content still matches its digest.

        The append-only store retains historical records across resume.  A
        manuscript reader must therefore apply both the latest-step authority
        ledger and the on-disk digest check instead of trusting the first-write
        alias table or the evidence index by itself.

        ``None`` preserves the legacy reader contract for callers that do not
        have a step ledger.  Production manuscript paths always pass the
        ledger and therefore receive the strict current/digest-verified view.
        """

        records = self.records()
        if per_step_records is None:
            return records

        # Local import keeps EvidenceStore usable by the low-level runtime
        # artifact module without introducing a module-import cycle.
        from .runtime_artifacts import (
            current_evidence_records,
            verified_run_evidence_path,
        )

        return [
            record
            for record in current_evidence_records(records, per_step_records)
            if verified_run_evidence_path(self.root, record) is not None
        ]

    def current_resolvable_names(
        self,
        per_step_records: Optional[Sequence[Mapping[str, Any]]],
    ) -> List[str]:
        """Names resolving to current, digest-verified evidence only."""

        if per_step_records is None:
            return self.resolvable_names()
        current_ids = {
            record.evidence_id
            for record in self.current_verified_records(per_step_records)
        }
        with self._lock:
            current_aliases = {
                alias
                for alias, evidence_id in self._aliases.items()
                if evidence_id in current_ids
            }
        return sorted(current_ids | current_aliases)

    def authoritative_numeric_claims(
        self,
        per_step_records: Optional[Sequence[Mapping[str, Any]]],
    ) -> List[NumericClaim]:
        """Numeric claims bound to current evidence owner and file digest.

        A flat active-evidence-id set is not sufficient: a claim from one step
        could borrow an id owned by another step, and a deleted or modified
        evidence blob would still look registered.  Step claims must therefore
        match the exact producing step and its current evidence ids.  The two
        host-owned run-level numeric families have explicit kind/producer
        contracts and remain citable without an analysis-step record.
        """

        claims = self.numeric_claims()
        if per_step_records is None:
            return claims

        from .runtime_artifacts import (
            active_step_evidence_ids_by_step,
            run_level_evidence_matches_claim_owner,
        )

        records_by_id = {
            record.evidence_id: record
            for record in self.current_verified_records(per_step_records)
        }
        active_ids_by_step = active_step_evidence_ids_by_step(per_step_records)
        run_level_contracts = {
            "research_context": ("log", "pipeline"),
            "robustness_panel": ("statistic", "pipeline"),
        }
        authoritative: List[NumericClaim] = []
        for claim in claims:
            record = records_by_id.get(claim.evidence_id)
            if record is None:
                continue
            claim_step = str(claim.step_id or "").strip()
            record_step = str(record.produced_by_step or "").strip()
            if record_step:
                if (
                    record_step == claim_step
                    and claim.evidence_id in active_ids_by_step.get(record_step, set())
                ):
                    authoritative.append(claim)
                continue

            contract = run_level_contracts.get(claim_step)
            if contract is None:
                continue
            expected_kind, expected_producer = contract
            if (
                record.kind == expected_kind
                and str(record.producer or "").strip() == expected_producer
                and run_level_evidence_matches_claim_owner(
                    claim_step_id=claim_step,
                    evidence_id=claim.evidence_id,
                )
            ):
                authoritative.append(claim)
        return authoritative

    # ------------------------------------------------------------------
    # Manuscript binding
    # ------------------------------------------------------------------

    def enforce_evidence_bound_scaffold(self, scaffold: str) -> tuple[str, List[str]]:
        """Drop result-like sentences that lack an explicit evidence placeholder.

        The writer is allowed to draft prose freely, but anything that looks like
        a numerical result or analytical claim must cite ``{evidence:<id>}``
        before it can enter the final manuscript. We keep headings, structural
        Markdown, and non-result narrative intact, but list/blockquote markers
        do not exempt the claim that follows them. The filtered scaffold and a
        list of sentences that were removed are returned.

        In ``STRICT`` mode, raises :class:`EvidenceEnforcementError` when any
        sentence would have been dropped, so a CI / submission run fails loudly
        instead of shipping a silently shortened manuscript.
        """
        removed: List[str] = []
        filtered_lines: List[str] = []
        for raw_line in scaffold.splitlines():
            line = raw_line.rstrip()
            stripped = line.strip()
            if stripped.startswith(("```", "~~~")):
                filtered_lines.append(line)
                continue
            if not stripped:
                filtered_lines.append(line)
                continue
            structure_prefix, content = _split_markdown_structure_prefix(line)
            heading_prefix, heading_content = _split_markdown_heading_prefix(content)
            if heading_prefix:
                if not _heading_requires_evidence(heading_content):
                    filtered_lines.append(line)
                    continue
                structure_prefix += heading_prefix
                content = heading_content
            content_stripped = content.strip()
            if not content_stripped:
                filtered_lines.append(line)
                continue
            sentences = _split_sentences(content)
            if len(sentences) == 1 and not _looks_result_like_sentence(sentences[0]):
                filtered_lines.append(line)
                continue
            kept: List[str] = []
            for sentence in sentences:
                if (
                    _looks_result_like_sentence(sentence)
                    and "{evidence:" not in sentence
                ):
                    removed.append(sentence.strip())
                    continue
                kept.append(sentence.strip())
            kept_content = " ".join(part for part in kept if part).strip()
            filtered_lines.append(
                f"{structure_prefix}{kept_content}".rstrip() if kept_content else ""
            )
        if removed and self.enforcement_mode is EvidenceEnforcementMode.STRICT:
            raise EvidenceEnforcementError(
                f"STRICT evidence mode: {len(removed)} result-like sentence(s) "
                f"without {{evidence:<id>}} placeholders. The writer must cite "
                f"registered evidence ids for every analytical claim.",
                detail={"removed_sentences": removed},
            )
        return "\n".join(filtered_lines).strip() + "\n", removed

    def bind_manuscript(
        self,
        scaffold: str,
        *,
        verbose: bool = False,
        per_step_records: Optional[Sequence[Mapping[str, Any]]] = None,
    ) -> str:
        """Replace ``{evidence:<id>}`` placeholders with provenance links.

        Default mode emits a compact markdown link
        ``[<id>](relative_path)`` that reads naturally inside a sentence
        ("see Table 1 [table_one](evidence/...)"). Set ``verbose=True``
        to keep the older ``[description | path | sha256=…]`` form for
        machine-readable tracking.

        Unbound placeholders are replaced with ``[evidence missing: id]``
        so a reviewer can immediately see what the writer expected to
        cite.

        In ``STRICT`` mode, any unresolved placeholder raises
        :class:`EvidenceEnforcementError` so the run fails before a
        manuscript containing ``[evidence missing: …]`` markers can be
        written out.
        """
        current_ids = (
            {
                record.evidence_id
                for record in self.current_verified_records(per_step_records)
            }
            if per_step_records is not None
            else None
        )
        out: List[str] = []
        all_missing: List[str] = []
        i = 0
        n = len(scaffold)
        while i < n:
            j = scaffold.find("{evidence:", i)
            if j < 0:
                out.append(scaffold[i:])
                break
            k = scaffold.find("}", j)
            if k < 0:
                out.append(scaffold[j:])
                break
            double_wrapped = (
                j > i
                and scaffold[j - 1] == "{"
                and k + 1 < n
                and scaffold[k + 1] == "}"
            )
            out.append(scaffold[i : j - 1 if double_wrapped else j])
            eid = scaffold[j + len("{evidence:") : k]
            requested_ids = [
                _normalize_requested_evidence_id(item)
                for item in eid.split(",")
                if _normalize_requested_evidence_id(item)
            ]
            if not requested_ids:
                requested_ids = [eid]
            bound_parts: List[str] = []
            missing: List[str] = []
            for requested_id in requested_ids:
                rec = self.get(requested_id)
                if rec is None or (
                    current_ids is not None and rec.evidence_id not in current_ids
                ):
                    missing.append(requested_id)
                    continue
                if verbose:
                    suffix = _binding_caveat(rec, verbose=verbose)
                    bound_parts.append(
                        f"[{rec.description} | {rec.relative_path} | sha256={rec.sha256[:8]}]{suffix}"
                    )
                else:
                    bound_parts.append(
                        f'[{requested_id}]({rec.relative_path} "sha256={rec.sha256[:8]}")'
                        f"{_binding_caveat(rec, verbose=verbose)}"
                    )
            if missing:
                bound_parts.extend(f"[evidence missing: {item}]" for item in missing)
                all_missing.extend(missing)
            if bound_parts:
                out.append("; ".join(bound_parts))
            elif verbose:
                out.append(f"[evidence missing: {eid}]")
                all_missing.append(eid)
            i = k + 2 if double_wrapped else k + 1
        if all_missing and self.enforcement_mode is EvidenceEnforcementMode.STRICT:
            unique_missing = sorted(set(all_missing))
            raise EvidenceEnforcementError(
                f"STRICT evidence mode: {len(unique_missing)} manuscript "
                f"placeholder(s) do not resolve to a registered evidence id: "
                f"{', '.join(unique_missing)}. Register the underlying "
                f"artefact, or correct the placeholder before binding.",
                detail={"missing_evidence_ids": unique_missing},
            )
        return "".join(out)


def _normalize_requested_evidence_id(value: str) -> str:
    """Normalize a manuscript placeholder item before alias lookup.

    Some writer models emit comma placeholders as
    ``{evidence:a, evidence:b}`` rather than ``{evidence:a, b}``.
    Treat the repeated prefix as harmless syntax noise.
    """

    item = (value or "").strip()
    if item.lower().startswith("evidence:"):
        item = item.split(":", 1)[1].strip()
    return item


def _target_basename_stem(target: Path, evidence_id: str) -> str:
    """Return the original filename stem from ``<evidence_id>__<filename>``.

    Evidence ids themselves may contain a doubled underscore when the id prefix
    ends with ``_``. Splitting on the first ``__`` then corrupts the alias.
    """

    prefix = f"{evidence_id}__"
    name = target.name
    if name.startswith(prefix):
        return Path(name[len(prefix) :]).stem
    return Path(name.split("__", 1)[-1]).stem


def _id_prefix(kind: str, stem: str) -> str:
    safe = "".join(c for c in stem if c.isalnum() or c in "_-").strip("_")[:32]
    return f"{kind}_{safe}" if safe else kind


def _binding_caveat(record: EvidenceRecord, *, verbose: bool = False) -> str:
    severity = record.finding_severity
    if severity in {"warning", "error"}:
        if verbose:
            return f" ({severity}: see manifest)"
        return f"<!-- {severity}: see manifest -->"
    return ""


_RESULT_TOKEN_RE = re.compile(
    r"(\bOR\b|\bHR\b|\bRR\b|\bAUC\b|\bAUROC\b|\bBrier\b|\bcalibration\b|"
    r"\bdiscrimination\b|\bperformance\b|\brobust(?:ness)?\b|"
    r"\boverfitting\b|\bmiscalibration\b|\bmissingness\b|\bconsistent\b|"
    r"\bgeneralisa(?:bility|ble)\b|"
    r"\bgeneraliza(?:bility|ble)\b|"
    r"\bmedian\b|\bmean\b|\bincidence\b|\bmortality\b|\bhazard\b|"
    r"\bconfidence interval\b|\bCI\b|\bp\s*[<=>]|%|\d)",
    re.I,
)
_MANUSCRIPT_METADATA_PREFIX_RE = re.compile(
    r"^\s*(?:\*\*)?"
    r"(?:keywords?|key words|funding|conflicts?\s+of\s+interest|"
    r"data\s+(?:and\s+code\s+)?availability|code\s+availability|"
    r"ethics\s+approval|acknowledg(?:e)?ments?)"
    r"\s*(?:\*\*)?\s*[:：]",
    re.I,
)
_AVAILABILITY_BOILERPLATE_RE = re.compile(
    r"\b(?:generated scripts?|sha-?256|evidence store|reproducibility envelope|"
    r"strobe checklist|supplementary tables?|released alongside|available from|"
    r"data availability|code availability)\b",
    re.I,
)
_AVAILABILITY_ACTION_RE = re.compile(
    r"\b(?:released|available|deposited|archived|provided|shared|included)\b",
    re.I,
)


def _looks_manuscript_metadata_sentence(sentence: str) -> bool:
    """Return True for non-analytic manuscript front/back matter."""
    stripped = sentence.strip()
    if _MANUSCRIPT_METADATA_PREFIX_RE.search(stripped):
        return True
    if _AVAILABILITY_BOILERPLATE_RE.search(stripped) and _AVAILABILITY_ACTION_RE.search(
        stripped
    ):
        return True
    return False


def _split_sentences(text: str) -> List[str]:
    parts = [
        part.strip()
        for part in re.split(r"(?<=[.!?。！？])\s+", text.strip())
        if part.strip()
    ]
    return parts or ([text.strip()] if text.strip() else [])


def _split_markdown_structure_prefix(line: str) -> tuple[str, str]:
    """Separate list/blockquote syntax from the prose it cannot exempt."""

    cursor = len(line) - len(line.lstrip())
    marker_re = re.compile(r"(?:>\s*|[-+*]\s+|\d+[.)]\s+)")
    while match := marker_re.match(line, cursor):
        cursor = match.end()
    return line[:cursor], line[cursor:]


def _split_markdown_heading_prefix(content: str) -> tuple[str, str]:
    """Separate an ATX heading marker from its potentially assertive text."""

    match = re.match(r"^(\s*#{1,6}(?:\s+|$))", content)
    if match is None:
        return "", content
    return match.group(1), content[match.end() :]


_HEADING_RESULT_ASSERTION_RE = re.compile(
    r"\b(?:higher|lower|greater|less|increas(?:e|ed|es|ing)|"
    r"decreas(?:e|ed|es|ing)|reduc(?:e|ed|es|ing)|elevated|"
    r"declin(?:e|ed|es|ing)|improv(?:e|ed|es|ing)|worsen(?:ed|s|ing)?|"
    r"associated|correlated|predicted|significant(?:ly)?|unchanged|similar)\b",
    re.I,
)
_HEADING_RESULT_CONTEXT_RE = re.compile(
    r"\b(?:OR|HR|RR|AUC|AUROC|Brier|calibration|discrimination|"
    r"median|mean|incidence|mortality|hazard|confidence interval|CI|p)\b",
    re.I,
)
_HEADING_NUMERIC_RE = re.compile(r"(?:\d|%|\bp\s*[<=>])", re.I)
_HEADING_RESULT_VERB_RE = re.compile(
    r"\b(?:was|were|had|showed|demonstrated|differ(?:ed|s)?|varied)\b",
    re.I,
)


def _heading_requires_evidence(content: str) -> bool:
    """Return whether a heading states a result rather than naming a section."""

    stripped = content.strip()
    if not stripped:
        return False
    # A leading outline number is structural (``2. Results``), not a result.
    semantic = re.sub(r"^\d+(?:\.\d+)*[.)]?\s+", "", stripped, count=1)
    if _HEADING_RESULT_ASSERTION_RE.search(semantic):
        return True
    if _HEADING_RESULT_CONTEXT_RE.search(semantic) and _HEADING_RESULT_VERB_RE.search(
        semantic
    ):
        return True
    return bool(
        _HEADING_NUMERIC_RE.search(semantic)
        and _HEADING_RESULT_CONTEXT_RE.search(semantic)
    )


def _looks_result_like_sentence(sentence: str) -> bool:
    if "{evidence:" in sentence:
        return False
    if _looks_manuscript_metadata_sentence(sentence):
        return False
    return bool(_RESULT_TOKEN_RE.search(sentence))


__all__ = [
    "EvidenceStore",
    "EvidenceEnforcementMode",
    "EvidenceEnforcementError",
    "NumericClaim",
    "DerivedFormulaError",
    "sha256_of_file",
    "sha256_of_bytes",
]
