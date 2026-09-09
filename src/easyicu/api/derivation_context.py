"""Private, content-addressed inputs to derived labels, separate from data views.

This owner persists calls, not clinical algorithms. SOFA-1 Sepsis replay uses
the existing kernel and native publication filter. References never authorize
these patient-level inputs as research variables or external-LLM material.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Iterator, Mapping, Sequence

import pandas as pd

PRIVATE_DIRECTORY = ".derivation-context"
CONTEXT_SCHEMA = "easyicu_derivation_context_v1"
CONCEPT = "sep3_sofa1"
MODULE = "sepsis3_sofa1"
MAX_CONTEXT_STAYS = 128
MAX_CONTEXT_ROWS = 65536
MAX_CONTEXT_DECODED_BYTES = 32 * 1024 * 1024


class DerivationContextError(ValueError):
    """Complete derivation verification is unavailable or has failed."""


@dataclass(frozen=True)
class ContextReference:
    file: str
    sha256: str
    bytes: int

    def to_dict(self) -> dict:
        return asdict(self)


def _digest(handle) -> str:
    digest = hashlib.sha256()
    for block in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()


def _json_bytes(value) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def _safe_path(root: Path, name: str, *, private: bool = True) -> Path:
    relative = Path(name)
    if (
        not name
        or relative.is_absolute()
        or ".." in relative.parts
        or (private and relative.parts[0] != PRIVATE_DIRECTORY)
    ):
        raise DerivationContextError("invalid derivation context path")
    current = root
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            raise DerivationContextError("derivation context symlink refused")
    return current


@contextmanager
def _snapshot(root: Path, ref: Mapping, *, private: bool = True) -> Iterator:
    """Hash the exact anonymous snapshot consumed, with bounded copy buffers."""
    try:
        path = _safe_path(root, ref["file"], private=private)
        with path.open("rb") as source, tempfile.TemporaryFile() as snapshot:
            shutil.copyfileobj(source, snapshot, length=1024 * 1024)
            size = snapshot.tell()
            snapshot.seek(0)
            digest = _digest(snapshot)
            if digest != ref["sha256"] or size != ref["bytes"]:
                raise DerivationContextError(
                    "derivation context content binding mismatch"
                )
            snapshot.seek(0)
            yield snapshot
    except (KeyError, OSError, TypeError) as exc:
        raise DerivationContextError("derivation context missing or invalid") from exc


def _store(root: Path, *, frame=None, payload=None, source=None, suffix=None) -> dict:
    directory = _safe_path(root, PRIVATE_DIRECTORY)
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    suffix = suffix or (".parquet" if frame is not None else ".json")
    descriptor, name = tempfile.mkstemp(prefix=".partial-", dir=directory)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            if source is not None:
                shutil.copyfileobj(source, handle, length=1024 * 1024)
            elif frame is not None:
                frame.to_parquet(handle, index=False, engine="pyarrow")
            else:
                handle.write(_json_bytes(payload))
        with temporary.open("rb") as handle:
            digest = _digest(handle)
        destination = directory / (digest + suffix)
        ref = ContextReference(
            destination.relative_to(root).as_posix(), digest, temporary.stat().st_size
        ).to_dict()
        # Exclusive creation; a content address is never replaced or repaired.
        try:
            os.link(temporary, destination)
            destination.chmod(0o400)
        except FileExistsError:
            with _snapshot(root, ref):
                pass
        return ref
    finally:
        temporary.unlink(missing_ok=True)


def _read_json(root: Path, ref: Mapping) -> dict:
    with _snapshot(root, ref) as handle:
        try:
            value = json.load(handle)
        except (ValueError, UnicodeError) as exc:
            raise DerivationContextError("invalid derivation context JSON") from exc
    if not isinstance(value, dict):
        raise DerivationContextError("invalid derivation context object")
    return value


def _read_frame(root: Path, ref: Mapping, *, dependency: bool = False) -> pd.DataFrame:
    with _snapshot(root, ref) as handle:
        try:
            if dependency:
                import pyarrow.parquet as pq

                metadata = pq.ParquetFile(handle).metadata
                decoded = sum(
                    metadata.row_group(i).total_byte_size
                    for i in range(metadata.num_row_groups)
                )
                if (
                    metadata.num_rows > MAX_CONTEXT_ROWS
                    or decoded > MAX_CONTEXT_DECODED_BYTES
                ):
                    raise DerivationContextError(
                        "derivation dependency shard exceeds read bound"
                    )
                handle.seek(0)
            return pd.read_parquet(handle)
        except DerivationContextError:
            raise
        except Exception as exc:
            raise DerivationContextError("invalid derivation context table") from exc


def _ids(values: Sequence) -> list[int]:
    numeric = pd.to_numeric(pd.Series(list(values), dtype="object"), errors="coerce")
    if numeric.isna().any() or (numeric % 1 != 0).any():
        raise DerivationContextError("invalid derivation cohort identity")
    ids = numeric.astype("int64").tolist()
    if len(ids) != len(set(ids)):
        raise DerivationContextError("duplicate derivation cohort identity")
    return sorted(ids)


def _cohort_frame(ids) -> pd.DataFrame:
    return pd.DataFrame({"stay_id": pd.Series(_ids(ids), dtype="int64")})


def _read_cohort(root: Path, ref: Mapping) -> list[int]:
    import pyarrow.parquet as pq

    ids = set()
    with _snapshot(root, ref) as handle:
        for batch in pq.ParquetFile(handle).iter_batches(columns=["stay_id"]):
            current = set(_ids(batch.column(0).to_pylist()))
            if ids.intersection(current):
                raise DerivationContextError("duplicate derivation cohort identity")
            ids.update(current)
    return sorted(ids)


def _label_events(frame):
    if frame.empty:
        return set()
    if frame["stay_id"].duplicated().any() or not frame[CONCEPT].eq(True).all():
        raise DerivationContextError("published label grain or value mismatch")
    return set(zip(frame.stay_id.astype(int), frame.charttime.astype(float)))


def _code_binding() -> dict:
    package = Path(__file__).resolve().parents[1]
    # Bind actual executable owners, including the publication policy constants.
    names = [
        "api/derivation_context.py",
        "api/extraction.py",
        "scores/sepsis.py",
        "utils/time_units.py",
        "data/concept-dict.json",
        "data/data-sources.json",
    ]
    return {
        name: hashlib.sha256((package / name).read_bytes()).hexdigest()
        for name in names
    }


def _kernel(sofa, suspicion, *, id_col: str, time_col: str):
    from ..scores.sepsis import sep3

    return sep3(sofa, suspicion, id_cols=[id_col], index_col=time_col)


class SepsisContextRecorder:
    """Record complete actual kernel arguments and every requested batch.

    A skipped batch records the SI input and the existing short-circuit reason;
    it does not invent an unconsumed score timeline or turn absence into a label.
    """

    def __init__(self, root: Path, *, database: str, data_path: str):
        self.root = Path(root)
        self.database = database
        self.data_path = str(data_path)
        self.code = _code_binding()
        self.batches: list[dict] = []
        self.cohort: set[int] = set()

    def record(self, sofa, suspicion, *, ids, id_col, time_col, action="kernel"):
        cohort = _ids(ids)
        if self.cohort.intersection(cohort):
            raise DerivationContextError("derivation batch cohort overlap")
        for frame in (sofa, suspicion):
            if frame is not None and not frame.empty:
                if id_col not in frame or not frame[id_col].isin(cohort).all():
                    raise DerivationContextError("derivation input cohort drift")
        # Partition only between stays. A stay's entire consumed timeline must
        # remain together for firstSI and cumulative-minimum semantics. This
        # also bounds replay when the original producer call was one-shot.
        counts = pd.Series(0, index=cohort, dtype="int64")
        for frame in (sofa, suspicion):
            if frame is not None and not frame.empty:
                counts = counts.add(frame[id_col].value_counts(), fill_value=0)
        if (counts > MAX_CONTEXT_ROWS).any():
            raise DerivationContextError(
                "single-stay context exceeds replay bound; no timeline truncated"
            )
        shards, shard, rows = [], [], 0
        for identity in cohort:
            size = int(counts.loc[identity])
            if shard and (
                len(shard) >= MAX_CONTEXT_STAYS or rows + size > MAX_CONTEXT_ROWS
            ):
                shards.append(shard)
                shard, rows = [], 0
            shard.append(identity)
            rows += size
        if shard or not shards:
            shards.append(shard)
        for shard in shards:

            def select(frame):
                if frame is None or frame.empty:
                    return frame
                selected = frame.loc[frame[id_col].isin(shard)]
                if (
                    selected.memory_usage(index=False, deep=True).sum()
                    > MAX_CONTEXT_DECODED_BYTES
                ):
                    raise DerivationContextError(
                        "derivation argument shard exceeds memory bound"
                    )
                return selected

            self._record_shard(
                select(sofa),
                select(suspicion),
                ids=shard,
                id_col=id_col,
                time_col=time_col,
                action=action,
            )
        self.cohort.update(cohort)

    def _record_shard(self, sofa, suspicion, *, ids, id_col, time_col, action):
        self.batches.append(
            {
                "cohort": _store(self.root, frame=_cohort_frame(ids)),
                "sofa": _store(self.root, frame=sofa) if sofa is not None else None,
                "suspicion": _store(self.root, frame=suspicion),
                "id_col": id_col,
                "time_col": time_col,
                "action": action,
                "sofa_dtypes": {name: str(dtype) for name, dtype in sofa.dtypes.items()}
                if sofa is not None
                else None,
                "suspicion_dtypes": {
                    name: str(dtype) for name, dtype in suspicion.dtypes.items()
                },
            }
        )

    def derive(self, sofa, suspicion, *, ids, id_col, time_col):
        self.record(sofa, suspicion, ids=ids, id_col=id_col, time_col=time_col)
        return _kernel(sofa, suspicion, id_col=id_col, time_col=time_col)

    def finish(self, *, expected_ids=None) -> dict:
        if expected_ids is not None and set(_ids(expected_ids)) != self.cohort:
            raise DerivationContextError("derivation batch queue incomplete")
        if self.code != _code_binding():
            raise DerivationContextError(
                "derivation producer code changed during capture"
            )
        return _store(
            self.root,
            payload={
                "schema_version": CONTEXT_SCHEMA,
                "concept": CONCEPT,
                "database": self.database,
                "data_path": self.data_path,
                "code": self.code,
                "kernel": "easyicu.scores.sepsis.sep3:defaults",
                "cohort": _store(self.root, frame=_cohort_frame(self.cohort)),
                "batches": self.batches,
            },
        )


def transfer_context(source: Path, destination: Path, reference: dict) -> dict:
    """Retain worker inputs before the ordinary worker-directory cleanup."""
    payload = _read_json(source, reference)
    refs = [payload["cohort"]]
    for batch in payload["batches"]:
        refs.extend(
            batch[name] for name in ("cohort", "sofa", "suspicion") if batch[name]
        )
    for ref in refs:
        with _snapshot(source, ref) as handle:
            copied = _store(destination, source=handle, suffix=".parquet")
        if copied != ref:
            raise DerivationContextError(
                "derivation context transfer changed serialization"
            )
    copied = _store(destination, payload=payload)
    if copied != reference:
        raise DerivationContextError("derivation context transfer changed descriptor")
    return copied


def _binding(manifest: Mapping) -> dict:
    return {
        key: manifest.get(key)
        for key in (
            "database",
            "data_path",
            "runtime_provenance",
            "time_window_authority",
            "files",
            "concept_selection",
        )
    }


def seal_context(
    root: Path,
    manifest: dict,
    *,
    context: dict | None,
    upper_bounds: dict,
    expected_ids=None,
) -> dict:
    """Bind captured dependencies to these exact public views and label P."""
    if context is None:
        return {"status": "unavailable", "reason": "producer_context_missing"}
    captured = _read_json(root, context)
    cohort = _read_cohort(root, captured["cohort"])
    if expected_ids is not None and cohort != _ids(expected_ids):
        raise DerivationContextError("derivation publication cohort drift")
    if (
        captured["database"] != manifest["database"]
        or captured["data_path"] != manifest["data_path"]
        or captured["code"] != _code_binding()
    ):
        raise DerivationContextError("derivation producer provenance drift")
    bounds = pd.DataFrame(
        {
            "stay_id": pd.Series(list(upper_bounds), dtype="int64"),
            "upper": pd.Series(list(upper_bounds.values()), dtype="float64"),
        }
    )
    reference = _store(
        root,
        payload={
            "schema_version": CONTEXT_SCHEMA,
            "inputs": context,
            "binding": _binding(manifest),
            "upper_bounds": _store(root, frame=bounds),
            "label_publication": "easyicu.api.extraction._enforce_native_export_time_axis",
        },
    )
    return {
        "status": "bound",
        "scope": "private_derivation_inputs",
        "research_variables": False,
        "external_llm_allowed": False,
        "reference": reference,
    }


def validate_native_derivation_context(
    output_dir: str | Path,
    *,
    expected_manifest_sha256: str,
    expected_patient_ids: Sequence | None = None,
) -> dict:
    """Replay bound full inputs, then the SAME label publication filter.

    Caller supplies its retained manifest hash. Returns aggregate engineering
    consistency only; never source/clinical approval or case-level records.
    Legacy or incomplete packages cannot pass this stronger validation.
    """
    root = Path(output_dir)
    path = _safe_path(root, "_manifest.json", private=False)
    try:
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected_manifest_sha256:
            raise DerivationContextError("native manifest binding mismatch")
        manifest = json.loads(raw)
    except (ValueError, OSError) as exc:
        raise DerivationContextError("native manifest invalid or missing") from exc
    receipt = validate_derivation_manifest(
        root, manifest, expected_patient_ids=expected_patient_ids
    )
    receipt["manifest_sha256"] = expected_manifest_sha256
    return receipt


def validate_derivation_manifest(
    root: Path, manifest: Mapping, *, expected_patient_ids=None
) -> dict:
    """Publisher and readback share one validator; publisher passes its sealed draft."""
    try:
        descriptor = manifest["derivation_contexts"][CONCEPT]
        if descriptor["status"] != "bound":
            raise DerivationContextError("complete derivation context unavailable")
        if (
            descriptor.get("scope") != "private_derivation_inputs"
            or descriptor.get("research_variables") is not False
            or descriptor.get("external_llm_allowed") is not False
        ):
            raise DerivationContextError("invalid private derivation scope")
        envelope = _read_json(root, descriptor["reference"])
        inputs = _read_json(root, envelope["inputs"])
        if (
            envelope["schema_version"] != CONTEXT_SCHEMA
            or inputs["schema_version"] != CONTEXT_SCHEMA
            or inputs["concept"] != CONCEPT
            or inputs["kernel"] != "easyicu.scores.sepsis.sep3:defaults"
            or envelope["label_publication"]
            != "easyicu.api.extraction._enforce_native_export_time_axis"
            or envelope["binding"] != _binding(manifest)
            or inputs["code"] != _code_binding()
            or inputs["database"] != manifest["database"]
            or inputs["data_path"] != manifest["data_path"]
        ):
            raise DerivationContextError("derivation provenance or executable drift")
        cohort = set(_read_cohort(root, inputs["cohort"]))
        if expected_patient_ids is not None and cohort != set(
            _ids(expected_patient_ids)
        ):
            raise DerivationContextError("derivation expected cohort drift")
        import pyarrow.parquet as pq

        upper = {}
        with _snapshot(root, envelope["upper_bounds"]) as handle:
            for batch in pq.ParquetFile(handle).iter_batches(
                columns=["stay_id", "upper"]
            ):
                keys, values = batch.column(0).to_pylist(), batch.column(1).to_pylist()
                if len(keys) != len(set(keys)) or set(keys).intersection(upper):
                    raise DerivationContextError("derivation time bounds duplicated")
                upper.update(zip(keys, values))
        actual_events, actual_ids = set(), set()
        found_label = False
        for entry in manifest["files"]:
            if entry["module"] not in {
                MODULE,
                "sofa1_score",
                "sepsis_shared",
                "outcome",
            }:
                continue
            ref = {
                "file": entry["file"],
                "sha256": entry["parquet_sha256"],
                "bytes": entry["parquet_bytes"],
            }
            with _snapshot(root, ref, private=False) as handle:
                # Scan just identities except for the small derived-label file.
                if entry["module"] == MODULE:
                    if found_label:
                        raise DerivationContextError("duplicate published label module")
                    found_label = True
                    for batch in pq.ParquetFile(handle).iter_batches():
                        frame = batch.to_pandas()
                        current_ids = set(frame.stay_id)
                        if not current_ids.issubset(cohort):
                            raise DerivationContextError("published view cohort drift")
                        if current_ids.intersection(actual_ids):
                            raise DerivationContextError(
                                "published label grain mismatch"
                            )
                        actual_ids.update(current_ids)
                        actual_events.update(_label_events(frame))
                else:
                    for batch_ids in pq.ParquetFile(handle).iter_batches(
                        columns=["stay_id"]
                    ):
                        if not set(batch_ids.column(0).to_pylist()).issubset(cohort):
                            raise DerivationContextError("published view cohort drift")
        if not found_label:
            raise DerivationContextError("published label missing")
        seen: set[int] = set()
        expected_events = set()
        skipped = {"empty_si": 0, "no_positive_si": 0}
        for batch in inputs["batches"]:
            ids = set(_read_cohort(root, batch["cohort"]))
            if seen.intersection(ids) or not ids.issubset(cohort):
                raise DerivationContextError("derivation batch queue drift")
            seen.update(ids)
            # Object/numeric SI dtypes select different existing kernel branches.
            # Preserve the actual pandas argument types, not Arrow's inference.
            if len(ids) > MAX_CONTEXT_STAYS:
                raise DerivationContextError(
                    "derivation cohort shard exceeds read bound"
                )
            suspicion = _read_frame(root, batch["suspicion"], dependency=True).astype(
                batch["suspicion_dtypes"]
            )
            sofa = (
                _read_frame(root, batch["sofa"], dependency=True).astype(
                    batch["sofa_dtypes"]
                )
                if batch["sofa"]
                else None
            )
            id_col, time_col = batch["id_col"], batch["time_col"]
            for frame in (sofa, suspicion):
                if (
                    frame is not None
                    and not frame.empty
                    and not frame[id_col].isin(ids).all()
                ):
                    raise DerivationContextError("derivation dependency cohort drift")
            action = batch["action"]
            if action == "empty_si" and suspicion.empty and sofa is None:
                skipped[action] += 1
                continue
            if (
                action == "no_positive_si"
                and sofa is None
                and "susp_inf" in suspicion
                and not suspicion.susp_inf.eq(True).fillna(False).any()
            ):
                skipped[action] += 1
                continue
            if action != "kernel" or sofa is None:
                raise DerivationContextError("invalid derivation short-circuit receipt")
            expected = _kernel(
                sofa, suspicion, id_col=id_col, time_col=time_col
            ).rename(
                columns={id_col: "stay_id", time_col: "charttime", "sep3": CONCEPT}
            )
            # Same extraction owner; no second implementation of clinical P.
            from .extraction import _enforce_native_export_time_axis

            expected, _ = _enforce_native_export_time_axis(
                expected,
                module=MODULE,
                stay_time_upper_bounds=upper,
                database=manifest["database"],
            )
            expected_events.update(_label_events(expected))
        if seen != cohort:
            raise DerivationContextError("derivation batch queue incomplete")
        if expected_events != actual_events:
            raise DerivationContextError(
                "derived label replay mismatch "
                f"(expected_only={len(expected_events - actual_events)}, "
                f"actual_only={len(actual_events - expected_events)})"
            )
        return {
            "status": "verified",
            "scope": "derivation_replay",
            "concept": CONCEPT,
            "cohort_stays": len(cohort),
            "batches": len(inputs["batches"]),
            "positive_stays": len(actual_events),
            "short_circuit_batches": skipped,
            "membership_or_onset_differences": 0,
        }
    except DerivationContextError:
        raise
    except (KeyError, TypeError, ValueError, OSError, AttributeError) as exc:
        raise DerivationContextError(
            "complete derivation context invalid or missing"
        ) from exc
