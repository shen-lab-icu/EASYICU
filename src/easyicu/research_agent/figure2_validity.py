"""V2-only objective-validity replay over a verified evidence snapshot.

The historical scorecard module is part of the byte-frozen Figure 2 v1 scorer
bundle.  Paper v2 therefore reuses its existing objective validity logic without
editing that module: current, digest-verified evidence documents and checkpoint-
selected step summaries are projected into an isolated temporary run view, then
the frozen scorer is invoked against that view.  No mutable run-root file is read
after the scoring-input authority has been established.
"""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from .evaluation_scorecard import DimensionScore, score_run_from_dir
from .figure2_scoring_inputs import LoadedFigure2ScoringInputs
from .icu_agent_bench import ICUAgentBenchTask


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def score_verified_result_validity(
    task: ICUAgentBenchTask,
    loaded: LoadedFigure2ScoringInputs,
) -> DimensionScore:
    """Replay the existing validity teeth using only authority-bound inputs."""

    with TemporaryDirectory(prefix="easyicu-figure2-validity-") as tmp:
        root = Path(tmp)
        materialized: set[Path] = set()
        for document in loaded.review_documents:
            target = root / document.relative_path
            if target in materialized:
                raise ValueError("verified review corpus contains a duplicate path")
            materialized.add(target)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(document.text.encode("utf-8"))

        # The legacy validity detector reads successful step summaries from this
        # fallback layout when no runtime manifest is present.  These summaries
        # came from the digest-bound checkpoint selected by the input loader.
        for index, summary in enumerate(loaded.current_step_summaries, start=1):
            target = root / "steps" / f"{index:04d}" / "outputs" / "step_summary.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(_canonical_json_bytes(summary))

        # Only the gate values needed by result-validity scoring are projected.
        # Other legacy dimensions are discarded after replay.
        (root / "run_status.json").write_bytes(
            _canonical_json_bytes(
                {
                    "run_id": loaded.authority.run_id,
                    "gates": loaded.gates,
                }
            )
        )
        return score_run_from_dir(
            task,
            root,
            exposure_concept=loaded.authority.exposure_concept,
            outcome_concept=loaded.authority.outcome_concept,
            locked_reference_frozen=False,
            run_id=loaded.authority.run_id,
        ).result_validity


__all__ = ["score_verified_result_validity"]
