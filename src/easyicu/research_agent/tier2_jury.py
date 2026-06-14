"""[Layer 5: Evaluation And Submission Scaffold] Tier-2 jury orchestration.

The jury is off by default. ``--enable-real-judges`` and
``EASYICU_ENABLE_REAL_JUDGES=1`` are both required before real API clients are
used. Without that explicit opt-in, ``MockJudge`` is used for all positions so
reviewers and CI can verify the pipeline shape without producing publishable
Tier-2 scores.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .tier2_rubric import JuryRubric

REAL_JUDGE_ENV_FLAG = "EASYICU_ENABLE_REAL_JUDGES"


@dataclass(frozen=True)
class JudgeIdentity:
    judge_id: str
    provider: str
    snapshot: str
    family: str


@dataclass
class JudgeScore:
    """One judge's score for one run on one rubric dimension."""

    run_id: str
    judge_id: str
    dimension_id: str
    score: int
    rationale: str
    prompt_sha256: str

    def __post_init__(self) -> None:
        if self.score < 0 or self.score > 3:
            raise ValueError(f"Tier-2 score must be in 0..3, got {self.score}")


@dataclass
class JudgeClient(ABC):
    identity: JudgeIdentity

    @abstractmethod
    def score_run(
        self,
        *,
        run_artifact_bundle: Dict[str, str],
        rubric: JuryRubric,
        prompt_text: str,
        prompt_hash: str,
    ) -> List[JudgeScore]:
        ...


class MockJudge(JudgeClient):
    """Deterministic scorer for tests and reviewer smoke runs.

    The default score is 2 for every dimension. Tests can set
    ``EASYICU_MOCK_JUDGE_OVERRIDES`` to a JSON object using keys such as
    ``"run-a|evidence_binding"`` or
    ``"mock_judge_1|run-a|evidence_binding"`` to force deterministic
    disagreement.
    """

    def score_run(
        self,
        *,
        run_artifact_bundle: Dict[str, str],
        rubric: JuryRubric,
        prompt_text: str,
        prompt_hash: str,
    ) -> List[JudgeScore]:
        run_id = _run_id_from_bundle(run_artifact_bundle)
        overrides = _load_mock_overrides()
        scores: List[JudgeScore] = []
        for dimension in rubric.dimensions:
            score = _mock_override_score(
                overrides=overrides,
                judge_id=self.identity.judge_id,
                run_id=run_id,
                dimension_id=dimension.dimension_id,
                default=2,
            )
            scores.append(
                JudgeScore(
                    run_id=run_id,
                    judge_id=self.identity.judge_id,
                    dimension_id=dimension.dimension_id,
                    score=score,
                    rationale="MockJudge deterministic scaffold score; not a publishable Tier-2 judgment.",
                    prompt_sha256=prompt_hash,
                )
            )
        return scores


class OpenAIJudge(JudgeClient):
    """Real-API judge via an OpenAI-compatible endpoint.

    This class does not import the OpenAI SDK or make network calls at import
    time. It fails closed unless ``EASYICU_ENABLE_REAL_JUDGES=1`` is set.
    """

    def __init__(
        self,
        identity: JudgeIdentity,
        *,
        model: str,
        api_key_env: str,
        base_url_env: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> None:
        super().__init__(identity=identity)
        self.model = model
        self.api_key_env = api_key_env
        self.base_url_env = base_url_env
        self.api_key = api_key
        self.base_url = base_url

    def score_run(
        self,
        *,
        run_artifact_bundle: Dict[str, str],
        rubric: JuryRubric,
        prompt_text: str,
        prompt_hash: str,
    ) -> List[JudgeScore]:
        if os.environ.get(REAL_JUDGE_ENV_FLAG) != "1":
            raise RuntimeError(
                f"real Tier-2 judges require {REAL_JUDGE_ENV_FLAG}=1; "
                "mock judges are used by default"
            )
        api_key = self.api_key or os.environ.get(self.api_key_env)
        if not api_key:
            raise RuntimeError(
                f"missing API key for {self.identity.judge_id}: set {self.api_key_env}"
            )
        base_url = self.base_url
        if base_url is None and self.base_url_env:
            base_url = os.environ.get(self.base_url_env)

        from .llm import LLMMessage, OpenAIClient

        client = OpenAIClient(
            model=self.model,
            api_key=api_key,
            base_url=base_url,
            max_retries=2,
        )
        response = client.complete(
            [
                LLMMessage(
                    role="system",
                    content=(
                        "You are an independent Tier-2 process-quality judge. "
                        "Return JSON only."
                    ),
                ),
                LLMMessage(role="user", content=prompt_text),
            ],
            max_tokens=1600,
            temperature=0.0,
        )
        parsed = _parse_real_judge_response(response)
        run_id = _run_id_from_bundle(run_artifact_bundle)
        expected_dimensions = {dimension.dimension_id for dimension in rubric.dimensions}
        scores: List[JudgeScore] = []
        for item in parsed:
            dimension_id = str(item.get("dimension_id", ""))
            if dimension_id not in expected_dimensions:
                raise ValueError(f"real judge returned unknown dimension {dimension_id!r}")
            scores.append(
                JudgeScore(
                    run_id=run_id,
                    judge_id=self.identity.judge_id,
                    dimension_id=dimension_id,
                    score=int(item["score"]),
                    rationale=str(item.get("rationale", "")).strip(),
                    prompt_sha256=prompt_hash,
                )
            )
        if {score.dimension_id for score in scores} != expected_dimensions:
            raise ValueError("real judge response did not score every rubric dimension")
        return scores


@dataclass
class JuryReport:
    rubric_version: str
    judges: List[JudgeIdentity]
    scores: List[JudgeScore]
    inter_judge_alpha: Dict[str, float]
    flagged_dimensions: List[str]
    retired_dimensions: List[str]
    run_order: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rubric_version": self.rubric_version,
            "judges": [asdict(judge) for judge in self.judges],
            "scores": [asdict(score) for score in self.scores],
            "inter_judge_alpha": self.inter_judge_alpha,
            "flagged_dimensions": self.flagged_dimensions,
            "retired_dimensions": self.retired_dimensions,
            "run_order": list(self.run_order),
        }


class JuryRunner:
    def __init__(self, judges: Sequence[JudgeClient], rubric: JuryRubric, seed: int):
        if len(judges) < 3:
            raise ValueError("Tier-2 jury requires at least 3 judges")
        non_mock = [judge for judge in judges if judge.identity.family != "mock"]
        if non_mock:
            families = [judge.identity.family for judge in non_mock]
            if len(non_mock) < 3 or len(set(families)) != len(non_mock):
                raise ValueError(
                    "Tier-2 jury requires real judges from disjoint model families"
                )
        self.judges = list(judges)
        self.rubric = rubric
        self._rng = random.Random(seed)

    def _build_prompt(
        self,
        run_artifact_bundle: Dict[str, str],
        judge_position: int,
    ) -> Tuple[str, str]:
        """Return byte-identical prompt text and SHA-256 for all judges."""

        del judge_position  # Position is intentionally not leaked to the judge.
        text = _render_prompt(run_artifact_bundle, self.rubric)
        return text, hashlib.sha256(text.encode("utf-8")).hexdigest()

    def score_runs(self, runs: List[Dict[str, str]]) -> JuryReport:
        indexed_runs = list(runs)
        self._rng.shuffle(indexed_runs)
        scores: List[JudgeScore] = []
        run_order = [_run_id_from_bundle(bundle) for bundle in indexed_runs]
        for bundle in indexed_runs:
            prompt_text, prompt_hash = self._build_prompt(bundle, judge_position=0)
            for position, judge in enumerate(self.judges):
                scores.extend(
                    judge.score_run(
                        run_artifact_bundle=bundle,
                        rubric=self.rubric,
                        prompt_text=prompt_text,
                        prompt_hash=prompt_hash,
                    )
                )
        alpha = _alpha_by_dimension(scores, self.judges, self.rubric, run_order)
        flagged = [
            dimension_id
            for dimension_id, value in alpha.items()
            if value < 0.4
        ]
        retired = [
            dimension_id
            for dimension_id, value in alpha.items()
            if value < 0.2
        ]
        return JuryReport(
            rubric_version=self.rubric.version,
            judges=[judge.identity for judge in self.judges],
            scores=scores,
            inter_judge_alpha=alpha,
            flagged_dimensions=flagged,
            retired_dimensions=retired,
            run_order=run_order,
        )


def default_mock_judges() -> List[MockJudge]:
    return [
        MockJudge(JudgeIdentity("mock_judge_1", "mock", "mock-v1", "mock")),
        MockJudge(JudgeIdentity("mock_judge_2", "mock", "mock-v1", "mock")),
        MockJudge(JudgeIdentity("mock_judge_3", "mock", "mock-v1", "mock")),
    ]


REAL_JUDGE_SPECS: Dict[str, Dict[str, str]] = {
    "claude_opus_4_7": {
        "provider": "anthropic",
        "family": "anthropic",
        "snapshot": "claude-opus-4-7",
        "api_key_env": "EASYICU_JUDGE_CLAUDE_OPUS_4_7_API_KEY",
        "base_url_env": "EASYICU_JUDGE_CLAUDE_OPUS_4_7_BASE_URL",
    },
    "gpt_5_5": {
        "provider": "openai",
        "family": "openai",
        "snapshot": "gpt-5.5",
        "api_key_env": "EASYICU_JUDGE_GPT_5_5_API_KEY",
        "base_url_env": "EASYICU_JUDGE_GPT_5_5_BASE_URL",
    },
    "gemini_2_5_pro": {
        "provider": "google",
        "family": "google",
        "snapshot": "gemini-2.5-pro",
        "api_key_env": "EASYICU_JUDGE_GEMINI_2_5_PRO_API_KEY",
        "base_url_env": "EASYICU_JUDGE_GEMINI_2_5_PRO_BASE_URL",
    },
    # OpenRouter-backed judges: three disjoint open-weight families served
    # through a single OPENROUTER_API_KEY. This makes the Tier-2 jury runnable
    # without three separate frontier-provider accounts; the disjoint-family
    # invariant still holds (meta / qwen / openai). Free-tier snapshots are
    # rate-limited and lower-capability than the frontier specs above — use
    # them for jury smoke runs and reproducibility demos, not as the canonical
    # submission jury. NOTE: OpenRouter rotates which slugs are free; if a run
    # 404s with "unavailable for free", refresh these slugs against
    # GET /models (filter id endswith ":free").
    "or_llama_70b": {
        "provider": "openrouter",
        "family": "meta",
        "snapshot": "meta-llama/llama-3.3-70b-instruct:free",
        "api_key_env": "OPENROUTER_API_KEY",
        "base_url_env": "OPENROUTER_BASE_URL",
    },
    "or_qwen3_next": {
        "provider": "openrouter",
        "family": "qwen",
        "snapshot": "qwen/qwen3-next-80b-a3b-instruct:free",
        "api_key_env": "OPENROUTER_API_KEY",
        "base_url_env": "OPENROUTER_BASE_URL",
    },
    "or_gpt_oss_120b": {
        "provider": "openrouter",
        "family": "openai",
        "snapshot": "openai/gpt-oss-120b:free",
        "api_key_env": "OPENROUTER_API_KEY",
        "base_url_env": "OPENROUTER_BASE_URL",
    },
}


def make_real_judges(judge_ids: Sequence[str]) -> List[OpenAIJudge]:
    judges: List[OpenAIJudge] = []
    for judge_id in judge_ids:
        try:
            spec = REAL_JUDGE_SPECS[judge_id]
        except KeyError as exc:
            known = ", ".join(sorted(REAL_JUDGE_SPECS))
            raise ValueError(f"unknown real judge {judge_id!r}; known: {known}") from exc
        identity = JudgeIdentity(
            judge_id=judge_id,
            provider=spec["provider"],
            snapshot=spec["snapshot"],
            family=spec["family"],
        )
        judges.append(
            OpenAIJudge(
                identity,
                model=spec["snapshot"],
                api_key_env=spec["api_key_env"],
                base_url_env=spec["base_url_env"],
            )
        )
    return judges


def krippendorff_alpha(scores_by_judge: List[List[Optional[int]]]) -> float:
    """Krippendorff's alpha for ordinal 0-3 scores.

    ``scores_by_judge`` is shaped as judges x units. Missing scores may be
    represented as ``None``.
    """

    if not scores_by_judge:
        return 1.0
    unit_count = max((len(row) for row in scores_by_judge), default=0)
    units: List[List[int]] = []
    for unit_idx in range(unit_count):
        values = [
            int(row[unit_idx])
            for row in scores_by_judge
            if unit_idx < len(row) and row[unit_idx] is not None
        ]
        if len(values) >= 2:
            units.append(values)
    if not units:
        return 1.0

    categories = sorted({value for unit in units for value in unit})
    frequencies = {category: 0 for category in categories}
    for unit in units:
        for value in unit:
            frequencies[value] += 1
    total = sum(frequencies.values())
    if len(categories) <= 1 or total <= 1:
        return 1.0

    def ordinal_delta(a: int, b: int) -> float:
        if a == b:
            return 0.0
        low, high = sorted((a, b))
        between = [
            category
            for category in categories
            if low <= category <= high
        ]
        metric = sum(frequencies[category] for category in between)
        metric -= (frequencies[low] + frequencies[high]) / 2.0
        return float(metric * metric)

    observed = 0.0
    for unit in units:
        m = len(unit)
        pair_sum = 0.0
        for a in unit:
            for b in unit:
                pair_sum += ordinal_delta(a, b)
        observed += pair_sum / (m - 1)
    observed /= total

    expected = 0.0
    for a in categories:
        for b in categories:
            expected += frequencies[a] * frequencies[b] * ordinal_delta(a, b)
    expected /= total * (total - 1)
    if expected == 0:
        return 1.0
    return 1.0 - (observed / expected)


def _alpha_by_dimension(
    scores: Sequence[JudgeScore],
    judges: Sequence[JudgeClient],
    rubric: JuryRubric,
    run_order: Sequence[str],
) -> Dict[str, float]:
    by_key = {
        (score.judge_id, score.dimension_id, score.run_id): score.score
        for score in scores
    }
    result: Dict[str, float] = {}
    for dimension in rubric.dimensions:
        rows: List[List[Optional[int]]] = []
        for judge in judges:
            rows.append(
                [
                    by_key.get((judge.identity.judge_id, dimension.dimension_id, run_id))
                    for run_id in run_order
                ]
            )
        result[dimension.dimension_id] = krippendorff_alpha(rows)
    return result


def _render_prompt(run_artifact_bundle: Dict[str, str], rubric: JuryRubric) -> str:
    run_id = _run_id_from_bundle(run_artifact_bundle)
    lines = [
        "EasyICU Tier-2 process-quality jury",
        "",
        "Status: advisory process audit only. Do not judge clinical validity, novelty, or causal correctness.",
        "Return JSON with key 'scores', one object per dimension: dimension_id, score, rationale.",
        "",
        f"Rubric version: {rubric.version}",
        "",
        "Rubric anchors:",
    ]
    for dimension in rubric.dimensions:
        lines.append(f"- {dimension.dimension_id} ({dimension.label})")
        for value in sorted(dimension.anchors):
            lines.append(f"  {value}: {dimension.anchors[value]}")
    lines.extend(["", f"Run id: {run_id}", "", "Artifact bundle:"])
    for name in sorted(k for k in run_artifact_bundle if not k.startswith("__")):
        lines.extend([f"--- {name} ---", run_artifact_bundle[name].strip(), ""])
    return "\n".join(lines).rstrip() + "\n"


def _run_id_from_bundle(bundle: Dict[str, str]) -> str:
    explicit = bundle.get("__run_id__")
    if explicit:
        return explicit.strip()
    for key in ("run_status.json", "manifest.json", "run_manifest.json"):
        raw = bundle.get(key)
        if not raw:
            continue
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            continue
        for field_name in ("run_id", "id", "name"):
            value = parsed.get(field_name)
            if isinstance(value, str) and value.strip():
                return value.strip()
    digest = hashlib.sha256()
    for name in sorted(bundle):
        if name.startswith("__"):
            continue
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bundle[name].encode("utf-8"))
        digest.update(b"\0")
    return "run_" + digest.hexdigest()[:12]


def _load_mock_overrides() -> Dict[str, Any]:
    raw = os.environ.get("EASYICU_MOCK_JUDGE_OVERRIDES")
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("EASYICU_MOCK_JUDGE_OVERRIDES must be valid JSON") from exc
    if not isinstance(parsed, dict):
        raise ValueError("EASYICU_MOCK_JUDGE_OVERRIDES must be a JSON object")
    return parsed


def _mock_override_score(
    *,
    overrides: Dict[str, Any],
    judge_id: str,
    run_id: str,
    dimension_id: str,
    default: int,
) -> int:
    candidate_keys = (
        f"{judge_id}|{run_id}|{dimension_id}",
        f"{run_id}|{dimension_id}",
        f"{run_id}.{dimension_id}",
        dimension_id,
    )
    value: Any = None
    for key in candidate_keys:
        if key in overrides:
            value = overrides[key]
            break
    if value is None:
        nested = overrides.get(judge_id)
        if isinstance(nested, dict):
            run_nested = nested.get(run_id)
            if isinstance(run_nested, dict) and dimension_id in run_nested:
                value = run_nested[dimension_id]
            elif dimension_id in nested:
                value = nested[dimension_id]
    if value is None:
        return default
    score = int(value)
    if score < 0 or score > 3:
        raise ValueError(f"mock override score must be in 0..3, got {score}")
    return score


def _extract_json_payload(text: str) -> Any:
    """Parse a judge response that may be wrapped in code fences or preceded
    by reasoning/preamble. Tries a direct parse first, then falls back to the
    first balanced ``{...}`` or ``[...]`` block. Reasoning/thinking models
    (e.g. nemotron-style judges) routinely emit prose before the JSON, so a
    strict whole-string ``json.loads`` would reject otherwise-valid scores."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.lower().startswith("json"):
            stripped = stripped[4:].strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    # Fall back: scan for the first balanced JSON object/array.
    for opener, closer in (("{", "}"), ("[", "]")):
        start = stripped.find(opener)
        if start == -1:
            continue
        depth = 0
        in_str = False
        escape = False
        for idx in range(start, len(stripped)):
            ch = stripped[idx]
            if in_str:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(stripped[start : idx + 1])
                    except json.JSONDecodeError:
                        break
    raise ValueError("real judge response did not contain parseable JSON")


def _parse_real_judge_response(response: str) -> List[Dict[str, Any]]:
    parsed = _extract_json_payload(response)
    if isinstance(parsed, dict):
        scores = parsed.get("scores")
    else:
        scores = parsed
    if not isinstance(scores, list):
        raise ValueError("real judge response must contain a list of scores")
    return [item for item in scores if isinstance(item, dict)]


__all__ = [
    "JudgeClient",
    "JudgeIdentity",
    "JudgeScore",
    "JuryReport",
    "JuryRunner",
    "MockJudge",
    "OpenAIJudge",
    "REAL_JUDGE_ENV_FLAG",
    "REAL_JUDGE_SPECS",
    "default_mock_judges",
    "krippendorff_alpha",
    "make_real_judges",
]
