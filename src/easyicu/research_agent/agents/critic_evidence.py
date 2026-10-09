"""Which manuscript sentences the critic finds without evidence.

Owner
-----
The critic flags a result-like sentence that carries no evidence token: a
number beside a claim word, or an unquantified result word such as
"consistent" or "robust".  This module decides what counts as a sentence's
evidence (an evidence placeholder or link inside it, or one the host placed
directly after its full stop) and which cited sentences are literature
rather than results of this study.  ``agents.roles.CriticAgent`` asks it.
"""

from __future__ import annotations

import re
from typing import List, Sequence


_TRAILING_EVIDENCE_RE = re.compile(
    r"(?P<stop>[.!?。！？])\s+"
    r"(?P<evidence>\{evidence:[^}]+\}|\[[^\[\]]*\]\(\s*evidence/[^)]+\))"
)


def sentences_missing_evidence_tokens(
    scaffold: str,
    *,
    available_evidence_ids: Sequence[str] = (),
) -> List[str]:
    unsupported: List[str] = []
    text = re.sub(r"```.*?```", " ", scaffold, flags=re.S)
    available_evidence = {
        str(evidence_id).strip().lower()
        for evidence_id in available_evidence_ids
        if str(evidence_id).strip()
    }
    bound_claim_footnotes = {
        match.group("claim_id").lower()
        for raw_line in text.splitlines()
        if (
            match := re.match(
                r"^\s*\[\^(?P<claim_id>claim_[^\]]+)\]:.*\bevidence=(?P<evidence>\S+)",
                raw_line,
                flags=re.I,
            )
        )
        and match.group("evidence").strip().rstrip(";,. ").lower() in available_evidence
    }
    cleaned_lines: List[str] = []
    section_label_re = re.compile(
        r"^\*\*(?:background|methods?|results?|conclusions?|discussion|limitations?)\s*:\*\*\s*",
        flags=re.I,
    )
    metadata_line_re = re.compile(
        r"^\s*(?:#{1,6}\s*)?(?:\*\*)?"
        r"(?:keywords?|key words|data\s+(?:and\s+code\s+)?availability|"
        r"code\s+availability|funding|conflicts?\s+of\s+interest|"
        r"acknowledg(?:e)?ments?|ethics\s+approval)"
        r"\s*(?:\*\*)?\s*[:：]?",
        flags=re.I,
    )
    in_metadata_section = False
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            cleaned_lines.append(" ")
            continue
        if re.match(r"^#{1,6}\s+", stripped):
            in_metadata_section = bool(metadata_line_re.match(stripped))
            continue
        if in_metadata_section or metadata_line_re.match(stripped):
            continue
        # Skip footnote/provenance DEFINITION lines (``[^claim_1]: value=...;
        # step=...; evidence=<name>``). These are auto-appended by the numeric
        # binder as machine provenance, not author-written result sentences:
        # they carry numbers + claimy words (auroc/brier/death) but reference
        # evidence via a plaintext ``evidence=<step>`` token (no ``](evidence/)``
        # link) when a claim binds to a step-level virtual evidence, so the
        # support check can mis-flag the whole footnote block as unsupported
        # prose. The block proves the claims are bound and is not prose to audit.
        if re.match(r"^\[\^[^\]]+\]:", stripped):
            continue
        match = section_label_re.match(stripped)
        if match:
            stripped = stripped[match.end() :].strip()
            if not stripped:
                continue
        cleaned_lines.append(stripped)
    text = " ".join(cleaned_lines)
    # A host-rendered sentence carries its evidence after the full stop
    # ("... (adjusted odds ratio). {evidence:<id>}", a link once bound). That
    # provenance belongs to the sentence it directly follows.
    text = _TRAILING_EVIDENCE_RE.sub(
        lambda match: f" {match.group('evidence')}{match.group('stop')}", text
    )
    for raw_sentence in re.split(r"(?<=[.!?。！？])\s+", text):
        sentence = raw_sentence.strip()
        if not sentence:
            continue
        if "{evidence:" in sentence or re.search(
            r"\]\(\s*evidence/[^)]+\)", sentence, flags=re.I
        ):
            continue
        claim_refs = {
            claim_ref.lower()
            for claim_ref in re.findall(r"\[\^(claim_[^\]]+)\]", sentence, flags=re.I)
        }
        if claim_refs and claim_refs.issubset(bound_claim_footnotes):
            continue
        if re.search(
            r"(?:\[evidence missing:\s*[^\]]+\]|<!--\s*evidence missing:\s*[^>]+-->)",
            sentence,
            flags=re.I,
        ):
            unsupported.append(sentence)
            continue
        # Citation keys commonly contain publication years.  Their digits are
        # literature provenance, not quantitative manuscript results; citation
        # validity is enforced independently by the literature audit.
        has_literature_citation = bool(
            re.search(r"\[[^\[\]]*@[A-Za-z0-9_.:-]+[^\[\]]*\]", sentence)
        )
        prose_for_result_detection = re.sub(r"\[@[^\]]+\]", " ", sentence)
        # Versioned scientific names and database releases (for example
        # SOFA-2, Sepsis-3, and MIMIC-IV) are identifiers, not quantitative
        # results.  Preserve ordinary values such as ``2-fold`` or ``10%``.
        prose_for_result_detection = re.sub(
            r"\b(?:[A-Za-z][A-Za-z0-9]*-\d+[A-Za-z0-9-]*|[A-Z]{2,}-[IVXLCDM]+)\b",
            " ",
            prose_for_result_detection,
        )
        has_number = bool(re.search(r"\d", prose_for_result_detection))
        has_claimy_word = bool(
            re.search(
                r"\b(cohort|stays|patients|mortality|death|auroc|auc|hazard|odds|risk|cluster|survival|ci|p=|calibration|brier|discrimination|performance|robust(?:ness)?|overfitting|miscalibration|missingness|generalisability|generalizability)\b",
                prose_for_result_detection,
                flags=re.I,
            )
        )
        is_literature_attribution = bool(
            re.search(
                r"\b(?:prior|previous|published|recent)\s+"
                r"(?:stud(?:y|ies)|work|reports?|evaluations?|literature)\b",
                prose_for_result_detection,
                flags=re.I,
            )
        )
        # A cited statement with no number that does not speak about this
        # study is literature background (for example a guideline definition),
        # not an unquantified result of the analysis.
        refers_to_this_study = bool(
            re.search(
                r"\b(?:our|we|this\s+(?:study|analysis|cohort|work)|"
                r"the\s+present\s+(?:study|analysis)|"
                r"(?:these|the)\s+(?:results|findings|estimates))\b",
                prose_for_result_detection,
                flags=re.I,
            )
        )
        is_literature_background = (
            has_literature_citation and not has_number and not refers_to_this_study
        )
        has_unquantified_result_claim = bool(
            re.search(
                r"\b(performance|robust(?:ness)?|consistent|overfitting|miscalibration|missingness|generalisability|generalizability)\b",
                prose_for_result_detection,
                flags=re.I,
            )
        ) and not (
            (has_literature_citation and is_literature_attribution)
            or is_literature_background
        )
        if (has_number and has_claimy_word) or has_unquantified_result_claim:
            unsupported.append(sentence)
    return unsupported


__all__ = ["sentences_missing_evidence_tokens"]
