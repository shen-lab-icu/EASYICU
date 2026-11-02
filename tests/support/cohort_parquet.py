"""Shared synthetic cohort export mutation and summary helpers."""
from easyicu.webserver import cohort_review


def _run(monkeypatch, path, desc, selected=None):
    monkeypatch.setattr(cohort_review, "_SUMMARY_CACHE", {})
    monkeypatch.setattr(
        cohort_review,
        "_resolve_registered_source",
        lambda body: ({"id": "synthetic", "path": str(path)}, desc),
    )
    return cohort_review.cohort_review_summary(
        {} if selected is None else {"selected_features": selected}
    )

def _replace_module(root, desc, module, frame):
    item = next(entry for entry in desc["files"] if entry["module"] == module)
    frame.to_parquet(root / item["file"], index=False)
    item["columns"] = list(frame.columns)
    item["rows"] = len(frame)
    item["concept_ids"] = [
        column for column in frame if column not in {"stay_id", "charttime"}
    ]
