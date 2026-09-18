from easyicu.webserver import cohort_review, sources as source_store
from tests.webserver.test_webserver_workspace_summary import _write_csv_export


def test_parallel_cohort_summaries_keep_bounded_isolated_cache(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor

    monkeypatch.setattr(source_store, "_CONFIG_DIR", tmp_path / "cfg")
    monkeypatch.setattr(source_store, "_CONFIG_PATH", tmp_path / "cfg" / "sources.json")
    monkeypatch.setattr(source_store, "_autodiscovered_paths", lambda: [])
    monkeypatch.setattr(cohort_review, "_SUMMARY_CACHE", {})
    paths = []
    for index in range(12):
        path = _write_csv_export(tmp_path / f"export{index}", database="miiv")
        source_store.register_source(
            str(path), label=f"Export {index}", active=index == 0
        )
        paths.append(path)
    with ThreadPoolExecutor(max_workers=12) as workers:
        results = list(
            workers.map(
                lambda path: cohort_review.cohort_review_summary(
                    {"source_path": str(path)}
                ),
                paths,
            )
        )
    assert all(result["summary"]["cohort_size"] == 3 for result in results)
    assert len(cohort_review._SUMMARY_CACHE) <= cohort_review._SUMMARY_CACHE_MAX
    last = cohort_review.cohort_review_summary({"source_path": str(paths[-1])})
    last["summary"]["cohort_size"] = 999
    assert (
        cohort_review.cohort_review_summary({"source_path": str(paths[-1])})["summary"][
            "cohort_size"
        ]
        == 3
    )
