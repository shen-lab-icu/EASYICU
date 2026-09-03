from __future__ import annotations

from easyicu.research_agent.repairs.source import _deterministic_runner_repair


_ERROR = "ValueError: Clustering visualization panel has no traceable plotted values"

_SCRIPT = '''
fig, axes = plt.subplots(1, 3)
axes[2].bar([0, 1], [10, 20])
source_data_files = []
figure_panels = []
for panel_index, panel_ax in enumerate(fig.axes):
    panel_rows = []
    for artist_index, collection in enumerate(panel_ax.collections):
        offsets = np.asarray(collection.get_offsets())
        for source_row_index, offset in enumerate(offsets):
            panel_rows.append({
                "source_row_index": int(source_row_index),
                "artist_index": int(artist_index),
                "x_coordinate": float(offset[0]),
                "y_coordinate": float(offset[1]),
            })
    for artist_index, line in enumerate(panel_ax.lines):
        x_values = np.asarray(line.get_xdata())
        y_values = np.asarray(line.get_ydata())
        for source_row_index, (x_value, y_value) in enumerate(zip(x_values, y_values)):
            panel_rows.append({
                "source_row_index": int(source_row_index),
                "artist_index": int(artist_index),
                "x_coordinate": float(x_value),
                "y_coordinate": float(y_value),
            })
    if not panel_rows:
        raise ValueError("Clustering visualization panel has no traceable plotted values")
    panel_source_name = f"panel_{panel_index}.csv"
'''.lstrip()


def test_runner_repair_does_not_relabel_matplotlib_artists_as_source_data() -> None:
    repair = _deterministic_runner_repair(code=_SCRIPT, run_log=_ERROR)

    assert repair is None


def test_runner_repair_requires_exact_failure_and_complete_existing_extractors() -> None:
    assert (
        _deterministic_runner_repair(
            code=_SCRIPT,
            run_log="ValueError: unrelated figure failure",
        )
        is None
    )
    assert (
        _deterministic_runner_repair(
            code=_SCRIPT.replace("panel_ax.lines", "panel_ax.artists"),
            run_log=_ERROR,
        )
        is None
    )
    assert (
        _deterministic_runner_repair(
            code=_SCRIPT.replace("axes[2].bar([0, 1], [10, 20])\n", ""),
            run_log=_ERROR,
        )
        is None
    )


def test_runner_repair_refuses_ambiguous_or_already_handled_guards() -> None:
    duplicated = _SCRIPT + "\n" + _SCRIPT.replace("panel_index", "other_index")
    assert _deterministic_runner_repair(code=duplicated, run_log=_ERROR) is None

    assert _deterministic_runner_repair(code=_SCRIPT, run_log=_ERROR) is None
