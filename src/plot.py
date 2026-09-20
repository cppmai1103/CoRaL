"""PNG plots for score ranges, ranking, and test error summaries.

Called by src.train and src.test; all functions accept the current run's
metrics and an explicit output path. Plotting uses a headless backend.
"""

import math
from pathlib import Path

SCORE_RANGE_PLOT_NOTE = (
    "Test predictions clipped to [0,5]; min/max describe RATER predictions, not human scores.\n"
    "MAE color scale: 0–3 (possible range: 0–5). n = documents; * = test n < 10; gray = no data.\n"
    "Spearman appears only in ALL; NA = fewer than 2 examples or constant scores.\n"
    "All row pools languages; ALL column pools score ranges. Metrics use pooled examples, not language macro-averages."
)


def draw_score_range_panels(fig, axes, rows: list[dict]) -> None:
    """Shared layout for per-dimension figures and the combined overview."""
    languages = sorted({r["language"] for r in rows} - {"All"})
    if any(r["language"] == "All" for r in rows):
        languages.append("All")
    ranges = ["[0,1)", "[1,2)", "[2,3)", "[3,4)", "[4,5]", "ALL"]
    import matplotlib.pyplot as plt

    lookup = {(r["language"], r["score_range"]): r for r in rows}
    for ax, metric, title, cmap_name, limit in zip(
        axes, ["train_percent", "mae"],
        ["Training score distribution", "Test MAE by human score"],
        ["Blues", "YlOrRd"], [100, 3],
    ):
        values = [[lookup[lang, b][metric] if lookup[lang, b][metric] is not None else float("nan")
                   for b in ranges] for lang in languages]
        cmap = plt.get_cmap(cmap_name).copy()
        cmap.set_bad("#e6e6e6")
        im = ax.imshow(values, cmap=cmap, vmin=0, vmax=limit, aspect="auto")
        ax.set_xticks(range(len(ranges)), ranges)
        ax.set_yticks(range(len(languages)), languages)
        ax.set_xlabel("Human score range")
        ax.set_title(title, weight="bold")
        for i, lang in enumerate(languages):
            for j, b in enumerate(ranges):
                row = lookup[lang, b]
                if metric == "train_percent":
                    share = f"{row[metric]:.1f}%" if row[metric] is not None else "NA"
                    label = f"{share}\nn={row['train_n']}"
                elif row["test_n"]:
                    rho = f"{row['spearman']:.2f}" if row["spearman"] is not None else "NA"
                    flag = "*" if row["test_n"] < 10 else ""
                    rank_line = f"\nSpearman: {rho}" if b == "ALL" else ""
                    label = (f"MAE: {row['mae']:.2f}{rank_line}\nn={row['test_n']}{flag}\n"
                             f"min: {row['prediction_min']:.2f}; max: {row['prediction_max']:.2f}")
                else:
                    label = "No test data\nn=0\nmin: NA; max: NA"
                color = "white" if math.isfinite(values[i][j]) and values[i][j] / limit > .58 else "#17202a"
                ax.text(j, i, label, ha="center", va="center", fontsize=8, color=color)
        ax.set_xticks([i - .5 for i in range(len(ranges) + 1)], minor=True)
        ax.set_yticks([i - .5 for i in range(len(languages) + 1)], minor=True)
        ax.grid(which="minor", color="white", linewidth=2)
        ax.axvline(len(ranges) - 1.5, color="#17202a", linewidth=2)
        if "All" in languages:
            ax.axhline(len(languages) - 1.5, color="#17202a", linewidth=2)
        ax.tick_params(which="minor", bottom=False, left=False)
        fig.colorbar(im, ax=ax, fraction=.04, pad=.025,
                     label="Training share (%)" if metric == "train_percent" else "MAE (points)")


def plot_score_ranges(path: Path, dimension: str, rows: list[dict]) -> None:
    """Plot current-run training distribution and test errors by human score range."""
    if not rows:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_languages = len({r["language"] for r in rows})
    fig, axes = plt.subplots(1, 2, figsize=(24, max(5, .8 * n_languages + 3)))
    draw_score_range_panels(fig, axes, rows)
    fig.suptitle(dimension.replace("_", " ").title(), fontsize=19, weight="bold")
    fig.subplots_adjust(top=.85, bottom=.25, left=.07, right=.97, wspace=.30)
    fig.text(.07, .04, SCORE_RANGE_PLOT_NOTE, fontsize=9)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"  Saved score-range plot: {path}")


def plot_score_range_overview(path: Path, rows_by_dimension: dict[str, list[dict]]) -> None:
    """Combine current-run score-range plots, retaining the same metrics and layout."""
    available = {dim: rows for dim, rows in rows_by_dimension.items() if rows}
    if not available:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_languages = max(len({r["language"] for r in rows}) for rows in available.values())
    height = max(4, .7 * n_languages + 1.5) * len(available) + 2
    fig, axes = plt.subplots(len(available), 2, figsize=(24, height), squeeze=False)
    for (dimension, rows), axis_pair in zip(available.items(), axes):
        draw_score_range_panels(fig, axis_pair, rows)
        title = dimension.replace("_", " ").title()
        axis_pair[0].set_title(f"{title} — training score distribution", weight="bold")
        axis_pair[1].set_title(f"{title} — test MAE by human score", weight="bold")
    fig.suptitle("Score distributions and test performance by range", fontsize=20, weight="bold", y=.99)
    fig.subplots_adjust(left=.07, right=.97, top=1-.65/height, bottom=1.4/height, hspace=.45, wspace=.30)
    fig.text(.07, .2/height, SCORE_RANGE_PLOT_NOTE, fontsize=10)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"  Saved score-range overview: {path}")


def plot_test_ranking(path: Path, cells_by_dimension: dict[str, dict[str, dict]]) -> None:
    """Plot full-scale test Spearman for the evaluated dimensions and languages."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dimensions = list(cells_by_dimension)
    languages = sorted({lang for cells in cells_by_dimension.values() for lang in cells})
    if not languages:
        return
    values = []
    for lang in languages:
        values.append([
            cells_by_dimension[dim].get(lang, {}).get("spearman")
            if cells_by_dimension[dim].get(lang, {}).get("spearman") is not None else float("nan")
            for dim in dimensions
        ])
    fig, ax = plt.subplots(figsize=(max(6, 2 * len(dimensions) + 2), max(4, .65 * len(languages) + 2)))
    cmap = plt.get_cmap("RdBu").copy()
    cmap.set_bad("#e6e6e6")
    im = ax.imshow(values, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(dimensions)), [d.replace("_", "\n").title() for d in dimensions])
    ax.set_yticks(range(len(languages)), languages)
    ax.set_title("Test ranking agreement")
    for i, lang in enumerate(languages):
        for j, dim in enumerate(dimensions):
            cell = cells_by_dimension[dim].get(lang, {})
            rho = cell.get("spearman")
            label = f"{rho:.3f}" if rho is not None else "NA"
            ax.text(j, i, f"{label}\nn={cell.get('n', 0)}", ha="center", va="center",
                    color="white" if rho is not None and abs(rho) > .6 else "#17202a")
    fig.colorbar(im, ax=ax, label="Spearman correlation (higher is better)")
    fig.subplots_adjust(left=.17, bottom=.25, top=.88)
    fig.text(.10, .04, "All test scores per language; predictions clipped to [0,5].\n"
             "+1: identical ranking; 0: no rank correlation; -1: reversed ranking.\n"
             "NA: fewer than two examples or constant scores. Not an average of score-range correlations.", fontsize=9)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"  Saved test ranking plot: {path}")


def pool_error_metrics(cells: list[dict]) -> dict:
    """Pool document-dimension observations, including squared errors for RMSE."""
    cells = [cell for cell in cells if cell.get("n", 0) > 0]
    n = sum(cell["n"] for cell in cells)
    if not n:
        return {"n": 0, "mae": None, "rmse": None, "within_1_0": None}
    return {
        "n": n,
        "mae": sum(cell["n"] * cell["mae"] for cell in cells) / n,
        "rmse": math.sqrt(sum(cell["n"] * cell["rmse"] ** 2 for cell in cells) / n),
        "within_1_0": sum(cell["n"] * cell["within_1_0"] for cell in cells) / n,
    }


def plot_test_metrics(path: Path, cells_by_dimension: dict[str, dict[str, dict]], provenance: str = "Current-run test predictions.") -> None:
    """Plot MAE, RMSE and within-1 accuracy, with pooled language/dimension totals."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dimensions = list(cells_by_dimension)
    languages = sorted({lang for cells in cells_by_dimension.values() for lang in cells})
    if not dimensions or not languages:
        return
    # Construct all totals from original cells to avoid double-counting totals.
    lookup = {(lang, dim): cells_by_dimension[dim].get(lang, {})
              for lang in languages for dim in dimensions}
    for lang in languages:
        lookup[lang, "ALL"] = pool_error_metrics([lookup[lang, dim] for dim in dimensions])
    for dim in dimensions:
        lookup["All", dim] = pool_error_metrics([lookup[lang, dim] for lang in languages])
    lookup["All", "ALL"] = pool_error_metrics([lookup[lang, dim] for lang in languages for dim in dimensions])
    row_labels, columns = languages + ["All"], dimensions + ["ALL"]
    fig, axes = plt.subplots(1, 3, figsize=(max(17, 4.6 * len(columns)), max(6, .72 * len(row_labels) + 2.5)))
    specs = [("mae", "MAE — lower is better", "YlOrRd", 1, 5, "Rubric points"),
             ("rmse", "RMSE — lower is better", "YlOrRd", 1, 5, "Rubric points"),
             ("within_1_0", "Within ±1 point — higher is better", "Blues", 100, 100, "Test predictions (%)")]
    for ax, (metric, title, cmap_name, scale, upper, unit) in zip(axes, specs):
        values = [[lookup[lang, dim].get(metric) * scale if lookup[lang, dim].get(metric) is not None else float("nan")
                   for dim in columns] for lang in row_labels]
        cmap = plt.get_cmap(cmap_name).copy()
        cmap.set_bad("#e6e6e6")
        im = ax.imshow(values, cmap=cmap, vmin=0, vmax=upper, aspect="auto")
        ax.set_title(title, weight="bold", pad=12)
        ax.set_xticks(range(len(columns)), [dim.replace("_", "\n").title() if dim != "ALL" else dim for dim in columns])
        ax.set_yticks(range(len(row_labels)), row_labels)
        for i, lang in enumerate(row_labels):
            for j, dim in enumerate(columns):
                value = values[i][j]
                formatted = (f"{value:.1f}%" if metric == "within_1_0" else f"{value:.3f}") if math.isfinite(value) else "NA"
                ax.text(j, i, f"{formatted}\nn={lookup[lang, dim].get('n', 0)}", ha="center", va="center", fontsize=9,
                        color="white" if math.isfinite(value) and value / upper > .58 else "#17202a")
        ax.set_xticks([i - .5 for i in range(len(columns) + 1)], minor=True)
        ax.set_yticks([i - .5 for i in range(len(row_labels) + 1)], minor=True)
        ax.grid(which="minor", color="white", linewidth=2)
        ax.tick_params(which="minor", bottom=False, left=False)
        ax.axhline(len(row_labels) - 1.5, color="#17202a", linewidth=1.5)
        ax.axvline(len(columns) - 1.5, color="#17202a", linewidth=1.5)
        fig.colorbar(im, ax=ax, fraction=.04, pad=.02, label=unit)
    fig.suptitle("Test performance by language and dimension", fontsize=19, weight="bold")
    fig.subplots_adjust(left=.04, right=.98, bottom=.25, top=.87, wspace=.25)
    fig.text(.04, .045, provenance + " Predictions are clipped to [0,5]. Within ±1 includes absolute error exactly 1.\n"
             "All row: pooled languages. ALL column: pooled dimensions. Bottom-right: all document-dimension predictions, not unique documents.\n"
             "Totals weight each prediction equally; they are not language macro-averages. Pooled RMSE is the square root of pooled squared error.\n"
             "n = evaluated document-dimension predictions; gray = no data.", fontsize=10)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"  Saved test metrics plot: {path}")
