"""Inference, evaluation metrics, and reports for the SEA-Rater heads.

Training calls evaluate_and_report after restoring the best validation
checkpoint. This module also supplies the validation metrics used for early
stopping. It does not train or select checkpoints on the test split.
"""

import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn

if __package__:
    from .plot import plot_score_ranges, plot_test_metrics, plot_test_ranking
else:
    from plot import plot_score_ranges, plot_test_metrics, plot_test_ranking

def rankdata(values: list[float]) -> list[float]:
    n = len(values)
    order = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg_rank = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


def spearman(x: list[float], y: list[float]) -> float | None:
    if len(set(x)) <= 1 or len(set(y)) <= 1:
        return None
    rx, ry = rankdata(x), rankdata(y)
    n = len(x)
    mx, my = sum(rx) / n, sum(ry) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = sum((a - mx) ** 2 for a in rx)
    vy = sum((b - my) ** 2 for b in ry)
    if vx == 0 or vy == 0:
        return None
    return cov / math.sqrt(vx * vy)


def compute_cell_metrics(human: list[float], raw_pred: list[float]) -> dict:
    clipped = [min(5.0, max(0.0, p)) for p in raw_pred]
    n = len(human)
    errors = [c - h for c, h in zip(clipped, human)]
    abs_errors = [abs(e) for e in errors]
    oob = sum(1 for p in raw_pred if p < 0.0 or p > 5.0) / n
    return {
        "n": n,
        "mae": statistics.mean(abs_errors),
        "rmse": math.sqrt(statistics.mean(e * e for e in errors)),
        "bias": statistics.mean(errors),
        "within_0_5": sum(1 for e in abs_errors if e <= 0.5) / n,
        "within_1_0": sum(1 for e in abs_errors if e <= 1.0) / n,
        "spearman": spearman(human, clipped),
        "out_of_range_rate": oob,
    }


def macro_mae(cell_metrics_by_language: dict[str, dict]) -> float:
    return statistics.mean(m["mae"] for m in cell_metrics_by_language.values())


@torch.no_grad()
def predict(head: nn.Module, records: list[dict], device) -> list[float]:
    if not records:
        return []
    head.eval()
    embeddings = torch.stack([r["embedding"] for r in records]).float().to(device)
    return head(embeddings).squeeze(-1).cpu().tolist()


def evaluate_split(head: nn.Module, records: list[dict], device) -> dict[str, dict]:
    """Per-language-cell metrics for one split."""
    return cells_from_predictions(records, predict(head, records, device))


def cells_from_predictions(records: list[dict], raw_preds: list[float]) -> dict[str, dict]:
    """Per-language-cell metrics from precomputed raw predictions."""
    by_lang: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_lang[r["language"]].append(r)
    preds_by_lang: dict[str, list[float]] = defaultdict(list)
    idx = 0
    for r in records:
        preds_by_lang[r["language"]].append(raw_preds[idx])
        idx += 1
    cells = {}
    for lang, recs in by_lang.items():
        human = [r["score"] for r in recs]
        preds = preds_by_lang[lang]
        cells[lang] = compute_cell_metrics(human, preds)
    return cells


def constant_baselines(train_records: list[dict]) -> dict[str, dict]:
    by_lang: dict[str, list[float]] = defaultdict(list)
    for r in train_records:
        by_lang[r["language"]].append(r["score"])
    return {
        lang: {"mean": statistics.mean(vals), "median": statistics.median(vals)}
        for lang, vals in by_lang.items()
    }


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_predictions(path: Path, records: list[dict], raw_preds: list[float], split: str, run_id: str) -> None:
    rows = []
    for r, p in zip(records, raw_preds):
        rows.append(
            {
                "doc_id": r["doc_id"],
                "language": r["language"],
                "human_mean": r["score"],
                "raw_prediction": p,
                "clipped_prediction": min(5.0, max(0.0, p)),
                "split": split,
                "run_id": run_id,
            }
        )
    write_csv(path, rows, ["doc_id", "language", "human_mean", "raw_prediction", "clipped_prediction", "split", "run_id"])


def cells_to_markdown(cells: dict[str, dict], baselines: dict[str, dict]) -> list[str]:
    lines = ["| Language | N | MAE | RMSE | Bias | Within 0.5 | Within 1 | Spearman | OOB rate | Baseline MAE (mean/median) |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for lang in sorted(cells):
        m = cells[lang]
        sp = f"{m['spearman']:.3f}" if m["spearman"] is not None else "NA"
        base = baselines.get(lang, {})
        base_str = f"{base.get('mean', float('nan')):.2f} / {base.get('median', float('nan')):.2f}" if base else "NA"
        lines.append(
            f"| {lang} | {m['n']} | {m['mae']:.3f} | {m['rmse']:.3f} | {m['bias']:+.3f} | "
            f"{m['within_0_5']*100:.1f}% | {m['within_1_0']*100:.1f}% | {sp} | {m['out_of_range_rate']*100:.1f}% | {base_str} |"
        )
    macro = macro_mae(cells)
    lines.append(f"\nMacro-MAE (equal weight per language): **{macro:.3f}**")
    return lines


def score_range_report(train_records: list[dict], test_records: list[dict], raw_preds: list[float]) -> tuple[list[dict], list[str]]:
    """Group by human scores; errors use clipped predictions, as in overall evaluation."""
    if len(test_records) != len(raw_preds):
        raise ValueError("Test records and predictions must have the same length")
    ranges = ["[0,1)", "[1,2)", "[2,3)", "[3,4)", "[4,5]", "ALL"]
    train_by_lang = defaultdict(list)
    test_by_bin = defaultdict(list)
    for r in train_records:
        train_by_lang[r["language"]].append(r["score"])
    for r, p in zip(test_records, raw_preds):
        test_by_bin[(r["language"], min(4, int(r["score"])))].append((r["score"], p))
    languages = sorted(set(train_by_lang) | {r["language"] for r in test_records})
    if languages:
        train_by_lang["All"] = [r["score"] for r in train_records]
        for b in range(5):
            test_by_bin[("All", b)] = [pair for lang in languages for pair in test_by_bin[(lang, b)]]
        languages.append("All")
    rows = []
    lines = ["### Test metrics by human score range", "",
             "Errors use predictions clipped to [0,5]; bias = prediction - human score. "
             "Training percentages are within each language. All pools documents across languages; its baseline uses each document's language-specific training median. * marks test N < 10; NA means no test data or undefined correlation (fewer than two examples or constant scores). Spearman measures ranking within each range; use overall per-language Spearman for full-scale ranking.", "",
             "| Language | Human range | Train N | Train % | Test N | MAE | RMSE | Bias | Spearman | Median baseline MAE |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for lang in languages:
        train = train_by_lang[lang]
        median = statistics.median(train) if train else None
        for b, label in enumerate(ranges):
            train_n = len(train) if label == "ALL" else sum(min(4, int(y)) == b for y in train)
            share = 100 * train_n / len(train) if train else None
            pairs = ([pair for bin_id in range(5) for pair in test_by_bin[(lang, bin_id)]]
                     if label == "ALL" else test_by_bin[(lang, b)])
            clipped = [min(5.0, max(0.0, p)) for _, p in pairs]
            metrics = compute_cell_metrics([y for y, _ in pairs], [p for _, p in pairs]) if pairs else {}
            baseline_mae = statistics.mean(abs(median - y) for y, _ in pairs) if pairs and median is not None else None
            if lang == "All":
                contributing = [r for r in rows if r["language"] != "All" and r["score_range"] == label and r["test_n"]]
                baseline_mae = (
                    sum(r["median_baseline_mae"] * r["test_n"] for r in contributing) / len(pairs)
                    if pairs and all(r["median_baseline_mae"] is not None for r in contributing) else None
                )
            row = {"language": lang, "score_range": label, "train_n": train_n,
                   "train_percent": share, "test_n": len(pairs),
                   "mae": metrics.get("mae"), "rmse": metrics.get("rmse"),
                   "bias": metrics.get("bias"), "spearman": metrics.get("spearman"), "median_baseline_mae": baseline_mae,
                   "prediction_min": min(clipped) if clipped else None,
                   "prediction_max": max(clipped) if clipped else None}
            rows.append(row)
            values = [f"{row[k]:.3f}" if row[k] is not None else "NA"
                      for k in ["mae", "rmse", "bias", "spearman", "median_baseline_mae"]]
            share_text = f"{share:.1f}%" if share is not None else "NA"
            count = f"{len(pairs)}{'*' if 0 < len(pairs) < 10 else ''}"
            lines.append(f"| {lang} | {label} | {train_n} | {share_text} | {count} | " + " | ".join(values) + " |")
    return rows, lines


def evaluate_and_report(
    head: nn.Module, dataset: dict[str, list[dict]], dimension: str,
    out_dir: Path, device, best_epoch: int,
) -> tuple[dict[str, dict], list[dict]]:
    """Save validation/test predictions, reports, and per-dimension PNG plots."""
    val_raw = predict(head, dataset.get("validation", []), device)
    test_raw = predict(head, dataset.get("test", []), device)
    return report_from_predictions(dataset, val_raw, test_raw, dimension, out_dir, best_epoch)


def report_from_predictions(
    dataset: dict[str, list[dict]], val_raw: list[float], test_raw: list[float],
    dimension: str, out_dir: Path, best_epoch: int,
) -> tuple[dict[str, dict], list[dict]]:
    """Same outputs as evaluate_and_report, from raw predictions already computed
    (used by models that don't take cached embeddings, e.g. fine-tuning)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    baselines = constant_baselines(dataset.get("train", []))
    val_cells = cells_from_predictions(dataset.get("validation", []), val_raw)
    test_cells = cells_from_predictions(dataset.get("test", []), test_raw)

    write_predictions(out_dir / "predictions_validation.csv", dataset.get("validation", []), val_raw, "validation", dimension)
    write_predictions(out_dir / "predictions_test.csv", dataset.get("test", []), test_raw, "test", dimension)

    range_rows, range_lines = score_range_report(dataset.get("train", []), dataset.get("test", []), test_raw)
    write_csv(out_dir / "test_score_ranges.csv", range_rows,
              ["language", "score_range", "train_n", "train_percent", "test_n",
               "mae", "rmse", "bias", "spearman", "median_baseline_mae", "prediction_min", "prediction_max"])
    print(f"\n{dimension}\n" + "\n".join(range_lines))

    report_lines = [f"# {dimension}", "", f"Best epoch: {best_epoch}", "",
                     "## Validation", ""] + cells_to_markdown(val_cells, baselines) + \
                    ["", "## Test", ""] + cells_to_markdown(test_cells, baselines) + [""] + range_lines
    (out_dir / "evaluation_report.md").write_text("\n".join(report_lines), encoding="utf-8")
    plot_test_ranking(out_dir / "test_ranking.png", {dimension: test_cells})
    plot_score_ranges(out_dir / "test_score_ranges.png", dimension, range_rows)
    plot_test_metrics(out_dir / "test_metrics.png", {dimension: test_cells})
    return test_cells, range_rows
