"""Compare raters trained on different labels (e.g. human vs LLM scores) on the same HUMAN test split.

Reads <rater-dir>/<dimension>/predictions_test.csv written by src.train_rater.finetune (or train) for each rater
and scores the clipped predictions against the human mean, on the documents every rater has (the LLM-label
dataset reuses the human test split, so they are the same documents).

    MAE / bias          absolute error and mean signed error (prediction - human): sensitive to the label SCALE,
                        so a rater trained on LLM scores inherits the LLM's offset (e.g. cultural_nuances)
    Spearman            rank agreement with humans, pooled over languages and as the mean of per-language values:
                        what matters for top-k selection, which only uses the ranking within a language

Usage:
    python -m src.train_rater.compare_raters \\
        --raters human=checkpoints/rater/finetuned/mean llm=checkpoints/rater_llm/finetuned/mean \\
        --output checkpoints/rater_llm/rater_comparison.md
"""

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path

if __package__:
    from .test import spearman
    from .train import DIMENSIONS
else:
    from test import spearman
    from train import DIMENSIONS


def load_predictions(path: Path) -> dict[tuple[str, str], tuple[float, float]]:
    """(language, doc_id) -> (human_mean, clipped_prediction)."""
    with path.open(encoding="utf-8") as f:
        return {(r["language"], r["doc_id"]): (float(r["human_mean"]), float(r["clipped_prediction"]))
                for r in csv.DictReader(f)}


def metrics(pairs: dict[tuple[str, str], tuple[float, float]]) -> dict:
    human = [h for h, _ in pairs.values()]
    pred = [p for _, p in pairs.values()]
    by_lang = defaultdict(list)
    for (language, _), (h, p) in pairs.items():
        by_lang[language].append((h, p))
    per_lang = {l: spearman([h for h, _ in v], [p for _, p in v]) for l, v in sorted(by_lang.items())}
    valid = [s for s in per_lang.values() if s is not None]
    return {"n": len(pairs), "mae": statistics.fmean(abs(p - h) for h, p in zip(human, pred)),
            "bias": statistics.fmean(p - h for h, p in zip(human, pred)),
            "spearman": spearman(human, pred), "spearman_lang_mean": statistics.fmean(valid) if valid else None,
            "spearman_per_language": per_lang}


def fmt(x, spec=".3f"):
    return "NA" if x is None else format(x, spec)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raters", nargs="+", required=True, help="label=rater_dir pairs, e.g. human=checkpoints/...")
    parser.add_argument("--dimensions", nargs="+", default=DIMENSIONS)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    raters = dict(item.split("=", 1) for item in args.raters)
    lines = ["# Rater comparison on the human test split", "",
             "Clipped predictions vs the human mean, on documents every rater was tested on. MAE/bias depend on the "
             "label scale; Spearman (rank agreement) is what top-k selection uses.", "",
             "| Dimension | Rater | n | MAE | Bias | Spearman (pooled) | Spearman (mean over languages) |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    per_language_rows = []
    for dim in args.dimensions:
        preds = {}
        for label, rater_dir in raters.items():
            path = Path(rater_dir) / dim / "predictions_test.csv"
            if path.is_file():
                preds[label] = load_predictions(path)
            else:
                print(f"  [{dim}] {label}: no {path}, skipping")
        if not preds:
            continue
        common = set.intersection(*(set(p) for p in preds.values()))
        for label, p in preds.items():
            m = metrics({k: p[k] for k in common})
            lines.append(f"| {dim} | {label} | {m['n']} | {fmt(m['mae'])} | {fmt(m['bias'], '+.3f')} | "
                         f"{fmt(m['spearman'])} | {fmt(m['spearman_lang_mean'])} |")
            per_language_rows.append((dim, label, m["spearman_per_language"]))
            print(f"  {dim:18s} {label:6s} n={m['n']} MAE={fmt(m['mae'])} bias={fmt(m['bias'], '+.3f')} "
                  f"Spearman={fmt(m['spearman'])} (language mean {fmt(m['spearman_lang_mean'])})")
    if per_language_rows:
        languages = sorted({l for _, _, s in per_language_rows for l in s})
        lines += ["", "## Spearman per language", "", "| Dimension | Rater | " + " | ".join(languages) + " |",
                  "| --- | --- |" + " ---: |" * len(languages)]
        for dim, label, s in per_language_rows:
            lines.append(f"| {dim} | {label} | " + " | ".join(fmt(s.get(l)) for l in languages) + " |")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
