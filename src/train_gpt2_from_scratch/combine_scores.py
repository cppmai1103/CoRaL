"""Combine the rater scores of several dimensions into one importance score per document.

Reads the score folders written by score_pool.py (one per dimension, e.g. data/pilot_scores/reasoning_mean)
and writes a folder in the same format whose score is the equal-weight average over the dimensions:

    importance(doc) = mean over dimensions of score_d(doc)

so prepare_data.py --method top-score can rank by it unchanged. Every folder must have scored exactly
the same documents. By default the *clipped* per-dimension score (0-5, the deployed score) is averaged:
the unclipped Cleanliness output frequently exceeds 5 (up to 80% of documents in one language), and
averaging it raw would let that saturation, not quality, decide the ranking. --column score_raw averages
the unclipped outputs instead.

Also writes, in --output-dir:
    <language>.csv               doc_id, language, score_raw (the average), score (clipped), mmbert_tokens,
                                 and one column per dimension holding the value that was averaged
    score_distribution.png       histogram of the combined score per language with the top-k threshold
    thresholds.csv               thresholds and summary statistics per language
    dimension_correlations.csv   Spearman correlation between the dimensions over the whole pool

Run from the project root, after score_pool has finished for every dimension:
    python -m src.train_gpt2_from_scratch.combine_scores
    python -m src.train_gpt2_from_scratch.combine_scores --scores-dirs data/pilot_scores/educational_value_mean \\
        data/pilot_scores/reasoning_mean --output-dir data/pilot_scores/avg2_mean --top-k 10000
"""

import argparse
import csv
import statistics
from pathlib import Path

if __package__:
    from .score_pool import OUT_FIELDS, draw_and_report
else:
    from score_pool import OUT_FIELDS, draw_and_report

DEFAULT_DIMENSIONS = ["educational_value", "reasoning", "professionalism", "cleanliness", "cultural_nuances"]
DEFAULT_SCORES_DIRS = [Path("data/pilot_scores") / f"{d}_mean" for d in DEFAULT_DIMENSIONS]


NOT_LANGUAGES = {"thresholds", "dimension_correlations"}   # other CSVs written next to the per-language scores


def language_files(scores_dir: Path) -> set[str]:
    """Languages that have a score file in `scores_dir` (thresholds.csv and friends are not languages)."""
    return {p.stem for p in scores_dir.glob("*.csv")} - NOT_LANGUAGES


def load_score_file(path: Path) -> dict[str, dict]:
    with path.open(encoding="utf-8", newline="") as f:
        return {r["doc_id"]: r for r in csv.DictReader(f)}


def combine_language(language: str, scores_dirs: list[Path], column: str) -> list[dict]:
    """One row per document: the average of `column` over the dimensions, plus each dimension's value."""
    per_dim = [load_score_file(d / f"{language}.csv") for d in scores_dirs]
    reference = set(per_dim[0])
    for d, scores in zip(scores_dirs[1:], per_dim[1:]):
        if set(scores) != reference:
            only_a, only_b = len(reference - set(scores)), len(set(scores) - reference)
            raise ValueError(f"{language}: {d} and {scores_dirs[0]} scored different documents "
                             f"({only_a} only in the first, {only_b} only in the second)")
    rows = []
    for doc_id in per_dim[0]:  # file order of the first dimension
        values = [float(scores[doc_id][column]) for scores in per_dim]
        average = statistics.mean(values)
        row = {"doc_id": doc_id, "language": language, "score_raw": f"{average:.6f}",
               "score": f"{min(5.0, max(0.0, average)):.6f}", "mmbert_tokens": per_dim[0][doc_id]["mmbert_tokens"]}
        for d, v in zip(scores_dirs, values):
            row[f"dim_{d.name}"] = f"{v:.6f}"
        rows.append(row)
    return rows


def rank(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def spearman(x: list[float], y: list[float]) -> float | None:
    if len(set(x)) < 2 or len(set(y)) < 2:
        return None
    rx, ry = rank(x), rank(y)
    mx, my = statistics.mean(rx), statistics.mean(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = sum((a - mx) ** 2 for a in rx)
    vy = sum((b - my) ** 2 for b in ry)
    return cov / (vx * vy) ** 0.5


def correlations(columns: dict[str, list[float]]) -> list[dict]:
    names = list(columns)
    return [{"dimension_a": a, "dimension_b": b, "spearman": None if (r := spearman(columns[a], columns[b])) is None else round(r, 4)}
            for i, a in enumerate(names) for b in names[i + 1:]]


def write_language(path: Path, rows: list[dict], dim_fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUT_FIELDS + dim_fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scores-dirs", type=Path, nargs="+", default=DEFAULT_SCORES_DIRS,
                        help="One score_pool.py output folder per dimension (default: the five fine-tuned mean-pooled raters)")
    parser.add_argument("--output-dir", type=Path, default=Path("data/pilot_scores/avg5_mean"))
    parser.add_argument("--column", choices=["score", "score_raw"], default="score",
                        help="Which per-dimension value to average: clipped (default) or unclipped")
    parser.add_argument("--languages", nargs="+", default=None, help="Default: every language present in all folders")
    parser.add_argument("--top-k", type=int, default=10000, help="Documents kept per language; sets the plotted threshold")
    args = parser.parse_args()

    for d in args.scores_dirs:
        if not d.is_dir():
            raise SystemExit(f"{d} not found. Run score_pool.py for that dimension first.")
    languages = args.languages or sorted(set.intersection(*[language_files(d) for d in args.scores_dirs]))
    if not languages:
        raise SystemExit("No language has score files in every folder; finish score_pool.py for all dimensions first.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dim_fields = [f"dim_{d.name}" for d in args.scores_dirs]
    pooled = {f: [] for f in dim_fields}
    for language in languages:
        rows = combine_language(language, args.scores_dirs, args.column)
        write_language(args.output_dir / f"{language}.csv", rows, dim_fields)
        for f in dim_fields:
            pooled[f] += [float(r[f]) for r in rows]
        print(f"[{language}] combined {len(rows)} documents from {len(args.scores_dirs)} dimensions", flush=True)

    corr = correlations(pooled)
    with (args.output_dir / "dimension_correlations.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dimension_a", "dimension_b", "spearman"])
        writer.writeheader()
        writer.writerows(corr)
    print("\nSpearman correlation between dimensions (whole pool):")
    for c in corr:
        print(f"  {c['dimension_a'].removeprefix('dim_'):<28}{c['dimension_b'].removeprefix('dim_'):<28}{c['spearman']}")

    kind = "clipped (0-5)" if args.column == "score" else "unclipped"
    draw_and_report(args.output_dir, languages, args.top_k,
                    title=f"Importance score (average of {len(args.scores_dirs)} dimensions, {kind}): {args.output_dir.name}",
                    xlabel="importance score = mean of the dimension scores",
                    note=f"Red line: lowest score kept when selecting the top documents per language (green = kept). "
                         f"Each dimension's score is {kind} before averaging.")


if __name__ == "__main__":
    main()
