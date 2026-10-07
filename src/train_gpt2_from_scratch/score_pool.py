"""Score the candidate training pool with a fine-tuned SEA-Rater (guide 02, Step 5).

Loads a fully fine-tuned rater (default: the Educational Value model with mean pooling,
checkpoints/rater/finetuned/mean/educational_value/model) and predicts its 0-5 score for every
document in the candidate pool, i.e. those the split manifest marks "train" by default (pass
--splits validation or --splits test to score held-out documents instead, e.g. to compare the
importance-score distribution of what was selected against the held-out set it's evaluated on;
point --output-dir elsewhere so it doesn't mix with the candidate-pool scores). The rater is
applied exactly as it was trained: mmBERT tokenizer, one pass per document, truncated at
--max-length (4096) tokens.

For each language it writes <output-dir>/<language>.csv with:
    doc_id, language, score_raw (unclipped model output), score (clipped to [0, 5]), mmbert_tokens
Selection should rank by score_raw: clipping creates many ties at exactly 5.0.
Languages that already have a complete file are skipped (--force to redo).

When scoring is done it also draws the score distribution of every language with the threshold that
keeps the top --top-k documents (default 10,000), and writes the thresholds as a table:
    <output-dir>/score_distribution.png      one histogram per language, threshold marked
    <output-dir>/thresholds.csv              per language: documents scored, threshold, share kept, mean, median, ...
The threshold is the k-th highest raw score, i.e. the lowest score prepare_data.py --method top-score keeps.
--plot-only redraws both from existing score files without loading the rater.

Run from the project root (GPU strongly recommended; the pool is ~48,500 documents per language):
    python -m src.train_gpt2_from_scratch.score_pool
    python -m src.train_gpt2_from_scratch.score_pool --languages vie --limit 200 --device cpu   # quick check
Requires torch, transformers and tqdm.
"""

import argparse
import csv
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

if __package__:
    from . import train_gpt2 as gpt
else:
    import train_gpt2 as gpt

DEFAULT_RATER_DIR = Path("checkpoints/rater/finetuned/mean/educational_value")
OUT_FIELDS = ["doc_id", "language", "score_raw", "score", "mmbert_tokens"]
THRESHOLD_FIELDS = ["language", "scored", "top_k", "threshold", "share_kept", "mean", "median", "p90", "share_above_5"]


def load_candidates(pool_dir: Path, manifest: Path | None, language: str, limit: int = 0,
                    splits: tuple[str, ...] = ("train",)) -> list[dict]:
    """Pool documents of `language` whose split is in `splits` (default: the candidate pool), in file order.
    manifest=None: every document of <language>.csv (corpus not split yet)."""
    csv.field_size_limit(sys.maxsize)
    wanted = None
    if manifest is not None:
        wanted = set()
        with manifest.open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                if row["language"] == language and row["split"] in splits:
                    wanted.add(row["doc_id"])
    docs = []
    with (pool_dir / f"{language}.csv").open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if wanted is None or row["doc_id"] in wanted:
                docs.append({"doc_id": row["doc_id"], "text": row["text"]})
                if limit and len(docs) >= limit:
                    break
    return docs


@torch.no_grad()
def score_documents(model, tokenizer, texts: list[str], max_length: int, batch_size: int, device, use_bf16: bool,
                    desc: str | None = None) -> tuple[list[float], list[int]]:
    """Raw regression output and token count for every text, in input order.

    Texts are batched by character length (so padding is small) and tokenised batch by batch."""
    model.eval()
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    scores = [float("nan")] * len(texts)
    lengths = [0] * len(texts)
    pad_id = tokenizer.pad_token_id
    starts = range(0, len(order), batch_size)
    for s in (gpt.progress(starts, desc=desc, unit="batch") if desc else starts):
        idx = order[s:s + batch_size]
        ids = tokenizer([texts[i] for i in idx], truncation=True, max_length=max_length)["input_ids"]
        longest = max(len(x) for x in ids)
        input_ids = torch.full((len(idx), longest), pad_id, dtype=torch.long)
        mask = torch.zeros((len(idx), longest), dtype=torch.long)
        for row, x in enumerate(ids):
            input_ids[row, :len(x)] = torch.tensor(x)
            mask[row, :len(x)] = 1
        with gpt.autocast_ctx(device, use_bf16):
            logits = model(input_ids=input_ids.to(device), attention_mask=mask.to(device)).logits
        for row, i in enumerate(idx):
            scores[i] = logits[row, 0].float().item()
            lengths[i] = len(ids[row])
    return scores, lengths


def write_scores(path: Path, language: str, docs: list[dict], scores: list[float], lengths: list[int]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUT_FIELDS)
        writer.writeheader()
        for d, raw, n in zip(docs, scores, lengths):
            writer.writerow({"doc_id": d["doc_id"], "language": language, "score_raw": f"{raw:.6f}",
                             "score": f"{min(5.0, max(0.0, raw)):.6f}", "mmbert_tokens": n})


def describe(scores: list[float]) -> str:
    q = statistics.quantiles(scores, n=20)  # 5% steps
    return (f"mean {statistics.mean(scores):.2f}, sd {statistics.pstdev(scores):.2f}, "
            f"5%/50%/95% = {q[0]:.2f}/{q[9]:.2f}/{q[18]:.2f}, above 5: {sum(s > 5 for s in scores) / len(scores):.1%}")


def load_raw_scores(scores_dir: Path, languages: list[str]) -> dict[str, list[float]]:
    out = {}
    for language in languages:
        with (scores_dir / f"{language}.csv").open(encoding="utf-8", newline="") as f:
            out[language] = [float(r["score_raw"]) for r in csv.DictReader(f)]
    return out


def top_k_threshold(scores: list[float], k: int) -> float | None:
    """The k-th highest score: the lowest score kept when the top k documents are selected.
    None if fewer than k documents were scored."""
    if k > len(scores) or k < 1:
        return None
    return sorted(scores, reverse=True)[k - 1]


def threshold_row(language: str, scores: list[float], k: int) -> dict:
    threshold = top_k_threshold(scores, k)
    ordered = sorted(scores)
    return {"language": language, "scored": len(scores), "top_k": k,
            "threshold": "" if threshold is None else round(threshold, 4),
            "share_kept": "" if threshold is None else round(k / len(scores), 4),
            "mean": round(statistics.mean(scores), 4), "median": round(statistics.median(scores), 4),
            "p90": round(ordered[int(0.9 * (len(ordered) - 1))], 4),
            "share_above_5": round(sum(x > 5 for x in scores) / len(scores), 4)}


def plot_score_distributions(scores_dir: Path, languages: list[str], k: int, out_png: Path, title: str = "",
                             xlabel: str = "rater score (unclipped)", note: str | None = None) -> list[dict]:
    """One histogram per language with the top-k threshold; returns the threshold rows."""
    import math

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scores = load_raw_scores(scores_dir, languages)
    rows = [threshold_row(lang, scores[lang], k) for lang in languages]
    lo = min(min(v) for v in scores.values())
    hi = max(max(v) for v in scores.values())
    bins = [lo + (hi - lo) * i / 50 for i in range(51)]
    ncols = min(3, len(languages))
    nrows = math.ceil(len(languages) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 3.8 * nrows), squeeze=False, sharex=True)
    for ax, row in zip(axes.ravel(), rows):
        values = scores[row["language"]]
        ax.hist(values, bins=bins, color="#6c8ebf", edgecolor="white", linewidth=.4)
        ax.axvline(5.0, color="#7f8c8d", linestyle=":", linewidth=1)
        heading = f"{row['language']}  (n={row['scored']:,}, mean {row['mean']:.2f})"
        if row["threshold"] != "":
            t = float(row["threshold"])
            ax.axvspan(t, hi, color="#2ecc71", alpha=.15)
            ax.axvline(t, color="#c0392b", linewidth=1.8)
            ax.text(.98, .95, f"threshold {t:.2f}\ntop {k:,} = {row['share_kept']:.1%} kept",
                    transform=ax.transAxes, ha="right", va="top", fontsize=9,
                    bbox=dict(boxstyle="round", facecolor="white", edgecolor="#c0392b", alpha=.9))
        else:
            heading += f"  (fewer than {k:,} scored: no threshold)"
        ax.set_title(heading, fontsize=10)
        ax.set_ylabel("documents")
        ax.grid(alpha=.25)
    for ax in axes.ravel()[len(rows):]:
        ax.axis("off")
    for ax in axes[-1]:
        ax.set_xlabel(xlabel)
    fig.suptitle(title or f"Rater score distribution of the candidate pool: {scores_dir.name}", fontsize=13)
    fig.text(.5, .005, note or "Red line: lowest score kept when selecting the top documents per language (green = kept). "
             "Dotted line: 5.0, the top of the rating scale; scores above it are unclipped model outputs.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .03, 1, .95))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return rows


def write_thresholds(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=THRESHOLD_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def draw_and_report(scores_dir: Path, languages: list[str], k: int, **plot_kwargs) -> None:
    rows = plot_score_distributions(scores_dir, languages, k, scores_dir / "score_distribution.png", **plot_kwargs)
    write_thresholds(scores_dir / "thresholds.csv", rows)
    print(f"\nTop-{k:,} thresholds (lowest kept score):", flush=True)
    print(f"  {'language':<8}{'scored':>8}{'threshold':>11}{'kept':>8}{'mean':>7}{'median':>8}{'p90':>7}{'>5':>7}")
    for r in rows:
        kept = f"{r['share_kept']:.1%}" if r["share_kept"] != "" else "n/a"
        thr = f"{r['threshold']:.3f}" if r["threshold"] != "" else "n/a"
        print(f"  {r['language']:<8}{r['scored']:>8,}{thr:>11}{kept:>8}{r['mean']:>7.2f}{r['median']:>8.2f}{r['p90']:>7.2f}"
              f"{r['share_above_5']:>7.1%}", flush=True)
    print(f"Wrote {scores_dir / 'score_distribution.png'} and {scores_dir / 'thresholds.csv'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rater-dir", type=Path, default=DEFAULT_RATER_DIR,
                        help="Folder holding model/ (save_pretrained weights + tokenizer) and config.json")
    parser.add_argument("--pool-dir", type=Path, default=Path("data/pilot_corpus"))
    parser.add_argument("--split-manifest", type=Path, default=None, help="Default: <pool-dir>/split_manifest.csv")
    parser.add_argument("--output-dir", type=Path, default=Path("data/pilot_scores/educational_value_mean"))
    parser.add_argument("--splits", nargs="+", default=["train"],
                        help="Manifest split(s) to score (default: train, the candidate pool). Use "
                             "'validation'/'test' to score held-out documents instead (point --output-dir "
                             "elsewhere so it doesn't mix with the candidate-pool scores)")
    parser.add_argument("--no-split-manifest", action="store_true",
                        help="Score every document of each <language>.csv (corpus not split yet); selection later "
                             "keeps only the manifest's train documents")
    parser.add_argument("--languages", nargs="+", default=None,
                        help="Default: every language in the manifest (with --no-split-manifest: every <language>.csv)")
    parser.add_argument("--max-length", type=int, default=4096, help="Truncation length; keep it equal to training")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int, default=0, help="Debug: only the first N candidate documents per language")
    parser.add_argument("--top-k", type=int, default=10000,
                        help="Documents kept per language; sets the threshold drawn on the distribution plot")
    parser.add_argument("--plot-only", action="store_true",
                        help="Do not score: redraw the distribution and thresholds from existing score files")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-bf16", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    gpt.PROGRESS = not args.no_progress
    manifest = None if args.no_split_manifest else (args.split_manifest or args.pool_dir / "split_manifest.csv")
    use_bf16 = not args.no_bf16
    languages = args.languages
    if languages is None and manifest is None:
        languages = sorted(p.stem for p in args.pool_dir.glob("*.csv") if p.stem != "split_manifest")
    elif languages is None:
        with manifest.open(encoding="utf-8", newline="") as f:
            languages = sorted({r["language"] for r in csv.DictReader(f)})

    if args.plot_only:
        draw_and_report(args.output_dir, languages, args.top_k)
        return

    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model_dir = args.rater_dir / "model"
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(args.device).eval()
    print(f"Rater: {model_dir} | pooling={model.config.classifier_pooling} | labels={model.config.num_labels} | "
          f"device={args.device} bf16={use_bf16 and args.device != 'cpu'}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for language in languages:
        out_path = args.output_dir / f"{language}.csv"
        docs = load_candidates(args.pool_dir, manifest, language, args.limit, splits=tuple(args.splits))
        if out_path.is_file() and not args.force:
            with out_path.open(encoding="utf-8", newline="") as f:
                done = sum(1 for _ in csv.DictReader(f))
            if done == len(docs):
                print(f"[{language}] already scored ({done} documents), skipping (use --force to redo)", flush=True)
                continue
        started = time.time()
        scores, lengths = score_documents(model, tokenizer, [d["text"] for d in docs], args.max_length,
                                          args.batch_size, args.device, use_bf16, desc=f"scoring {language}")
        write_scores(out_path, language, docs, scores, lengths)
        took = time.time() - started
        print(f"[{language}] scored {len(docs)} documents in {took / 60:.1f} min ({len(docs) / took:.1f} docs/s, "
              f"{sum(lengths) / took:,.0f} tokens/s): {describe(scores)}", flush=True)
    print(f"Wrote scores to {args.output_dir}")
    if not args.no_plot:
        draw_and_report(args.output_dir, languages, args.top_k)


if __name__ == "__main__":
    main()
