"""Evaluate the trained GPT-2 pilot models against the human-annotated SEA-Rater reference dataset.

Three steps, all on data/rater_dataset/document_table.csv (~5,835 documents / 6 languages, the dataset
the raters were trained/evaluated on -- disjoint from the GPT-2 pilot corpus, so this is unseen text
for every GPT-2 checkpoint below):

  1. Importance scores from the HUMAN labels (no rater model involved): edu = the human
     educational_value score; avg4 = mean of educational_value/reasoning/professionalism/cleanliness
     (the human labels, matching combine_scores.py's dimension set); avg5 = mean of all 5 human
     dimensions. Plots the per-language distribution of each, with the per-language avg5 cutoff for
     --high-quality-fraction (default top 50%) marked on the avg5 panels.
  2. Each trained GPT-2 checkpoint (--runs, default: the 12 combinations of {random, edu, avg4, avg5}
     x {5M, 10M, 20M} tokens/language) is run twice -- the same evaluate() used for validation/test
     during training, just on this different, never-seen corpus: once over every document, and once
     over only the top --high-quality-fraction of each language's documents by avg5 (the "high
     quality" half), to see whether the selection methods separate more once low-quality documents
     (by the human labels) are excluded from the evaluation set.
  3. Reports loss and perplexity per run (per language, macro, and token-weighted) for both, as JSON
     and a markdown summary each.

Outputs in --output-dir (default data/rater_dataset/annotated_eval/):
    importance_score_distribution.png   edu/avg4/avg5 human-score histograms, one row per language,
                                        with each language's top-fraction avg5 cutoff marked
    importance_scores.csv               doc_id, language, edu, avg4, avg5, high_quality (bool)
    gpt2_results.json / gpt2_summary.md                       full dataset
    gpt2_results_highquality.json / gpt2_summary_highquality.md   top --high-quality-fraction by avg5 only

Run from the project root (GPU recommended for step 2; the GPT-2 checkpoints are small so CPU is
also workable for a quick check):
    python -m src.train_gpt2_from_scratch.score_annotated_dataset
    python -m src.train_gpt2_from_scratch.score_annotated_dataset --runs random_5M_ep1_seed42 --device cpu
"""

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path

import torch

if __package__:
    from . import train_gpt2 as gpt
else:
    import train_gpt2 as gpt

DEFAULT_TABLE = Path("data/rater_dataset/document_table.csv")
DEFAULT_RUNS_DIR = Path("checkpoints/gpt2_top_doc")
DEFAULT_OUTPUT_DIR = Path("data/rater_dataset/annotated_eval")
SIZES = ["5M", "10M", "20M"]
METHODS = ["random", "edu", "avg4", "avg5"]
DEFAULT_RUNS = [f"{m}_{s}_ep1_seed42" for m in METHODS for s in SIZES]
AVG4_DIMS = ["educational_value", "reasoning", "professionalism", "cleanliness"]
AVG5_DIMS = AVG4_DIMS + ["cultural_nuances"]


# --------------------------------------------------------------------------
# Step 1: importance scores from the human labels
# --------------------------------------------------------------------------

def load_table(table: Path) -> list[dict]:
    """One row per doc_id. A handful of doc_ids (all the <urn:pdid:...> placeholder-style ones from
    batch2, never the normal <urn:uuid:...> ones) are reused across several different documents
    (different text_hash) in the raw file -- keeping every duplicate would silently let one row's
    membership in a selection (e.g. select_high_quality) drag its differently-scored, same-ID
    siblings along for the ride. Keeps the first row seen per doc_id."""
    csv.field_size_limit(sys.maxsize)
    with table.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    seen = {}
    for r in rows:
        seen.setdefault(r["doc_id"], r)
    deduped = list(seen.values())
    if len(deduped) < len(rows):
        print(f"load_table: {len(rows) - len(deduped)} duplicate-doc_id rows dropped "
              f"(kept first occurrence of {len(deduped)} unique doc_ids)", flush=True)
    return deduped


def compute_importance_scores(rows: list[dict]) -> list[dict]:
    """doc_id, language, edu, avg4, avg5 -- each None if that document is missing a needed human
    label (a handful of documents lack one or two dimensions; see the module docstring)."""
    out = []
    for r in rows:
        def val(dim):
            raw = r.get(dim, "")
            return float(raw) if raw not in ("", None) else None

        edu = val("educational_value")
        avg4_vals = [val(d) for d in AVG4_DIMS]
        avg5_vals = [val(d) for d in AVG5_DIMS]
        out.append({
            "doc_id": r["doc_id"], "language": r["language"],
            "edu": edu,
            "avg4": statistics.mean(avg4_vals) if all(v is not None for v in avg4_vals) else None,
            "avg5": statistics.mean(avg5_vals) if all(v is not None for v in avg5_vals) else None,
        })
    return out


def write_importance_scores(path: Path, scores: list[dict], high_quality_ids: set[str] | None = None) -> None:
    fields = ["doc_id", "language", "edu", "avg4", "avg5"] + (["high_quality"] if high_quality_ids is not None else [])
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for s in scores:
            row = dict(s)
            if high_quality_ids is not None:
                row["high_quality"] = s["doc_id"] in high_quality_ids
            writer.writerow(row)


def select_high_quality(scores: list[dict], fraction: float = 0.5) -> tuple[dict[str, set], dict[str, float]]:
    """Per language, the top `fraction` of documents by avg5 (ties broken by doc_id). Returns
    {language: {doc_id, ...}} and {language: threshold}, where threshold is the lowest avg5 score
    kept -- the cutoff between the "high quality" and "low quality" halves."""
    by_language: dict[str, list[dict]] = {}
    for s in scores:
        if s["avg5"] is not None:
            by_language.setdefault(s["language"], []).append(s)
    kept_ids, thresholds = {}, {}
    for language, docs in by_language.items():
        ranked = sorted(docs, key=lambda s: (-s["avg5"], s["doc_id"]))
        keep_n = round(len(ranked) * fraction)
        kept = ranked[:keep_n]
        kept_ids[language] = {s["doc_id"] for s in kept}
        thresholds[language] = kept[-1]["avg5"] if kept else float("inf")
    return kept_ids, thresholds


def plot_importance_distribution(scores: list[dict], out_png: Path,
                                 high_quality_thresholds: dict[str, float] | None = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    languages = sorted({s["language"] for s in scores})
    labels = ["edu", "avg4", "avg5"]
    label_colors = {"edu": "#c0392b", "avg4": "#2980b9", "avg5": "#27ae60"}
    bins = [0 + 5 * i / 40 for i in range(41)]

    fig, axes = plt.subplots(len(languages), 3, figsize=(14, 2.8 * len(languages)), sharex=True)
    for row, language in enumerate(languages):
        for col, label in enumerate(labels):
            ax = axes[row, col]
            vals = [s[label] for s in scores if s["language"] == language and s[label] is not None]
            ax.hist(vals, bins=bins, color=label_colors[label], alpha=.85, edgecolor="white", linewidth=.3)
            ax.axvline(statistics.mean(vals), color="black", linestyle="--", linewidth=1, label="mean")
            title = f"{language} — {label}  (n={len(vals)}, mean={statistics.mean(vals):.2f})"
            if label == "avg5" and high_quality_thresholds is not None:
                t = high_quality_thresholds[language]
                ymax = ax.get_ylim()[1]
                ax.set_ylim(0, ymax * 1.22)
                ax.axvline(t, color="#8e44ad", linestyle="-", linewidth=2)
                ax.text(t, ymax * 1.2, f" top 50% cutoff\n = {t:.2f}",
                        color="#8e44ad", fontsize=8, ha="left", va="top", fontweight="bold")
            ax.set_title(title, fontsize=10)
            ax.grid(alpha=.25)
            if col == 0:
                ax.set_ylabel("documents")
            if row == len(languages) - 1:
                ax.set_xlabel("score")
    fig.suptitle("Importance score distribution by language (from human annotation)", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
# Step 2: run each trained GPT-2 checkpoint over the dataset
# --------------------------------------------------------------------------

def load_documents_by_language(rows: list[dict], keep_ids: dict[str, set] | None = None) -> dict[str, list[str]]:
    """`keep_ids` (language -> set of doc_id), if given, restricts to only those documents --
    e.g. the top 50% by avg5 from select_high_quality."""
    by_lang: dict[str, list[str]] = {}
    for r in rows:
        if keep_ids is not None and r["doc_id"] not in keep_ids.get(r["language"], ()):
            continue
        by_lang.setdefault(r["language"], []).append(r["text"])
    return by_lang


def evaluate_run(run_dir: Path, docs_by_language: dict[str, list[str]], languages: list[str],
                 batch_size: int, device: str, use_bf16: bool) -> dict:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(run_dir / "tokenizer")
    model = AutoModelForCausalLM.from_pretrained(run_dir / "final").to(device).eval()
    seq_len = model.config.n_positions
    pad_id = tokenizer.pad_token_id

    encoded = {lang: gpt.encode_documents(tokenizer, texts) for lang, texts in docs_by_language.items()}
    blocks = gpt.pack_streams(encoded, languages, seq_len, pad_id, seed=0)
    result = gpt.evaluate(model, blocks, languages, batch_size, device, use_bf16,
                          desc=f"{run_dir.name} on human-annotated data")
    result["data_stats"] = blocks["stats"]
    del model
    if device != "cpu":
        torch.cuda.empty_cache()
    return result


def format_eval_table(results: dict[str, dict]) -> list[str]:
    lines = ["| Run | Tokens | Loss | Perplexity |", "| --- | --- | --- | --- |"]
    for name, r in results.items():
        tokens = sum(s["tokens"] for s in r["data_stats"].values())
        lines.append(f"| {name} | {tokens:,} | {r['macro']['loss']:.4f} | {r['macro']['perplexity']:.1f} |")
    return lines


def write_gpt2_summary(path: Path, results: dict[str, dict], languages: list[str], description: str) -> None:
    lines = ["# GPT-2 pilot checkpoints evaluated on the human-annotated dataset", "", description, ""]
    lines += format_eval_table(results)
    lines += ["", "## Per-language perplexity", "",
             "| Run | " + " | ".join(languages) + " |", "| --- |" + " --- |" * len(languages)]
    for name, r in results.items():
        cells = [f"{r['per_language'][lang]['perplexity']:.1f}" for lang in languages]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_all(run_names: list[str], runs_dir: Path, docs_by_language: dict[str, list[str]], languages: list[str],
           batch_size: int, device: str, use_bf16: bool, results_path: Path | None = None,
           force: bool = False) -> dict[str, dict]:
    """Evaluates each run, skipping ones already present in `results_path` (unless `force`), and
    rewrites `results_path` after every run -- so a SLURM timeout partway through only costs the
    runs after the last checkpoint, and a rerun picks up exactly where it left off."""
    results = {}
    if results_path is not None and results_path.is_file() and not force:
        results = json.loads(results_path.read_text())
        if results:
            print(f"  resuming from {results_path.name}: {len(results)} run(s) already done, skipping them",
                  flush=True)
    for name in run_names:
        if name in results:
            print(f"  [{name}] already in {results_path.name}, skipping (use --force to redo everything)",
                  flush=True)
            continue
        run_dir = runs_dir / name
        if not (run_dir / "final").is_dir():
            print(f"  [{name}] no trained checkpoint at {run_dir}/final, skipping", flush=True)
            continue
        result = evaluate_run(run_dir, docs_by_language, languages, batch_size, device, use_bf16)
        results[name] = result
        print(f"  [{name}] loss={result['macro']['loss']:.4f}  perplexity={result['macro']['perplexity']:.1f}",
              flush=True)
        if results_path is not None:
            results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--table", type=Path, default=DEFAULT_TABLE)
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--runs", nargs="+", default=DEFAULT_RUNS,
                        help="Run directory names under --runs-dir (default: the 12 "
                             "{random,edu,avg4,avg5} x {5M,10M,20M} ep1_seed42 runs)")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--high-quality-fraction", type=float, default=0.5,
                        help="Per-language top fraction by avg5 to also evaluate separately (default: top 50%%)")
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--skip-full-eval", action="store_true",
                        help="Skip step 2+3a (the full-dataset GPT-2 pass) and only run the "
                             "high-quality-subset pass -- e.g. when gpt2_results.json is already up to date "
                             "and you just want to (re)run the high-quality half, or to fit within a shorter "
                             "SLURM time limit")
    parser.add_argument("--force", action="store_true",
                        help="Re-evaluate every run even if it's already in the results JSON on disk "
                             "(default: skip runs already recorded, so a timed-out job can be resumed by "
                             "just resubmitting)")
    parser.add_argument("--no-bf16", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    gpt.PROGRESS = not args.no_progress
    use_bf16 = not args.no_bf16
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_table(args.table)
    languages = sorted({r["language"] for r in rows})

    print("=== step 1: importance scores from human labels ===", flush=True)
    scores = compute_importance_scores(rows)
    high_quality_ids, thresholds = select_high_quality(scores, args.high_quality_fraction)
    all_high_quality_ids = {doc_id for ids in high_quality_ids.values() for doc_id in ids}
    write_importance_scores(args.output_dir / "importance_scores.csv", scores, all_high_quality_ids)
    plot_importance_distribution(scores, args.output_dir / "importance_score_distribution.png", thresholds)
    for label in ["edu", "avg4", "avg5"]:
        vals = [s[label] for s in scores if s[label] is not None]
        print(f"  {label}: n={len(vals)}, mean={statistics.mean(vals):.3f}", flush=True)
    for language, t in sorted(thresholds.items()):
        print(f"  top {args.high_quality_fraction:.0%} avg5 cutoff [{language}]: {t:.3f} "
              f"({len(high_quality_ids[language])} of {sum(1 for s in scores if s['language'] == language and s['avg5'] is not None)} documents)",
              flush=True)
    print(f"  wrote {args.output_dir / 'importance_scores.csv'} and importance_score_distribution.png", flush=True)

    if args.skip_full_eval:
        print("\n=== step 2+3a: skipped (--skip-full-eval) ===", flush=True)
    else:
        print("\n=== step 2+3a: evaluate each GPT-2 checkpoint on the FULL human-annotated text ===", flush=True)
        docs_by_language = load_documents_by_language(rows)
        results_path = args.output_dir / "gpt2_results.json"
        results = run_all(args.runs, args.runs_dir, docs_by_language, languages, args.eval_batch_size, args.device,
                          use_bf16, results_path, args.force)
        write_gpt2_summary(args.output_dir / "gpt2_summary.md", results, languages,
                           "Macro loss/perplexity (mean over languages) on the FULL data/rater_dataset/document_table.csv "
                           "-- text the raters were trained on, disjoint from and unseen by every GPT-2 checkpoint "
                           "here.")
        print(f"Wrote {args.output_dir / 'gpt2_results.json'} and gpt2_summary.md")

    print(f"\n=== step 2+3b: evaluate each GPT-2 checkpoint on the top {args.high_quality_fraction:.0%} "
          f"by avg5 (per language) ===", flush=True)
    hq_docs_by_language = load_documents_by_language(rows, high_quality_ids)
    hq_results_path = args.output_dir / "gpt2_results_highquality.json"
    hq_results = run_all(args.runs, args.runs_dir, hq_docs_by_language, languages, args.eval_batch_size, args.device,
                         use_bf16, hq_results_path, args.force)
    write_gpt2_summary(args.output_dir / "gpt2_summary_highquality.md", hq_results, languages,
                       f"Macro loss/perplexity on only the top {args.high_quality_fraction:.0%} of documents per "
                       f"language by avg5 (human-derived) -- the per-language avg5 cutoffs are marked on "
                       f"importance_score_distribution.png.")
    print(f"Wrote {args.output_dir / 'gpt2_results_highquality.json'} and gpt2_summary_highquality.md")


if __name__ == "__main__":
    main()
