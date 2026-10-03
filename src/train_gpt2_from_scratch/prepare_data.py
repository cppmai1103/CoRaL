"""Select the training documents for a GPT-2 pilot run (guide 02, Step 5).

Every selection method draws from the same candidate pool: the documents in
`data/pilot_corpus/<language>.csv` that the split manifest marks "train". Validation and test
documents (split_corpus.py) are never eligible. Two independent choices:

  --method     random     uniform order, seeded and reproducible per language (the baseline)
               top-score  ranked by the rater scores that score_pool.py wrote to --scores-dir
                          (default: the Educational Value rater). Ranking uses the unclipped
                          model output, ties broken by doc_id; every candidate document must
                          have a score.

  --select-by  docs       (default) keep exactly --num-docs documents per language.
               tokens     keep documents in the method's order (random or ranked), growing the
                          selection --group-size documents at a time (default 5), until each
                          language's running SeaLLM token count (with one EOS per document)
                          reaches --target-tokens. The rater and the token budget both operate
                          on whole documents, never partial ones, so the group that crosses the
                          target is kept in full: the actual total can overshoot the target
                          (reported in summary.md) but never falls short of it, matching guide
                          02 Step 5's "accumulate ranked documents until reaching the token
                          target, retaining the final whole document" procedure.

Tokens are counted with the language-model tokenizer (SeaLLM v2), so methods can be compared on
the same per-language token budget either way. The output is self-contained (it includes the
text), so the pool can be deleted afterwards if the space is needed.

Run from the project root, after extract_corpus and split_corpus (top-score also needs score_pool):
    python -m src.train_gpt2_from_scratch.prepare_data                              # random baseline, docs
    python -m src.train_gpt2_from_scratch.prepare_data --method top-score \\
        --scores-dir data/pilot_scores/educational_value_mean --output-dir data/pilot_selected/edu_top10k
    python -m src.train_gpt2_from_scratch.prepare_data --select-by tokens --target-tokens 5000000 \\
        --output-dir data/pilot_selected/random_5M
    python -m src.train_gpt2_from_scratch.prepare_data --select-by tokens --target-tokens 5000000 \\
        --method top-score --scores-dir data/pilot_scores/avg5_mean --output-dir data/pilot_selected/avg5_5M

Outputs in --output-dir (default data/pilot_selected/random/, data/pilot_selected/top_score/, or with a
_<N>Mtok suffix when --select-by tokens):
    documents.csv   doc_id, language, source, text, char_len, tokens, tokens_with_eos, language_score, url, dump, date
                    (+ rater_score for top-score)
    summary.json    per-language pool size, selected count, token totals (and scores for top-score;
                    target vs. actual tokens for --select-by tokens)
    summary.md      the same as a table
Train on the result with:  METHOD=<output folder name> sbatch sh/train_gpt2.sh
"""

import argparse
import csv
import json
import random
import statistics
import sys
from pathlib import Path

DEFAULT_NUM_DOCS = 10000
DEFAULT_SEED = 42
DEFAULT_GROUP_SIZE = 5
DEFAULT_TOKENIZER = "SeaLLMs/SeaLLM-7B-v2"
OUT_FIELDS = ["doc_id", "language", "source", "text", "char_len", "tokens", "tokens_with_eos",
              "language_score", "url", "dump", "date"]
METHODS = ["random", "top-score"]
SELECT_BY = ["docs", "tokens"]


# --------------------------------------------------------------------------
# Selection
# --------------------------------------------------------------------------

def select_random(rows: list[dict], num_docs: int, seed: int, language: str) -> list[dict]:
    """Uniform sample without replacement, reproducible per (seed, language).
    Returned in pool order."""
    if num_docs > len(rows):
        raise ValueError(f"{language}: asked for {num_docs} documents but the pool has {len(rows)}")
    rng = random.Random(f"{seed}:{language}")
    order = sorted(range(len(rows)), key=lambda i: rows[i]["doc_id"])  # independent of file order
    chosen = sorted(rng.sample(order, num_docs))
    return [rows[i] for i in chosen]


def rank_all_by_score(rows: list[dict], scores: dict[str, float], language: str) -> list[dict]:
    """Every pool document, best-scoring first; ties broken by doc_id.

    Every pool document must have a score: a partial scoring run would silently rank from a
    smaller pool than intended."""
    missing = sum(1 for r in rows if r["doc_id"] not in scores)
    if missing:
        raise ValueError(f"{language}: {missing} of {len(rows)} candidate documents have no rater score; "
                         f"score the whole pool with score_pool.py first")
    return sorted(rows, key=lambda r: (-scores[r["doc_id"]], r["doc_id"]))


def select_top_score(rows: list[dict], scores: dict[str, float], num_docs: int, language: str) -> list[dict]:
    """The `num_docs` highest-scoring pool documents, best first. Each returned row gets a
    `rater_score` field."""
    ranked = rank_all_by_score(rows, scores, language)
    if num_docs > len(ranked):
        raise ValueError(f"{language}: asked for {num_docs} documents but the pool has {len(ranked)}")
    return [dict(r, rater_score=f"{scores[r['doc_id']]:.6f}") for r in ranked[:num_docs]]


def order_random_all(rows: list[dict], seed: int, language: str) -> list[dict]:
    """Every pool document, in a full seeded shuffle order (reproducible per (seed, language)).

    Not the same order select_random draws its sample in: that function keeps its subset in
    original pool order. This one is for --select-by tokens, where the walk order itself decides
    which documents end up selected, so it needs to be a genuine permutation of the whole pool."""
    rng = random.Random(f"{seed}:{language}")
    order = sorted(range(len(rows)), key=lambda i: rows[i]["doc_id"])  # independent of file order
    rng.shuffle(order)
    return [rows[i] for i in order]


def select_by_token_budget(ordered: list[dict], token_counts_with_eos: list[int], target_tokens: int,
                          group_size: int, language: str) -> tuple[list[dict], int]:
    """Walk `ordered` (already ranked or shuffled), `group_size` documents at a time, accumulating
    `token_counts_with_eos` (same order and length as `ordered`), until the running total reaches
    `target_tokens`. Selection only ever grows by whole groups, so the returned total can overshoot
    the target (by at most one group's worth of tokens) but never falls short of it, unless the
    entire pool is exhausted first, which raises instead of silently under-delivering.

    Returns (selected_prefix_of_ordered, actual_token_total)."""
    if group_size < 1:
        raise ValueError("group_size must be at least 1")
    available = sum(token_counts_with_eos)
    if available < target_tokens:
        raise ValueError(f"{language}: candidate pool has only {available:,} tokens ({len(ordered):,} documents), "
                         f"short of the {target_tokens:,}-token target")
    cumulative, kept = 0, 0
    for start in range(0, len(ordered), group_size):
        end = min(start + group_size, len(ordered))
        cumulative += sum(token_counts_with_eos[start:end])
        kept = end
        if cumulative >= target_tokens:
            break
    return ordered[:kept], cumulative


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def load_pool(path: Path) -> list[dict]:
    csv.field_size_limit(sys.maxsize)
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["char_len"] = int(r["char_len"])
    return rows


def load_train_ids(manifest: Path) -> set[tuple[str, str]]:
    """(language, doc_id) of every candidate-pool document; validation/test are left out."""
    with manifest.open(encoding="utf-8", newline="") as f:
        return {(r["language"], r["doc_id"]) for r in csv.DictReader(f) if r["split"] == "train"}


def load_scores(path: Path) -> dict[str, float]:
    """doc_id -> unclipped rater output, from a score_pool.py file."""
    with path.open(encoding="utf-8", newline="") as f:
        return {r["doc_id"]: float(r["score_raw"]) for r in csv.DictReader(f)}


def count_tokens(texts: list[str], tokenizer, batch_size: int = 1000) -> list[int]:
    counts = []
    for i in range(0, len(texts), batch_size):
        counts += [len(ids) for ids in tokenizer(texts[i:i + batch_size], add_special_tokens=False)["input_ids"]]
    return counts


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def summarise(language: str, pool: list[dict], selected: list[dict], scores: dict[str, float] | None = None,
             target_tokens: int | None = None) -> dict:
    out = {
        "language": language,
        "pool_docs": len(pool),
        "selected_docs": len(selected),
        "pool_mean_chars": round(statistics.mean(r["char_len"] for r in pool), 1),
        "selected_mean_chars": round(statistics.mean(r["char_len"] for r in selected), 1),
        "selected_median_chars": int(statistics.median(r["char_len"] for r in selected)),
    }
    if scores is not None:
        chosen = [scores[r["doc_id"]] for r in selected]
        out["pool_mean_score"] = round(statistics.mean(scores[r["doc_id"]] for r in pool), 3)
        out["selected_mean_score"] = round(statistics.mean(chosen), 3)
        out["selected_min_score"] = round(min(chosen), 3)
    if selected and "tokens" in selected[0]:
        out["selected_tokens"] = sum(r["tokens"] for r in selected)
        out["selected_tokens_with_eos"] = sum(r["tokens_with_eos"] for r in selected)
        out["selected_mean_tokens"] = round(statistics.mean(r["tokens"] for r in selected), 1)
        if target_tokens is not None:
            out["target_tokens"] = target_tokens
            out["tokens_over_target"] = out["selected_tokens_with_eos"] - target_tokens
    return out


def build_markdown(summaries: list[dict], args, baseline: dict[str, int] | None = None) -> str:
    """`baseline` maps language -> tokens_with_eos of the Random baseline, for a token ratio column
    (only used for --select-by docs; --select-by tokens reports target vs. actual instead)."""
    has_tokens = "selected_tokens" in summaries[0]
    has_scores = "selected_mean_score" in summaries[0]
    has_target = "target_tokens" in summaries[0]
    by_tokens = args.select_by == "tokens"
    columns = [("Language", lambda s: s["language"]), ("Candidate pool", lambda s: s["pool_docs"]),
               ("Selected", lambda s: s["selected_docs"]),
               ("Pool mean chars", lambda s: s["pool_mean_chars"]), ("Selected mean chars", lambda s: s["selected_mean_chars"])]
    if has_scores:
        columns += [("Pool mean score", lambda s: s["pool_mean_score"]),
                    ("Selected mean score", lambda s: s["selected_mean_score"]),
                    ("Lowest selected score", lambda s: s["selected_min_score"])]
    if has_tokens:
        columns += [("Tokens", lambda s: f"{s['selected_tokens']:,}"),
                    ("Tokens with EOS", lambda s: f"{s['selected_tokens_with_eos']:,}"),
                    ("Mean tokens/doc", lambda s: s["selected_mean_tokens"])]
        if has_target:
            columns += [("Target tokens", lambda s: f"{s['target_tokens']:,}"),
                        ("Over target", lambda s: f"+{s['tokens_over_target']:,}")]
        elif baseline:
            columns += [("Random baseline tokens", lambda s: f"{baseline[s['language']]:,}" if s["language"] in baseline else ""),
                        ("Ratio to random", lambda s: f"{s['selected_tokens_with_eos'] / baseline[s['language']]:.2f}"
                         if s["language"] in baseline else "")]
    method_label = "Random" if args.method == "random" else "Top-score"
    if by_tokens:
        title = f"{method_label} selection, token budget"
        order_desc = (f"random order (seed {args.seed})" if args.method == "random"
                      else f"ranked order by the rater score in `{args.scores_dir}` (ties by doc_id)")
        how = (f"Documents kept in {order_desc}, grown {args.group_size} at a time until each language's running "
               f"token count reaches {args.target_tokens:,}. ")
    elif args.method == "random":
        title = "Random baseline selection"
        how = f"Uniform random sample of {args.num_docs} documents per language, seed {args.seed}. "
    else:
        title = "Top-score selection"
        how = (f"Top {args.num_docs} documents per language by the rater score in `{args.scores_dir}` "
               f"(ranked by the unclipped output, ties by doc_id). ")
    lines = [f"# {title}", "",
             how + f"Candidate pool of `{args.pool_dir}` only (validation/test documents excluded). "
             + (f"Tokens counted with `{args.tokenizer}` (no special tokens; one EOS per document is added in the "
                "`with EOS` column).\n" if has_tokens else "Tokens were not counted (`--no-count-tokens`).\n"),
             "| " + " | ".join(c[0] for c in columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for s in summaries:
        lines.append("| " + " | ".join(str(c[1](s)) for c in columns) + " |")
    if has_tokens:
        total = sum(s["selected_tokens_with_eos"] for s in summaries)
        cells = {"Language": "**Total**", "Selected": sum(s["selected_docs"] for s in summaries),
                 "Tokens": f"{sum(s['selected_tokens'] for s in summaries):,}", "Tokens with EOS": f"{total:,}"}
        if has_target:
            target_total = sum(s["target_tokens"] for s in summaries)
            cells["Target tokens"] = f"{target_total:,}"
            cells["Over target"] = f"+{total - target_total:,}"
        elif baseline:
            base = sum(baseline[s["language"]] for s in summaries if s["language"] in baseline)
            cells["Random baseline tokens"] = f"{base:,}"
            cells["Ratio to random"] = f"{total / base:.2f}"
        lines.append("| " + " | ".join(str(cells.get(c[0], "")) for c in columns) + " |")
    if by_tokens:
        lines += ["", f"Selection grows in whole groups of {args.group_size} documents, so the actual token total can "
                  "overshoot the target (see 'Over target') but never falls short of it. Compare methods at the same "
                  "target to hold the token budget fixed while the selection itself varies."]
    elif args.method == "random":
        lines += ["", "A random sample should look like its pool: compare the two mean-character columns.",
                  "The per-language token totals are the budgets that rater-based methods should be matched to."]
    else:
        lines += ["", "Documents are chosen by count, not by token budget, so the token total can differ from the Random "
                  "baseline: a ratio far from 1.00 means this method trains on more or fewer tokens for the same number of "
                  "documents."]
    return "\n".join(lines + [""])


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--method", choices=METHODS, default="random")
    parser.add_argument("--pool-dir", type=Path, default=Path("data/pilot_corpus"))
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Default: data/pilot_selected/random or data/pilot_selected/top_score")
    parser.add_argument("--scores-dir", type=Path, default=Path("data/pilot_scores/educational_value_mean"),
                        help="top-score only: the folder written by score_pool.py")
    parser.add_argument("--compare-with", type=Path, default=Path("data/pilot_selected/random/summary.json"),
                        help="top-score only: Random-baseline summary.json for the token ratio column (skipped if missing)")
    parser.add_argument("--languages", nargs="+", default=None, help="Default: every <language>.csv in --pool-dir")
    parser.add_argument("--split-manifest", type=Path, default=None,
                        help="Default: <pool-dir>/split_manifest.csv. Only its 'train' documents can be selected")
    parser.add_argument("--no-split-manifest", action="store_true",
                        help="Select from the whole pool (no held-out documents are protected)")
    parser.add_argument("--select-by", choices=SELECT_BY, default="docs",
                        help="docs: keep --num-docs documents per language. "
                             "tokens: keep documents (grown --group-size at a time) until --target-tokens is reached")
    parser.add_argument("--num-docs", type=int, default=DEFAULT_NUM_DOCS, help="select-by=docs: documents per language")
    parser.add_argument("--target-tokens", type=int, default=None,
                        help="select-by=tokens: SeaLLM tokens (with EOS) to reach per language, e.g. 5000000")
    parser.add_argument("--group-size", type=int, default=DEFAULT_GROUP_SIZE,
                        help="select-by=tokens: grow the selection this many documents at a time (default 5)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="random method only")
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--no-count-tokens", action="store_true", help="select-by=docs only: skip loading the tokenizer")
    args = parser.parse_args()

    if args.select_by == "tokens":
        if args.target_tokens is None:
            raise SystemExit("--target-tokens is required with --select-by tokens")
        if args.no_count_tokens:
            raise SystemExit("--no-count-tokens cannot be used with --select-by tokens: token counts drive the selection")

    if args.output_dir is None:
        base = "random" if args.method == "random" else "top_score"
        if args.select_by == "tokens":
            size = f"{args.target_tokens // 1_000_000}M" if args.target_tokens % 1_000_000 == 0 else str(args.target_tokens)
            base = f"{base}_{size}tok"
        args.output_dir = Path("data/pilot_selected") / base

    languages = args.languages or sorted(p.stem for p in args.pool_dir.glob("*.csv") if p.stem != "split_manifest")
    if not languages:
        raise SystemExit(f"No <language>.csv files found in {args.pool_dir}")

    train_ids = None
    if not args.no_split_manifest:
        manifest = args.split_manifest or args.pool_dir / "split_manifest.csv"
        if not manifest.is_file():
            raise SystemExit(f"{manifest} not found. Run split_corpus first, or pass --no-split-manifest "
                             f"to select without protecting held-out documents.")
        train_ids = load_train_ids(manifest)
    if args.method == "top-score":
        missing = [l for l in languages if not (args.scores_dir / f"{l}.csv").is_file()]
        if missing:
            raise SystemExit(f"No rater scores for {missing} in {args.scores_dir}. Run score_pool.py first.")

    tokenizer = None
    if not args.no_count_tokens:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    fields = OUT_FIELDS + (["rater_score"] if args.method == "top-score" else [])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    with (args.output_dir / "documents.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for language in languages:
            pool = load_pool(args.pool_dir / f"{language}.csv")
            if train_ids is not None:
                held_out = len(pool)
                pool = [r for r in pool if (language, r["doc_id"]) in train_ids]
                held_out -= len(pool)
                print(f"[{language}] {held_out} held-out documents excluded; candidate pool = {len(pool)}", flush=True)
            scores = None
            if args.select_by == "docs":
                if args.method == "random":
                    selected = select_random(pool, args.num_docs, args.seed, language)
                else:
                    scores = load_scores(args.scores_dir / f"{language}.csv")
                    selected = select_top_score(pool, scores, args.num_docs, language)
                if tokenizer is not None:
                    for r, n in zip(selected, count_tokens([r["text"] for r in selected], tokenizer)):
                        r["tokens"], r["tokens_with_eos"] = n, n + 1
            else:  # tokens
                if args.method == "random":
                    ordered = order_random_all(pool, args.seed, language)
                else:
                    scores = load_scores(args.scores_dir / f"{language}.csv")
                    ordered = rank_all_by_score(pool, scores, language)
                token_counts = count_tokens([r["text"] for r in ordered], tokenizer)
                token_counts_with_eos = [n + 1 for n in token_counts]
                selected, _ = select_by_token_budget(ordered, token_counts_with_eos, args.target_tokens,
                                                     args.group_size, language)
                for r, n, we in zip(selected, token_counts, token_counts_with_eos):
                    r["tokens"], r["tokens_with_eos"] = n, we
                if scores is not None:
                    for r in selected:
                        r["rater_score"] = f"{scores[r['doc_id']]:.6f}"
            writer.writerows(selected)
            target = args.target_tokens if args.select_by == "tokens" else None
            summaries.append(summarise(language, pool, selected, scores, target))
            s = summaries[-1]
            print(f"[{language}] selected {len(selected)} of {len(pool)}"
                  + (f" (mean score {s['pool_mean_score']} -> {s['selected_mean_score']}, lowest kept {s['selected_min_score']})"
                     if scores is not None else "")
                  + (f", {s['selected_tokens_with_eos']:,} tokens"
                     + (f" (target {target:,}, +{s['tokens_over_target']:,} over)" if target is not None else "")
                     if tokenizer is not None else ""), flush=True)

    baseline = None
    if args.method == "top-score" and tokenizer is not None and args.compare_with.is_file():
        baseline = {r["language"]: r["selected_tokens_with_eos"] for r in json.loads(args.compare_with.read_text())
                    if "selected_tokens_with_eos" in r}
    (args.output_dir / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    (args.output_dir / "summary.md").write_text(build_markdown(summaries, args, baseline), encoding="utf-8")
    print(f"Wrote {args.output_dir / 'documents.csv'}, summary.json and summary.md")


if __name__ == "__main__":
    main()
