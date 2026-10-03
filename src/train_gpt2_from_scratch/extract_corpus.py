"""Extract the pilot corpus: FineWeb2 train documents that were NOT used for the raters.

For each language this streams `HuggingFaceFW/fineweb-2/data/<config>/train`
(the filtered "clean" config; `--clean-ratio` < 1 also mixes in the heuristic-excluded
"_removed" config, as pipeline_revise.ipynb does for annotation) and keeps the first
documents that pass the same filters as the annotated pool:

  - length window [500, 4000] characters
  - no word from datatrove's banned-words (adult content) list
  - unique document id and unique text within the language

Any document annotated for the raters is excluded, whether it ended up in the
rater train, validation or test split: its FineWeb2 id or its exact text
(SHA-256 of the stripped text) appears in data/annotation_batches/<lang>/<lang>_clean.csv
or <lang>_flag.csv. The result is checked for zero overlap before it is written.

Run from the project root (needs `datatrove[io]`, network access to the Hub):
    python -m src.train_gpt2_from_scratch.extract_corpus
    python -m src.train_gpt2_from_scratch.extract_corpus --languages vie thai --target-per-language 5000

Outputs in --output-dir (default data/pilot_corpus/):
    <language>.csv          one row per document: doc_id, language, hf_config, source, text,
                            text_hash, char_len, language_score, url, dump, date
    <language>.stats.json   what was streamed, skipped (and why) and kept
    extraction_report.md    all languages together
Re-running skips languages whose CSV already holds the target count (use --force to redo).
"""

import argparse
import csv
import hashlib
import json
import re
import statistics
import sys
from pathlib import Path

# language folder in data/annotation_batches -> FineWeb2 config (see pipeline_revise.ipynb)
LANGUAGE_CONFIGS = {
    "indo": "ind_Latn",
    "vie": "vie_Latn",
    "thai": "tha_Thai",
    "malay": "zsm_Latn",
    "fil": "fil_Latn",
    "khmer": "khm_Khmr",
}
HF_DATASET = "HuggingFaceFW/fineweb-2"
SPLIT = "train"
MIN_CHARS = 500
MAX_CHARS = 4000
DEFAULT_TARGET = 5000
DEFAULT_CLEAN_RATIO = 1.0
PROGRESS_EVERY = 500  # raised for large targets, see collect_documents

CSV_FIELDS = ["doc_id", "language", "hf_config", "source", "text", "text_hash", "char_len",
              "language_score", "url", "dump", "date"]
SKIP_REASONS = ["length", "overlap_id", "overlap_text", "duplicate_id", "duplicate_text", "adult"]

_WORD_SPLIT = re.compile(r"[^a-zA-Z0-9]+")


def text_hash(text: str) -> str:
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def load_annotated(dataset_dir: Path, language: str) -> tuple[set[str], set[str]]:
    """IDs and text hashes of every document annotated for this language (clean + flagged)."""
    ids, hashes = set(), set()
    csv.field_size_limit(sys.maxsize)
    for name in (f"{language}_clean.csv", f"{language}_flag.csv"):
        path = dataset_dir / language / name
        if not path.is_file():
            raise FileNotFoundError(f"Annotated data not found: {path}")
        with path.open(encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                if row.get("id"):
                    ids.add(row["id"].strip())
                if row.get("text", "").strip():
                    hashes.add(text_hash(row["text"]))
    return ids, hashes


def load_banned_words() -> set[str]:
    from datatrove.utils._import_utils import ASSETS_PATH

    words = set()
    with (Path(ASSETS_PATH) / "banned_words.txt").open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                words.add(_WORD_SPLIT.sub("", line).lower())
    return words - {""}


def has_banned_word(text: str, banned: set[str]) -> bool:
    return bool(set(_WORD_SPLIT.split(text.lower())) & banned)


def stream_fineweb2(config: str):
    from datatrove.pipeline.readers import ParquetReader

    return ParquetReader(f"hf://datasets/{HF_DATASET}/data/{config}/{SPLIT}")()


def collect_documents(stream, target: int, *, language: str, config: str, source: str,
                      exclude_ids: set, exclude_hashes: set, seen_ids: set, seen_hashes: set,
                      banned: set, min_chars: int = MIN_CHARS, max_chars: int = MAX_CHARS,
                      log=print) -> tuple[list[dict], dict]:
    """Take documents from `stream` (objects with .id, .text, .metadata) until `target` are kept.

    seen_ids / seen_hashes are updated in place so a second source cannot repeat a document."""
    rows = []
    progress_every = max(PROGRESS_EVERY, target // 20)
    stats = {"streamed": 0, **{r: 0 for r in SKIP_REASONS}, "kept": 0, "stream_exhausted": True}
    for doc in stream:
        if len(rows) >= target:
            stats["stream_exhausted"] = False
            break
        stats["streamed"] += 1
        text = (doc.text or "").strip()
        doc_id = (doc.id or "").strip()
        if not min_chars <= len(text) <= max_chars:
            stats["length"] += 1
            continue
        digest = text_hash(text)
        if doc_id in exclude_ids:
            stats["overlap_id"] += 1
            continue
        if digest in exclude_hashes:
            stats["overlap_text"] += 1
            continue
        if doc_id in seen_ids:
            stats["duplicate_id"] += 1
            continue
        if digest in seen_hashes:
            stats["duplicate_text"] += 1
            continue
        if has_banned_word(text, banned):
            stats["adult"] += 1
            continue
        seen_ids.add(doc_id)
        seen_hashes.add(digest)
        meta = doc.metadata or {}
        score = meta.get("language_score")
        rows.append({
            "doc_id": doc_id, "language": language, "hf_config": config, "source": source,
            "text": text, "text_hash": digest, "char_len": len(text),
            "language_score": "" if score is None else float(score),
            "url": meta.get("url", ""), "dump": meta.get("dump", ""), "date": meta.get("date", ""),
        })
        if len(rows) % progress_every == 0:
            log(f"    {language} {source}: kept {len(rows)}/{target} "
                f"(streamed {stats['streamed']}, skipped {sum(stats[r] for r in SKIP_REASONS)})")
    else:
        stats["stream_exhausted"] = len(rows) < target
    stats["kept"] = len(rows)
    return rows, stats


def empty_stats() -> dict:
    return {"streamed": 0, **{r: 0 for r in SKIP_REASONS}, "kept": 0, "stream_exhausted": False}


def extract_language(language: str, args, banned: set) -> tuple[list[dict], dict]:
    config = LANGUAGE_CONFIGS[language]
    exclude_ids, exclude_hashes = load_annotated(args.dataset_dir, language)
    print(f"[{language}] {len(exclude_ids)} annotated ids / {len(exclude_hashes)} texts to exclude; "
          f"FineWeb2 config {config}", flush=True)
    seen_ids, seen_hashes = set(), set()
    common = dict(language=language, exclude_ids=exclude_ids, exclude_hashes=exclude_hashes,
                  seen_ids=seen_ids, seen_hashes=seen_hashes, banned=banned,
                  log=lambda m: print(m, flush=True))

    target_removed = round(args.target_per_language * (1 - args.clean_ratio))
    if target_removed > 0:
        removed_rows, removed_stats = collect_documents(
            stream_fineweb2(f"{config}_removed"), target_removed, config=f"{config}_removed",
            source="removed", **common)
    else:  # clean only: do not touch the _removed config at all
        removed_rows, removed_stats = [], empty_stats()
    target_clean = args.target_per_language - len(removed_rows)  # a short "removed" set is topped up from clean
    clean_rows, clean_stats = collect_documents(
        stream_fineweb2(config), target_clean, config=config, source="clean", **common)

    rows = clean_rows + removed_rows
    stats = {"language": language, "hf_config": config, "target": args.target_per_language,
             "annotated_ids_excluded": len(exclude_ids), "annotated_texts_excluded": len(exclude_hashes),
             "clean": clean_stats, "removed": removed_stats, "kept": len(rows)}
    if len(rows) < args.target_per_language:
        print(f"[{language}] WARNING: only {len(rows)}/{args.target_per_language} documents available", flush=True)
    check_no_overlap(rows, exclude_ids, exclude_hashes)
    return rows, stats


def check_no_overlap(rows: list[dict], exclude_ids: set, exclude_hashes: set) -> None:
    ids = [r["doc_id"] for r in rows]
    digests = [r["text_hash"] for r in rows]
    assert len(set(ids)) == len(ids), "duplicate document ids in the extracted corpus"
    assert len(set(digests)) == len(digests), "duplicate texts in the extracted corpus"
    assert not set(ids) & exclude_ids, "extracted corpus overlaps annotated ids"
    assert not set(digests) & exclude_hashes, "extracted corpus overlaps annotated texts"
    assert all(MIN_CHARS <= r["char_len"] <= MAX_CHARS for r in rows), "length window violated"


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def summarise(rows: list[dict]) -> dict:
    lens = [r["char_len"] for r in rows]
    scores = [r["language_score"] for r in rows if r["language_score"] != ""]
    return {
        "char_len_min": min(lens), "char_len_median": int(statistics.median(lens)),
        "char_len_mean": int(statistics.mean(lens)), "char_len_max": max(lens),
        "language_score_median": round(statistics.median(scores), 4) if scores else None,
    }


def build_report(all_stats: list[dict], args) -> str:
    lines = ["# Pilot corpus extraction report", "",
             f"FineWeb2 `{SPLIT}`, length window [{MIN_CHARS}, {MAX_CHARS}] chars, adult filter on, "
             f"target {args.target_per_language} per language ({args.clean_ratio:.0%} clean / "
             f"{1 - args.clean_ratio:.0%} `_removed`; 0% means the `_removed` config is not read).", "",
             "Documents annotated for the raters (clean + flagged files, i.e. rater train/validation/test) "
             "are excluded by id and by exact text; overlap is asserted to be zero before writing.", "",
             "## Documents kept", "",
             "| Language | Config | Kept | Clean | Removed | Annotated ids excluded | chars min / median / max | median language score |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for s in all_stats:
        m = s["summary"]
        lines.append(f"| {s['language']} | {s['hf_config']} | {s['kept']} | {s['clean']['kept']} | {s['removed']['kept']} | "
                     f"{s['annotated_ids_excluded']} | {m['char_len_min']} / {m['char_len_median']} / {m['char_len_max']} | "
                     f"{m['language_score_median']} |")
    lines += ["", "## Streamed documents and why they were skipped", "",
              "| Language | Source | Streamed | Kept | " + " | ".join(SKIP_REASONS) + " | Stream exhausted |",
              "| --- | --- | --- | --- | " + " | ".join(["---"] * len(SKIP_REASONS)) + " | --- |"]
    for s in all_stats:
        for source in ("clean", "removed"):
            st = s[source]
            lines.append(f"| {s['language']} | {source} | {st['streamed']} | {st['kept']} | "
                         + " | ".join(str(st[r]) for r in SKIP_REASONS) + f" | {st['stream_exhausted']} |")
    lines += ["", "## Notes", "",
              "- Documents are the first eligible ones in FineWeb2 stream order (not a random sample), which is also "
              "how the annotated pool was drawn, so the candidates come from the same part of the corpus as the rater's data.",
              "- `overlap_id` / `overlap_text` count streamed documents skipped because they were annotated for the raters.",
              "- `stream_exhausted = True` with fewer kept documents than the target means the config ran out of eligible documents.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--languages", nargs="+", default=list(LANGUAGE_CONFIGS), choices=list(LANGUAGE_CONFIGS))
    parser.add_argument("--target-per-language", type=int, default=DEFAULT_TARGET)
    parser.add_argument("--clean-ratio", type=float, default=DEFAULT_CLEAN_RATIO,
                        help="Share taken from the filtered config; the rest comes from the _removed config")
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/annotation_batches"),
                        help="Annotated data used to train the raters (documents to exclude)")
    parser.add_argument("--output-dir", type=Path, default=Path("data/pilot_corpus"))
    parser.add_argument("--force", action="store_true", help="Re-extract languages that already have a complete CSV")
    args = parser.parse_args()
    if not 0 <= args.clean_ratio <= 1:
        raise SystemExit("--clean-ratio must be between 0 and 1")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    banned = load_banned_words()
    print(f"Loaded {len(banned)} banned words", flush=True)

    all_stats = []
    for language in args.languages:
        csv_path = args.output_dir / f"{language}.csv"
        stats_path = args.output_dir / f"{language}.stats.json"
        if not args.force and csv_path.is_file() and stats_path.is_file():
            stats = json.loads(stats_path.read_text(encoding="utf-8"))
            if stats.get("kept") == stats.get("target") == args.target_per_language:
                print(f"[{language}] already extracted ({stats['kept']} documents), skipping (use --force to redo)", flush=True)
                all_stats.append(stats)
                continue
        rows, stats = extract_language(language, args, banned)
        stats["summary"] = summarise(rows)
        write_csv(csv_path, rows)
        stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
        print(f"[{language}] wrote {len(rows)} documents -> {csv_path}", flush=True)
        all_stats.append(stats)

    (args.output_dir / "extraction_report.md").write_text(build_report(all_stats, args), encoding="utf-8")
    print(f"Report: {args.output_dir / 'extraction_report.md'}")


if __name__ == "__main__":
    main()
