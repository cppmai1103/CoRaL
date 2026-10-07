"""Prepare the SEA-Rater annotation dataset for training: validate records
and write an independent train/validation/test split *per dimension*.

Each of the 5 rating dimensions gets its own clean subset and its own
per-language stratified split on that dimension's own score distribution.
This means a document can be `train` for professionalism and `test` for
educational_value if both scores happen to be valid for it -- the 5 splits
are not required to agree with each other.

Practical consequence for training: because a document's split status can
differ across dimensions, the 5 regression heads cannot be trained from one
shared batch with one combined 5-way MSE loss (that would leak a document
held out for one head's test set into another head's training step). Train
each head as an independent regression over its own document list, reusing
the same cached frozen embeddings.

Implements workflow Step 1 / guide section 3-4 in 01_train_rater.md, adapted
for per-dimension splits:
  - one record per document, human_mean already averaged by upstream cleaning
  - drop (never zero-fill) a document from a dimension's dataset if that
    dimension's score is missing/invalid/out-of-range for it; log it for
    review. A document can be valid for some dimensions and excluded from
    others.
  - split whole duplicate-text groups together, never individual rows
  - stratify by a coarse bin of the dimension's own score, within each
    language, so that dimension's score distribution is preserved across
    train/validation/test

Input: data/annotation_batches/<language>_clean.csv (from cppmai/sea-rater).
Output per dimension: data/rater_dataset/split_manifest_<dimension>.csv
Shared output: data/rater_dataset/document_table.csv (all 5 raw scores and all 5
    per-dimension split columns), data/rater_dataset/excluded_documents.csv
    (long format, one row per excluded document x dimension),
    data/rater_dataset/data_quality_report.md

Usage:
    python -m src.prepare_dataset
    python -m src.prepare_dataset --input-dir data/annotation_batches --output-dir data/rater_dataset
"""

import argparse
import csv
import hashlib
import random
import statistics
from pathlib import Path

DIMENSIONS = [
    "educational_value",
    "reasoning",
    "professionalism",
    "cleanliness",
    "cultural_nuances",
]

# Workflow Step 1 default (README.md section on split); overridable via CLI.
DEFAULT_SPLIT_RATIOS = {"train": 0.7, "validation": 0.1, "test": 0.2}
DEFAULT_SEED = 42
GUIDELINE_VERSION = "2.1"

# All eight SEA-Rater target languages (README.md section 2); only the ones
# with a *_clean.csv under --input-dir are actually processed. The rest are
# reported as missing so gaps in language coverage stay visible.
TARGET_LANGUAGES = ["indo", "vie", "thai", "malay", "fil", "khmer", "lao", "burmese"]


def discover_language_files(input_dir: Path) -> dict[str, Path]:
    files = {}
    for path in sorted(input_dir.glob("*_clean.csv")):
        lang = path.name.removesuffix("_clean.csv")
        files[lang] = path
    return files


def parse_score(raw: str) -> tuple[float | None, str | None]:
    """Return (value, error_reason). value is None if invalid."""
    text = raw.strip()
    if text == "":
        return None, "missing"
    try:
        value = float(text)
    except ValueError:
        return None, f"nonnumeric:{text!r}"
    if not (0.0 <= value <= 5.0):
        return None, f"out_of_range:{value}"
    return value, None


def load_language(path: Path, lang: str) -> list[dict]:
    """Load one language's clean.csv. One dict per row, with per-dimension
    values/reasons kept independent (a row can be valid for some dimensions
    and invalid for others)."""
    rows = []
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            doc_id = row["id"].strip()
            text = row["text"]
            text_ok = bool(text.strip())

            values: dict[str, float | None] = {}
            reasons: dict[str, str | None] = {}
            for dim in DIMENSIONS:
                if not text_ok:
                    values[dim] = None
                    reasons[dim] = "empty_text"
                    continue
                value, err = parse_score(row.get(dim, ""))
                values[dim] = value
                reasons[dim] = err

            text_hash = (
                hashlib.sha256(text.strip().encode("utf-8")).hexdigest() if text_ok else None
            )
            rows.append(
                {
                    "doc_id": doc_id,
                    "language": lang,
                    "source_language": lang,
                    "batch": row.get("batch", ""),
                    "text": text,
                    "text_hash": text_hash,
                    "guideline_version": GUIDELINE_VERSION,
                    "values": values,
                    "reasons": reasons,
                }
            )
    return rows


def assign_duplicate_groups(rows: list[dict]) -> None:
    """Exact-text duplicates (any language, any dimension) share one
    duplicate_group_id, computed once and reused for every dimension's
    split. Mutates rows in place; rows with no text_hash (empty text) get
    a unique group of their own since they are excluded everywhere anyway."""
    first_seen: dict[str, str] = {}
    for rec in rows:
        h = rec["text_hash"]
        if h is None:
            rec["duplicate_group_id"] = f"notext_{rec['language']}_{rec['doc_id']}"
            continue
        group_id = first_seen.setdefault(h, f"dup_{h[:16]}")
        rec["duplicate_group_id"] = group_id


def score_bin(value: float) -> int:
    """Coarse rubric-scale bin: [0,1) -> 0, [1,2) -> 1, ... [4,5] -> 4."""
    return min(4, int(value // 1))


def largest_remainder_allocation(n: int, ratios: dict[str, float]) -> dict[str, int]:
    """Split n items across ratios.keys() as close to `ratios` as possible,
    all counts summing exactly to n (largest-remainder / Hamilton method)."""
    names = list(ratios.keys())
    raw = {name: n * ratios[name] for name in names}
    floors = {name: int(raw[name]) for name in names}
    remainder = n - sum(floors.values())
    # Largest fractional part first; stable tie-break by declared ratio order.
    order = sorted(names, key=lambda name: (-(raw[name] - floors[name]), names.index(name)))
    for name in order[:remainder]:
        floors[name] += 1
    return floors


def split_dimension_for_language(
    records: list[dict], ratios: dict[str, float], seed: int
) -> dict[str, str]:
    """Stratified group split of one dimension's valid records within one
    language, binned on that dimension's own score. Returns
    {duplicate_group_id: split}."""
    groups: dict[str, list[dict]] = {}
    for rec in records:
        groups.setdefault(rec["duplicate_group_id"], []).append(rec)

    bins: dict[int, list[str]] = {}
    for group_id, members in groups.items():
        group_value = statistics.mean(m["value"] for m in members)
        bins.setdefault(score_bin(group_value), []).append(group_id)

    rng = random.Random(seed)
    assignment: dict[str, str] = {}
    for bin_id in sorted(bins):
        group_ids = sorted(bins[bin_id])  # deterministic order before shuffling
        rng.shuffle(group_ids)
        counts = largest_remainder_allocation(len(group_ids), ratios)
        cursor = 0
        for split_name, count in counts.items():
            for group_id in group_ids[cursor : cursor + count]:
                assignment[group_id] = split_name
            cursor += count
    return assignment


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_report(
    lang_files: dict[str, Path],
    all_rows: dict[str, list[dict]],
    dim_splits: dict[str, dict[str, dict[str, str]]],  # dim -> lang -> {doc_id: split}
    ratios: dict[str, float],
    seed: int,
) -> str:
    lines = ["# SEA-Rater data preparation report (per-dimension splits)", ""]
    lines.append(f"Split seed: {seed}. Target ratios: {ratios}.")
    lines.append("Each dimension below has its own independent split, stratified on that "
                 "dimension's own score within each language. The same document can fall "
                 "into different splits for different dimensions.")
    lines.append("")

    missing_langs = [l for l in TARGET_LANGUAGES if l not in lang_files]
    if missing_langs:
        lines.append(f"**Languages with no `*_clean.csv` found (excluded from this run):** {', '.join(missing_langs)}")
        lines.append("")

    for dim in DIMENSIONS:
        lines.append(f"## {dim}")
        lines.append("")
        lines.append("| Language | Valid | Excluded | Train | Validation | Test |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for lang in sorted(lang_files):
            rows = all_rows[lang]
            valid_rows = [r for r in rows if r["reasons"][dim] is None]
            excluded_rows = [r for r in rows if r["reasons"][dim] is not None]
            splits = dim_splits[dim][lang]
            counts = {s: sum(1 for r in valid_rows if splits[r["doc_id"]] == s) for s in ["train", "validation", "test"]}
            lines.append(
                f"| {lang} | {len(valid_rows)} | {len(excluded_rows)} | "
                f"{counts['train']} | {counts['validation']} | {counts['test']} |"
            )
        lines.append("")

        lines.append("Per-split mean (+/- std) for this dimension's own score:")
        lines.append("")
        lines.append("| Language | Train | Validation | Test |")
        lines.append("| --- | --- | --- | --- |")
        for lang in sorted(lang_files):
            rows = all_rows[lang]
            valid_rows = [r for r in rows if r["reasons"][dim] is None]
            splits = dim_splits[dim][lang]
            cells = []
            for s in ["train", "validation", "test"]:
                vals = [r["values"][dim] for r in valid_rows if splits[r["doc_id"]] == s]
                if vals:
                    cells.append(f"{statistics.mean(vals):.2f} (+/-{statistics.pstdev(vals):.2f}, n={len(vals)})")
                else:
                    cells.append("NA")
            lines.append(f"| {lang} | " + " | ".join(cells) + " |")
        lines.append("")

    lines.append("## Notes and limitations")
    lines.append("")
    lines.append("- Splits are independent per dimension: a document's split for "
                 "`professionalism` can differ from its split for `educational_value`. "
                 "Each dimension's head must therefore be trained as an independent "
                 "regression (own document list, own optimizer steps) on the shared "
                 "frozen embeddings, not jointly with one combined 5-way batch/loss -- "
                 "otherwise a document held out for one head leaks into training for another.")
    lines.append("- `*_clean.csv` already stores the averaged human_mean per dimension; the two "
                 "individual annotator ratings are not present in these files, so `human_scores` "
                 "(the two raw ratings) could not be reconstructed or preserved here.")
    lines.append("- A document is excluded from a dimension's dataset only if that dimension's "
                 "own score is missing/invalid/out-of-range for it (or its text is empty, which "
                 "excludes it from all 5 dimensions); it can still be valid for other dimensions.")
    lines.append("- No exact-duplicate texts were found within any language in this snapshot, so "
                 "duplicate-group splitting had no effect here; the logic is kept for future data.")
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", default="data/annotation_batches", type=Path)
    parser.add_argument("--output-dir", default="data/rater_dataset", type=Path)
    parser.add_argument("--seed", default=DEFAULT_SEED, type=int)
    parser.add_argument("--train-ratio", default=DEFAULT_SPLIT_RATIOS["train"], type=float)
    parser.add_argument("--val-ratio", default=DEFAULT_SPLIT_RATIOS["validation"], type=float)
    parser.add_argument("--test-ratio", default=DEFAULT_SPLIT_RATIOS["test"], type=float)
    args = parser.parse_args()

    ratios = {"train": args.train_ratio, "validation": args.val_ratio, "test": args.test_ratio}
    total_ratio = sum(ratios.values())
    if abs(total_ratio - 1.0) > 1e-6:
        raise SystemExit(f"train/val/test ratios must sum to 1.0, got {total_ratio}")

    lang_files = discover_language_files(args.input_dir)
    if not lang_files:
        raise SystemExit(f"No <language>_clean.csv files found under {args.input_dir}")

    all_rows: dict[str, list[dict]] = {}
    for lang, path in lang_files.items():
        rows = load_language(path, lang)
        assign_duplicate_groups(rows)
        all_rows[lang] = rows
        n_valid_any = sum(1 for r in rows if any(r["reasons"][d] is None for d in DIMENSIONS))
        print(f"{lang}: {len(rows)} rows, {n_valid_any} valid for at least one dimension (from {path})")

    # dim_splits[dim][lang] = {doc_id: split}, computed independently per dimension.
    dim_splits: dict[str, dict[str, dict[str, str]]] = {dim: {} for dim in DIMENSIONS}
    for dim in DIMENSIONS:
        for lang, rows in all_rows.items():
            valid_records = [
                {"doc_id": r["doc_id"], "duplicate_group_id": r["duplicate_group_id"], "value": r["values"][dim]}
                for r in rows
                if r["reasons"][dim] is None
            ]
            group_split = split_dimension_for_language(valid_records, ratios, seed=args.seed)
            dim_splits[dim][lang] = {
                rec["doc_id"]: group_split[rec["duplicate_group_id"]] for rec in valid_records
            }

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # document_table.csv: one row per document with text, all 5 raw scores,
    # and all 5 per-dimension split columns ("excluded" if invalid for that dim).
    doc_table_rows = []
    for lang, rows in all_rows.items():
        for r in rows:
            if r["text_hash"] is None:
                continue  # empty text: useless for every dimension, not worth a row
            out = {
                "doc_id": r["doc_id"],
                "language": r["language"],
                "source_language": r["source_language"],
                "batch": r["batch"],
                "text": r["text"],
                "text_hash": r["text_hash"],
                "duplicate_group_id": r["duplicate_group_id"],
                "guideline_version": r["guideline_version"],
            }
            for dim in DIMENSIONS:
                out[dim] = r["values"][dim] if r["values"][dim] is not None else ""
                out[f"split_{dim}"] = (
                    dim_splits[dim][lang][r["doc_id"]] if r["reasons"][dim] is None else "excluded"
                )
            doc_table_rows.append(out)

    doc_fields = (
        ["doc_id", "language", "source_language", "batch", "text", "text_hash",
         "duplicate_group_id", "guideline_version"]
        + DIMENSIONS
        + [f"split_{d}" for d in DIMENSIONS]
    )
    write_csv(args.output_dir / "document_table.csv", doc_table_rows, doc_fields)

    # One lean manifest per dimension: only the documents valid for that dimension.
    for dim in DIMENSIONS:
        manifest_rows = []
        for lang, rows in all_rows.items():
            for r in rows:
                if r["reasons"][dim] is not None:
                    continue
                manifest_rows.append(
                    {
                        "doc_id": r["doc_id"],
                        "language": lang,
                        "duplicate_group_id": r["duplicate_group_id"],
                        "score": r["values"][dim],
                        "split": dim_splits[dim][lang][r["doc_id"]],
                    }
                )
        write_csv(
            args.output_dir / f"split_manifest_{dim}.csv",
            manifest_rows,
            ["doc_id", "language", "duplicate_group_id", "score", "split"],
        )

    # excluded_documents.csv: long format, one row per (document, dimension) exclusion.
    excluded_rows = []
    for lang, rows in all_rows.items():
        for r in rows:
            for dim in DIMENSIONS:
                reason = r["reasons"][dim]
                if reason is not None:
                    excluded_rows.append(
                        {"doc_id": r["doc_id"], "language": lang, "batch": r["batch"], "dimension": dim, "reason": reason}
                    )
    write_csv(
        args.output_dir / "excluded_documents.csv",
        excluded_rows,
        ["doc_id", "language", "batch", "dimension", "reason"],
    )

    report = build_report(lang_files, all_rows, dim_splits, ratios, args.seed)
    (args.output_dir / "data_quality_report.md").write_text(report, encoding="utf-8")

    print(f"\nWrote document_table.csv, split_manifest_<dimension>.csv x5, "
          f"excluded_documents.csv, data_quality_report.md to {args.output_dir}")


if __name__ == "__main__":
    main()
