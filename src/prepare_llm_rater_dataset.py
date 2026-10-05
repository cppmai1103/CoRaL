"""Build a rater dataset whose TRAINING labels are LLM scores instead of human scores.

Same file layout as src/prepare_dataset.py's data/rater_dataset/, so src.train_rater.finetune runs on it unchanged
(--document-table / --split-manifest-dir). Two pools (--pool):

--pool annotated (default): only the human-annotated documents, with EXACTLY the human dataset's per-dimension
train/validation/test split -- the label is the only difference from the human-label raters:

    train / validation   the human train/validation documents, relabelled with the LLM's integer 0-5 score
                         (documents the LLM never scored -- mostly the <urn:pdid:...> items -- are left out)
    test                 the human test split, unchanged, with human-mean labels

--pool all: every LLM-scored document (~5,000 per language), described below.

    train / validation   LLM-scored documents (data/llm_score/<config>.jsonl, ~5,000 per language), labelled with
                         the LLM's integer 0-5 score; validation = a seeded 10% per language, stratified by the
                         LLM score (checkpoint selection never sees a human label)
    test                 exactly the HUMAN test split of data/rater_dataset/split_manifest_<dimension>.csv, labelled
                         with the human mean -- so the LLM-trained raters are evaluated against humans on the same
                         documents as the human-trained raters, and the two are directly comparable

Leakage guards on the LLM pool (train/validation):
    - drop documents in the human VALIDATION or TEST split of any dimension (matched by id or by text with
      whitespace collapsed);
      documents in the human train split are kept, with their LLM label
    - drop documents in the GPT-2 pilot corpus validation/test split (data/pilot_corpus/split_manifest.csv), so the
      raters never train on text the GPT-2 runs are evaluated on
    - drop duplicate texts (whitespace collapsed) within a language (keep the first) and LLM parse failures (per dimension)

File -> language: by the file-name prefix (fil, indo/ind, khm, zsm/msa, tha, vie); other files (e.g. mya_Mymr,
Burmese) and names containing "checkpoint" are skipped and listed.

Outputs in --output-dir (default data/rater_dataset_llm/):
    document_table.csv            doc_id, language, label_source (llm|human), text_hash, text, llm_<dim>, human_<dim>
    split_manifest_<dimension>.csv   doc_id, language, duplicate_group_id, score, split, label_source
    data_report.md                counts per language/split/dimension, exclusions, LLM score distributions

Usage:
    python -m src.prepare_llm_rater_dataset                      # annotated documents, human splits
    python -m src.prepare_llm_rater_dataset --pool all --output-dir data/rater_dataset_llm_all
"""

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

if __package__:
    from .extract_llm_scores import DIMENSIONS, content_hash, match_records
else:
    from extract_llm_scores import DIMENSIONS, content_hash, match_records

FILE_PREFIX_TO_LANGUAGE = {"fil": "fil", "indo": "indo", "ind": "indo", "khm": "khmer", "zsm": "malay",
                           "msa": "malay", "malay": "malay", "tha": "thai", "vie": "vie"}


def language_of_file(path: Path) -> str | None:
    return FILE_PREFIX_TO_LANGUAGE.get(path.stem.split("_")[0].lower())


def read_csv(path: Path) -> list[dict]:
    csv.field_size_limit(sys.maxsize)
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_llm_documents(llm_dir: Path) -> tuple[dict[str, list[dict]], list[str]]:
    """{language: [record, ...]} from every usable file, and the skipped file names."""
    docs: dict[str, list[dict]] = defaultdict(list)
    skipped = []
    for path in sorted(llm_dir.glob("*.jsonl")):
        language = language_of_file(path)
        if language is None or "checkpoint" in path.name:
            skipped.append(path.name)
            continue
        with path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    rec["_language"] = language
                    docs[language].append(rec)
    return dict(docs), skipped


def heldout_human_keys(rater_dir: Path) -> tuple[set[tuple[str, str]], set[str]]:
    """(language, doc_id) and text hashes of every document in a human validation/test split of any dimension."""
    table = {(r["language"], r["doc_id"]): r for r in read_csv(rater_dir / "document_table.csv")}
    keys, hashes = set(), set()
    for dim in DIMENSIONS:
        for r in read_csv(rater_dir / f"split_manifest_{dim}.csv"):
            if r["split"] in ("validation", "test"):
                key = (r["language"], r["doc_id"])
                keys.add(key)
                if key in table and table[key]["text"].strip():
                    hashes.add(content_hash(table[key]["text"]))
    return keys, hashes


def pilot_heldout_ids(manifest: Path) -> set[tuple[str, str]]:
    return {(r["language"], r["doc_id"]) for r in read_csv(manifest) if r["split"] in ("validation", "test")}


def filter_pool(llm_docs: dict[str, list[dict]], human_keys: set, human_hashes: set, pilot_keys: set
                ) -> tuple[dict[str, list[dict]], dict[str, Counter]]:
    """LLM documents allowed into train/validation, and why the others were dropped."""
    pool, dropped = {}, {}
    for language, recs in llm_docs.items():
        seen, kept, why = set(), [], Counter()
        for rec in recs:
            text = rec.get("text") or ""
            h = content_hash(text)
            if not text.strip():
                why["empty_text"] += 1
            elif (language, rec["id"]) in human_keys or h in human_hashes:
                why["human_validation_or_test"] += 1
            elif (language, rec["id"]) in pilot_keys:
                why["pilot_validation_or_test"] += 1
            elif h in seen:
                why["duplicate_text"] += 1
            else:
                seen.add(h)
                rec["_hash"] = h
                kept.append(rec)
        pool[language], dropped[language] = kept, why
    return pool, dropped


def split_validation(recs: list[dict], dim: str, fraction: float, seed: int) -> dict[str, str]:
    """doc_id -> train/validation for one language and dimension: per LLM score value, a seeded `fraction`
    goes to validation (at least one document per score value with 2+ documents)."""
    by_score: dict[int, list[str]] = defaultdict(list)
    for rec in recs:
        if rec.get(dim) is not None:
            by_score[int(rec[dim])].append(rec["id"])
    split = {}
    for value, ids in sorted(by_score.items()):
        ids = sorted(ids)
        random.Random(f"{seed}-{dim}-{value}").shuffle(ids)
        n_val = round(len(ids) * fraction)
        if n_val == 0 and len(ids) >= 2:
            n_val = 1
        for i, doc_id in enumerate(ids):
            split[doc_id] = "validation" if i < n_val else "train"
    return split


def usable_llm_files(llm_dir: Path) -> tuple[list[Path], list[str]]:
    files, skipped = [], []
    for path in sorted(llm_dir.glob("*.jsonl")):
        if language_of_file(path) is None or "checkpoint" in path.name:
            skipped.append(path.name)
        else:
            files.append(path)
    return files, skipped


def relabel_human_splits(manifests: dict[str, list[dict]], llm_scores: dict[tuple[str, str], dict]
                         ) -> tuple[dict[str, list[dict]], dict[str, Counter]]:
    """Per dimension: the human manifest with train/validation scores replaced by the LLM score; rows without one
    are dropped. Test rows keep the human label. Returns (rows per dimension, dropped train/validation per dimension)."""
    out, dropped = {}, {}
    for dim, rows in manifests.items():
        kept, why = [], Counter()
        for r in rows:
            if r["split"] == "test":
                kept.append({**r, "label_source": "human"})
                continue
            llm = llm_scores.get((r["language"], r["doc_id"]))
            if llm is None:
                why[f"{r['split']}: no LLM record"] += 1
            elif llm.get(dim) is None:
                why[f"{r['split']}: LLM parse failed"] += 1
            else:
                kept.append({**r, "score": float(llm[dim]), "label_source": "llm"})
        out[dim], dropped[dim] = kept, why
    return out, dropped


def build_annotated(args) -> None:
    table = read_csv(args.human_dir / "document_table.csv")
    files, skipped = usable_llm_files(args.llm_dir)
    if not files:
        raise SystemExit(f"No usable LLM score files in {args.llm_dir}")
    docs, _, _ = match_records(table, files)
    keys = Counter((d["language"], d["doc_id"]) for d in docs)
    # (language, doc_id) is the key finetune.py uses; a reused placeholder id matched twice is ambiguous -> no label
    llm_scores = {(d["language"], d["doc_id"]): d["llm"] for d in docs if keys[(d["language"], d["doc_id"])] == 1}
    manifests = {dim: read_csv(args.human_dir / f"split_manifest_{dim}.csv") for dim in DIMENSIONS}
    relabelled, dropped = relabel_human_splits(manifests, llm_scores)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fields = ["doc_id", "language", "duplicate_group_id", "score", "split", "label_source"]
    counts = {}
    for dim, rows in relabelled.items():
        write_csv(args.output_dir / f"split_manifest_{dim}.csv", [{k: r[k] for k in fields} for r in rows], fields)
        counts[dim] = Counter((r["language"], r["split"]) for r in rows)
    needed = {(r["language"], r["doc_id"]) for rows in relabelled.values() for r in rows}
    table_rows = {}
    for h in table:
        key = (h["language"], h["doc_id"])
        if key in needed:
            llm = llm_scores.get(key, {})
            table_rows[key] = {"doc_id": h["doc_id"], "language": h["language"], "label_source": "annotated",
                               "text_hash": h["text_hash"], "text": h["text"],
                               **{f"llm_{d}": "" if llm.get(d) is None else llm[d] for d in DIMENSIONS},
                               **{f"human_{d}": h.get(d, "") for d in DIMENSIONS}}
    write_csv(args.output_dir / "document_table.csv", sorted(table_rows.values(), key=lambda r: (r["language"], r["doc_id"])),
              ["doc_id", "language", "label_source", "text_hash", "text"] + [f"llm_{d}" for d in DIMENSIONS]
              + [f"human_{d}" for d in DIMENSIONS])

    languages = sorted({l for c in counts.values() for l, _ in c})
    lines = ["# LLM-label rater dataset (human-annotated documents, human splits)", "",
             "Same documents and the same per-dimension train/validation/test split as data/rater_dataset/; train and "
             "validation are relabelled with the LLM score (integer 0-5), test keeps the human mean. Train/validation "
             "documents without an LLM score are left out, so the training sets are slightly smaller than the human ones.",
             "", "## Documents per split (train / validation / test)", "",
             "| Dimension | " + " | ".join(languages) + " | total | human-label train / validation |",
             "| --- |" + " ---: |" * (len(languages) + 2)]
    for dim in DIMENSIONS:
        c = counts[dim]
        human = Counter(r["split"] for r in manifests[dim])
        cells = [f"{c[(l, 'train')]} / {c[(l, 'validation')]} / {c[(l, 'test')]}" for l in languages]
        tot = [sum(c[(l, s)] for l in languages) for s in ("train", "validation", "test")]
        lines.append(f"| {dim} | " + " | ".join(cells) + f" | {tot[0]:,} / {tot[1]:,} / {tot[2]:,} | "
                     f"{human['train']:,} / {human['validation']:,} |")
    lines += ["", "## Left out of train/validation", "", "| Dimension | Reason | Documents |", "| --- | --- | ---: |"]
    for dim in DIMENSIONS:
        for reason, n in sorted(dropped[dim].items()):
            lines.append(f"| {dim} | {reason} | {n} |")
    lines += ["", "## Mean label on the training documents: LLM vs human", "",
              "| Dimension | LLM mean | Human mean (same documents) |", "| --- | ---: | ---: |"]
    for dim in DIMENSIONS:
        train = [r for r in relabelled[dim] if r["split"] == "train"]
        human = {(r["language"], r["doc_id"]): float(r["score"]) for r in manifests[dim]}
        llm_mean = sum(r["score"] for r in train) / len(train)
        human_mean = sum(human[(r["language"], r["doc_id"])] for r in train) / len(train)
        lines.append(f"| {dim} | {llm_mean:.2f} | {human_mean:.2f} |")
    if skipped:
        lines += ["", f"Skipped files: {', '.join(skipped)}"]
    (args.output_dir / "data_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    for dim in DIMENSIONS:
        c = counts[dim]
        total = {s: sum(v for (l, sp), v in c.items() if sp == s) for s in ("train", "validation", "test")}
        print(f"  {dim}: train {total['train']:,} | validation {total['validation']:,} | test (human) {total['test']:,}"
              f"  (left out: {dict(dropped[dim])})")
    if skipped:
        print(f"  skipped files: {', '.join(skipped)}")
    print(f"Wrote {args.output_dir}/ (document_table.csv, split_manifest_<dimension>.csv, data_report.md)")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pool", choices=["annotated", "all"], default="annotated",
                        help="annotated: the human-annotated documents with the human splits (default); "
                             "all: every LLM-scored document")
    parser.add_argument("--llm-dir", type=Path, default=Path("data/llm_score"))
    parser.add_argument("--human-dir", type=Path, default=Path("data/rater_dataset"),
                        help="Human rater dataset (document_table.csv + split_manifest_<dim>.csv)")
    parser.add_argument("--pilot-manifest", type=Path, default=Path("data/pilot_corpus/split_manifest.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/rater_dataset_llm"))
    parser.add_argument("--validation-fraction", type=float, default=0.1, help="--pool all only")
    parser.add_argument("--seed", type=int, default=42, help="--pool all only")
    args = parser.parse_args(argv)
    if args.pool == "annotated":
        build_annotated(args)
        return

    llm_docs, skipped = load_llm_documents(args.llm_dir)
    if not llm_docs:
        raise SystemExit(f"No usable LLM score files in {args.llm_dir}")
    human_keys, human_hashes = heldout_human_keys(args.human_dir)
    pool, dropped = filter_pool(llm_docs, human_keys, human_hashes, pilot_heldout_ids(args.pilot_manifest))
    human_table = {(r["language"], r["doc_id"]): r for r in read_csv(args.human_dir / "document_table.csv")}
    languages = sorted(pool)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    table_rows: dict[tuple[str, str], dict] = {}
    for language in languages:
        for rec in pool[language]:
            table_rows[(language, rec["id"])] = {
                "doc_id": rec["id"], "language": language, "label_source": "llm", "text_hash": rec["_hash"],
                "text": rec["text"], **{f"llm_{d}": rec.get(d) if rec.get(d) is not None else "" for d in DIMENSIONS},
                **{f"human_{d}": "" for d in DIMENSIONS}}

    counts = {}
    for dim in DIMENSIONS:
        rows = []
        for language in languages:
            split = split_validation(pool[language], dim, args.validation_fraction, args.seed)
            for rec in pool[language]:
                if rec["id"] in split:
                    rows.append({"doc_id": rec["id"], "language": language, "duplicate_group_id": f"llm_{rec['_hash'][:16]}",
                                 "score": float(rec[dim]), "split": split[rec["id"]], "label_source": "llm"})
        for r in read_csv(args.human_dir / f"split_manifest_{dim}.csv"):
            if r["split"] != "test" or r["language"] not in languages:
                continue
            rows.append({**{k: r[k] for k in ("doc_id", "language", "duplicate_group_id", "score", "split")},
                         "label_source": "human"})
            key = (r["language"], r["doc_id"])
            if key not in table_rows:
                h = human_table[key]
                table_rows[key] = {"doc_id": h["doc_id"], "language": h["language"], "label_source": "human",
                                   "text_hash": h["text_hash"], "text": h["text"],
                                   **{f"llm_{d}": "" for d in DIMENSIONS},
                                   **{f"human_{d}": h.get(d, "") for d in DIMENSIONS}}
        write_csv(args.output_dir / f"split_manifest_{dim}.csv", rows,
                  ["doc_id", "language", "duplicate_group_id", "score", "split", "label_source"])
        counts[dim] = Counter((r["language"], r["split"]) for r in rows)

    write_csv(args.output_dir / "document_table.csv", sorted(table_rows.values(), key=lambda r: (r["language"], r["doc_id"])),
              ["doc_id", "language", "label_source", "text_hash", "text"] + [f"llm_{d}" for d in DIMENSIONS]
              + [f"human_{d}" for d in DIMENSIONS])

    lines = ["# LLM-label rater dataset", "",
             "train/validation: LLM-scored documents with LLM labels (integer 0-5); test: the human test split with "
             f"human-mean labels. Validation = {args.validation_fraction:.0%} per language, stratified by LLM score, "
             f"seed {args.seed}.", "", "## LLM documents per language", "",
             "| Language | LLM records | Kept | In human validation/test | In GPT-2 validation/test | Duplicate text | Empty |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for language in languages:
        d = dropped[language]
        lines.append(f"| {language} | {len(llm_docs[language]):,} | {len(pool[language]):,} | "
                     f"{d['human_validation_or_test']} | {d['pilot_validation_or_test']} | {d['duplicate_text']} | "
                     f"{d['empty_text']} |")
    if skipped:
        lines += ["", f"Skipped files (language not in this project, or in-progress checkpoints): {', '.join(skipped)}"]
    lines += ["", "## Documents per split (train / validation / test)", "",
              "| Dimension | " + " | ".join(languages) + " | total |", "| --- |" + " ---: |" * (len(languages) + 1)]
    for dim in DIMENSIONS:
        c = counts[dim]
        cells = [f"{c[(l, 'train')]} / {c[(l, 'validation')]} / {c[(l, 'test')]}" for l in languages]
        tot = [sum(c[(l, s)] for l in languages) for s in ("train", "validation", "test")]
        lines.append(f"| {dim} | " + " | ".join(cells) + f" | {tot[0]:,} / {tot[1]:,} / {tot[2]:,} |")
    lines += ["", "## LLM score distribution in train+validation (all languages)", "",
              "| Dimension | " + " | ".join(str(v) for v in range(6)) + " | mean |", "| --- |" + " ---: |" * 7]
    for dim in DIMENSIONS:
        vals = [int(rec[dim]) for l in languages for rec in pool[l] if rec.get(dim) is not None]
        dist = Counter(vals)
        lines.append(f"| {dim} | " + " | ".join(f"{dist[v] / len(vals):.0%}" for v in range(6))
                     + f" | {sum(vals) / len(vals):.2f} |")
    (args.output_dir / "data_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    for language in languages:
        print(f"  {language:6s} {len(llm_docs[language]):,} LLM records -> {len(pool[language]):,} kept "
              f"(dropped: {dict(dropped[language])})")
    if skipped:
        print(f"  skipped files: {', '.join(skipped)}")
    for dim in DIMENSIONS:
        c = counts[dim]
        print(f"  {dim}: train {sum(v for (l, s), v in c.items() if s == 'train'):,} | validation "
              f"{sum(v for (l, s), v in c.items() if s == 'validation'):,} | test (human) "
              f"{sum(v for (l, s), v in c.items() if s == 'test'):,}")
    print(f"Wrote {args.output_dir}/ (document_table.csv, split_manifest_<dimension>.csv, data_report.md)")


if __name__ == "__main__":
    main()
