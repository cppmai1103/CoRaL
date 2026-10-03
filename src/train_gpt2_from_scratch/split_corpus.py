"""Hold out validation and test documents from the pilot corpus, per language.

The remaining documents are the candidate training pool ("train" in the manifest):
every selection method draws its training documents from that pool only.

Guide 02, Step 2: reserve validation and test documents before any quality-based
selection, and keep near-identical pages from crossing splits. FineWeb2 pages are
concentrated by website (e.g. one Khmer news site is ~10% of the sample), so a
random document-level split would leak site templates into validation/test.

Method (per language, seeded):
  1. Group documents by website (host name without "www.").
  2. Fill test, then validation, with whole groups drawn at random until each has
     exactly the requested size. Only groups of at most --max-eval-group-size
     documents are eligible, so no single site can dominate a small evaluation set.
  3. Everything else is train. A website therefore lives in exactly one split.

Consequence: validation/test come from smaller sites, while train also holds the
big ones. This is stricter than a random split, and the report shows the difference.

Run from the project root, after extract_corpus:
    python -m src.train_gpt2_from_scratch.split_corpus                      # 500 validation + 1000 test per language
    python -m src.train_gpt2_from_scratch.split_corpus --validation 250 --test 250

Outputs in --corpus-dir (default data/pilot_corpus/):
    split_manifest.csv   doc_id, language, split, domain, char_len, source
    split_report.md      sizes, sites per split and length/language-score comparison
"""

import argparse
import csv
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlparse

SPLITS = ["train", "validation", "test"]
DEFAULT_SIZES = {"validation": 500, "test": 1000}  # train = everything else
DEFAULT_SEED = 42
DEFAULT_MAX_EVAL_GROUP = 10
MANIFEST_FIELDS = ["doc_id", "language", "split", "domain", "char_len", "source"]


def domain_of(url: str, fallback: str) -> str:
    host = (urlparse(url or "").hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    return host or f"no-url:{fallback}"


def split_documents(docs: list[dict], sizes: dict[str, int], seed: int, max_eval_group: int) -> dict[str, str]:
    """docs need 'doc_id' and 'domain'. Returns {doc_id: split}."""
    if sum(sizes.values()) != len(docs):
        raise ValueError(f"split sizes {sizes} sum to {sum(sizes.values())} but there are {len(docs)} documents")
    groups: dict[str, list[str]] = defaultdict(list)
    for d in docs:
        groups[d["domain"]].append(d["doc_id"])

    rng = random.Random(seed)
    eligible = sorted(g for g, ids in groups.items() if len(ids) <= max_eval_group)
    rng.shuffle(eligible)

    assignment: dict[str, str] = {}
    used = set()
    for split in ("test", "validation"):
        remaining = sizes[split]
        for g in eligible:
            if remaining == 0:
                break
            if g in used or len(groups[g]) > remaining:
                continue
            used.add(g)
            remaining -= len(groups[g])
            for doc_id in groups[g]:
                assignment[doc_id] = split
        if remaining:
            raise ValueError(f"could not fill '{split}' exactly: {remaining} documents short "
                             f"(groups of <= {max_eval_group} documents are eligible)")
    for d in docs:
        assignment.setdefault(d["doc_id"], "train")
    return assignment


def load_language(path: Path) -> list[dict]:
    csv.field_size_limit(sys.maxsize)
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["domain"] = domain_of(r.get("url", ""), r["doc_id"])
        r["char_len"] = int(r["char_len"])
    return rows


def build_report(per_language: dict[str, list[dict]], sizes: dict[str, int], args) -> str:
    lines = ["# Pilot corpus split report", "",
             f"Per language: validation {sizes['validation']} / test {sizes['test']} documents held out, the rest "
             f"({sizes['train']} here) is the candidate training pool ('train'), "
             f"seed {args.seed}. Split by website; validation/test use only sites with at most "
             f"{args.max_eval_group_size} documents.", "",
             "## Sizes and sites", "",
             "| Language | Split | Documents | Sites | Largest site (docs) | Median chars | Mean language score |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for lang, rows in per_language.items():
        for split in SPLITS:
            sub = [r for r in rows if r["split"] == split]
            sites = defaultdict(int)
            for r in sub:
                sites[r["domain"]] += 1
            scores = [float(r["language_score"]) for r in sub if r.get("language_score") not in ("", None)]
            lines.append(f"| {lang} | {split} | {len(sub)} | {len(sites)} | {max(sites.values())} | "
                         f"{int(statistics.median(r['char_len'] for r in sub))} | "
                         f"{statistics.mean(scores):.3f} |" if scores else
                         f"| {lang} | {split} | {len(sub)} | {len(sites)} | {max(sites.values())} | "
                         f"{int(statistics.median(r['char_len'] for r in sub))} | NA |")
    lines += ["", "## Checks", ""]
    for lang, rows in per_language.items():
        by_domain = defaultdict(set)
        for r in rows:
            by_domain[r["domain"]].add(r["split"])
        crossing = sum(1 for s in by_domain.values() if len(s) > 1)
        lines.append(f"- {lang}: {len(rows)} documents, {crossing} sites appear in more than one split.")
    lines += ["", "## Notes", "",
              "- Validation and test contain only small sites, so they are not identical in site mix to train, "
              "which also holds the large sites. Compare selection methods on the same held-out documents.",
              "- Rater-supervision documents are already excluded from the whole corpus by extract_corpus.",
              "- Exact duplicates were removed at extraction; near-duplicate detection beyond identical text and "
              "same-site grouping has not been run.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus-dir", type=Path, default=Path("data/pilot_corpus"))
    parser.add_argument("--languages", nargs="+", default=None,
                        help="Default: every <language>.csv found in --corpus-dir")
    parser.add_argument("--train", type=int, default=None,
                        help="Candidate-pool size; default = all documents not held out")
    parser.add_argument("--validation", type=int, default=DEFAULT_SIZES["validation"])
    parser.add_argument("--test", type=int, default=DEFAULT_SIZES["test"])
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-eval-group-size", type=int, default=DEFAULT_MAX_EVAL_GROUP,
                        help="Largest site allowed in validation/test (documents)")
    args = parser.parse_args()

    languages = args.languages or sorted(p.stem for p in args.corpus_dir.glob("*.csv") if p.stem != "split_manifest")
    if not languages:
        raise SystemExit(f"No <language>.csv files found in {args.corpus_dir}")

    per_language, manifest = {}, []
    for lang in languages:
        rows = load_language(args.corpus_dir / f"{lang}.csv")
        sizes = {"train": args.train if args.train is not None else len(rows) - args.validation - args.test,
                 "validation": args.validation, "test": args.test}
        assignment = split_documents(rows, sizes, args.seed, args.max_eval_group_size)
        for r in rows:
            r["split"] = assignment[r["doc_id"]]
        per_language[lang] = rows
        manifest += [{k: r.get(k, "") for k in MANIFEST_FIELDS} for r in rows]
        print(f"[{lang}] " + ", ".join(f"{s}={sum(r['split'] == s for r in rows)}" for s in SPLITS), flush=True)

    with (args.corpus_dir / "split_manifest.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(manifest)
    (args.corpus_dir / "split_report.md").write_text(build_report(per_language, sizes, args), encoding="utf-8")
    print(f"Wrote {args.corpus_dir / 'split_manifest.csv'} and split_report.md")


if __name__ == "__main__":
    main()
