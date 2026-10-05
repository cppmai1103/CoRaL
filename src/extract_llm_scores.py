"""Extract the LLM scores of the human-annotated SEA-Rater documents into one JSON file.

Input:
    data/llm_score/<config>.jsonl      one LLM-scored document per line: id, parse_status, error, the 5
                                       dimension scores (int 0-5, null if parsing failed), justification, text.
                                       Every *.jsonl except names containing "checkpoint" (in-progress files).
    data/rater_dataset/document_table.csv   the human-annotated documents (src/prepare_dataset.py)

Matching: by document id. A handful of table ids are reused by different documents (the <urn:pdid:...>
placeholders); those are told apart by a hash of the text with whitespace runs collapsed (the annotation CSV
cleaning changed whitespace in some texts, e.g. most Vietnamese documents). An LLM record whose id is not in the table at all falls back to that text hash (catches table ids
damaged when the annotation CSVs were cleaned). A matched id whose LLM text differs from the annotated text
(beyond whitespace) is kept but counted in the report. Most annotated documents without an LLM record are <urn:pdid:...> items that
were never part of the LLM-scored pool.
The language comes from the document table, so file names do not matter; a file with no annotated documents
(e.g. a language outside the annotation set) is reported and contributes nothing.

Output (default data/rater_dataset/llm_scores_annotated.json):
    {"dimensions": [...], "source_files": {file: {"records", "matched"}}, "documents": [
        {"doc_id", "language", "text_hash", "llm_source", "llm_parse_status", "llm_text_matches",
         "llm": {dim: int | null}, "human": {dim: float | null}, "split": {dim: "train"/"validation"/"test"/""}}]}
    one entry per annotated document that has an LLM record (documents without one are listed in the report).
And data/rater_dataset/llm_scores_annotated.md: coverage per language and LLM-vs-human agreement (MAE,
Pearson, Spearman) per dimension, on documents where both scores exist; and
data/rater_dataset/llm_scores_annotated_distribution.png: the score distribution of both, per dimension.

Usage:
    python -m src.extract_llm_scores
    python -m src.extract_llm_scores --llm-dir data/llm_score --output data/rater_dataset/llm_scores_annotated.json
"""

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

DIMENSIONS = ["educational_value", "reasoning", "professionalism", "cleanliness", "cultural_nuances"]


def text_hash(text: str) -> str:
    """Same hash as the document table's text_hash (src/prepare_dataset.py)."""
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def content_hash(text: str) -> str:
    """Whitespace-insensitive hash: the annotation CSV cleaning changed whitespace in some texts (e.g. most
    Vietnamese documents), so 'same document' checks compare texts with whitespace runs collapsed."""
    return hashlib.sha256(" ".join(text.split()).encode("utf-8")).hexdigest()


def load_table(path: Path) -> list[dict]:
    csv.field_size_limit(sys.maxsize)
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def score(raw):
    return float(raw) if raw not in ("", None) else None


def llm_files(llm_dir: Path) -> list[Path]:
    return sorted(p for p in llm_dir.glob("*.jsonl") if "checkpoint" not in p.name)


def match_records(table: list[dict], files: list[Path]) -> tuple[list[dict], dict, dict]:
    """Annotated documents with their LLM record. Returns (documents, per-file stats, per-language stats)."""
    by_id: dict[str, list[dict]] = defaultdict(list)
    by_hash: dict[str, list[dict]] = defaultdict(list)
    for row in table:
        row["_content"] = content_hash(row.get("text") or "")
        by_id[row["doc_id"]].append(row)
        by_hash[row["_content"]].append(row)
    matched: dict[tuple[str, str], dict] = {}  # (doc_id, text_hash) -> output record
    file_stats = {}
    for path in files:
        records = matched_in_file = 0
        with path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                records += 1
                h = content_hash(rec.get("text") or "")
                rows = by_id.get(rec["id"]) or by_hash.get(h)
                if not rows:
                    continue
                row = next((r for r in rows if r["_content"] == h), None)
                if row is None and len(rows) > 1:
                    continue  # reused placeholder id, but none of its documents has this text
                row = row or rows[0]
                key = (row["doc_id"], row["text_hash"])
                if key in matched:
                    continue  # first file wins if a document was scored twice
                matched[key] = {
                    "doc_id": row["doc_id"], "language": row["language"], "text_hash": row["text_hash"],
                    "llm_source": path.name, "llm_parse_status": rec.get("parse_status"),
                    "llm_text_matches": h == row["_content"],
                    "llm": {d: rec.get(d) for d in DIMENSIONS},
                    "human": {d: score(row.get(d)) for d in DIMENSIONS},
                    "split": {d: row.get(f"split_{d}", "") for d in DIMENSIONS},
                }
                matched_in_file += 1
        file_stats[path.name] = {"records": records, "matched": matched_in_file}

    documents = sorted(matched.values(), key=lambda r: (r["language"], r["doc_id"], r["text_hash"]))
    lang_stats = {}
    for language in sorted({r["language"] for r in table}):
        rows = [r for r in table if r["language"] == language]
        docs = [d for d in documents if d["language"] == language]
        have = {(d["doc_id"], d["text_hash"]) for d in docs}
        lang_stats[language] = {
            "annotated": len(rows), "with_llm": len(docs),
            "parse_failed": sum(d["llm_parse_status"] != "ok" for d in docs),
            "text_mismatch": sum(not d["llm_text_matches"] for d in docs),
            "missing_ids": sorted(r["doc_id"] for r in rows if (r["doc_id"], r["text_hash"]) not in have),
        }
    return documents, file_stats, lang_stats


def _ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1  # average rank for ties
        i = j + 1
    return ranks


def pearson(x: list[float], y: list[float]) -> float:
    if len(x) < 2 or statistics.pstdev(x) == 0 or statistics.pstdev(y) == 0:
        return float("nan")
    mx, my = statistics.fmean(x), statistics.fmean(y)
    cov = sum((a - mx) * (b - my) for a, b in zip(x, y))
    return cov / math.sqrt(sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y))


def agreement(documents: list[dict], dim: str, language: str | None = None) -> dict:
    pairs = [(d["llm"][dim], d["human"][dim]) for d in documents
             if (language is None or d["language"] == language)
             and d["llm"][dim] is not None and d["human"][dim] is not None]
    if not pairs:
        return {"n": 0, "mae": float("nan"), "pearson": float("nan"), "spearman": float("nan"),
                "llm_mean": float("nan"), "human_mean": float("nan")}
    llm, human = [float(a) for a, _ in pairs], [b for _, b in pairs]
    return {"n": len(pairs), "mae": statistics.fmean(abs(a - b) for a, b in pairs),
            "pearson": pearson(llm, human), "spearman": pearson(_ranks(llm), _ranks(human)),
            "llm_mean": statistics.fmean(llm), "human_mean": statistics.fmean(human)}


def write_report(path: Path, documents: list[dict], file_stats: dict, lang_stats: dict) -> None:
    languages = list(lang_stats)
    lines = ["# LLM scores of the human-annotated documents", "",
             "Matched by document id (text hash for reused placeholder ids). Agreement uses documents where both the "
             "LLM score and the human mean are present; LLM scores are integers 0-5, human scores are the mean of two "
             "annotators.", "", "## Source files", "", "| File | LLM records | Annotated documents matched |",
             "| --- | ---: | ---: |"]
    for name, s in file_stats.items():
        lines.append(f"| {name} | {s['records']:,} | {s['matched']:,} |")
    lines += ["", "## Coverage per language", "",
              "| Language | Annotated | With LLM record | Parse failed | LLM text differs | Missing |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for language, s in lang_stats.items():
        lines.append(f"| {language} | {s['annotated']} | {s['with_llm']} | {s['parse_failed']} | "
                     f"{s['text_mismatch']} | {len(s['missing_ids'])} |")
    lines += ["", "## LLM vs human agreement (all languages)", "",
              "| Dimension | n | MAE | Pearson | Spearman | LLM mean | Human mean |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for dim in DIMENSIONS:
        a = agreement(documents, dim)
        lines.append(f"| {dim} | {a['n']} | {a['mae']:.3f} | {a['pearson']:.3f} | {a['spearman']:.3f} | "
                     f"{a['llm_mean']:.2f} | {a['human_mean']:.2f} |")
    lines += ["", "## Spearman correlation per language", "", "| Dimension | " + " | ".join(languages) + " |",
              "| --- |" + " ---: |" * len(languages)]
    for dim in DIMENSIONS:
        cells = []
        for language in languages:
            a = agreement(documents, dim, language)
            cells.append(f"{a['spearman']:.3f} (n={a['n']})" if a["n"] else "–")
        lines.append(f"| {dim} | " + " | ".join(cells) + " |")
    missing = {l: s["missing_ids"] for l, s in lang_stats.items() if s["missing_ids"]}
    if missing:
        lines += ["", "## Annotated documents without an LLM record", ""]
        for language, ids in missing.items():
            shown = ", ".join(ids[:10]) + (f", ... ({len(ids)} total)" if len(ids) > 10 else "")
            lines.append(f"- {language}: {shown}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


HUMAN_COLOR, LLM_COLOR = "#2a78d6", "#eb6834"  # categorical slots 1 and 2 of the dataviz reference palette


def plot_distributions(path: Path, documents: list[dict]) -> None:
    """One panel per dimension: share of documents at each score (0-5 in 0.5 steps), human mean vs LLM score,
    on documents with both. Human means can fall on .5 (two annotators); LLM scores are integers."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, grid = "#0b0b0b", "#52514e", "#e4e3df"
    positions = [i / 2 for i in range(11)]
    width = 0.2
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.2), sharey=True)
    for ax, dim in zip(axes.flat, DIMENSIONS):
        pairs = [(d["human"][dim], d["llm"][dim]) for d in documents
                 if d["human"][dim] is not None and d["llm"][dim] is not None]
        n = len(pairs)
        human = [round(h * 2) / 2 for h, _ in pairs]
        llm = [float(l) for _, l in pairs]
        for values, offset, color, label in [(human, -width / 2 - 0.01, HUMAN_COLOR, "Human (mean of 2 annotators)"),
                                             (llm, width / 2 + 0.01, LLM_COLOR, "LLM")]:
            share = [100 * sum(v == x for v in values) / n for x in positions]
            # half-point scores only exist for humans: centre those bars instead of leaving a gap beside them
            xs = [x + (offset if x == int(x) else 0) for x in positions]
            ax.bar(xs, share, width=width, color=color, label=label, zorder=3)
        h_mean, l_mean = statistics.fmean(h for h, _ in pairs), statistics.fmean(llm)
        for mean, color, style in [(h_mean, HUMAN_COLOR, "-"), (l_mean, LLM_COLOR, "--")]:
            ax.axvline(mean, color=color, linestyle=style, linewidth=1.5, zorder=4)
        ax.set_title(dim.replace("_", " ").capitalize(), fontsize=12, color=ink, loc="left", fontweight="bold", pad=20)
        ax.text(0.0, 1.02, f"mean: human {h_mean:.2f} · LLM {l_mean:.2f}   (n = {n:,})", transform=ax.transAxes,
                fontsize=9, color=muted, va="bottom", ha="left")
        ax.set_xticks(range(6))
        ax.set_xlim(-0.4, 5.4)
        ax.grid(axis="y", color=grid, linewidth=0.8, zorder=0)
        ax.tick_params(colors=muted, labelsize=9, length=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(grid)
        ax.set_xlabel("score", fontsize=9, color=muted)
    for ax in axes[:, 0]:
        ax.set_ylabel("% of documents", fontsize=9, color=muted)
    legend_ax = axes.flat[-1]
    legend_ax.axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    handles += [plt.Line2D([], [], color=HUMAN_COLOR, linewidth=1.5), plt.Line2D([], [], color=LLM_COLOR, linewidth=1.5,
                                                                                   linestyle="--")]
    labels += ["Human mean", "LLM mean"]
    legend_ax.legend(handles, labels, loc="center left", frameon=False, fontsize=11, labelcolor=ink)
    legend_ax.text(0.0, 0.18, "Human-annotated documents with both scores.\nHuman scores fall on 0.5 steps (mean of two\n"
                   "annotators); LLM scores are whole numbers.", transform=legend_ax.transAxes, fontsize=9, color=muted,
                   va="top")
    fig.suptitle("Score distribution: human annotation vs LLM, per dimension", fontsize=14, color=ink, x=0.01,
                 ha="left", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--llm-dir", type=Path, default=Path("data/llm_score"))
    parser.add_argument("--table", type=Path, default=Path("data/rater_dataset/document_table.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/rater_dataset/llm_scores_annotated.json"))
    args = parser.parse_args(argv)

    table = load_table(args.table)
    files = llm_files(args.llm_dir)
    if not files:
        raise SystemExit(f"No *.jsonl files in {args.llm_dir}")
    documents, file_stats, lang_stats = match_records(table, files)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"dimensions": DIMENSIONS, "source_files": file_stats, "documents": documents},
                                      ensure_ascii=False, indent=1), encoding="utf-8")
    report = args.output.with_suffix(".md")
    write_report(report, documents, file_stats, lang_stats)
    figure = args.output.with_name(args.output.stem + "_distribution.png")
    plot_distributions(figure, documents)

    for name, s in file_stats.items():
        note = "" if s["matched"] else "  (no annotated documents: language not in the annotation set?)"
        print(f"  {name}: {s['records']:,} LLM records, {s['matched']:,} annotated documents matched{note}")
    for language, s in lang_stats.items():
        print(f"  {language:6s} {s['with_llm']:>4}/{s['annotated']} annotated documents have an LLM score "
              f"(parse failed {s['parse_failed']}, text differs {s['text_mismatch']})")
    print(f"Wrote {args.output} ({len(documents):,} documents), {report} and {figure}")


if __name__ == "__main__":
    main()
