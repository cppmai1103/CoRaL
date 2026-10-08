"""Per-language accuracy plots of lm-evaluation-harness runs (eval_lm_harness.py outputs), one line per model.

One PNG per benchmark and one grid of all of them. Default (--style delta): grouped bars, x = language (+ Average),
y = each trained run's accuracy minus the untrained --reference model's, in percentage points (the reference itself is
the zero line). --style lines: absolute accuracy, one line per run, chance level as a gray hairline. Also, plus results_by_language.csv with every plotted value (the table view). SEA-NLI normal and hard are
pooled into one benchmark: per language, their item-weighted accuracy (both scored on the same model; n = normal + hard).

Colors: the reference categorical palette in its fixed order (blue, orange, aqua, yellow, magenta) for the trained
runs, the untrained base model in neutral gray; every run also has its own marker, so identity never rests on color.

    python -m src.train_gpt2_from_scratch.plot_lm_harness \\
        --runs base=checkpoints/lora_cpt/gemma-3-1b-pt/base/lm_eval_all_5shot \\
               random=checkpoints/lora_cpt/gemma-3-1b-pt/random_50M_7languages_ep1_seed42/lm_eval_all_5shot ... \\
        --output-dir checkpoints/lora_cpt/gemma-3-1b-pt/plots
"""

from __future__ import annotations  # also runs on the system Python 3.8 (no conda environment needed)

import argparse
import csv
from pathlib import Path

LANGUAGE_NAMES = {"burmese": "Burmese", "fil": "Filipino", "indo": "Indonesian", "khmer": "Khmer",
                  "malay": "Malay", "thai": "Thai", "vie": "Vietnamese"}
BENCHMARKS = {  # name -> (title, chance level in %)
    "belebele": ("Belebele", 25.0),
    "global_piqa": ("Global PIQA", 50.0),
    "global_mmlu": ("Global-MMLU", 25.0),
    "include": ("INCLUDE", 25.0),
    "sib200": ("SIB-200", 100 / 7),
    "sea_nli": ("SEA-NLI", 100 / 3),
}
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
REFERENCE = "#52514e"   # secondary ink: the untrained base model
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
INK, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"


def load_run(folder: Path) -> dict[str, dict[str, tuple[float, int]]]:
    """benchmark -> language -> (accuracy %, n); SEA-NLI normal + hard pooled per language."""
    with (folder / "summary.csv").open(encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["score"] not in ("", "None")]
    out = {}
    for r in rows:
        bench = "sea_nli" if r["benchmark"].startswith("sea_nli") else r["benchmark"]
        if bench not in BENCHMARKS:
            continue
        correct, n = out.setdefault(bench, {}).get(r["language"], (0.0, 0))
        out[bench][r["language"]] = (correct + float(r["score"]) * int(r["n"]), n + int(r["n"]))
    return {b: {l: (100 * c / n, n) for l, (c, n) in per.items()} for b, per in out.items()}


def style(name: str, i_trained: int, reference: str | None):
    if name == reference:
        return {"color": REFERENCE, "marker": "o", "markerfacecolor": SURFACE, "markeredgecolor": REFERENCE,
                "zorder": 2}
    return {"color": SERIES[i_trained], "marker": MARKERS[1 + i_trained], "markerfacecolor": SERIES[i_trained],
            "markeredgecolor": SURFACE, "zorder": 3}


def draw(ax, bench: str, runs: dict, reference: str | None):
    title, chance = BENCHMARKS[bench]
    languages = [l for l in LANGUAGE_NAMES if any(l in runs[r].get(bench, {}) for r in runs)]
    x = list(range(len(languages)))
    values = []
    i_trained = 0
    for name, data in runs.items():
        per = data.get(bench, {})
        y = [per[l][0] if l in per else float("nan") for l in languages]
        values += [v for v in y if v == v]
        kw = style(name, i_trained, reference)
        if name != reference:
            i_trained += 1
        ax.plot(x, y, linewidth=2, markersize=8, markeredgewidth=1.5, label=name, **kw)
    lo, hi = min(values), max(values)
    # chance level as a hairline, labelled in the right margin; left out (named in the title instead) when it is so
    # far below the data that showing it would flatten the differences between runs
    show_chance = lo - chance <= 1.5 * (hi - lo)
    if show_chance:
        lo, hi = min(lo, chance), max(hi, chance)
    pad = max(1.0, 0.08 * (hi - lo))
    ax.set_ylim(lo - pad, hi + pad)
    if show_chance:
        ax.axhline(chance, color=AXIS, linewidth=1, zorder=1)
        ax.annotate(f"chance\n{chance:.0f}%", (1.0, chance), xycoords=("axes fraction", "data"), xytext=(4, 0),
                    textcoords="offset points", color=MUTED, fontsize=8, va="center", ha="left")
    ax.set_title(title + ("" if show_chance else f"  ·  chance {chance:.0f}%"),
                 color=INK, fontsize=11, loc="left")
    ax.set_xticks(x, [LANGUAGE_NAMES[l] for l in languages], fontsize=9, color=INK)
    ax.set_xlim(-0.4, len(languages) - 0.6)
    ax.set_ylabel("Accuracy (%)", color=INK, fontsize=9)
    ax.tick_params(colors=MUTED, labelcolor=INK, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)


def draw_delta(ax, bench: str, runs: dict, reference: str):
    """Grouped bars: per language, each trained run's accuracy minus the reference (untrained) model's, in percentage
    points, plus an "Average" group (macro over the benchmark's languages) after a hairline divider."""
    title, _ = BENCHMARKS[bench]
    ref = runs[reference].get(bench, {})
    languages = [l for l in LANGUAGE_NAMES if l in ref]
    trained = [r for r in runs if r != reference]
    width = 0.8 / len(trained)
    groups = list(range(len(languages))) + [len(languages) + 0.4]  # Average set apart, after a divider
    values = []
    for i, name in enumerate(trained):
        per = runs[name].get(bench, {})
        deltas = [per[l][0] - ref[l][0] if l in per else float("nan") for l in languages]
        known = [d for d in deltas if d == d]
        deltas.append(sum(known) / len(known) if known else float("nan"))  # Average group
        values += [d for d in deltas if d == d]
        x = [g + (i - (len(trained) - 1) / 2) * width for g in groups]
        ax.bar(x, deltas, width=width, color=SERIES[i], edgecolor=SURFACE, linewidth=1.0, label=name, zorder=3)
    ax.axhline(0, color=AXIS, linewidth=1, zorder=4)  # over the bars, so it reads as one continuous baseline
    ax.axvline(len(languages) - 0.3, color=GRID, linewidth=1, zorder=1)
    lo, hi = min(values + [0]), max(values + [0])
    pad = max(0.5, 0.08 * (hi - lo))
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.set_xticks(groups, [LANGUAGE_NAMES[l] for l in languages] + ["Average"], fontsize=9, color=INK)
    ax.set_xlim(-0.5, len(languages) + 0.9)
    ax.set_ylabel(f"Accuracy − {reference} (pp)", color=INK, fontsize=9)
    ax.tick_params(colors=MUTED, labelcolor=INK, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="+", required=True, metavar="LABEL=FOLDER",
                        help="Legend label and eval_lm_harness output folder, in legend/color order")
    parser.add_argument("--reference", default="base", help="Label drawn in neutral gray (the untrained model)")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--style", choices=["delta", "lines"], default="delta",
                        help="delta: bars of each trained run minus --reference (not drawn itself), per language; "
                             "lines: absolute accuracy, one line per run")
    args = parser.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "sans-serif"

    runs = {}
    for item in args.runs:
        label, _, folder = item.partition("=")
        runs[label] = load_run(Path(folder))
    reference = args.reference if args.reference in runs else None
    if args.style == "delta" and reference is None:
        parser.error(f"--style delta needs the --reference run ({args.reference!r}) among --runs")
    plot = (lambda ax, b: draw_delta(ax, b, runs, reference)) if args.style == "delta" else \
        (lambda ax, b: draw(ax, b, runs, reference))
    benches = [b for b in BENCHMARKS if any(b in r for r in runs.values())]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with (args.output_dir / "results_by_language.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["benchmark", "language", "n"] + list(runs))
        for b in benches:
            for l in LANGUAGE_NAMES:
                cells = [runs[r].get(b, {}).get(l) for r in runs]
                if any(cells):
                    n = next(c[1] for c in cells if c)
                    w.writerow([b, LANGUAGE_NAMES[l], n] + [f"{c[0]:.2f}" if c else "" for c in cells])

    def legend(fig_or_ax, **kw):
        leg = fig_or_ax.legend(frameon=False, fontsize=9, labelcolor=INK, **kw)
        return leg

    for b in benches:
        fig, ax = plt.subplots(figsize=(7.5, 4.2), facecolor=SURFACE)
        plot(ax, b)
        legend(ax, loc="upper left", bbox_to_anchor=(1.08, 1.0))
        fig.tight_layout()
        fig.savefig(args.output_dir / f"{b}.png", dpi=200, facecolor=SURFACE)
        plt.close(fig)

    cols = 3
    rows = (len(benches) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6.2 * cols, 4.0 * rows + 0.6), facecolor=SURFACE, squeeze=False)
    for ax, b in zip(axes.flat, benches):
        plot(ax, b)
    for ax in list(axes.flat)[len(benches):]:
        ax.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False, fontsize=10, labelcolor=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95), w_pad=3.0, h_pad=2.5)
    fig.savefig(args.output_dir / "all_benchmarks.png", dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print(f"Wrote {len(benches)} benchmark plots, all_benchmarks.png and results_by_language.csv to {args.output_dir}")


if __name__ == "__main__":
    main()
