"""k-shot evaluation with lm-evaluation-harness (docs/03_downstream_evaluation.md, docs/03b_additional_benchmarks.md).

Replaces the hand-written scorers (eval_downstream.py, eval_wiki_sib200.py) for the 7-language runs: harness task
configs, prompts and metrics, on the official evaluation sets. The three largest are cut to fixed subsets (SUBSETS;
--full-test-sets scores all of them in full, --full-test-sets global_mmlu only that one), the same items for every
model and language, written to subsets.json:
  belebele     500 questions per language (of 900, after the holdout)
  global_mmlu  100 questions per subject category (6 categories -> 600 per language, of 14,042), every subject kept
  flores_plus  3 devtest sentences per topic (182 topics -> 520 per language, of 1,012)

  suite  benchmark        harness task(s)                              languages (7 project languages)
  core   belebele         belebele_{lang}                              burmese fil indo khmer malay thai vie
  core   global_piqa      global_piqa_nonparallel_cloze_{lang}         fil indo malay thai vie
  core   global_mmlu      global_mmlu_full_{lang} (57 subjects)        fil indo malay vie
  core   include          include_base_44_{lang}                       fil indo malay vie
  core   xcopa            xcopa_{lang}                                 indo thai vie   (DISABLED for now, see TASKS)
  extra  sib200           sib200_{iso}            (custom, topic)      all 7
  extra  sea_nli_normal   sea_nli_normal_{iso}    (custom, NLI)        all 7 (Khmer: 42 items)
  extra  sea_nli_hard     sea_nli_hard_{iso}      (custom, NLI)        all 7 (Khmer: 23 items)
  extra  flores_plus      flores_plus_eng_{iso}   (custom, eng->lang)  all 7   (DISABLED for now, see TASKS)
The custom task configs are in lm_eval_tasks/ next to this file (--include_path). --benchmarks takes benchmark
names and/or the suites core (default), extra, all.

Global PIQA uses the non-parallel split (the culturally specific questions written by native speakers) in cloze format,
which suits base models.

k-shot (--num-fewshot K, default 5; per benchmark: --shots sib200=0 flores_plus=1 ...). SEA-NLI follows --num-fewshot too
(docs/03b recommends 0-shot; --shots sea_nli_normal=0 sea_nli_hard=0 gives that). Demonstrations never come from scored
items:
  global_mmlu  the dev split (harness default, first_n)
  sib200       the train split (seeded per item with --seed)
  flores_plus  the dev split (seeded per item with --seed); devtest is scored
  xcopa        the validation split (seeded; benchmark disabled for now)
  belebele, global_piqa, include, sea_nli_normal, sea_nli_hard
               only a test split exists, so --holdout (default 5) test items are drawn once with
               random.Random(f"{seed}:{task}") as that task's fixed demonstration pool (the first K are shown, in a fixed
               order) and removed from scoring; INCLUDE holds out once per language for its three domain subtasks.
               The held-out indices are written to holdout.json. With K = 0 nothing is held out and the full test set
               is scored, so a 0-shot and a k-shot run of these benchmarks score slightly different item sets.

Main metric (others are kept in results.json): acc, except global_piqa acc_norm (cloze answers differ in length) and
flores_plus chrF++ and spBLEU, recomputed with sacrebleu from the logged hypotheses (chrF++: word_order=2; spBLEU:
the official flores200 SentencePiece tokenizer). Accuracy benchmarks are averaged per benchmark over their languages;
the overall macro-average covers the core suite only, and FLORES+ is reported in its own table (accuracy, chrF++ and
spBLEU are never averaged together).

Outputs (--output-dir): results.json (harness output, with config and versions), <model>/samples_*.jsonl (every
prompt, per-choice log-likelihood or generation), holdout.json, flores_plus/<iso>.{hyp,ref}.txt, summary.csv and
summary.md, and breakdown.csv/.md: the main metric per value of the dataset fields in BREAKDOWN (Global-MMLU culturally
sensitive/agnostic, subject category and subject; INCLUDE regional feature, domain, level and subject; Global PIQA
cultural score; SIB-200 topic; SEA-NLI concept category and label). Compare several runs (difference from a baseline
run, e.g. random selection), and the breakdown of runs evaluated before it existed:
    python -m src.train_gpt2_from_scratch.eval_lm_harness compare --runs <dir> ... --baseline <dir> --output <md>
    python -m src.train_gpt2_from_scratch.eval_lm_harness breakdown --run-dirs <dir> ...

Run inside .venv/lm-eval (lm-eval on top of the sea-rater environment); see sh/eval/eval_lm_harness.sh.
    python -m src.train_gpt2_from_scratch.eval_lm_harness eval --model <hub id or merged model folder> \\
        --output-dir <dir> [--benchmarks extra] [--num-fewshot 5] [--shots sib200=0]
    python -m src.train_gpt2_from_scratch.eval_lm_harness prefetch --benchmarks all   (login node: download data)
"""

import argparse
import csv
import hashlib
import json
import random
import sys
from pathlib import Path

LANGUAGES = ["burmese", "fil", "indo", "khmer", "malay", "thai", "vie"]
ISO = {"burmese": "mya", "fil": "fil", "indo": "ind", "khmer": "khm", "malay": "zsm", "thai": "tha", "vie": "vie"}
TASK_DIR = Path(__file__).resolve().parent / "lm_eval_tasks"  # custom task configs (SIB-200, SEA-NLI, FLORES+)
# benchmark -> project language -> harness task (a task or a group of subtasks)
TASKS = {
    "belebele": {"burmese": "belebele_mya_Mymr", "fil": "belebele_tgl_Latn", "indo": "belebele_ind_Latn",
                 "khmer": "belebele_khm_Khmr", "malay": "belebele_zsm_Latn", "thai": "belebele_tha_Thai",
                 "vie": "belebele_vie_Latn"},
    "global_piqa": {lang: f"global_piqa_nonparallel_cloze_{code}" for lang, code in
                    [("fil", "tgl_latn"), ("indo", "ind_latn"), ("malay", "zsm_latn"), ("thai", "tha_thai"),
                     ("vie", "vie_latn")]},
    "global_mmlu": {"fil": "global_mmlu_full_fil", "indo": "global_mmlu_full_id", "malay": "global_mmlu_full_ms",
                    "vie": "global_mmlu_full_vi"},
    "include": {"fil": "include_base_44_tagalog", "indo": "include_base_44_indonesian",
                "malay": "include_base_44_malay", "vie": "include_base_44_vietnamese"},
    # XCOPA disabled for now; uncomment to evaluate it again (and add it back to BENCHMARKS in sh/eval/eval_lm_harness.sh)
    # "xcopa": {"indo": "xcopa_id", "thai": "xcopa_th", "vie": "xcopa_vi"},
    "sib200": {lang: f"sib200_{iso}" for lang, iso in ISO.items()},
    "sea_nli_normal": {lang: f"sea_nli_normal_{iso}" for lang, iso in ISO.items()},
    "sea_nli_hard": {lang: f"sea_nli_hard_{iso}" for lang, iso in ISO.items()},
    # FLORES+ disabled for now (generation is slow: hours per model); uncomment to evaluate it again (and add it back
    # to SUITES["extra"])
    # "flores_plus": {lang: f"flores_plus_eng_{iso}" for lang, iso in ISO.items()},
}
SUITES = {"core": ["belebele", "global_piqa", "global_mmlu", "include"],
          "extra": ["sib200", "sea_nli_normal", "sea_nli_hard"]}  # + "flores_plus" (disabled for now, see TASKS)
SUITES["all"] = SUITES["core"] + SUITES["extra"]
CORE = set(SUITES["core"]) | {"xcopa"}  # benchmarks of the overall macro-average
# xcopa stays here while disabled, so `compare` still reads summaries written before (they contain xcopa rows).
# flores_plus is reported twice: chrF++ (row benchmark flores_plus) and spBLEU (row benchmark flores_plus_spbleu).
MAIN_METRIC = {"belebele": "acc", "global_piqa": "acc_norm", "global_mmlu": "acc", "include": "acc", "xcopa": "acc",
               "sib200": "acc", "sea_nli_normal": "acc", "sea_nli_hard": "acc",
               "flores_plus": "chrF++", "flores_plus_spbleu": "spBLEU"}
GENERATION = {"flores_plus", "flores_plus_spbleu"}  # scores on sacrebleu's 0-100 scale, not fractions
# Per-benchmark default k, overriding --num-fewshot (none now: SEA-NLI is run k-shot like the others, with held-out
# demonstrations; docs/03b recommends 0-shot for it, e.g. {"sea_nli_normal": 0, "sea_nli_hard": 0})
DEFAULT_SHOTS = {}
# test-only benchmarks: demonstrations are held out of the test split
HOLDOUT_BENCHMARKS = ("belebele", "global_piqa", "include", "sea_nli_normal", "sea_nli_hard")
# Fixed scored subsets of the largest benchmarks (--full-test-sets scores everything): `size` items per `per` group
# (one group if None), drawn after the holdout and before process_docs. Each group's items are ranked by
# sha256(f"{seed}:{benchmark}:{group}:{key}") and taken in that order, so the selection depends on the data and the
# seed only: the same items for every model, and (the benchmarks are parallel) for every language; a Belebele
# question held out in one language is replaced there by the next one of the order. `cover`: every value of that column
# gets at least one item of its group (Global-MMLU: every subject task keeps a document).
SUBSETS = {
    "belebele": {"size": 500, "per": None, "key": lambda d: f"{d['link']}#{d['question_number']}"},
    "global_mmlu": {"size": 100, "per": "subject_category", "cover": "subject", "key": lambda d: d["sample_id"]},
    "flores_plus": {"size": 3, "per": "topic", "key": lambda d: d["id"]},
}
# Dataset columns the main metric is also reported by (breakdown.csv, breakdown.md), per (benchmark, language), from the
# logged samples. Belebele has no such column.
BREAKDOWN = {
    "global_mmlu": ["cultural_sensitivity_label", "subject_category", "subject"],
    "include": ["regional_feature", "domain", "level", "subject"],
    "global_piqa": ["approx_cultural_score"],
    "sib200": ["category"],
    "sea_nli_normal": ["concept_category", "true_label"],
    "sea_nli_hard": ["concept_category", "true_label"],
}
BREAKDOWN_NOTES = {
    ("global_mmlu", "cultural_sensitivity_label"): "CS = culturally sensitive, CA = culturally agnostic (annotated "
                                                   "subset), - = not annotated",
    ("include", "regional_feature"): "how much the question depends on regional knowledge",
    ("global_piqa", "approx_cultural_score"): "1 = culturally specific (dataset's approximate score), 0 = not",
}


def expand_benchmarks(names: list[str]) -> list[str]:
    return list(dict.fromkeys(b for n in names for b in SUITES.get(n, [n])))


def selected_tasks(benchmarks, languages) -> dict[str, dict[str, str]]:
    return {b: {l: t for l, t in TASKS[b].items() if l in languages} for b in benchmarks}


def configured_task(name: str) -> tuple[str, str] | None:
    """(benchmark, configured task/group) a harness task belongs to. Groups such as INCLUDE (one per-language test set
    split into domain subtasks by process_docs) and Global-MMLU (57 subjects) share the configured group name."""
    for b, per_lang in TASKS.items():
        for configured in per_lang.values():
            if name == configured or name.startswith(configured + "_"):
                return b, configured
    return None


def draw_subset(rows: list[dict], bench: str, seed: int) -> list[int]:
    """Row indices of the fixed SUBSETS[bench] selection (see SUBSETS)."""
    rule = SUBSETS[bench]
    groups = {}
    for i, row in enumerate(rows):
        groups.setdefault(row[rule["per"]] if rule["per"] else "", []).append(i)
    keep = []
    for group in sorted(groups):
        # a seeded hash of each item's own key: an item's rank does not depend on which other items are present
        order = sorted(groups[group], key=lambda i: hashlib.sha256(
            f"{seed}:{bench}:{group}:{rule['key'](rows[i])}".encode()).hexdigest())
        picked = []
        if rule.get("cover"):
            first = {}
            for i in order:
                first.setdefault(rows[i][rule["cover"]], i)
            picked = list(first.values())[:rule["size"]]
        taken = set(picked)
        picked += [i for i in order if i not in taken][:rule["size"] - len(picked)]
        keep += picked
    return sorted(keep)


def make_task_manager(shots: dict[str, int], holdout: int, seed: int, record: dict, subsets: dict, full: set[str]):
    """A TaskManager (with the custom tasks) that, right after loading, sets each task's number of demonstrations and,
    for test-only benchmarks with k > 0, moves `holdout` seeded test items of the task family into its demonstration
    pool (first_n sampler, fixed order) and out of every scored test split. The SUBSETS benchmarks not in `full` are
    then cut to their fixed scored subset (`subsets` is filled with the selected item keys)."""
    from lm_eval.api.samplers import FirstNSampler
    from lm_eval.tasks import TaskManager

    class HoldoutTaskManager(TaskManager):
        def load(self, task_list):
            loaded = super().load(task_list)
            for name, task in loaded["tasks"].items():
                name = getattr(task.config, "task", None) or str(name)
                bench, family = configured_task(name)
                k = shots[bench]
                task.set_config(key="num_fewshot", value=k)
                if k > 0 and bench in HOLDOUT_BENCHMARKS:
                    apply_holdout(task, name, family)
                if bench in SUBSETS and bench not in full:
                    apply_subset(task, name, bench, family)
            return loaded

    selections = {}  # family -> selected rows of its (shared) test set, drawn once for e.g. all 57 Global-MMLU subjects

    def apply_subset(task, name, bench, family):
        split = task.config.test_split
        docs = task.dataset[split]  # after the holdout, before process_docs (Global-MMLU: the per-subject filter)
        if family not in selections:
            keep = draw_subset(list(docs), bench, seed)
            key = SUBSETS[bench]["key"]
            selections[family] = keep
            subsets[family] = {"pool": len(docs), "selected": len(keep), "scored": {},
                               "keys": [key(docs[i]) for i in keep]}
        task.dataset[split] = docs.select(selections[family])
        task.task_docs = task.eval_docs
        subsets[family]["scored"][name] = len(task.eval_docs)

    def apply_holdout(task, name, family):
        split = task.config.test_split
        raw = task.dataset[split]  # before process_docs
        process = task.config.process_docs
        if process is None or family != name:
            # No processing, or a family sharing one test set (INCLUDE domains): hold out of the raw per-language set
            # and keep the demonstrations unfiltered, so every domain subtask gets the same `holdout` of them.
            docs = raw
        else:
            # process_docs selects this task's rows from a shared set (SEA-NLI: one culture): hold out of the result.
            docs = process(raw)
        if len(docs) <= holdout:
            raise SystemExit(f"{name}: only {len(docs)} test items, cannot hold out {holdout}")
        demo_idx = sorted(random.Random(f"{seed}:{family}").sample(range(len(docs)), holdout))
        demo_set = set(demo_idx)
        demos = docs.select(demo_idx)
        rest = docs.select([i for i in range(len(docs)) if i not in demo_set])
        if docs is raw:
            task.dataset[split] = rest  # process_docs (if any, e.g. INCLUDE's domain filter) still applies
        else:
            task.config.process_docs = lambda _dataset, scored=rest: scored
        task.sampler = FirstNSampler(list(demos), rnd=None)
        task.task_docs = task.eval_docs
        record.setdefault(family, {"test_size": len(docs), "demonstration_indices": demo_idx, "scored": {}})
        record[family]["scored"][name] = len(task.eval_docs)

    return HoldoutTaskManager(include_path=str(TASK_DIR))


def leaf_tasks(name: str, group_subtasks: dict) -> list[str]:
    children = group_subtasks.get(name)
    if not children:
        return [name]
    return [leaf for c in children for leaf in leaf_tasks(c, group_subtasks)]


def flores_scores(task_samples: list[dict], out_dir: Path, iso: str) -> tuple[float, float, int]:
    """Corpus chrF++ and spBLEU (flores200 SentencePiece tokenizer) of one direction; also writes the hypotheses and
    references, one sentence per line, in doc order."""
    import sacrebleu

    task_samples = sorted(task_samples, key=lambda s: s["doc_id"])
    hyps = []
    for s in task_samples:
        resp = s["filtered_resps"]
        hyps.append((resp[0] if isinstance(resp, list) else resp).strip())
    refs = [str(s["target"]).strip() for s in task_samples]
    folder = out_dir / "flores_plus"
    folder.mkdir(exist_ok=True)
    (folder / f"{iso}.hyp.txt").write_text("\n".join(h.replace("\n", " ") for h in hyps) + "\n", encoding="utf-8")
    (folder / f"{iso}.ref.txt").write_text("\n".join(refs) + "\n", encoding="utf-8")
    chrfpp = sacrebleu.corpus_chrf(hyps, [refs], word_order=2).score
    spbleu = sacrebleu.corpus_bleu(hyps, [refs], tokenize="flores200").score
    return chrfpp, spbleu, len(hyps)


def summarize(results: dict, samples: dict, tasks: dict[str, dict[str, str]], shots: dict[str, int],
              out_dir: Path) -> list[dict]:
    """One row per (benchmark, language): main metric (and its stderr), number of scored examples, shots."""
    rows = []
    groups = results.get("group_subtasks", {})
    for bench, per_lang in tasks.items():
        for lang, task in per_lang.items():
            if bench == "flores_plus":
                chrfpp, spbleu, n = flores_scores(samples.get(task, []), out_dir, ISO[lang])
                for b, score in (("flores_plus", chrfpp), ("flores_plus_spbleu", spbleu)):
                    rows.append({"benchmark": b, "language": lang, "task": task, "metric": MAIN_METRIC[b],
                                 "score": score, "stderr": None, "n": n, "shots": shots[bench]})
                continue
            metric = MAIN_METRIC[bench]
            r = results["results"].get(task, {})
            n = sum(results["n-samples"].get(leaf, {}).get("effective", 0) for leaf in leaf_tasks(task, groups))
            rows.append({"benchmark": bench, "language": lang, "task": task, "metric": metric,
                         "score": r.get(f"{metric},none"), "stderr": r.get(f"{metric}_stderr,none"), "n": n,
                         "shots": shots[bench]})
    return rows


def macro(rows: list[dict]) -> tuple[dict[str, float], float | None]:
    """Macro-average per benchmark over its languages, and overall over the core benchmarks present."""
    per_bench = {}
    for bench in dict.fromkeys(r["benchmark"] for r in rows):
        scores = [r["score"] for r in rows if r["benchmark"] == bench and isinstance(r["score"], (int, float))]
        if scores:
            per_bench[bench] = sum(scores) / len(scores)
    core = [v for b, v in per_bench.items() if b in CORE]
    overall = sum(core) / len(core) if core else None
    return per_bench, overall


def fmt(x, bench=None):
    if x is None:
        return "N/A"
    return f"{x:.2f}" if bench in GENERATION else f"{100 * x:.2f}"


def table(rows: list[dict], benches: list[str], per_bench: dict) -> list[str]:
    lines = ["| language | " + " | ".join(f"{b} ({MAIN_METRIC[b]})" for b in benches) + " |",
             "|---|" + "---|" * len(benches)]
    for lang in LANGUAGES:
        cells = []
        for b in benches:
            r = next((r for r in rows if r["benchmark"] == b and r["language"] == lang), None)
            cells.append("N/A" if r is None else f"{fmt(r['score'], b)} (n={r['n']:,})")
        if any(c != "N/A" for c in cells):
            lines.append(f"| {lang} | " + " | ".join(cells) + " |")
    lines.append("| **macro** | " + " | ".join(f"**{fmt(per_bench.get(b), b)}**" for b in benches) + " |")
    return lines


def write_summary(out_dir: Path, rows: list[dict], header: str):
    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    per_bench, overall = macro(rows)
    benches = list(dict.fromkeys(r["benchmark"] for r in rows))
    acc = [b for b in benches if b not in GENERATION]
    gen = [b for b in benches if b in GENERATION]
    shots = {r["benchmark"]: r["shots"] for r in rows}
    lines = [header, "", "Demonstrations per benchmark: " + ", ".join(f"{b} {shots[b]}-shot" for b in benches
                                                                       if b != "flores_plus_spbleu"), ""]
    if acc:
        lines += ["Accuracy in % (main metric per benchmark); N/A = benchmark has no data for that language; "
                  "n = scored examples.", ""] + table(rows, acc, per_bench) + [""]
        core = [b for b in acc if b in CORE]
        if core:
            lines += [f"Overall macro-average over the core benchmarks ({', '.join(core)}): **{fmt(overall)}**", ""]
    if gen:
        lines += ["FLORES+ English -> language, devtest (sacrebleu, 0-100; not averaged with accuracy).", ""]
        lines += table(rows, gen, per_bench) + [""]
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines), flush=True)


def load_samples(run_dir: Path) -> dict[str, list[dict]]:
    """Logged samples of an `eval` output folder: harness (leaf) task -> samples, from <model>/samples_<task>_<time>.jsonl."""
    samples = {}
    for path in sorted(run_dir.glob("*/samples_*.jsonl")):
        task = path.stem[len("samples_"):].rsplit("_", 1)[0]
        samples.setdefault(task, []).extend(json.loads(line) for line in path.open(encoding="utf-8"))
    return samples


def breakdown_rows(samples: dict[str, list[dict]]) -> list[dict]:
    """Main metric per value of each BREAKDOWN column, per (benchmark, language), from the logged samples (each one
    carries its dataset row in `doc` and its own metric values)."""
    lang_of = {t: l for per_lang in TASKS.values() for l, t in per_lang.items()}
    groups = {}
    for name, task_samples in samples.items():
        found = configured_task(name)
        if not found or found[0] not in BREAKDOWN:
            continue
        bench, family = found
        for s in task_samples:
            for col in BREAKDOWN[bench]:
                if col in s["doc"]:
                    key = (bench, lang_of[family], col, str(s["doc"][col]))
                    groups.setdefault(key, []).append(float(s[MAIN_METRIC[bench]]))
    # fields in BREAKDOWN order, then languages, then values
    order = lambda g: (g[0][0], BREAKDOWN[g[0][0]].index(g[0][2]), LANGUAGES.index(g[0][1]), g[0][3])
    return [{"benchmark": b, "language": l, "field": c, "value": v, "metric": MAIN_METRIC[b],
             "score": sum(x) / len(x), "n": len(x)} for (b, l, c, v), x in sorted(groups.items(), key=order)]


def write_breakdown(out_dir: Path, rows: list[dict], header: str):
    """breakdown.csv (one row per benchmark, language, field, value) and breakdown.md (a table per benchmark and field:
    values x languages, plus the macro-average over the languages that have the value)."""
    if not rows:
        return
    with (out_dir / "breakdown.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    lines = [header, "", "Main metric in % per value of a dataset field; macro = mean over the languages with that "
             "value; n = scored examples over all languages. Small n: read with care.", ""]
    for bench, col in dict.fromkeys((r["benchmark"], r["field"]) for r in rows):
        sel = [r for r in rows if r["benchmark"] == bench and r["field"] == col]
        langs = [l for l in LANGUAGES if any(r["language"] == l for r in sel)]
        lines += [f"## {bench} by {col} ({MAIN_METRIC[bench]})", ""]
        if (bench, col) in BREAKDOWN_NOTES:
            lines += [BREAKDOWN_NOTES[bench, col], ""]
        lines += ["| value | n | " + " | ".join(langs) + " | macro |", "|---|---|" + "---|" * (len(langs) + 1)]
        for value in dict.fromkeys(r["value"] for r in sel):
            cells = {r["language"]: r for r in sel if r["value"] == value}
            scores = [c["score"] for c in cells.values()]
            lines.append(f"| {value} | {sum(c['n'] for c in cells.values()):,} | "
                         + " | ".join(f"{fmt(cells[l]['score'])} ({cells[l]['n']})" if l in cells else "N/A"
                                      for l in langs) + f" | **{fmt(sum(scores) / len(scores))}** |")
        lines.append("")
    (out_dir / "breakdown.md").write_text("\n".join(lines), encoding="utf-8")


def parse_shots(args) -> dict[str, int]:
    shots = {b: DEFAULT_SHOTS.get(b, args.num_fewshot) for b in TASKS}
    for item in args.shots or []:
        bench, _, k = item.partition("=")
        if bench not in TASKS or not k.isdigit():
            sys.exit(f"--shots expects BENCHMARK=K with a benchmark of {', '.join(TASKS)}; got {item!r}")
        shots[bench] = int(k)
    if args.num_fewshot == 0 and not args.shots:  # --num-fewshot 0: everything 0-shot
        shots = {b: 0 for b in TASKS}
    for b in HOLDOUT_BENCHMARKS:
        if b in TASKS and shots[b] > args.holdout:
            sys.exit(f"{b}: {shots[b]} shots but --holdout {args.holdout}; test-only benchmarks draw their "
                     "demonstrations from the held-out items")
    return shots


def cmd_eval(args):
    import lm_eval
    from lm_eval.loggers import EvaluationTracker

    benchmarks = expand_benchmarks(args.benchmarks)
    shots = parse_shots(args)
    tasks = selected_tasks(benchmarks, args.languages)
    task_names = [t for per_lang in tasks.values() for t in per_lang.values()]
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    record = {}
    subsets = {}
    # --full-test-sets alone: every SUBSETS benchmark in full; with names: only those
    full = set(SUBSETS) if args.full_test_sets == [] else set(args.full_test_sets or [])
    model_args = {"pretrained": args.model, "dtype": args.dtype, "add_bos_token": True}
    print(f"lm-eval {lm_eval.__version__} | model {args.model} | shots "
          + ", ".join(f"{b}={shots[b]}" for b in benchmarks) + f" | holdout {args.holdout} | "
          f"{len(task_names)} tasks: {' '.join(task_names)}", flush=True)
    tracker = EvaluationTracker(output_path=str(out_dir))
    results = lm_eval.simple_evaluate(
        model="hf", model_args=model_args, tasks=task_names, num_fewshot=None,  # set per task by the TaskManager
        batch_size=args.batch_size, max_batch_size=args.max_batch_size, device=args.device, limit=args.limit,
        log_samples=True, evaluation_tracker=tracker,
        task_manager=make_task_manager(shots, args.holdout, args.seed, record, subsets, full),
        random_seed=args.seed, numpy_random_seed=args.seed, torch_random_seed=args.seed,
        fewshot_random_seed=args.seed,
    )
    samples = results.pop("samples")
    tracker.save_results_aggregated(results=results, samples=samples)
    for name in results["configs"]:
        tracker.save_results_samples(task_name=name, samples=samples[name])
    (out_dir / "results.json").write_text(json.dumps(results, indent=2, default=str, ensure_ascii=False),
                                          encoding="utf-8")
    (out_dir / "holdout.json").write_text(json.dumps({"seed": args.seed, "holdout": args.holdout, "tasks": record},
                                                     indent=2), encoding="utf-8")
    if subsets:
        (out_dir / "subsets.json").write_text(json.dumps({"seed": args.seed, "tasks": subsets}, indent=2,
                                                         ensure_ascii=False), encoding="utf-8")
    rows = summarize(results, samples, tasks, shots, out_dir)
    subset_note = ""
    if subsets:
        subset_note = ("\n\nFixed scored subsets (identical for every model; item keys in subsets.json): "
                       + ", ".join(f"{b} {SUBSETS[b]['size']} per {SUBSETS[b]['per'] or 'language'}"
                                   for b in benchmarks if b in SUBSETS and b not in full) + ".")
    if full & set(benchmarks):
        subset_note += f"\n\nComplete test sets: {', '.join(b for b in benchmarks if b in full)}."
    header = (f"# lm-evaluation-harness results: {args.model}"
              + (f" (--limit {args.limit}: NOT a final result)" if args.limit else "") + subset_note)
    write_summary(out_dir, rows, header)
    write_breakdown(out_dir, breakdown_rows(samples), header.replace("results:", "results by subset:", 1))


def cmd_prefetch(args):
    """Download every benchmark dataset (and the spBLEU tokenizer) into the shared caches (login node), so jobs can
    run offline."""
    from lm_eval.tasks import TaskManager

    benchmarks = expand_benchmarks(args.benchmarks)
    tasks = selected_tasks(benchmarks, args.languages)
    names = [t for per_lang in tasks.values() for t in per_lang.values()]
    loaded = TaskManager(include_path=str(TASK_DIR)).load(names)
    if "flores_plus" in benchmarks:
        from sacrebleu.tokenizers.tokenizer_spm import Flores200Tokenizer
        Flores200Tokenizer()  # downloads the flores200 SentencePiece model into ~/.sacrebleu
    print(f"cached {len(loaded['tasks'])} tasks for {len(names)} benchmark-language pairs", flush=True)


def cmd_extract(args):
    """Subset results from a finished --full-test-sets run, without a GPU: every item is scored on its own, so the
    subset score is the main metric over the SUBSETS items of the logged samples (selected exactly as `eval` does,
    from the same post-holdout pool when the run used the same --seed and --holdout). Other benchmarks are copied."""
    import statistics

    src, out_dir = args.run_dir, args.output_dir
    holdout = json.loads((src / "holdout.json").read_text(encoding="utf-8"))
    seed = holdout["seed"]
    results = json.loads((src / "results.json").read_text(encoding="utf-8"))
    with (src / "summary.csv").open(encoding="utf-8") as f:
        old = [r for r in csv.DictReader(f) if r["benchmark"] in TASKS]  # drops xcopa and flores_plus rows (disabled)
    # one samples file per leaf task (Global-MMLU: per subject) -> family -> samples
    by_family = {}
    for task, task_samples in load_samples(src).items():
        found = configured_task(task)
        if found and found[0] in SUBSETS:
            by_family.setdefault(found[1], []).extend(task_samples)
    subsets, rows = {}, []
    for r in old:
        bench, lang = r["benchmark"], r["language"]
        shots = int(r["shots"]) if r.get("shots") else results["n-shot"].get(r["task"], args.num_fewshot)
        if bench not in SUBSETS:
            rows.append({**r, "score": float(r["score"]) if r["score"] not in ("", "None") else None,
                         "stderr": float(r["stderr"]) if r["stderr"] not in ("", "None") else None,
                         "n": int(r["n"]), "shots": shots})
            continue
        if bench == "flores_plus":
            sys.exit("extract: FLORES+ subsets need the per-sentence topic, rerun FLORES+ instead")
        samples = by_family.get(r["task"])
        if not samples:
            sys.exit(f"extract: no logged samples for {r['task']} in {src}")
        keep = draw_subset([s["doc"] for s in samples], bench, seed)
        key = SUBSETS[bench]["key"]
        values = [float(samples[i][MAIN_METRIC[bench]]) for i in keep]
        subsets[r["task"]] = {"pool": len(samples), "selected": len(keep), "scored": {r["task"]: len(keep)},
                              "keys": [key(samples[i]["doc"]) for i in keep]}
        rows.append({"benchmark": bench, "language": lang, "task": r["task"], "metric": MAIN_METRIC[bench],
                     "score": statistics.fmean(values), "stderr": statistics.stdev(values) / len(values) ** 0.5,
                     "n": len(keep), "shots": shots})
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "holdout.json").write_text(json.dumps(holdout, indent=2), encoding="utf-8")
    (out_dir / "subsets.json").write_text(json.dumps({"seed": seed, "extracted_from": str(src), "tasks": subsets},
                                                     indent=2, ensure_ascii=False), encoding="utf-8")
    benches = list(dict.fromkeys(r["benchmark"] for r in rows if r["benchmark"] in SUBSETS))
    write_summary(out_dir, rows, f"# lm-evaluation-harness results: {results.get('config', {}).get('model_args', src)}"
                  f"\n\nExtracted from the full-test-set run {src}. Fixed scored subsets (identical for every model; "
                  "item keys in subsets.json): "
                  + ", ".join(f"{b} {SUBSETS[b]['size']} per {SUBSETS[b]['per'] or 'language'}" for b in benches)
                  + ".")


def cmd_breakdown(args):
    """breakdown.csv/.md of a finished `eval` folder from its logged samples, no GPU (runs from before BREAKDOWN)."""
    for run_dir in args.run_dirs:
        rows = breakdown_rows(load_samples(run_dir))
        if not rows:
            print(f"{run_dir}: no samples of {', '.join(BREAKDOWN)}", flush=True)
            continue
        write_breakdown(run_dir, rows, f"# lm-evaluation-harness results by subset: {run_dir}")
        print(f"{run_dir}: {len(rows)} rows -> breakdown.csv, breakdown.md", flush=True)


def cmd_compare(args):
    # A run is one `eval` output folder, or several comma-separated ones of the same model (e.g. its core and extra
    # suites); the first folder holding a (benchmark, language) wins.
    runs = {}
    for item in args.runs:
        rows, seen = [], set()
        for d in item.split(","):
            with (Path(d) / "summary.csv").open(encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    if (r["benchmark"], r["language"]) in seen:
                        continue
                    seen.add((r["benchmark"], r["language"]))
                    r["score"] = float(r["score"]) if r["score"] not in ("", "None") else None
                    r["n"] = int(r["n"])
                    rows.append(r)
        runs[item] = rows
    # Only benchmarks every run has (e.g. runs evaluated before XCOPA was disabled also have xcopa rows), so the
    # overall macro-average covers the same benchmarks for every run.
    all_benches = list(dict.fromkeys(r["benchmark"] for rows in runs.values() for r in rows))
    benches = [b for b in all_benches if all(any(r["benchmark"] == b for r in rows) for rows in runs.values())]
    dropped = [b for b in all_benches if b not in benches]
    runs = {d: [r for r in rows if r["benchmark"] in benches] for d, rows in runs.items()}
    # --baseline: one of --runs, or one of the folders of a run
    base_dir = next((item for item in runs if args.baseline and (item == args.baseline or args.baseline in
                                                                    item.split(","))), None)
    base_bench, base_overall = macro(runs[base_dir]) if base_dir in runs else ({}, None)
    run_name = lambda d: (p.parent.name if (p := Path(d)).name.startswith("lm_eval") else p.name)
    name = lambda item: run_name(item.split(",")[0]) if item else None
    has_core = any(b in CORE for b in benches)
    lines = ["# lm-evaluation-harness comparison", "",
             "Macro-average per benchmark (accuracy in %, FLORES+ chrF++/spBLEU on 0-100); in brackets the difference "
             f"from {name(base_dir) if base_dir else 'no baseline'}. Overall = core benchmarks only."
             + (f" Left out (not evaluated for every run): {', '.join(dropped)}." if dropped else ""), "",
             "| run | " + " | ".join(benches) + (" | overall |" if has_core else " |"),
             "|---|" + "---|" * (len(benches) + has_core)]
    for d, rows in runs.items():
        per_bench, overall = macro(rows)

        def cell(v, b_val, bench=None):
            if v is None:
                return "N/A"
            diff = ""
            if b_val is not None and d != base_dir:
                scale = 1 if bench in GENERATION else 100
                diff = f" ({scale * (v - b_val):+.2f})"
            return fmt(v, bench) + diff

        lines.append(f"| {name(d)} | " + " | ".join(cell(per_bench.get(b), base_bench.get(b), b) for b in benches)
                     + (f" | {cell(overall, base_overall)} |" if has_core else " |"))
    lines += ["", "## Per language", ""]
    for b in benches:
        langs = [l for l in LANGUAGES if any(r["benchmark"] == b and r["language"] == l for rows in runs.values()
                                             for r in rows)]
        lines += [f"### {b} ({MAIN_METRIC[b]})", "", "| run | " + " | ".join(langs) + " |",
                  "|---|" + "---|" * len(langs)]
        for d, rows in runs.items():
            get = lambda l: next((r["score"] for r in rows if r["benchmark"] == b and r["language"] == l), None)
            lines.append(f"| {name(d)} | " + " | ".join(fmt(get(l), b) for l in langs) + " |")
        lines.append("")
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("eval", "prefetch"):
        p = sub.add_parser(name)
        p.add_argument("--benchmarks", nargs="+", default=["core"], choices=list(TASKS) + list(SUITES),
                       help="Benchmarks and/or suites (core, extra, all)")
        p.add_argument("--languages", nargs="+", default=LANGUAGES, choices=LANGUAGES)
        if name == "eval":
            p.add_argument("--model", required=True, help="Hugging Face Hub ID or local model folder (e.g. <run>/final)")
            p.add_argument("--output-dir", type=Path, required=True)
            p.add_argument("--num-fewshot", type=int, default=5,
                           help="Demonstrations for every benchmark (override per benchmark with --shots); 0 = all 0-shot")
            p.add_argument("--shots", nargs="+", metavar="BENCHMARK=K",
                           help="Per-benchmark override of --num-fewshot, e.g. --shots sib200=0 sea_nli_hard=5")
            p.add_argument("--holdout", type=int, default=5,
                           help="Test items per test-only task held out as its demonstrations (k-shot only)")
            p.add_argument("--seed", type=int, default=1234)
            p.add_argument("--full-test-sets", nargs="*", choices=list(SUBSETS), metavar="BENCHMARK",
                           help="Score complete test sets instead of the fixed SUBSETS (Belebele, Global-MMLU, "
                                "FLORES+): all of them without names, or only the named ones (e.g. global_mmlu)")
            p.add_argument("--batch-size", default="auto")
            p.add_argument("--max-batch-size", type=int, default=64)
            p.add_argument("--dtype", default="bfloat16")
            p.add_argument("--device", default="cuda")
            p.add_argument("--limit", type=float, default=None, help="Debug only: examples per task")
    p = sub.add_parser("extract", help="Subset results (SUBSETS) from a finished --full-test-sets run, no GPU")
    p.add_argument("--run-dir", type=Path, required=True, help="Output folder of a full-test-set `eval`")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--num-fewshot", type=int, default=5, help="Shots column for old summaries that lack it")
    p = sub.add_parser("breakdown", help="Scores by dataset field (BREAKDOWN) of finished `eval` folders, no GPU")
    p.add_argument("--run-dirs", type=Path, nargs="+", required=True)
    p = sub.add_parser("compare")
    p.add_argument("--runs", nargs="+", required=True,
                   help="Output folders of `eval`; several folders of one model comma-separated (core,extra)")
    p.add_argument("--baseline", default=None, help="One of --runs (or one of its folders), e.g. the base model")
    p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    {"eval": cmd_eval, "prefetch": cmd_prefetch, "extract": cmd_extract, "breakdown": cmd_breakdown,
     "compare": cmd_compare}[args.command](args)


if __name__ == "__main__":
    main()
