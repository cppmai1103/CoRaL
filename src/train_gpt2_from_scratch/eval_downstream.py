"""Zero-shot and few-shot downstream evaluation of the GPT-2 pilot checkpoints (docs/03_downstream_evaluation.md).

Three benchmarks, each reduced to the same shape -- {example_id, language, prompt, choices,
gold_index} -- and scored by summed log-probability of the full candidate answer text (no
fine-tuning, no gradient updates, model weights frozen):

  belebele  facebook/belebele   reading comprehension, 4 choices, all 6 project languages
            (fil->tgl_Latn, indo->ind_Latn, khmer->khm_Khmr, malay->zsm_Latn, thai->tha_Thai,
            vie->vie_Latn, burmese->mya_Mymr), 900 questions/language.
  xcopa     cambridgeltl/xcopa  cause/effect reasoning, 2 choices, only 3 languages have this
            dataset (indo->id, thai->th, vie->vi), 500 test examples/language. The premise's
            trailing punctuation is replaced by a fixed, language-specific connector ("karena"/
            "maka" id, "เพราะ"/"จึง" th, "vi"/"nên" vie) per protocol section 4.2 -- these are
            the standard connector translations, not from the dataset itself.
  sib200    Davlan/sib200       topic classification, 7 choices (the label set is already fixed
            English strings even for non-English text, so no label translation is needed), all
            6 project languages, 204 test examples/language.

Per language, the whole official test split is scored by default (--num-samples 0), as in the SEA LLM
papers. --num-samples N > 0 instead draws N examples with a seeded RNG (same convention as the rest of
this project: random.Random(f"{seed}:{language}")), for quick checks only.

Few-shot (--num-shots K, e.g. 5 or 10; default 0 = zero-shot): K solved examples ("<prompt> <gold answer>",
separated by a blank line) are put before each test prompt. Shots never come from the scored test items:
  belebele  only a test split exists, so shots are other Belebele questions about a DIFFERENT passage
  xcopa     the validation split (100 examples/language)
  sib200    the train split (701 examples/language)
Shots are drawn per test example with random.Random(f"{seed}:{language}:{example_id}:shots") and balanced over
the gold answer position, so every option appears as a correct answer when K >= #options (Belebele 4, XCOPA 2,
SIB-200 7; 5-shot SIB-200 covers 5 distinct topics). They are ordered as one partial round followed by full
rounds of all options (each round shuffled): if a prompt does not fit the context window, whole shots are
dropped from the FRONT, and the shots kept still cover every option as long as a full round fits. The number of
shots actually used is reported per language (GPT-2 has a 1024-token window, so long Belebele passages lose
shots at K=10). Outputs go to downstream_eval_<K>shot/ instead of downstream_eval/.

Scoring (protocol section 5-6): combine prompt + " " + candidate, tokenize the whole string and
the prompt alone, take candidate token ids as the suffix after their longest common prefix (not
just len(prompt_ids), since retokenization at the boundary can differ -- section 6 point 2). One
EOS token is prepended to the context, matching how this GPT-2 was trained (documents start right
after an EOS). Sum log-probs over candidate tokens only; the primary rule is summed
log-probability (not averaged), per protocol section 5.

Per --run-dir, writes <run-dir>/downstream_eval/<benchmark>/<language>.csv (one row per example:
example_id, language, prompt, choices, candidate_scores, prediction, gold_index, correct) and
summary.json/summary.md (accuracy per language, macro average, chance/majority baselines). With
several --run-dir values, also writes a side-by-side comparison table.

Pretrained Hub models (--model, e.g. google/gemma-3-270m) are scored the same way, except that the context
starts with the model's own BOS token when it has one (Gemma is trained with <bos> at the start of every
sequence; our GPT-2 runs with EOS, see above), and the context window is capped at --max-context tokens.
Results go to checkpoints/hub_models/<model name>/downstream_eval/. Gated models (Gemma) need HF_TOKEN.
Hub models and run dirs can be mixed in one call; the comparison file lists them side by side.

Run from the project root (GPU recommended; the GPT-2 checkpoints are small so CPU also works for
a quick check). Requires torch, transformers, huggingface_hub, pyarrow (for XCOPA's parquet):
    python -m src.train_gpt2_from_scratch.eval_downstream --run-dir checkpoints/gpt2_top_doc/random_20M_ep1_seed42
    python -m src.train_gpt2_from_scratch.eval_downstream --run-dir checkpoints/gpt2_top_doc/random_20M_ep1_seed42 \\
        checkpoints/gpt2_top_doc/edu_20M_ep1_seed42 checkpoints/gpt2_top_doc/avg4_20M_ep1_seed42 \\
        checkpoints/gpt2_top_doc/avg5_20M_ep1_seed42 --comparison-file checkpoints/gpt2_top_doc/downstream_comparison.md
    python -m src.train_gpt2_from_scratch.eval_downstream --model google/gemma-3-270m \
        --run-dir checkpoints/gpt2_top_doc/random_20M_ep1_seed42 --comparison-file checkpoints/hub_models/downstream_comparison.md
"""

import argparse
import csv
import json
import math
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

PROGRESS = True


def progress(iterable, **kwargs):
    if not PROGRESS:
        return iterable
    from tqdm import tqdm
    return tqdm(iterable, **kwargs)


# --------------------------------------------------------------------------
# Dataset adapters: each returns a list of {example_id, language, prompt, choices, gold_index}
# --------------------------------------------------------------------------

BELEBELE_LANGUAGES = {"fil": "tgl_Latn", "indo": "ind_Latn", "khmer": "khm_Khmr",
                      "malay": "zsm_Latn", "thai": "tha_Thai", "vie": "vie_Latn", "burmese": "mya_Mymr"}
XCOPA_LANGUAGES = {"indo": "id", "thai": "th", "vie": "vi"}
SIB200_LANGUAGES = {"fil": "tgl_Latn", "indo": "ind_Latn", "khmer": "khm_Khmr",
                    "malay": "zsm_Latn", "thai": "tha_Thai", "vie": "vie_Latn", "burmese": "mya_Mymr"}
# "because" / "so", fixed per-language connector translations (not from the dataset itself) --
# see module docstring. Premise's trailing punctuation is replaced by " " + this connector.
XCOPA_CONNECTORS = {"indo": {"cause": "karena", "effect": "maka"},
                    "thai": {"cause": "เพราะ", "effect": "จึง"},
                    "vie": {"cause": "vì", "effect": "nên"}}


def sample_indices(n_total: int, n_sample: int, seed: int, language: str) -> list[int]:
    if n_sample <= 0 or n_sample >= n_total:
        return list(range(n_total))  # full test split
    rng = random.Random(f"{seed}:{language}")
    return sorted(rng.sample(range(n_total), min(n_sample, n_total)))


def belebele_examples(language: str) -> list[dict]:
    from huggingface_hub import hf_hub_download

    code = BELEBELE_LANGUAGES[language]
    path = hf_hub_download("facebook/belebele", f"data/{code}.jsonl", repo_type="dataset")
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    out = []
    for i, r in enumerate(rows):
        prompt = f"Passage:\n{r['flores_passage']}\n\nQuestion: {r['question']}\nAnswer:"
        choices = [r[f"mc_answer{k}"] for k in range(1, 5)]
        # question_number is only 1/2 within a passage, so the row index makes the id unique
        out.append({"example_id": f"belebele-{code}-{i}-q{r['question_number']}", "language": language,
                   "prompt": prompt, "choices": choices, "gold_index": int(r["correct_answer_num"]) - 1,
                   "group": r["flores_passage"]})  # questions about the same passage share a group
    return out


def xcopa_examples(language: str, split: str = "test") -> list[dict]:
    import pandas as pd
    from huggingface_hub import hf_hub_download

    code = XCOPA_LANGUAGES[language]
    connectors = XCOPA_CONNECTORS[language]
    path = hf_hub_download("cambridgeltl/xcopa", f"{code}/{split}-00000-of-00001.parquet", repo_type="dataset")
    out = []
    for r in pd.read_parquet(path).itertuples(index=False):
        premise = r.premise.rstrip(" .!?؟。").rstrip()
        example_id = f"xcopa-{code}-{r.idx}" if split == "test" else f"xcopa-{code}-{split}-{r.idx}"
        out.append({"example_id": example_id, "language": language, "prompt": f"{premise} {connectors[r.question]}",
                   "choices": [r.choice1, r.choice2], "gold_index": int(r.label), "group": example_id})
    return out


def sib200_examples(language: str, split: str = "test") -> list[dict]:
    from huggingface_hub import hf_hub_download

    code = SIB200_LANGUAGES[language]
    labels_path = hf_hub_download("Davlan/sib200", f"data/{code}/labels.txt", repo_type="dataset")
    labels = [l.strip() for l in open(labels_path, encoding="utf-8") if l.strip()]
    path = hf_hub_download("Davlan/sib200", f"data/{code}/{split}.tsv", repo_type="dataset")
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    topics = ", ".join(labels)
    out = []
    for r in rows:
        example_id = f"sib200-{code}-{r['index_id']}"
        out.append({"example_id": example_id, "language": language,
                   "prompt": f"Topics: {topics}.\n\nText:\n{r['text']}\n\nTopic:",
                   "choices": list(labels), "gold_index": labels.index(r["category"]), "group": example_id})
    return out


def sampled(examples: list[dict], seed: int, num_samples: int, language: str) -> list[dict]:
    return [examples[i] for i in sample_indices(len(examples), num_samples, seed, language)]


def load_belebele(language: str, seed: int, num_samples: int) -> list[dict]:
    return sampled(belebele_examples(language), seed, num_samples, language)


def load_xcopa(language: str, seed: int, num_samples: int) -> list[dict]:
    return sampled(xcopa_examples(language), seed, num_samples, language)


def load_sib200(language: str, seed: int, num_samples: int) -> list[dict]:
    return sampled(sib200_examples(language), seed, num_samples, language)


# "shot_pool": where few-shot examples come from (never the scored test items, see the module docstring).
BENCHMARKS = {
    "belebele": {"languages": BELEBELE_LANGUAGES, "loader": load_belebele, "chance": 0.25,
                 "shot_pool": belebele_examples},  # test split, other passages only (excluded by "group")
    "xcopa": {"languages": XCOPA_LANGUAGES, "loader": load_xcopa, "chance": 0.50,
              "shot_pool": lambda language: xcopa_examples(language, "validation")},
    "sib200": {"languages": SIB200_LANGUAGES, "loader": load_sib200, "chance": 1 / 7,
               "shot_pool": lambda language: sib200_examples(language, "train")},
}
SHOT_SEPARATOR = "\n\n"


def select_shots(pool: list[dict], example: dict, num_shots: int, seed: int) -> list[dict]:
    """num_shots solved examples for one test example, balanced over the gold answer position.

    Order: a partial round of distinct options first, then full rounds of every option, each round shuffled --
    so dropping shots from the front (context overflow) keeps every option covered while one full round
    remains. Shots sharing the example's group (same Belebele passage, or the example itself) are excluded."""
    rng = random.Random(f"{seed}:{example['language']}:{example['example_id']}:shots")
    by_label: dict[int, list[dict]] = defaultdict(list)
    for p in pool:
        if p["group"] != example["group"]:
            by_label[p["gold_index"]].append(p)
    labels = sorted(by_label)
    for label in labels:
        rng.shuffle(by_label[label])
    full_rounds, rest = divmod(num_shots, len(labels))
    rounds = ([rng.sample(labels, rest)] if rest else []) + [rng.sample(labels, len(labels)) for _ in range(full_rounds)]
    return [by_label[label].pop() for round_ in rounds for label in round_ if by_label[label]]


def attach_shots(examples: list[dict], pool: list[dict], num_shots: int, seed: int) -> None:
    for ex in examples:
        ex["shots"] = [f"{s['prompt']} {s['choices'][s['gold_index']]}" for s in select_shots(pool, ex, num_shots, seed)]


def fit_prompt(tokenizer, example: dict, seq_len: int) -> tuple[str, int]:
    """Test prompt preceded by as many of its shots as fit (dropping whole shots from the front) next to the
    longest candidate in seq_len - 1 tokens (1 for the start token). Returns (prompt, number of shots used)."""
    shots = example.get("shots", [])
    for start in range(len(shots) + 1):
        prompt = SHOT_SEPARATOR.join(shots[start:] + [example["prompt"]])
        longest = max(len(tokenizer(prompt + " " + c, add_special_tokens=False)["input_ids"]) for c in example["choices"])
        if longest <= seq_len - 1 or start == len(shots):
            return prompt, len(shots) - start
    raise AssertionError("unreachable")


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------

def common_prefix_len(a: list[int], b: list[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


@torch.no_grad()
def loglikelihood(model, tokenizer, prompt: str, candidate: str, device, use_bf16: bool,
                  seq_len: int, start_id: int | None = None) -> tuple[float, bool]:
    """Summed log-probability of `candidate` as a continuation of `prompt`, with one start token prepended:
    `start_id`, or EOS by default (matching how the GPT-2 runs were trained). Returns (score, truncated)."""
    eos_id = tokenizer.eos_token_id if start_id is None else start_id
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(prompt + " " + candidate, add_special_tokens=False)["input_ids"]
    prefix = common_prefix_len(prompt_ids, full_ids)
    candidate_ids = full_ids[prefix:]
    if not candidate_ids:
        candidate_ids = full_ids[-1:]
        prefix = len(full_ids) - 1

    context_ids = prompt_ids[:prefix]
    truncated = False
    budget = seq_len - 1 - len(candidate_ids)  # -1 for the prepended EOS
    if budget < 1:
        raise ValueError(f"Candidate alone ({len(candidate_ids)} tokens) exceeds the context window")
    if len(context_ids) > budget:
        context_ids = context_ids[-budget:]  # keep the END of the prompt (closest to the answer)
        truncated = True

    input_ids = torch.tensor([[eos_id] + context_ids + candidate_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    with _autocast(device, use_bf16):
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    log_probs = F.log_softmax(logits[0].float(), dim=-1)
    n_candidate = len(candidate_ids)
    target_positions = range(input_ids.size(1) - n_candidate, input_ids.size(1))
    score = sum(log_probs[t - 1, input_ids[0, t]].item() for t in target_positions)
    return score, truncated


def _autocast(device, use_bf16):
    if use_bf16 and torch.device(device).type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return torch.autocast(device_type="cpu", enabled=False)


def evaluate_examples(model, tokenizer, examples: list[dict], device, use_bf16: bool, seq_len: int,
                      desc: str, start_id: int | None = None) -> list[dict]:
    records = []
    truncated_count = 0
    for ex in progress(examples, desc=desc, unit="ex"):
        prompt, n_shots = fit_prompt(tokenizer, ex, seq_len) if ex.get("shots") else (ex["prompt"], 0)
        scores = []
        for choice in ex["choices"]:
            score, truncated = loglikelihood(model, tokenizer, prompt, choice, device, use_bf16, seq_len,
                                             start_id)
            scores.append(score)
            truncated_count += truncated
        prediction = max(range(len(scores)), key=lambda j: scores[j])
        records.append({"example_id": ex["example_id"], "language": ex["language"], "prompt": prompt,
                        "n_shots": n_shots, "candidate_scores": scores, "prediction": prediction,
                        "gold_index": ex["gold_index"], "correct": prediction == ex["gold_index"]})
    if truncated_count:
        print(f"  [{desc}] {truncated_count} candidate(s) required prompt truncation", flush=True)
    return records


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def write_predictions(path: Path, examples: list[dict], records: list[dict]) -> None:
    by_id = {e["example_id"]: e for e in examples}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["example_id", "language", "n_shots", "prompt", "choices", "candidate_scores",
                         "prediction", "gold_index", "correct"])
        for r in records:
            ex = by_id[r["example_id"]]
            writer.writerow([r["example_id"], r["language"], r["n_shots"], r["prompt"],
                            json.dumps(ex["choices"], ensure_ascii=False),
                            json.dumps([round(s, 4) for s in r["candidate_scores"]]),
                            r["prediction"], r["gold_index"], r["correct"]])


def accuracy(records: list[dict]) -> float:
    return sum(r["correct"] for r in records) / len(records) if records else float("nan")


def majority_baseline(examples: list[dict]) -> float:
    counts = defaultdict(int)
    for e in examples:
        counts[e["gold_index"]] += 1
    return max(counts.values()) / len(examples) if examples else float("nan")


def run_benchmark(model, tokenizer, benchmark: str, languages: list[str], seed: int, num_samples: int,
                  device, use_bf16: bool, seq_len: int, out_dir: Path, start_id: int | None = None,
                  num_shots: int = 0) -> dict:
    spec = BENCHMARKS[benchmark]
    per_language = {}
    for language in languages:
        if language not in spec["languages"]:
            continue
        examples = spec["loader"](language, seed, num_samples)
        if num_shots:
            attach_shots(examples, spec["shot_pool"](language), num_shots, seed)
        records = evaluate_examples(model, tokenizer, examples, device, use_bf16, seq_len,
                                    desc=f"{benchmark}/{language}", start_id=start_id)
        write_predictions(out_dir / benchmark / f"{language}.csv", examples, records)
        shots_used = [r["n_shots"] for r in records]
        per_language[language] = {"n": len(records), "accuracy": accuracy(records),
                                  "majority_baseline": majority_baseline(examples),
                                  "mean_shots": statistics.mean(shots_used) if shots_used else 0.0,
                                  "min_shots": min(shots_used, default=0)}
        print(f"  [{benchmark}/{language}] accuracy={per_language[language]['accuracy']:.3f} "
              f"(n={len(records)}, chance={spec['chance']:.3f}, majority={per_language[language]['majority_baseline']:.3f}"
              + (f", shots used mean={per_language[language]['mean_shots']:.2f} min={per_language[language]['min_shots']}"
                 if num_shots else "") + ")", flush=True)
    macro = statistics.mean(v["accuracy"] for v in per_language.values()) if per_language else float("nan")
    return {"benchmark": benchmark, "chance": spec["chance"], "num_shots": num_shots,
            "per_language": per_language, "macro": macro}


def write_summary(path: Path, results: list[dict]) -> None:
    lines = ["# Downstream evaluation summary", ""]
    for r in results:
        k = r.get("num_shots", 0)
        lines += [f"## {r['benchmark']} (chance={r['chance']:.3f}, {k}-shot)", "",
                 "| Language | n | Accuracy | Majority baseline | Shots used (mean / min) |",
                 "| --- | --- | --- | --- | --- |"]
        for lang, v in r["per_language"].items():
            lines.append(f"| {lang} | {v['n']} | {v['accuracy']:.3f} | {v['majority_baseline']:.3f} | "
                         f"{v.get('mean_shots', 0):.2f} / {v.get('min_shots', 0)} |")
        lines += [f"| **macro** | | **{r['macro']:.3f}** | | |", ""]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_comparison(path: Path, run_names: list[str], all_results: dict[str, list[dict]]) -> None:
    lines = ["# Downstream evaluation: run comparison", ""]
    for benchmark in BENCHMARKS:
        if not any(any(x["benchmark"] == benchmark for x in all_results[name]) for name in run_names):
            continue  # not run this time (e.g. --benchmarks excluded it) -- skip the section entirely
        k = next((x.get("num_shots", 0) for name in run_names for x in all_results[name] if x["benchmark"] == benchmark), 0)
        lines += [f"## {benchmark} ({k}-shot)", "", "| Run | " + " | ".join(BENCHMARKS[benchmark]["languages"]) + " | macro |",
                 "| --- |" + " --- |" * (len(BENCHMARKS[benchmark]["languages"]) + 1)]
        for name in run_names:
            r = next((x for x in all_results[name] if x["benchmark"] == benchmark), None)
            if r is None:
                continue
            cells = [f"{r['per_language'].get(l, {}).get('accuracy', float('nan')):.3f}"
                    if l in r["per_language"] else "N/A" for l in BENCHMARKS[benchmark]["languages"]]
            lines.append(f"| {name} | " + " | ".join(cells) + f" | {r['macro']:.3f} |")
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def unique_names(entries: list[tuple]) -> list[tuple]:
    """Entries whose display names collide (e.g. the GPT-2 random_20M run and a LoRA random_20M run, or several
    merged models all called 'final') get the folder above prefixed, so comparison tables keep every row."""
    names = [e[0] for e in entries]
    out = []
    for e in entries:
        name = e[0]
        if names.count(name) > 1:
            src = Path(str(e[2]))
            run = src.parent if src.name == "final" else src
            name = f"{run.parent.name}/{run.name}"
        out.append((name,) + tuple(e[1:]))
    return out

def default_out_root(entries) -> Path:
    """Where the comparison table goes by default: next to the first run dir, or checkpoints/hub_models."""
    return entries[0][3].parent.parent if not entries[0][4] else Path("checkpoints/hub_models")


def main():
    global PROGRESS
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, nargs="+", default=[],
                        help="GPT-2 pilot run folders (final/ + tokenizer/)")
    parser.add_argument("--model", nargs="+", default=[],
                        help="Pretrained causal LMs: Hugging Face Hub IDs or local model folders, e.g. google/gemma-3-270m")
    parser.add_argument("--max-context", type=int, default=2048,
                        help="--model only: context window cap (prompt is truncated from the start beyond it)")
    parser.add_argument("--benchmarks", nargs="+", default=list(BENCHMARKS), choices=list(BENCHMARKS))
    parser.add_argument("--languages", nargs="+", default=None,
                        help="Default: every language available for each benchmark")
    parser.add_argument("--num-samples", type=int, default=0,
                        help="Examples per language; 0 (default) = the full official test split")
    parser.add_argument("--num-shots", type=int, default=0,
                        help="Solved examples before each test prompt (e.g. 5 or 10); 0 (default) = zero-shot")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Default: <run-dir>/downstream_eval (zero-shot) or <run-dir>/downstream_eval_<K>shot")
    parser.add_argument("--comparison-file", type=Path, default=None)
    parser.add_argument("--no-bf16", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    PROGRESS = not args.no_progress
    use_bf16 = not args.no_bf16
    if not args.run_dir and not args.model:
        parser.error("give at least one --run-dir or --model")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if args.num_shots < 0:
        parser.error("--num-shots must be >= 0")
    eval_name = "downstream_eval" if args.num_shots == 0 else f"downstream_eval_{args.num_shots}shot"
    # (name, tokenizer source, model source, default output dir, is a pretrained Hub/local model)
    entries = [(d.name, d / "tokenizer", d / "final", d / eval_name, False) for d in args.run_dir]
    for m in args.model:
        local = Path(m)
        if local.name == "final":  # a LoRA run's merged model: name and outputs follow the run
            entries.append((local.parent.name, m, m, local.parent / eval_name, True))
        else:
            entries.append((m.rstrip("/").split("/")[-1], m, m,
                            Path("checkpoints/hub_models") / m.rstrip("/").split("/")[-1] / eval_name, True))
    entries = unique_names(entries)
    all_results = {}
    for name, tok_src, model_src, default_out, pretrained in entries:
        print(f"=== {name} ===", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(tok_src)
        model = AutoModelForCausalLM.from_pretrained(model_src).to(args.device).eval()
        if pretrained:
            window = getattr(model.config, "max_position_embeddings", None) or getattr(model.config, "n_positions", None)
            seq_len = min(window or args.max_context, args.max_context)
            start_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
            print(f"  pretrained model: context {seq_len} tokens, start token {tokenizer.convert_ids_to_tokens(start_id)!r}, "
                  f"vocab {len(tokenizer):,}, {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M parameters", flush=True)
        else:
            seq_len, start_id = model.config.n_positions, None
        out_dir = args.out_dir or default_out
        results = []
        for benchmark in args.benchmarks:
            languages = args.languages or list(BENCHMARKS[benchmark]["languages"])
            results.append(run_benchmark(model, tokenizer, benchmark, languages, args.seed, args.num_samples,
                                         args.device, use_bf16, seq_len, out_dir, start_id, args.num_shots))
        (out_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        write_summary(out_dir / "summary.md", results)
        print(f"Wrote {out_dir / 'results.json'} and summary.md", flush=True)
        all_results[name] = results
        del model
        if args.device != "cpu":
            torch.cuda.empty_cache()

    if len(entries) > 1:
        suffix = "" if args.num_shots == 0 else f"_{args.num_shots}shot"
        comparison_file = args.comparison_file or (default_out_root(entries) / f"downstream_comparison{suffix}.md")
        write_comparison(comparison_file, [e[0] for e in entries], all_results)
        print(f"Wrote {comparison_file}", flush=True)


if __name__ == "__main__":
    main()
