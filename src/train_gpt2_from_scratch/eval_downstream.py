"""Zero-shot downstream evaluation of the GPT-2 pilot checkpoints (docs/03_downstream_evaluation.md).

Three benchmarks, each reduced to the same shape -- {example_id, language, prompt, choices,
gold_index} -- and scored by summed log-probability of the full candidate answer text (no
fine-tuning, no gradient updates, model weights frozen):

  belebele  facebook/belebele   reading comprehension, 4 choices, all 6 project languages
            (fil->tgl_Latn, indo->ind_Latn, khmer->khm_Khmr, malay->zsm_Latn, thai->tha_Thai,
            vie->vie_Latn), 900 questions/language.
  xcopa     cambridgeltl/xcopa  cause/effect reasoning, 2 choices, only 3 languages have this
            dataset (indo->id, thai->th, vie->vi), 500 test examples/language. The premise's
            trailing punctuation is replaced by a fixed, language-specific connector ("karena"/
            "maka" id, "เพราะ"/"จึง" th, "vi"/"nên" vie) per protocol section 4.2 -- these are
            the standard connector translations, not from the dataset itself.
  sib200    Davlan/sib200       topic classification, 7 choices (the label set is already fixed
            English strings even for non-English text, so no label translation is needed), all
            6 project languages, 204 test examples/language.

Per language, --num-samples (default 100) are drawn with a seeded RNG (same convention as the
rest of this project: random.Random(f"{seed}:{language}")) from the official test split.

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
                      "malay": "zsm_Latn", "thai": "tha_Thai", "vie": "vie_Latn"}
XCOPA_LANGUAGES = {"indo": "id", "thai": "th", "vie": "vi"}
SIB200_LANGUAGES = {"fil": "tgl_Latn", "indo": "ind_Latn", "khmer": "khm_Khmr",
                    "malay": "zsm_Latn", "thai": "tha_Thai", "vie": "vie_Latn"}
# "because" / "so", fixed per-language connector translations (not from the dataset itself) --
# see module docstring. Premise's trailing punctuation is replaced by " " + this connector.
XCOPA_CONNECTORS = {"indo": {"cause": "karena", "effect": "maka"},
                    "thai": {"cause": "เพราะ", "effect": "จึง"},
                    "vie": {"cause": "vì", "effect": "nên"}}


def sample_indices(n_total: int, n_sample: int, seed: int, language: str) -> list[int]:
    rng = random.Random(f"{seed}:{language}")
    return sorted(rng.sample(range(n_total), min(n_sample, n_total)))


def load_belebele(language: str, seed: int, num_samples: int) -> list[dict]:
    from huggingface_hub import hf_hub_download

    code = BELEBELE_LANGUAGES[language]
    path = hf_hub_download("facebook/belebele", f"data/{code}.jsonl", repo_type="dataset")
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    out = []
    for i in sample_indices(len(rows), num_samples, seed, language):
        r = rows[i]
        prompt = f"Passage:\n{r['flores_passage']}\n\nQuestion: {r['question']}\nAnswer:"
        choices = [r[f"mc_answer{k}"] for k in range(1, 5)]
        out.append({"example_id": f"belebele-{code}-{r['question_number']}", "language": language,
                   "prompt": prompt, "choices": choices, "gold_index": int(r["correct_answer_num"]) - 1})
    return out


def load_xcopa(language: str, seed: int, num_samples: int) -> list[dict]:
    import pandas as pd
    from huggingface_hub import hf_hub_download

    code = XCOPA_LANGUAGES[language]
    connectors = XCOPA_CONNECTORS[language]
    path = hf_hub_download("cambridgeltl/xcopa", f"{code}/test-00000-of-00001.parquet", repo_type="dataset")
    df = pd.read_parquet(path)
    out = []
    for i in sample_indices(len(df), num_samples, seed, language):
        r = df.iloc[i]
        premise = r["premise"].rstrip(" .!?؟。").rstrip()
        prompt = f"{premise} {connectors[r['question']]}"
        out.append({"example_id": f"xcopa-{code}-{r['idx']}", "language": language, "prompt": prompt,
                   "choices": [r["choice1"], r["choice2"]], "gold_index": int(r["label"])})
    return out


def load_sib200(language: str, seed: int, num_samples: int) -> list[dict]:
    from huggingface_hub import hf_hub_download

    code = SIB200_LANGUAGES[language]
    labels_path = hf_hub_download("Davlan/sib200", f"data/{code}/labels.txt", repo_type="dataset")
    labels = [l.strip() for l in open(labels_path, encoding="utf-8") if l.strip()]
    test_path = hf_hub_download("Davlan/sib200", f"data/{code}/test.tsv", repo_type="dataset")
    with open(test_path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    topics = ", ".join(labels)
    out = []
    for i in sample_indices(len(rows), num_samples, seed, language):
        r = rows[i]
        prompt = f"Topics: {topics}.\n\nText:\n{r['text']}\n\nTopic:"
        out.append({"example_id": f"sib200-{code}-{r['index_id']}", "language": language, "prompt": prompt,
                   "choices": list(labels), "gold_index": labels.index(r["category"])})
    return out


BENCHMARKS = {
    "belebele": {"languages": BELEBELE_LANGUAGES, "loader": load_belebele, "chance": 0.25},
    "xcopa": {"languages": XCOPA_LANGUAGES, "loader": load_xcopa, "chance": 0.50},
    "sib200": {"languages": SIB200_LANGUAGES, "loader": load_sib200, "chance": 1 / 7},
}


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
        scores = []
        for choice in ex["choices"]:
            score, truncated = loglikelihood(model, tokenizer, ex["prompt"], choice, device, use_bf16, seq_len,
                                             start_id)
            scores.append(score)
            truncated_count += truncated
        prediction = max(range(len(scores)), key=lambda j: scores[j])
        records.append({"example_id": ex["example_id"], "language": ex["language"],
                        "candidate_scores": scores, "prediction": prediction,
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
        writer.writerow(["example_id", "language", "prompt", "choices", "candidate_scores",
                         "prediction", "gold_index", "correct"])
        for r in records:
            ex = by_id[r["example_id"]]
            writer.writerow([r["example_id"], r["language"], ex["prompt"], json.dumps(ex["choices"], ensure_ascii=False),
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
                  device, use_bf16: bool, seq_len: int, out_dir: Path, start_id: int | None = None) -> dict:
    spec = BENCHMARKS[benchmark]
    per_language = {}
    for language in languages:
        if language not in spec["languages"]:
            continue
        examples = spec["loader"](language, seed, num_samples)
        records = evaluate_examples(model, tokenizer, examples, device, use_bf16, seq_len,
                                    desc=f"{benchmark}/{language}", start_id=start_id)
        write_predictions(out_dir / benchmark / f"{language}.csv", examples, records)
        per_language[language] = {"n": len(records), "accuracy": accuracy(records),
                                  "majority_baseline": majority_baseline(examples)}
        print(f"  [{benchmark}/{language}] accuracy={per_language[language]['accuracy']:.3f} "
              f"(n={len(records)}, chance={spec['chance']:.3f}, majority={per_language[language]['majority_baseline']:.3f})",
              flush=True)
    macro = statistics.mean(v["accuracy"] for v in per_language.values()) if per_language else float("nan")
    return {"benchmark": benchmark, "chance": spec["chance"], "per_language": per_language, "macro": macro}


def write_summary(path: Path, results: list[dict]) -> None:
    lines = ["# Downstream evaluation summary", ""]
    for r in results:
        lines += [f"## {r['benchmark']} (chance={r['chance']:.3f})", "",
                 "| Language | n | Accuracy | Majority baseline |", "| --- | --- | --- | --- |"]
        for lang, v in r["per_language"].items():
            lines.append(f"| {lang} | {v['n']} | {v['accuracy']:.3f} | {v['majority_baseline']:.3f} |")
        lines += [f"| **macro** | | **{r['macro']:.3f}** | |", ""]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_comparison(path: Path, run_names: list[str], all_results: dict[str, list[dict]]) -> None:
    lines = ["# Downstream evaluation: run comparison", ""]
    for benchmark in BENCHMARKS:
        if not any(any(x["benchmark"] == benchmark for x in all_results[name]) for name in run_names):
            continue  # not run this time (e.g. --benchmarks excluded it) -- skip the section entirely
        lines += [f"## {benchmark}", "", "| Run | " + " | ".join(BENCHMARKS[benchmark]["languages"]) + " | macro |",
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
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=None, help="Default: <run-dir>/downstream_eval")
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

    # (name, tokenizer source, model source, default output dir, is a pretrained Hub/local model)
    entries = [(d.name, d / "tokenizer", d / "final", d / "downstream_eval", False) for d in args.run_dir]
    for m in args.model:
        local = Path(m)
        if local.name == "final":  # a LoRA run's merged model: name and outputs follow the run
            entries.append((local.parent.name, m, m, local.parent / "downstream_eval", True))
        else:
            entries.append((m.rstrip("/").split("/")[-1], m, m,
                            Path("checkpoints/hub_models") / m.rstrip("/").split("/")[-1] / "downstream_eval", True))
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
                                         args.device, use_bf16, seq_len, out_dir, start_id))
        (out_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        write_summary(out_dir / "summary.md", results)
        print(f"Wrote {out_dir / 'results.json'} and summary.md", flush=True)
        all_results[name] = results
        del model
        if args.device != "cpu":
            torch.cuda.empty_cache()

    if len(entries) > 1:
        comparison_file = args.comparison_file or (default_out_root(entries) / "downstream_comparison.md")
        write_comparison(comparison_file, [e[0] for e in entries], all_results)
        print(f"Wrote {comparison_file}", flush=True)


if __name__ == "__main__":
    main()
