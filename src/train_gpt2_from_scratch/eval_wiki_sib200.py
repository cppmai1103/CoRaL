"""Wikipedia and SIB-200 loss/perplexity evaluation of the GPT-2 pilot checkpoints
(docs/04_wikipedia_sib200_evaluation.md).

Unlike eval_downstream.py (zero-shot task accuracy), this measures ordinary, unweighted
next-token loss/perplexity -- the same metric used throughout this project -- on two corpora
independent of the pilot corpus:

  wikipedia  wikimedia/wikipedia  20231101 dump, 1,000 randomly sampled ELIGIBLE articles per
             language (fil->tl, indo->id, khmer->km, malay->ms, thai->th, vie->vi), after
             rejecting empty/too-short/list-navigation-like text (section 2's cleaning criteria;
             see is_eligible_wikipedia_article's docstring for the exact, documented heuristic)
             and excluding any article whose exact text (sha256 of the stripped string, same
             text_hash() as extract_corpus.py) appears anywhere in data/pilot_corpus/<language>.csv
             -- the union of every document any compared model could have trained on.
  sib200     Davlan/sib200        ALL 1,004 sentences/language (train+dev+test pooled, 701+99+204
             per the SIB-200 paper's Table 1), used purely as TEXT -- topic labels only group
             results afterward, never a model target.

Scoring (protocol section 4): one EOS token prepended to context (matching how this GPT-2 was
trained -- documents start right after an EOS), then a sliding window over the content tokens so
every long Wikipedia article is scored fully without truncation: each window re-uses the previous
window's tail tokens purely as unscored context and only the new tokens at the end of each window
are scored as targets, so every content token is counted exactly once regardless of article
length (--stride, default half the model's context window). SIB-200 sentences are short enough
that this reduces to a single window per sentence. Group-level loss is the SUM of per-document
negative log-likelihoods divided by the SUM of per-document valid-target-token counts (never an
average of per-document perplexities -- protocol section 4's explicit warning).

Pragmatic deviation from the protocol text: articles longer than --max-article-chars (default
50,000, well above the p99 length) are truncated before scoring, to bound worst-case runtime
against the rare pathologically long article (one sampled Khmer article was 714,766 characters).

Per --run-dir, writes <run-dir>/wiki_sib200_eval/wikipedia/<language>.csv and
<run-dir>/wiki_sib200_eval/sib200/<language>.csv (one row per example: model_id, training_seed,
dataset, dataset_revision, language, example_id, original_split, topic, source_url, nll_sum,
valid_target_tokens -- protocol section 5's exact per-example schema), plus results.json/
summary.md (loss/PPL per language, per SIB-200 topic, and pooled; loss difference vs. the first
--run-dir, treated as the random baseline). With several --run-dir values, also writes a
side-by-side comparison table.

Bits per byte (summed NLL / ln 2 / UTF-8 bytes of the scored text) is reported next to loss/PPL: loss and PPL
are per TOKEN and only comparable between models with the same tokenizer; bits per byte compares any models.

Pretrained Hub models (--model, e.g. google/gemma-3-270m) are scored the same way, except that the context
starts with the model's own BOS token when it has one (Gemma is trained with <bos> first; our GPT-2 runs with
EOS), and the window is capped at --max-context tokens. Results go to
checkpoints/hub_models/<model name>/wiki_sib200_eval/. Gated models (Gemma) need HF_TOKEN.

Run from the project root (GPU recommended; Wikipedia download is ~1.85GB total across the 6
languages, cached under --cache-dir after the first run). Requires torch, transformers,
huggingface_hub, pyarrow, pandas:
    python -m src.train_gpt2_from_scratch.eval_wiki_sib200 --run-dir checkpoints/gpt2_top_doc/random_20M_ep1_seed42
    python -m src.train_gpt2_from_scratch.eval_wiki_sib200 --run-dir checkpoints/gpt2_top_doc/random_20M_ep1_seed42 \\
        checkpoints/gpt2_top_doc/edu_20M_ep1_seed42 checkpoints/gpt2_top_doc/avg4_20M_ep1_seed42 \\
        checkpoints/gpt2_top_doc/avg5_20M_ep1_seed42 --comparison-file checkpoints/gpt2_top_doc/wiki_sib200_comparison.md
    python -m src.train_gpt2_from_scratch.eval_wiki_sib200 --run-dir checkpoints/gpt2_top_doc/random_20M_ep1_seed42 \
        --model google/gemma-3-270m --comparison-file checkpoints/hub_models/wiki_sib200_comparison.md
"""

import argparse
import csv
import json
import math
import random
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

PROGRESS = True
WIKIPEDIA_REVISION = "20231101"
SIB200_SPLITS = ["train", "dev", "test"]


def progress(iterable, **kwargs):
    if not PROGRESS:
        return iterable
    from tqdm import tqdm
    return tqdm(iterable, **kwargs)


WIKIPEDIA_LANGUAGES = {"fil": "tl", "indo": "id", "khmer": "km", "malay": "ms", "thai": "th", "vie": "vi"}
SIB200_LANGUAGES = {"fil": "tgl_Latn", "indo": "ind_Latn", "khmer": "khm_Khmr",
                    "malay": "zsm_Latn", "thai": "tha_Thai", "vie": "vie_Latn"}


# --------------------------------------------------------------------------
# Wikipedia: load, clean, dedup, sample
# --------------------------------------------------------------------------

def text_hash(text: str) -> str:
    """Identical to extract_corpus.py's text_hash -- same rule, so hashes from
    data/pilot_corpus/<language>.csv directly compare against Wikipedia article text."""
    import hashlib
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def is_eligible_wikipedia_article(text: str, min_chars: int = 200) -> bool:
    """Protocol section 2's cleaning criteria, made concrete and documented here since the
    protocol only gives examples ("such as"), not an exact algorithm:
      - reject empty or near-empty text (fewer than `min_chars` characters after stripping)
      - reject text containing raw HTML tags (<div, <table, <span -- signs of corrupted
        extraction, e.g. this dump's "Main Page" entries that weren't stripped of templates)
      - reject text that is almost entirely short lines (a navigation/list page: >= 8 non-empty
        lines and an average line length under 25 characters, vs. ordinary prose paragraphs)
    """
    stripped = text.strip()
    if len(stripped) < min_chars:
        return False
    if re.search(r"<(div|table|span)[\s>]", stripped, re.IGNORECASE):
        return False
    lines = [l for l in stripped.split("\n") if l.strip()]
    if len(lines) >= 8 and statistics.mean(len(l) for l in lines) < 25:
        return False
    return True


def load_pilot_corpus_hashes(language: str, pilot_corpus_dir: Path) -> set[str]:
    import pandas as pd
    path = pilot_corpus_dir / f"{language}.csv"
    if not path.is_file():
        print(f"Note: {path} not found, skipping the training-overlap exclusion for {language}", flush=True)
        return set()
    return set(pd.read_csv(path, usecols=["text_hash"])["text_hash"])


def load_wikipedia(language: str, seed: int, num_samples: int, pilot_corpus_dir: Path,
                   min_chars: int) -> list[dict]:
    import pandas as pd
    from huggingface_hub import HfApi, hf_hub_download

    code = WIKIPEDIA_LANGUAGES[language]
    config = f"{WIKIPEDIA_REVISION}.{code}"
    api = HfApi()
    files = sorted(f for f in api.list_repo_files("wikimedia/wikipedia", repo_type="dataset")
                  if f.startswith(f"{config}/"))
    frames = [pd.read_parquet(hf_hub_download("wikimedia/wikipedia", f, repo_type="dataset")) for f in files]
    df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]

    exclude = load_pilot_corpus_hashes(language, pilot_corpus_dir)
    eligible = []
    excluded_overlap = 0
    for row in df.itertuples(index=False):
        if not is_eligible_wikipedia_article(row.text, min_chars):
            continue
        h = text_hash(row.text)
        if h in exclude:
            excluded_overlap += 1
            continue
        eligible.append(row)
    if excluded_overlap:
        print(f"  [wikipedia/{language}] excluded {excluded_overlap} article(s) overlapping the pilot "
              f"training corpus", flush=True)

    rng = random.Random(f"{seed}:{language}")
    n = min(num_samples, len(eligible))
    if n < num_samples:
        print(f"  [wikipedia/{language}] only {len(eligible)} eligible articles, short of the requested "
              f"{num_samples}", flush=True)
    sampled = rng.sample(eligible, n)
    return [{"example_id": f"wiki-{code}-{r.id}", "language": language, "text": r.text,
             "original_split": "train", "topic": "", "source_url": r.url} for r in sampled]


# --------------------------------------------------------------------------
# SIB-200: load (all splits pooled, no sampling)
# --------------------------------------------------------------------------

def load_sib200_text(language: str) -> list[dict]:
    from huggingface_hub import hf_hub_download

    code = SIB200_LANGUAGES[language]
    out = []
    for split in SIB200_SPLITS:
        path = hf_hub_download("Davlan/sib200", f"data/{code}/{split}.tsv", repo_type="dataset")
        with open(path, encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                out.append({"example_id": f"sib200-{code}-{row['index_id']}", "language": language,
                           "text": row["text"], "original_split": split, "topic": row["category"],
                           "source_url": ""})
    return out


# --------------------------------------------------------------------------
# Scoring: sliding window so every content token is scored exactly once
# --------------------------------------------------------------------------

def _autocast(device, use_bf16):
    if use_bf16 and torch.device(device).type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return torch.autocast(device_type="cpu", enabled=False)


@torch.no_grad()
def score_document(model, tokenizer, text: str, device, use_bf16: bool, seq_len: int, stride: int,
                   max_chars: int, start_id: int | None = None) -> tuple[float, int]:
    """Summed negative log-likelihood and valid-target-token count for `text`, with one EOS
    prepended as context (matching training) and a sliding window over long content so nothing is
    truncated: each window after the first only scores the tokens beyond what the previous window
    already scored, using the rest purely as context (protocol section 4's "count each scored
    target token once; exclude overlap used only as context"). Returns (nll_sum, valid_tokens)."""
    eos_id = tokenizer.eos_token_id if start_id is None else start_id
    content_ids = tokenizer(text[:max_chars], add_special_tokens=False)["input_ids"]
    ids = [eos_id] + content_ids
    total = len(ids)

    total_nll, total_count, scored_until, begin = 0.0, 0, 0, 0
    while True:
        end = min(begin + seq_len, total)
        window = ids[begin:end]
        input_ids = torch.tensor([window], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)
        with _autocast(device, use_bf16):
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        log_probs = F.log_softmax(logits[0].float(), dim=-1)
        for rel_t in range(1, len(window)):
            abs_t = begin + rel_t
            if abs_t <= scored_until:
                continue
            total_nll += -log_probs[rel_t - 1, window[rel_t]].item()
            total_count += 1
        scored_until = end - 1
        if end == total:
            break
        begin += stride
    return total_nll, total_count


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def write_predictions(path: Path, model_id: str, training_seed, dataset: str, dataset_revision: str,
                      records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model_id", "training_seed", "dataset", "dataset_revision", "language", "example_id",
                         "original_split", "topic", "source_url", "nll_sum", "valid_target_tokens"])
        for r in records:
            writer.writerow([model_id, training_seed, dataset, dataset_revision, r["language"], r["example_id"],
                            r["original_split"], r["topic"], r["source_url"],
                            f"{r['nll_sum']:.6f}", r["valid_target_tokens"]])


def group_loss(records: list[dict]) -> dict:
    """L_G = sum(S_d) / sum(N_d); never average per-document perplexities (protocol section 4)."""
    s = sum(r["nll_sum"] for r in records)
    n = sum(r["valid_target_tokens"] for r in records)
    n_bytes = sum(r.get("text_bytes", 0) for r in records)
    loss = s / n if n else float("nan")
    return {"n_documents": len(records), "valid_target_tokens": n, "loss": loss,
           "perplexity": math.exp(loss) if n else float("nan"), "text_bytes": n_bytes,
           "bits_per_byte": s / math.log(2) / n_bytes if n_bytes else float("nan")}


def run_wikipedia(model, tokenizer, model_id: str, training_seed, languages: list[str], seed: int,
                  num_samples: int, pilot_corpus_dir: Path, min_chars: int, device, use_bf16: bool,
                  seq_len: int, stride: int, max_chars: int, out_dir: Path, start_id: int | None = None) -> dict:
    per_language = {}
    for language in languages:
        examples = load_wikipedia(language, seed, num_samples, pilot_corpus_dir, min_chars)
        records = []
        for ex in progress(examples, desc=f"wikipedia/{language}", unit="doc"):
            nll, n = score_document(model, tokenizer, ex["text"], device, use_bf16, seq_len, stride, max_chars,
                                    start_id)
            records.append({**ex, "nll_sum": nll, "valid_target_tokens": n,
                            "text_bytes": len(ex["text"][:max_chars].encode("utf-8"))})
        write_predictions(out_dir / "wikipedia" / f"{language}.csv", model_id, training_seed, "wikimedia/wikipedia",
                          WIKIPEDIA_REVISION, records)
        per_language[language] = group_loss(records)
        g = per_language[language]
        print(f"  [wikipedia/{language}] articles={g['n_documents']} loss={g['loss']:.4f} "
              f"perplexity={g['perplexity']:.1f} bits/byte={g['bits_per_byte']:.4f}", flush=True)
    macro = statistics.mean(v["loss"] for v in per_language.values()) if per_language else float("nan")
    macro_bpb = statistics.mean(v["bits_per_byte"] for v in per_language.values()) if per_language else float("nan")
    return {"dataset": "wikipedia", "per_language": per_language, "macro_loss": macro, "macro_bits_per_byte": macro_bpb}


def run_sib200(model, tokenizer, model_id: str, training_seed, languages: list[str], device, use_bf16: bool,
              seq_len: int, stride: int, out_dir: Path, start_id: int | None = None) -> dict:
    per_language = {}
    for language in languages:
        examples = load_sib200_text(language)
        records = []
        for ex in progress(examples, desc=f"sib200/{language}", unit="sent"):
            nll, n = score_document(model, tokenizer, ex["text"], device, use_bf16, seq_len, stride, max_chars=10 ** 9,
                                    start_id=start_id)
            records.append({**ex, "nll_sum": nll, "valid_target_tokens": n, "text_bytes": len(ex["text"].encode("utf-8"))})
        write_predictions(out_dir / "sib200" / f"{language}.csv", model_id, training_seed, "Davlan/sib200",
                          "main", records)
        by_topic = defaultdict(list)
        for r in records:
            by_topic[r["topic"]].append(r)
        per_language[language] = {"pooled": group_loss(records),
                                  "by_topic": {t: group_loss(rs) for t, rs in by_topic.items()}}
        g = per_language[language]["pooled"]
        print(f"  [sib200/{language}] sentences={g['n_documents']} loss={g['loss']:.4f} "
              f"perplexity={g['perplexity']:.1f} bits/byte={g['bits_per_byte']:.4f}", flush=True)
    macro = statistics.mean(v["pooled"]["loss"] for v in per_language.values()) if per_language else float("nan")
    macro_bpb = (statistics.mean(v["pooled"]["bits_per_byte"] for v in per_language.values()) if per_language
                 else float("nan"))
    return {"dataset": "sib200", "per_language": per_language, "macro_loss": macro, "macro_bits_per_byte": macro_bpb}


def add_loss_diff(results: list[dict], baseline: list[dict] | None) -> None:
    """In place: adds loss_diff_vs_random to every per-language (and sib200 by_topic) entry."""
    if baseline is None:
        return
    base_by_dataset = {b["dataset"]: b for b in baseline}
    for r in results:
        b = base_by_dataset.get(r["dataset"])
        if b is None:
            continue
        for language, v in r["per_language"].items():
            bv = b["per_language"].get(language)
            if bv is None:
                continue
            if r["dataset"] == "sib200":
                v["pooled"]["loss_diff_vs_random"] = v["pooled"]["loss"] - bv["pooled"]["loss"]
                for topic, tv in v["by_topic"].items():
                    btv = bv["by_topic"].get(topic)
                    if btv is not None:
                        tv["loss_diff_vs_random"] = tv["loss"] - btv["loss"]
            else:
                v["loss_diff_vs_random"] = v["loss"] - bv["loss"]


def write_summary(path: Path, results: list[dict]) -> None:
    lines = ["# Wikipedia / SIB-200 evaluation summary", ""]
    for r in results:
        if r["dataset"] == "wikipedia":
            lines += ["## Wikipedia", "",
                     "| Language | Articles | Valid target tokens | Loss | PPL | Bits/byte | Loss vs. baseline |",
                     "| --- | --- | --- | --- | --- | --- | --- |"]
            for lang, v in r["per_language"].items():
                diff = v.get("loss_diff_vs_random")
                lines.append(f"| {lang} | {v['n_documents']} | {v['valid_target_tokens']:,} | {v['loss']:.4f} | "
                            f"{v['perplexity']:.1f} | {v.get('bits_per_byte', float('nan')):.4f} | "
                            f"{'' if diff is None else f'{diff:+.4f}'} |")
        else:
            lines += ["## SIB-200 (pooled per language)", "",
                     "| Language | Sentences | Valid target tokens | Loss | PPL | Bits/byte | Loss vs. baseline |",
                     "| --- | --- | --- | --- | --- | --- | --- |"]
            for lang, v in r["per_language"].items():
                p = v["pooled"]
                diff = p.get("loss_diff_vs_random")
                lines.append(f"| {lang} | {p['n_documents']} | {p['valid_target_tokens']:,} | {p['loss']:.4f} | "
                            f"{p['perplexity']:.1f} | {p.get('bits_per_byte', float('nan')):.4f} | "
                            f"{'' if diff is None else f'{diff:+.4f}'} |")
            lines += ["", "### SIB-200 by topic", "",
                     "| Language | Topic | Sentences | Loss | PPL |", "| --- | --- | --- | --- | --- |"]
            for lang, v in r["per_language"].items():
                for topic, tv in sorted(v["by_topic"].items()):
                    lines.append(f"| {lang} | {topic} | {tv['n_documents']} | {tv['loss']:.4f} | {tv['perplexity']:.1f} |")
        lines += [f"| **macro** | | | **{r['macro_loss']:.4f}** | | **{r.get('macro_bits_per_byte', float('nan')):.4f}** | |", ""]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_comparison(path: Path, run_names: list[str], all_results: dict[str, list[dict]]) -> None:
    lines = ["# Wikipedia / SIB-200 evaluation: run comparison", "", f"Baseline: `{run_names[0]}`", "",
             "Loss and PPL are per token: compare them only between models with the same tokenizer. Bits per byte "
             "(lower is better) compares any models, e.g. our GPT-2 runs against a pretrained model.", ""]
    metrics = [("loss", "loss", "{:.4f}", "macro_loss"), ("perplexity", "PPL", "{:.1f}", None),
               ("bits_per_byte", "bits per byte", "{:.4f}", "macro_bits_per_byte")]
    for dataset in ("wikipedia", "sib200"):
        if not any(any(x["dataset"] == dataset for x in all_results[name]) for name in run_names):
            continue
        languages = sorted({l for name in run_names for x in all_results[name] if x["dataset"] == dataset
                           for l in x["per_language"]})
        for key, label, fmt, macro_key in metrics:
            lines += [f"## {dataset}: {label}", "", "| Run | " + " | ".join(languages) + " | macro |",
                     "| --- |" + " --- |" * (len(languages) + 1)]
            for name in run_names:
                r = next((x for x in all_results[name] if x["dataset"] == dataset), None)
                if r is None:
                    continue
                cells, values = [], []
                for l in languages:
                    v = r["per_language"].get(l)
                    if v is None:
                        cells.append("N/A")
                    else:
                        value = (v["pooled"] if dataset == "sib200" else v).get(key, float("nan"))
                        values.append(value)
                        cells.append(fmt.format(value))
                macro = r.get(macro_key, float("nan")) if macro_key else math.exp(r["macro_loss"])
                lines.append(f"| {name} | " + " | ".join(cells) + f" | {fmt.format(macro)} |")
            lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def training_seed_of(run_dir: Path) -> str:
    m = re.search(r"seed(\d+)$", run_dir.name)
    return m.group(1) if m else "unknown"


def main():
    global PROGRESS
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, nargs="+", default=[],
                        help="GPT-2 pilot run folders; the first entry (run dir, else model) is the baseline for loss_diff")
    parser.add_argument("--model", nargs="+", default=[],
                        help="Pretrained causal LMs: Hugging Face Hub IDs or local model folders, e.g. google/gemma-3-270m")
    parser.add_argument("--max-context", type=int, default=2048, help="--model only: context window cap")
    parser.add_argument("--datasets", nargs="+", default=["wikipedia", "sib200"], choices=["wikipedia", "sib200"])
    parser.add_argument("--languages", nargs="+", default=None, help="Default: all 6 project languages")
    parser.add_argument("--num-samples", type=int, default=1000, help="Wikipedia articles per language")
    parser.add_argument("--min-chars", type=int, default=200, help="Wikipedia eligibility threshold")
    parser.add_argument("--max-article-chars", type=int, default=50_000,
                        help="Truncate pathologically long Wikipedia articles before scoring (bounds worst-case "
                             "runtime; well above the typical article length)")
    parser.add_argument("--stride", type=int, default=None, help="Sliding-window stride; default seq_len // 2")
    parser.add_argument("--pilot-corpus-dir", type=Path, default=Path("data/pilot_corpus"),
                        help="Used for the Wikipedia/training-overlap exclusion check")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=None, help="Default: <run-dir>/wiki_sib200_eval")
    parser.add_argument("--comparison-file", type=Path, default=None)
    parser.add_argument("--no-bf16", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    PROGRESS = not args.no_progress
    use_bf16 = not args.no_bf16
    if not args.run_dir and not args.model:
        parser.error("give at least one --run-dir or --model")
    languages = args.languages or list(WIKIPEDIA_LANGUAGES)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # (name, tokenizer source, model source, default output dir, training seed, is a pretrained Hub/local model)
    entries = [(d.name, d / "tokenizer", d / "final", d / "wiki_sib200_eval", training_seed_of(d), False)
               for d in args.run_dir]
    for m in args.model:
        local = Path(m)
        if local.name == "final" and local.is_dir():  # a LoRA run's merged model: name and outputs follow the run
            entries.append((local.parent.name, m, m, local.parent / "wiki_sib200_eval", training_seed_of(local.parent), True))
        else:
            short = m.rstrip("/").split("/")[-1]
            entries.append((short, m, m, Path("checkpoints/hub_models") / short / "wiki_sib200_eval", "pretrained", True))
    all_results, baseline = {}, None
    for i, (name, tok_src, model_src, default_out, training_seed, pretrained) in enumerate(entries):
        print(f"=== {name} ===", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(tok_src)
        model = AutoModelForCausalLM.from_pretrained(model_src).to(args.device).eval()
        if pretrained:
            window = getattr(model.config, "max_position_embeddings", None) or getattr(model.config, "n_positions", None)
            seq_len = min(window or args.max_context, args.max_context)
            start_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
            print(f"  pretrained model: context {seq_len} tokens, start token {tokenizer.convert_ids_to_tokens(start_id)!r}, "
                  f"vocab {len(tokenizer):,}", flush=True)
        else:
            seq_len, start_id = model.config.n_positions, None
        stride = args.stride or seq_len // 2
        out_dir = args.out_dir or default_out

        results = []
        if "wikipedia" in args.datasets:
            results.append(run_wikipedia(model, tokenizer, name, training_seed, languages, args.seed,
                                         args.num_samples, args.pilot_corpus_dir, args.min_chars, args.device,
                                         use_bf16, seq_len, stride, args.max_article_chars, out_dir, start_id))
        if "sib200" in args.datasets:
            results.append(run_sib200(model, tokenizer, name, training_seed, languages, args.device,
                                      use_bf16, seq_len, stride, out_dir, start_id))
        add_loss_diff(results, baseline)
        if i == 0:
            baseline = results

        (out_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        write_summary(out_dir / "summary.md", results)
        print(f"Wrote {out_dir / 'results.json'} and summary.md", flush=True)
        all_results[name] = results
        del model
        if args.device != "cpu":
            torch.cuda.empty_cache()

    if len(entries) > 1:
        comparison_file = args.comparison_file or (
            entries[0][3].parent.parent / "wiki_sib200_comparison.md" if not entries[0][5]
            else Path("checkpoints/hub_models/wiki_sib200_comparison.md"))
        write_comparison(comparison_file, [e[0] for e in entries], all_results)
        print(f"Wrote {comparison_file}", flush=True)


if __name__ == "__main__":
    main()
