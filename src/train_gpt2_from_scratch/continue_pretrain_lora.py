"""Continued pretraining of a pretrained LM (default google/gemma-3-270m) with LoRA on a pilot selection.

Same data, packing, loss and evaluation as train_gpt2.py -- only the model differs: instead of a GPT-2 trained from
scratch, a pretrained base model is adapted with LoRA adapters (the base weights stay frozen) on one of the
data/pilot_selected/<method>/documents.csv selections (random_20M, edu_20M, avg4_20M, avg5_20M: the same documents as
the GPT-2 runs). Documents are encoded with the model's own tokenizer as BOS + text + EOS when the model has a BOS
token (Gemma: <bos> ... <eos>, the format it was pretrained on) or text + EOS otherwise, shuffled and packed into --seq-len blocks per language; ordinary
next-token cross-entropy; AdamW over the LoRA parameters only, cosine schedule with warmup; validation every
1/--evals-per-epoch epoch; final validation/test on the pilot corpus held-out splits.

LoRA: rank --lora-r, alpha --lora-alpha, dropout --lora-dropout, on every attention and MLP projection
(q/k/v/o, gate/up/down). Embeddings and the LM head stay frozen, so the tokenizer is unchanged.

--eval-base-only evaluates the pretrained model without any training: the reference every adapted run is compared
against (written to <output-root>/base/).

Outputs in --output-dir (default checkpoints/lora_cpt/<model name>/<method>_ep<epochs>_seed<seed>):
    config.json  results.json  summary.md  train_log.csv  eval_log.csv
    adapter/   the LoRA adapter only (small; load with peft on top of the base model)
    final/     the adapter merged into the base weights, plus the tokenizer -- a plain transformers model, so the
               evaluation scripts take it with --model <run>/final (eval_downstream.py, eval_wiki_sib200.py), and
               eval_test_set.py / score_annotated_dataset.py take the run folder (they read final/ and final/ tokenizer)
    tokenizer/ the tokenizer (as in the GPT-2 run folders)

Loss/perplexity are per model token (Gemma: 262K vocabulary): compare them only with other runs of the same base
model, never with the GPT-2 runs (SeaLLM tokenizer); eval_test_set.py also reports bits per byte, which compares across tokenizers.

Gemma is gated: set HF_TOKEN (sh/lora_cpt.sh reads it from .env).

--budget B (e.g. 10M, SeaLLM tokens with EOS per language, as prepare_data.py counts them) trains on the B part of a
larger selection instead of the whole file: selections made with prepare_data.py --select-by tokens are nested, so
the B selection is the start of the larger one, up to the first group of --group-size documents (or round-robin
batch) after which the running token count reaches B -- exactly the documents prepare_data.py would have selected
for B. Training is then the same as for any selection (all documents shuffled together, cosine schedule), so one
selection at the largest budget serves every budget, each trained as its own run.

Run from the project root (GPU):
    python -m src.train_gpt2_from_scratch.continue_pretrain_lora --eval-base-only
    python -m src.train_gpt2_from_scratch.continue_pretrain_lora --method avg4_20M
    python -m src.train_gpt2_from_scratch.continue_pretrain_lora --method avg5_50M_7languages --budget 10M \\
        --pool-dir data/pilot_corpus_7languages
"""

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

from src.train_gpt2_from_scratch import train_gpt2 as base

DEFAULT_MODEL = "google/gemma-3-270m"
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def short_name(model_id: str) -> str:
    return model_id.rstrip("/").split("/")[-1]


def load_model(args, device):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)  # bf16 via autocast
    if args.eval_base_only:
        return model.to(device)
    from peft import LoraConfig, get_peft_model

    lora = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                      target_modules=LORA_TARGETS, task_type="CAUSAL_LM", bias="none")
    model = get_peft_model(model, lora)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    return model.to(device)


def parse_budget(text: str) -> int:
    t = text.strip().upper()
    scale = {"K": 1_000, "M": 1_000_000, "B": 1_000_000_000}.get(t[-1:], 1)
    return int(float(t[:-1] if scale > 1 else t) * scale)


def budget_label(tokens: int) -> str:
    return f"{tokens // 1_000_000}M" if tokens % 1_000_000 == 0 else str(tokens)


def load_budget_documents(path: Path, budget: int, group_size: int, limit: int = 0) -> tuple[dict[str, list[str]], dict]:
    """The `budget` part of a selection made with prepare_data.py --select-by tokens (documents.csv, in selection
    order, with tokens_with_eos): per language, the documents up to the first group of `group_size` documents (or
    round-robin batch, when the file has a `round` column) after which the running token count reaches `budget`."""
    csv.field_size_limit(sys.maxsize)
    by_lang: dict[str, list[tuple[str, int, str | None]]] = defaultdict(list)
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if "tokens_with_eos" not in reader.fieldnames:
            raise SystemExit(f"{path} has no tokens_with_eos column: make it with prepare_data.py --select-by tokens")
        has_round = "round" in reader.fieldnames
        for row in reader:
            by_lang[row["language"]].append((row["text"], int(row["tokens_with_eos"]), row["round"] if has_round else None))
    docs, info = {}, {}
    for language, rows in by_lang.items():
        cum, end = 0, None
        for i, (_, n, rnd) in enumerate(rows):
            cum += n
            last = i + 1 == len(rows)
            group_end = last or (rows[i + 1][2] != rnd if has_round else (i + 1) % group_size == 0)
            if group_end and cum >= budget:
                end = i + 1
                break
        if end is None:
            raise SystemExit(f"{language}: the selection has {cum:,} tokens, fewer than the {budget:,} budget")
        texts = [r[0] for r in rows[:end]]
        docs[language] = texts[:limit] if limit else texts
        info[language] = {"documents": end, "tokens_with_eos": cum, "documents_in_file": len(rows)}
    return docs, info


def count_parameters(model) -> tuple[int, int]:
    return sum(p.numel() for p in model.parameters()), sum(p.numel() for p in model.parameters() if p.requires_grad)


def write_summary(path: Path, results: dict) -> None:
    cfg = results["config"]
    lines = [f"# {short_name(cfg['model'])} + LoRA continued pretraining: {cfg['method']}", "",
             f"Base model `{cfg['model']}` ({results['parameters'] / 1e6:.0f}M parameters, frozen); LoRA r={cfg['lora_r']}, "
             f"alpha={cfg['lora_alpha']}, dropout={cfg['lora_dropout']} on {', '.join(LORA_TARGETS)} "
             f"({results['trainable_parameters'] / 1e6:.2f}M trainable). {cfg['epochs']} epoch(s), lr {cfg['lr']}, "
             f"seed {cfg['seed']}, {cfg['seq_len']}-token blocks.", ""]
    if "run" in results:
        run, d = results["run"], results["data"]
        lines += [f"- Training tokens ({short_name(cfg['model'])} tokenizer): {d['train_unique_tokens']:,} unique, "
                  f"{run['tokens_consumed']:,} consumed",
                  f"- Steps: {run['steps']}, throughput {run['tokens_per_sec']:,.0f} tokens/s, peak GPU memory "
                  f"{run['peak_mem_gib']:.1f} GiB, wall time {run['wall_seconds'] / 60:.1f} min", ""]
    lines += ["## Validation", ""] + base.eval_table(results["validation"]) + \
             ["", "## Test", ""] + base.eval_table(results["test"]) + \
             ["", "Perplexity = exp(mean next-token loss) per model token: compare only with runs of the same base model.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Pretrained causal LM (Hub ID or local folder)")
    parser.add_argument("--method", default="random_20M", help="Selection under data/pilot_selected/ (output folder name)")
    parser.add_argument("--train-data", type=Path, default=None, help="Default: data/pilot_selected/<method>/documents.csv")
    parser.add_argument("--pool-dir", type=Path, default=Path("data/pilot_corpus"))
    parser.add_argument("--split-manifest", type=Path, default=None, help="Default: <pool-dir>/split_manifest.csv")
    parser.add_argument("--output-root", type=Path, default=Path("checkpoints/lora_cpt"))
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Default: <output-root>/<model name>/<method>_ep<epochs>_seed<seed> (or .../base)")
    parser.add_argument("--budget", default=None,
                        help="Train on this budget's part of the selection (e.g. 10M SeaLLM tokens per language); "
                             "the selection must have been made for a budget at least as large")
    parser.add_argument("--group-size", type=int, default=5,
                        help="--budget: documents per selection group in prepare_data.py (round-robin uses its batches)")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=32, help="Sequences per optimizer step")
    parser.add_argument("--micro-batch-size", type=int, default=4,
                        help="Sequences per forward/backward pass; lower it if the GPU runs out of memory")
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--evals-per-epoch", type=int, default=2)
    parser.add_argument("--no-eval-at-start", dest="eval_at_start", action="store_false")
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--gradient-checkpointing", action="store_true", help="Less memory, ~30%% slower")
    parser.add_argument("--eval-base-only", action="store_true",
                        help="Evaluate the pretrained model on validation/test without training")
    parser.add_argument("--limit-docs", type=int, default=0, help="Debug: keep only N documents per language and set")
    parser.add_argument("--no-bf16", action="store_true")
    parser.add_argument("--no-save-model", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)

    base.PROGRESS = not args.no_progress
    if args.batch_size % args.micro_batch_size:
        raise SystemExit("--batch-size must be a multiple of --micro-batch-size")
    args.train_data = args.train_data or Path("data/pilot_selected") / args.method / "documents.csv"
    budget = parse_budget(args.budget) if args.budget else None
    method_name = f"{args.method}_budget{budget_label(budget)}" if budget else args.method
    run_name = "base" if args.eval_base_only else f"{method_name}_ep{args.epochs}_seed{args.seed}"
    out_dir = args.output_dir or args.output_root / short_name(args.model) / run_name
    manifest = args.split_manifest or args.pool_dir / "split_manifest.csv"
    device, use_bf16 = args.device, not args.no_bf16
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(args.seed)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # the attention mask, not the id, marks padding
    print(f"Model {args.model}: documents encoded as {tokenizer.bos_token or ''} ... {tokenizer.eos_token}, "
          f"vocab {len(tokenizer):,}", flush=True)

    print("Loading and encoding documents ...", flush=True)
    budget_info = None
    if budget:
        train_docs, budget_info = load_budget_documents(args.train_data, budget, args.group_size, args.limit_docs)
        print(f"  budget {budget_label(budget)} of {args.train_data}: "
              + ", ".join(f"{l} {v['documents']:,}/{v['documents_in_file']:,} docs" for l, v in budget_info.items()), flush=True)
    else:
        train_docs = base.load_train_documents(args.train_data, args.limit_docs)
    languages = sorted(train_docs)
    val_docs = base.load_heldout_documents(args.pool_dir, manifest, "validation", args.limit_docs)
    test_docs = base.load_heldout_documents(args.pool_dir, manifest, "test", args.limit_docs)
    pad_id = tokenizer.pad_token_id
    pack = lambda docs: base.pack_streams({l: base.encode_documents(tokenizer, docs[l], bos=True) for l in languages},
                                          languages, args.seq_len, pad_id, args.seed)
    val_blocks, test_blocks = pack(val_docs), pack(test_docs)
    train_blocks, unique_tokens = None, 0
    if not args.eval_base_only:
        train_blocks = pack(train_docs)
        unique_tokens = sum(s["tokens"] for s in train_blocks["stats"].values())
        print(f"  train: {unique_tokens:,} tokens in {train_blocks['input_ids'].size(0)} blocks", flush=True)
    print(f"  validation: {sum(s['tokens'] for s in val_blocks['stats'].values()):,} | test: "
          f"{sum(s['tokens'] for s in test_blocks['stats'].values()):,} tokens", flush=True)

    model = load_model(args, device)
    n_params, n_trainable = count_parameters(model)
    print(f"Model: {n_params / 1e6:.0f}M parameters, {n_trainable / 1e6:.2f}M trainable, on {device}", flush=True)

    import transformers
    config = {**{k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
              "vocab_size": len(tokenizer), "torch": torch.__version__, "transformers": transformers.__version__}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    tokenizer.save_pretrained(out_dir / "tokenizer")
    eval_log = base.CsvLog(out_dir / "eval_log.csv", ["step", "epoch", "tokens_consumed", "split", "language", "loss",
                                                      "perplexity", "target_tokens"])

    results = {"config": config, "parameters": n_params, "trainable_parameters": n_trainable}
    steps, tokens = 0, 0
    if not args.eval_base_only:
        run = base.train(model, train_blocks, val_blocks, languages, args, device, use_bf16, out_dir, eval_log)
        results["run"] = run
        results["data"] = {"train_unique_tokens": unique_tokens, "train": train_blocks["stats"]}
        if budget_info:
            results["data"]["budget"] = {"tokens_per_language": budget, "selection": budget_info}
        steps, tokens = run["steps"], run["tokens_consumed"]

    print("Final evaluation ...", flush=True)
    started = time.time()
    val_result = base.evaluate(model, val_blocks, languages, args.eval_batch_size, device, use_bf16, desc="validation")
    test_result = base.evaluate(model, test_blocks, languages, args.eval_batch_size, device, use_bf16, desc="test")
    base.log_eval(eval_log, val_result, "validation", steps, float(args.epochs if steps else 0), tokens)
    base.log_eval(eval_log, test_result, "test", steps, float(args.epochs if steps else 0), tokens)
    eval_log.close()
    print(f"  validation: {base.format_eval(val_result)}\n  test:       {base.format_eval(test_result)} "
          f"({time.time() - started:.0f}s)", flush=True)
    results.update({"validation": val_result, "test": test_result})
    results.setdefault("data", {}).update({"validation": val_blocks["stats"], "test": test_blocks["stats"]})
    (out_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    write_summary(out_dir / "summary.md", results)

    if not args.no_save_model:
        if not args.eval_base_only:
            model.save_pretrained(out_dir / "adapter")
            model = model.merge_and_unload()
        model.to(torch.bfloat16).save_pretrained(out_dir / "final")
        tokenizer.save_pretrained(out_dir / "final")
    print(f"Wrote results to {out_dir}")


if __name__ == "__main__":
    main()
