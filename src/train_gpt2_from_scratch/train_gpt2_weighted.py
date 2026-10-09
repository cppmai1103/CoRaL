"""Train a small GPT-2 from scratch with a quality-weighted training loss (guide 05_loss.md).

This is the same model, data pipeline and evaluation as train_gpt2.py, with one change: every
training token is multiplied by its parent document's importance-score weight before the
token losses are combined, instead of being averaged with weight 1. Held-out validation/test
are still scored with the *ordinary*, unweighted loss (05_loss.md section 13): weighted
training loss is not comparable across runs, only the downstream unweighted perplexity is.

    L_quality = sum_d w_d * sum_{t in T_d} loss(d,t)  /  sum_d w_d * n_d

Document weight
    w_d = weight-offset + score_d * weight-scale, where score_d is the rater's combined
    importance score (already clipped to 0-5 by score_pool.py/combine_scores.py) for one of
    three scoring methods:
        edu    data/pilot_scores/educational_value_mean  (single dimension)
        avg4   data/pilot_scores/avg4_mean                (4-dimension average)
        avg5   data/pilot_scores/avg5_mean                (5-dimension average)
    Defaults (weight-offset=0.5, weight-scale=0.2) reproduce the guide's illustrative
    mapping 0.5 + score/5, giving weight 0.5 at score 0 and 1.5 at score 5.
    --normalize-per-language (on by default) rescales weights so the token-weighted mean
    weight is 1 within each language (guide section 10), so a language with systematically
    higher rater scores does not silently receive more total loss weight than the others.
    Pass --no-normalize-per-language to use the raw 0.5-1.5 mapping instead.

Training data
    Fixed to the already-selected Random 20M-token manifest (data/pilot_selected/random_20M
    /documents.csv) by default, so the three weighted runs (edu/avg4/avg5) use exactly the
    same documents and the same shared initial weights as checkpoints/gpt2_top_doc
    /random_20M_ep1_seed42 -- that unweighted run is the baseline this experiment compares
    against. Only the loss weighting differs.

Run from the project root:
    python -m src.train_gpt2_from_scratch.train_gpt2_weighted --sanity-check --score-method edu
    python -m src.train_gpt2_from_scratch.train_gpt2_weighted --score-method edu --epochs 1
    python -m src.train_gpt2_from_scratch.train_gpt2_weighted --score-method avg4 --epochs 1
    python -m src.train_gpt2_from_scratch.train_gpt2_weighted --score-method avg5 --epochs 1

Outputs in --output-dir (default checkpoints/gpt2_weighted_loss/<score-method>_20M_ep<epochs>_seed<seed>):
    config.json  results.json  summary.md  train_log.csv  eval_log.csv  tokenizer/  final/
Requires torch, transformers and tqdm.
"""

import argparse
import csv
import json
import random
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

from . import train_gpt2 as base

SCORE_DIRS = {
    "edu": Path("data/pilot_scores/educational_value_mean"),
    "avg4": Path("data/pilot_scores/avg4_mean"),
    "avg5": Path("data/pilot_scores/avg5_mean"),
}


# --------------------------------------------------------------------------
# Data: training documents paired with an importance-score weight
# --------------------------------------------------------------------------

def load_train_documents_with_ids(path: Path, limit: int = 0) -> dict[str, list[tuple[str, str]]]:
    """Like train_gpt2.load_train_documents, but keeps doc_id alongside text (needed to look up scores)."""
    csv.field_size_limit(sys.maxsize)
    docs: dict[str, list[tuple[str, str]]] = defaultdict(list)
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            docs[row["language"]].append((row["doc_id"], row["text"]))
    if limit:
        docs = {lang: items[:limit] for lang, items in docs.items()}
    return dict(docs)


def load_importance_scores(scores_dir: Path, languages: list[str]) -> dict[str, dict[str, float]]:
    """doc_id -> combined importance score (score_pool.py/combine_scores.py "score" column, clipped 0-5)."""
    scores: dict[str, dict[str, float]] = {}
    for language in languages:
        path = scores_dir / f"{language}.csv"
        if not path.is_file():
            raise SystemExit(f"No importance scores for '{language}' in {scores_dir}")
        with path.open(encoding="utf-8", newline="") as f:
            scores[language] = {row["doc_id"]: float(row["score"]) for row in csv.DictReader(f)}
    return scores


def score_to_weight(score: float, offset: float, scale: float) -> float:
    return offset + score * scale


def encode_weighted_documents(tokenizer, items: list[tuple[str, str]], scores: dict[str, float], offset: float,
                              scale: float, batch_size: int = 1000) -> list[tuple[list[int], float]]:
    """Token ids (with EOS appended) paired with the document's raw weight u_d, for one language."""
    doc_ids = [doc_id for doc_id, _ in items]
    texts = [text for _, text in items]
    out = []
    for i in range(0, len(texts), batch_size):
        batch_ids = tokenizer(texts[i:i + batch_size], add_special_tokens=False)["input_ids"]
        for doc_id, ids in zip(doc_ids[i:i + batch_size], batch_ids):
            if doc_id not in scores:
                raise SystemExit(f"Training document {doc_id} has no importance score")
            weight = score_to_weight(scores[doc_id], offset, scale)
            out.append((ids + [tokenizer.eos_token_id], weight))
    return out


def normalize_weights_per_language(encoded: dict[str, list[tuple[list[int], float]]]) -> dict[str, float]:
    """Rescale weights in place so the token-weighted mean weight is 1 within each language
    (guide section 10: m_lambda = sum(n_d*u_d)/sum(n_d), w_d = u_d/m_lambda). Returns m_lambda per language."""
    m: dict[str, float] = {}
    for language, docs in encoded.items():
        total_tokens = sum(len(ids) for ids, _ in docs)
        total_weight = sum(len(ids) * w for ids, w in docs)
        m_lang = total_weight / total_tokens if total_tokens else 1.0
        m[language] = m_lang
        encoded[language] = [(ids, w / m_lang) for ids, w in docs]
    return m


def pack_weighted_streams(encoded: dict[str, list[tuple[list[int], float]]], languages: list[str], seq_len: int,
                          pad_id: int, seed: int) -> dict:
    """Like train_gpt2.pack_streams, but every token also carries its document's weight. Padding
    positions get weight 0 (they are already excluded from the loss via the attention mask)."""
    all_ids, all_mask, all_lang, all_weight = [], [], [], []
    stats = {}
    for li, language in enumerate(languages):
        docs = list(encoded.get(language, []))
        random.Random(f"{seed}:{language}").shuffle(docs)
        id_stream = [t for ids, _ in docs for t in ids]
        weight_stream = [w for ids, w in docs for _ in ids]
        for i in range(0, len(id_stream), seq_len):
            b, wb = id_stream[i:i + seq_len], weight_stream[i:i + seq_len]
            pad = seq_len - len(b)
            all_ids.append(b + [pad_id] * pad)
            all_mask.append([1] * len(b) + [0] * pad)
            all_weight.append(wb + [0.0] * pad)
            all_lang.append(li)
        n_blocks = -(-len(id_stream) // seq_len) if id_stream else 0
        stats[language] = {"documents": len(docs), "tokens": len(id_stream), "blocks": n_blocks,
                           "padding_tokens": n_blocks * seq_len - len(id_stream),
                           "mean_token_weight": statistics.mean(weight_stream) if weight_stream else float("nan")}
    return {"input_ids": torch.tensor(all_ids, dtype=torch.long),
            "attention_mask": torch.tensor(all_mask, dtype=torch.long),
            "lang": torch.tensor(all_lang, dtype=torch.long),
            "weight": torch.tensor(all_weight, dtype=torch.float32), "stats": stats}


def format_weighted_data_summary(train: dict, languages: list[str]) -> str:
    header = f"{'language':<10}{'train docs':>11}{'train tokens':>15}{'blocks':>9}{'mean token weight':>19}"
    lines = [header, "-" * len(header)]
    for lang in languages:
        s = train["stats"][lang]
        lines.append(f"{lang:<10}{s['documents']:>11,}{s['tokens']:>15,}{s['blocks']:>9,}"
                     f"{s['mean_token_weight']:>19.4f}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Weighted loss
# --------------------------------------------------------------------------

def weighted_block_losses(model, input_ids, attention_mask, weight, device, use_bf16):
    """Per-block weighted loss sum and weight sum of valid target tokens (05_loss.md section 4/11)."""
    input_ids, attention_mask, weight = input_ids.to(device), attention_mask.to(device), weight.to(device)
    labels = input_ids.masked_fill(attention_mask == 0, -100)
    with base.autocast_ctx(device, use_bf16):
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    shifted_labels = labels[:, 1:]
    shifted_weight = weight[:, 1:]
    per_token = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.size(-1)), shifted_labels.reshape(-1),
                                ignore_index=-100, reduction="none").view(shifted_labels.shape)
    valid = (shifted_labels != -100).float()
    shifted_weight = shifted_weight * valid  # belt-and-suspenders: padding weight is already 0
    return (per_token * shifted_weight).sum(dim=1), shifted_weight.sum(dim=1)


# --------------------------------------------------------------------------
# Sanity check (weighted analogue of train_gpt2.run_sanity_check)
# --------------------------------------------------------------------------

def run_sanity_check(model, train_blocks, tokenizer, args, device, use_bf16) -> bool:
    ok = True
    vocab = len(tokenizer)
    ids, mask, weight = train_blocks["input_ids"], train_blocks["attention_mask"], train_blocks["weight"]
    print("\nSanity check")
    in_range = bool(((ids >= 0) & (ids < vocab)).all())
    print(f"  token ids inside the vocabulary (0..{vocab - 1}): {in_range}")
    ok &= in_range
    positive = bool((weight[mask.bool()] > 0).all())
    print(f"  every unmasked token has a strictly positive weight: {positive}")
    ok &= positive

    model.train()
    batch = slice(0, min(args.micro_batch_size, ids.size(0)))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    losses = []
    for _ in range(100):
        loss_sum, weight_sum = weighted_block_losses(model, ids[batch], mask[batch], weight[batch], device, use_bf16)
        loss = loss_sum.sum() / weight_sum.sum()
        if not torch.isfinite(loss):
            print("  loss became non-finite")
            return False
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(loss.item())
    import math
    expected = math.log(vocab)
    print(f"  initial loss {losses[0]:.3f} (about ln(vocab) = {expected:.3f} for random weights)")
    print(f"  loss on one repeated batch: {losses[0]:.3f} -> {losses[-1]:.3f}")
    decreased = losses[-1] < 0.5 * losses[0]
    print(f"  loss falls by more than half on a repeated batch: {decreased}")
    ok &= decreased and abs(losses[0] - expected) < 1.5
    print("  RESULT:", "PASS" if ok else "FAIL")
    return ok


# --------------------------------------------------------------------------
# Training (weighted analogue of train_gpt2.train)
# --------------------------------------------------------------------------

def train(model, train_blocks, val_blocks, languages, args, device, use_bf16, out_dir: Path, eval_log) -> dict:
    from transformers import get_cosine_schedule_with_warmup

    n_blocks = train_blocks["input_ids"].size(0)
    steps_per_epoch = n_blocks // args.batch_size
    if steps_per_epoch == 0:
        raise SystemExit(f"Only {n_blocks} training blocks, fewer than --batch-size {args.batch_size}")
    total_steps = steps_per_epoch * args.epochs
    dropped = n_blocks - steps_per_epoch * args.batch_size
    optimizer = base.build_optimizer(model, args.lr, args.weight_decay, device)
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(args.warmup_ratio * total_steps), total_steps)
    checkpointer = base.Checkpointer(args, model, optimizer, scheduler)
    resumed = checkpointer.resume()
    start_step = resumed["step"] if resumed else 0

    eval_steps = {e * steps_per_epoch + round(k * steps_per_epoch / args.evals_per_epoch)
                  for e in range(args.epochs) for k in range(1, args.evals_per_epoch + 1)}
    train_log = base.CsvLog(out_dir / "train_log.csv",
                            ["step", "epoch", "tokens_consumed", "train_loss", "mean_batch_weight", "lr",
                             "tokens_per_sec", "peak_mem_gib"], keep_upto_step=start_step if resumed else None)
    print(f"Training: {n_blocks} blocks/epoch -> {steps_per_epoch} steps/epoch x {args.epochs} epochs = {total_steps} steps "
          f"({args.batch_size} sequences x {args.seq_len} tokens per step; {dropped} blocks left out per epoch)")

    if args.eval_at_start and not resumed:
        result = base.evaluate(model, val_blocks, languages, args.eval_batch_size, device, use_bf16,
                               desc="validation @ step 0", nested=True)
        base.log_eval(eval_log, result, "validation", 0, 0.0, 0)
        print(f"  step 0 validation: {base.format_eval(result)}", flush=True)

    tokens_consumed, step = (resumed["tokens_consumed"], start_step) if resumed else (0, 0)
    window_loss, window_steps, window_tokens = 0.0, 0, 0
    window_start = time.time()
    started = time.time()
    if torch.device(device).type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    history = resumed["history"] if resumed else []
    for epoch in range(start_step // steps_per_epoch, args.epochs):
        perm = torch.randperm(n_blocks, generator=torch.Generator().manual_seed(args.seed * 1000 + epoch))
        epoch_loss = 0.0
        first = max(start_step - epoch * steps_per_epoch, 0)  # resuming: skip the batches already trained
        bar = base.progress(range(first, steps_per_epoch), desc=f"epoch {epoch + 1}/{args.epochs}", unit="step")
        for s in bar:
            idx = perm[s * args.batch_size:(s + 1) * args.batch_size]
            ids, mask, weight = (train_blocks["input_ids"][idx], train_blocks["attention_mask"][idx],
                                 train_blocks["weight"][idx])
            batch_weight_total = float((weight[:, 1:] * (mask[:, 1:] == 1)).sum().item())
            model.train()
            optimizer.zero_grad(set_to_none=True)
            step_loss = 0.0
            for m in range(0, ids.size(0), args.micro_batch_size):
                mb = slice(m, m + args.micro_batch_size)
                loss_sum, _ = weighted_block_losses(model, ids[mb], mask[mb], weight[mb], device, use_bf16)
                loss = loss_sum.sum() / batch_weight_total
                loss.backward()
                step_loss += loss.item()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            optimizer.step()
            scheduler.step()
            step += 1
            step_tokens = int(mask.sum().item())
            tokens_consumed += step_tokens
            window_loss += step_loss
            window_steps += 1
            window_tokens += step_tokens
            epoch_loss += step_loss

            lr = scheduler.get_last_lr()[0]
            bar.set_postfix(loss=f"{epoch_loss / (s + 1 - first):.4f}", lr=f"{lr:.2e}", refresh=False)
            if step % args.log_every == 0 or step == total_steps:
                if torch.device(device).type == "cuda":
                    torch.cuda.synchronize()
                elapsed = max(time.time() - window_start, 1e-9)
                peak = torch.cuda.max_memory_allocated() / 2 ** 30 if torch.device(device).type == "cuda" else 0.0
                train_log.write({"step": step, "epoch": round(step / steps_per_epoch, 3),
                                 "tokens_consumed": tokens_consumed, "train_loss": window_loss / window_steps,
                                 "mean_batch_weight": batch_weight_total / max(step_tokens, 1), "lr": lr,
                                 "tokens_per_sec": window_tokens / elapsed, "peak_mem_gib": round(peak, 2)})
                window_loss, window_steps, window_tokens, window_start = 0.0, 0, 0, time.time()
            if step in eval_steps and step != total_steps:
                result = base.evaluate(model, val_blocks, languages, args.eval_batch_size, device, use_bf16,
                                       desc=f"validation @ step {step}", nested=True)
                base.log_eval(eval_log, result, "validation", step, step / steps_per_epoch, tokens_consumed)
                history.append({"step": step, "epoch": step / steps_per_epoch, "tokens": tokens_consumed,
                                "validation": result["macro"]})
                print(f"  step {step} (epoch {step / steps_per_epoch:.2f}) validation: {base.format_eval(result)}",
                      flush=True)
            checkpointer.maybe_save(step, total_steps, {"tokens_consumed": tokens_consumed, "history": history})
        bar.close()

    train_log.close()
    wall = time.time() - started
    peak = torch.cuda.max_memory_allocated() / 2 ** 30 if torch.device(device).type == "cuda" else 0.0
    return {"steps": total_steps, "steps_per_epoch": steps_per_epoch, "blocks_left_out_per_epoch": dropped,
            "tokens_consumed": tokens_consumed, "wall_seconds": wall,
            "tokens_per_sec": tokens_consumed / wall, "peak_mem_gib": peak, "validation_history": history}


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def write_summary(path: Path, results: dict) -> None:
    d, run, w = results["data"], results["run"], results["weighting"]
    lines = [f"# GPT-2 pilot: quality-weighted loss ({results['config']['score_method']})", "",
             f"{results['config']['n_layer']} layers, {results['config']['n_embd']} hidden, "
             f"{results['parameters'] / 1e6:.1f}M parameters, {results['config']['epochs']} epoch(s), "
             f"seed {results['config']['seed']}.", "",
             f"Loss weighting: w_d = {w['offset']} + score_d * {w['scale']} "
             f"(score_d = combined rater importance score, clipped 0-5, from `{w['scores_dir']}`); "
             f"per-language normalization: {w['normalize_per_language']}.", "",
             "Mean token weight per language (after any normalization):", ""]
    for lang, s in d["train"].items():
        lines.append(f"- {lang}: {s['mean_token_weight']:.4f}")
    lines += ["",
             f"- Unique training tokens: {d['train_unique_tokens']:,}; tokens consumed: {run['tokens_consumed']:,} "
             f"(repeated exposure {run['tokens_consumed'] / d['train_unique_tokens']:.2f}x)",
             f"- Steps: {run['steps']} ({run['steps_per_epoch']} per epoch), throughput {run['tokens_per_sec']:,.0f} tokens/s, "
             f"peak GPU memory {run['peak_mem_gib']:.1f} GiB, wall time {run['wall_seconds'] / 60:.1f} min", "",
             "## Validation (final, ordinary unweighted loss)", ""] + base.eval_table(results["validation"]) + \
            ["", "## Test (final, ordinary unweighted loss)", ""] + base.eval_table(results["test"]) + \
            ["", "Perplexity = exp(mean next-token loss), evaluated with ordinary (unweighted) loss -- "
             "comparable to checkpoints/gpt2_top_doc/random_20M_ep1_seed42, the unweighted baseline trained on the "
             "same documents. Compare within a language, not across languages.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--score-method", choices=sorted(SCORE_DIRS), required=True,
                        help="Which importance score weights the training loss")
    parser.add_argument("--scores-dir", type=Path, default=None,
                        help="Default: SCORE_DIRS[--score-method], e.g. data/pilot_scores/avg5_mean")
    parser.add_argument("--weight-offset", type=float, default=0.5)
    parser.add_argument("--weight-scale", type=float, default=0.2, help="1/5: maps a 0-5 score to 0.5-1.5")
    parser.add_argument("--no-normalize-per-language", dest="normalize_per_language", action="store_false",
                        help="Disable the default per-language rescaling (weights stay the raw 0.5-1.5 mapping)")
    parser.add_argument("--train-data", type=Path, default=Path("data/pilot_selected/random_20M/documents.csv"),
                        help="Same Random 20M-token selection used by checkpoints/gpt2_top_doc/random_20M_ep1_seed42")
    parser.add_argument("--pool-dir", type=Path, default=Path("data/pilot_corpus"))
    parser.add_argument("--split-manifest", type=Path, default=None, help="Default: <pool-dir>/split_manifest.csv")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Default: checkpoints/gpt2_weighted_loss/<score-method>_20M_ep<epochs>_seed<seed>")
    parser.add_argument("--init-dir", type=Path, default=Path("checkpoints/gpt2_top_doc/init"),
                        help="Shared initial weights (same ones used by checkpoints/gpt2_top_doc runs)")
    parser.add_argument("--tokenizer", default=base.DEFAULT_TOKENIZER)
    parser.add_argument("--tokenizer-revision", default=None)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--n-layer", type=int, default=6)
    parser.add_argument("--n-embd", type=int, default=512)
    parser.add_argument("--n-head", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=32, help="Sequences per optimizer step")
    parser.add_argument("--micro-batch-size", type=int, default=8,
                        help="Sequences per forward/backward pass; lower it if the GPU runs out of memory")
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--evals-per-epoch", type=int, default=2)
    parser.add_argument("--no-eval-at-start", dest="eval_at_start", action="store_false")
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--limit-docs", type=int, default=0, help="Debug: keep only N documents per language and set")
    parser.add_argument("--sanity-check", action="store_true", help="Run the pre-flight checks and exit")
    parser.add_argument("--no-bf16", action="store_true")
    parser.add_argument("--no-save-model", action="store_true")
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    base.PROGRESS = not args.no_progress
    if args.batch_size % args.micro_batch_size:
        raise SystemExit("--batch-size must be a multiple of --micro-batch-size")
    scores_dir = args.scores_dir or SCORE_DIRS[args.score_method]
    manifest = args.split_manifest or args.pool_dir / "split_manifest.csv"
    out_dir = args.output_dir or Path("checkpoints/gpt2_weighted_loss") / f"{args.score_method}_20M_ep{args.epochs}_seed{args.seed}"
    device, use_bf16 = args.device, not args.no_bf16
    torch.backends.cuda.matmul.allow_tf32 = True

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, revision=args.tokenizer_revision)
    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading and encoding documents (loss weighted by '{args.score_method}' importance scores "
          f"from {scores_dir}) ...", flush=True)
    train_items = load_train_documents_with_ids(args.train_data, args.limit_docs)
    languages = sorted(train_items)
    val_docs = base.load_heldout_documents(args.pool_dir, manifest, "validation", args.limit_docs)
    test_docs = base.load_heldout_documents(args.pool_dir, manifest, "test", args.limit_docs)
    if set(val_docs) != set(languages) or set(test_docs) != set(languages):
        raise SystemExit("Training, validation and test must cover the same languages")
    pad_id = tokenizer.pad_token_id
    val_blocks = base.pack_streams({l: base.encode_documents(tokenizer, val_docs[l]) for l in languages},
                                   languages, args.seq_len, pad_id, args.seed)
    test_blocks = base.pack_streams({l: base.encode_documents(tokenizer, test_docs[l]) for l in languages},
                                    languages, args.seq_len, pad_id, args.seed)

    scores = load_importance_scores(scores_dir, languages)
    encoded = {l: encode_weighted_documents(tokenizer, train_items[l], scores[l], args.weight_offset,
                                            args.weight_scale) for l in languages}
    language_mean_weight = (normalize_weights_per_language(encoded) if args.normalize_per_language
                            else {l: float("nan") for l in languages})
    train_blocks = pack_weighted_streams(encoded, languages, args.seq_len, pad_id, args.seed)
    unique_tokens = sum(s["tokens"] for s in train_blocks["stats"].values())
    print(f"  train: {unique_tokens:,} tokens in {train_blocks['input_ids'].size(0)} blocks | validation: "
          f"{sum(s['tokens'] for s in val_blocks['stats'].values()):,} | test: "
          f"{sum(s['tokens'] for s in test_blocks['stats'].values()):,}", flush=True)
    print(format_weighted_data_summary(train_blocks, languages), flush=True)

    model, init_path = base.build_model(tokenizer, args)
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params / 1e6:.1f}M parameters on {device} (bf16={use_bf16 and device != 'cpu'})", flush=True)

    import transformers

    config = {**{k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
              "init_weights": str(init_path), "scores_dir": str(scores_dir), "vocab_size": len(tokenizer),
              "parameters": n_params, "torch": torch.__version__, "transformers": transformers.__version__}

    if args.sanity_check:
        raise SystemExit(0 if run_sanity_check(model, train_blocks, tokenizer, args, device, use_bf16) else 1)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    tokenizer.save_pretrained(out_dir / "tokenizer")

    eval_log = base.CsvLog(out_dir / "eval_log.csv", ["step", "epoch", "tokens_consumed", "split", "language",
                                                      "loss", "perplexity", "target_tokens"])
    run = train(model, train_blocks, val_blocks, languages, args, device, use_bf16, out_dir, eval_log)

    print("Final evaluation (ordinary unweighted loss) ...", flush=True)
    val_result = base.evaluate(model, val_blocks, languages, args.eval_batch_size, device, use_bf16,
                               desc="final validation")
    test_result = base.evaluate(model, test_blocks, languages, args.eval_batch_size, device, use_bf16,
                                desc="final test")
    base.log_eval(eval_log, val_result, "validation", run["steps"], float(args.epochs), run["tokens_consumed"])
    base.log_eval(eval_log, test_result, "test", run["steps"], float(args.epochs), run["tokens_consumed"])
    eval_log.close()
    print(f"  validation: {base.format_eval(val_result)}\n  test:       {base.format_eval(test_result)}", flush=True)

    results = {"config": config, "parameters": n_params, "run": run, "validation": val_result, "test": test_result,
              "weighting": {"score_method": args.score_method, "scores_dir": str(scores_dir),
                           "offset": args.weight_offset, "scale": args.weight_scale,
                           "normalize_per_language": args.normalize_per_language,
                           "language_mean_weight_before_normalization": language_mean_weight},
              "data": {"train_unique_tokens": unique_tokens, "train": train_blocks["stats"],
                       "validation": val_blocks["stats"], "test": test_blocks["stats"]}}
    (out_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    write_summary(out_dir / "summary.md", results)
    if not args.no_save_model:
        model.save_pretrained(out_dir / "final")
    print(f"Wrote results to {out_dir}")


if __name__ == "__main__":
    main()
