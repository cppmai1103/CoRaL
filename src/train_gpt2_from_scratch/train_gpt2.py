"""Train a small GPT-2 from scratch on one selection method's documents (guide 02, Steps 6-9).

Data
    Training documents come from a selection manifest (default: the Random baseline,
    data/pilot_selected/random/documents.csv). Held-out validation and test documents come
    from the split manifest written by split_corpus.py. Every method must be trained
    and evaluated on the same held-out documents.

Sequences (Step 6)
    Text is encoded with the SeaLLM v2 tokenizer (no special tokens) and one EOS is
    appended per document. Documents are shuffled with a recorded seed, concatenated
    per language, and cut into blocks of --seq-len tokens; the final partial block of
    each language is kept and padded. Padding positions get attention_mask 0 and label
    -100; genuine EOS tokens stay prediction targets (EOS is also the padding id, so
    the attention mask, never the id, decides what is padding). Blocks from all
    languages are then mixed and reshuffled every epoch.

Model and training (Steps 7-8)
    GPT-2 architecture with random weights (default 6 layers, 512 hidden, 8 heads,
    tied embeddings). The initial weights are saved once per (architecture, seed) in
    --init-dir and reused, so every method starts from the same state. AdamW, cosine
    schedule with warmup, gradient clipping, bf16 on GPU. Loss is next-token cross
    entropy summed over target tokens and normalised over each optimizer step, so
    accumulation and padding do not bias it.

Evaluation (Step 9)
    Loss and perplexity (exp of mean loss) per language on the validation set at the
    chosen intervals and on validation + test at the end, plus the macro average
    (each language weighted equally) and the token-weighted overall value. Targets are
    counted after the causal shift; padding is excluded.

Initial-weights baseline
    --eval-init-only evaluates the freshly initialized (untrained, shared) model on
    validation and test and exits, without touching the training loop or encoding the
    training documents. Its perplexity is the chance-level reference: a random model
    should score close to the vocabulary size, since next-token loss starts near
    ln(vocab). Compare a trained run's validation/test perplexity against this to see
    how much training actually improved over doing nothing.

Run from the project root:
    python -m src.train_gpt2_from_scratch.train_gpt2 --sanity-check
    python -m src.train_gpt2_from_scratch.train_gpt2 --epochs 4
    python -m src.train_gpt2_from_scratch.train_gpt2 --epochs 1 --method random
    python -m src.train_gpt2_from_scratch.train_gpt2 --eval-init-only --train-data data/pilot_selected/random/documents.csv

Outputs in --output-dir (default checkpoints/gpt2_top_doc/<method>_ep<epochs>_seed<seed>, or checkpoints/gpt2_top_doc/init_seed<seed>
for --eval-init-only):
    config.json  results.json  summary.md  train_log.csv  eval_log.csv  tokenizer/  final/
    (--eval-init-only skips train_log.csv: there is no training)
Requires torch, transformers and tqdm.
"""

import argparse
import csv
import json
import math
import random
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

DEFAULT_TOKENIZER = "SeaLLMs/SeaLLM-7B-v2"
PROGRESS = True  # set from --no-progress


# --------------------------------------------------------------------------
# Progress bars (interactive on a terminal, one plain line per update in a log file)
# --------------------------------------------------------------------------

class _LineWriter:
    def __init__(self, stream):
        self.stream = stream

    def write(self, text: str) -> int:
        text = text.replace("\r", "").rstrip()
        if text:
            self.stream.write(text + "\n")
        return len(text)

    def flush(self) -> None:
        self.stream.flush()


def progress(iterable, nested: bool = False, **kwargs):
    """`nested` bars are drawn only on a terminal: in a log file, a bar opened while another is active
    prints cursor-movement codes. The evaluation result line is printed either way."""
    if nested and not sys.stdout.isatty():
        return iterable
    if sys.stdout.isatty():
        return tqdm(iterable, disable=not PROGRESS, file=sys.stdout, **kwargs)
    return tqdm(iterable, disable=not PROGRESS, file=_LineWriter(sys.stdout), mininterval=30, maxinterval=60,
                dynamic_ncols=False, ncols=120, **kwargs)


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

def load_train_documents(path: Path, limit: int = 0) -> dict[str, list[str]]:
    csv.field_size_limit(sys.maxsize)
    docs: dict[str, list[str]] = defaultdict(list)
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            docs[row["language"]].append(row["text"])
    if limit:
        docs = {lang: texts[:limit] for lang, texts in docs.items()}
    return dict(docs)


def load_heldout_documents(pool_dir: Path, manifest: Path, split: str, limit: int = 0) -> dict[str, list[str]]:
    """Texts of the validation/test documents, looked up in the pool CSVs by (language, doc_id)."""
    csv.field_size_limit(sys.maxsize)
    wanted: dict[str, list[str]] = defaultdict(list)
    with manifest.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row["split"] == split:
                wanted[row["language"]].append(row["doc_id"])
    docs: dict[str, list[str]] = {}
    for language, ids in wanted.items():
        ids = sorted(ids)[:limit] if limit else sorted(ids)
        keep = set(ids)
        found = {}
        with (pool_dir / f"{language}.csv").open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                if row["doc_id"] in keep:
                    found[row["doc_id"]] = row["text"]
        missing = keep - set(found)
        if missing:
            raise ValueError(f"{language}: {len(missing)} {split} documents from the manifest are not in the pool CSV")
        docs[language] = [found[i] for i in ids]
    return docs


def encode_documents(tokenizer, texts: list[str], batch_size: int = 1000, bos: bool = False) -> list[list[int]]:
    """Token ids of each document with one EOS appended (the document boundary), and with `bos` also the
    tokenizer's BOS prepended (pretrained models such as Gemma, which start every document with <bos>)."""
    start = [tokenizer.bos_token_id] if bos and tokenizer.bos_token_id is not None else []
    out = []
    for i in range(0, len(texts), batch_size):
        for ids in tokenizer(texts[i:i + batch_size], add_special_tokens=False)["input_ids"]:
            out.append(start + ids + [tokenizer.eos_token_id])
    return out


def eval_format(model, run_dir: Path | None = None) -> tuple[int, bool]:
    """(block length, prepend BOS) to evaluate a run folder the way it was trained: GPT-2 pilot runs use their
    n_positions and no BOS; LoRA continued-pretraining runs (continue_pretrain_lora.py) use the seq_len in their
    config.json and BOS-prefixed documents."""
    if getattr(model.config, "model_type", "gpt2") == "gpt2":
        return model.config.n_positions, False
    seq_len = 1024
    if run_dir is not None and (run_dir / "config.json").is_file():
        seq_len = json.loads((run_dir / "config.json").read_text(encoding="utf-8")).get("seq_len", seq_len)
    return seq_len, True


def pack_streams(encoded: dict[str, list[list[int]]], languages: list[str], seq_len: int, pad_id: int,
                 seed: int) -> dict:
    """Shuffle documents per language (seeded), concatenate, cut into blocks of `seq_len`.

    The last block of each language is padded. Returns tensors input_ids [N, L], attention_mask [N, L],
    lang [N] (index into `languages`) and per-language statistics."""
    all_ids, all_mask, all_lang = [], [], []
    stats = {}
    for li, language in enumerate(languages):
        docs = list(encoded.get(language, []))
        random.Random(f"{seed}:{language}").shuffle(docs)
        stream = [t for d in docs for t in d]
        blocks = [stream[i:i + seq_len] for i in range(0, len(stream), seq_len)]
        for b in blocks:
            all_ids.append(b + [pad_id] * (seq_len - len(b)))
            all_mask.append([1] * len(b) + [0] * (seq_len - len(b)))
            all_lang.append(li)
        stats[language] = {"documents": len(docs), "tokens": len(stream), "blocks": len(blocks),
                           "padding_tokens": len(blocks) * seq_len - len(stream)}
    return {"input_ids": torch.tensor(all_ids, dtype=torch.long),
            "attention_mask": torch.tensor(all_mask, dtype=torch.long),
            "lang": torch.tensor(all_lang, dtype=torch.long), "stats": stats}


def format_data_summary(train: dict, val: dict, test: dict, languages: list[str]) -> str:
    """Documents and tokens per language and split, after encoding (tokens include one EOS per document)."""
    header = (f"{'language':<10}{'train docs':>11}{'train tokens':>15}{'blocks':>9}"
              f"{'val docs':>10}{'val tokens':>12}{'test docs':>11}{'test tokens':>13}")
    lines = [header, "-" * len(header)]
    totals = defaultdict(int)
    for lang in languages:
        tr, va, te = train["stats"][lang], val["stats"][lang], test["stats"][lang]
        lines.append(f"{lang:<10}{tr['documents']:>11,}{tr['tokens']:>15,}{tr['blocks']:>9,}"
                     f"{va['documents']:>10,}{va['tokens']:>12,}{te['documents']:>11,}{te['tokens']:>13,}")
        for key, value in (("td", tr["documents"]), ("tt", tr["tokens"]), ("tb", tr["blocks"]), ("vd", va["documents"]),
                           ("vt", va["tokens"]), ("ed", te["documents"]), ("et", te["tokens"])):
            totals[key] += value
    lines.append("-" * len(header))
    lines.append(f"{'total':<10}{totals['td']:>11,}{totals['tt']:>15,}{totals['tb']:>9,}"
                 f"{totals['vd']:>10,}{totals['vt']:>12,}{totals['ed']:>11,}{totals['et']:>13,}")
    return "\n".join(lines)


def target_count(attention_mask: torch.Tensor) -> int:
    """Scored next-token targets: positions 1..L-1 that hold a real token."""
    return int(attention_mask[:, 1:].sum().item())


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------

def build_config(tokenizer, args):
    from transformers import GPT2Config

    return GPT2Config(
        vocab_size=len(tokenizer), n_positions=args.seq_len, n_embd=args.n_embd, n_layer=args.n_layer,
        n_head=args.n_head, bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id, tie_word_embeddings=True, use_cache=False,
    )


def build_model(tokenizer, args):
    """Random-weight GPT-2, identical across methods for a given (architecture, seed)."""
    from transformers import GPT2LMHeadModel, set_seed

    config = build_config(tokenizer, args)
    set_seed(args.seed)
    model = GPT2LMHeadModel(config)
    init_dir = Path(args.init_dir)
    init_dir.mkdir(parents=True, exist_ok=True)
    name = f"gpt2_L{args.n_layer}_d{args.n_embd}_h{args.n_head}_ctx{args.seq_len}_V{len(tokenizer)}_seed{args.seed}.pt"
    path = init_dir / name
    if path.is_file():
        model.load_state_dict(torch.load(path, map_location="cpu"))
        print(f"Loaded shared initial weights: {path}")
    else:
        torch.save(model.state_dict(), path)
        print(f"Saved shared initial weights: {path}")
    return model, path


def build_optimizer(model, lr: float, weight_decay: float, device) -> torch.optim.Optimizer:
    decay = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
    groups = [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
    kwargs = {"fused": True} if torch.device(device).type == "cuda" else {}
    return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95), **kwargs)


def autocast_ctx(device, use_bf16: bool):
    if use_bf16 and torch.device(device).type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return torch.autocast(device_type="cpu", enabled=False)


def block_losses(model, input_ids, attention_mask, device, use_bf16):
    """Per-block summed cross-entropy and target counts. Labels are the inputs with padding masked
    to -100; the shift happens here, once."""
    input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
    labels = input_ids.masked_fill(attention_mask == 0, -100)
    with autocast_ctx(device, use_bf16):
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    shifted = labels[:, 1:]
    token_loss = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.size(-1)), shifted.reshape(-1),
                                 ignore_index=-100, reduction="none").view(shifted.shape)
    return token_loss.sum(dim=1), (shifted != -100).sum(dim=1)


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, blocks: dict, languages: list[str], batch_size: int, device, use_bf16: bool,
             desc: str | None = None, nested: bool = False) -> dict:
    """Loss and perplexity per language, macro average, and token-weighted overall."""
    model.eval()
    sums = torch.zeros(len(languages), dtype=torch.float64)
    counts = torch.zeros(len(languages), dtype=torch.long)
    n = blocks["input_ids"].size(0)
    starts = range(0, n, batch_size)
    for s in (progress(starts, nested=nested, desc=desc, unit="batch", leave=False) if desc else starts):
        sl = slice(s, s + batch_size)
        loss_sum, count = block_losses(model, blocks["input_ids"][sl], blocks["attention_mask"][sl], device, use_bf16)
        lang = blocks["lang"][sl]
        sums.index_add_(0, lang, loss_sum.double().cpu())
        counts.index_add_(0, lang, count.cpu())
    return summarise_losses(sums, counts, languages)


def summarise_losses(sums: torch.Tensor, counts: torch.Tensor, languages: list[str]) -> dict:
    per_language = {}
    for i, language in enumerate(languages):
        n = int(counts[i])
        loss = float(sums[i] / n) if n else float("nan")
        per_language[language] = {"loss": loss, "perplexity": math.exp(loss) if n else float("nan"), "target_tokens": n}
    valid = [v["loss"] for v in per_language.values() if v["target_tokens"]]
    macro = statistics.mean(valid)
    micro = float(sums.sum() / counts.sum())
    return {"per_language": per_language,
            "macro": {"loss": macro, "perplexity": math.exp(macro)},
            "token_weighted": {"loss": micro, "perplexity": math.exp(micro), "target_tokens": int(counts.sum())}}


# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------

class CsvLog:
    def __init__(self, path: Path, fields: list[str], keep_upto_step: int | None = None):
        """keep_upto_step (resuming from a checkpoint): keep the rows already logged up to that step, drop later ones
        (logged after the checkpoint, so they will be logged again) and append; otherwise start a new file."""
        kept = []
        if keep_upto_step is not None and path.is_file():
            with path.open(encoding="utf-8", newline="") as f:
                kept = [r for r in csv.DictReader(f) if int(r["step"]) <= keep_upto_step]
        self.f = path.open("w", encoding="utf-8", newline="")
        self.writer = csv.DictWriter(self.f, fieldnames=fields)
        self.writer.writeheader()
        self.writer.writerows(kept)
        self.f.flush()

    def write(self, row: dict) -> None:
        self.writer.writerow(row)
        self.f.flush()

    def close(self) -> None:
        self.f.close()


class Checkpointer:
    """Periodic training checkpoints, so a run killed part-way can resume where it stopped.

    Off unless the caller sets args.checkpoint_dir and args.save_every (> 0); the GPT-2 scripts never do. Every
    args.save_every steps it saves the trainable parameters (only the LoRA adapter for LoRA runs), the optimizer and
    scheduler state, the RNG states and the step reached, replacing the previous checkpoint (written to a temporary
    file, then renamed, so a crash while saving leaves the previous one intact). With args.resume, training restarts
    after the saved step: block order is a fixed function of the seed and epoch, so the resumed run sees exactly the
    batches the uninterrupted run would have. args.checkpoint_fingerprint (the settings that must not change) is
    stored and checked on resume."""

    def __init__(self, args, model, optimizer, scheduler):
        self.dir = getattr(args, "checkpoint_dir", None)
        self.every = getattr(args, "save_every", 0) if self.dir else 0
        self.resume_enabled = bool(self.dir) and getattr(args, "resume", False)
        self.fingerprint = getattr(args, "checkpoint_fingerprint", {})
        self.model, self.optimizer, self.scheduler = model, optimizer, scheduler

    @staticmethod
    def saved_step(checkpoint_dir) -> int | None:
        path = Path(checkpoint_dir) / "state.json" if checkpoint_dir else None
        return json.loads(path.read_text(encoding="utf-8"))["step"] if path and path.is_file() else None

    def resume(self) -> dict | None:
        if not self.resume_enabled or not (Path(self.dir) / "checkpoint.pt").is_file():
            return None
        ckpt = torch.load(Path(self.dir) / "checkpoint.pt", map_location="cpu", weights_only=False)
        if ckpt["fingerprint"] != self.fingerprint:
            changed = sorted(k for k in set(ckpt["fingerprint"]) | set(self.fingerprint)
                             if ckpt["fingerprint"].get(k) != self.fingerprint.get(k))
            raise SystemExit(f"Checkpoint {self.dir} was made with other settings ({', '.join(changed)}): delete it "
                             "or pass --no-resume to start over")
        params = dict(self.model.named_parameters())
        with torch.no_grad():
            for name, value in ckpt["trainable"].items():
                params[name].copy_(value)
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scheduler.load_state_dict(ckpt["scheduler"])
        torch.set_rng_state(ckpt["rng_cpu"])
        if ckpt["rng_cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(ckpt["rng_cuda"])
        print(f"Resumed from {self.dir} at step {ckpt['state']['step']}", flush=True)
        return ckpt["state"]

    def maybe_save(self, step: int, total_steps: int, state: dict) -> None:
        if not self.every or step % self.every or step == total_steps:
            return
        out = Path(self.dir)
        out.mkdir(parents=True, exist_ok=True)
        ckpt = {"trainable": {n: p.detach().cpu() for n, p in self.model.named_parameters() if p.requires_grad},
                "optimizer": self.optimizer.state_dict(), "scheduler": self.scheduler.state_dict(),
                "rng_cpu": torch.get_rng_state(),
                "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "fingerprint": self.fingerprint, "state": {**state, "step": step}}
        torch.save(ckpt, out / "checkpoint.pt.tmp")
        (out / "checkpoint.pt.tmp").replace(out / "checkpoint.pt")
        (out / "state.json").write_text(json.dumps({"step": step, "total_steps": total_steps}), encoding="utf-8")


def log_eval(eval_log: CsvLog, result: dict, split: str, step: int, epoch: float, tokens: int) -> None:
    base = {"step": step, "epoch": round(epoch, 3), "tokens_consumed": tokens, "split": split}
    for language, v in result["per_language"].items():
        eval_log.write({**base, "language": language, "loss": v["loss"], "perplexity": v["perplexity"],
                        "target_tokens": v["target_tokens"]})
    eval_log.write({**base, "language": "macro", "loss": result["macro"]["loss"],
                    "perplexity": result["macro"]["perplexity"], "target_tokens": ""})
    eval_log.write({**base, "language": "token_weighted", "loss": result["token_weighted"]["loss"],
                    "perplexity": result["token_weighted"]["perplexity"],
                    "target_tokens": result["token_weighted"]["target_tokens"]})


def format_eval(result: dict) -> str:
    m, t = result["macro"], result["token_weighted"]
    return (f"macro loss {m['loss']:.4f} (ppl {m['perplexity']:.1f}) | "
            f"token-weighted loss {t['loss']:.4f} (ppl {t['perplexity']:.1f})")


# --------------------------------------------------------------------------
# Sanity check (guide Step 8: run this before the real comparison)
# --------------------------------------------------------------------------

def run_sanity_check(model, train_blocks, tokenizer, args, device, use_bf16) -> bool:
    ok = True
    vocab = len(tokenizer)
    ids, mask = train_blocks["input_ids"], train_blocks["attention_mask"]
    print("\nSanity check")
    in_range = bool(((ids >= 0) & (ids < vocab)).all())
    print(f"  token ids inside the vocabulary (0..{vocab - 1}): {in_range}")
    ok &= in_range
    monotone = bool((mask[:, 1:] <= mask[:, :-1]).all())
    padding_ok = bool((mask.sum(dim=1) > 0).all())
    print(f"  padding only at the end of a block: {monotone} | no empty blocks: {padding_ok}")
    ok &= monotone and padding_ok
    eos_in_targets = int(((ids == tokenizer.eos_token_id) & (mask == 1)).sum().item())
    print(f"  genuine EOS tokens kept as real targets: {eos_in_targets}")
    ok &= eos_in_targets > 0

    model.train()
    batch = slice(0, min(args.micro_batch_size, ids.size(0)))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    losses = []
    for _ in range(100):
        loss_sum, count = block_losses(model, ids[batch], mask[batch], device, use_bf16)
        loss = loss_sum.sum() / count.sum()
        if not torch.isfinite(loss):
            print("  loss became non-finite")
            return False
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(loss.item())
    expected = math.log(vocab)
    print(f"  initial loss {losses[0]:.3f} (about ln(vocab) = {expected:.3f} for random weights)")
    print(f"  loss on one repeated batch: {losses[0]:.3f} -> {losses[-1]:.3f}")
    decreased = losses[-1] < 0.5 * losses[0]
    print(f"  loss falls by more than half on a repeated batch: {decreased}")
    ok &= decreased and abs(losses[0] - expected) < 1.5
    print("  RESULT:", "PASS" if ok else "FAIL")
    return ok


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------

def dist_info() -> tuple[int, int]:
    """(rank, world size) under torchrun once the caller has initialised torch.distributed, else (0, 1)."""
    import torch.distributed as dist
    return (dist.get_rank(), dist.get_world_size()) if dist.is_available() and dist.is_initialized() else (0, 1)


class NullLog:
    """Stands in for CsvLog on the non-main ranks of a multi-GPU run (only rank 0 writes the logs)."""

    def write(self, row: dict) -> None:
        pass

    def close(self) -> None:
        pass


def train(model, train_blocks, val_blocks, languages, args, device, use_bf16, out_dir: Path, eval_log: CsvLog):
    from transformers import get_cosine_schedule_with_warmup

    # Multi-GPU (torchrun): every rank takes the same batches; rank r runs micro-batches r, r + world, ... of each
    # step and the gradients are summed across ranks, so a step is the same update as on one GPU. Rank 0 alone
    # evaluates, logs and saves checkpoints while the others wait.
    rank, world = dist_info()
    is_main = rank == 0
    if args.batch_size % (args.micro_batch_size * world):
        raise SystemExit(f"--batch-size {args.batch_size} must be a multiple of --micro-batch-size x GPUs "
                         f"({args.micro_batch_size} x {world})")
    if world > 1:
        import torch.distributed as dist
    n_blocks = train_blocks["input_ids"].size(0)
    steps_per_epoch = n_blocks // args.batch_size
    if steps_per_epoch == 0:
        raise SystemExit(f"Only {n_blocks} training blocks, fewer than --batch-size {args.batch_size}")
    total_steps = steps_per_epoch * args.epochs
    dropped = n_blocks - steps_per_epoch * args.batch_size
    optimizer = build_optimizer(model, args.lr, args.weight_decay, device)
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(args.warmup_ratio * total_steps), total_steps)
    checkpointer = Checkpointer(args, model, optimizer, scheduler)
    resumed = checkpointer.resume()
    start_step = resumed["step"] if resumed else 0

    eval_steps = {e * steps_per_epoch + round(k * steps_per_epoch / args.evals_per_epoch)
                  for e in range(args.epochs) for k in range(1, args.evals_per_epoch + 1)}
    train_log = CsvLog(out_dir / "train_log.csv",
                       ["step", "epoch", "tokens_consumed", "train_loss", "lr", "tokens_per_sec", "peak_mem_gib"],
                       keep_upto_step=start_step if resumed else None) if is_main else NullLog()
    print(f"Training: {n_blocks} blocks/epoch -> {steps_per_epoch} steps/epoch x {args.epochs} epochs = {total_steps} steps "
          f"({args.batch_size} sequences x {args.seq_len} tokens per step{f', split over {world} GPUs' if world > 1 else ''}; "
          f"{dropped} blocks left out per epoch)")

    if args.eval_at_start and not resumed:
        if is_main:
            result = evaluate(model, val_blocks, languages, args.eval_batch_size, device, use_bf16, desc="validation @ step 0", nested=True)
            log_eval(eval_log, result, "validation", 0, 0.0, 0)
            print(f"  step 0 validation: {format_eval(result)}", flush=True)
        if world > 1:
            dist.barrier()

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
        bar = progress(range(first, steps_per_epoch), desc=f"epoch {epoch + 1}/{args.epochs}", unit="step")
        for s in bar:
            idx = perm[s * args.batch_size:(s + 1) * args.batch_size]
            ids, mask = train_blocks["input_ids"][idx], train_blocks["attention_mask"][idx]
            targets = target_count(mask)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            step_loss = 0.0
            for m in range(rank * args.micro_batch_size, ids.size(0), world * args.micro_batch_size):
                loss_sum, _ = block_losses(model, ids[m:m + args.micro_batch_size], mask[m:m + args.micro_batch_size],
                                           device, use_bf16)
                loss = loss_sum.sum() / targets
                loss.backward()
                step_loss += loss.item()
            if world > 1:
                grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
                flat = torch.cat([g.reshape(-1) for g in grads] + [torch.tensor([step_loss], device=grads[0].device)])
                dist.all_reduce(flat)
                offset = 0
                for g in grads:
                    g.copy_(flat[offset:offset + g.numel()].view_as(g))
                    offset += g.numel()
                step_loss = flat[-1].item()
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
                                 "lr": lr, "tokens_per_sec": window_tokens / elapsed, "peak_mem_gib": round(peak, 2)})
                window_loss, window_steps, window_tokens, window_start = 0.0, 0, 0, time.time()
            if step in eval_steps and step != total_steps:
                if is_main:
                    result = evaluate(model, val_blocks, languages, args.eval_batch_size, device, use_bf16,
                                      desc=f"validation @ step {step}", nested=True)
                    log_eval(eval_log, result, "validation", step, step / steps_per_epoch, tokens_consumed)
                    history.append({"step": step, "epoch": step / steps_per_epoch, "tokens": tokens_consumed,
                                    "validation": result["macro"]})
                    print(f"  step {step} (epoch {step / steps_per_epoch:.2f}) validation: {format_eval(result)}", flush=True)
                if world > 1:
                    dist.barrier()
            if is_main:
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

def eval_table(result: dict) -> list[str]:
    lines = ["| Language | Loss | Perplexity | Target tokens |", "| --- | --- | --- | --- |"]
    for language, v in result["per_language"].items():
        lines.append(f"| {language} | {v['loss']:.4f} | {v['perplexity']:.2f} | {v['target_tokens']:,} |")
    m, t = result["macro"], result["token_weighted"]
    lines.append(f"| **macro average** | {m['loss']:.4f} | {m['perplexity']:.2f} | |")
    lines.append(f"| **token-weighted** | {t['loss']:.4f} | {t['perplexity']:.2f} | {t['target_tokens']:,} |")
    return lines


def write_summary(path: Path, results: dict) -> None:
    d, run = results["data"], results["run"]
    lines = [f"# GPT-2 pilot: {results['config']['method']}", "",
             f"{results['config']['n_layer']} layers, {results['config']['n_embd']} hidden, "
             f"{results['parameters'] / 1e6:.1f}M parameters, {results['config']['epochs']} epoch(s), "
             f"seed {results['config']['seed']}.", "",
             f"- Unique training tokens: {d['train_unique_tokens']:,}; tokens consumed: {run['tokens_consumed']:,} "
             f"(repeated exposure {run['tokens_consumed'] / d['train_unique_tokens']:.2f}x)",
             f"- Steps: {run['steps']} ({run['steps_per_epoch']} per epoch), throughput {run['tokens_per_sec']:,.0f} tokens/s, "
             f"peak GPU memory {run['peak_mem_gib']:.1f} GiB, wall time {run['wall_seconds'] / 60:.1f} min", "",
             "## Validation (final)", ""] + eval_table(results["validation"]) + \
            ["", "## Test (final)", ""] + eval_table(results["test"]) + \
            ["", "Perplexity = exp(mean next-token loss). Compare methods only with the same tokenizer and held-out "
             "documents; compare within a language, not across languages.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_init_summary(path: Path, results: dict) -> None:
    """Same layout as write_summary, for the untrained --eval-init-only baseline (no `run`/`data.train*` fields)."""
    lines = ["# GPT-2 pilot: untrained initial weights (chance-level reference)", "",
             f"{results['config']['n_layer']} layers, {results['config']['n_embd']} hidden, "
             f"{results['parameters'] / 1e6:.1f}M parameters, seed {results['config']['seed']}. No training was run: "
             f"a randomly initialized model should score close to chance, i.e. perplexity near the vocabulary size "
             f"({results['config']['vocab_size']:,}).", "",
             f"- Evaluation wall time: {results['wall_seconds'] / 60:.1f} min", "",
             "## Validation", ""] + eval_table(results["validation"]) + \
            ["", "## Test", ""] + eval_table(results["test"]) + \
            ["", "Perplexity = exp(mean next-token loss). Compare against a trained run's validation/test perplexity "
             "(same held-out documents) to see how much training improved over doing nothing.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--method", default="random", help="Label for this selection method (output folder name)")
    parser.add_argument("--train-data", type=Path, default=Path("data/pilot_selected/random/documents.csv"))
    parser.add_argument("--pool-dir", type=Path, default=Path("data/pilot_corpus"))
    parser.add_argument("--split-manifest", type=Path, default=None, help="Default: <pool-dir>/split_manifest.csv")
    parser.add_argument("--output-dir", type=Path, default=None, help="Default: checkpoints/gpt2_top_doc/<method>_ep<epochs>_seed<seed>")
    parser.add_argument("--init-dir", type=Path, default=Path("checkpoints/gpt2_top_doc/init"), help="Shared initial weights")
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--tokenizer-revision", default=None)
    parser.add_argument("--epochs", type=int, default=4)
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
    parser.add_argument("--evals-per-epoch", type=int, default=1)
    parser.add_argument("--no-eval-at-start", dest="eval_at_start", action="store_false")
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--limit-docs", type=int, default=0, help="Debug: keep only N documents per language and set")
    parser.add_argument("--sanity-check", action="store_true", help="Run the pre-flight checks and exit")
    parser.add_argument("--eval-init-only", action="store_true",
                        help="Evaluate the untrained initial weights on validation/test and exit (no training)")
    parser.add_argument("--no-bf16", action="store_true")
    parser.add_argument("--no-save-model", action="store_true")
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    global PROGRESS
    PROGRESS = not args.no_progress
    if args.batch_size % args.micro_batch_size:
        raise SystemExit("--batch-size must be a multiple of --micro-batch-size")
    manifest = args.split_manifest or args.pool_dir / "split_manifest.csv"
    out_dir = args.output_dir or Path("checkpoints/gpt2_top_doc") / f"{args.method}_ep{args.epochs}_seed{args.seed}"
    device, use_bf16 = args.device, not args.no_bf16
    torch.backends.cuda.matmul.allow_tf32 = True

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, revision=args.tokenizer_revision)
    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token  # the attention mask, not the id, marks padding

    print("Loading and encoding documents ...", flush=True)
    train_docs = load_train_documents(args.train_data, args.limit_docs)
    languages = sorted(train_docs)
    val_docs = load_heldout_documents(args.pool_dir, manifest, "validation", args.limit_docs)
    test_docs = load_heldout_documents(args.pool_dir, manifest, "test", args.limit_docs)
    if set(val_docs) != set(languages) or set(test_docs) != set(languages):
        raise SystemExit("Training, validation and test must cover the same languages")
    pad_id = tokenizer.pad_token_id
    val_blocks = pack_streams({l: encode_documents(tokenizer, val_docs[l]) for l in languages},
                              languages, args.seq_len, pad_id, args.seed)
    test_blocks = pack_streams({l: encode_documents(tokenizer, test_docs[l]) for l in languages},
                               languages, args.seq_len, pad_id, args.seed)
    train_blocks, unique_tokens = None, 0
    if args.eval_init_only:
        print(f"  validation: {sum(s['tokens'] for s in val_blocks['stats'].values()):,} tokens | test: "
              f"{sum(s['tokens'] for s in test_blocks['stats'].values()):,} tokens "
              f"(training documents not encoded: --eval-init-only)", flush=True)
    else:
        train_blocks = pack_streams({l: encode_documents(tokenizer, train_docs[l]) for l in languages},
                                    languages, args.seq_len, pad_id, args.seed)
        unique_tokens = sum(s["tokens"] for s in train_blocks["stats"].values())
        print(f"  train: {unique_tokens:,} tokens in {train_blocks['input_ids'].size(0)} blocks | validation: "
              f"{sum(s['tokens'] for s in val_blocks['stats'].values()):,} | test: "
              f"{sum(s['tokens'] for s in test_blocks['stats'].values()):,}", flush=True)
        print("Tokens per language after loading (SeaLLM tokenizer, one EOS per document included):", flush=True)
        print(format_data_summary(train_blocks, val_blocks, test_blocks, languages), flush=True)

    model, init_path = build_model(tokenizer, args)
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params / 1e6:.1f}M parameters on {device} (bf16={use_bf16 and device != 'cpu'})", flush=True)

    import transformers

    config = {**{k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
              "init_weights": str(init_path), "vocab_size": len(tokenizer), "parameters": n_params,
              "torch": torch.__version__, "transformers": transformers.__version__}

    if args.sanity_check:
        if train_blocks is None:
            raise SystemExit("--sanity-check needs encoded training data; remove --eval-init-only")
        raise SystemExit(0 if run_sanity_check(model, train_blocks, tokenizer, args, device, use_bf16) else 1)

    if args.eval_init_only:
        out_dir = args.output_dir or Path("checkpoints/gpt2_top_doc") / f"init_seed{args.seed}"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        tokenizer.save_pretrained(out_dir / "tokenizer")

        eval_log = CsvLog(out_dir / "eval_log.csv", ["step", "epoch", "tokens_consumed", "split", "language", "loss",
                                                     "perplexity", "target_tokens"])
        print("Evaluating the untrained initial weights (no training) ...", flush=True)
        started = time.time()
        val_result = evaluate(model, val_blocks, languages, args.eval_batch_size, device, use_bf16,
                              desc="validation (untrained)")
        test_result = evaluate(model, test_blocks, languages, args.eval_batch_size, device, use_bf16,
                               desc="test (untrained)")
        wall = time.time() - started
        log_eval(eval_log, val_result, "validation", 0, 0.0, 0)
        log_eval(eval_log, test_result, "test", 0, 0.0, 0)
        eval_log.close()
        print(f"  validation: {format_eval(val_result)}\n  test:       {format_eval(test_result)}", flush=True)

        results = {"config": config, "parameters": n_params, "wall_seconds": wall,
                  "validation": val_result, "test": test_result,
                  "data": {"validation": val_blocks["stats"], "test": test_blocks["stats"]}}
        (out_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        write_init_summary(out_dir / "summary.md", results)
        if not args.no_save_model:
            model.save_pretrained(out_dir / "final")
        print(f"Wrote results to {out_dir}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    tokenizer.save_pretrained(out_dir / "tokenizer")

    eval_log = CsvLog(out_dir / "eval_log.csv", ["step", "epoch", "tokens_consumed", "split", "language", "loss",
                                                 "perplexity", "target_tokens"])
    run = train(model, train_blocks, val_blocks, languages, args, device, use_bf16, out_dir, eval_log)

    print("Final evaluation ...", flush=True)
    val_result = evaluate(model, val_blocks, languages, args.eval_batch_size, device, use_bf16, desc="final validation")
    test_result = evaluate(model, test_blocks, languages, args.eval_batch_size, device, use_bf16, desc="final test")
    log_eval(eval_log, val_result, "validation", run["steps"], float(args.epochs), run["tokens_consumed"])
    log_eval(eval_log, test_result, "test", run["steps"], float(args.epochs), run["tokens_consumed"])
    eval_log.close()
    print(f"  validation: {format_eval(val_result)}\n  test:       {format_eval(test_result)}", flush=True)

    results = {"config": config, "parameters": n_params, "run": run, "validation": val_result, "test": test_result,
               "data": {"train_unique_tokens": unique_tokens, "train": train_blocks["stats"],
                        "validation": val_blocks["stats"], "test": test_blocks["stats"]}}
    (out_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    write_summary(out_dir / "summary.md", results)
    if not args.no_save_model:
        model.save_pretrained(out_dir / "final")
    print(f"Wrote results to {out_dir}")


if __name__ == "__main__":
    main()
