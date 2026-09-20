"""Fully fine-tune mmBERT-base as a SEA-Rater regressor (Meta-rater style).

Unlike src.train (frozen encoder + MLP head on cached embeddings), this trains
the whole model end to end: `AutoModelForSequenceClassification` with
`num_labels=1` / `problem_type="regression"`, so the loss is MSE against the
human-mean score. One model per dimension, each on its own independent split
(prepared_data/split_manifest_<dimension>.csv). Documents are fed in one pass
(truncated at --max-length; the longest document in the current data is 3,947
tokens, so 4096 truncates nothing).

Pooling (--pooling) selects how the encoder output becomes one vector for the
regression head:
    cls   first-token state (the Meta-rater / transformers default)
    mean  mean over non-padding tokens

The pretrained prediction head (dense + norm) is kept; only the final
`classifier` layer is new. It gets its own higher learning rate and its bias is
initialised to the training-mean score. Checkpoints are selected by validation
macro-MAE, as in src.train; the test split is never used for selection.

Run from the project root:
    python -m src.finetune --dimension all --pooling cls
    python -m src.finetune --dimension all --pooling mean
    python -m src.finetune --dimension reasoning --pooling mean --limit 32 --max-length 128 \\
        --max-epochs 1 --batch-size 8 --micro-batch-size 4 --output-dir .scratch/finetune_smoke

Outputs per dimension (same layout as src.train): config.json, training_log.csv,
predictions, evaluation_report.md, plots, and model/ (save_pretrained weights
plus tokenizer, loadable with AutoModelForSequenceClassification). Combined PNGs
are written to the output root. Requires torch, transformers and matplotlib.
"""

import argparse
import csv
import gc
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

if __package__:
    from .plot import plot_score_range_overview, plot_test_metrics, plot_test_ranking
    from .test import cells_from_predictions, macro_mae, report_from_predictions, write_csv
    from .train import DIMENSIONS, LanguageBalancedSampler, load_split_manifest
else:
    from plot import plot_score_range_overview, plot_test_metrics, plot_test_ranking
    from test import cells_from_predictions, macro_mae, report_from_predictions, write_csv
    from train import DIMENSIONS, LanguageBalancedSampler, load_split_manifest

DEFAULT_ENCODER = "jhu-clsp/mmBERT-base"

DEFAULT_CONFIG = {
    "pooling": "cls",
    "max_length": 4096,
    "batch_size": 16,
    "micro_batch_size": 4,
    "eval_batch_size": 16,
    "lr": 2e-5,
    "head_lr": 1e-3,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "max_epochs": 10,
    "patience": 3,
    "min_delta": 0.001,
    "grad_clip_norm": 1.0,
    "seed": 42,
}


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

def load_texts(document_table: Path) -> dict[tuple[str, str], str]:
    with document_table.open(encoding="utf-8") as f:
        return {(r["language"], r["doc_id"]): r["text"] for r in csv.DictReader(f)}


def build_dataset(manifest_rows: list[dict], token_ids: dict[tuple[str, str], list[int]]) -> dict[str, list[dict]]:
    by_split: dict[str, list[dict]] = defaultdict(list)
    missing = 0
    for row in manifest_rows:
        ids = token_ids.get((row["language"], row["doc_id"]))
        if ids is None:
            missing += 1
            continue
        by_split[row["split"]].append(
            {"doc_id": row["doc_id"], "language": row["language"], "score": row["score"], "input_ids": ids}
        )
    if missing:
        print(f"  warning: {missing} manifest rows had no text in the document table and were skipped")
    return by_split


def tokenize_documents(tokenizer, texts: dict[tuple[str, str], str], keys: set, max_length: int) -> dict:
    ordered = sorted(keys)
    ids = tokenizer([texts[k] for k in ordered], truncation=True, max_length=max_length)["input_ids"]
    truncated = sum(1 for x in ids if len(x) >= max_length)
    print(f"Tokenized {len(ordered)} documents (max_length={max_length}); "
          f"{truncated} at/over the limit, longest={max(len(x) for x in ids)} tokens")
    return dict(zip(ordered, ids))


def collate(records: list[dict], pad_id: int, device):
    longest = max(len(r["input_ids"]) for r in records)
    input_ids = torch.full((len(records), longest), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(records), longest), dtype=torch.long)
    for i, r in enumerate(records):
        n = len(r["input_ids"])
        input_ids[i, :n] = torch.tensor(r["input_ids"])
        attention_mask[i, :n] = 1
    targets = torch.tensor([r["score"] for r in records], dtype=torch.float32)
    return input_ids.to(device), attention_mask.to(device), targets.to(device)


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------

def load_model(encoder: str, pooling: str, train_mean: float, gradient_checkpointing: bool, device):
    from transformers import AutoModelForSequenceClassification

    model = AutoModelForSequenceClassification.from_pretrained(
        encoder, num_labels=1, problem_type="regression", classifier_pooling=pooling
    )
    with torch.no_grad():
        model.classifier.bias.fill_(train_mean)
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
    return model.to(device)


def build_optimizer(model, lr: float, head_lr: float, weight_decay: float) -> torch.optim.Optimizer:
    """Encoder + pretrained prediction head at `lr`; only the new classifier at
    `head_lr`. Biases and norm weights get no weight decay."""
    groups: dict[tuple[bool, bool], list] = defaultdict(list)
    for name, param in model.named_parameters():
        is_new = name.startswith("classifier.")
        no_decay = name.endswith("bias") or "norm" in name
        groups[(is_new, no_decay)].append(param)
    param_groups = [
        {"params": params, "lr": head_lr if is_new else lr, "weight_decay": 0.0 if no_decay else weight_decay}
        for (is_new, no_decay), params in groups.items()
    ]
    return torch.optim.AdamW(param_groups)


def autocast_ctx(device, use_bf16: bool):
    if use_bf16 and torch.device(device).type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return torch.autocast(device_type="cpu", enabled=False)


@torch.no_grad()
def predict_records(model, records: list[dict], pad_id: int, device, batch_size: int, use_bf16: bool) -> list[float]:
    """Raw predictions in the same order as `records`."""
    if not records:
        return []
    model.eval()
    order = sorted(range(len(records)), key=lambda i: len(records[i]["input_ids"]))
    preds = [0.0] * len(records)
    for start in range(0, len(order), batch_size):
        idx = order[start:start + batch_size]
        input_ids, attention_mask, _ = collate([records[i] for i in idx], pad_id, device)
        with autocast_ctx(device, use_bf16):
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        for i, p in zip(idx, logits.squeeze(-1).float().cpu().tolist()):
            preds[i] = p
    return preds


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------

def train_dimension(dimension: str, dataset: dict[str, list[dict]], tokenizer, cfg: dict, device):
    from transformers import get_cosine_schedule_with_warmup

    seed = cfg["seed"]
    torch.manual_seed(seed)
    random.seed(seed)

    train, val = dataset.get("train", []), dataset.get("validation", [])
    if not train:
        raise SystemExit(f"No training records for dimension={dimension}")
    pad_id = tokenizer.pad_token_id
    use_bf16 = cfg["bf16"]

    train_mean = sum(r["score"] for r in train) / len(train)
    model = load_model(cfg["encoder"], cfg["pooling"], train_mean, cfg["gradient_checkpointing"], device)
    optimizer = build_optimizer(model, cfg["lr"], cfg["head_lr"], cfg["weight_decay"])

    indices_by_lang: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(train):
        indices_by_lang[r["language"]].append(i)
    sampler = LanguageBalancedSampler(indices_by_lang, cfg["batch_size"], seed=seed)

    steps_per_epoch = math.ceil(len(train) / cfg["batch_size"])
    total_steps = steps_per_epoch * cfg["max_epochs"]
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(cfg["warmup_ratio"] * total_steps), total_steps)

    best_val_macro_mae, best_state, best_epoch = float("inf"), None, -1
    epochs_without_improvement = 0
    log_rows = []

    for epoch in range(1, cfg["max_epochs"] + 1):
        model.train()
        epoch_loss, started = 0.0, time.time()
        for _ in range(steps_per_epoch):
            batch = [train[i] for i in sampler.sample_batch()]
            batch.sort(key=lambda r: len(r["input_ids"]))  # less padding per micro-batch
            optimizer.zero_grad(set_to_none=True)
            step_loss = 0.0
            for s in range(0, len(batch), cfg["micro_batch_size"]):
                micro = batch[s:s + cfg["micro_batch_size"]]
                input_ids, attention_mask, targets = collate(micro, pad_id, device)
                with autocast_ctx(device, use_bf16):
                    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits.squeeze(-1)
                loss = F.mse_loss(logits.float(), targets)
                (loss * len(micro) / len(batch)).backward()
                step_loss += loss.item() * len(micro) / len(batch)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg["grad_clip_norm"])
            optimizer.step()
            scheduler.step()
            epoch_loss += step_loss
        epoch_loss /= steps_per_epoch

        val_raw = predict_records(model, val, pad_id, device, cfg["eval_batch_size"], use_bf16)
        val_macro = macro_mae(cells_from_predictions(val, val_raw)) if val else float("nan")
        log_rows.append({"epoch": epoch, "train_loss": epoch_loss, "val_macro_mae": val_macro})

        improved = val_macro < best_val_macro_mae - cfg["min_delta"]
        if improved:
            best_val_macro_mae, best_epoch = val_macro, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        print(f"    epoch={epoch:2d} train_loss={epoch_loss:.4f} val_macro_mae={val_macro:.4f}"
              f" ({time.time() - started:.0f}s){' *' if improved else ''}")
        if epochs_without_improvement >= cfg["patience"]:
            print(f"    early stopping at epoch {epoch} (best epoch {best_epoch}, val_macro_mae={best_val_macro_mae:.4f})")
            break

    if best_state is None:
        raise SystemExit("Training never produced a valid validation score; check the validation split.")
    model.load_state_dict(best_state)
    return model, best_epoch, best_val_macro_mae, log_rows


def run_dimension(dimension: str, args, cfg: dict, tokenizer, token_ids: dict, device):
    print(f"\n=== {dimension} (pooling={cfg['pooling']}) ===")
    manifest_rows = load_split_manifest(args.split_manifest_dir / f"split_manifest_{dimension}.csv")
    dataset = build_dataset(manifest_rows, token_ids)
    if args.limit:
        dataset = {split: recs[: args.limit] for split, recs in dataset.items()}
    for split in ["train", "validation", "test"]:
        print(f"  {split}: {len(dataset.get(split, []))} documents")

    out_dir = args.output_dir / dimension
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps({"dimension": dimension, **cfg}, indent=2), encoding="utf-8")

    model, best_epoch, best_val_macro_mae, log_rows = train_dimension(dimension, dataset, tokenizer, cfg, device)
    write_csv(out_dir / "training_log.csv", log_rows, ["epoch", "train_loss", "val_macro_mae"])

    pad_id = tokenizer.pad_token_id
    val_raw = predict_records(model, dataset.get("validation", []), pad_id, device, cfg["eval_batch_size"], cfg["bf16"])
    test_raw = predict_records(model, dataset.get("test", []), pad_id, device, cfg["eval_batch_size"], cfg["bf16"])
    test_cells, range_rows = report_from_predictions(dataset, val_raw, test_raw, dimension, out_dir, best_epoch)

    if not args.no_save_model:
        model.save_pretrained(out_dir / "model")
        tokenizer.save_pretrained(out_dir / "model")
    print(f"  best_epoch={best_epoch} val_macro_mae={best_val_macro_mae:.4f} "
          f"test_macro_mae={macro_mae(test_cells):.4f}")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return test_cells, range_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dimension", required=True, choices=DIMENSIONS + ["all"])
    parser.add_argument("--pooling", choices=["cls", "mean"], default=DEFAULT_CONFIG["pooling"])
    parser.add_argument("--encoder", default=DEFAULT_ENCODER)
    parser.add_argument("--document-table", default="prepared_data/document_table.csv", type=Path)
    parser.add_argument("--split-manifest-dir", default="prepared_data", type=Path)
    parser.add_argument("--output-dir", default=None, type=Path,
                        help="Default: checkpoints_finetune/<pooling>")
    parser.add_argument("--max-length", type=int, default=DEFAULT_CONFIG["max_length"])
    parser.add_argument("--batch-size", type=int, default=DEFAULT_CONFIG["batch_size"],
                        help="Effective batch per optimizer step (language-balanced)")
    parser.add_argument("--micro-batch-size", type=int, default=DEFAULT_CONFIG["micro_batch_size"],
                        help="Documents per forward/backward pass; lower it if the GPU runs out of memory")
    parser.add_argument("--eval-batch-size", type=int, default=DEFAULT_CONFIG["eval_batch_size"])
    parser.add_argument("--lr", type=float, default=DEFAULT_CONFIG["lr"], help="Encoder + pretrained head")
    parser.add_argument("--head-lr", type=float, default=DEFAULT_CONFIG["head_lr"], help="New final classifier layer")
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_CONFIG["weight_decay"])
    parser.add_argument("--warmup-ratio", type=float, default=DEFAULT_CONFIG["warmup_ratio"])
    parser.add_argument("--max-epochs", type=int, default=DEFAULT_CONFIG["max_epochs"])
    parser.add_argument("--patience", type=int, default=DEFAULT_CONFIG["patience"])
    parser.add_argument("--min-delta", type=float, default=DEFAULT_CONFIG["min_delta"])
    parser.add_argument("--grad-clip-norm", type=float, default=DEFAULT_CONFIG["grad_clip_norm"])
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG["seed"])
    parser.add_argument("--no-bf16", action="store_true", help="Disable bf16 autocast on GPU")
    parser.add_argument("--gradient-checkpointing", action="store_true", help="Trade speed for memory")
    parser.add_argument("--no-save-model", action="store_true", help="Skip saving model weights (~600 MB per dimension)")
    parser.add_argument("--limit", type=int, default=0, help="Debug: keep only the first N documents per split")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = Path("checkpoints_finetune") / args.pooling
    if args.micro_batch_size > args.batch_size:
        raise SystemExit("--micro-batch-size cannot exceed --batch-size")

    cfg = {
        "encoder": args.encoder,
        "pooling": args.pooling,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "micro_batch_size": args.micro_batch_size,
        "eval_batch_size": args.eval_batch_size,
        "lr": args.lr,
        "head_lr": args.head_lr,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "min_delta": args.min_delta,
        "grad_clip_norm": args.grad_clip_norm,
        "seed": args.seed,
        "bf16": not args.no_bf16,
        "gradient_checkpointing": args.gradient_checkpointing,
    }

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.encoder)
    dimensions = DIMENSIONS if args.dimension == "all" else [args.dimension]

    keys = set()
    for dimension in dimensions:
        for row in load_split_manifest(args.split_manifest_dir / f"split_manifest_{dimension}.csv"):
            keys.add((row["language"], row["doc_id"]))
    texts = load_texts(args.document_table)
    token_ids = tokenize_documents(tokenizer, texts, keys & set(texts), args.max_length)
    del texts

    results, ranges_by_dimension = {}, {}
    for dimension in dimensions:
        results[dimension], ranges_by_dimension[dimension] = run_dimension(
            dimension, args, cfg, tokenizer, token_ids, args.device
        )
    plot_test_ranking(args.output_dir / "test_ranking.png", results)
    plot_test_metrics(args.output_dir / "test_metrics.png", results,
                      provenance=f"Fine-tuned {args.encoder}, {args.pooling} pooling.")
    plot_score_range_overview(args.output_dir / "test_score_ranges.png", ranges_by_dimension)


if __name__ == "__main__":
    main()
