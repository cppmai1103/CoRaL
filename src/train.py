"""Train SEA-Rater regression heads and generate evaluation reports and plots.

Each dimension uses its own split and the same frozen embedding cache.
Train with MSE and language-balanced batches; select checkpoints by validation
macro-MAE. One run per dimension uses a fixed internal seed for reproducibility.

Run from the project root:
    python -m src.train --dimension all
    python src/train.py --dimension educational_value

Outputs per dimension: checkpoint.pt, config.json, predictions, reports,
test_score_ranges.png, test_ranking.png, and test_metrics.png. Combined PNGs
are saved in the output root. Requires torch and matplotlib.
"""

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

if __package__:
    from .test import evaluate_and_report, evaluate_split, macro_mae, write_csv
    from .plot import plot_score_range_overview, plot_test_metrics, plot_test_ranking
else:
    from test import evaluate_and_report, evaluate_split, macro_mae, write_csv
    from plot import plot_score_range_overview, plot_test_metrics, plot_test_ranking

DIMENSIONS = [
    "educational_value",
    "reasoning",
    "professionalism",
    "cleanliness",
    "cultural_nuances",
]

TRAINING_SEED = 42

DEFAULT_CONFIG = {
    "hidden_size": 128,
    "dropout": 0.1,
    "lr": 1e-3,
    "weight_decay": 0.01,
    "batch_size": 64,
    "max_epochs": 30,
    "patience": 5,
    "min_delta": 0.001,
    "grad_clip_norm": 1.0,
}


def load_embeddings(path: Path) -> dict[tuple[str, str], torch.Tensor]:
    payload = torch.load(path, map_location="cpu")
    if "languages" not in payload:
        raise ValueError("Embedding cache has no languages; rebuild it with python -m src.build_embeddings")
    if not (len(payload["languages"]) == len(payload["doc_ids"]) == len(payload["embeddings"])):
        raise ValueError("Embedding cache languages, document IDs, and vectors have different lengths")
    embeddings = {}
    for i, (language, doc_id) in enumerate(zip(payload["languages"], payload["doc_ids"])):
        key = (language, doc_id)
        if key in embeddings:
            raise ValueError(f"Duplicate (language, document ID) in embedding cache: {key!r}")
        embeddings[key] = payload["embeddings"][i]
    return embeddings


def load_split_manifest(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["score"] = float(r["score"])
    return rows


def build_dataset(manifest_rows: list[dict], embeddings: dict[tuple[str, str], torch.Tensor]) -> dict[str, list[dict]]:
    """Returns {split_name: [{'doc_id','language','score','embedding'}, ...]}."""
    by_split: dict[str, list[dict]] = defaultdict(list)
    missing = 0
    for row in manifest_rows:
        emb = embeddings.get((row["language"], row["doc_id"]))
        if emb is None:
            missing += 1
            continue
        by_split[row["split"]].append(
            {"doc_id": row["doc_id"], "language": row["language"], "score": row["score"], "embedding": emb}
        )
    if missing:
        print(f"  warning: {missing} manifest rows had no cached embedding and were skipped")
    return by_split


class LanguageBalancedSampler:
    """Draws batches with as-equal-as-possible language counts, cycling and
    reshuffling each language's pool as it is exhausted. When batch_size does
    not divide evenly across languages, which languages get the extra slot
    rotates batch-to-batch so exposure stays balanced over an epoch."""

    def __init__(self, indices_by_language: dict[str, list[int]], batch_size: int, seed: int):
        self.languages = sorted(indices_by_language)
        self.pools = {lang: list(idxs) for lang, idxs in indices_by_language.items()}
        self.queues = {lang: [] for lang in self.languages}
        self.batch_size = batch_size
        self.rng = random.Random(seed)
        self.exposures = {lang: 0 for lang in self.languages}
        self._rotation = 0

    def _refill(self, lang: str) -> None:
        pool = list(self.pools[lang])
        self.rng.shuffle(pool)
        self.queues[lang] = pool

    def _draw_one(self, lang: str) -> int:
        if not self.queues[lang]:
            self._refill(lang)
        self.exposures[lang] += 1
        return self.queues[lang].pop()

    def sample_batch(self) -> list[int]:
        n_lang = len(self.languages)
        base, remainder = divmod(self.batch_size, n_lang)
        counts = [base] * n_lang
        for k in range(remainder):
            counts[(self._rotation + k) % n_lang] += 1
        self._rotation = (self._rotation + remainder) % n_lang if remainder else self._rotation

        batch = []
        for lang, count in zip(self.languages, counts):
            for _ in range(count):
                batch.append(self._draw_one(lang))
        self.rng.shuffle(batch)
        return batch


def build_head(embedding_dim: int, hidden_size: int, dropout: float) -> nn.Module:
    return nn.Sequential(
        nn.Linear(embedding_dim, hidden_size),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_size, 1),
    )


def build_optimizer(head: nn.Module, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    decay, no_decay = [], []
    for name, param in head.named_parameters():
        (decay if name.endswith("weight") else no_decay).append(param)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=lr,
    )


def train_head(dimension: str, dataset: dict[str, list[dict]], embedding_dim: int, cfg: dict, device):
    torch.manual_seed(TRAINING_SEED)
    random.seed(TRAINING_SEED)

    train, val = dataset.get("train", []), dataset.get("validation", [])
    if not train:
        raise SystemExit(f"No training records for dimension={dimension}")

    indices_by_lang: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(train):
        indices_by_lang[r["language"]].append(i)
    sampler = LanguageBalancedSampler(indices_by_lang, cfg["batch_size"], seed=TRAINING_SEED)

    head = build_head(embedding_dim, cfg["hidden_size"], cfg["dropout"]).to(device)
    optimizer = build_optimizer(head, cfg["lr"], cfg["weight_decay"])

    steps_per_epoch = math.ceil(len(train) / cfg["batch_size"])
    best_val_macro_mae = float("inf")
    best_state = None
    best_epoch = -1
    epochs_without_improvement = 0
    log_rows = []

    for epoch in range(1, cfg["max_epochs"] + 1):
        head.train()
        epoch_loss = 0.0
        for _ in range(steps_per_epoch):
            batch_idx = sampler.sample_batch()
            batch = [train[i] for i in batch_idx]
            embeddings = torch.stack([r["embedding"] for r in batch]).float().to(device)
            targets = torch.tensor([r["score"] for r in batch], dtype=torch.float32, device=device)

            preds = head(embeddings).squeeze(-1)
            loss = F.mse_loss(preds, targets)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), max_norm=cfg["grad_clip_norm"])
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= steps_per_epoch

        val_cells = evaluate_split(head, val, device) if val else {}
        val_macro = macro_mae(val_cells) if val_cells else float("nan")
        log_rows.append({"epoch": epoch, "train_loss": epoch_loss, "val_macro_mae": val_macro})

        improved = val_macro < best_val_macro_mae - cfg["min_delta"]
        if improved:
            best_val_macro_mae = val_macro
            best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        print(f"    epoch={epoch:2d} train_loss={epoch_loss:.4f} val_macro_mae={val_macro:.4f}"
              f"{' *' if improved else ''}")
        if epochs_without_improvement >= cfg["patience"]:
            print(f"    early stopping at epoch {epoch} (best epoch {best_epoch}, val_macro_mae={best_val_macro_mae:.4f})")
            break

    if best_state is None:
        raise SystemExit("Training never produced a valid validation score; check the validation split.")
    head.load_state_dict(best_state)
    return head, best_epoch, best_val_macro_mae, log_rows, sampler.exposures


def run_dimension(dimension: str, args, cfg: dict, device) -> tuple[dict[str, dict], list[dict]]:
    print(f"\n=== {dimension} ===")
    embeddings = load_embeddings(args.embeddings)
    embedding_dim = next(iter(embeddings.values())).shape[0]
    manifest_rows = load_split_manifest(args.split_manifest_dir / f"split_manifest_{dimension}.csv")
    dataset = build_dataset(manifest_rows, embeddings)
    for split in ["train", "validation", "test"]:
        print(f"  {split}: {len(dataset.get(split, []))} documents")

    out_dir = args.output_dir / dimension
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps({"dimension": dimension, **cfg}, indent=2), encoding="utf-8")

    head, best_epoch, best_val_macro_mae, log_rows, exposures = train_head(
        dimension, dataset, embedding_dim, cfg, device
    )
    write_csv(out_dir / "training_log.csv", log_rows, ["epoch", "train_loss", "val_macro_mae"])
    torch.save(
        {
            "state_dict": head.state_dict(),
            "epoch": best_epoch,
            "val_macro_mae": best_val_macro_mae,
            "dimension": dimension,
            "seed": TRAINING_SEED,
            "embedding_dim": embedding_dim,
            "language_exposures": exposures,
        },
        out_dir / "checkpoint.pt",
    )

    test_cells, range_rows = evaluate_and_report(
        head, dataset, dimension, out_dir, device, best_epoch
    )
    print(f"  best_epoch={best_epoch} val_macro_mae={best_val_macro_mae:.4f} "
          f"test_macro_mae={macro_mae(test_cells):.4f}")
    return test_cells, range_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dimension", required=True, choices=DIMENSIONS + ["all"])
    parser.add_argument("--split-manifest-dir", default="prepared_data", type=Path)
    parser.add_argument("--embeddings", default="prepared_data/embeddings/embeddings.pt", type=Path)
    parser.add_argument("--output-dir", default="checkpoints", type=Path)
    parser.add_argument("--hidden-size", type=int, default=DEFAULT_CONFIG["hidden_size"])
    parser.add_argument("--dropout", type=float, default=DEFAULT_CONFIG["dropout"])
    parser.add_argument("--lr", type=float, default=DEFAULT_CONFIG["lr"])
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_CONFIG["weight_decay"])
    parser.add_argument("--batch-size", type=int, default=DEFAULT_CONFIG["batch_size"])
    parser.add_argument("--max-epochs", type=int, default=DEFAULT_CONFIG["max_epochs"])
    parser.add_argument("--patience", type=int, default=DEFAULT_CONFIG["patience"])
    parser.add_argument("--min-delta", type=float, default=DEFAULT_CONFIG["min_delta"])
    parser.add_argument("--grad-clip-norm", type=float, default=DEFAULT_CONFIG["grad_clip_norm"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg = {
        "hidden_size": args.hidden_size,
        "dropout": args.dropout,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "min_delta": args.min_delta,
        "grad_clip_norm": args.grad_clip_norm,
    }

    dimensions = DIMENSIONS if args.dimension == "all" else [args.dimension]
    results, ranges_by_dimension = {}, {}
    for dimension in dimensions:
        results[dimension], ranges_by_dimension[dimension] = run_dimension(dimension, args, cfg, args.device)
    plot_test_ranking(args.output_dir / "test_ranking.png", results)
    plot_test_metrics(args.output_dir / "test_metrics.png", results)
    plot_score_range_overview(args.output_dir / "test_score_ranges.png", ranges_by_dimension)


if __name__ == "__main__":
    main()
