"""Build the frozen multilingual document-embedding cache (01_train_rater.md
section 5). Runs once; every dimension's rater training reuses this cache.

For each document in prepared_data/document_table.csv:
  1. Tokenize without truncation.
  2. Split into non-overlapping chunks of at most --max-chunk-tokens tokens
     (including special tokens).
  3. Encode each chunk with the frozen encoder and pool the final hidden
     state (--pooling): `mean` averages content tokens only (padding/special
     tokens excluded), `cls` takes the first-token (CLS) state, `cls+mean`
     concatenates both (2x hidden size).
  4. Combine chunk embeddings with a weighted mean (weight = content-token
     count per chunk) into one embedding per document.

Requires a GPU-capable environment with `torch` and `transformers` installed
(see run_job.sh / script_cpu.sh) -- this does not run in a plain sandbox.

Usage:
    python -m src.build_embeddings
    python -m src.build_embeddings --pooling cls      # -> prepared_data/embeddings_cls
    python -m src.build_embeddings --encoder jhu-clsp/mmBERT-base --revision main \
        --document-table prepared_data/document_table.csv \
        --output-dir prepared_data/embeddings
"""

import argparse
import csv
import json
from pathlib import Path

import torch

DEFAULT_ENCODER = "jhu-clsp/mmBERT-base"
DEFAULT_MAX_CHUNK_TOKENS = 2048

# Recorded in the cache config so a cache built with one pooling is never
# reused for another. The `mean` string is unchanged from earlier caches.
POOLING_CONFIG_NAMES = {
    "mean": "content_token_mean_then_token_count_weighted_chunk_mean",
    "cls": "cls_token_then_token_count_weighted_chunk_mean",
    "cls+mean": "cls_and_content_token_mean_concat_then_token_count_weighted_chunk_mean",
}


def load_documents(document_table: Path) -> list[dict]:
    with document_table.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [{"doc_id": r["doc_id"], "language": r["language"], "text": r["text"]} for r in rows]


@torch.no_grad()
def embed_document(text: str, tokenizer, model, device, max_chunk_tokens: int, pooling: str = "mean") -> torch.Tensor:
    # Let the tokenizer's own truncation/overflow machinery do the chunking
    # (stride=0 -> non-overlapping windows, special tokens added per chunk).
    # Manually re-wrapping raw ids with build_inputs_with_special_tokens() /
    # prepare_for_model() is not portable across tokenizer backends (e.g.
    # mmBERT/ModernBERT's Rust-based TokenizersBackend implements neither);
    # this call is verified to produce the same non-overlapping chunk count
    # as the old manual split, against the real mmBERT-base tokenizer.
    encoded = tokenizer(
        text,
        add_special_tokens=True,
        truncation=True,
        max_length=max_chunk_tokens,
        stride=0,
        return_overflowing_tokens=True,
        return_attention_mask=True,
        return_special_tokens_mask=True,
    )

    chunk_embeddings = []
    chunk_weights = []
    for ids, attn, special in zip(
        encoded["input_ids"], encoded["attention_mask"], encoded["special_tokens_mask"]
    ):
        input_ids = torch.tensor([ids], device=device)
        attention_mask = torch.tensor([attn], device=device)
        special_mask = torch.tensor(special, device=device).bool()

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state[0]  # [seq_len, H]

        content_mask = attention_mask[0].bool() & ~special_mask
        num_content = int(content_mask.sum().item())
        if num_content == 0:
            continue  # degenerate chunk (only special tokens); skip rather than divide by zero
        mean_vec = hidden[content_mask].mean(dim=0)
        if pooling == "mean":
            pooled = mean_vec
        else:
            if not bool(special_mask[0]):
                raise RuntimeError("Position 0 is not a special (CLS) token; cannot use CLS pooling")
            cls_vec = hidden[0]
            pooled = cls_vec if pooling == "cls" else torch.cat([cls_vec, mean_vec], dim=0)
        if not torch.isfinite(pooled).all():
            raise RuntimeError("Non-finite embedding produced; check precision/config")
        chunk_embeddings.append(pooled)
        chunk_weights.append(num_content)

    if not chunk_embeddings:
        raise RuntimeError("Document produced no content tokens at all")

    weights = torch.tensor(chunk_weights, dtype=torch.float32, device=device)
    weights = weights / weights.sum()
    stacked = torch.stack(chunk_embeddings, dim=0)
    return (stacked * weights.unsqueeze(1)).sum(dim=0), len(chunk_embeddings), sum(chunk_weights)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--document-table", default="prepared_data/document_table.csv", type=Path)
    parser.add_argument("--output-dir", default=None, type=Path,
                        help="Default: prepared_data/embeddings for mean pooling, "
                             "prepared_data/embeddings_<pooling> otherwise (never overwrites the mean cache)")
    parser.add_argument("--pooling", default="mean", choices=list(POOLING_CONFIG_NAMES))
    parser.add_argument("--encoder", default=DEFAULT_ENCODER)
    parser.add_argument("--revision", default=None, help="Pin a specific model/tokenizer revision once verified")
    parser.add_argument("--max-chunk-tokens", default=DEFAULT_MAX_CHUNK_TOKENS, type=int)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--force", action="store_true", help="Recompute even if a matching cache already exists")
    args = parser.parse_args()

    from transformers import AutoModel, AutoTokenizer  # deferred: only needed here, not for smoke tests

    config = {
        "encoder": args.encoder,
        "revision": args.revision,
        "max_chunk_tokens": args.max_chunk_tokens,
        "dtype": args.dtype,
        "pooling": POOLING_CONFIG_NAMES[args.pooling],
    }

    if args.output_dir is None:
        suffix = "" if args.pooling == "mean" else "_" + args.pooling.replace("+", "_")
        args.output_dir = Path("prepared_data") / f"embeddings{suffix}"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.json"
    embeddings_path = args.output_dir / "embeddings.pt"

    if not args.force and manifest_path.exists() and embeddings_path.exists():
        existing = json.loads(manifest_path.read_text())
        if existing.get("config") == config:
            print(f"Matching embedding cache already at {args.output_dir} (use --force to recompute)")
            return

    documents = load_documents(args.document_table)
    print(f"Embedding {len(documents)} documents with {args.encoder} "
          f"(revision={args.revision or 'default'}) on {args.device}, dtype={args.dtype}, pooling={args.pooling}")

    tokenizer = AutoTokenizer.from_pretrained(args.encoder, revision=args.revision)
    model = AutoModel.from_pretrained(args.encoder, revision=args.revision)
    model = model.to(args.device).to(getattr(torch, args.dtype))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    doc_ids, languages, vectors, provenance = [], [], [], []
    for i, doc in enumerate(documents):
        embedding, num_chunks, num_content_tokens = embed_document(
            doc["text"], tokenizer, model, args.device, args.max_chunk_tokens, args.pooling
        )
        doc_ids.append(doc["doc_id"])
        languages.append(doc["language"])
        vectors.append(embedding.float().cpu())
        provenance.append({"doc_id": doc["doc_id"], "num_chunks": num_chunks, "num_content_tokens": num_content_tokens})
        if (i + 1) % 200 == 0 or (i + 1) == len(documents):
            print(f"  {i + 1}/{len(documents)} embedded")

    embeddings = torch.stack(vectors, dim=0)  # [N, H]
    torch.save({"doc_ids": doc_ids, "languages": languages, "embeddings": embeddings}, embeddings_path)

    manifest = {
        "config": config,
        "num_documents": len(doc_ids),
        "embedding_dim": embeddings.shape[1],
        "installed_versions": {
            "torch": torch.__version__,
        },
        "provenance": provenance,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {embeddings.shape} embeddings to {embeddings_path} and manifest to {manifest_path}")


if __name__ == "__main__":
    main()
