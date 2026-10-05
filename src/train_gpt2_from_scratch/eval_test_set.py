"""Loss, perplexity and bits per byte of trained GPT-2 pilot runs on the pilot corpus held-out TEST split.

    Test split: data/pilot_corpus/split_manifest.csv rows with split=test (1,000 documents per language, chosen
    at random by website when the corpus was split, never trained on). Documents are encoded and packed exactly
    as train_gpt2.py does for its final evaluation (same loader, one EOS per document, seeded pack_streams with the
    run's seed and the model's context length), and scored with the same ordinary unweighted next-token loss -- so
    for a run trained with these defaults, loss/perplexity reproduce the "test" section of its results.json.

    Bits per byte = summed next-token loss (nats) / ln 2 / UTF-8 bytes of the documents. Unlike loss and perplexity,
    which are per TOKEN, it does not depend on how the tokenizer splits text: use it to compare runs trained with
    different tokenizers (e.g. SeaLLM 48K vs the custom 16K BPE). Loss/perplexity are compared only among runs
    with the same tokenizer; the comparison file flags runs whose tokenizer differs from the first (baseline) run.

    The first --run-dir is the baseline; the comparison reports each run's loss and bits/byte difference to it.

Outputs:
    <run-dir>/<split>_eval/results.json, summary.md     per run
    --comparison-file (default <first run's parent>/<split>_comparison.md) when more than one run is given

Run: python -m src.train_gpt2_from_scratch.eval_test_set --run-dir <baseline> <run> ... --device cuda
"""

import argparse
import json
import math
from pathlib import Path

import torch

from src.train_gpt2_from_scratch import train_gpt2 as base


def bits_per_byte(loss: float, target_tokens: int, n_bytes: int) -> float:
    return loss * target_tokens / math.log(2) / n_bytes if n_bytes else float("nan")


def add_bits_per_byte(result: dict, bytes_per_language: dict[str, int]) -> dict:
    """Bits/byte per language, macro (mean over languages) and byte-weighted overall, added in place."""
    nats = total_bytes = 0.0
    for language, v in result["per_language"].items():
        n_bytes = bytes_per_language[language]
        v["bytes"] = n_bytes
        v["bits_per_byte"] = bits_per_byte(v["loss"], v["target_tokens"], n_bytes)
        nats += v["loss"] * v["target_tokens"]
        total_bytes += n_bytes
    values = [v["bits_per_byte"] for v in result["per_language"].values()]
    result["macro"]["bits_per_byte"] = sum(values) / len(values)
    result["token_weighted"]["bits_per_byte"] = nats / math.log(2) / total_bytes
    result["token_weighted"]["bytes"] = int(total_bytes)
    return result


def tokenizer_id(run_dir: Path) -> str:
    try:
        return json.loads((run_dir / "config.json").read_text(encoding="utf-8")).get("tokenizer") or "?"
    except FileNotFoundError:
        return "?"


def run_seed(run_dir: Path, default: int) -> int:
    try:
        return int(json.loads((run_dir / "config.json").read_text(encoding="utf-8")).get("seed", default))
    except FileNotFoundError:
        return default


def result_table(result: dict) -> list[str]:
    lines = ["| Language | Loss | Perplexity | Bits/byte | Target tokens | Bytes |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for language, v in result["per_language"].items():
        lines.append(f"| {language} | {v['loss']:.4f} | {v['perplexity']:.2f} | {v['bits_per_byte']:.4f} | "
                     f"{v['target_tokens']:,} | {v['bytes']:,} |")
    m, t = result["macro"], result["token_weighted"]
    lines.append(f"| **macro average** | {m['loss']:.4f} | {m['perplexity']:.2f} | {m['bits_per_byte']:.4f} | | |")
    lines.append(f"| **token-weighted** | {t['loss']:.4f} | {t['perplexity']:.2f} | {t['bits_per_byte']:.4f} | "
                 f"{t['target_tokens']:,} | {t['bytes']:,} |")
    return lines


def write_summary(path: Path, name: str, split: str, tokenizer: str, result: dict) -> None:
    lines = [f"# {name}: pilot corpus {split} split", "", f"Tokenizer: `{tokenizer}`. Ordinary unweighted next-token "
             "loss; perplexity = exp(loss); bits/byte = summed loss / ln 2 / UTF-8 bytes.", ""] + result_table(result)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_comparison(path: Path, split: str, runs: list[dict]) -> None:
    base_run = runs[0]
    languages = list(base_run["result"]["per_language"])
    lines = [f"# GPT-2 pilot: {split} split comparison", "",
             f"Baseline (first run): `{base_run['name']}`. Δ = run − baseline; negative is better.", "",
             "Loss/perplexity are per token: compare them only between runs with the same tokenizer. Bits/byte is "
             "tokenizer-independent.", "", "## Runs", "", "| Run | Tokenizer | Same tokenizer as baseline |",
             "| --- | --- | --- |"]
    for r in runs:
        lines.append(f"| {r['name']} | `{r['tokenizer']}` | {'yes' if r['tokenizer'] == base_run['tokenizer'] else '**no**'} |")
    for label, key in [("macro average over languages", "macro"), ("token-weighted overall", "token_weighted")]:
        lines += ["", f"## Overall ({label})", "", "| Run | Loss | Δ loss | Perplexity | Bits/byte | Δ bits/byte |",
                  "| --- | ---: | ---: | ---: | ---: | ---: |"]
        b = base_run["result"][key]
        for r in runs:
            v = r["result"][key]
            lines.append(f"| {r['name']} | {v['loss']:.4f} | {v['loss'] - b['loss']:+.4f} | {v['perplexity']:.2f} | "
                         f"{v['bits_per_byte']:.4f} | {v['bits_per_byte'] - b['bits_per_byte']:+.4f} |")
    for metric, fmt in [("loss", "{:.4f}"), ("perplexity", "{:.2f}"), ("bits_per_byte", "{:.4f}")]:
        lines += ["", f"## {metric} per language", "", "| Language | " + " | ".join(r["name"] for r in runs) + " |",
                  "| --- |" + " ---: |" * len(runs)]
        for language in languages:
            cells = []
            for r in runs:
                v = r["result"]["per_language"][language][metric]
                cell = fmt.format(v)
                if r is not base_run and metric != "perplexity":
                    cell += f" ({v - base_run['result']['per_language'][language][metric]:+.4f})"
                cells.append(cell)
            lines.append(f"| {language} | " + " | ".join(cells) + " |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, nargs="+", required=True,
                        help="Run folders with final/ and tokenizer/; the first is the baseline")
    parser.add_argument("--split", choices=["test", "validation"], default="test")
    parser.add_argument("--pool-dir", type=Path, default=Path("data/pilot_corpus"))
    parser.add_argument("--split-manifest", type=Path, default=None, help="Default: <pool-dir>/split_manifest.csv")
    parser.add_argument("--comparison-file", type=Path, default=None,
                        help="Default: <first run's parent>/<split>_comparison.md")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42, help="Packing seed if a run has no config.json")
    parser.add_argument("--limit-docs", type=int, default=0, help="Debug: keep only N documents per language")
    parser.add_argument("--no-bf16", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)

    base.PROGRESS = not args.no_progress
    use_bf16 = not args.no_bf16
    manifest = args.split_manifest or args.pool_dir / "split_manifest.csv"
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading the {split_name(args.split)} documents ...", flush=True)
    docs = base.load_heldout_documents(args.pool_dir, manifest, args.split, args.limit_docs)
    languages = sorted(docs)
    # One EOS is appended per document; it is a target too, but every tokenizer predicts exactly one per document.
    n_bytes = {l: sum(len(t.encode("utf-8")) for t in docs[l]) for l in languages}

    runs = []
    for run_dir in args.run_dir:
        name = run_dir.name
        print(f"=== {name} ===", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(run_dir / "tokenizer")
        tokenizer.pad_token = tokenizer.eos_token  # as in training: the attention mask, not the id, marks padding
        model = AutoModelForCausalLM.from_pretrained(run_dir / "final").to(args.device).eval()
        seq_len, bos = base.eval_format(model, run_dir)  # GPT-2 runs: n_positions, no BOS; LoRA runs: seq_len + BOS
        blocks = base.pack_streams({l: base.encode_documents(tokenizer, docs[l], bos=bos) for l in languages},
                                   languages, seq_len, tokenizer.pad_token_id, run_seed(run_dir, args.seed))
        with torch.no_grad():
            result = base.evaluate(model, blocks, languages, args.batch_size, args.device, use_bf16,
                                   desc=f"{name} {args.split}")
        add_bits_per_byte(result, n_bytes)
        tok = tokenizer_id(run_dir)
        out_dir = run_dir / f"{args.split}_eval"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "results.json").write_text(json.dumps(
            {"run": str(run_dir), "split": args.split, "tokenizer": tok, "vocab_size": len(tokenizer),
             "documents": {l: len(docs[l]) for l in languages}, **result}, indent=2), encoding="utf-8")
        write_summary(out_dir / "summary.md", name, args.split, tok, result)
        m, t = result["macro"], result["token_weighted"]
        print(f"  macro loss {m['loss']:.4f} (ppl {m['perplexity']:.2f}, {m['bits_per_byte']:.4f} bits/byte) | "
              f"token-weighted loss {t['loss']:.4f} (ppl {t['perplexity']:.2f}) -> {out_dir}", flush=True)
        runs.append({"name": name, "tokenizer": tok, "result": result})
        del model
        if args.device != "cpu":
            torch.cuda.empty_cache()

    if len(runs) > 1:
        comparison_file = args.comparison_file or args.run_dir[0].parent / f"{args.split}_comparison.md"
        write_comparison(comparison_file, args.split, runs)
        print(f"Wrote {comparison_file}", flush=True)


def split_name(split: str) -> str:
    return "held-out test" if split == "test" else "validation"


if __name__ == "__main__":
    main()
