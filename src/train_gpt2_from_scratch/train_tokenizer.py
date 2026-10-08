"""Train a small byte-level BPE tokenizer for the SEA pilot languages (fil, indo, khmer, malay, thai, vie).

    Text: only the TRAIN split of data/pilot_corpus (split_manifest.csv), so validation/test documents never
    shape the vocabulary. Languages are balanced by characters: every language contributes the same number of
    characters (default: as many as the smallest language has), drawn from a seeded shuffle of its train
    documents, so the large languages cannot take most of the merges.

    Model: byte-level BPE (GPT-2 style: 256 byte tokens + learned merges, so no unknown tokens in any script),
    NFC-normalized. The pre-tokenizer is GPT-2's word-splitting regex with letters widened to letters + combining
    marks ([\\p{L}\\p{M}]): plain \\p{L}+ would cut Thai vowel/tone marks and Khmer subscript signs off their base
    letter. Thai/Khmer have no spaces between words, so a whole phrase is one pre-token and BPE learns the
    subword units inside it. One special token, <|endoftext|>, used as EOS and padding (train_gpt2.py appends one
    EOS per document itself; encoding adds no special tokens).

    Report: on the VALIDATION split, per language, characters/bytes per token, tokens per whitespace word and an
    exact encode->decode round trip, side by side with --compare tokenizers (default: the SeaLLM v2 tokenizer
    the existing runs use). Train characters/token are not shown: the tokenizer was fitted on that text.

Outputs in --output-dir (default checkpoints/tokenizers/sea_bpe_<vocab/1000>k):
    tokenizer.json  tokenizer_config.json  special_tokens_map.json   (load with AutoTokenizer.from_pretrained)
    training_data.json  report.md  report.json

Run (CPU, a few minutes):
    python -m src.train_gpt2_from_scratch.train_tokenizer --vocab-size 16000
Use it for GPT-2 training: TOKENIZER=checkpoints/tokenizers/sea_bpe_16k sbatch sh/train/train_gpt2_weighted.sh
"""

import argparse
import csv
import json
import random
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

EOS_TOKEN = "<|endoftext|>"
LANGUAGES = ["fil", "indo", "khmer", "malay", "thai", "vie"]
# GPT-2's pre-tokenizer pattern with \p{L} -> [\p{L}\p{M}] so combining marks stay attached to their letter.
PRETOKENIZE_PATTERN = (r"'(?:[sdmt]|ll|ve|re)| ?[\p{L}\p{M}]+| ?\p{N}+| ?[^\s\p{L}\p{M}\p{N}]+|\s+(?!\S)|\s+")


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

def load_split_ids(manifest: Path, split: str) -> dict[str, set[str]]:
    ids: dict[str, set[str]] = defaultdict(set)
    with manifest.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row["split"] == split:
                ids[row["language"]].add(row["doc_id"])
    return dict(ids)


def load_pool_texts(pool_dir: Path, language: str, keep: set[str]) -> dict[str, str]:
    """doc_id -> text for the given ids of one language's pool CSV."""
    csv.field_size_limit(sys.maxsize)
    texts = {}
    with (pool_dir / f"{language}.csv").open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row["doc_id"] in keep:
                texts[row["doc_id"]] = row["text"]
    missing = keep - set(texts)
    if missing:
        raise ValueError(f"{language}: {len(missing)} documents from the manifest are not in the pool CSV")
    return texts


def balanced_sample(texts_by_lang: dict[str, dict[str, str]], chars_per_language: int, seed: int
                    ) -> tuple[dict[str, list[str]], dict[str, dict]]:
    """Per language, whole documents from a seeded shuffle until chars_per_language is reached (0 = as many
    characters as the smallest language has). Returns the chosen texts and per-language counts."""
    totals = {lang: sum(len(t) for t in texts.values()) for lang, texts in texts_by_lang.items()}
    budget = chars_per_language or min(totals.values())
    chosen, stats = {}, {}
    for lang in sorted(texts_by_lang):
        ids = sorted(texts_by_lang[lang])
        random.Random(f"{seed}-{lang}").shuffle(ids)
        picked, chars = [], 0
        for doc_id in ids:
            if chars >= budget:
                break
            text = texts_by_lang[lang][doc_id]
            picked.append(text)
            chars += len(text)
        chosen[lang] = picked
        stats[lang] = {"documents": len(picked), "characters": chars, "available_documents": len(ids),
                       "available_characters": totals[lang]}
    return chosen, stats


def interleave(chosen: dict[str, list[str]]):
    """Round-robin over languages (BPE counts do not depend on order; this just keeps progress even)."""
    longest = max(len(v) for v in chosen.values())
    for i in range(longest):
        for lang in sorted(chosen):
            if i < len(chosen[lang]):
                yield chosen[lang][i]


# --------------------------------------------------------------------------
# Tokenizer
# --------------------------------------------------------------------------

def build_tokenizer():
    from tokenizers import Regex, Tokenizer, decoders, models, normalizers, pre_tokenizers

    tok = Tokenizer(models.BPE())
    tok.normalizer = normalizers.NFC()
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(PRETOKENIZE_PATTERN), behavior="isolated"),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    ])
    tok.decoder = decoders.ByteLevel()
    return tok


def train_tokenizer(texts, vocab_size: int, min_frequency: int = 2, length: int | None = None):
    """Train byte-level BPE on an iterable of texts and wrap it as a transformers fast tokenizer."""
    from tokenizers import pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    tok = build_tokenizer()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, min_frequency=min_frequency, special_tokens=[EOS_TOKEN],
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=True)
    tok.train_from_iterator(texts, trainer=trainer, length=length)
    return PreTrainedTokenizerFast(tokenizer_object=tok, eos_token=EOS_TOKEN, pad_token=EOS_TOKEN,
                                   clean_up_tokenization_spaces=False)


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------

def measure(tokenizer, texts: list[str]) -> dict:
    """Compression and exact round trip of a tokenizer on some documents (texts compared after NFC)."""
    tokens = chars = nbytes = words = exact = 0
    for i in range(0, len(texts), 256):
        batch = [unicodedata.normalize("NFC", t) for t in texts[i:i + 256]]
        ids = tokenizer(batch, add_special_tokens=False)["input_ids"]
        for text, seq in zip(batch, ids):
            tokens += len(seq)
            chars += len(text)
            nbytes += len(text.encode("utf-8"))
            words += len(text.split())
            exact += tokenizer.decode(seq, clean_up_tokenization_spaces=False) == text
    return {"documents": len(texts), "tokens": tokens, "chars_per_token": chars / max(tokens, 1),
            "bytes_per_token": nbytes / max(tokens, 1), "tokens_per_word": tokens / max(words, 1),
            "round_trip_exact": exact / max(len(texts), 1)}


def write_report(out_dir: Path, name: str, vocab_size: int, train_stats: dict, results: dict[str, dict]) -> None:
    lines = [f"# Tokenizer report: {name}", "",
             f"Byte-level BPE, vocab {vocab_size:,} (incl. {EOS_TOKEN}), trained on the train split only.", "",
             "## Training text (balanced by characters)", "",
             "| Language | Documents used | Characters used | Train documents available | Train characters available |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for lang, s in train_stats.items():
        lines.append(f"| {lang} | {s['documents']:,} | {s['characters']:,} | {s['available_documents']:,} | "
                     f"{s['available_characters']:,} |")
    names = list(results)
    lines += ["", "## Validation split", "",
              "Higher characters/token = shorter sequences for the same text. Tokens/word is not meaningful for "
              "thai/khmer (few spaces). Round trip = share of documents decoded back exactly (after NFC).", ""]
    for metric, fmt in [("chars_per_token", "{:.2f}"), ("tokens_per_word", "{:.2f}"), ("round_trip_exact", "{:.1%}")]:
        lines += [f"### {metric}", "", "| Language | " + " | ".join(names) + " |",
                  "| --- |" + " ---: |" * len(names)]
        for lang in next(iter(results.values()))["languages"]:
            lines.append(f"| {lang} | " + " | ".join(fmt.format(results[n]["languages"][lang][metric])
                                                    for n in names) + " |")
        lines.append("")
    vocab_line = ", ".join(f"{n}: {results[n]['vocab_size']:,}" for n in names)
    lines += ["### Vocabulary sizes", "", vocab_line, ""]
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    (out_dir / "report.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--vocab-size", type=int, default=16000)
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument("--pool-dir", type=Path, default=Path("data/pilot_corpus"))
    parser.add_argument("--split-manifest", type=Path, default=None, help="Default: <pool-dir>/split_manifest.csv")
    parser.add_argument("--languages", nargs="+", default=LANGUAGES)
    parser.add_argument("--chars-per-language", type=int, default=0,
                        help="Training characters per language (0 = as many as the smallest language has)")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Default: checkpoints/tokenizers/sea_bpe_<vocab/1000>k")
    parser.add_argument("--compare", nargs="*", default=["SeaLLMs/SeaLLM-7B-v2"],
                        help="Tokenizers (Hub IDs or folders) to compare against on the validation split")
    parser.add_argument("--limit-docs", type=int, default=0, help="Debug: keep only N train/validation docs per language")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    out_dir = args.output_dir or Path("checkpoints/tokenizers") / f"sea_bpe_{args.vocab_size // 1000}k"
    manifest = args.split_manifest or args.pool_dir / "split_manifest.csv"
    train_ids, val_ids = load_split_ids(manifest, "train"), load_split_ids(manifest, "validation")
    if args.limit_docs:
        train_ids = {l: set(sorted(v)[:args.limit_docs]) for l, v in train_ids.items()}
        val_ids = {l: set(sorted(v)[:args.limit_docs]) for l, v in val_ids.items()}

    print(f"Loading train documents for {', '.join(args.languages)} ...", flush=True)
    train_texts = {lang: load_pool_texts(args.pool_dir, lang, train_ids[lang]) for lang in args.languages}
    chosen, train_stats = balanced_sample(train_texts, args.chars_per_language, args.seed)
    del train_texts
    for lang, s in train_stats.items():
        print(f"  {lang:6s} {s['documents']:>7,} docs  {s['characters']:>13,} chars  "
              f"(of {s['available_documents']:,} docs / {s['available_characters']:,} chars)", flush=True)

    print(f"Training byte-level BPE, vocab {args.vocab_size:,} ...", flush=True)
    n = sum(len(v) for v in chosen.values())
    tokenizer = train_tokenizer(interleave(chosen), args.vocab_size, args.min_frequency, length=n)
    del chosen
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(out_dir)
    (out_dir / "training_data.json").write_text(json.dumps(
        {"vocab_size": len(tokenizer), "min_frequency": args.min_frequency, "seed": args.seed,
         "pool_dir": str(args.pool_dir), "split_manifest": str(manifest), "split": "train",
         "pretokenize_pattern": PRETOKENIZE_PATTERN, "languages": train_stats}, indent=2), encoding="utf-8")
    print(f"Saved tokenizer ({len(tokenizer):,} tokens) to {out_dir}", flush=True)

    print("Measuring on the validation split ...", flush=True)
    val_texts = {lang: list(load_pool_texts(args.pool_dir, lang, val_ids[lang]).values()) for lang in args.languages}
    from transformers import AutoTokenizer
    candidates = {out_dir.name: AutoTokenizer.from_pretrained(out_dir)}
    for name in args.compare:
        try:
            candidates[name] = AutoTokenizer.from_pretrained(name)
        except Exception as exc:  # comparison is optional (e.g. offline node)
            print(f"  Note: could not load {name} for comparison ({exc.__class__.__name__}), skipping", flush=True)
    results = {}
    for name, tok in candidates.items():
        results[name] = {"vocab_size": len(tok),
                         "languages": {lang: measure(tok, texts) for lang, texts in val_texts.items()}}
    write_report(out_dir, out_dir.name, len(tokenizer), train_stats, results)
    for lang in args.languages:
        print(f"  {lang:6s} " + "  ".join(f"{name}: {results[name]['languages'][lang]['chars_per_token']:.2f} chars/tok "
                                          f"(round trip {results[name]['languages'][lang]['round_trip_exact']:.0%})"
                                          for name in results), flush=True)
    print(f"Wrote {out_dir / 'report.md'}")


if __name__ == "__main__":
    main()
