# SEA-Rater: Held-Out Validation and Test Sets

## 1. Purpose

Every data-selection or loss-weighting method in this project is compared on the **same fixed held-out documents**. The documents are reserved from the FineWeb2 pilot corpus **before** any quality scoring or selection, so no method can train on them and no method gets to choose its own evaluation set.

There are two held-out splits:

| Split | Use |
|---|---|
| `validation` | Choose settings: checkpoints, learning rates, score combinations, weighting strength. |
| `test` | Final reporting only, after all choices are fixed. Do not tune on it. |

Everything else in the corpus is `train`, the **candidate pool** that every selection method draws from.

## 2. Where it lives

| File | Contents |
|---|---|
| `data/pilot_corpus_7languages/split_manifest.csv` | One row per document: `doc_id, language, split, domain, char_len, source`. `split` is `train`, `validation`, `test` or `excluded`. |
| `data/pilot_corpus_7languages/split_report.md` | Sizes, site counts, length and language-score comparison, leakage checks. |
| `data/pilot_corpus_7languages/<language>.csv` | Document text. The manifest refers to it by `(language, doc_id)`. |

`data/` is git-ignored. The older single-corpus setup used the same layout under `data/pilot_corpus/`.

## 3. How the split is built

Script: `src/train_gpt2_from_scratch/split_corpus.py`, run after `extract_corpus`.

For each language, with seed `42`:

1. Group documents by **website** (host name without `www.`).
2. Draw whole websites at random and fill `test`, then `validation`, until each has exactly the requested number of documents. Only websites with **at most 10 documents** are eligible, so a single site cannot dominate a small evaluation set.
3. Everything else is `train`. A website is in **exactly one split**.

Why split by website instead of by document: FineWeb2 pages are concentrated on a few sites (for example, one Khmer news site is about 10% of the sample). A random document-level split would put near-identical templated pages from the same site in both train and test, which inflates test scores.

```bash
# Defaults: 500 validation + 1,000 test per language
python -m src.train_gpt2_from_scratch.split_corpus --corpus-dir data/pilot_corpus_7languages

# Keep the held-out documents of an older manifest (for a larger re-extraction of the same stream)
python -m src.train_gpt2_from_scratch.split_corpus --corpus-dir data/pilot_corpus_7languages \
    --keep-eval-from data/pilot_corpus/split_manifest.csv
```

With `--keep-eval-from`, the old validation/test documents keep their split. Any new document from a website already in validation/test is marked `excluded` and is never read by any consumer, so no site crosses splits.

## 4. Current sizes (7-language corpus)

300,000 documents per language: **1,000 test**, **500 validation**, 298,500 train. No website appears in more than one split.

| Language | Test sites | Test median chars | Train median chars | Test tokens (OLMo) |
|---|---:|---:|---:|---:|
| Burmese | 391 | 1,559 | 1,703 | 2.68M |
| Filipino | 518 | 1,923 | 1,463 | 0.70M |
| Indonesian | 580 | 2,278 | 2,021 | 0.82M |
| Khmer | 421 | 1,679 | 1,410 | 3.31M |
| Malay | 485 | 1,583 | 1,669 | 0.66M |
| Thai | 537 | 1,880 | 1,652 | 2.05M |
| Vietnamese | 538 | 2,271 | 2,163 | 1.43M |

Token counts depend on the tokenizer. Use these OLMo counts as a rough size guide only.

## 5. Who reads which split

| Step | Reads | Code |
|---|---|---|
| Rater scoring of the candidate pool | `train` only | `score_pool.py` (default `--splits train`) |
| Data selection (random / top-score) | `train` only | `prepare_data.py` |
| GPT-2 training, LoRA continued pretraining | train on the selection; evaluate on `validation` during training and on `validation` + `test` at the end | `train_gpt2.py`, `train_gpt2_weighted.py`, `continue_pretrain_lora.py` (`load_heldout_documents`) |
| Re-scoring a finished run on test | `test` | `eval_test_set.py`, `sh/eval/eval_test_set.sh` |
| Quality-score distribution of the test set (diagnostic only) | `test` | `sh/rater/score_test_set.sh` → `data/pilot_scores/*_mean_test` |
| Wikipedia evaluation | excludes any article whose text matches a pilot-corpus document | `eval_wiki_sib200.py --pilot-corpus-dir` |

Rater-supervision documents are removed from the whole corpus by `extract_corpus`, so they are in none of the splits.

## 6. Evaluation protocol on the test set

- **Ordinary unweighted next-token loss** for every model, including models trained with a weighted loss. Weighted training loss is not comparable to ordinary loss (see `docs/05_loss.md`).
- Documents are encoded as the model was trained (BOS + text + EOS when the model has a BOS token, otherwise text + EOS), packed into context-length blocks with the run's seed, and every target token is scored once.
- Report per language and the macro average (mean over languages): **loss**, **perplexity** and **bits per byte**.
- Loss and perplexity are per token, so compare them **only between models with the same tokenizer**. Bits per byte (summed loss in nats / ln 2 / UTF-8 bytes) does not depend on the tokenizer, so use it to compare across tokenizers.
- The first run listed is the baseline. The comparison file reports each run's difference from it.

```bash
# Weighted-loss GPT-2 runs vs. the unweighted random_20M baseline
sbatch sh/eval/eval_test_set.sh

# Any set of runs
RUN_DIRS="<baseline run> <run> ..." COMPARISON_FILE=<out>.md sbatch sh/eval/eval_test_set.sh
```

Outputs: `<run>/test_eval/{results.json,summary.md}` per run, plus the comparison file. The LoRA runs also store their final test result in `<run>/results.json` under `"test"`.

## 7. Example: OLMo-1B LoRA, 50M tokens per language

Test loss (macro over 7 languages), same documents in the same order, only the loss weighting differs:

| Run | Test loss | Test perplexity |
|---|---:|---:|
| `random_50M_7languages` (unweighted) | 1.5993 | 4.949 |
| `wavg5_50M_7languages` (quality-weighted) | 1.6000 | 4.953 |

On this web-text test set, quality-weighted training does not change held-out loss. Per-language differences are at most 0.007 (Khmer).

## 8. Rules

1. **Do not change the split** for an existing comparison. All compared runs must use the same `split_manifest.csv`.
2. **Do not tune on test.** Choose checkpoints, hyperparameters and score combinations on validation. A change prompted by looking at test results is exploratory and needs fresh held-out evidence.
3. **Do not select a different test set per method.** In particular, do not pick a "high-quality" test subset with the same rater used for selection, because that favors the rater-based method. Report quality-stratified subsets only as extra diagnostics, next to the full test set.
4. Report the run trained on the full matched token budget as the main comparison.

## 9. Limitations

- **Small sites only.** Validation/test come from websites with at most 10 documents, while train also contains the large sites. The test set is therefore not identical in site mix to train. This is fine for comparing methods, but it is not an unbiased sample of the whole corpus.
- **Web text.** The test set has the same noise as FineWeb2 (boilerplate, ads, mixed quality). A method that prefers cleaner text may not show a gain here. Complement it with clean, human-written evaluation sets: Wikipedia / SIB-200 (`docs/04_wikipedia_sib200_evaluation.md`), and Bible / NTREX / ALT / FLORES+ (`sh/eval/eval_ppl_clean.sh`).
- **Near-duplicates.** Exact duplicates were removed at extraction, and website grouping blocks same-site leakage. Near-duplicate detection across different sites (for example, syndicated news) has not been run.
- **Sample size.** 1,000 documents per language can miss very small differences. Use paired comparisons on the same documents when testing whether a difference is real.
