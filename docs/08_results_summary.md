# SEA-Rater: Results Summary (OLMo-1B and Gemma-3-1B, 50M tokens per language)

## 1. Setup

Every run is LoRA continued pretraining (r=16, alpha=32, 1 epoch, seed 42) of a pretrained model on 50M tokens per language in 7 languages (burmese, fil, indo, khmer, malay, thai, vie). Only the training data, or the loss weighting, differs between runs:

| Run | Training data |
|---|---|
| base | no training (the pretrained model) |
| random | random documents (the baseline) |
| wavg5 | the random documents, loss weighted by the avg5 rater score (OLMo only, `docs/05_loss.md`) |
| cult | top documents by the cultural-nuances rater (Gemma only) |
| avg5 | top documents by the mean of the 5 rater scores |
| edu | top documents by the educational-value rater |
| rr5 | round-robin over the 5 rater rankings |

OLMo selections count tokens with the OLMo tokenizer; Gemma selections count them with the Gemma tokenizer. Per-run outputs are in `checkpoints/lora_cpt/<model>/<run>_50M_7languages_ep1_seed42/` (Hub: `cppmai/sea7-checkpoints`), with full per-language tables in the `*_comparison.md` files of each model folder.

Evaluation sets (`sh/eval/eval_ppl_clean.sh`, `docs/04_wikipedia_sib200_evaluation.md`, `docs/03_downstream_evaluation.md`):

- **Perplexity**: Wikipedia (1,000 articles per language), Bloom library books, the Bible (one document per chapter), NTREX-128 news, FLORES+ Wikibooks / Wikivoyage passages, and ALT news (the same 1,000 articles in every language). Bloom has no Malay and the Bible set has no Khmer.
- **Benchmarks**: lm-evaluation-harness, 5-shot.

## Table 1. Perplexity (lower is better)

exp(macro-average loss over the languages of each set). Compare models only within a family (OLMo and Gemma have different tokenizers). Best per row in bold.

| | base | random | wavg5 | cult | avg5 | edu | rr5 |
|---|---|---|---|---|---|---|---|
| **OLMo-1B** | | | | | | | |
| wikipedia (7 lang) | 6.58 | 5.44 | 5.43 | – | 5.43 | 5.45 | **5.39** |
| bloom (6 lang) | 5.29 | 4.54 | 4.54 | – | 4.52 | **4.44** | 4.45 |
| bible (6 lang) | 6.78 | 5.84 | 5.82 | – | 5.74 | 5.70 | **5.67** |
| ntrex (7 lang) | 6.67 | **5.03** | **5.03** | – | 5.38 | 5.42 | 5.22 |
| flores_wikibooks (7 lang) | 7.20 | 5.26 | 5.25 | – | 5.19 | **5.04** | 5.06 |
| flores_wikivoyage (7 lang) | 7.45 | 5.36 | 5.35 | – | 5.41 | **5.22** | 5.26 |
| alt (7 lang) | 6.77 | 4.92 | **4.91** | – | 5.12 | 5.20 | 5.02 |
| **Gemma-3-1B** | | | | | | | |
| wikipedia (7 lang) | **9.62** | 9.82 | – | 9.91 | 9.80 | 9.72 | 9.77 |
| bloom (6 lang) | **8.56** | 8.96 | – | 8.90 | 9.13 | 9.00 | 9.09 |
| bible (6 lang) | **10.56** | 11.17 | – | 11.28 | 11.13 | 11.13 | 11.38 |
| ntrex (7 lang) | 12.89 | **11.86** | – | 12.29 | 12.15 | 11.96 | 11.93 |
| flores_wikibooks (7 lang) | 14.94 | 14.62 | – | 15.25 | 14.45 | **14.31** | 14.52 |
| flores_wikivoyage (7 lang) | 16.40 | 15.65 | – | 16.39 | 15.64 | **15.37** | 15.67 |
| alt (7 lang) | 13.11 | **12.19** | – | 12.73 | 12.30 | 12.23 | 12.22 |

## Table 2. 5-shot benchmarks (accuracy %, higher is better)

Macro-average over languages per benchmark; *core avg* = mean of belebele, global_piqa, global_mmlu, include. OLMo scores the full Global MMLU test set, Gemma the fixed subset. Best per row in bold.

| | base | random | wavg5 | cult | avg5 | edu | rr5 |
|---|---|---|---|---|---|---|---|
| **OLMo-1B** | | | | | | | |
| belebele | **26.51** | 25.83 | 26.17 | – | 25.97 | 26.17 | 26.49 |
| global_piqa | 47.79 | 48.63 | 48.84 | – | 50.11 | 49.47 | **52.21** |
| global_mmlu | 25.36 | 25.40 | 25.30 | – | 25.59 | **26.24** | 25.63 |
| include | 25.24 | **26.38** | 25.78 | – | 24.62 | 24.66 | 24.90 |
| sib200 | 26.61 | 28.01 | 27.59 | – | 25.00 | **28.71** | 26.47 |
| sea_nli_normal | 33.23 | **33.35** | 32.75 | – | 33.23 | 31.20 | 33.18 |
| sea_nli_hard | 48.03 | **49.33** | 49.26 | – | 46.02 | 43.84 | 46.85 |
| core avg | 31.23 | 31.56 | 31.52 | – | 31.57 | 31.64 | **32.31** |
| **Gemma-3-1B** | | | | | | | |
| belebele | **27.34** | 25.86 | – | 26.34 | 25.43 | 25.37 | 25.77 |
| global_piqa | 59.16 | 60.00 | – | 60.00 | 60.63 | **60.84** | 60.00 |
| global_mmlu | **24.12** | 23.67 | – | 22.83 | 23.83 | 23.79 | 23.96 |
| include | 23.92 | 25.56 | – | 25.42 | **26.01** | 25.45 | 25.06 |
| sib200 | 69.89 | 68.07 | – | 69.82 | 69.96 | 69.96 | **70.24** |
| sea_nli_normal | 32.68 | 32.79 | – | 32.74 | 32.68 | **32.94** | 32.74 |
| sea_nli_hard | 51.05 | 51.56 | – | **51.66** | 51.18 | 51.05 | 51.18 |
| core avg | 33.64 | 33.77 | – | 33.65 | **33.98** | 33.86 | 33.70 |

## 2. Observations

- **Training helps perplexity a lot for OLMo, little for Gemma.** Every OLMo run beats base by 15-30% on every set. Gemma base is already strong in these languages: it has the lowest perplexity on Wikipedia, Bloom and the Bible, and training only helps on the news (NTREX, ALT) and FLORES sets.
- **Rater selections help on book-like and encyclopedic text, not on news.** For OLMo, rr5 and edu are best on Wikipedia, Bloom, the Bible and both FLORES sets (2-6% below random), but on NTREX and ALT news random is best and avg5 / edu are 4-8% worse. Gemma shows the same pattern, with smaller gaps.
- **cult (Gemma) is the weakest selection on perplexity**, worse than random on 6 of 7 sets.
- **wavg5 is indistinguishable from random** (within 0.01 perplexity on every set, within 0.6 points on every benchmark): the avg5 loss weighting barely changes training.
- **Benchmarks are close to chance and differ by ~1 point**, so most differences are within noise. The exception is OLMo rr5 on Global PIQA (52.2 vs 48.6 for random), which lifts it to the best OLMo core average (32.3 vs 31.6). For Gemma, avg5 has the best core average (34.0 vs 33.8 for random).
