# SEA-Rater: Results Summary (OLMo-1B and Gemma-3-1B, 50M tokens per language)

## 1. Setup

Every run is LoRA continued pretraining (r=16, alpha=32, 1 epoch, seed 42) of a pretrained model on 50M tokens per language in 7 languages (burmese, fil, indo, khmer, malay, thai, vie). Only the training data, or the loss weighting, differs between runs:

| Run | Training data |
|---|---|
| base | no training (the pretrained model) |
| random | random documents (the baseline) |
| wavg5 | the random documents, loss weighted by the avg5 rater score (OLMo only, `docs/05_loss.md`) |
| edu | top documents by the educational-value rater |
| cult | top documents by the cultural-nuances rater |
| avg5 | top documents by the mean of the 5 rater scores |
| rr5 | round-robin over the 5 rater rankings |

OLMo selections count tokens with the OLMo tokenizer; Gemma selections count them with the Gemma tokenizer. Per-run outputs are in `checkpoints/lora_cpt/<model>/<run>_50M_7languages_ep1_seed42/` (Hub: `cppmai/sea7-checkpoints`), with full per-language tables in the `*_comparison.md` files of each model folder.

Evaluation (`sh/eval/eval_ppl_clean.sh`, `sh/eval/eval_lm_harness_per_benchmark.sh`, `docs/04_wikipedia_sib200_evaluation.md`, `docs/03_downstream_evaluation.md`):

- **Perplexity** on held-out text, reported as **bits per byte** (summed loss / ln 2 / UTF-8 bytes): unlike loss and perplexity, which are per token, it does not depend on the tokenizer, so OLMo and Gemma are directly comparable. Sets: Wikipedia (1,000 articles per language), Bloom library storybooks, the Bible (one document per chapter), NTREX-128 news (all 123 articles), ALT news (the same 1,000 articles in every language), FLORES+ Wikibooks / Wikivoyage passages. Bloom has no Malay and the Bible set has no Khmer.
- **Benchmarks**: lm-evaluation-harness, 5-shot (5 held-out test items as demonstrations for test-only tasks). OLMo scores the full Global-MMLU test set, Gemma the fixed subset (100 questions per subject category, 600 per language). SEA-NLI normal and hard are pooled into one score (every question counts once).

In the tables, the number in brackets is the difference from the same model's `base`; **bold** is the best trained run of each model.

## Table 1. Perplexity: bits per byte (lower is better)

Macro-average over the languages of each set.

| Model | Run | Wikipedia | Storybooks | Bible | NTREX (news) | ALT (news) | FLORES Wikibooks | FLORES Wikivoyage |
|---|---|---|---|---|---|---|---|---|
| **Gemma-3-1B** | base | 0.7359 | 0.7396 | 0.7873 | 0.7970 | 0.7562 | 0.7729 | 0.7948 |
| | random | 0.7423 (+0.0064) | 0.7527 (+0.0131) | 0.8021 (+0.0148) | **0.7710 (−0.0260)** | **0.7354 (−0.0207)** | 0.7646 (−0.0082) | 0.7794 (−0.0154) |
| | edu | **0.7397 (+0.0037)** | 0.7535 (+0.0139) | **0.8000 (+0.0128)** | 0.7737 (−0.0233) | 0.7361 (−0.0201) | **0.7586 (−0.0143)** | **0.7743 (−0.0205)** |
| | cult | 0.7462 (+0.0103) | **0.7505 (+0.0109)** | 0.8060 (+0.0187) | 0.7814 (−0.0156) | 0.7480 (−0.0081) | 0.7765 (+0.0037) | 0.7928 (−0.0021) |
| | avg5 | 0.7426 (+0.0067) | 0.7575 (+0.0179) | 0.8007 (+0.0134) | 0.7792 (−0.0178) | 0.7380 (−0.0182) | 0.7616 (−0.0112) | 0.7796 (−0.0152) |
| | rr5 | 0.7411 (+0.0052) | 0.7564 (+0.0168) | 0.8066 (+0.0193) | 0.7730 (−0.0240) | 0.7359 (−0.0202) | 0.7624 (−0.0104) | 0.7795 (−0.0153) |
| **OLMo-1B** | base | 1.1478 | 1.0846 | 1.1609 | 1.1583 | 1.1618 | 1.2115 | 1.2264 |
| | random | 1.0148 (−0.1330) | 0.9836 (−0.1009) | 1.0717 (−0.0892) | 0.9757 (−0.1826) | 0.9456 (−0.2162) | 1.0003 (−0.2112) | 1.0063 (−0.2201) |
| | wavg5 | 1.0140 (−0.1338) | 0.9829 (−0.1017) | 1.0705 (−0.0904) | **0.9757 (−0.1826)** | **0.9453 (−0.2164)** | 0.9990 (−0.2125) | 1.0053 (−0.2211) |
| | edu | 1.0190 (−0.1288) | **0.9686 (−0.1159)** | 1.0578 (−0.1031) | 1.0254 (−0.1330) | 0.9837 (−0.1781) | 0.9761 (−0.2354) | **0.9925 (−0.2339)** |
| | cult | 1.0403 (−0.1075) | 0.9841 (−0.1005) | 1.0753 (−0.0856) | 1.0352 (−0.1231) | 1.0136 (−0.1482) | 1.0573 (−0.1542) | 1.0635 (−0.1629) |
| | avg5 | 1.0130 (−0.1348) | 0.9817 (−0.1029) | 1.0618 (−0.0992) | 1.0219 (−0.1365) | 0.9705 (−0.1912) | 0.9919 (−0.2195) | 1.0119 (−0.2145) |
| | rr5 | **1.0087 (−0.1391)** | 0.9706 (−0.1140) | **1.0538 (−0.1071)** | 1.0005 (−0.1578) | 0.9581 (−0.2036) | **0.9758 (−0.2356)** | 0.9953 (−0.2310) |

OLMo random and wavg5 tie on NTREX at four decimals (wavg5 is lower in later digits).

FineWeb2 test split (1,000 random web documents per language, same distribution as the training pool; OLMo only): base 1.1653, random 0.9439 (−0.2214), wavg5 0.9446 (−0.2207).

## Table 2. 5-shot benchmarks (accuracy %, higher is better)

Macro-average over languages per benchmark; *overall* = mean of Belebele, Global PIQA, INCLUDE and Global-MMLU.

| Model | Run | Belebele | Global PIQA | INCLUDE | Global-MMLU | SIB-200 | SEA-NLI (pooled) | Overall |
|---|---|---|---|---|---|---|---|---|
| *Chance / majority answer* | | *25 / 25* | *50 / 50* | *25 / 25* | *25 / 25* | *14.3 / 25.0* | *33.3 / 39.9* | |
| **Gemma-3-1B** | base | 27.34 | 59.16 | 23.92 | 24.12 | 69.89 | 39.62 | 33.64 |
| | random | 25.86 (−1.49) | 60.00 (+0.84) | 25.56 (+1.64) | 23.67 (−0.46) | 68.07 (−1.82) | **39.98 (+0.36)** | 33.77 (+0.13) |
| | edu | 25.37 (−1.97) | **60.84 (+1.68)** | 25.45 (+1.53) | 23.79 (−0.33) | 69.96 (+0.07) | 39.86 (+0.24) | 33.86 (+0.23) |
| | cult | **26.34 (−1.00)** | 60.00 (+0.84) | 25.42 (+1.50) | 22.83 (−1.29) | 69.82 (−0.07) | 39.92 (+0.30) | 33.65 (+0.01) |
| | avg5 | 25.43 (−1.91) | 60.63 (+1.47) | **26.01 (+2.10)** | 23.83 (−0.29) | 69.96 (+0.07) | 39.68 (+0.06) | **33.98 (+0.34)** |
| | rr5 | 25.77 (−1.57) | 60.00 (+0.84) | 25.06 (+1.14) | **23.96 (−0.17)** | **70.24 (+0.35)** | 39.74 (+0.12) | 33.70 (+0.06) |
| **OLMo-1B** | base | 26.51 | 47.79 | 25.24 | 25.36 | 26.61 | 39.08 | 31.23 |
| | random | 25.83 (−0.69) | 48.63 (+0.84) | **26.38 (+1.14)** | 25.40 (+0.04) | 28.01 (+1.40) | 38.78 (−0.30) | 31.56 (+0.34) |
| | wavg5 | 26.17 (−0.34) | 48.84 (+1.05) | 25.78 (+0.54) | 25.30 (−0.06) | 27.59 (+0.98) | 38.42 (−0.66) | 31.52 (+0.30) |
| | edu | 26.17 (−0.34) | 49.47 (+1.68) | 24.66 (−0.58) | **26.24 (+0.88)** | **28.71 (+2.10)** | 35.71 (−3.36) | 31.64 (+0.41) |
| | cult | 25.91 (−0.60) | 49.89 (+2.11) | 25.04 (−0.20) | 25.13 (−0.23) | 24.51 (−2.10) | **39.08 (0.00)** | 31.49 (+0.27) |
| | avg5 | 25.97 (−0.54) | 50.11 (+2.32) | 24.62 (−0.62) | 25.59 (+0.24) | 25.00 (−1.61) | 37.45 (−1.62) | 31.57 (+0.35) |
| | rr5 | **26.49 (−0.03)** | **52.21 (+4.42)** | 24.90 (−0.34) | 25.63 (+0.27) | 26.47 (−0.14) | 37.76 (−1.32) | **32.31 (+1.08)** |

Chance = 1 / number of options; majority answer = accuracy of always giving the most frequent label of the scored items (SIB-200: science/technology, 51 of 204 sentences per language). SEA-NLI pools normal (1,105 items) and hard (561 items) over all languages: 1,666 questions.

## Table 3. Culture-related subsets (5-shot accuracy %, pooled over languages)

Subsets from the datasets' own annotations (`eval_lm_harness breakdown`). Questions per model: INCLUDE culture 508, region implicit 1,048; Global PIQA culturally specific 412; Global-MMLU culturally sensitive (CS) 136 for Gemma (subset) and 3,168 for OLMo (full set).

| Model | Run | INCLUDE culture | INCLUDE region implicit | Global PIQA cultural | Global-MMLU CS |
|---|---|---|---|---|---|
| **Gemma-3-1B** | base | 22.64 | 23.57 | 58.74 | 13.24 |
| | random | **28.15 (+5.51)** | 24.43 (+0.86) | 59.71 (+0.97) | 22.79 (+9.56) |
| | edu | 25.59 (+2.95) | **25.48 (+1.91)** | **60.92 (+2.18)** | 25.00 (+11.76) |
| | cult | 27.17 (+4.53) | 24.43 (+0.86) | 59.71 (+0.97) | 18.38 (+5.15) |
| | avg5 | 27.17 (+4.53) | 23.95 (+0.38) | 60.44 (+1.70) | **27.21 (+13.97)** |
| | rr5 | 25.59 (+2.95) | 24.14 (+0.57) | 59.95 (+1.21) | 21.32 (+8.09) |
| **OLMo-1B** | base | 23.82 | 24.81 | 46.84 | 25.44 |
| | random | **29.13 (+5.31)** | 24.62 (−0.19) | 48.54 (+1.70) | 25.28 (−0.16) |
| | wavg5 | **29.13 (+5.31)** | 23.57 (−1.24) | 48.79 (+1.94) | 24.49 (−0.95) |
| | edu | 24.41 (+0.59) | 24.05 (−0.76) | 49.51 (+2.67) | 25.09 (−0.35) |
| | cult | 24.02 (+0.20) | **25.57 (+0.76)** | 50.00 (+3.16) | **27.90 (+2.46)** |
| | avg5 | 22.24 (−1.57) | 24.62 (−0.19) | 49.76 (+2.91) | 27.30 (+1.86) |
| | rr5 | 23.23 (−0.59) | 25.48 (+0.67) | **52.43 (+5.58)** | 26.33 (+0.88) |

Gemma's CS subset has only 136 questions and its base scores far below chance, so its CS deltas are mostly noise; OLMo's CS scores (3,168 questions) stay within ~2.5 points of chance.

## Table 4. Per-language effect of selection (vs random)

Does selection help each language? **Perplexity**: mean relative change in bits per byte against `random`, averaged over the held-out sets that cover the language (negative = better); in brackets, on how many sets the method beats `random` (Khmer has no Bible, Malay no Bloom: 6 sets). **Benchmarks**: accuracy change against `random` in points, for the two benchmarks above chance (Global PIQA: 95 questions per language, about ±5 points per language; SIB-200: 204 sentences, about ±3 points). Computed from the per-language `bits_per_byte` in each run's `ppl_*/results.json` and the per-language benchmark `summary.csv` files.

**OLMo-1B, perplexity**

| Method | burmese | fil | indo | khmer | malay | thai | vie |
|---|---|---|---|---|---|---|---|
| wavg5 | −0.06% (5/7) | −0.17% (7/7) | −0.05% (5/7) | +0.01% (2/6) | −0.06% (4/6) | −0.15% (7/7) | −0.03% (5/7) |
| rr5 | **−1.11% (5/7)** | −1.13% (5/7) | −0.15% (5/7) | +0.39% (3/6) | −0.09% (3/6) | −0.42% (5/7) | **−0.55% (5/7)** |
| edu | +1.12% (3/7) | **−1.72% (5/7)** | +0.32% (4/7) | +5.23% (0/6) | −0.25% (3/6) | −0.33% (5/7) | +1.20% (3/7) |
| avg5 | +0.59% (5/7) | −0.07% (3/7) | +0.85% (2/7) | +2.78% (1/6) | +0.57% (2/6) | +0.44% (4/7) | +1.21% (2/7) |
| cult | +1.34% (3/7) | +5.04% (1/7) | +3.87% (0/7) | +5.22% (0/6) | +5.99% (0/6) | +3.14% (2/7) | +3.60% (1/7) |

**Gemma-3-1B, perplexity**

| Method | burmese | fil | indo | khmer | malay | thai | vie |
|---|---|---|---|---|---|---|---|
| edu | **−0.64% (4/7)** | **−0.80% (6/7)** | −0.12% (4/7) | +0.73% (2/6) | −0.33% (4/6) | +0.03% (6/7) | −0.07% (5/7) |
| rr5 | +0.32% (4/7) | −0.53% (5/7) | −0.10% (6/7) | +1.71% (0/6) | −0.19% (5/6) | +0.44% (5/7) | −0.11% (5/7) |
| avg5 | −0.22% (4/7) | −0.17% (4/7) | +0.15% (2/7) | +1.10% (1/6) | +0.04% (2/6) | +0.90% (4/7) | +0.03% (3/7) |
| cult | +0.94% (2/7) | +1.23% (1/7) | +0.92% (0/7) | +2.21% (0/6) | +1.39% (0/6) | +0.33% (3/7) | +0.53% (1/7) |

**Benchmarks** (accuracy change vs random, points; – = benchmark has no data for the language)

| Model / benchmark | Method | burmese | fil | indo | khmer | malay | thai | vie |
|---|---|---|---|---|---|---|---|---|
| OLMo, Global PIQA | rr5 | – | +7.4 | 0.0 | – | +2.1 | +5.3 | +3.2 |
| | edu | – | +2.1 | +2.1 | – | +4.2 | −3.2 | −1.1 |
| | avg5 | – | 0.0 | +1.1 | – | +1.1 | +4.2 | +1.1 |
| OLMo, SIB-200 | edu | +2.9 | +2.0 | +2.0 | +1.0 | +0.5 | +1.5 | −4.9 |
| | rr5 | +4.4 | −3.4 | −4.4 | +3.4 | −5.9 | 0.0 | −4.9 |
| | avg5 | +0.5 | −3.9 | −7.4 | +0.5 | −7.4 | +1.0 | −4.4 |
| Gemma, SIB-200 | edu | −0.5 | 0.0 | +2.0 | +2.9 | +6.9 | +2.5 | −0.5 |
| | avg5 | −1.5 | +0.5 | +3.4 | +0.5 | +5.4 | +2.0 | +2.9 |
| | rr5 | 0.0 | +0.5 | +2.0 | +2.0 | +5.4 | +3.9 | +1.5 |

**By language**

| Language | Does selection help? | Best method |
|---|---|---|
| Filipino | Yes, most consistently: better under almost every method for both models; OLMo rr5 +7.4 on Global PIQA | edu (OLMo −1.7%, Gemma −0.8%), rr5 |
| Burmese | Yes with rr5 on OLMo (−1.1%) and edu on Gemma (−0.6%); edu and avg5 hurt OLMo | rr5 (OLMo), edu (Gemma) |
| Vietnamese | Slightly with rr5 (OLMo −0.55%); edu and avg5 hurt OLMo (+1.2%) | rr5 |
| Thai | Slightly on OLMo (rr5 −0.4%, edu −0.3%; rr5 +5.3 on Global PIQA); neutral or slightly worse on Gemma | rr5 (OLMo) |
| Indonesian | Neutral on perplexity (best methods within ±0.3%) | – |
| Malay | Neutral on perplexity, but the largest and most consistent Gemma SIB-200 gains (+5.4 to +6.9 for every selection) | edu (Gemma) |
| Khmer | **No: selection hurts for both models.** Every selection is worse than random on almost every set (OLMo edu +5.2%, 0/6 sets; Gemma rr5 +1.7%, 0/6) | random |

- Selection helps some languages, not all: the best methods (rr5 for OLMo, edu for Gemma) lower perplexity in 4-5 of 7 languages, mostly Filipino, Burmese and Vietnamese, never in Khmer.
- **Khmer is the clear failure case** for both models and all four selections. A plausible cause is the Khmer rater scores (less reliable, or the top-scored Khmer documents cover a narrow set of topics); to check: the raters' per-language test errors and the content of the top-scored Khmer documents.
- rr5 is the most language-robust selection on OLMo (better in 5 languages, at most +0.4% worse elsewhere); edu varies most (best for Filipino, worst for Khmer). cult hurts every language for both models; wavg5 stays within ±0.2% everywhere.
- Per-language benchmark differences are mostly within noise; the one consistent pattern is Gemma SIB-200 in Malay (+5 to +7 points for every selection).

## 2. Observations

- **Gemma is the much stronger base model** in these languages: 0.74-0.81 bits per byte against 0.95-1.07 for trained OLMo, and well above chance on SIB-200 (~70%) and Global PIQA (~60%), where OLMo is close to chance.
- **Training helps OLMo on every held-out set** (−0.09 to −0.24 bits per byte), most on news and FLORES. **Gemma gets worse than its base** on Wikipedia, the storybooks and the Bible with every training set, and improves only on news and FLORES.
- **Rater selection is a domain trade-off, not a uniform gain.** edu and rr5 are best on book-like and encyclopedic text (Wikipedia, storybooks, the Bible, both FLORES sets), while random (and wavg5) are best on news (NTREX, ALT); avg5, edu and cult lose most on news. Both models show this pattern.
- **rr5 is the best OLMo method overall**: best benchmark average (32.31, +1.08 over base, driven by Global PIQA +4.4), best on 3 of 7 perplexity sets, and the smallest loss on news among the selections.
- **cult is the weakest selection on perplexity for both models** (OLMo: worse than random on all 7 sets; Gemma: on 6 of 7) and no better than random on the overall benchmark average. On the culture-related subsets it is mixed: OLMo cult is best on INCLUDE region-implicit and Global-MMLU culturally sensitive questions (+0.8 / +2.5 over base), but plain random training gains most on INCLUDE culture questions (+5.3 OLMo, +5.5 Gemma).
- **wavg5 is indistinguishable from random**: within ~0.001 bits per byte on every set and within ~0.7 points on every benchmark. It is consistently a little better on clean or edited text and a little worse on random web text, but the effect is ~100 times smaller than training itself.
- **The effect of selection differs by language** (Table 4): the best selections help Filipino, Burmese and Vietnamese most, are neutral for Indonesian and Malay, and hurt Khmer for every method and both models.
- **Benchmarks are mostly at chance.** Belebele, INCLUDE, Global-MMLU and SEA-NLI sit at chance or majority-answer level for every model (no model beats the 39.9% SEA-NLI majority answer), so their differences are noise. Global PIQA (both models) and SIB-200 (Gemma) are the only benchmarks that separate the runs; on culturally specific Global PIQA questions the rater selections lead (OLMo rr5 +5.6 over base against +1.7 for random).
