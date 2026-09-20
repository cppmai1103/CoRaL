# SEA-Rater

Multidimensional data quality assessment and selection for Southeast Asian language-model training.

**Current status:** Human annotation is complete. The next milestone is to prepare the annotation dataset for training and establish a multilingual quality-rater baseline.

This README describes the project idea and overall workflow. The first implementation guide is [01_train_rater.md](01_train_rater.md), covering annotation preparation and multilingual rater training. These documents specify the workflow; they do not imply that training has already been run.

## Running the code

Python code lives in `src/`. Run commands from the project root so the default
dataset, embedding, and checkpoint paths resolve correctly:

```bash
python -m src.download_from_huggingface --local-dir final_dataset
python -m src.prepare_dataset --input-dir final_dataset --output-dir prepared_data
python -m src.build_embeddings --document-table prepared_data/document_table.csv --output-dir prepared_data/embeddings
python -m src.train --dimension all --output-dir checkpoints
```

`src/train.py` loads data, trains the regression heads, and orchestrates the run.
`src/test.py` handles inference, validation/test metrics, and reports.
`src/plot.py` generates PNGs for score ranges, ranking, and MAE/RMSE/within-1
accuracy. Evaluation and plotting run automatically after training. Each
dimension's outputs go under `<output-dir>/<dimension>/`, with combined plots
in `<output-dir>/`. The launchers use the same module entry points.

## Agreed training decisions

| Decision | Current choice |
| --- | --- |
| Human supervision target | Arithmetic mean of the two valid human ratings for each document and dimension; retain the original ratings |
| Prediction | Five continuous scores on the original 0–5 rubric scale |
| Main training loss | Mean squared error (MSE), replacing the earlier weighted-kappa-loss proposal |
| Label transformation | Preserve fractional averages; do not round, z-score, or percentile-transform human training targets |
| Initial experiment | Human-only supervision; one multilingual rater shared across languages |
| Data splits | Independent 70% train / 10% validation / 20% test per dimension, stratified within each language |

The starting architecture is a frozen pretrained multilingual encoder with five small regression heads. Encoder checkpoint, pooling, optimization settings, and later ablations are working defaults to validate, rather than established best choices. Keep unbounded regression outputs for diagnostics and clip to [0, 5] for deployed scores without rounding.

## 1. Project idea

SEA-Rater investigates whether quality signals adapted to Southeast Asian (SEA) languages can identify more useful language-model training data.

The project connects three kinds of evidence:

1. **Rater reliability:** Small multilingual models predict document scores that agree with native-speaker judgments.
2. **Selection effectiveness:** Data selected using these scores improves language-model performance under a controlled training budget.
3. **Explanation:** Analysis of the selected documents shows what changes in language coverage, domains, noise, and cultural content accompany those improvements.

The work is inspired by Meta-rater's combination of multiple quality signals and JQL's use of lightweight annotators built on pretrained multilingual representations. SEA-Rater adds native-speaker supervision across eight SEA languages and investigates the value of a Cultural Nuances dimension. The contribution must be established through controlled comparisons; better rater agreement alone does not establish better pre-training data.

## 2. Scope

**Target languages:** Indonesian, Vietnamese, Thai, Malay, Tagalog, Khmer, Lao, and Burmese. Dataset-specific language and script identifiers will be mapped explicitly during preparation.

The completed annotation follows *SEA-Rater Human Annotation Guidelines v2.1*, using scores from 0 to 5 on five dimensions:

| Dimension | Intended property |
| --- | --- |
| Educational Value | Usefulness for primary- and middle-school learning |
| Reasoning | Logical relationships, explanations, and depth of argumentation |
| Professionalism | Expertise and prerequisite knowledge needed to understand the content |
| Cleanliness | Freedom from extraction artifacts, intrusive boilerplate, and formatting noise |
| Cultural Nuances | Natural local expression, cultural context, and community-specific perspective as defined in the annotation rubric |

Cultural Nuances is a specific selection signal. A document with little local cultural content can still be valuable for other purposes. Natural code-switching and valid script variation should be preserved throughout preparation and evaluation.

## 3. Research questions

- **RQ1 — Quality prediction:** How reliably can small multilingual raters reproduce human judgments across dimensions and languages?
- **RQ2 — Supervision:** How do human labels, LLM labels, and their combination affect rater reliability and subsequent data selection?
- **RQ3 — Selection:** Does multidimensional selection improve language-model performance relative to random selection and simpler quality policies? Do language-specific policies help beyond one shared policy?
- **RQ4 — Cultural contribution:** What changes when Cultural Nuances is included, both in the selected corpus and in downstream cultural performance?

## 4. Model roles

| Component | Input and training signal | Output | Role |
| --- | --- | --- | --- |
| Quality rater | Documents and human or LLM quality labels | Scores for each dimension | Score a large corpus efficiently |
| Proxy language model, if used | Candidate selected corpora and a language-modeling objective | Loss on held-out language-model validation data | Help search for score-combination weights |
| Evaluation language model | A selected corpus and a language-modeling objective | Downstream benchmark results | Test whether the selected data is useful |

These are separate training decisions. A quality rater can reuse a pretrained encoder while a language model used to evaluate selection is trained from random initialization. Proxy models are needed only if the chosen selection method uses proxy-based weight search.

## 5. Overall workflow

### Step 0 — Complete human annotation

**Status: Complete.**

Native-speaker annotators have scored documents on the five dimensions. Preserve both annotators' original scores, explanations, flags, document identifiers, and guideline versions.

**Output:** Original human annotation records.

### Step 1 — Prepare the training and evaluation data

Consolidate annotation files, check score ranges and missing values, and inspect disagreement. For each dimension, use the mean of the two valid human scores as the baseline target: ratings 2 and 3 become 2.5. Preserve individual scores and review flags. Do not fill missing labels with zero or use LLM scores to resolve them automatically. Record exclusions and any adjudication separately.

Use a fixed, independent **70% train / 10% validation / 20% test split for each dimension**, stratified on that dimension’s score ranges within each language (split seed 42). A document may belong to different splits for different dimensions. Within each dimension, keep paired annotations, document chunks, and known duplicate groups together. Reuse the existing per-dimension manifests for every comparison; small count differences from the target ratios arise from integer allocation and grouping. Measure language coverage, score distributions, and tokenized document lengths.

**Output:** A versioned annotation dataset, five `split_manifest_<dimension>.csv` files, and a short data-quality report.

### Step 2 — Train a multilingual quality-rater baseline

**Working default:** Reuse a pretrained multilingual encoder, initially frozen, and train five small regression heads, one per dimension. Each head learns from all eight languages. The training guide proposes mmBERT-base and document embeddings pooled across token chunks, with settings to verify in a pilot.

**Agreed objective:** Predict continuous human-average scores with MSE. Train each head independently on its own dimension’s training split, using language-balanced batches and the shared frozen embeddings. Use a language–dimension training-mean predictor as a reference baseline. A small Huber-loss comparison can follow; it does not replace MSE by default.

Evaluate MAE, RMSE, signed bias, within-0.5/within-1 accuracy, and Spearman separately for every language and dimension. Select the initial checkpoint using validation macro-MAE, with Spearman and per-group errors reported alongside it. Keep the human test set for final reporting. Compare encoder fine-tuning only after the baseline is established.

See [01_train_rater.md](01_train_rater.md) for the data contract, loss, starting configuration, training sequence, and required outputs.

**Output:** A baseline rater, per-language results, and reproducible training settings.

### Step 3 — Compare human and LLM supervision

Compare human-only, LLM-only, and combined supervision while keeping the rater architecture and human evaluation set fixed. To isolate label-source effects, first compare labels on the same training documents. Separately test the benefit of a larger weak-labeled training pool.

Audit LLM scores against the human averages before using them as weak labels. The observed Cultural Nuances results show useful rank association alongside poor numerical agreement; that alone does not identify the direction or cause of the errors. Compare paired scores, signed errors, and score distributions per language and dimension.

Separate human and LLM z-scores can diagnose differences in their means and spreads. Z-scoring does not change Spearman or establish agreement on the original rubric. If numerical calibration is useful, fit bias correction or a linear mapping to human scores using training/calibration pairs, choose on validation data, and evaluate on untouched test pairs. Report raw and calibrated LLM results separately. Keep human labels unchanged.

Exact matching needs care when integer LLM ratings are compared with fractional human averages. Use continuous-error metrics and tolerance-based agreement as the primary numerical checks.

LLM labeling means assigning scores to documents; generating new synthetic documents is a separate augmentation experiment. Human validation and test documents must remain excluded from rater training and augmentation.

**Output:** Evidence about supervision quality, data quantity, and performance by language.

This comparison can follow the human-only baseline; it does not have to block an initial corpus-scoring pilot.

### Step 4 — Build and score the candidate corpus

Use a fixed, documented corpus release. FineWeb2 is the initial candidate source. Audit language identification, script handling, duplicate contamination, and available unique tokens before fixing a training budget.

Apply the trained raters to the corpus and retain the five scores, document metadata, and any selected heuristic signals. Audit high- and low-scoring examples in each language. Keep corpus preparation consistent with the text representation used to train the raters.

**Output:** A scored candidate corpus with token counts, metadata, and scoring provenance.

### Step 5 — Define and compare selection policies

Begin with interpretable baselines, then add learned weighting if warranted. Candidate comparisons include random selection, Educational Value alone, equal weighting of a fixed signal set, learned global weights, and language-specific weights.

Score transformation at this stage is a selection-policy decision, separate from human-target preparation and human–LLM calibration. Compare raw scores with z-scores and tie-preserving percentiles if useful; no transformation is assumed best from histograms alone. Specify whether statistics are calculated globally or within languages, fix a representative scored-corpus reference population, and apply the same procedure across comparable experiments. Do not use human test labels to fit a selection transformation.

Per-language transformations express relative standing within each language and do not establish equal absolute quality across languages. Guard against zero or very small standard deviations and preserve ties for percentile scores. Under fixed language token budgets, within-language ranking is a natural comparison; global selection requires an explicit language-allocation policy.

**Statistical clarification:** Ordinary z-score standardization preserves ranks and therefore does not change Spearman correlation for the same nonconstant paired score vectors. It is not a required preprocessing step for calculating Spearman and does not make a skewed distribution Gaussian. Human regression targets remain on their defined scoring scale.

If Meta-rater-style search is adopted, train proxy models on candidate selections, predict their validation losses, and validate promising weights. Check that proxy preferences transfer to the evaluation-model setting.

**Output:** Versioned selected datasets and their selection policies, with token budgets, retention rates, and repetition policies.

### Step 6 — Test selection through language-model training

Train comparable language models on the selected datasets. Scratch training and continued pre-training answer different questions and are possible experimental settings; the initial setting, architecture, adaptation method, and budget remain open.

Within each comparison, control model initialization, training recipe, consumed tokens, and language allocation unless one of those factors is deliberately being studied. Report unique selected tokens and repeated exposures separately.

For the cultural ablation, select equal-budget datasets with and without Cultural Nuances, train comparable models, and evaluate both general language ability and cultural understanding. Refit the reduced selection policy if its weights are normally learned.

**Output:** Downstream performance comparisons with per-language results and uncertainty estimates.

### Step 7 — Analyze results and prepare the publication

Analyze what each policy retains: source domains, document lengths, language and script composition, code-switching, cultural references, noise, and overlap with other selections. Use independent human spot checks to support interpretations of automatic scores.

Build a benchmark coverage matrix for the eight languages. Candidate evaluations include Belebele, SIB-200, suitable reasoning tasks, and culturally grounded SEA benchmarks. Account for missing language coverage and shared source material; Belebele and SIB-200 both draw on FLORES-200.

Connect rater reliability, changes in the corpus, and downstream outcomes. Report negative results and language-specific trade-offs alongside aggregate improvements.

**Output:** Main results, ablations, corpus analysis, documented limitations, and reproducibility materials.

## 6. Decisions to settle in the detailed guides

| Area | Open decision |
| --- | --- |
| Annotation preparation | Missing/invalid labels, review flags, and duplicate groups; independent 70/10/20 per-dimension splits and human-mean targets are fixed |
| Rater training | Validate the proposed checkpoint, context handling, and hyperparameters; MSE regression is the agreed baseline |
| LLM supervision | Labeling model, annotation budget, and combination with human labels |
| Corpus | Release, language/script mappings, candidate-pool sizes, and unique-token availability |
| Selection | Normalization scope, signal set, global versus language-specific weights, and whether to use proxies |
| Language-model experiments | Scratch or continued training, model architecture, adaptation method, and feasible budget |
| Evaluation | Task-language coverage, cultural evaluation, downstream language-model checkpoint selection, and uncertainty reporting; the initial rater uses validation macro-MAE |

Detailed guides specify the inputs, procedure, checks, outputs, and experiment settings for each step. [01_train_rater.md](01_train_rater.md) is the first guide and covers workflow Steps 1 and 2 together. Later guides will cover supervision comparisons, corpus scoring, selection, and language-model experiments.

## 7. References

- [Meta-rater: A Multi-dimensional Data Selection Method for Pre-training Language Models](https://arxiv.org/html/2504.14194v4) — multidimensional score aggregation and proxy-based selection-weight search.
- [Judging Quality Across Languages (JQL)](https://aclanthology.org/2025.emnlp-main.449/) — lightweight multilingual annotators using pretrained representations.
- [mmBERT model card](https://huggingface.co/jhu-clsp/mmBERT-base) — candidate multilingual encoder.
- [FineWeb2 dataset card](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) — candidate corpus and language-specific data documentation.
- [Belebele](https://github.com/facebookresearch/belebele) and [SIB-200](https://arxiv.org/abs/2309.07445) — candidate multilingual evaluation datasets.
- [SEA-NLI](https://arxiv.org/html/2606.03284v1) — candidate evaluation of culturally grounded SEA understanding.
- [SciPy Spearman correlation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html) and [z-score](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.zscore.html) — statistical definitions.
- [PyTorch MSELoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.MSELoss.html) — the agreed regression loss; see the training guide for its application to human averages.
- *SEA-Rater Human Annotation Guidelines v2.1* — internal project document used for the completed annotation.
