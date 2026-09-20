# 01 — Train the multilingual SEA-Rater

This guide covers annotation preparation and the first rater-training experiment: workflow Steps 1 and 2 in [README.md](README.md). The filename numbers the guide, not a separate model-training stage.

**Agreed method:** Learn continuous human-average scores on the original 0–5 scale using mean squared error (MSE).

**Deliverable:** One multilingual rater that assigns five quality scores to a document, accompanied by per-language evaluation, a reproducible configuration, and document-level predictions.

This is a training specification. The annotated rows, training scripts, and trained checkpoints have not been supplied or produced as part of writing this guide. Examples below are illustrative. Numerical hyperparameters are starting settings to validate, not measured best settings.

## 1. What the model learns

The input is the document text seen by the annotators. The model produces one continuous score for each dimension in this fixed order:

1. `educational_value`
2. `reasoning`
3. `professionalism`
4. `cleanliness`
5. `cultural_nuances`

For document i and dimension d, the training target is:

$$
y_{i,d}=\frac{r^{(1)}_{i,d}+r^{(2)}_{i,d}}{2}
$$

For example, ratings 2 and 3 give a target of 2.5. A prediction of 2.4 is valid. Do not round the target to 2 or 3, and do not convert the task into eleven classes for half-point averages.

This formulation treats a one-point difference as comparable across the rubric. It estimates the mean human judgment; it does not imply that the rubric measures quality with arbitrary precision. Preserve both original ratings because averaging removes information about disagreement.

## 2. Fixed decisions and starting defaults

| Component | Choice | Status |
| --- | --- | --- |
| Target | Mean of two valid human ratings per dimension | Agreed |
| Output | Five continuous scores on the 0–5 scale | Agreed |
| Main loss | MSE | Agreed |
| Human-label normalization | None; retain raw rubric scores | Agreed |
| Initial supervision | Human-only | Baseline |
| Encoder | `jhu-clsp/mmBERT-base`, frozen | Proposed starting checkpoint |
| Prediction heads | Five independent small MLP regression heads | Starting architecture |
| Data splits | Independent 70/10/20 train/validation/test per dimension, stratified within language | Agreed |
| Text representation | Mean pooling across all document chunks | Starting document strategy |
| Training sampling | Equal language representation in each batch | Starting multilingual policy |
| Checkpoint selection | Lowest validation macro-MAE | Fixed rule for the first experiment |
| Deployment output | Clip raw predictions to [0, 5]; do not round | Baseline inference rule |

Each head is shared across all target languages. Do not train forty separate language–dimension models for this baseline. The five heads share the same frozen text representation; the encoder itself does not learn from these labels in the initial experiment.

## 3. Prepare a document-level dataset

### 3.1 Required fields

Keep one record per document with these fields:

| Field | Purpose |
| --- | --- |
| `doc_id` | Stable document identifier |
| `language` | Canonical project language identifier |
| `source_language` | Original identifier, such as `vie`, `thai`, or `fil` |
| `text` | Exact text representation used for the ratings |
| `text_hash` | Detect changed text and exact duplicates |
| `duplicate_group_id` | Group exact/near duplicates and known alternate versions |
| `annotator_ids` | Identify the two human raters without putting identities into model input |
| `human_scores` | Two five-element vectors in the fixed dimension order |
| `human_mean` | Five arithmetic means computed from those vectors |
| `guideline_version` | Annotation rubric version, currently v2.1 |
| `flags` | Missing-label, uncertainty, script, review, and other audit information |
| `split_<dimension>` | `train`, `validation`, `test`, or `excluded` from that dimension’s fixed manifest |

Preserve source/domain, annotation explanations, and any existing final/adjudicated scores when available. These are audit fields, not input features for the first rater. Never include annotation scores, explanations, LLM judgments, or split labels in the text fed to the encoder.

An illustrative record, omitting hashes and provenance details for readability:

```json
{
  "doc_id": "example_vie_0001",
  "language": "vie",
  "source_language": "vie",
  "text": "<the original document text>",
  "duplicate_group_id": "example_group_0001",
  "annotator_ids": ["annotator_A", "annotator_B"],
  "human_scores": [[2, 1, 2, 5, 3], [3, 2, 2, 4, 4]],
  "human_mean": [2.5, 1.5, 2.0, 4.5, 3.5],
  "guideline_version": "2.1",
  "flags": [],
  "split_educational_value": "train",
  "split_reasoning": "validation",
  "split_professionalism": "train",
  "split_cleanliness": "test",
  "split_cultural_nuances": "train"
}
```

Create an explicit language mapping. Confirm how Filipino/Tagalog and Malay identifiers are used by the annotation source and candidate corpus; do not silently equate every dataset's labels. Use the actual available language inventory and counts, rather than assuming that all eight languages have exactly 1,000 usable records.

### 3.2 Validation and review

1. Check document IDs, nonempty text, dimension order, and finite numeric scores in [0, 5].
2. Confirm that the two ratings refer to the same document text and rubric version.
3. Recompute `human_mean` and compare it with any supplied average. Preserve the supplied value separately if it differs, and resolve the discrepancy before training.
4. Count missing labels, invalid records, duplicates, review flags, and disagreements by language and dimension.
5. Preserve natural code-switching, punctuation, script variants, and noise visible to annotators. Removing quality defects after annotation can make the text inconsistent with its label, especially for Cleanliness.

Determine label validity independently for each dimension. Use documents with a valid human-average target for that dimension; a missing score on another dimension does not exclude the document. Log exclusions by language and dimension and never replace missing labels with zero. The current cleaned CSVs already contain averaged targets; original annotator ratings must be retained separately where available.

Large human disagreement is not by itself an exclusion rule. Review it, retain the original scores, and document any adjudication. Do not remove examples because their LLM score disagrees with humans, or because they belong to a rare score category.

**Output:** A versioned document table plus an audit report with actual retained and excluded counts.

## 4. Fix independent train/validation/test splits per dimension

The agreed policy is **70% train / 10% validation / 20% test for each dimension**, using split seed **42**. The five manifests are independent: a document can be training data for one head and test data for another.

- Filter labels independently for each dimension, retaining its valid documents.
- Stratify within each language on that dimension's human-score bins: [0,1), [1,2), [2,3), [3,4), [4,5]. These bins affect splitting only; fractional training targets remain unchanged.
- Keep paired annotations, chunks, and duplicate groups in one split within each dimension. Audit known translated/template variants when present; do not infer that repeated IDs across languages necessarily identify identical text.
- Integer allocation and group sizes can produce small deviations from 70/10/20; report actual counts.
- Write `split_manifest_<dimension>.csv` once and reuse it for every comparison on that dimension. The existing prepared manifests already follow this policy and should be retained.
- Do not oversample or balance validation/test distributions by score.

Train each head independently on its own training manifest using the shared frozen embeddings. A head's held-out labels must never enter that head's optimization. If introducing a shared trainable encoder or other shared learned parameters later, design a compatible held-out protocol first.

Training data fits the heads and any learned calibration. Validation data selects hyperparameters and checkpoints. Test data is reserved for reporting after choices are fixed. Human and LLM labels for a document inherit the same split **for the dimension being evaluated**.

**Output:** Five fixed split manifests and per-dimension, per-split language/score counts.

## 5. Build a frozen multilingual document encoder

### 5.1 Load and freeze the encoder

Use `AutoTokenizer` and the encoder representation from `AutoModel` for the proposed `jhu-clsp/mmBERT-base` checkpoint. Pin the model/tokenizer revision and record the installed library versions once a pilot load succeeds. The official [mmBERT model card](https://huggingface.co/jhu-clsp/mmBERT-base) documents the checkpoint and loading interface.

Set encoder parameters to `requires_grad=False`, keep the encoder in evaluation mode, and extract representations without gradients. Freezing parameters alone does not disable dropout. If the encoder is placed inside a larger module, ensure that calling the larger module's `train()` does not accidentally reactivate encoder dropout.

### 5.2 Handle long documents consistently

Human labels describe documents, so the default representation should cover the complete rated text:

1. Tokenize without silently truncating the document.
2. Split token IDs into non-overlapping chunks with a maximum encoded length of **2,048 tokens including special tokens**. Reserve the tokenizer's required special-token budget.
3. Encode each chunk. Mean-pool the final hidden states over content tokens, excluding padding and added special tokens.
4. Combine chunk embeddings using a weighted mean, where each weight is the number of content tokens in that chunk.
5. Produce one embedding per document and apply the training loss once per document.

Do not label every chunk as if it were an independent document or let long documents receive more supervised weight merely because they have more chunks. Do not count overlapping tokens twice; the baseline uses no overlap.

This pooling strategy limits memory and retains coverage, but it cannot model interactions across chunk boundaries as a single full-context encoder would. Audit token lengths and chunk counts before the full embedding pass. If extreme documents make the pass infeasible, document a deterministic handling policy and rerun the same policy across training and evaluation; do not introduce an undisclosed chunk cap.

### 5.3 Cache the embeddings

Because the encoder is frozen, compute embeddings once and reuse them across head-training runs. Cache by (language, document ID), text hash, checkpoint revision, tokenizer revision, chunking/pooling configuration, and numeric precision.

A shared label-free embedding cache can serve all five per-dimension manifests. No labels are needed to extract embeddings. Any future learned transformation of embeddings must respect the training split for the head being evaluated. If the encoder or text representation changes, invalidate the relevant caches.

## 6. Add five regression heads

For an encoder embedding of size H, use one head per dimension:

`Linear(H, 128) → GELU → Dropout(0.1) → Linear(128, 1)`

Read H from the encoder configuration. During training, each head produces one score per document in its own batch. During corpus inference, concatenate all five heads’ outputs in the fixed dimension order to obtain `[batch_size, 5]`.

The final layer has no softmax, sigmoid, or rounding operation. Compute the training loss on raw outputs. During evaluation and deployment, also calculate `clip(raw_prediction, 0, 5)` and preserve both values.

This is a compact baseline, not an architecture claim. If it overfits, simplify to a linear head before making the model larger. Do not fine-tune the encoder until the frozen baseline is understood.

## 7. Train with MSE

For a batch of B valid documents from dimension d’s training split, optimize that head’s loss:

$$
\mathcal L_d=\frac{1}{B}\sum_{i=1}^{B}(\hat y_{i,d}-y_{i,d})^2
$$

Each head has its own optimizer and batches; there is no combined five-dimension loss. With balanced language counts in each batch, each language contributes equally in expectation. For eight languages and batch size 64, draw eight training documents from each language. Shuffle each language's pool and cycle/reshuffle smaller pools as needed. Log actual exposures and define an epoch as `ceil(N_train / 64)` optimizer steps for this sampler.

Do not additionally use inverse-frequency language loss weights with this balanced sampler. Do not add inverse-score-frequency weights or oversample rare score bins in the first experiment: these change the objective and should be explicit later comparisons.

The essential PyTorch operations are:

```python
import torch
import torch.nn.functional as F

# Batch contains only training documents for the current dimension.
# Predictions and targets have shape [batch_size].
# Targets are this dimension’s human averages, stored as float32.
predictions = head(document_embeddings.float()).squeeze(-1)
loss = F.mse_loss(predictions, targets.float())

optimizer.zero_grad(set_to_none=True)
loss.backward()
torch.nn.utils.clip_grad_norm_(head.parameters(), max_norm=1.0)
optimizer.step()

# In a separate evaluation pass, with head.eval() and no gradients:
# deployed_scores = raw_predictions.clamp(0.0, 5.0)
```

This illustrates the loss and update, not a complete training program. The data loader must enforce the shape, dimension order, and language-sampling contract. See [PyTorch MSELoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.MSELoss.html) for the reduction behavior.

**Do not normalize human labels before this loss.** Their scale is already defined. Standardizing each language/dimension separately would change what the model learns and the relative weight given to errors.

The earlier weighted-kappa-loss proposal is superseded by MSE for the baseline. Spearman is an evaluation metric, not the training loss.

## 8. Starting configuration

These settings apply to training the small heads on cached frozen embeddings. They are not recommendations for full encoder fine-tuning.

| Setting | Initial value |
| --- | --- |
| Encoder | `jhu-clsp/mmBERT-base`; pin a verified revision |
| Maximum encoded chunk length | 2,048 tokens |
| Pooling | Content-token mean, then token-count-weighted chunk mean |
| Head hidden size / dropout | 128 / 0.1 |
| Loss | MSE, equal dimension weights |
| Head optimizer | AdamW |
| Learning rate | 0.001 |
| Weight decay | 0.01 on weight matrices; 0 on biases |
| Learning-rate schedule | Constant for the initial baseline |
| Head-training batch size | 64 documents, balanced by language |
| Maximum epochs | 30 |
| Gradient norm limit | 1.0 |
| Validation frequency | Once per epoch |
| Early stopping | Patience 5 epochs; improvement of at least 0.001 macro-MAE points |
| Checkpoint metric | Validation macro-MAE of clipped predictions; lower is better |
| Split seed | 42, unless a valid split already exists |
| Training seeds | Pilot: 42; repeated experiment: 42, 43, 44 |
| Head precision | Float32 |

Choose the encoder's chunk batch size from available hardware; it is separate from the 64-document head batch. Mixed precision for frozen feature extraction is optional, must be recorded, and should not produce nonfinite embeddings. No specific GPU runtime is assumed here.

Set seeds for Python, NumPy, PyTorch, and the sampler. Keep the split fixed across training seeds. Record hardware and software versions; exact reproducibility across different releases and devices is not guaranteed by seeds alone. See [PyTorch reproducibility notes](https://docs.pytorch.org/docs/stable/notes/randomness.html).

## 9. Run the experiment in this order

1. **Audit and split:** Complete the checks in Sections 3–4. Save immutable input and split versions.
2. **Prepare a baseline predictor:** Calculate the training-set mean for each language and dimension. Predict that constant for every validation/test document in that group. Also include a training-median predictor when comparing MAE, because the median is the optimal constant under absolute error.
3. **Pilot the representation:** Encode a small sample covering all available languages, short and long documents, and script conditions. Confirm finite embeddings, stable document alignment, and frozen encoder behavior.
4. **Cache document embeddings:** Apply the same deterministic representation to each split and retain provenance.
5. **Train the MSE heads:** Run seed 42 using the initial settings. Log training loss, validation metrics, and prediction ranges each epoch.
6. **Select a checkpoint:** Save the best eligible validation macro-MAE checkpoint. For ties within the improvement tolerance, retain the earlier checkpoint. Restore it after early stopping.
7. **Resolve concrete problems on validation data:** Inspect underfitting, overfitting, rare-score errors, and language-specific failures. If necessary, test a small learning-rate set such as 0.0003 and 0.001 or a linear-head alternative. Log every attempt; do not search on the test set.
8. **Repeat the selected recipe:** Run training seeds 42, 43, and 44 with the same split, reusing the seed-42 run if its recipe is unchanged. Select each run's checkpoint using the same validation rule. Report every seed; do not pick the best test seed.
9. **Freeze the reporting protocol and evaluate:** Produce the final held-out results, predictions, uncertainty estimates, and model card.

All candidate loss/architecture variants intended for the same comparison should be chosen using validation data before their final test evaluation. Further changes prompted by test inspection are exploratory and require fresh held-out evidence for confirmatory claims.

## 10. Evaluate ranking and numerical agreement separately

Evaluate against the unrounded human averages. Report results for every available language–dimension pair, with the number of documents in each cell.

| Metric | Interpretation |
| --- | --- |
| MAE | Average absolute error in rubric points |
| RMSE | Error measure that emphasizes large mistakes |
| Signed bias | Mean(prediction − human mean); positive means overscoring |
| Within 0.5 | Percentage with absolute error ≤ 0.5 |
| Within 1 | Percentage with absolute error ≤ 1.0 |
| Spearman | Agreement in document ranking |
| Out-of-range rate | Fraction of raw outputs below 0 or above 5 |

**Primary evaluation convention:** Report metrics for clipped predictions, because those are the deployed scores. Also save and report raw-prediction diagnostics, including raw Spearman, bias, and out-of-range rate. Clipping can introduce ties and change Spearman; keep the convention consistent across models.

Select a checkpoint independently for each dimension. Calculate MAE on that dimension's validation documents within each language, then average equally over languages. This gives the head's validation macro-MAE; other dimensions do not affect its checkpoint selection. A pooled document-level metric can be supplementary; it must not replace the per-language table.

For Spearman, constant target or prediction vectors make the statistic undefined. Report `NA` with a reason and the number of valid cells included in any macro-average. Do not hide undefined cells or replace them with invented correlations. The constant baselines will have undefined Spearman. [SciPy documents this behavior](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html).

Exact match is not a primary regression metric. Human averages can be fractional, and continuous predictions almost never match exactly. QWK, accuracy, and macro-F1 require a separately defined categorical evaluation protocol; do not apply them directly to arbitrary floating-point scores or round the reference merely to improve agreement.

### Distribution-aware checks

Your plots show concentrated distributions, especially for Cleanliness and some Professionalism labels. Therefore:

- Compare against both constant baselines; low MAE can reflect the dominant score.
- Report errors by human-score band, with band edges and sample counts fixed before model comparison.
- Inspect prediction histograms and prediction-versus-human scatterplots per language/dimension.
- Inspect whether high-quality tails remain distinguishable rather than all predictions collapsing near the mean.
- Inspect Cultural Nuances separately, given the earlier human–LLM disagreement.

Include human–human agreement as context, but do not call it a strict performance ceiling: agreement between two individuals is a different comparison from prediction against their mean.

For uncertainty, report mean and standard deviation across training seeds. Use paired bootstrap intervals for model differences on the same test documents, resampling duplicate groups within languages where appropriate. Compare model predictions on the same held-out documents within each dimension. Bootstrap within that dimension’s languages and duplicate groups; do not assume a shared test set across dimensions. Separate test-sample uncertainty from training-seed variation.

## 11. Keep normalization and LLM calibration separate

| Stage | What to do |
| --- | --- |
| Human-supervised MSE training | Use original human averages without z-scoring |
| Human–LLM diagnostic analysis | Separate z-scores may reveal location/spread differences; they do not improve Spearman |
| LLM weak-label preparation | If needed, learn a human-scale calibration from training pairs and validate it on held-out pairs |
| Later corpus-score aggregation | Compare raw scores, z-scores, or tie-preserving percentiles as explicit selection policies |

For the first rater, no LLM score enters the target. A different loss cannot repair a systematic mismatch between an LLM's interpretation of the rubric and human judgments.

Later, compare human-only and LLM-only supervision on the same training documents, then compare combined supervision and larger weak-label pools. Preserve the same human evaluation set. Fit any LLM-to-human calibration per language/dimension only where enough paired training data exists, and report raw versus calibrated weak-label experiments separately.

Never include held-out human documents, their duplicates, or generated variants in the weak-label training pool. Generating new documents is a separate experiment from asking an LLM to score existing documents.

## 12. Limited follow-up experiments

The following are optional after the MSE baseline:

| Experiment | Question |
| --- | --- |
| Huber loss, initially delta = 1.0 | Does reducing the influence of large residuals improve held-out performance? |
| Linear head | Is the MLP overfitting the available annotations? |
| Encoder adaptation | Does updating encoder representations help languages poorly served by the frozen baseline? |
| Annotator-distribution prediction | Does preserving disagreement improve score prediction or selection? |

Keep data splits, supervision, document representation, and evaluation fixed when isolating a loss change. Huber becomes linear for large errors, so it may reduce sensitivity to noisy labels, but may also underemphasize informative rare examples. It is a comparison, not an automatic improvement. [PyTorch HuberLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.HuberLoss.html)

An annotator-distribution model would predict category probabilities and use their expected score for selection. That is a later formulation; MSE remains the agreed main experiment. Architectural experiments should support the paper's data-selection contribution rather than consume the entire experimental budget.

## 13. Save the outputs needed for reuse

The following are planned artifacts to produce when training is implemented:

| Artifact | Required contents |
| --- | --- |
| Annotation audit | Counts, exclusions, disagreements, text changes, and guideline version |
| Split manifests | One per dimension: document/group IDs, languages, scores, splits, split seed 42, and dataset version |
| Embedding manifest | Encoder/tokenizer revisions, text hashes, chunking, pooling, and precision |
| Training configuration | Dimensions, target construction, architecture, optimizer, sampler, seeds, and checkpoint rule |
| Selected checkpoint | Head weights, epoch, validation score, and exact frozen encoder reference |
| Training logs | Epoch losses, per-language validation metrics, exposures, and raw prediction ranges |
| Validation/test predictions | Document ID, language, dimension, both human ratings, human mean, raw/clipped prediction, split, and run ID |
| Evaluation report | Per-cell and macro metrics, baseline comparisons, sample counts, seed variation, and limitations |
| Rater model card | Intended use, training data, language coverage, preprocessing, inference rule, and known weaknesses |

Preserve the tokenizer and encoder revision needed to reconstruct inference, or package the permitted checkpoint alongside the heads. Reload the selected model and confirm that it reproduces saved validation predictions within a documented numerical tolerance.

## 14. Criteria for moving to a corpus-scoring pilot

- Dataset targets and splits are documented and free of known duplicate leakage.
- The rater produces finite, correctly ordered five-dimensional outputs for every supported language.
- Its errors and ranking behavior have been compared with constant predictors and inspected by language/dimension.
- Weak or nearly constant dimensions are identified explicitly; they are not hidden by an overall average.
- Inference uses the same text, chunking, pooling, checkpoint, and clipping policies as evaluation.
- The trained checkpoint, prediction files, configuration, and limitations are saved.

Start corpus scoring with a small per-language audit sample before a full run. There is no justified universal Spearman or MAE threshold that guarantees useful selection. If a dimension cannot distinguish documents reliably, investigate or treat it as an explicit ablation before relying on it for large-scale selection. Downstream language-model experiments remain necessary to establish the value of the selection method.
