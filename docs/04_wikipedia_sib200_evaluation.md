# SEA-Rater: Wikipedia and SIB-200 Evaluation Protocol

## 1. Objective and scope

Compare decoder language models trained on different FineWeb2 selections or with different loss-weighting methods. Evaluate ordinary, unweighted next-token loss and perplexity on:

- **Wikipedia:** 1,000 randomly sampled eligible articles per target language.
- **SIB-200:** all 1,004 sentences per target language from its three official splits, reported overall and by topic.

This protocol uses SIB-200 as a topic-labeled text corpus. It does **not** ask the model to predict topic labels and does not measure topic-classification accuracy. Neither evaluation requires fine-tuning.

These sample sizes define the initial pilot. They do not guarantee enough precision to distinguish very small improvements.

## 2. Datasets

| Dataset | Text and provenance | Evaluation unit | Main results |
|---|---|---|---|
| [Wikimedia Wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia) | Cleaned encyclopedia articles from language-specific Wikipedia editions | Article | Loss and perplexity per language |
| [SIB-200](https://huggingface.co/datasets/Davlan/sib200) | Topic-labeled sentences from FLORES-200 and their aligned translations; original sources are Wikinews, Wikijunior, and Wikivoyage | Sentence | Loss and perplexity per language and topic |

Pin the dataset revision and language configuration. Wikipedia is a reference-text source, not a guarantee that every article meets a quality threshold. SIB-200 contains translated text and should be described as such.

### Wikipedia preparation

1. Load the chosen Wikipedia snapshot and language subset.
2. Apply predefined cleaning criteria, such as rejecting empty text, severely corrupted extraction, and pages containing almost entirely lists or navigation.
3. Check for exact and near-duplicate overlap with the union of the training documents used by all compared models. Exclude overlapping evaluation articles using one common rule.
4. Randomly sample **1,000 eligible articles per language** with a fixed seed. A seed such as `42` is a reproducibility choice, not a methodological requirement.
5. Save the selected article IDs and URLs, dataset revision, preprocessing settings, and sampling seed.

If fewer than 1,000 eligible articles remain in a language, use the available articles without replacement and report the actual number. Do not filter articles using model loss or SEA-Rater scores to obtain a favorable evaluation set.

The released Wikipedia dataset has a split called `train`. An article sampled from it can serve as held-out evaluation text for our experiment only if it is kept independent of our training and model-selection process. Keep all chunks of an article in the same experimental split.

### SIB-200 preparation

Combine the standard seven-class version's official splits within each language:

| Original split | Sentences |
|---|---:|
| Train | 701 |
| Validation/development | 99 |
| Test | 204 |
| **Combined** | **1,004** |

Retain each sentence's ID, original split, text, and topic. Use the text as the model input; use the topic only to group evaluation results. Check training overlap and duplicate records before evaluation. If exclusions change the counts, report both the original and retained counts.

| Topic | Train | Development | Test | Combined per language |
|---|---:|---:|---:|---:|
| Science/technology | 176 | 25 | 51 | 252 |
| Travel | 138 | 20 | 40 | 198 |
| Politics | 102 | 14 | 30 | 146 |
| Sports | 85 | 12 | 25 | 122 |
| Health | 77 | 11 | 22 | 110 |
| Entertainment | 65 | 9 | 19 | 93 |
| Geography | 58 | 8 | 17 | 83 |
| **Total** | **701** | **99** | **204** | **1,004** |

Counts are from Table 1 of the [SIB-200 paper](https://aclanthology.org/2024.eacl-long.14.pdf). Keep all retained sentences; there is no need to downsample larger topics to match smaller ones.

## 3. Development versus final testing

Pooling the SIB-200 splits is suitable for this custom language-model diagnostic when the evaluated text has not been used to train or select the models. Preserve the original split metadata even after pooling.

- If these results guide dimension weights, model choices, or other training decisions, label the pooled set as **development/diagnostic data** and reserve separate untouched data for final reporting.
- If all decisions are already fixed and the data have remained independent, the pooled corpus can serve as a custom held-out evaluation set. Clearly state the pooling procedure.
- If standard SIB-200 topic-classification accuracy is added later, use a separately documented classification protocol and its official test split. Do not claim that this test is untouched if its examples already influenced development through the pooled evaluation.

For continued pretraining, overlap checks cover the training data we control. Exposure during the base model's original pretraining may remain unknown.

## 4. Computing loss and perplexity

Use the same LM tokenizer, context limit, special-token policy, and scoring procedure for every compared model. Evaluate in inference mode with dropout disabled.

For document or sentence d, save:

- N_d: number of valid next-token prediction targets.
- S_d: sum of negative log-probabilities assigned to those targets.

Using natural logarithms:

$$
S_d = -\sum_{t\in T_d}\log P_\theta(x_{d,t}\mid c_{d,t}),
\qquad N_d = |T_d|,
$$

where T_d contains the scored positions and c_{d,t} is the preceding context available under the fixed evaluation procedure.

For any evaluation group G:

$$
L_G = \frac{\sum_{d\in G} S_d}{\sum_{d\in G} N_d},
\qquad
\mathrm{PPL}_G = \exp(L_G).
$$

Groups include all Wikipedia articles in one language, all SIB-200 sentences in one language, or one language-topic combination.

**Do not average per-document perplexities or unweighted batch losses.** Aggregate loss sums and valid-token counts first. Do not apply SEA-Rater importance weights during evaluation, even if a model used weighted loss during training.

### Context handling

- **Wikipedia:** use a fixed procedure for long articles, such as sliding windows with a recorded stride. Count each scored target token once; exclude overlap used only as context. Reset context at article boundaries.
- **SIB-200:** score each sentence independently. Group its results by topic afterward. Do not concatenate unrelated sentences from one topic into a shared context.
- Exclude padding from loss. Apply the same BOS/EOS policy to every model and document whether the first content token and EOS are scored.
- Accumulate across all batches or devices before computing group means.

## 5. Reporting

### Wikipedia: one row per model and language

| Model/method | Language | Articles | Valid target tokens | Loss | PPL | Loss difference vs random |
|---|---|---:|---:|---:|---:|---:|
| Random | ... | ... | ... | ... | ... | 0 |
| Quality selection or weighting | ... | ... | ... | ... | ... | ... |

### SIB-200: one row per model, language, and topic

| Model/method | Language | Topic | Sentences | Valid target tokens | Loss | PPL | Loss difference vs random |
|---|---|---|---:|---:|---:|---:|---:|
| Random | ... | Science/technology | ... | ... | ... | ... | 0 |
| Quality selection or weighting | ... | Science/technology | ... | ... | ... | ... | ... |

Also report the pooled SIB-200 result per language. For each fixed evaluation group:

$$
\Delta L_G = L_{\mathrm{method},G} - L_{\mathrm{random},G}.
$$

Negative values indicate improvement over the random-training baseline on that group.

Keep Wikipedia and SIB-200 results separate. Report per-language results, followed by an equal-weight average of the language losses for each corpus. This macro loss is not the token-pooled loss across languages. If an equal-topic summary is also reported, identify it separately from the pooled SIB-200 loss.

Save per-example results so that aggregation and uncertainty analysis do not require rerunning the models:

```text
model_id, training_seed, dataset, dataset_revision, language,
example_id, original_split, topic, source_url,
nll_sum, valid_target_tokens
```

Use blank topic fields where no topic label exists; preserve URLs when available.

## 6. Interpretation and uncertainty

- Compare models on identical text within the same language and topic.
- Lower loss on travel than science does not show that travel text is higher quality or that the model has better travel knowledge.
- Wikipedia uses article context while SIB-200 uses short isolated sentences. Their absolute perplexities measure different conditions.
- Per-topic SIB-200 groups contain only 83–252 sentences before exclusions. Report uncertainty instead of treating every small difference as meaningful.
- For paired bootstrap intervals, resample the same evaluation units for both models and recompute the token-weighted loss difference. Use articles for Wikipedia; for SIB-200, cluster by source article when that metadata is available, or describe sentence-level resampling as an approximation.
- SIB-200 translations are aligned across languages. Preserve shared sentence-ID grouping when estimating uncertainty on an aggregate across languages.
- Evaluation-sampling uncertainty and training-seed variation are different. Report training variation separately if multiple runs are available.

This experiment measures encyclopedic and topic-specific next-token prediction. It does not by itself establish general reasoning ability, cultural competence, or topic-classification performance. Retain the existing random-web held-out results as complementary evidence.

## 7. References

1. [Wikimedia Wikipedia dataset card](https://huggingface.co/datasets/wikimedia/wikipedia).
2. [SIB-200 dataset card and files](https://huggingface.co/datasets/Davlan/sib200).
3. [SIB-200: A Simple, Inclusive, and Big Evaluation Dataset for Topic Classification in 200+ Languages and Dialects](https://aclanthology.org/2024.eacl-long.14/), especially Section 2 and Table 1.

The sampling and reporting choices above are the proposed SEA-Rater protocol. They are not the original SIB-200 classification evaluation procedure.
