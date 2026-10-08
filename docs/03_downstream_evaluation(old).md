# SEA-Rater: Downstream Evaluation Without Fine-Tuning

Protocol date: 2026-10-02

## 1. Objective and scope

Evaluate whether SEA-Rater data selection or quality-weighted training improves a pretrained causal language model's reading comprehension, causal reasoning, topic classification, and culturally grounded inference.

Use our own evaluation code for all four benchmarks. Keep the pretrained model parameters fixed: no task fine-tuning, classification-head training, or gradient updates. The main setting is zero-shot candidate-likelihood scoring.

This document specifies a proposed SEA-Rater evaluation protocol. The illustrative prompts are custom templates, not exact reproductions of each benchmark's published evaluation setup. Record and release the final templates and scoring rules with the results.

## 2. Benchmarks

| Benchmark | Task | Languages in this project | Main reporting |
|---|---|---|---|
| Belebele | Reading comprehension with four answer choices | Indonesian, Malay, Tagalog, Thai, Khmer, Vietnamese | Accuracy |
| XCOPA | Choose one of two plausible causes or effects | Indonesian, Thai, Vietnamese | Accuracy |
| SIB-200 | Classify text into seven topics | Indonesian, Malay, Tagalog, Thai, Khmer, Vietnamese | Accuracy and macro-F1 |
| SEA-NLI | Classify a culturally grounded premise–hypothesis relationship | Indonesian, Malay, Tagalog, Thai, Khmer, Vietnamese | Accuracy and macro-F1; additionally weighted-F1 |

Use the official test splits. Belebele has 900 questions per language, XCOPA has 500 test questions per language, and SIB-200 has 204 test examples per language. SEA-NLI's language subsets differ in size; record their actual counts and report its normal and hard subsets separately. The SEA-NLI paper reports weighted-F1. See the primary sources in Section 10.

Uniform random guessing gives 25% accuracy for Belebele, 50% for XCOPA, about 14.3% for SIB-200, and about 33.3% for three-label SEA-NLI. Also report majority-class accuracy for the classification datasets: exceeding uniform random guessing alone can be misleading when classes are imbalanced.

## 3. One common example format

Each dataset adapter produces the same structure:

```python
example = {
    "example_id": "unique_dataset_example_id",
    "language": "vie",
    "prompt": "...",
    "choices": ["candidate 1", "candidate 2", "..."],
    "gold_index": 0,
}
```

Use zero-based indices. The gold index is used only after prediction to calculate metrics; it must never enter the model prompt.

The evaluation code supplies every candidate. The model does not have to generate its own candidate list or produce a free-form answer.

### Should choices appear in the prompt?

| Scoring format | Choice-list requirement |
|---|---|
| Score the full candidate answer text | A list in the shared prompt is optional; the evaluator already has the candidates. |
| Score answer letters such as A/B/C/D | The prompt must provide the mapping from letters to answers. |

For this protocol, score full answer texts for Belebele and XCOPA, and label texts for SIB-200 and SEA-NLI. Include the topic list for SIB-200 and label definitions for SEA-NLI. Listing possible labels does not make an evaluation few-shot: no labeled demonstrations have been provided.

## 4. Prompt and candidate examples

All examples below are invented illustrations in English, not real benchmark test items. For the actual experiments, use each benchmark's target-language text and fixed, checked prompt and label translations. Preserve a mapping from translated labels to the original categories.

### 4.1 Belebele: passage and question

Shared prompt:

```text
Passage:
Lan visited the library on Saturday to borrow history books.
Afterwards, she returned home to read.

Question: Why did Lan visit the library?
Answer:
```

Candidates held by the evaluator:

```python
choices = [
    "to borrow history books",
    "to meet her teacher",
    "to buy food",
    "to play sports",
]
gold_index = 0
```

Include the passage as well as the question. In this custom answer-text format, the four choices are not listed inside the shared prompt.

### 4.2 XCOPA: cause or effect

For a question asking for a cause, use a cue equivalent to “because.”

```text
The road was wet because
```

```python
choices = [
    "it had just rained.",
    "the sun was shining.",
]
gold_index = 0
```

For a question asking for an effect, use a cue equivalent to “so.”

```text
It rained heavily, so
```

```python
choices = [
    "the road became wet.",
    "the road dried quickly.",
]
gold_index = 0
```

Read the cause/effect field from each dataset record. Use grammatically appropriate language-specific connectors and consistent punctuation handling.

### 4.3 SIB-200: topic classification

Shared prompt:

```text
Topics: science/technology, travel, politics, sports,
health, entertainment, geography.

Text:
The Vietnamese team won the final after a penalty shootout.

Topic:
```

```python
choices = [
    "science/technology",
    "travel",
    "politics",
    "sports",
    "health",
    "entertainment",
    "geography",
]
gold_index = 3
```

These are the seven original topic categories. For Vietnamese, for example, “thể thao” maps to “sports.” Use the same label wording and order for every checkpoint evaluated in a given language.

### 4.4 SEA-NLI: culturally grounded inference

Shared prompt:

```text
Assume the premise is true. Classify the hypothesis:
- entailment: the hypothesis follows from the premise.
- contradiction: the hypothesis conflicts with the premise.
- neutral: the premise does not determine whether it is true.

Premise: Nam brought bánh chưng to the family gathering.
Hypothesis: Nam brought food to the family gathering.

Relationship:
```

```python
choices = [
    "entailment",
    "contradiction",
    "neutral",
]
gold_index = 0
```

The intended label is entailment, using the knowledge that bánh chưng is food. Neutral means insufficient information, not that the hypothesis is false.

For real SEA-NLI records, use the native-language premise and hypothesis fields for the main experiment. Do not include the gold label, explanatory reasoning, cultural concept description, or other answer-supporting metadata in the prompt. Classify each premise–hypothesis pair separately.

## 5. Calculate candidate likelihood

For prompt $x$ and candidate answer $a_j$ containing $m_j$ tokens:

$$
s_j = \log P_\theta(a_j\mid x)
    = \sum_{t=1}^{m_j}\log P_\theta(a_{j,t}\mid x,a_{j,<t}).
$$

Each answer token is conditioned on the shared prompt and preceding answer tokens. Select:

$$
\hat y = \operatorname*{arg\,max}_j s_j.
$$

For the Belebele example, suppose the following illustrative scores are obtained:

| Candidate | Summed log-probability |
|---|---:|
| to borrow history books | **−3.2** |
| to meet her teacher | −5.8 |
| to buy food | −6.1 |
| to play sports | −7.4 |

The largest value is −3.2, so the prediction is index 0. It matches the gold index and contributes one correct prediction.

There is no need to exponentiate the scores or normalize them across candidates to identify the winner. Use a fixed tie-breaking rule and log ties, particularly during initial implementation checks.

### Primary and optional scoring rules

Use summed log-probability as the primary rule in this initial custom protocol. Longer answers can receive lower scores partly because they contain more tokens.

An optional sensitivity analysis uses mean log-probability per answer token:

$$
s_j^{\mathrm{mean}} = \frac{s_j}{m_j}.
$$

This can change the selected answer; it is a separate scoring variant, not a transformation of the final accuracy. Label the two variants explicitly. Choose the primary rule before examining test results. Per-token normalization also does not remove all biases from label frequency or wording.

## 6. Evaluation loop and implementation checks

The following is pseudocode, not a complete executable evaluator:

```python
predictions = []
gold_labels = []
records = []

for example in dataset:
    # loglikelihood scores continuation tokens only.
    scores = [
        loglikelihood(model, tokenizer, example["prompt"], candidate)
        for candidate in example["choices"]
    ]
    prediction = argmax(scores)

    predictions.append(prediction)
    gold_labels.append(example["gold_index"])
    records.append({
        "example_id": example["example_id"],
        "language": example["language"],
        "gold_index": example["gold_index"],
        "prediction": prediction,
        "candidate_scores": scores,
    })
```

Inside the likelihood scorer:

1. Set `model.eval()` and disable gradient computation, for example with `torch.inference_mode()`.
2. Combine the prompt, a fixed separator, and the candidate with an explicit tokenization policy. Verify the prompt–continuation boundary; tokenizing the strings separately and concatenating their IDs is not automatically identical to tokenizing the combined text.
3. Obtain token logits and apply log-softmax over the vocabulary.
4. Align logits with the next token: logits at position $t-1$ score the token at position $t$.
5. Gather the log-probability assigned to each actual candidate token.
6. Mask out all prompt and padding positions and sum over candidate tokens only. The first candidate token must still be included.
7. Preserve identical prompt context across candidates. If truncation is necessary, reserve space for the longest candidate and apply one documented truncation policy. Record truncation counts by task and language.
8. Use consistent BOS/EOS and delimiter handling. Do not accidentally add an EOS token to only some candidates or double-shift targets when using a model's built-in loss.

Score candidate sequences in batches for efficiency. Batching must not change token masks, context, or scores beyond ordinary numerical tolerance.

The causal model can receive a candidate's tokens to calculate its likelihood without receiving the gold answer label. The causal attention mask ensures that a token is predicted from preceding tokens; the evaluator repeats this operation for every candidate.

Before a full run, inspect a few rendered examples and confirm correct label mapping, inclusion of all answer tokens, and agreement between batched and individual scoring.

## 7. Calculate metrics

### Accuracy

$$
\mathrm{Accuracy} = \frac{1}{N}\sum_{i=1}^{N}\mathbf{1}[\hat y_i=y_i].
$$

Multiply by 100 to report a percentage. For example, 315 correct answers out of 900 gives 35% accuracy.

### Macro-F1

For class $c$:

$$
F1_c = \frac{2TP_c}{2TP_c+FP_c+FN_c}.
$$

Then average equally over the fixed label set:

$$
\mathrm{MacroF1} = \frac{1}{C}\sum_{c=1}^{C}F1_c.
$$

Use seven classes for SIB-200 and three for SEA-NLI. Specify the behavior for undefined class scores, such as zero with `zero_division=0`, and report class support counts.

### Weighted-F1

For comparison with the SEA-NLI paper, also calculate:

$$
\mathrm{WeightedF1} = \sum_{c=1}^{C}\frac{n_c}{N}F1_c,
$$

where $n_c$ is the number of gold examples in class $c$. Weighted-F1 weights classes by support; macro-F1 weights classes equally. A different prompt or likelihood-scoring protocol still prevents treating our results as an exact reproduction of published scores.

## 8. Compare SEA-Rater training methods

Evaluate the pretrained checkpoints from Random, Educational-value selection, Avg4, Avg5, and quality-weighted-loss training. Use the same benchmark questions, prompt templates, label translations, tokenizer, context policy, and scoring rule across comparable checkpoints. Compare checkpoints at matched training budgets.

Report:

- Scores for every language.
- A separate macro-average across languages for each benchmark and metric.
- Normal and hard SEA-NLI results separately.
- Mean and sample standard deviation across the three pretraining seeds.
- Improvements over the Random baseline in percentage points.
- Paired confidence intervals for the principal method comparisons where feasible.

The language macro-average is:

$$
\mathrm{LanguageMacro} = \frac{1}{L}\sum_{\ell=1}^{L}M_\ell.
$$

This differs from macro-F1: the former averages languages, while the latter averages classes. Do not merge the four benchmarks into one unqualified accuracy number.

For paired uncertainty estimates, compare methods on the same examples. Account for shared passages and parallel translations when resampling Belebele, and shared premises/concepts where applicable in SEA-NLI. Training-seed variation and test-sample uncertainty describe different sources of variation.

Use this results-table structure once per benchmark, metric, and subset:

| Training method | Indonesian | Malay | Tagalog | Thai | Khmer | Vietnamese | Language macro |
|---|---:|---:|---:|---:|---:|---:|---:|
| Random | | | | | | | |
| Educational value | | | | | | | |
| Avg4 | | | | | | | |
| Avg5 | | | | | | | |
| Quality-weighted loss | | | | | | | |

For XCOPA, mark unsupported language cells as N/A and average only Indonesian, Thai, and Vietnamese. Fill cells with mean ± standard deviation across pretraining seeds.

## 9. Reproducibility and interpretation

Save per-example predictions and candidate scores, along with model checkpoint, training seed, dataset revision, subset, prompt version, translated label mapping, tokenizer revision, scoring rule, and truncation policy.

Keep benchmark test examples out of pretraining and data-weight optimization. Select prompts, hyperparameters, and checkpoints using development data or a previously fixed rule, not final benchmark test scores.

Zero-shot is the main protocol. If few-shot evaluation is added later, put demonstrations in the prompt without changing model weights. Use non-test examples, identical demonstrations across compared models, and check source-passage overlap across benchmarks before constructing demonstrations. Belebele and SIB-200 both use FLORES-derived material.

A small LM can remain near chance on prompted tasks. Report that outcome honestly; these evaluations do not guarantee a usable task capability. Consistent improvements over Random across languages and seeds provide evidence for the data method. An Avg5 improvement on general benchmarks alone does not establish cultural understanding; SEA-NLI provides a more targeted, still limited test.

## 10. Primary sources

- [Belebele dataset and task description](https://huggingface.co/datasets/facebook/belebele)
- [XCOPA dataset and task description](https://huggingface.co/datasets/cambridgeltl/xcopa)
- [SIB-200 dataset, labels, and splits](https://huggingface.co/datasets/Davlan/sib200)
- [SIB-200 paper](https://aclanthology.org/2024.eacl-long.14/)
- [SEA-NLI dataset and field descriptions](https://huggingface.co/datasets/aisingapore/SEA-NLI)
- [SEA-NLI paper: task definition, subsets, and weighted-F1 reporting](https://arxiv.org/html/2606.03284v1)
- [lm-evaluation-harness model guide: conditional continuation likelihood and causal alignment](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/model_guide.md)

Dataset facts come from these sources. The custom prompt examples, combined evaluation design, and recommended reporting procedure are the proposed SEA-Rater protocol discussed in this project.
