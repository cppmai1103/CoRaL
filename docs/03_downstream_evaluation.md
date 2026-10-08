# Evaluation Benchmark Plan Using `lm-evaluation-harness`

## Evaluation objective

The project evaluates multilingual continual-pretraining and data-selection methods for seven Southeast Asian languages: Burmese, Filipino, Indonesian, Khmer, Malay, Thai, and Vietnamese.

The benchmark suite is selected using three criteria:

1. Language coverage across the seven target languages.
2. Simple, reproducible tasks that are suitable for base language models.
3. Coverage of different abilities, including comprehension, knowledge, and reasoning.

## Selected five-benchmark suite

The primary evaluation uses the following five benchmarks:

| Benchmark | Main ability | Available project languages | Primary harness task(s) | Main metric | Final evaluation set |
| --- | --- | --- | --- | --- | --- |
| Belebele | Reading comprehension | Burmese, Filipino, Indonesian, Khmer, Malay, Thai, Vietnamese | `belebele_{lang}` | `acc`, `acc_norm` | Full official test set |
| Global PIQA | Cultural and physical commonsense reasoning | Filipino, Indonesian, Malay, Thai, Vietnamese | `global_piqa_{split}_cloze_{lang}` | `acc`, `acc_norm`, `acc_bytes` | Official test split |
| Global-MMLU Full | Broad academic knowledge and reasoning | Filipino, Indonesian, Malay, Vietnamese | `global_mmlu_full_{lang}` | `acc` | Full test split |
| INCLUDE | Regional knowledge and academic reasoning | Filipino, Indonesian, Malay, Vietnamese | `include_base_44_{lang}` | `acc` | Full INCLUDE-base test set |
| XCOPA | Causal commonsense reasoning | Indonesian, Thai, Vietnamese | `xcopa_{lang}` | `acc` | Full test split |

The five benchmarks should be evaluated using their complete official evaluation sets. The number of examples should not be artificially equalized across datasets.

## Evaluation protocol with k-shot prompting

`lm-evaluation-harness` supports k-shot evaluation through:

```bash
lm-eval run \
  --model hf \
  --model_args pretrained=<checkpoint> \
  --tasks <task_names> \
  --num_fewshot 5 \
  --batch_size auto \
  --log_samples
```

Here, `--num_fewshot 5` adds five labeled demonstrations to every test prompt. It does not mean that only five test examples are evaluated. The model is scored on every example in the official test split unless `--limit` or `--samples` is explicitly used.

The evaluation procedure is:

1. Train each model size and data-selection method under the same token budget.
2. Evaluate every checkpoint on the same benchmark tasks and language configurations.
3. Add the same number of demonstrations to each test prompt for a k-shot run.
4. Evaluate the complete official test split.
5. Save model inputs and outputs with `--log_samples` for verification.
6. Report results separately for every language and benchmark.

The recommended main experiment is 0-shot for the cleanest comparison of data-selection methods. A 5-shot experiment can be reported as a secondary robustness analysis.

### Demonstration sources

Demonstrations must come from a separate training, development, or validation pool whenever possible. The final test examples must only be used for scoring.

| Benchmark | Recommended demonstration source |
| --- | --- |
| Belebele | A separate English task-training pool, following the original benchmark protocol; otherwise use 0-shot or a custom demonstration split |
| Global PIQA | A separate fixed demonstration pool; the current task configuration does not provide a clean train/dev split, so 0-shot is safer |
| INCLUDE | A separate fixed demonstration pool; the current task configuration is test-only, so 0-shot is safer unless a custom split is created |
| Global-MMLU Full | `dev` split |
| XCOPA | `validation` split |

## Metrics and aggregation

For each model, method, benchmark, and language, report the official harness metric. The main results should include:

- Per-language scores.
- Macro-average across available languages within each benchmark.
- An overall macro-average across the five selected benchmarks.
- The number of evaluated examples for every benchmark-language pair.
- The difference from the random-selection baseline when comparing data-selection methods.

Do not pool all examples from all benchmarks into one accuracy score, because larger datasets would dominate the result. Use the same test examples for every model and method so that differences are directly comparable.

For languages that are not supported by a benchmark, report `N/A` and clearly state the coverage. Do not replace missing languages with translated or unofficial versions without defining a separate experiment.

## Reproducibility checklist

- Pin the `lm-evaluation-harness` version or commit.
- Record the exact task names and task configurations.
- Record the benchmark version and evaluation split.
- Record the model checkpoint and tokenizer.
- Record `num_fewshot`, prompt format, and random seeds.
- Do not use `--limit` or an arbitrary subset for the final results.
- Keep the final test set separate from method, checkpoint, and hyperparameter selection.

## Section 1: Comprehensive benchmark inventory

This section is intentionally kept at the end so that additional benchmarks can be added later without changing the main evaluation protocol. The first table records benchmark availability and harness details; the second table records the main ability measured by each benchmark.

### 1.1 Benchmark availability and harness information

| Dataset          | Harness task(s)                                 | Metric(s)                                                        | Burmese | Filipino | Indonesian | Khmer | Malay | Thai | Vietnamese | # Languages |
| ---------------- | ----------------------------------------------- | ---------------------------------------------------------------- | ------- | -------- | ---------- | ----- | ----- | ---- | ---------- | ----------- |
| Belebele         | `belebele_{lang}`                               | `acc`, `acc_norm`                                                | ✓       | ✓        | ✓          | ✓     | ✓     | ✓    | ✓          | **7**       |
| Global PIQA      | `global_piqa_{split}_{format}_{lang}`           | Cloze: `acc`, `acc_norm`, `acc_bytes`; Generation: `exact_match` |         | ✓        | ✓          |       | ✓     | ✓    | ✓          | **5**       |
| Global-MMLU      | `global_mmlu_{lang}`; `global_mmlu_full_{lang}` | `acc`                                                            |         | ✓        | ✓          |       | ✓     |      | ✓          | **4**       |
| INCLUDE          | `include_base_44_{lang}`; few-shot variants     | `acc`                                                            |         | ✓        | ✓          |       | ✓     |      | ✓          | **4**       |
| MMLU-ProX        | `mmlu_prox_{lang}`; `mmlu_prox_lite_{lang}`     | `exact_match`                                                    |         |          | ✓          |       |       | ✓    | ✓          | **3**       |
| XCOPA            | `xcopa_id`, `xcopa_th`, `xcopa_vi`              | `acc`                                                            |         |          | ✓          |       |       | ✓    | ✓          | **3**       |
| Okapi ARC        | `arc_id`, `arc_vi`                              | `acc`, `acc_norm`                                                |         |          | ✓          |       |       |      | ✓          | **2**       |
| Okapi HellaSwag  | `hellaswag_id`, `hellaswag_vi`                  | `acc`, `acc_norm`                                                |         |          | ✓          |       |       |      | ✓          | **2**       |
| Okapi MMLU       | `m_mmlu_id`, `m_mmlu_vi`                        | `acc`                                                            |         |          | ✓          |       |       |      | ✓          | **2**       |
| Okapi TruthfulQA | `truthfulqa_{id,vi}_{mc1,mc2}`                  | `acc`                                                            |         |          | ✓          |       |       |      | ✓          | **2**       |
| XNLI             | `xnli_th`, `xnli_vi`                            | `acc`                                                            |         |          |            |       |       | ✓    | ✓          | **2**       |
| XQuAD            | `xquad_th`, `xquad_vi`                          | `exact_match`, `F1`                                              |         |          |            |       |       | ✓    | ✓          | **2**       |
| XStoryCloze      | `xstorycloze_my`, `xstorycloze_id`              | `acc`                                                            | ✓       |          | ✓          |       |       |      |            | **2**       |
| COPAL-ID         | `copal_id_standard`, `copal_id_colloquial`      | `acc`                                                            |         |          | ✓          |       |       |      |            | **1**       |
| MGSM             | `mgsm_direct_th`; native/en COT variants        | `exact_match`                                                    |         |          |            |       |       | ✓    |            | **1**       |
| MLQA             | `mlqa_{context-lang}_{question-lang}`           | `exact_match`, `F1`                                              |         |          |            |       |       |      | ✓          | **1**       |
| MMMLU (OpenAI)   | `mmmlu_id_id` and subject tasks                 | `acc`, `acc_norm`                                                |         |          | ✓          |       |       |      |            | **1**       |
| TyDiQA           | `tydiqa_goldp_id`                               | `exact_match`, `F1`                                              |         |          | ✓          |       |       |      |            | **1**       |

### 1.2 Main model ability tested

| Dataset          | Main model ability                        | What it mainly tests                                                            |
| ---------------- | ----------------------------------------- | ------------------------------------------------------------------------------- |
| Belebele         | Reading comprehension                     | Understanding a passage and answering questions based on it                     |
| Global PIQA      | Cultural commonsense reasoning            | Choosing plausible solutions using everyday and local cultural knowledge        |
| Global-MMLU      | Broad academic knowledge and reasoning    | Knowledge across many academic subjects, including cultural sensitivity         |
| INCLUDE          | Regional knowledge and academic reasoning | Academic/professional exam knowledge and local regional understanding           |
| MMLU-ProX        | Advanced academic reasoning               | Multi-step reasoning across 14 academic subjects and languages                  |
| XCOPA            | Causal commonsense reasoning              | Identifying plausible causes and effects                                        |
| Okapi ARC        | Scientific knowledge and reasoning        | Answering elementary science questions                                          |
| Okapi HellaSwag  | Situational commonsense reasoning         | Selecting the most plausible continuation of a situation                        |
| Okapi MMLU       | Broad academic knowledge                  | Multiple-choice knowledge and reasoning across academic subjects                |
| Okapi TruthfulQA | Truthfulness and factuality               | Avoiding false, misleading, or common-misconception answers                     |
| XNLI             | Natural-language inference                | Determining entailment, contradiction, or neutrality between sentences          |
| XQuAD            | Extractive reading comprehension          | Finding the correct answer span in a passage                                    |
| XStoryCloze      | Narrative understanding                   | Predicting the most coherent ending of a short story                            |
| COPAL-ID         | Causal and cultural commonsense reasoning | Causal reasoning involving Indonesian local language and culture                |
| MGSM             | Mathematical reasoning                    | Solving multilingual grade-school math problems, often requiring multiple steps |
| MLQA             | Cross-lingual question answering          | Understanding passages and extracting answers across languages                  |
| MMMLU            | Broad academic knowledge and reasoning    | 57 multiple-choice academic subjects in multiple languages                      |
| TyDiQA           | Information-seeking question answering    | Finding answer spans in passages for factual questions                          |

