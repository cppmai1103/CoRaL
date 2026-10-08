
# Adding SIB-200, SEA-NLI, and FLORES+ to lm-evaluation-harness

This guide explains how to add three additional benchmarks to the evaluation pipeline for:

- Burmese
- Filipino
- Indonesian
- Khmer
- Malay
- Thai
- Vietnamese

The existing Belebele, Global PIQA, Global-MMLU, and INCLUDE tasks can continue using their built-in lm-evaluation-harness implementations.

## 1. Evaluation protocol

Use the official evaluation split for the reported score. Few-shot examples are demonstrations added to the prompt; they are not included in the metric calculation.

| Benchmark | Scored split | Demonstration source | Output type | Main metric |
|---|---|---|---|---|
| SIB-200 | test | train or validation | multiple_choice | acc |
| SEA-NLI normal | test | 0-shot recommended | multiple_choice | normalized accuracy / acc |
| SEA-NLI hard | test | 0-shot recommended | multiple_choice | normalized accuracy / acc |
| FLORES+ | devtest | dev | generate_until | chrF++, spBLEU |

Important:

1. Use the same prompt, demonstrations, order, seed, and decoding settings for every model and data-selection method.
2. Do not use scored test examples as demonstrations in the main experiment.
3. SEA-NLI normal and hard are test-only configurations. Use 0-shot unless you create a separate demonstration set.
4. FLORES+ has public dev and devtest splits; its blind test set is not in the public repository. Use devtest for the final score and dev for prompt decisions or demonstrations.

The harness task configuration guide is available at:

https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/task_guide.md

## 2. Installation

    git clone https://github.com/EleutherAI/lm-evaluation-harness.git
    cd lm-evaluation-harness
    pip install -e .

    pip install datasets transformers sentencepiece sacrebleu

FLORES+ may require accepting its dataset terms and logging in to Hugging Face:

    huggingface-cli login

## 3. Suggested directory structure

Keep custom tasks outside the main harness source code.

    your-evaluation-project/
    ├── custom_tasks/
    │   ├── sib200/
    │   │   ├── sib200_ind.yaml
    │   │   └── utils.py
    │   ├── sea_nli/
    │   │   ├── sea_nli_ind_normal.yaml
    │   │   ├── sea_nli_ind_hard.yaml
    │   │   └── utils.py
    │   └── flores_plus/
    │       └── flores_plus_eng_ind.yaml
    ├── data/
    │   └── flores_plus/
    ├── scripts/
    └── results/

Run custom tasks by adding:

    --include_path ./custom_tasks

## 4. Add SIB-200

SIB-200 is a seven-way topic-classification task. It provides train, validation, and test splits for each language.

Dataset:

https://huggingface.co/datasets/Davlan/sib200

### 4.1 Language configurations

Use these SIB-200 configuration names:

    Burmese      mya_Mymr
    Filipino     tgl_Latn
    Indonesian   ind_Latn
    Khmer        khm_Khmr
    Malay        zsm_Latn
    Thai         tha_Thai
    Vietnamese   vie_Latn

### 4.2 Helper functions

Create custom_tasks/sib200/utils.py:

    TOPICS = [
        "science/technology",
        "travel",
        "politics",
        "sports",
        "health",
        "entertainment",
        "geography",
    ]


    def sib_choices(doc):
        return TOPICS


    def sib_target(doc):
        return int(doc["label"])

Verify the label order before running all languages:

    python - <<'PY'
    from datasets import load_dataset

    ds = load_dataset("Davlan/sib200", "ind_Latn")
    print(ds)
    print(ds["train"].features["label"])
    PY

The names in label.names must match the order in TOPICS.

### 4.3 SIB-200 YAML task

Create custom_tasks/sib200/sib200_ind.yaml:

    task: sib200_ind
    dataset_path: Davlan/sib200
    dataset_name: ind_Latn

    test_split: test
    fewshot_split: train

    output_type: multiple_choice

    doc_to_text: |-
      Classify the topic of the following sentence.

      Sentence:
      {{text}}

      Topic:

    doc_to_choice: !function utils.sib_choices
    doc_to_target: !function utils.sib_target

    target_delimiter: " "
    fewshot_delimiter: "\n\n"

    metric_list:
      - metric: acc
        aggregation: mean
        higher_is_better: true
      - metric: acc_norm
        aggregation: mean
        higher_is_better: true

    metadata:
      version: 1.0

Duplicate this YAML for the other languages and change task and dataset_name. For example:

    task: sib200_vie
    dataset_name: vie_Latn

Use acc as the primary score. Report acc_norm as an additional diagnostic if desired.

### 4.4 SIB-200 smoke test

Run five examples first:

    lm_eval \
      --include_path ./custom_tasks \
      --model hf \
      --model_args pretrained=/path/to/gemma-3-1b-pt,dtype=bfloat16 \
      --tasks sib200_ind \
      --num_fewshot 0 \
      --limit 5 \
      --device cuda:0 \
      --log_samples \
      --output_path results/sib200_ind_smoke

Then run the complete test split:

    lm_eval \
      --include_path ./custom_tasks \
      --model hf \
      --model_args pretrained=/path/to/gemma-3-1b-pt,dtype=bfloat16 \
      --tasks sib200_ind \
      --num_fewshot 0 \
      --device cuda:0 \
      --log_samples \
      --output_path results/sib200_ind_0shot

For a secondary 5-shot result, change:

    --num_fewshot 5

Because fewshot_split is train, the demonstrations come from train and the score is calculated on test.

## 5. Add SEA-NLI

SEA-NLI is a three-way natural-language-inference task with culturally grounded premise-hypothesis pairs.

Dataset:

https://huggingface.co/datasets/aisingapore/SEA-NLI

Native-language prompt templates:

https://github.com/aisingapore/SEA-HELM/blob/main/docs/datasets_and_prompts.md

Important fields:

    premise_native
    hypothesis_native
    true_label
    culture

Possible labels:

    entailment
    contradiction
    neutral

Evaluate normal and hard as separate tasks.

### 5.1 Helper functions

Create custom_tasks/sea_nli/utils.py:

    LABELS = ["entailment", "contradiction", "neutral"]


    def sea_choices(doc):
        return LABELS


    def sea_target(doc):
        label = str(doc["true_label"]).strip().lower()
        if label not in LABELS:
            raise ValueError(f"Unknown SEA-NLI label: {label}")
        return LABELS.index(label)


    def _filter_culture(dataset, culture):
        return dataset.filter(lambda row: row["culture"] == culture)


    def filter_myanmar(dataset):
        return _filter_culture(dataset, "Myanmar")


    def filter_philippines(dataset):
        return _filter_culture(dataset, "Philippines")


    def filter_indonesia(dataset):
        return _filter_culture(dataset, "Indonesia")


    def filter_cambodia(dataset):
        return _filter_culture(dataset, "Cambodia")


    def filter_malaysia(dataset):
        return _filter_culture(dataset, "Malaysia")


    def filter_thailand(dataset):
        return _filter_culture(dataset, "Thailand")


    def filter_vietnam(dataset):
        return _filter_culture(dataset, "Vietnam")

Check the actual culture values before running:

    python - <<'PY'
    from datasets import load_dataset

    ds = load_dataset("aisingapore/SEA-NLI", "normal", split="test")
    print(sorted(set(ds["culture"])))
    print(ds.column_names)
    PY

If the country names differ, update the filtering functions.

### 5.2 SEA-NLI normal YAML

Create custom_tasks/sea_nli/sea_nli_ind_normal.yaml:

    task: sea_nli_ind_normal
    dataset_path: aisingapore/SEA-NLI
    dataset_name: normal

    test_split: test
    process_docs: !function utils.filter_indonesia

    output_type: multiple_choice

    doc_to_text: |-
      Determine the relation between the following two sentences.

      Premise:
      {{premise_native}}

      Hypothesis:
      {{hypothesis_native}}

      Relation:

    doc_to_choice: !function utils.sea_choices
    doc_to_target: !function utils.sea_target

    target_delimiter: " "

    metric_list:
      - metric: acc
        aggregation: mean
        higher_is_better: true
      - metric: acc_norm
        aggregation: mean
        higher_is_better: true

    metadata:
      version: 1.0

For a native-language evaluation, replace the English instruction with the official SEA-HELM prompt for that language. Keep the label order fixed.

### 5.3 SEA-NLI hard YAML

Copy the normal YAML and change:

    task: sea_nli_ind_hard
    dataset_name: hard

Repeat the task for all seven languages by changing process_docs and task.

Use 0-shot for both normal and hard:

    lm_eval \
      --include_path ./custom_tasks \
      --model hf \
      --model_args pretrained=/path/to/gemma-3-1b-pt,dtype=bfloat16 \
      --tasks sea_nli_ind_normal,sea_nli_ind_hard \
      --num_fewshot 0 \
      --device cuda:0 \
      --log_samples \
      --output_path results/sea_nli_ind

Report normal and hard separately. Do not combine them into one primary score.

## 6. Add FLORES+

FLORES+ is a generation-based machine-translation benchmark, not a multiple-choice task.

Dataset:

https://huggingface.co/datasets/openlanguagedata/flores_plus

Official scoring instructions:

https://github.com/facebookresearch/flores/blob/main/flores200/README.md

Use:

    dev      -> optional demonstrations and prompt decisions
    devtest  -> final evaluation

The English and target-language sentences are aligned by id and split. First create paired examples.

### 6.1 Create aligned FLORES+ pairs

Create scripts/make_flores_pairs.py:

    import argparse
    import json
    from pathlib import Path

    from datasets import load_dataset


    def main():
        parser = argparse.ArgumentParser()
        parser.add_argument("--source", default="eng_Latn")
        parser.add_argument("--target", required=True)
        parser.add_argument("--split", default="devtest")
        parser.add_argument("--output", required=True)
        args = parser.parse_args()

        dataset_name = "openlanguagedata/flores_plus"

        source = load_dataset(
            dataset_name,
            args.source,
            split=args.split,
        )
        target = load_dataset(
            dataset_name,
            args.target,
            split=args.split,
        )

        source_by_id = {row["id"]: row["text"] for row in source}

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        count = 0
        with output_path.open("w", encoding="utf-8") as handle:
            for row in target:
                if row["id"] not in source_by_id:
                    continue

                item = {
                    "id": row["id"],
                    "source": source_by_id[row["id"]],
                    "target": row["text"],
                }
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
                count += 1

        print(f"Wrote {count} aligned examples to {output_path}")


    if __name__ == "__main__":
        main()

Example for Vietnamese:

    python scripts/make_flores_pairs.py \
      --target vie_Latn \
      --split devtest \
      --output data/flores_plus/eng_vie_devtest.jsonl

Repeat for:

    mya_Mymr
    tgl_Latn
    ind_Latn
    khm_Khmr
    zsm_Latn
    tha_Thai
    vie_Latn

Use the exact configuration names shown by the current FLORES+ version.

### 6.2 FLORES+ generation task

Create custom_tasks/flores_plus/flores_plus_eng_vie.yaml:

    task: flores_plus_eng_vie
    dataset_path: json

    dataset_kwargs:
      data_files:
        test: data/flores_plus/eng_vie_devtest.jsonl

    test_split: test

    output_type: generate_until

    doc_to_text: |-
      Translate the following sentence from English into Vietnamese.

      English:
      {{source}}

      Vietnamese:

    doc_to_target: "{{target}}"

    generation_kwargs:
      until:
        - "\n\n"
      do_sample: false

    metric_list:
      - metric: chrf
        aggregation: mean
        higher_is_better: true

    metadata:
      version: 1.0

Replace Vietnamese with the target language in each language-specific YAML file.

Run the generation task:

    lm_eval \
      --include_path ./custom_tasks \
      --model hf \
      --model_args pretrained=/path/to/gemma-3-1b-pt,dtype=bfloat16 \
      --tasks flores_plus_eng_vie \
      --num_fewshot 0 \
      --gen_kwargs max_gen_toks=128,do_sample=False \
      --device cuda:0 \
      --log_samples \
      --output_path results/flores_plus_vie

The harness chrF result is useful for a smoke test. For the final paper score, calculate official chrF++ and spBLEU using the saved hypotheses, references, and official FLORES tokenization.

### 6.3 FLORES+ scoring

Save one hypothesis and one reference per line:

    results/flores_plus_vie/hypotheses.txt
    results/flores_plus_vie/references.txt

Calculate chrF++ with SacreBLEU:

    sacrebleu \
      -m chrf \
      --chrf-word-order 2 \
      results/flores_plus_vie/references.txt \
      < results/flores_plus_vie/hypotheses.txt

For spBLEU, tokenize both hypotheses and references with the official FLORES SentencePiece model, then calculate BLEU with SacreBLEU. Follow the official FLORES evaluation README.

Do not use exact match as the main FLORES+ metric because multiple translations can be valid.

## 7. Running all languages

Run language-specific tasks separately so that outputs are easy to audit:

    lm_eval \
      --include_path ./custom_tasks \
      --model hf \
      --model_args pretrained=/path/to/gemma-3-1b-pt,dtype=bfloat16 \
      --tasks sib200_mya,sib200_fil,sib200_ind,sib200_khm,sib200_msa,sib200_tha,sib200_vie \
      --num_fewshot 0 \
      --device cuda:0 \
      --output_path results/sib200_0shot

Use the exact task names created in your YAML files. For SEA-NLI, save normal and hard results separately.

## 8. Smoke-test checklist

Before running the complete evaluation, check that:

- the intended language configuration is loaded;
- the correct test or devtest split is used for scoring;
- no scored test labels appear in demonstrations;
- the correct answer index matches the dataset label;
- prompts are identical across random, selected, and full-data models;
- do_sample is false for translation;
- model checkpoint, harness commit, dataset revision, prompt version, seed, and metrics are recorded;
- the number of evaluated examples is reported for every language.

A useful smoke-test command is:

    lm_eval \
      --include_path ./custom_tasks \
      --model hf \
      --model_args pretrained=/path/to/gemma-3-1b-pt,dtype=bfloat16 \
      --tasks sib200_ind \
      --num_fewshot 0 \
      --limit 5 \
      --device cuda:0 \
      --log_samples \
      --output_path results/smoke_test

## 9. Recommended paper reporting

Use separate tables:

1. SIB-200 accuracy by language and macro-average.
2. SEA-NLI normal accuracy by language and macro-average.
3. SEA-NLI hard accuracy by language and macro-average.
4. FLORES+ chrF++ and spBLEU by language direction.

Do not average acc, chrF++, and spBLEU into a single overall score because they measure different abilities.

## 10. References

- lm-evaluation-harness task guide: https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/task_guide.md
- SIB-200: https://huggingface.co/datasets/Davlan/sib200
- SEA-NLI: https://huggingface.co/datasets/aisingapore/SEA-NLI
- SEA-HELM prompts: https://github.com/aisingapore/SEA-HELM/blob/main/docs/datasets_and_prompts.md
- FLORES+: https://huggingface.co/datasets/openlanguagedata/flores_plus
- FLORES evaluation: https://github.com/facebookresearch/flores/blob/main/flores200/README.md

