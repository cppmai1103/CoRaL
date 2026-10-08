"""Package and upload SEA-Rater data and experiment artifacts to private Hub repos.

Dataset repo (--dataset-repo): data/annotation_batches + data/rater_dataset (always), plus
data/pilot_corpus, data/pilot_scores, data/pilot_selected if present (skipped with a printed note
otherwise, never a hard error).
Model repo (--model-repo): checkpoints/rater/frozen (always, validated), plus
checkpoints/rater/finetuned and checkpoints/gpt2_top_doc if present (same skip-if-missing rule).

This is a sizeable upload once everything is included (pilot_selected alone is several GB, the
fine-tuned raters another ~12GB) -- expect a long `--dry-run` packaging step and a slow real upload;
there's nothing to tune for that, it's just how much data there is.

Run from the project root:
    python -m src.upload_to_huggingface --dry-run
    python -m src.upload_to_huggingface

Authenticate with `hf auth login` or HF_TOKEN. A dry run is entirely offline.
All staging files stay under .scratch/hf_upload; source files are not modified.
"""

import argparse
import csv
import hashlib
import json
import re
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIMENSIONS = ["educational_value", "reasoning", "professionalism", "cleanliness", "cultural_nuances"]
DATA_SUFFIXES = {".csv", ".json", ".jsonl", ".parquet", ".xlsx", ".xls", ".png", ".md", ".txt"}
MODEL_SUFFIXES = {".pt", ".safetensors", ".bin", ".json", ".csv", ".md", ".png", ".txt", ".model", ".jinja"}
SOURCE_FILES = [
    "__init__.py", "prepare_dataset.py", "train_rater/__init__.py", "train_rater/train.py",
    "train_rater/test.py", "train_rater/plot.py", "train_rater/build_embeddings.py",
    "train_rater/finetune.py", "train_gpt2_from_scratch/__init__.py", "train_gpt2_from_scratch/train_gpt2.py",
    "train_gpt2_from_scratch/prepare_data.py", "train_gpt2_from_scratch/score_pool.py",
    "train_gpt2_from_scratch/combine_scores.py", "train_gpt2_from_scratch/extract_corpus.py",
    "train_gpt2_from_scratch/split_corpus.py",
]
# Pilot/checkpoint directories that may not exist in every local checkout or test fixture --
# copied if present, skipped with a printed note otherwise (never a hard error).
OPTIONAL_DATA_DIRS = [("pilot_corpus", "pilot_corpus_dir"), ("pilot_scores", "pilot_scores_dir"),
                      ("pilot_selected", "pilot_selected_dir"), ("llm_score", "llm_score_dir"),
                      ("rater_dataset_llm", "rater_dataset_llm_dir"), ("pilot_scores_llm", "pilot_scores_llm_dir")]
OPTIONAL_MODEL_DIRS = [("rater/finetuned", "finetuned_dir"), ("gpt2_top_doc", "gpt2_dir"),
                       ("rater_llm", "rater_llm_dir"), ("gpt2_weighted_loss", "weighted_dir"),
                       ("gpt2_top_doc_sea_bpe_16k", "gpt2_16k_dir"), ("tokenizers", "tokenizers_dir"),
                       ("lora_cpt", "lora_cpt_dir"), ("hub_models", "hub_models_dir")]
# One line per optional folder for the generated cards (folders not present are not mentioned).
EXTRA_DATA_DESCRIPTIONS = {
    "llm_score": "LLM (Gemini) scores of ~5,000 documents per language on the 5 dimensions, with justifications.",
    "rater_dataset_llm": "the rater dataset relabelled with LLM scores: same human-annotated documents and "
                         "per-dimension splits, train/validation labels = LLM score, test = human mean.",
    "pilot_scores_llm": "scores of the GPT-2 candidate pool from the LLM-label raters (per dimension, avg4, avg5); "
                        "the selections they produce are pilot_selected/llm_{edu,avg4,avg5}_20M.",
}
EXTRA_MODEL_DESCRIPTIONS = {
    "rater_llm": "mmBERT raters fine-tuned on LLM-score labels (same documents/splits as the human-label raters), "
                 "with rater_comparison.md: both kinds of rater on the human test split.",
    "gpt2_weighted_loss": "GPT-2 runs trained on the random 20M documents with a rater-score-weighted loss "
                          "(edu, avg4, avg5) instead of document selection.",
    "gpt2_top_doc_sea_bpe_16k": "the 20M GPT-2 runs (random, edu, avg4, avg5) with the custom 16K SEA tokenizer "
                                "(tokenizer ablation; compare with the SeaLLM runs by bits per byte only).",
    "tokenizers": "the custom byte-level BPE tokenizer(s) trained on the pilot corpus train split, with reports.",
    "lora_cpt": "google/gemma-3-270m continued-pretrained with LoRA on the 20M selections (adapter/ and the merged "
                "model final/), plus the untrained base reference; Gemma license applies to these weights.",
    "hub_models": "evaluation results of pretrained Hub models (e.g. gemma-3-270m) on the same benchmarks.",
}


def repo_id(value):
    if not re.fullmatch(r"[\w-]+/[\w.-]+", value, flags=re.ASCII):
        raise argparse.ArgumentTypeError("Use a repository ID such as cppmai/sea-rater, not a URL.")
    if any(".." in p or "--" in p or p.startswith((".", "-")) or p.endswith((".", "-", ".git")) for p in value.split("/")):
        raise argparse.ArgumentTypeError("Invalid Hugging Face repository ID.")
    return value


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-repo", type=repo_id, default="cppmai/sea-rater-data-7languages")
    parser.add_argument("--model-repo", type=repo_id, default="cppmai/sea-rater-models-7languages")
    parser.add_argument("--only", choices=["both", "dataset", "model"], default="both")
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "data/annotation_batches")
    parser.add_argument("--prepared-dir", type=Path, default=ROOT / "data/rater_dataset")
    parser.add_argument("--pilot-corpus-dir", type=Path, default=ROOT / "data/pilot_corpus",
                        help="GPT-2 pilot candidate corpus; skipped with a note if missing")
    parser.add_argument("--pilot-scores-dir", type=Path, default=ROOT / "data/pilot_scores",
                        help="Rater scores over the pilot corpus; skipped with a note if missing")
    parser.add_argument("--pilot-selected-dir", type=Path, default=ROOT / "data/pilot_selected",
                        help="Selected GPT-2 training sets per method/budget; skipped with a note if missing")
    parser.add_argument("--checkpoint-dir", type=Path, default=ROOT / "checkpoints/rater/frozen",
                        help="All experiment runs, or one run such as checkpoints/rater/frozen/mlp_mean")
    parser.add_argument("--finetuned-dir", type=Path, default=ROOT / "checkpoints/rater/finetuned",
                        help="Fully fine-tuned raters; skipped with a note if missing")
    parser.add_argument("--gpt2-dir", type=Path, default=ROOT / "checkpoints/gpt2_top_doc",
                        help="Trained GPT-2 pilot checkpoints; skipped with a note if missing")
    parser.add_argument("--llm-score-dir", type=Path, default=ROOT / "data/llm_score", help="LLM scores; skipped with a note if missing")
    parser.add_argument("--rater-dataset-llm-dir", type=Path, default=ROOT / "data/rater_dataset_llm", help="LLM-label rater dataset; skipped with a note if missing")
    parser.add_argument("--pilot-scores-llm-dir", type=Path, default=ROOT / "data/pilot_scores_llm", help="Pool scores from the LLM-label raters; skipped with a note if missing")
    parser.add_argument("--rater-llm-dir", type=Path, default=ROOT / "checkpoints/rater_llm", help="LLM-label raters; skipped with a note if missing")
    parser.add_argument("--weighted-dir", type=Path, default=ROOT / "checkpoints/gpt2_weighted_loss", help="Weighted-loss GPT-2 runs; skipped with a note if missing")
    parser.add_argument("--gpt2-16k-dir", type=Path, default=ROOT / "checkpoints/gpt2_top_doc_sea_bpe_16k", help="16K-tokenizer GPT-2 runs; skipped with a note if missing")
    parser.add_argument("--tokenizers-dir", type=Path, default=ROOT / "checkpoints/tokenizers", help="Custom tokenizers; skipped with a note if missing")
    parser.add_argument("--lora-cpt-dir", type=Path, default=ROOT / "checkpoints/lora_cpt", help="Gemma LoRA continued-pretraining runs; skipped with a note if missing")
    parser.add_argument("--hub-models-dir", type=Path, default=ROOT / "checkpoints/hub_models", help="Hub-model evaluation results; skipped with a note if missing")
    parser.add_argument("--staging-dir", type=Path, default=ROOT / ".scratch/hf_upload")
    parser.add_argument("--include-embeddings", action="store_true",
                        help="Also upload data/rater_dataset/embeddings*/embeddings.pt to the dataset repo")
    parser.add_argument("--dry-run", action="store_true", help="Prepare local bundles and manifests without contacting the Hub")
    return parser.parse_args(argv)


def copy_file(source, destination):
    if source.is_symlink():
        raise ValueError(f"Refusing to package a symbolic link: {source}")
    if not source.is_file():
        raise ValueError(f"Required file is missing: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def copy_artifacts(source, destination, suffixes):
    if not source.is_dir():
        raise ValueError(f"Directory is missing: {source}")
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        if any(part.startswith(".") or part == "__pycache__" for part in relative.parts):
            continue
        if path.is_file() and path.suffix.lower() in suffixes:
            copy_file(path, destination / relative)


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def validate_prepared(directory):
    documents = read_csv(directory / "document_table.csv")
    keys = {(r["language"], r["doc_id"]): r for r in documents}
    if not keys or len(keys) != len(documents):
        raise ValueError("document_table.csv is empty or has duplicate (language, document ID) keys.")
    counts = {}
    for dim in DIMENSIONS:
        rows = read_csv(directory / f"split_manifest_{dim}.csv")
        seen = set()
        counts[dim] = {split: 0 for split in ["train", "validation", "test"]}
        for r in rows:
            key = (r["language"], r["doc_id"])
            if key in seen or key not in keys or r["split"] not in counts[dim]:
                raise ValueError(f"Invalid or duplicate split entry for {dim}: {key}")
            score = float(r["score"])
            if not 0 <= score <= 5 or score != float(keys[key][dim]) or r["split"] != keys[key][f"split_{dim}"]:
                raise ValueError(f"Split/score disagrees with document table for {dim}: {key}")
            seen.add(key)
            counts[dim][r["split"]] += 1
        if not all(counts[dim].values()):
            raise ValueError(f"Missing train, validation or test examples for {dim}.")
    return len(documents), sorted({r["language"] for r in documents}), counts


def prepare_dataset(args, destination):
    if not list(args.dataset_dir.glob("*_clean.csv")):
        raise ValueError(f"No language clean CSV files in {args.dataset_dir}")
    copy_artifacts(args.dataset_dir, destination, DATA_SUFFIXES)
    prepared = destination / "rater_dataset"
    for name in ["document_table.csv", "excluded_documents.csv", "data_quality_report.md"] + [f"split_manifest_{d}.csv" for d in DIMENSIONS]:
        copy_file(args.prepared_dir / name, prepared / name)
    for path in sorted(args.prepared_dir.glob("embeddings*/manifest.json")):
        copy_file(path, prepared / path.relative_to(args.prepared_dir))
        if args.include_embeddings:
            copy_file(path.with_name("embeddings.pt"), prepared / path.relative_to(args.prepared_dir).with_name("embeddings.pt"))
    n, languages, counts = validate_prepared(prepared)
    # LLM-vs-human comparison and the GPT-2 evaluations on the annotated documents, if present
    for path in sorted(args.prepared_dir.glob("llm_scores_annotated.*")):
        copy_file(path, prepared / path.name)
    for path in sorted(p for p in args.prepared_dir.glob("annotated_eval*") if p.is_dir()):
        copy_artifacts(path, prepared / path.name, DATA_SUFFIXES)

    pilot_included = []
    for name, attr in OPTIONAL_DATA_DIRS:
        source = getattr(args, attr)
        if source.is_dir():
            copy_artifacts(source, destination / name, DATA_SUFFIXES)
            pilot_included.append(name)
        else:
            print(f"Note: {source} not found, skipping {name} in this export", flush=True)
    card = ["---", "tags:", "- text-quality", "- multilingual", "configs:",
            "- config_name: documents", "  data_files:", "  - split: documents",
            "    path: rater_dataset/document_table.csv", "---", "", "# SEA-Rater annotation dataset", "",
            f"Snapshot of {n:,} documents. Language identifiers: {', '.join(languages)}.", "",
            "Human-average targets retain their original fractional values on the 0–5 scale. Original annotation CSVs, spreadsheets, statistics and figures are preserved under the language directories.", "",
            "`rater_dataset/document_table.csv` includes text, scores and a separate split column for every dimension. Join manifests to documents using **(language, doc_id)**.", "",
            "## Splits", "", "The requested split proportions are 70/10/20 per dimension. The saved manifests are authoritative; counts reflect stratification and duplicate grouping. There is no single shared train/test split across dimensions.", "",
            "| Dimension | Train | Validation | Test |", "|---|---:|---:|---:|"]
    for dim, c in counts.items():
        card.append(f"| {dim} | {c['train']} | {c['validation']} | {c['test']} |")
    card += ["", "## Download", "", "```python", "from huggingface_hub import snapshot_download",
             f'snapshot_download(repo_id="{args.dataset_repo}", repo_type="dataset", local_dir="data/annotation_batches")',
             "```", "", "The `documents` viewer split is the complete document table, not a training split. Use each dimension's manifest for training and evaluation. Cached embeddings are optional; their manifests record encoder and pooling settings.", ""]
    if pilot_included:
        card += ["## GPT-2 pilot artifacts", "",
                 "The pretraining-data-selection pilot (guide 02) also ships here, built to exclude every "
                 "document used in rater annotation above (no train/eval leakage):", ""]
        if "pilot_corpus" in pilot_included:
            card.append("- `pilot_corpus/` — the GPT-2 candidate corpus per language, with held-out validation/test "
                        "documents (`split_manifest.csv`).")
        if "pilot_scores" in pilot_included:
            card.append("- `pilot_scores/` — rater-predicted scores over the candidate pool, per dimension and "
                        "combined (`avg4_mean`, `avg5_mean`), with distribution plots and thresholds.")
        if "pilot_selected" in pilot_included:
            card.append("- `pilot_selected/` — the actual GPT-2 training sets, one folder per selection "
                        "method x token budget (`documents.csv`, `summary.json`/`summary.md`).")
        card.append("")
    extra = [name for name in EXTRA_DATA_DESCRIPTIONS if name in pilot_included]
    if extra:
        card += ["## LLM-label experiment", ""] + [f"- `{name}/` — {EXTRA_DATA_DESCRIPTIONS[name]}" for name in extra]
        card += ["- `rater_dataset/llm_scores_annotated.*` and `rater_dataset/annotated_eval*/` — LLM vs human scores "
                 "on the annotated documents, and GPT-2 runs evaluated on them.", ""]
    card += [f"Model checkpoints and results: https://huggingface.co/{args.model_repo}", "",
             "## License", "", "This export does not assign a new license to the underlying source documents or annotations."]
    # Preserve a supplied dataset card when replacing it with an export card.
    if (destination / "README.md").exists():
        copy_file(destination / "README.md", destination / "original_dataset_card.md")
    (destination / "README.md").write_text("\n".join(card) + "\n", encoding="utf-8")


def prepare_model(args, destination):
    copy_artifacts(args.checkpoint_dir, destination / "rater/frozen", MODEL_SUFFIXES)
    checkpoints = sorted((destination / "rater/frozen").rglob("checkpoint.pt"))
    if not checkpoints:
        raise ValueError(f"No checkpoint.pt files in {args.checkpoint_dir}")
    for p in checkpoints:
        for name in ["config.json", "predictions_validation.csv", "predictions_test.csv", "evaluation_report.md"]:
            if not (p.parent / name).is_file():
                raise ValueError(f"Incomplete checkpoint/results directory: {p.parent} (missing {name})")
        cfg = json.loads((p.parent / "config.json").read_text())
        if cfg.get("dimension") not in DIMENSIONS:
            raise ValueError(f"Unexpected checkpoint dimension in {p.parent}")
    manifests = sorted(args.prepared_dir.glob("embeddings*/manifest.json"))
    if not manifests:
        print(f"Note: no embeddings*/manifest.json under {args.prepared_dir} (cache deleted/not built), "
              f"skipping embedding_manifests/ -- rebuild with sh/rater/rebuild_embeddings.sh if you want encoder "
              f"provenance in this export", flush=True)
    for p in manifests:
        copy_file(p, destination / "embedding_manifests" / f"{p.parent.name}.json")

    other_included = []
    for name, attr in OPTIONAL_MODEL_DIRS:
        source = getattr(args, attr)
        if not source.is_dir():
            print(f"Note: {source} not found, skipping {name} in this export", flush=True)
            continue
        copy_artifacts(source, destination / name, MODEL_SUFFIXES)
        if not any((destination / name).rglob("*")):
            raise ValueError(f"{source} exists but nothing matched MODEL_SUFFIXES -- check its contents")
        other_included.append(name)

    for name in SOURCE_FILES:
        copy_file(ROOT / "src" / name, destination / "src" / name)
    copy_file(ROOT / "docs/01_train_rater.md", destination / "training_guide.md")
    if (ROOT / "docs/02_pilot_gpt2_training.md").is_file():
        copy_file(ROOT / "docs/02_pilot_gpt2_training.md", destination / "pilot_gpt2_training_guide.md")
    (destination / "requirements.txt").write_text("torch\ntransformers\nmatplotlib\nhuggingface_hub\n")
    preferred = next((p for p in checkpoints if "mlp_mean" in p.parts), checkpoints[0])
    relative = preferred.parent.relative_to(destination).as_posix()
    card = ["---", "tags:", "- pytorch", "- text-quality", "- regression", "- multilingual",
            "datasets:", f"- {args.dataset_repo}", "---", "", "# SEA-Rater checkpoints and evaluation results", "",
            "Three kinds of trained model live in this repo: frozen-encoder rater heads (`<pooling>/<dimension>/`, "
            "this section), fully fine-tuned raters (`rater/finetuned/`), and GPT-2 pilot checkpoints "
            "(`gpt2_top_doc/`) -- see their own sections below if present in this export.", "",
            "## Frozen-encoder rater heads", "",
            "Custom PyTorch MLP regression heads over frozen multilingual document embeddings. Each dimension is trained independently with MSE and checkpoint selection by validation MAE averaged equally over languages. Outputs are clipped to [0,5] for reporting.", "",
            "These checkpoint files contain the trained heads, not the frozen encoder weights, and are not a Transformers `AutoModel.from_pretrained` package. Obtain the encoder and tokenizer from the model ID recorded in the matching embedding manifest.", "",
            "`mlp_mean` uses content-token mean pooling; `mlp_cls` uses first-token pooling. CLS refers to pooling, not a classification loss. Chunk embeddings are combined using content-token-count weights.", "",
            "### Included checkpoints", ""]
    card += [f"- `{p.relative_to(destination).as_posix()}`" for p in checkpoints]
    card += ["", "Each checkpoint directory includes configuration, predictions, training logs, evaluation reports and available PNG plots. Combined run plots are retained. `embedding_manifests/` records encoder, chunk size and pooling; `src/` contains the model and embedding code.", "",
             "### Reload a head", "", "After installing requirements and authenticating for the private repository:", "", "```python",
             "import json", "import sys", "from pathlib import Path", "import torch",
             "from huggingface_hub import snapshot_download", "",
             f'root = Path(snapshot_download(repo_id="{args.model_repo}"))',
             "sys.path.insert(0, str(root))", "from src.train_rater.train import build_head", "",
             f'folder = root / "{relative}"',
             'cfg = json.loads((folder / "config.json").read_text())',
             'saved = torch.load(folder / "checkpoint.pt", map_location="cpu", weights_only=True)',
             'head = build_head(saved["embedding_dim"], cfg["hidden_size"], cfg["dropout"])',
             'head.load_state_dict(saved["state_dict"])', "head.eval()", "```", "",
             "Generate document embeddings with the archived `src/train_rater/build_embeddings.py` and the matching manifest's encoder, revision, chunk size and pooling. Mean pooling uses `--pooling mean`; CLS pooling uses `--pooling cls`. Apply the head to float32 embeddings and clip its scalar output to [0,5].", ""]
    if "rater/finetuned" in other_included:
        card += ["## Fully fine-tuned raters (`rater/finetuned/`)", "",
                 "Unlike the frozen-encoder heads above, these fully fine-tune mmBERT-base end to end "
                 "(`AutoModelForSequenceClassification`, `num_labels=1`, `problem_type=\"regression\"`) -- one "
                 "model per dimension, under `rater/finetuned/<pooling>/<dimension>/model/`. This is the rater "
                 "actually used to score the GPT-2 pilot corpus below.", "",
                 "```python", "from pathlib import Path", "from transformers import AutoModelForSequenceClassification, AutoTokenizer",
                 "from huggingface_hub import snapshot_download", "",
                 f'root = Path(snapshot_download(repo_id="{args.model_repo}"))',
                 'folder = root / "rater/finetuned/mean/educational_value/model"',
                 "tokenizer = AutoTokenizer.from_pretrained(folder)",
                 "model = AutoModelForSequenceClassification.from_pretrained(folder).eval()", "```", ""]
    if "gpt2_top_doc" in other_included:
        card += ["## GPT-2 pilot checkpoints (`gpt2_top_doc/`)", "",
                 "Small GPT-2 language models (6 layers, 512-dim, ~44M params) trained from scratch on the pilot "
                 "corpus, comparing document-selection methods (`random`, `edu`, `avg4`, `avg5`) at matched "
                 "per-language token budgets (5M/10M/20M). Each run folder has `final/` (the trained weights), "
                 "`tokenizer/`, `results.json`/`summary.md` (validation/test loss and perplexity, per language "
                 "and macro-averaged).", "",
                 "```python", "from pathlib import Path", "from transformers import AutoModelForCausalLM, AutoTokenizer",
                 "from huggingface_hub import snapshot_download", "",
                 f'root = Path(snapshot_download(repo_id="{args.model_repo}"))',
                 'folder = root / "gpt2_top_doc/random_5M_ep1_seed42"',
                 'tokenizer = AutoTokenizer.from_pretrained(folder / "tokenizer")',
                 'model = AutoModelForCausalLM.from_pretrained(folder / "final").eval()', "```", ""]
    extra = [name for name in EXTRA_MODEL_DESCRIPTIONS if name in other_included]
    if extra:
        card += ["## Further experiments", ""] + [f"- `{name}/` — {EXTRA_MODEL_DESCRIPTIONS[name]}" for name in extra] + [""]
    card += ["## Evaluation and limitations", "",
             "Predictions and reports describe held-out human annotations. Rare score ranges have small test samples and can have substantially higher errors than overall averages. Inspect language/range results before selection. No downstream language-model benefit is established by rater agreement alone.", "",
             "In the current evaluation_report.md summary tables, the legacy column labelled 'Baseline MAE (mean/median)' contains training baseline score values. Actual median-baseline errors are in the by-range table/CSV. Do not interpret that legacy column as baseline errors.", "",
             "Original cache manifests may contain revision=null; this export preserves that provenance and does not invent a pinned encoder version. Source code is snapshotted at export time. The underlying encoder must be downloaded separately.", "",
             "## License", "", "This export does not assign a new license to the trained weights. The encoder retains its upstream license."]
    (destination / "README.md").write_text("\n".join(card) + "\n", encoding="utf-8")


def write_manifest(folder, repo, kind):
    entries = []
    for p in sorted(folder.rglob("*")):
        if p.is_file():
            digest = hashlib.sha256()
            with p.open("rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    digest.update(chunk)
            entries.append({"path": p.relative_to(folder).as_posix(), "bytes": p.stat().st_size, "sha256": digest.hexdigest()})
    manifest = {"repo_id": repo, "repo_type": kind, "private": True,
                "created_utc": datetime.now(timezone.utc).isoformat(), "files": entries}
    (folder / "upload_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return len(entries) + 1, sum(e["bytes"] for e in entries) + (folder / "upload_manifest.json").stat().st_size


def upload_bundles(bundles, api):
    # Secure every destination before sending any local artifacts. If permissions
    # do not allow making an existing repository private, fail before uploading.
    for kind, repo, folder in bundles:
        api.create_repo(repo_id=repo, repo_type=kind, private=True, exist_ok=True)
        if not api.repo_info(repo_id=repo, repo_type=kind).private:
            api.update_repo_settings(repo_id=repo, repo_type=kind, private=True)
        if not api.repo_info(repo_id=repo, repo_type=kind).private:
            raise RuntimeError(f"Repository is not private; upload stopped: {kind}/{repo}")
    for kind, repo, folder in bundles:
        print(f"Uploading private {kind} repository {repo} ...", flush=True)
        commit = api.upload_folder(repo_id=repo, repo_type=kind, folder_path=str(folder),
                                   commit_message="Upload SEA-Rater data and experiment artifacts")
        print(f"Done: {commit.commit_url}", flush=True)


def main(argv=None):
    args = parse_args(argv)
    args.staging_dir.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix="upload-", dir=args.staging_dir))
    bundles = []
    # Finish and validate all local packages before any remote mutation.
    if args.only in ("both", "dataset"):
        folder = stage / "dataset"
        prepare_dataset(args, folder)
        bundles.append(("dataset", args.dataset_repo, folder))
    if args.only in ("both", "model"):
        folder = stage / "model"
        prepare_model(args, folder)
        bundles.append(("model", args.model_repo, folder))
    for kind, repo, folder in bundles:
        count, size = write_manifest(folder, repo, kind)
        print(f"Private {kind}: {repo} — {count} files, {size / 1024**2:.1f} MiB")
        print(f"  Review: {folder / 'upload_manifest.json'}")
    print(f"Local packages: {stage}")
    if args.dry_run:
        print("Dry run complete. No network calls or uploads were made.")
        return
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise SystemExit("Install upload dependencies: python -m pip install --upgrade huggingface_hub pyyaml") from exc
    # HfApi reads HF_TOKEN or the token saved by `hf auth login`.
    upload_bundles(bundles, HfApi())


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError) as exc:
        raise SystemExit(f"Upload preparation failed: {exc}") from exc
