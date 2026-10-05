# LLM-label pipeline: running, logs and error handling

Raters trained on LLM scores instead of human scores -- on the same human-annotated documents and the same
per-dimension train/validation/test splits, so only the training label changes (test keeps the human labels) --
then the same GPT-2 selection, training and evaluation as the human-rater runs. This page says where every step logs, what success looks like, and what to do when a step
fails.

## 1. Run

```bash
# once, on the login node (CPU, seconds)
python -m src.extract_llm_scores           # -> data/rater_dataset/llm_scores_annotated.{json,md}
python -m src.prepare_llm_rater_dataset    # -> data/rater_dataset_llm/ (~3,960 train / ~566 validation / ~1,170 test per dimension)
# all GPU/CPU jobs, chained with SLURM dependencies
bash sh/submit_llm_pipeline.sh
```

**Save the job IDs the submit script prints**, e.g. `bash sh/submit_llm_pipeline.sh | tee llm_pipeline_jobs.txt`.
Every job writes `job-<id>.out` (progress, results) and `job-<id>.err` (warnings, tracebacks) in the project root.

| Stage | Job name | Jobs | Typical time | Output |
| --- | --- | ---: | --- | --- |
| rater | `ft-<dim>` (edu, reas, prof, clean, cult) | 5 | ~35 min each | `checkpoints/rater_llm/finetuned/mean/<dimension>/` |
| score | `sc-<dim>` | 5 | ~1 h each | `data/pilot_scores_llm/<dimension>_mean/` |
| select | `select` (CPU) | 1 | ~15 min | `checkpoints/rater_llm/rater_comparison.md`, `data/pilot_selected/llm_{edu,avg4,avg5}_20M/` |
| gpt2 | `gpt-edu`, `gpt-avg4`, `gpt-avg5` | 3 | ~35 min each | `checkpoints/gpt2_top_doc/llm_<m>_20M_ep1_seed42/` |
| eval | `ev-test`, `ev-hq`, `ev-clean`, `ev-down`, `ev-wiki` | 5 | 5–20 min each | `checkpoints/gpt2_top_doc/llm_*_comparison.md`, `data/rater_dataset/annotated_eval*_llm/` |

Only 2 jobs per user run at once (`QOSMaxJobsPerUserLimit`), so the whole chain takes ~6–7 h.

## 2. Monitor

```bash
squeue -u $USER -o "%.8i %.28j %.8T %.10M %R"                    # what is running / waiting, and why
sacct -u $USER -S today -o JobID,JobName%28,State,ExitCode,Elapsed | grep -v "\.ba\|\.ex"   # finished jobs
tail -f job-<id>.out                                               # follow one job
grep -l -iE "traceback|error|out of memory|killed|due to time" job-*.err   # which jobs had problems
```

`squeue` REASON column:

| Reason | Meaning | Action |
| --- | --- | --- |
| `QOSMaxJobsPerUserLimit` | waiting for one of your 2 slots | none |
| `Dependency` | waiting for the step before it | none |
| `DependencyNeverSatisfied` | a step it depends on **failed**; it will never start | fix the failed job, `scancel` the stuck ones, resubmit (section 5) |
| `Priority` / `Resources` | the GPU node is busy | none |

## 3. What success looks like, per stage

| Stage | Last lines of `job-<id>.out` | Files to check |
| --- | --- | --- |
| rater | `best_epoch=… val_macro_mae=… test_macro_mae=…` then `Done: checkpoints/rater_llm/…` | `<dimension>/model/`, `predictions_test.csv`, `evaluation_report.md` |
| score | one line per language (6), then `Done: data/pilot_scores_llm/…` | `fil.csv … vie.csv` (6 files), `score_distribution.png` |
| select | `Wrote …/rater_comparison.md`, `selected … tokens (target 20,000,000 …)` for each language and method, `Done: …` | `data/pilot_selected/llm_<m>_20M/documents.csv` + `summary.md` (×3) |
| gpt2 | `Loaded shared initial weights: …V48384_seed42.pt`, `test: macro loss …`, `Wrote results to …` | `results.json`, `final/`, `summary.md` |
| eval | `Wrote <comparison file>` / `Done: …` | the comparison `.md` files |

Sanity checks before trusting results:
- rater: `val_macro_mae` should fall over the first epochs; `test_macro_mae` is against **human** labels, so it will
  be higher than for the human-label raters (the LLM scale is shifted). Judge the raters by **Spearman** in
  `rater_comparison.md`, not MAE.
- select: each language should be within a few thousand tokens of 20,000,000 (`tokens_over_target`).
- gpt2: initial loss ≈ ln(48,384) ≈ 10.8 in the sanity check; test loss should be close to the human-rater
  20M runs (≈4.40–4.44).

## 4. Common errors and fixes

| Symptom (in `.err`/`.out`, or `sacct` State) | Stage | Cause | Fix |
| --- | --- | --- | --- |
| `CUDA out of memory` | rater | long documents × micro-batch 4 | `DIMENSION=<d> MICRO_BATCH_SIZE=2 sbatch sh/finetune_llm.sh` (still 16 docs per step); if it persists add `GRADIENT_CHECKPOINTING=1` |
| `DUE TO TIME LIMIT`, State `TIMEOUT` | rater | 10 epochs took longer than 1.5 h | rerun with more time: `DIMENSION=<d> sbatch --time=3:00:00 sh/finetune_llm.sh` (restarts from scratch) |
| `early stopping at epoch 2` | rater | validation MAE stopped improving | not an error; the best epoch is kept |
| `DUE TO TIME LIMIT` | score | slower GPU / long documents | just resubmit `DIMENSION=<d> sbatch sh/score_pool_llm.sh`: finished languages are skipped (`already scored …`) |
| `No such file … rater_llm/…/model` | score | its rater job failed or was cancelled | fix the rater first (it should show `DependencyNeverSatisfied`, not run) |
| `ERROR: data/pilot_scores_llm/<d>_mean has scores for N of 6 languages` | select | a scoring job did not finish (the check stops select before a language silently drops out) | rerun that scoring job (`DIMENSION=<d> sbatch sh/score_pool_llm.sh`), then `sbatch sh/select_llm.sh` |
| `compare_raters: … no predictions_test.csv, skipping` | select | a rater is missing | the comparison still writes; rerun select after the rater is done |
| `data/pilot_selected/llm_<m>_20M/documents.csv` missing | gpt2 | select failed | rerun select, then the gpt2 jobs (section 5) |
| `EOFError` in `torch.load` (init weights) | gpt2 | two jobs creating a new init file at once | cannot happen here (the SeaLLM init file `checkpoints/gpt2_top_doc/init/…V48384_seed42.pt` exists); if it does, just resubmit |
| `pip install …` / `ConnectionError` / `HTTPError 5xx` | gpt2, rater | no network on the node (train_gpt2.sh pip-installs each run; models load from the Hub cache) | resubmit; mmBERT and SeaLLM are cached, so `export HF_HUB_OFFLINE=1` before `sbatch` avoids Hub calls |
| `.env: No such file` | gpt2 | `train_gpt2.sh` sources the HF token file | keep `.env` in the project root, submit from the project root |
| `Disk quota exceeded` / `No space left` | any | 5 rater models ≈ 3 GB, 3 GPT-2 runs ≈ 0.6 GB | free space (`du -sh checkpoints/* data/*`), then resubmit the failed job |
| State `FAILED`, `.err` empty | any | killed by the node (memory) | look for `oom-kill`/`Killed` in `.out`; resubmit with `--mem=64GB` |

## 5. Resuming after a failure

1. Find the failed job: `sacct -u $USER -S today -o JobID,JobName%28,State | grep -v COMPLETED`.
2. Read its `job-<id>.err` (traceback at the end) and apply the fix from section 4.
3. Cancel jobs stuck on it: `squeue -u $USER -t PD -o "%i %j %R" | grep DependencyNeverSatisfied` → `scancel <ids>`.
4. Resubmit:
   - one job only: the single `sbatch` line from section 4 (e.g. one rater dimension);
   - everything after a finished stage: `STAGES="score select gpt2 eval" bash sh/submit_llm_pipeline.sh`
     (drop the stages that are already done; a stage only waits for stages submitted in the same call).
   - If one rater failed but the other four finished, rerun that rater alone, then submit its scoring job after it
     (`DIMENSION=<d> sbatch --dependency=afterok:<rater job> sh/score_pool_llm.sh`), then
     `STAGES="select gpt2 eval"` with `sbatch --dependency` on that scoring job, or simply wait for it to finish
     and run `STAGES="select gpt2 eval" bash sh/submit_llm_pipeline.sh`.

What is safe to rerun:

| Step | Rerun behaviour |
| --- | --- |
| `extract_llm_scores`, `prepare_llm_rater_dataset` | overwrite their outputs; deterministic (seed 42) |
| rater | retrains from scratch, overwrites `checkpoints/rater_llm/finetuned/mean/<d>/` |
| score | skips languages already scored; `--force` (edit the script) to redo |
| select | overwrites the avg4/avg5 scores and the three selections |
| gpt2 | retrains, overwrites the run folder |
| eval (annotated) | skips runs already in the results JSON (`FORCE=1` to redo); the others recompute |

## 6. Results to collect at the end

- `checkpoints/rater_llm/rater_comparison.md` — LLM-label vs human-label raters on the human test split (Spearman).
- `checkpoints/gpt2_top_doc/llm_test_comparison.md` — test split, random_20M vs LLM-rater vs human-rater runs.
- `data/rater_dataset/annotated_eval_llm/gpt2_summary_highquality.md` — high quality = avg5 top 50%.
- `data/rater_dataset/annotated_eval_cleanliness_llm/gpt2_summary_highquality.md` — high quality = cleanliness 5.
- `checkpoints/gpt2_top_doc/llm_downstream_comparison.md`, `llm_wiki_sib200_comparison.md`.
