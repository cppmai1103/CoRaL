from pathlib import Path
from huggingface_hub import HfApi

# ======== EDIT THESE ========
HF_USER      = "cppmai"
DATASET_REPO = f"{HF_USER}/sea7-data"
MODEL_REPO   = f"{HF_USER}/sea7-checkpoints"
DATASET_DIR  = "checkpoints"
CKPT_DIR     = "data"
PRIVATE      = True
# ============================

api = HfApi()

# 1) Dataset folder -> dataset repo
api.create_repo(DATASET_REPO, repo_type="dataset", private=PRIVATE, exist_ok=True)
api.upload_folder(repo_id=DATASET_REPO, repo_type="dataset",
                  folder_path=DATASET_DIR, commit_message="Upload dataset")

# 2) Checkpoint folder -> model repo (goes into a subfolder like checkpoint-1000/)
api.create_repo(MODEL_REPO, repo_type="model", private=PRIVATE, exist_ok=True)
api.upload_folder(repo_id=MODEL_REPO, repo_type="model",
                  folder_path=CKPT_DIR, path_in_repo=Path(CKPT_DIR).name,
                  ignore_patterns=["*.tmp", "__pycache__/*"],
                  commit_message=f"Upload {Path(CKPT_DIR).name}")