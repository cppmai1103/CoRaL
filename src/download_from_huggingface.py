"""Download the cppmai/sea-rater dataset from Hugging Face.

Setup (one-time):
    pip install huggingface_hub
    huggingface-cli login          # or: set HF_TOKEN env var (required if the repo is private)

Usage:
    python -m src.download_from_huggingface
    python -m src.download_from_huggingface --local-dir data/annotation_batches --revision main
"""

import argparse

from huggingface_hub import snapshot_download

REPO_ID = "cppmai/sea-rater"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--local-dir", default="data/annotation_batches", help="Directory to download into (default: data/annotation_batches)")
    parser.add_argument("--revision", default=None, help="Branch, tag, or commit hash to download (default: latest)")
    args = parser.parse_args()

    print(f"Downloading {REPO_ID} -> {args.local_dir} ...")
    path = snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        revision=args.revision,
        local_dir=args.local_dir,
    )
    print(f"Done: {path}")


if __name__ == "__main__":
    main()
