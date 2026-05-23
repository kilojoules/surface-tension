"""Lightweight LoRA-adapter push to Hub with a clear timeout target.

Avoids loading the base model just to push (which is what model.push_to_hub
does and is what hung in our last run).

Usage:
  python3 push_adapter.py FOLDER REPO_ID
"""
import os
import sys

from huggingface_hub import create_repo, upload_folder

if len(sys.argv) != 3:
    print("usage: push_adapter.py FOLDER REPO_ID")
    sys.exit(2)

folder, repo = sys.argv[1], sys.argv[2]
print(f"creating repo (idempotent): {repo}")
create_repo(repo_id=repo, repo_type="model", exist_ok=True)
print(f"uploading {folder} -> {repo}")
upload_folder(
    folder_path=folder,
    repo_id=repo,
    repo_type="model",
    commit_message=f"adapter from {folder}",
)
print(f"pushed: https://huggingface.co/{repo}")
