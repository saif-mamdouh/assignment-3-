# hf_upload.py
from huggingface_hub import HfApi
import os

def upload_checkpoint_local_to_hf(local_path: str, hf_repo_id: str, hf_token: str, repo_subpath: str = ""):
    """
    Upload a single file to a HF repo. repo_subpath is the path inside the repo, e.g. 'checkpoints/epoch_01.pth'
    Returns the HF raw file URL on success.
    """
    api = HfApi(token=hf_token)
    filename = os.path.basename(local_path)
    if repo_subpath:
        path_in_repo = repo_subpath
    else:
        path_in_repo = f"checkpoints/{filename}"
    api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo=path_in_repo,
        repo_id=hf_repo_id,
        token=hf_token
    )
    # Raw file URL (convenience)
    return f"https://huggingface.co/{hf_repo_id}/-/raw/main/{path_in_repo}?ref=main"
