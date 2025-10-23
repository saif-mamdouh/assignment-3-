import os
import zipfile
from huggingface_hub import snapshot_download
from pathlib import Path

# ✅ Target directory for dataset
target_dir = Path("E:/Fac_books/level 4/ip/assignment_3/dataset")

# ✅ Choose a few classes to limit size
chosen_classes = ["airplane", "bear", "bicycle"]

# ✅ Step 1: Download ZIP files only for chosen classes
print(f"📥 Downloading classes: {chosen_classes}")

snapshot_download(
    repo_id="l-lt/LaSOT",
    repo_type="dataset",
    local_dir=str(target_dir),
    allow_patterns=[f"{cls}.zip" for cls in chosen_classes]
)

print("\n✅ Download complete!")

# ✅ Step 2: Extract ZIPs
for cls in chosen_classes:
    zip_path = target_dir / f"{cls}.zip"
    if zip_path.exists():
        print(f"🗜️ Extracting {cls}.zip ...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(target_dir / cls)
    else:
        print(f"⚠️ Missing ZIP file: {zip_path}")

print("\n✅ All extraction done!")

# ✅ Step 3: Verify extracted folders
for cls in chosen_classes:
    cls_path = target_dir / cls
    if cls_path.exists():
        subdirs = [f.name for f in cls_path.iterdir() if f.is_dir()]
        print(f"✅ Found {len(subdirs)} sequences in '{cls}' (e.g., {subdirs[:3]})")
    else:
        print(f"⚠️ Folder not found for class: {cls}")
