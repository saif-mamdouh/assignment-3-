import os
import json
import random
from pathlib import Path

# ✅ إعدادات أساسية
DATASET_DIR = Path("E:/Fac_books/level 4/ip/assignment_3/dataset")
OUTPUT_DIR = Path("E:/Fac_books/level 4/ip/assignment_3/lasot_subset")
SELECTED_CLASSES = ["bicycle", "airplane"]   # ✅ الفئتان المختارتان
TRAIN_RATIO = 0.8
SEED = 42


def prepare():
    random.seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    subset_info = {}

    for cls in SELECTED_CLASSES:
        class_path = DATASET_DIR / cls
        if not class_path.exists():
            print(f"⚠️ Warning: class folder not found -> {class_path}")
            continue

        # جميع التسلسلات داخل الفئة
        sequences = [seq for seq in os.listdir(class_path) if os.path.isdir(class_path / seq)]
        sequences.sort()

        # تقسيم Train/Test
        random.shuffle(sequences)
        split_idx = int(len(sequences) * TRAIN_RATIO)
        train_seq = sequences[:split_idx]
        test_seq = sequences[split_idx:]

        subset_info[cls] = {
            "train": train_seq,
            "test": test_seq,
            "train_size": len(train_seq),
            "test_size": len(test_seq),
        }

        print(f"✅ {cls}: {len(train_seq)} train + {len(test_seq)} test sequences")

    # حفظ النتائج في ملف JSON
    out_path = OUTPUT_DIR / "subset_info.json"
    with open(out_path, "w") as f:
        json.dump(subset_info, f, indent=2)

    print(f"\n📁 Wrote subset info to {out_path}")
    print(json.dumps(subset_info, indent=2))


if __name__ == "__main__":
    prepare()
