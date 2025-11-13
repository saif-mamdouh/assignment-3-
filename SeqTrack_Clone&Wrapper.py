import torch
import torch.nn as nn
import importlib
from pathlib import Path
import sys
import subprocess

class SeqTrackWrapper(nn.Module):
    def __init__(self, version="v2", device=None):
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # المسار المحلي للمشروع
        seqtrack_path = Path("E:/Fac_books/level 4/ip/SeqTrackv2")

        # ✅ كلون للمشروع لو مش موجود
        if not seqtrack_path.exists():
            print("[INFO] 🔄 Cloning SeqTrackv2 repository from GitHub...")
            try:
                subprocess.run(
                    ["git", "clone", "https://github.com/jinyankai/SeqTrackv2", str(seqtrack_path)],
                    check=True,
                )
                print("[INFO] ✅ Successfully cloned SeqTrackv2 repository.")
            except subprocess.CalledProcessError as e:
                print(f"[Error] ❌ Failed to clone SeqTrackv2: {e}")
                print("Using dummy fallback model instead.")
                self._init_dummy()
                return

        # إضافة lib للمسار
        lib_path = seqtrack_path / "lib"
        if str(lib_path) not in sys.path:
            sys.path.insert(0, str(lib_path))

        # محاولة استيراد الموديل الحقيقي
        try:
            if version == "v2":
                seqtrack_module = importlib.import_module("models.seqtrackv2.seqtrackv2")
                if hasattr(seqtrack_module, "SeqTrackV2"):
                    self.model = seqtrack_module.SeqTrackV2()
                elif hasattr(seqtrack_module, "SeqTrack"):
                    self.model = seqtrack_module.SeqTrack()
                else:
                    raise ImportError("No SeqTrackV2 or SeqTrack found in seqtrackv2.py")
            else:
                seqtrack_module = importlib.import_module("models.seqtrack.seqtrack")
                if hasattr(seqtrack_module, "SeqTrack"):
                    self.model = seqtrack_module.SeqTrack()
                else:
                    raise ImportError("No SeqTrack found in seqtrack.py")

            print(f"[INFO] ✅ Successfully loaded real SeqTrack model from {seqtrack_module.__file__}")

        except Exception as e:
            print(f"[Warning] ❌ Failed to import real SeqTrack model: {e}")
            print("Using dummy fallback model instead.")
            self._init_dummy()

        self.to(self.device)

    def _init_dummy(self):
        """Fallback dummy model in case cloning or import fails."""
        self.model = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(32, 4)
        )

    def forward(self, x):
        return self.model(x)
