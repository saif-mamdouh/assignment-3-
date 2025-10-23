import torch
import torch.nn as nn
import importlib
from pathlib import Path
import sys

class SeqTrackWrapper(nn.Module):
    def __init__(self, version="v2", device=None):
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        seqtrack_path = Path("E:/Fac_books/level 4/ip/SeqTrackv2")
        lib_path = seqtrack_path / "lib"
        if str(lib_path) not in sys.path:
            sys.path.insert(0, str(lib_path))

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
            self.model = nn.Sequential(
                nn.Conv2d(3, 16, 3, 1, 1),
                nn.ReLU(),
                nn.Conv2d(16, 32, 3, 1, 1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(32, 4)
            )

        self.to(self.device)

    def forward(self, x):
        return self.model(x)
