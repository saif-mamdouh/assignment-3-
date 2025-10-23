import torch
import torch.nn as nn
import importlib

class SeqTrackWrapper(nn.Module):
    def __init__(self, version="v1", pretrained=False, device=None):
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        try:
            if version == "v2":
                seqtrack_module = importlib.import_module("SeqTrackv2.lib.models.seqtrackv2.seqtrackv2")
                self.model = seqtrack_module.SeqTrackV2()  # افترض أن الكلاس اسمه كده
            else:
                seqtrack_module = importlib.import_module("SeqTrackv2.lib.models.seqtrack.seqtrack")
                self.model = seqtrack_module.SeqTrack()  # الكلاس الأصلي في seqtrack.py
        except Exception as e:
            print(f"[Warning] Failed to import real SeqTrack model: {e}")
            print("Using dummy model for fallback.")
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
