# train.py
"""
Training script for assignment_3 — uses SeqTrack model from SeqTrackv2 repo when available.

Features:
- Adds SeqTrackv2 paths to sys.path so repo imports work.
- Loads LaSOT-style sequences using subset_info.json.
- Instantiates SeqTrackV2 / SeqTrack class from repo when possible; falls back to a small model if not.
- Checkpoints include optimizer, scheduler, and RNG states (python, numpy, torch, torch.cuda).
- Logs timing every log_every_samples via TrainingLogger from repo.
"""
import os
import sys
import argparse
import json
import time
import random
import glob
from pathlib import Path
import importlib

import numpy as np
import torch
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# ============== PATH FIXES ==============
PROJECT_ROOT = Path(__file__).resolve().parent
SEQTRACKV2_PATH = Path("F:\image\SeqTrackv2")
LIB_PATH = SEQTRACKV2_PATH / "lib"

for p in [str(LIB_PATH), str(SEQTRACKV2_PATH), str(PROJECT_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

print("[DEBUG] sys.path includes:")
for p in sys.path[:6]:
    print("   ", p)
# =======================================

# try to import repo utilities
try:
    from lib.utils.logger import TrainingLogger
except Exception as e:
    raise ImportError(f"Could not import TrainingLogger from repo (lib.utils.logger). Error: {e}")

try:
    from lib.utils.box_ops import box_iou as box_iou_from_repo
except Exception:
    from torchvision.ops import box_iou as tv_box_iou
    def box_iou_from_repo(a, b):
        return tv_box_iou(a, b), None

# ---------------- Checkpoint Handling ----------------
def save_checkpoint(path, epoch, model, optimizer, scheduler, extra=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ck = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer else None,
        "scheduler_state": scheduler.state_dict() if scheduler else None,
        "extra": extra or {},
        "rng_python_state": random.getstate(),
        "rng_numpy_state": np.random.get_state(),
        "rng_torch_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        ck["rng_torch_cuda_state_all"] = torch.cuda.get_rng_state_all()
    torch.save(ck, path)
    print(f"[INFO] Checkpoint saved at {path}")

def load_checkpoint(path, model, optimizer=None, scheduler=None, map_location="cpu", restore_rng=True):
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(ckpt.get("model_state", {}))
    if optimizer and ckpt.get("optimizer_state"):
        optimizer.load_state_dict(ckpt["optimizer_state"])
    if scheduler and ckpt.get("scheduler_state"):
        scheduler.load_state_dict(ckpt["scheduler_state"])
    if restore_rng:
        try:
            if "rng_python_state" in ckpt:
                random.setstate(ckpt["rng_python_state"])
            if "rng_numpy_state" in ckpt:
                np.random.set_state(ckpt["rng_numpy_state"])
            if "rng_torch_state" in ckpt:
                torch.set_rng_state(ckpt["rng_torch_state"])
            if torch.cuda.is_available() and "rng_torch_cuda_state_all" in ckpt:
                torch.cuda.set_rng_state_all(ckpt["rng_torch_cuda_state_all"])
        except Exception as e:
            print(f"[WARNING] Could not restore RNG states: {e}")
    print(f"[INFO] Checkpoint loaded from {path}")
    return ckpt

# ---------------- Utilities ----------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--team_number", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--subset_json", type=str, default="lasot_subset/subset_info.json")
    parser.add_argument("--dataset_root", type=str, default="dataset")
    parser.add_argument("--phase", type=int, choices=[1, 2], default=1)
    parser.add_argument("--log_every_samples", type=int, default=50)
    return parser.parse_args()

# ---------------- SeqTrack import ----------------
def import_seqtrack_class():
    candidates = [
        ("lib.models.seqtrack.seqtrack", ["SeqTrackV2", "SeqTrack"]),
        ("lib.models.seqtrackv2.seqtrackv2", ["SeqTrackV2", "SeqTrack"]),
        ("lib.models.seqtrackv2.seqtrack", ["SeqTrackV2", "SeqTrack"]),
    ]
    tried = []
    for mod_name, names in candidates:
        try:
            mod = importlib.import_module(mod_name)
        except Exception as e:
            tried.append((mod_name, str(e)))
            continue
        for nm in names:
            if hasattr(mod, nm):
                return getattr(mod, nm), f"Imported {nm} from {mod_name}"
        for alt in ("build_model", "build_seqtrack", "make_model"):
            if hasattr(mod, alt):
                return getattr(mod, alt), f"Using factory {alt} from {mod_name}"
        tried.append((mod_name, "no known symbol"))
    return None, f"Failed candidates: {tried}"

# ---------------- Loss ----------------
def get_loss_function():
    try:
        mod = importlib.import_module("lib.models.seqtrack.loss")
        for name in ("build_loss", "make_loss", "build_criterion"):
            if hasattr(mod, name):
                fn = getattr(mod, name)
                try: return fn()
                except: return fn
        for cname in ("SeqTrackLoss", "Loss"):
            if hasattr(mod, cname):
                cls = getattr(mod, cname)
                try: return cls()
                except: return cls
        print("[WARNING] Found lib.models.seqtrack.loss but couldn't instantiate; falling back.")
    except Exception: pass
    return torch.nn.MSELoss()

# ---------------- Dataset ----------------
class SimpleTrackingDataset(Dataset):
    def __init__(self, subset_json, dataset_root):
        with open(subset_json, "r") as f:
            self.subset_info = json.load(f)
        self.dataset_root = dataset_root
        self.samples = []

        for cls_name, info in self.subset_info.items():
            train_seqs = info.get("train", []) if isinstance(info, dict) else []
            test_seqs = info.get("test", []) if isinstance(info, dict) else []
            all_seqs = list(train_seqs) + list(test_seqs)

            for seq in all_seqs:
                img_dir = Path(dataset_root) / cls_name / seq / "img"
                gt_file = Path(dataset_root) / cls_name / seq / "groundtruth.txt"
                if not gt_file.exists() or not img_dir.exists():
                    continue
                imgs = sorted(glob.glob(str(img_dir / "*.jpg")) + glob.glob(str(img_dir / "*.png")))
                if len(imgs) == 0:
                    continue
                with open(gt_file, "r") as gtf:
                    lines = [ln.strip() for ln in gtf.readlines()]
                for i, img_path in enumerate(imgs):
                    if i < len(lines) and lines[i].strip():
                        parts = lines[i].replace(",", " ").split()
                        try: bbox = [float(x) for x in parts[:4]]
                        except: bbox = [0.0, 0.0, 0.0, 0.0]
                    else: bbox = [0.0, 0.0, 0.0, 0.0]
                    if len(bbox) == 4: self.samples.append((img_path, bbox))
        print(f"[DEBUG] Loaded {len(self.samples)} samples from {subset_json}")
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
        ])

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        img_path, bbox = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        bbox = torch.tensor(bbox, dtype=torch.float32)
        return img, bbox

def build_dataloaders_from_subset(subset_json, dataset_root, batch_size=8):
    ds = SimpleTrackingDataset(subset_json, dataset_root)
    if len(ds) == 0:
        raise ValueError("No samples found! Check dataset path or subset_json content.")
    val_size = max(1, len(ds)//10)
    train_len = len(ds) - val_size
    train_ds, val_ds = torch.utils.data.random_split(ds, [train_len, val_size])
    print(f"[DEBUG] Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader, len(train_ds)

# ---------------- Loss Wrapper ----------------
def compute_loss_wrapper(pred_boxes, target_boxes, criterion):
    try:
        if isinstance(criterion, torch.nn.MSELoss) or criterion==torch.nn.MSELoss:
            mask = (target_boxes.sum(dim=1)!=0).float()
            if mask.sum()>0:
                diff = (pred_boxes-target_boxes).pow(2).sum(dim=1)
                return (diff*mask).sum()/mask.sum()
            else:
                return torch.tensor(0.0, device=pred_boxes.device, requires_grad=True)
        else:
            return criterion(pred_boxes, target_boxes)
    except:
        return torch.nn.functional.mse_loss(pred_boxes, target_boxes)
    
# ---------------- upload to HuggingFace ----------------
from huggingface_hub import HfApi
api = HfApi()
api.upload_file(
    path_or_fileobj=ckpt_path,
    path_in_repo=f"checkpoints/epoch_{epoch:02d}.pth",
    repo_id="usaifmamdouh11/assignment_3",
    token="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"
)

# ---------------- Main ----------------
def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    phase = args.phase
    log_path = f"training_phase{phase}.log"
    metrics_json = f"metrics_phase{phase}.json"
    logger = TrainingLogger(log_path, print_every=args.log_every_samples)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device(args.device)

    seqtrack_class, note = import_seqtrack_class()
    model = None
    if seqtrack_class:
        print(f"[INFO] Found SeqTrack candidate: {note}")
        inst_ex = None
        for attempt in range(3):
            try:
                if attempt==0: candidate = seqtrack_class()
                elif attempt==1: candidate = seqtrack_class({})
                else: candidate = seqtrack_class(device=str(device))
                if hasattr(candidate, "to"):
                    model = candidate.to(device)
                    break
            except Exception as e: inst_ex = e
        if model is None:
            print(f"[WARNING] Could not instantiate SeqTrack: {inst_ex}")
    if model is None:
        print("[INFO] Using fallback dummy model.")
        model = torch.nn.Sequential(
            torch.nn.Conv2d(3,16,3,1,1), torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(16,4)
        ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    criterion = get_loss_function()
    train_loader, val_loader, total_train_samples = build_dataloaders_from_subset(
        args.subset_json, args.dataset_root, batch_size=args.batch_size
    )

    start_epoch = 1
    if args.resume_from:
        ckpt = load_checkpoint(args.resume_from, model, optimizer=optimizer, scheduler=scheduler, map_location=device)
        start_epoch = ckpt.get("epoch",1)+1
        print(f"[INFO] Resuming from epoch {start_epoch}")

    metrics_history = {}

    for epoch in range(start_epoch, args.epochs+1):
        set_seed(args.team_number)
        model.train()
        logger.start_epoch(epoch, total_train_samples)
        running_loss = 0.0
        samples_processed = 0

        for imgs, target_bboxes in train_loader:
            imgs, target_bboxes = imgs.to(device), target_bboxes.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            pred_boxes = outputs[:, :4] if outputs.dim()==2 and outputs.size(1)>=4 else outputs.view(outputs.size(0),-1)[:, :4]
            loss = compute_loss_wrapper(pred_boxes, target_bboxes, criterion)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item())
            samples_processed += imgs.size(0)
            logger.step_samples(imgs.size(0))

        scheduler.step()

        # Validation
        model.eval()
        val_losses, iou_vals = [], []
        with torch.no_grad():
            for imgs, target_bboxes in val_loader:
                imgs, target_bboxes = imgs.to(device), target_bboxes.to(device)
                outputs = model(imgs)
                pred_boxes = outputs[:, :4] if outputs.dim()==2 and outputs.size(1)>=4 else outputs.view(outputs.size(0),-1)[:, :4]
                loss = compute_loss_wrapper(pred_boxes, target_bboxes, criterion)
                val_losses.append(float(loss.item()))
                for pb, tb in zip(pred_boxes.cpu(), target_bboxes.cpu()):
                    try:
                        a, b = pb.unsqueeze(0), tb.unsqueeze(0)
                        iou_tensor, _ = box_iou_from_repo(a,b)
                        iou_val = float(iou_tensor.item()) if iou_tensor.numel()==1 else float(iou_tensor.reshape(-1)[0].item())
                    except: iou_val = 0.0
                    iou_vals.append(iou_val)

        avg_val = float(np.mean(val_losses)) if val_losses else None
        avg_train = float(running_loss/max(1,len(train_loader)))
        avg_iou = float(np.mean(iou_vals)) if iou_vals else 0.0

        ckpt_path = os.path.join(args.save_dir, f"epoch_{epoch:02d}.pth")
        save_checkpoint(ckpt_path, epoch, model, optimizer, scheduler, extra={"val_loss": avg_val, "avg_iou": avg_iou})
        print(f"✅ Saved checkpoint {ckpt_path} (local only).")

        logger.log_metrics(epoch, "end", train_loss=avg_train, val_loss=avg_val, iou=avg_iou, metrics_dict={"lr": scheduler.get_last_lr()})
        metrics_history[epoch] = {"train_loss": avg_train, "val_loss": avg_val, "iou": avg_iou}
        with open(metrics_json, "w") as f:
            json.dump(metrics_history, f, indent=2)

    print("🎯 Training finished successfully.")

if __name__ == "__main__":
    main()
