# checkpoint_utils.py
import torch
import random
import numpy as np
import os

def save_checkpoint(path: str, epoch: int, model: torch.nn.Module, optimizer, scheduler, extra: dict = None):
    if extra is None:
        extra = {}
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "rng_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
        },
        "extra": extra,
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(state, path)
    return path

def load_checkpoint(path: str, model: torch.nn.Module, optimizer=None, scheduler=None, map_location="cpu"):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and ckpt.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    rng = ckpt.get("rng_state", None)
    if rng is not None:
        import random as _random, numpy as _np, torch as _torch
        _random.setstate(rng["python"])
        _np.random.set_state(rng["numpy"])
        _torch.set_rng_state(rng["torch"])
    return ckpt
