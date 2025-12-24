import random
from typing import Any, Dict, Tuple
import numpy as np
import torch
from contextlib import contextmanager


def set_seed(seed: int, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def get_amp_dtype(precision: str):
    if precision == "fp16":
        return torch.float16
    if precision == "bf16":
        return torch.bfloat16
    return torch.float32


@contextmanager
def get_autocast_context(enabled: bool, dtype: torch.dtype, device_type: str):
    if enabled:
        with torch.autocast(device_type=device_type, dtype=dtype):
            yield
    else:
        yield


def save_checkpoint(path: str, epoch: int, model: torch.nn.Module, optimizer: torch.optim.Optimizer, best_score: float):
    ckpt = {
        "epoch": epoch,
        "best_score": best_score,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "task_configs": getattr(model, "task_configs", None),
    }
    torch.save(ckpt, path)


def load_checkpoint(path: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer, map_location=None) -> Tuple[int, float]:
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model"], strict=False)
    if "optimizer" in ckpt and optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    epoch = int(ckpt.get("epoch", 0))
    best_score = float(ckpt.get("best_score", -1e9))
    return epoch, best_score
