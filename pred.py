import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from collections import defaultdict
from typing import Dict, Optional
from contextlib import nullcontext
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

from dataset import MultiTaskDataset
from models.multitask_model import MultiTaskModel
from utils.data import build_task_configs_from_dataframe, multitask_collate_fn
from utils.misc import set_seed
from val_evaluate import evaluate

DATA_ROOT_PATH = "/root/train/train"
VAL_SPLIT = 0.2
FORCE_BACKBONE_IMAGE_SIZE = 448

BATCH_SIZE = 1
NUM_WORKERS = 4
PIN_MEMORY = True

SEED = 42
DETERMINISTIC = False

PRECISION = "bf16"
GRAD_CHECKPOINT = True

MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
TRUST_REMOTE_CODE = True
ATTN_IMPLEMENTATION = None

KEEP_LLM = False
LOAD_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

CKPT_PATH = "/root/ISBI-2026/FMCUIA-ISBI/best_model.pth"


def resolve_data_root(data_root: str) -> str:
    candidates = [
        data_root,
        os.path.join(data_root, "train"),
        os.path.join(data_root, "train", "train"),
        os.path.join(data_root, "data", "train"),
    ]
    for c in candidates:
        if os.path.isdir(os.path.join(c, "csv_files")):
            return c
    raise FileNotFoundError(
        "CSV path not found.\nTried:\n"
        + "\n".join([f"- {os.path.join(c, 'csv_files')}" for c in candidates])
    )


def build_transforms(is_train: bool, image_size: int, deterministic: bool) -> A.Compose:
    t_list = [A.Resize(image_size, image_size)]
    if is_train and not deterministic:
        t_list += [
            A.RandomBrightnessContrast(p=0.35),
            A.GaussNoise(var_limit=(5.0, 45.0), p=0.30),
            A.GaussianBlur(blur_limit=(3, 5), p=0.20),
            A.MotionBlur(blur_limit=(3, 5), p=0.10),
        ]
    t_list += [ToTensorV2()]
    return A.Compose(
        t_list,
        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["class_labels"],
            clip=True,
            min_visibility=0.0,
        ),
    )


def make_stratified_split(df, val_split: float, seed: int):
    rng = np.random.default_rng(seed)
    train_idx = []
    val_idx = []
    groups = df.groupby("task_id").indices
    for _, idxs in groups.items():
        idxs = np.array(list(idxs), dtype=np.int64)
        rng.shuffle(idxs)
        v = int(np.floor(len(idxs) * val_split))
        if v == 0 and val_split > 0 and len(idxs) > 1:
            v = 1
        if v >= len(idxs):
            v = max(0, len(idxs) - 1)
        val_idx.extend(idxs[:v].tolist())
        train_idx.extend(idxs[v:].tolist())
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


def _extract_state_dict(ckpt: Dict):
    if isinstance(ckpt, dict):
        for k in ("model", "state_dict", "model_state_dict"):
            v = ckpt.get(k, None)
            if isinstance(v, dict) and len(v) > 0:
                return v
    return ckpt


def _get_amp_dtype(precision: str):
    p = str(precision).lower()
    if p == "fp16":
        return torch.float16
    if p == "bf16":
        return torch.bfloat16
    return torch.float32


@torch.no_grad()
def _materialize_dynamic_modules(model: torch.nn.Module, task_configs: Dict[str, Dict], device: torch.device, image_size: int, precision: str):
    model.eval()
    use_amp = (device.type == "cuda") and (str(precision).lower() in ("bf16", "fp16"))
    amp_dtype = _get_amp_dtype(precision)
    ctx = torch.autocast(device_type="cuda", dtype=amp_dtype) if use_amp else nullcontext()

    def pick_task_id(tname: str):
        for tid, cfg in task_configs.items():
            if str(cfg.get("task_name", "")).lower() == tname.lower():
                return tid
        return None

    any_tid = next(iter(task_configs.keys()))
    cls_tid = pick_task_id("classification")
    reg_tid = pick_task_id("Regression")
    det_tid = pick_task_id("detection")
    seg_tid = pick_task_id("segmentation")

    dummy = torch.zeros((1, 3, image_size, image_size), device=device, dtype=torch.float32)

    with ctx:
        model(dummy, task_id=any_tid, labels=None)
        if seg_tid is not None:
            model(dummy, task_id=seg_tid, labels=None)
        if cls_tid is not None:
            model(dummy, task_id=cls_tid, labels=None)
        if reg_tid is not None:
            model(dummy, task_id=reg_tid, labels=None)
        if det_tid is not None:
            model(dummy, task_id=det_tid, labels=None)


def main():
    set_seed(SEED, deterministic=DETERMINISTIC)

    device = torch.device(LOAD_DEVICE if torch.cuda.is_available() else "cpu")
    print(f"[Info] Device: {device}")
    data_root = resolve_data_root(DATA_ROOT_PATH)
    print(f"[Info] Resolved data root: {data_root}")

    train_tfms = build_transforms(is_train=True, image_size=FORCE_BACKBONE_IMAGE_SIZE, deterministic=DETERMINISTIC)
    val_tfms = build_transforms(is_train=False, image_size=FORCE_BACKBONE_IMAGE_SIZE, deterministic=True)

    full_dataset_for_index = MultiTaskDataset(data_root, transforms=None)
    df = full_dataset_for_index.dataframe
    train_indices, val_indices = make_stratified_split(df, val_split=VAL_SPLIT, seed=SEED)

    train_dataset = MultiTaskDataset(data_root, transforms=train_tfms)
    val_dataset = MultiTaskDataset(data_root, transforms=val_tfms)

    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)

    train_subset.dataframe = train_dataset.dataframe.iloc[train_indices].reset_index(drop=True)
    val_subset.dataframe = val_dataset.dataframe.iloc[val_indices].reset_index(drop=True)

    print(f"[Info] Dataset split: {len(train_indices)} train samples, {len(val_indices)} val samples (val_split={VAL_SPLIT})")

    rows_by_task = defaultdict(int)
    task_name_map = {}
    vdf = val_subset.dataframe
    if "task_name" in vdf.columns:
        task_name_map = vdf.groupby("task_id")["task_name"].first().to_dict()
    for tid in vdf["task_id"].tolist():
        rows_by_task[str(tid)] += 1

    print("[Val Task Counts]")
    for tid in sorted(rows_by_task.keys()):
        tname = task_name_map.get(tid, "")
        if tname:
            print(f"{tid}: {rows_by_task[tid]} ({tname})")
        else:
            print(f"{tid}: {rows_by_task[tid]}")

    task_configs = build_task_configs_from_dataframe(train_dataset.dataframe)

    if str(PRECISION).lower() == "bf16":
        backbone_dtype = torch.bfloat16
    elif str(PRECISION).lower() == "fp16":
        backbone_dtype = torch.float16
    else:
        backbone_dtype = torch.float32

    model = MultiTaskModel(
        task_configs=task_configs,
        model_name=MODEL_NAME,
        trust_remote_code=TRUST_REMOTE_CODE,
        attn_implementation=ATTN_IMPLEMENTATION,
        keep_llm=KEEP_LLM,
        fpn_dim=256,
        backbone_dtype=backbone_dtype,
        load_device=LOAD_DEVICE,
        force_image_size=FORCE_BACKBONE_IMAGE_SIZE,
    )

    if GRAD_CHECKPOINT:
        model.enable_gradient_checkpointing()

    model.to(device)

    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    epoch = ckpt.get("epoch", None) if isinstance(ckpt, dict) else None
    best_score = ckpt.get("best_score", None) if isinstance(ckpt, dict) else None

    print(f"`torch_dtype` is deprecated! Use `dtype` instead!")
    print(f"[Info] Loaded checkpoint: {CKPT_PATH}")
    if epoch is not None:
        print(f"[Info] Ckpt epoch: {epoch}")
    if best_score is not None:
        print(f"[Info] Ckpt best_score: {best_score}")

    state = _extract_state_dict(ckpt)

    _materialize_dynamic_modules(model, task_configs, device, FORCE_BACKBONE_IMAGE_SIZE, PRECISION)

    load_res = model.load_state_dict(state, strict=False)
    missing = list(load_res.missing_keys) if hasattr(load_res, "missing_keys") else []
    unexpected = list(load_res.unexpected_keys) if hasattr(load_res, "unexpected_keys") else []

    print(f"[Info] Missing keys: {len(missing)}")
    print(f"[Info] Unexpected keys: {len(unexpected)}")
    if len(unexpected) > 0:
        print("[Info] Unexpected keys list:")
        for k in unexpected:
            print(k)

    val_loader = DataLoader(
        val_subset,
        batch_size=max(1, BATCH_SIZE),
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        collate_fn=multitask_collate_fn,
        persistent_workers=(NUM_WORKERS > 0),
        prefetch_factor=4 if NUM_WORKERS > 0 else None,
    )

    val_results_df = evaluate(model, val_loader, device)
    score_cols = [col for col in val_results_df.columns if "MAE" not in col and isinstance(val_results_df[col].iloc[0], (int, float, np.floating, np.integer))]
    avg_val_score = 0.0
    if not val_results_df.empty and score_cols:
        avg_val_score = float(val_results_df[score_cols].mean().mean())

    print("\n--- Validation Report ---")
    if not val_results_df.empty:
        print(val_results_df.to_string(index=False))
    print(f"--- Average Val Score (Higher is better): {avg_val_score:.4f} ---")


if __name__ == "__main__":
    main()
