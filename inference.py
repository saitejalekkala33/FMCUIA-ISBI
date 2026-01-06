import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import json
import glob
from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

from models.multitask_model import MultiTaskModel

DEFAULT_DATA_ROOT = "/root/train/val"
DEFAULT_OUTPUT_DIR = "/root/train/output"
DEFAULT_BATCH_SIZE = 1

MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
TRUST_REMOTE_CODE = True
ATTN_IMPLEMENTATION = None
KEEP_LLM = False
LOAD_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
PRECISION = "bf16"
FORCE_BACKBONE_IMAGE_SIZE = 448
GRAD_CHECKPOINT = True

CKPT_PATH = "/root/ISBI-2026/FMCUIA-ISBI/best_model.pth"


def resolve_data_root(data_root: str) -> str:
    data_root = os.path.abspath(str(data_root))
    candidates = [
        data_root,
        os.path.join(data_root, "val"),
        os.path.join(data_root, "test"),
        os.path.join(data_root, "data"),
    ]
    for c in candidates:
        if os.path.isdir(os.path.join(c, "csv_files")):
            return c
    for c in candidates:
        for sub in ("val", "test", "data"):
            cc = os.path.join(c, sub)
            if os.path.isdir(os.path.join(cc, "csv_files")):
                return cc
    raise FileNotFoundError(f"Could not find csv_files under: {data_root}")


def _read_all_csvs(csv_dir: str) -> pd.DataFrame:
    csvs = sorted(glob.glob(os.path.join(csv_dir, "*.csv")))
    if not csvs:
        raise FileNotFoundError(f"No CSV files found in: {csv_dir}")
    dfs = []
    for p in csvs:
        try:
            dfs.append(pd.read_csv(p))
        except Exception:
            dfs.append(pd.read_csv(p, encoding="latin-1"))
    return pd.concat(dfs, ignore_index=True).reset_index(drop=True)


def _build_task_configs_from_dataframe(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    task_configs: Dict[str, Dict[str, Any]] = {}
    for _, row in df.iterrows():
        tid = str(row["task_id"])
        if tid not in task_configs:
            task_configs[tid] = {
                "task_name": str(row["task_name"]),
                "num_classes": int(row["num_classes"]),
            }
    if not task_configs:
        raise RuntimeError("Failed to build task_configs from dataframe")
    return task_configs


def build_transforms(image_size: int) -> A.Compose:
    return A.Compose([A.Resize(image_size, image_size), ToTensorV2()])


def _extract_state_dict(ckpt: Any) -> Dict[str, torch.Tensor]:
    if isinstance(ckpt, dict):
        for k in ("model", "state_dict", "model_state_dict"):
            v = ckpt.get(k, None)
            if isinstance(v, dict) and len(v) > 0:
                return v
        return {k: v for k, v in ckpt.items() if isinstance(v, torch.Tensor)}
    raise RuntimeError("Checkpoint format not understood")


def _get_amp_dtype(precision: str) -> torch.dtype:
    p = str(precision).lower().strip()
    if p == "fp16":
        return torch.float16
    if p == "bf16":
        return torch.bfloat16
    return torch.float32


@torch.no_grad()
def _materialize_dynamic_modules(
    model: torch.nn.Module,
    task_configs: Dict[str, Dict[str, Any]],
    device: torch.device,
    image_size: int,
    precision: str,
):
    model.eval()
    use_amp = (device.type == "cuda") and (str(precision).lower() in ("bf16", "fp16"))
    amp_dtype = _get_amp_dtype(precision)
    ctx = torch.autocast(device_type="cuda", dtype=amp_dtype) if use_amp else nullcontext()

    def pick_task_id(tname: str) -> Optional[str]:
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


class InferenceDataset(Dataset):
    def __init__(self, data_root: str, transforms: Optional[A.Compose] = None):
        super().__init__()
        self.data_root = str(data_root)
        self.transforms = transforms
        self.csv_path = os.path.join(self.data_root, "csv_files")
        if not os.path.isdir(self.csv_path):
            raise FileNotFoundError(f"CSV path not found: {self.csv_path}")
        self.dataframe = _read_all_csvs(self.csv_path)
        print(f"[Info] Data loaded. Total samples: {len(self.dataframe)}")

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        record = self.dataframe.iloc[idx]
        task_id = str(record["task_id"])
        task_name = str(record["task_name"])
        image_rel_path = str(record["image_path"])
        image_abs_path = os.path.normpath(os.path.join(self.csv_path, image_rel_path))

        image = cv2.imread(image_abs_path, cv2.IMREAD_COLOR)
        if image is None:
            return self.__getitem__((idx + 1) % len(self))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h0, w0 = image.shape[:2]

        mask_path = None
        if task_name == "segmentation":
            if "mask_path" in record.index and pd.notna(record["mask_path"]):
                mask_path = str(record["mask_path"])

        if self.transforms is not None:
            aug = self.transforms(image=image)
            image_t = aug["image"]
        else:
            image_t = ToTensorV2()(image=image)["image"]

        return {
            "image": image_t,
            "task_id": task_id,
            "task_name": task_name,
            "image_path": image_rel_path,
            "mask_path": mask_path,
            "original_size": (h0, w0),
            "index": int(idx),
        }


def inference_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    images = torch.stack([item["image"] for item in batch], 0)
    task_ids = [item["task_id"] for item in batch]
    task_names = [item["task_name"] for item in batch]
    image_paths = [item["image_path"] for item in batch]
    mask_paths = [item["mask_path"] for item in batch]
    original_sizes = [item["original_size"] for item in batch]
    indices = [item["index"] for item in batch]
    return {
        "image": images,
        "task_id": task_ids,
        "task_name": task_names,
        "image_path": image_paths,
        "mask_path": mask_paths,
        "original_size": original_sizes,
        "index": indices,
    }


def _save_segmentation(pred_logits: torch.Tensor, image_path: str, mask_path: Optional[str], output_dir: str, original_size: Tuple[int, int]):
    logits = pred_logits.detach().float().cpu().numpy()
    if logits.ndim == 3:
        mask = np.argmax(logits, axis=0).astype(np.uint8)
    else:
        mask = logits.astype(np.uint8)

    h, w = original_size
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    if mask_path is not None and str(mask_path).strip() != "":
        mask_path_clean = str(mask_path).replace("../", "")
        out_path = os.path.join(output_dir, mask_path_clean)
    else:
        default_mask_path = str(image_path).replace("img", "mask").replace("IMG", "MASK")
        default_mask_path = default_mask_path.replace("../", "")
        out_path = os.path.join(output_dir, default_mask_path)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, mask)


def _process_classification(pred_logits: torch.Tensor, task_id: str, image_path: str) -> Dict[str, Any]:
    x = pred_logits.detach().float().cpu().numpy()
    pred_class = int(np.argmax(x))
    x = x - np.max(x)
    ex = np.exp(x)
    probs = ex / (np.sum(ex) + 1e-12)
    return {
        "image_path": image_path,
        "task_id": task_id,
        "predicted_class": pred_class,
        "predicted_probs": probs.tolist(),
    }


def _process_regression(pred_vec: torch.Tensor, task_id: str, image_path: str, original_size: Tuple[int, int]) -> Dict[str, Any]:
    coords = pred_vec.detach().float().cpu().numpy().reshape(-1).tolist()
    h, w = original_size
    pixel_coords: List[float] = []
    for i in range(0, len(coords), 2):
        x_norm, y_norm = float(coords[i]), float(coords[i + 1])
        pixel_coords.extend([x_norm * w, y_norm * h])
    return {
        "image_path": image_path,
        "task_id": task_id,
        "predicted_points_normalized": coords,
        "predicted_points_pixels": pixel_coords,
    }


def _process_detection(det_map: torch.Tensor, task_id: str, image_path: str, original_size: Tuple[int, int]) -> Dict[str, Any]:
    x = det_map.detach().float().cpu().numpy()
    _, h, w = x.shape
    scores = x[4, :, :].reshape(-1)
    best_idx = int(np.argmax(scores))
    best_h = best_idx // w
    best_w = best_idx % w
    bbox_norm = x[:4, best_h, best_w].tolist()
    img_h, img_w = original_size
    bbox_pixel = [
        float(bbox_norm[0]) * img_w,
        float(bbox_norm[1]) * img_h,
        float(bbox_norm[2]) * img_w,
        float(bbox_norm[3]) * img_h,
    ]
    return {
        "image_path": image_path,
        "task_id": task_id,
        "bbox_normalized": bbox_norm,
        "bbox_pixels": bbox_pixel,
    }


@torch.no_grad()
def run_inference(
    model: MultiTaskModel,
    dataset: InferenceDataset,
    output_dir: str,
    batch_size: int,
    device: torch.device,
    precision: str,
):
    os.makedirs(output_dir, exist_ok=True)

    dataloader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=inference_collate_fn,
    )

    rows_by_task: Dict[str, int] = {}
    task_name_map = dataset.dataframe.groupby("task_id")["task_name"].first().to_dict() if "task_name" in dataset.dataframe.columns else {}
    for tid in dataset.dataframe["task_id"].tolist():
        rows_by_task[str(tid)] = rows_by_task.get(str(tid), 0) + 1

    print("[Task Counts]")
    for tid in sorted(rows_by_task.keys()):
        tname = task_name_map.get(tid, "")
        if tname:
            print(f"{tid}: {rows_by_task[tid]} ({tname})")
        else:
            print(f"{tid}: {rows_by_task[tid]}")

    use_amp = (device.type == "cuda") and (str(precision).lower() in ("bf16", "fp16"))
    amp_dtype = _get_amp_dtype(precision)
    ctx = torch.autocast(device_type="cuda", dtype=amp_dtype) if use_amp else nullcontext()

    classification_results: List[Dict[str, Any]] = []
    detection_results: List[Dict[str, Any]] = []
    regression_results: List[Dict[str, Any]] = []

    model.eval()

    for batch in tqdm(dataloader, desc="Prediction progress"):
        images = batch["image"].to(device, non_blocking=True)
        task_ids = batch["task_id"]
        task_names = batch["task_name"]
        image_paths = batch["image_path"]
        mask_paths = batch["mask_path"]
        original_sizes = batch["original_size"]

        unique_tasks = list(set(task_ids))

        for task_id in unique_tasks:
            task_indices = [i for i, tid in enumerate(task_ids) if tid == task_id]
            task_images = images[task_indices]
            task_name = task_names[task_indices[0]]

            with ctx:
                outputs, _ = model(task_images, task_id=str(task_id), labels=None)

            if task_name == "segmentation":
                logits = outputs["seg_logits"]
                for j, batch_idx in enumerate(task_indices):
                    _save_segmentation(
                        pred_logits=logits[j],
                        image_path=image_paths[batch_idx],
                        mask_path=mask_paths[batch_idx],
                        output_dir=output_dir,
                        original_size=original_sizes[batch_idx],
                    )

            elif task_name == "classification":
                logits = outputs["cls_logits"]
                for j, batch_idx in enumerate(task_indices):
                    classification_results.append(
                        _process_classification(
                            pred_logits=logits[j],
                            task_id=str(task_id),
                            image_path=image_paths[batch_idx],
                        )
                    )

            elif task_name == "Regression":
                pred = outputs["reg_pred"]
                for j, batch_idx in enumerate(task_indices):
                    regression_results.append(
                        _process_regression(
                            pred_vec=pred[j],
                            task_id=str(task_id),
                            image_path=image_paths[batch_idx],
                            original_size=original_sizes[batch_idx],
                        )
                    )

            elif task_name == "detection":
                det_map = outputs["det_map"]
                for j, batch_idx in enumerate(task_indices):
                    detection_results.append(
                        _process_detection(
                            det_map=det_map[j],
                            task_id=str(task_id),
                            image_path=image_paths[batch_idx],
                            original_size=original_sizes[batch_idx],
                        )
                    )

    if classification_results:
        with open(os.path.join(output_dir, "classification_predictions.json"), "w", encoding="utf-8") as f:
            json.dump(classification_results, f, indent=2, ensure_ascii=False)

    if detection_results:
        with open(os.path.join(output_dir, "detection_predictions.json"), "w", encoding="utf-8") as f:
            json.dump(detection_results, f, indent=2, ensure_ascii=False)

    if regression_results:
        with open(os.path.join(output_dir, "regression_predictions.json"), "w", encoding="utf-8") as f:
            json.dump(regression_results, f, indent=2, ensure_ascii=False)


def main():
    import argparse
    import subprocess

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    args = parser.parse_args()

    device = torch.device(LOAD_DEVICE if torch.cuda.is_available() else "cpu")
    print(f"[Info] Device: {device}")

    data_root = resolve_data_root(args.data_root)
    print(f"[Info] Resolved data root: {data_root}")

    transforms = build_transforms(FORCE_BACKBONE_IMAGE_SIZE)
    dataset = InferenceDataset(data_root=data_root, transforms=transforms)

    df = dataset.dataframe
    task_configs = _build_task_configs_from_dataframe(df)

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

    print("`torch_dtype` is deprecated! Use `dtype` instead!")
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
    if unexpected:
        print("[Info] Unexpected keys list:")
        for k in unexpected:
            print(k)

    run_inference(
        model=model,
        dataset=dataset,
        output_dir=str(args.output_dir),
        batch_size=int(args.batch_size),
        device=device,
        precision=str(PRECISION),
    )

    subprocess.check_call(["bash", "-lc", f"cd {args.output_dir} && zip -r predictions.zip ."])
    print(os.path.join(args.output_dir, "predictions.zip"))


if __name__ == "__main__":
    main()
