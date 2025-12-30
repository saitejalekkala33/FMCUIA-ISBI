import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from collections import defaultdict
from typing import Dict, Optional
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

from dataset import MultiTaskDataset
from models.multitask_model import MultiTaskModel
from utils.data import build_task_configs_from_dataframe, multitask_collate_fn
from utils.misc import set_seed, get_autocast_context, get_amp_dtype, save_checkpoint, load_checkpoint
from val_evaluate import evaluate

DATA_ROOT_PATH = "/root/train/train"
VAL_SPLIT = 0.2
FORCE_BACKBONE_IMAGE_SIZE = 448

EPOCHS = 20
BATCH_SIZE = 1
GRAD_ACCUM = 16

HEAD_LR = 2e-4
BACKBONE_LR = 2e-5
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0

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

OUTPUT_DIR = "outputs"
SAVE_NAME_BEST = "best_model.pth"
SAVE_NAME_LAST = "last_model.pth"
RESUME_PATH = None

FREEZE_BACKBONE_EPOCHS = 0

EMA_BETA = 0.98
DYN_WEIGHT_GAMMA = 0.50
DYN_WEIGHT_CLAMP = (0.25, 4.0)

SAMPLE_TEMP = 0.70
SAMPLE_LOSS_BETA = 1.00
SAMPLE_COUNT_TAU = 0.50
STEPS_PER_EPOCH = None

SAMPLE_MIN_WEIGHT = 1.0
SAMPLE_MAX_WEIGHT = 1e6
LOSS_SCALE_FOR_SAMPLING = {
    "regression": 300.0,
    "Regression": 300.0,
    "segmentation": 1.0,
    "classification": 1.0,
    "detection": 1.0,
}

EARLY_STOPPING_PATIENCE = 3


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


def set_backbone_trainable(model: torch.nn.Module, trainable: bool):
    for p in model.backbone.parameters():
        p.requires_grad = bool(trainable)


def build_optimizer(model: torch.nn.Module):
    head_params = []
    backbone_params = []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n.startswith("backbone."):
            backbone_params.append(p)
        else:
            head_params.append(p)
    param_groups = []
    if head_params:
        param_groups.append({"params": head_params, "lr": float(HEAD_LR), "weight_decay": float(WEIGHT_DECAY)})
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": float(BACKBONE_LR), "weight_decay": float(WEIGHT_DECAY)})
    try:
        from transformers.optimization import Adafactor
        opt = Adafactor(param_groups, scale_parameter=False, relative_step=False, warmup_init=False)
        return opt
    except Exception:
        return torch.optim.AdamW(param_groups)


def make_stratified_split(df, val_split: float, data_frac: float, seed: int):
    rng = np.random.default_rng(seed)
    train_idx = []
    val_idx = []
    groups = df.groupby("task_id").indices
    for _, idxs in groups.items():
        idxs = np.array(list(idxs), dtype=np.int64)
        rng.shuffle(idxs)
        if data_frac < 1.0:
            k = int(np.ceil(len(idxs) * data_frac))
            k = max(1, k)
            idxs = idxs[:k]
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


class MultiTaskTemperatureSampler(torch.utils.data.Sampler):
    def __init__(
        self,
        subset,
        batch_size: int = 1,
        steps_per_epoch: int = None,
        seed: int = 42,
        temperature: float = 1.0,
        loss_beta: float = 1.0,
        count_tau: float = 0.5,
        min_weight: float = 1.0,
        max_weight: float = 1e6,
    ):
        self.subset = subset
        self.batch_size = int(batch_size)
        self.steps_per_epoch = steps_per_epoch
        self.seed = int(seed)
        self.temperature = float(temperature)
        self.loss_beta = float(loss_beta)
        self.count_tau = float(count_tau)
        self.min_weight = float(min_weight)
        self.max_weight = float(max_weight)

        df = getattr(subset, "dataframe", None)
        if df is None:
            raise RuntimeError("subset.dataframe is required")
        self.df = df

        self.task_ids = sorted(df["task_id"].unique().tolist())
        self.task_to_indices = {}
        for tid in self.task_ids:
            idxs = df.index[df["task_id"] == tid].to_numpy(dtype=np.int64)
            self.task_to_indices[tid] = idxs

        self.counts = {tid: float(len(self.task_to_indices[tid])) for tid in self.task_ids}

        self.task_to_name = {}
        if "task_name" in df.columns:
            tmp = df.groupby("task_id")["task_name"].first()
            for tid in self.task_ids:
                self.task_to_name[tid] = str(tmp.get(tid, "classification"))

        self._rng = np.random.default_rng(self.seed)
        self._task_probs = self._compute_probs(None)

    def __len__(self):
        if self.steps_per_epoch is not None:
            return int(self.steps_per_epoch)
        return max(1, int(np.ceil(len(self.df) / max(1, self.batch_size))))

    def _compute_probs(self, task_loss_ema: Optional[Dict[str, float]]):
        eps = 1e-8
        w = []
        for tid in self.task_ids:
            c = self.counts.get(tid, 1.0)
            cw = (c + eps) ** self.count_tau

            lw = 1.0
            if task_loss_ema is not None and tid in task_loss_ema:
                loss_val = float(task_loss_ema[tid])
                tname = self.task_to_name.get(tid, "classification")
                scale = float(LOSS_SCALE_FOR_SAMPLING.get(tname, LOSS_SCALE_FOR_SAMPLING.get(str(tname).lower(), 1.0)))
                loss_eff = max(loss_val * scale, 0.0)
                lw = (loss_eff + eps) ** self.loss_beta

            wt = cw * lw
            wt = float(np.clip(wt, 0.0, self.max_weight))
            wt = wt + self.min_weight
            w.append(wt)

        w = np.asarray(w, dtype=np.float64)
        w = np.maximum(w, eps)

        t = max(self.temperature, 1e-6)
        w = w ** (1.0 / t)

        s = float(w.sum())
        if not np.isfinite(s) or s <= 0:
            w = np.ones_like(w, dtype=np.float64) / float(len(w))
        else:
            w = w / s

        return w

    def update_from_losses(self, task_loss_ema: Dict[str, float]):
        self._task_probs = self._compute_probs(task_loss_ema)

    def __iter__(self):
        n_steps = len(self)
        for _ in range(n_steps):
            tid = self._rng.choice(self.task_ids, p=self._task_probs)
            idxs = self.task_to_indices[tid]
            if len(idxs) == 0:
                continue
            if self.batch_size == 1:
                yield [int(self._rng.choice(idxs))]
            else:
                replace = len(idxs) < self.batch_size
                batch = self._rng.choice(idxs, size=self.batch_size, replace=replace).tolist()
                yield [int(x) for x in batch]


def main():
    import argparse
    parser = argparse.ArgumentParser("FMC_UIA Qwen2.5-VL Multi-task Trainer")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--data_frac", type=float, default=1.0)
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    set_seed(SEED, deterministic=DETERMINISTIC)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    device = torch.device(LOAD_DEVICE if torch.cuda.is_available() else "cpu")
    print(f"[Info] Device: {device}")
    print(f"[Info] Using data root setting: {DATA_ROOT_PATH}")
    data_root = resolve_data_root(DATA_ROOT_PATH)
    print(f"[Info] Resolved data root: {data_root}")

    train_tfms = build_transforms(is_train=True, image_size=FORCE_BACKBONE_IMAGE_SIZE, deterministic=DETERMINISTIC)
    val_tfms = build_transforms(is_train=False, image_size=FORCE_BACKBONE_IMAGE_SIZE, deterministic=True)

    full_dataset_for_index = MultiTaskDataset(data_root, transforms=None)
    df = full_dataset_for_index.dataframe
    train_indices, val_indices = make_stratified_split(df, val_split=VAL_SPLIT, data_frac=float(args.data_frac), seed=SEED)

    train_dataset = MultiTaskDataset(data_root, transforms=train_tfms)
    val_dataset = MultiTaskDataset(data_root, transforms=val_tfms)

    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)

    train_subset.dataframe = train_dataset.dataframe.iloc[train_indices].reset_index(drop=True)
    val_subset.dataframe = val_dataset.dataframe.iloc[val_indices].reset_index(drop=True)

    print(f"[Info] Dataset split: {len(train_indices)} train samples, {len(val_indices)} val samples (data_frac={args.data_frac})")

    task_configs = build_task_configs_from_dataframe(train_dataset.dataframe)

    if PRECISION == "bf16":
        backbone_dtype = torch.bfloat16
    elif PRECISION == "fp16":
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

    start_epoch = 0
    best_score = -1e9

    optimizer = build_optimizer(model)
    if RESUME_PATH is not None and os.path.isfile(RESUME_PATH):
        start_epoch, best_score = load_checkpoint(RESUME_PATH, model, optimizer, map_location=device)
        print(f"[Resume] Loaded: {RESUME_PATH} (start_epoch={start_epoch}, best_score={best_score:.4f})")

    batch_sampler = MultiTaskTemperatureSampler(
        train_subset,
        batch_size=BATCH_SIZE,
        steps_per_epoch=STEPS_PER_EPOCH,
        seed=SEED,
        temperature=SAMPLE_TEMP,
        loss_beta=SAMPLE_LOSS_BETA,
        count_tau=SAMPLE_COUNT_TAU,
        min_weight=SAMPLE_MIN_WEIGHT,
        max_weight=SAMPLE_MAX_WEIGHT,
    )

    train_loader = DataLoader(
        train_subset,
        batch_sampler=batch_sampler,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        collate_fn=multitask_collate_fn,
        persistent_workers=(NUM_WORKERS > 0),
        prefetch_factor=4 if NUM_WORKERS > 0 else None,
    )

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

    amp_dtype = get_amp_dtype(PRECISION)
    use_amp = (device.type == "cuda") and (PRECISION in ["fp16", "bf16"])
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and PRECISION == "fp16"))

    ema_loss_by_task = {}
    ema_mean = 1.0
    es_bad = 0

    for epoch in range(start_epoch, EPOCHS):
        if epoch < int(FREEZE_BACKBONE_EPOCHS):
            set_backbone_trainable(model, False)
        else:
            set_backbone_trainable(model, True)

        model.train()
        epoch_train_losses = defaultdict(list)
        optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]", dynamic_ncols=True)
        step_in_epoch = 0

        for step, batch in enumerate(pbar):
            step_in_epoch += 1
            images = batch["images"].to(device, non_blocking=True)
            labels_list = batch["labels"]
            task_ids = batch["task_ids"]

            idx_by_task = defaultdict(list)
            for i, tid in enumerate(task_ids):
                idx_by_task[tid].append(i)

            total_loss_batch = torch.zeros((), device=device)

            with get_autocast_context(enabled=use_amp, dtype=amp_dtype, device_type=device.type):
                for tid, idxs in idx_by_task.items():
                    cfg = task_configs[tid]
                    task_name = cfg["task_name"]
                    imgs_g = images[idxs]
                    lbls_g = [labels_list[i] for i in idxs]
                    targets = torch.stack(lbls_g, dim=0).to(device, non_blocking=True)

                    _, loss_dict = model(imgs_g, task_id=tid, labels=targets)
                    base_loss = loss_dict["total_loss"]

                    cur = float(base_loss.detach().cpu().item())
                    if tid not in ema_loss_by_task:
                        ema_loss_by_task[tid] = cur
                    else:
                        ema_loss_by_task[tid] = EMA_BETA * float(ema_loss_by_task[tid]) + (1.0 - EMA_BETA) * cur

                    ema_vals = np.asarray(list(ema_loss_by_task.values()), dtype=np.float64)
                    ema_mean = float(np.mean(ema_vals)) if ema_vals.size > 0 else 1.0
                    denom = max(ema_mean, 1e-6)

                    if task_name in ("classification", "detection"):
                        dyn_w = (float(ema_loss_by_task[tid]) / denom) ** float(DYN_WEIGHT_GAMMA)
                        dyn_w = float(np.clip(dyn_w, DYN_WEIGHT_CLAMP[0], DYN_WEIGHT_CLAMP[1]))
                    else:
                        dyn_w = 1.0

                    scaled_loss = base_loss * float(dyn_w)

                    if task_name in ("classification", "detection"):
                        weighted_loss = model.loss_balancer(task_name, scaled_loss)
                    else:
                        weighted_loss = scaled_loss

                    epoch_train_losses[tid].append(float(weighted_loss.detach().cpu().item()))
                    total_loss_batch = total_loss_batch + weighted_loss

            loss_to_backprop = total_loss_batch / max(1, GRAD_ACCUM)

            if scaler.is_enabled():
                scaler.scale(loss_to_backprop).backward()
            else:
                loss_to_backprop.backward()

            if (step + 1) % GRAD_ACCUM == 0:
                if MAX_GRAD_NORM is not None and MAX_GRAD_NORM > 0:
                    if scaler.is_enabled():
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            pbar.set_postfix({"loss": float(total_loss_batch.detach().cpu().item())})

            if args.dry_run:
                break

        if (step_in_epoch % GRAD_ACCUM) != 0:
            if MAX_GRAD_NORM is not None and MAX_GRAD_NORM > 0:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        pbar.close()

        try:
            batch_sampler.update_from_losses(ema_loss_by_task)
        except Exception:
            pass

        print("\n--- Epoch {} Average Train Loss Report ---".format(epoch + 1))
        sorted_task_ids = sorted(epoch_train_losses.keys())
        for task_id in sorted_task_ids:
            avg_loss = np.mean(epoch_train_losses[task_id])
            print(f"  - Task '{task_id:<25}': {avg_loss:.4f}")
        print("-" * 40)

        val_results_df = evaluate(model, val_loader, device)
        score_cols = [col for col in val_results_df.columns if "MAE" not in col and isinstance(val_results_df[col].iloc[0], (int, float, np.floating, np.integer))]
        avg_val_score = 0.0
        if not val_results_df.empty and score_cols:
            avg_val_score = float(val_results_df[score_cols].mean().mean())

        print("\n--- Epoch {} Validation Report ---".format(epoch + 1))
        if not val_results_df.empty:
            print(val_results_df.to_string(index=False))
        print(f"--- Average Val Score (Higher is better): {avg_val_score:.4f} ---")

        last_path = os.path.join(OUTPUT_DIR, SAVE_NAME_LAST)
        save_checkpoint(last_path, epoch=epoch + 1, model=model, optimizer=optimizer, best_score=best_score)

        improved = avg_val_score > best_score
        if improved:
            best_score = float(avg_val_score)
            best_path = os.path.join(OUTPUT_DIR, SAVE_NAME_BEST)
            save_checkpoint(best_path, epoch=epoch + 1, model=model, optimizer=optimizer, best_score=best_score)
            print(f"[Checkpoint] New best saved: {best_path} (best_score={best_score:.4f})")
            es_bad = 0
        else:
            es_bad += 1
            print(f"[EarlyStopping] No improvement for {es_bad}/{EARLY_STOPPING_PATIENCE} epoch(s).")

        if es_bad >= int(EARLY_STOPPING_PATIENCE):
            print(f"[EarlyStopping] Patience {EARLY_STOPPING_PATIENCE} reached. Stopping training.")
            break

        if args.dry_run:
            print("[Dry Run] Completed 1 train step + 1 val pass. Exiting.")
            break


if __name__ == "__main__":
    main()




# regression neglected



# import os
# os.environ["TRANSFORMERS_NO_TF"] = "1"
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# from collections import defaultdict
# from typing import Dict, Optional
# import numpy as np
# import torch
# from torch.utils.data import DataLoader, Subset
# from tqdm import tqdm
# import albumentations as A
# from albumentations.pytorch import ToTensorV2

# from dataset import MultiTaskDataset
# from models.multitask_model import MultiTaskModel
# from utils.data import build_task_configs_from_dataframe, multitask_collate_fn
# from utils.misc import set_seed, get_autocast_context, get_amp_dtype, save_checkpoint, load_checkpoint
# from val_evaluate import evaluate

# DATA_ROOT_PATH = "/root/train/train"
# VAL_SPLIT = 0.2
# FORCE_BACKBONE_IMAGE_SIZE = 448

# EPOCHS = 20
# EARLY_STOP_PATIENCE = 3

# BATCH_SIZE = 1
# GRAD_ACCUM = 16

# HEAD_LR = 2e-4
# BACKBONE_LR = 2e-5
# WEIGHT_DECAY = 0.01
# MAX_GRAD_NORM = 1.0

# NUM_WORKERS = 4
# PIN_MEMORY = True

# SEED = 42
# DETERMINISTIC = False

# PRECISION = "bf16"
# GRAD_CHECKPOINT = True

# MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
# TRUST_REMOTE_CODE = True
# ATTN_IMPLEMENTATION = None

# KEEP_LLM = False
# LOAD_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# OUTPUT_DIR = "outputs"
# SAVE_NAME_BEST = "best_model.pth"
# SAVE_NAME_LAST = "last_model.pth"
# RESUME_PATH = None

# FREEZE_BACKBONE_EPOCHS = 0

# EMA_BETA = 0.98
# DYN_WEIGHT_GAMMA = 0.50
# DYN_WEIGHT_CLAMP = (0.25, 4.0)

# SAMPLE_TEMP = 0.70
# SAMPLE_LOSS_BETA = 1.00
# SAMPLE_COUNT_TAU = 0.50
# STEPS_PER_EPOCH = None


# def resolve_data_root(data_root: str) -> str:
#     candidates = [
#         data_root,
#         os.path.join(data_root, "train"),
#         os.path.join(data_root, "train", "train"),
#         os.path.join(data_root, "data", "train"),
#     ]
#     for c in candidates:
#         if os.path.isdir(os.path.join(c, "csv_files")):
#             return c
#     raise FileNotFoundError(
#         "CSV path not found.\nTried:\n"
#         + "\n".join([f"- {os.path.join(c, 'csv_files')}" for c in candidates])
#     )


# def build_transforms(is_train: bool, image_size: int, deterministic: bool) -> A.Compose:
#     t_list = [A.Resize(image_size, image_size)]
#     if is_train and not deterministic:
#         t_list += [
#             A.RandomBrightnessContrast(p=0.35),
#             A.GaussNoise(var_limit=(5.0, 45.0), p=0.30),
#             A.GaussianBlur(blur_limit=(3, 5), p=0.20),
#             A.MotionBlur(blur_limit=(3, 5), p=0.10),
#         ]
#     t_list += [ToTensorV2()]
#     return A.Compose(
#         t_list,
#         bbox_params=A.BboxParams(
#             format="pascal_voc",
#             label_fields=["class_labels"],
#             clip=True,
#             min_visibility=0.0,
#         ),
#     )


# def set_backbone_trainable(model: torch.nn.Module, trainable: bool):
#     for p in model.backbone.parameters():
#         p.requires_grad = bool(trainable)


# def build_optimizer(model: torch.nn.Module):
#     head_params = []
#     backbone_params = []
#     for n, p in model.named_parameters():
#         if not p.requires_grad:
#             continue
#         if n.startswith("backbone."):
#             backbone_params.append(p)
#         else:
#             head_params.append(p)
#     param_groups = []
#     if head_params:
#         param_groups.append({"params": head_params, "lr": float(HEAD_LR), "weight_decay": float(WEIGHT_DECAY)})
#     if backbone_params:
#         param_groups.append({"params": backbone_params, "lr": float(BACKBONE_LR), "weight_decay": float(WEIGHT_DECAY)})
#     try:
#         from transformers.optimization import Adafactor
#         opt = Adafactor(param_groups, scale_parameter=False, relative_step=False, warmup_init=False)
#         return opt
#     except Exception:
#         return torch.optim.AdamW(param_groups)


# def make_stratified_split(df, val_split: float, data_frac: float, seed: int):
#     rng = np.random.default_rng(seed)
#     train_idx = []
#     val_idx = []
#     groups = df.groupby("task_id").indices
#     for _, idxs in groups.items():
#         idxs = np.array(list(idxs), dtype=np.int64)
#         rng.shuffle(idxs)
#         if data_frac < 1.0:
#             k = int(np.ceil(len(idxs) * data_frac))
#             k = max(1, k)
#             idxs = idxs[:k]
#         v = int(np.floor(len(idxs) * val_split))
#         if v == 0 and val_split > 0 and len(idxs) > 1:
#             v = 1
#         if v >= len(idxs):
#             v = max(0, len(idxs) - 1)
#         val_idx.extend(idxs[:v].tolist())
#         train_idx.extend(idxs[v:].tolist())
#     rng.shuffle(train_idx)
#     rng.shuffle(val_idx)
#     return train_idx, val_idx


# class MultiTaskTemperatureSampler(torch.utils.data.Sampler):
#     def __init__(
#         self,
#         subset,
#         batch_size: int = 1,
#         steps_per_epoch: int = None,
#         seed: int = 42,
#         temperature: float = 1.0,
#         loss_beta: float = 1.0,
#         count_tau: float = 0.5,
#     ):
#         self.subset = subset
#         self.batch_size = int(batch_size)
#         self.steps_per_epoch = steps_per_epoch
#         self.seed = int(seed)
#         self.temperature = float(temperature)
#         self.loss_beta = float(loss_beta)
#         self.count_tau = float(count_tau)
#         df = getattr(subset, "dataframe", None)
#         if df is None:
#             raise RuntimeError("subset.dataframe is required")
#         self.df = df
#         self.task_ids = sorted(df["task_id"].unique().tolist())
#         self.task_to_indices = {}
#         for tid in self.task_ids:
#             idxs = df.index[df["task_id"] == tid].to_numpy(dtype=np.int64)
#             self.task_to_indices[tid] = idxs
#         self.counts = {tid: float(len(self.task_to_indices[tid])) for tid in self.task_ids}
#         self._rng = np.random.default_rng(self.seed)
#         self._task_probs = self._compute_probs(None)

#     def __len__(self):
#         if self.steps_per_epoch is not None:
#             return int(self.steps_per_epoch)
#         return max(1, int(np.ceil(len(self.df) / max(1, self.batch_size))))

#     def _compute_probs(self, task_loss_ema: Optional[Dict[str, float]]):
#         eps = 1e-8
#         w = []
#         for tid in self.task_ids:
#             c = self.counts.get(tid, 1.0)
#             cw = (c + eps) ** self.count_tau
#             lw = 1.0
#             if task_loss_ema is not None and tid in task_loss_ema:
#                 lw = (float(task_loss_ema[tid]) + eps) ** self.loss_beta
#             w.append(cw * lw)
#         w = np.asarray(w, dtype=np.float64)
#         w = np.maximum(w, eps)
#         t = max(self.temperature, 1e-6)
#         w = w ** (1.0 / t)
#         w = w / (w.sum() + eps)
#         return w

#     def update_from_losses(self, task_loss_ema: Dict[str, float]):
#         self._task_probs = self._compute_probs(task_loss_ema)

#     def __iter__(self):
#         n_steps = len(self)
#         for _ in range(n_steps):
#             tid = self._rng.choice(self.task_ids, p=self._task_probs)
#             idxs = self.task_to_indices[tid]
#             if len(idxs) == 0:
#                 continue
#             if self.batch_size == 1:
#                 yield [int(self._rng.choice(idxs))]
#             else:
#                 replace = len(idxs) < self.batch_size
#                 batch = self._rng.choice(idxs, size=self.batch_size, replace=replace).tolist()
#                 yield [int(x) for x in batch]


# def main():
#     import argparse
#     parser = argparse.ArgumentParser("FMC_UIA Qwen2.5-VL Multi-task Trainer")
#     parser.add_argument("--dry_run", action="store_true")
#     parser.add_argument("--data_frac", type=float, default=1.0)
#     args = parser.parse_args()

#     os.makedirs(OUTPUT_DIR, exist_ok=True)
#     set_seed(SEED, deterministic=DETERMINISTIC)

#     if torch.cuda.is_available():
#         torch.backends.cuda.matmul.allow_tf32 = True
#         torch.backends.cudnn.allow_tf32 = True

#     device = torch.device(LOAD_DEVICE if torch.cuda.is_available() else "cpu")
#     print(f"[Info] Device: {device}")
#     print(f"[Info] Using data root setting: {DATA_ROOT_PATH}")
#     data_root = resolve_data_root(DATA_ROOT_PATH)
#     print(f"[Info] Resolved data root: {data_root}")

#     train_tfms = build_transforms(is_train=True, image_size=FORCE_BACKBONE_IMAGE_SIZE, deterministic=DETERMINISTIC)
#     val_tfms = build_transforms(is_train=False, image_size=FORCE_BACKBONE_IMAGE_SIZE, deterministic=True)

#     full_dataset_for_index = MultiTaskDataset(data_root, transforms=None)
#     df = full_dataset_for_index.dataframe
#     train_indices, val_indices = make_stratified_split(df, val_split=VAL_SPLIT, data_frac=float(args.data_frac), seed=SEED)

#     train_dataset = MultiTaskDataset(data_root, transforms=train_tfms)
#     val_dataset = MultiTaskDataset(data_root, transforms=val_tfms)

#     train_subset = Subset(train_dataset, train_indices)
#     val_subset = Subset(val_dataset, val_indices)

#     train_subset.dataframe = train_dataset.dataframe.iloc[train_indices].reset_index(drop=True)
#     val_subset.dataframe = val_dataset.dataframe.iloc[val_indices].reset_index(drop=True)

#     print(f"[Info] Dataset split: {len(train_indices)} train samples, {len(val_indices)} val samples (data_frac={args.data_frac})")

#     task_configs = build_task_configs_from_dataframe(train_dataset.dataframe)

#     if PRECISION == "bf16":
#         backbone_dtype = torch.bfloat16
#     elif PRECISION == "fp16":
#         backbone_dtype = torch.float16
#     else:
#         backbone_dtype = torch.float32

#     model = MultiTaskModel(
#         task_configs=task_configs,
#         model_name=MODEL_NAME,
#         trust_remote_code=TRUST_REMOTE_CODE,
#         attn_implementation=ATTN_IMPLEMENTATION,
#         keep_llm=KEEP_LLM,
#         fpn_dim=256,
#         backbone_dtype=backbone_dtype,
#         load_device=LOAD_DEVICE,
#         force_image_size=FORCE_BACKBONE_IMAGE_SIZE,
#     )

#     if GRAD_CHECKPOINT:
#         model.enable_gradient_checkpointing()

#     model.to(device)

#     start_epoch = 0
#     best_score = -1e9
#     epochs_no_improve = 0

#     optimizer = build_optimizer(model)
#     if RESUME_PATH is not None and os.path.isfile(RESUME_PATH):
#         start_epoch, best_score = load_checkpoint(RESUME_PATH, model, optimizer, map_location=device)
#         print(f"[Resume] Loaded: {RESUME_PATH} (start_epoch={start_epoch}, best_score={best_score:.4f})")

#     batch_sampler = MultiTaskTemperatureSampler(
#         train_subset,
#         batch_size=BATCH_SIZE,
#         steps_per_epoch=STEPS_PER_EPOCH,
#         seed=SEED,
#         temperature=SAMPLE_TEMP,
#         loss_beta=SAMPLE_LOSS_BETA,
#         count_tau=SAMPLE_COUNT_TAU,
#     )

#     train_loader = DataLoader(
#         train_subset,
#         batch_sampler=batch_sampler,
#         num_workers=NUM_WORKERS,
#         pin_memory=PIN_MEMORY,
#         collate_fn=multitask_collate_fn,
#         persistent_workers=(NUM_WORKERS > 0),
#         prefetch_factor=4 if NUM_WORKERS > 0 else None,
#     )

#     val_loader = DataLoader(
#         val_subset,
#         batch_size=max(1, BATCH_SIZE),
#         shuffle=False,
#         num_workers=NUM_WORKERS,
#         pin_memory=PIN_MEMORY,
#         collate_fn=multitask_collate_fn,
#         persistent_workers=(NUM_WORKERS > 0),
#         prefetch_factor=4 if NUM_WORKERS > 0 else None,
#     )

#     amp_dtype = get_amp_dtype(PRECISION)
#     use_amp = (device.type == "cuda") and (PRECISION in ["fp16", "bf16"])
#     scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and PRECISION == "fp16"))

#     ema_loss_by_task = {}
#     ema_mean = 1.0

#     for epoch in range(start_epoch, EPOCHS):
#         if epoch < int(FREEZE_BACKBONE_EPOCHS):
#             set_backbone_trainable(model, False)
#         else:
#             set_backbone_trainable(model, True)

#         model.train()
#         epoch_train_losses = defaultdict(list)
#         optimizer.zero_grad(set_to_none=True)

#         pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]", dynamic_ncols=True)
#         step_in_epoch = 0

#         for step, batch in enumerate(pbar):
#             step_in_epoch += 1
#             images = batch["images"].to(device, non_blocking=True)
#             labels_list = batch["labels"]
#             task_ids = batch["task_ids"]

#             idx_by_task = defaultdict(list)
#             for i, tid in enumerate(task_ids):
#                 idx_by_task[tid].append(i)

#             total_loss_batch = torch.zeros((), device=device)

#             with get_autocast_context(enabled=use_amp, dtype=amp_dtype, device_type=device.type):
#                 for tid, idxs in idx_by_task.items():
#                     cfg = task_configs[tid]
#                     task_name = cfg["task_name"]
#                     imgs_g = images[idxs]
#                     lbls_g = [labels_list[i] for i in idxs]
#                     targets = torch.stack(lbls_g, dim=0).to(device, non_blocking=True)

#                     _, loss_dict = model(imgs_g, task_id=tid, labels=targets)
#                     base_loss = loss_dict["total_loss"]

#                     cur = float(base_loss.detach().cpu().item())
#                     if tid not in ema_loss_by_task:
#                         ema_loss_by_task[tid] = cur
#                     else:
#                         ema_loss_by_task[tid] = EMA_BETA * float(ema_loss_by_task[tid]) + (1.0 - EMA_BETA) * cur

#                     ema_vals = np.asarray(list(ema_loss_by_task.values()), dtype=np.float64)
#                     ema_mean = float(np.mean(ema_vals)) if ema_vals.size > 0 else 1.0
#                     denom = max(ema_mean, 1e-6)

#                     if task_name in ("classification", "detection"):
#                         dyn_w = (float(ema_loss_by_task[tid]) / denom) ** float(DYN_WEIGHT_GAMMA)
#                         dyn_w = float(np.clip(dyn_w, DYN_WEIGHT_CLAMP[0], DYN_WEIGHT_CLAMP[1]))
#                     else:
#                         dyn_w = 1.0

#                     scaled_loss = base_loss * float(dyn_w)

#                     if task_name in ("classification", "detection"):
#                         weighted_loss = model.loss_balancer(task_name, scaled_loss)
#                     else:
#                         weighted_loss = scaled_loss

#                     epoch_train_losses[tid].append(float(weighted_loss.detach().cpu().item()))
#                     total_loss_batch = total_loss_batch + weighted_loss

#             loss_to_backprop = total_loss_batch / max(1, GRAD_ACCUM)

#             if scaler.is_enabled():
#                 scaler.scale(loss_to_backprop).backward()
#             else:
#                 loss_to_backprop.backward()

#             if (step + 1) % GRAD_ACCUM == 0:
#                 if MAX_GRAD_NORM is not None and MAX_GRAD_NORM > 0:
#                     if scaler.is_enabled():
#                         scaler.unscale_(optimizer)
#                     torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
#                 if scaler.is_enabled():
#                     scaler.step(optimizer)
#                     scaler.update()
#                 else:
#                     optimizer.step()
#                 optimizer.zero_grad(set_to_none=True)

#             pbar.set_postfix({"loss": float(total_loss_batch.detach().cpu().item())})

#             if args.dry_run:
#                 break

#         if (step_in_epoch % GRAD_ACCUM) != 0:
#             if MAX_GRAD_NORM is not None and MAX_GRAD_NORM > 0:
#                 if scaler.is_enabled():
#                     scaler.unscale_(optimizer)
#                 torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
#             if scaler.is_enabled():
#                 scaler.step(optimizer)
#                 scaler.update()
#             else:
#                 optimizer.step()
#             optimizer.zero_grad(set_to_none=True)

#         pbar.close()

#         try:
#             batch_sampler.update_from_losses(ema_loss_by_task)
#         except Exception:
#             pass

#         print("\n--- Epoch {} Average Train Loss Report ---".format(epoch + 1))
#         sorted_task_ids = sorted(epoch_train_losses.keys())
#         for task_id in sorted_task_ids:
#             avg_loss = np.mean(epoch_train_losses[task_id])
#             print(f"  - Task '{task_id:<25}': {avg_loss:.4f}")
#         print("-" * 40)

#         val_results_df = evaluate(model, val_loader, device)
#         score_cols = [col for col in val_results_df.columns if "MAE" not in col and isinstance(val_results_df[col].iloc[0], (int, float, np.floating, np.integer))]
#         avg_val_score = 0.0
#         if not val_results_df.empty and score_cols:
#             avg_val_score = float(val_results_df[score_cols].mean().mean())

#         print("\n--- Epoch {} Validation Report ---".format(epoch + 1))
#         if not val_results_df.empty:
#             print(val_results_df.to_string(index=False))
#         print(f"--- Average Val Score (Higher is better): {avg_val_score:.4f} ---")

#         last_path = os.path.join(OUTPUT_DIR, SAVE_NAME_LAST)
#         save_checkpoint(last_path, epoch=epoch + 1, model=model, optimizer=optimizer, best_score=best_score)

#         if avg_val_score > best_score:
#             best_score = float(avg_val_score)
#             epochs_no_improve = 0
#             best_path = os.path.join(OUTPUT_DIR, SAVE_NAME_BEST)
#             save_checkpoint(best_path, epoch=epoch + 1, model=model, optimizer=optimizer, best_score=best_score)
#             print(f"[Checkpoint] New best saved: {best_path} (best_score={best_score:.4f})")
#         else:
#             epochs_no_improve += 1
#             if epochs_no_improve >= int(EARLY_STOP_PATIENCE):
#                 print(f"[Early Stopping] No improvement for {EARLY_STOP_PATIENCE} epochs. Stopping.")
#                 break

#         if args.dry_run:
#             print("[Dry Run] Completed 1 train step + 1 val pass. Exiting.")
#             break


# if __name__ == "__main__":
#     main()
