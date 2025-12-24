import os

os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

from dataset import MultiTaskDataset, MultiTaskUniformSampler
from models.multitask_model import MultiTaskModel
from utils.data import build_task_configs_from_dataframe, multitask_collate_fn
from utils.misc import set_seed, get_autocast_context, get_amp_dtype, save_checkpoint, load_checkpoint
from val_evaluate import evaluate

DATA_ROOT_PATH = "/root/train/train"
VAL_SPLIT = 0.2

FORCE_BACKBONE_IMAGE_SIZE = 448

EPOCHS = 10
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

W_SEG = 1.0
W_DET = 1.5
W_CLS = 1.0
W_REG = 1.0


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
            A.RandomBrightnessContrast(p=0.30),
            A.GaussNoise(var_limit=(5.0, 35.0), p=0.25),
            A.GaussianBlur(blur_limit=(3, 5), p=0.20),
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

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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

    if RESUME_PATH is not None and os.path.isfile(RESUME_PATH):
        optimizer = build_optimizer(model)
        start_epoch, best_score = load_checkpoint(RESUME_PATH, model, optimizer, map_location=device)
        print(f"[Resume] Loaded: {RESUME_PATH} (start_epoch={start_epoch}, best_score={best_score:.4f})")
    else:
        optimizer = build_optimizer(model)

    train_sampler = MultiTaskUniformSampler(train_subset, batch_size=BATCH_SIZE, steps_per_epoch=None)

    train_loader = DataLoader(
        train_subset,
        batch_sampler=train_sampler,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        collate_fn=multitask_collate_fn,
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=max(1, BATCH_SIZE),
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        collate_fn=multitask_collate_fn,
    )

    amp_dtype = get_amp_dtype(PRECISION)
    use_amp = (device.type == "cuda") and (PRECISION in ["fp16", "bf16"])
    scaler = torch.amp.GradScaler('cuda', enabled=(use_amp and PRECISION == "fp16"))

    task_type_weights = {
        "segmentation": W_SEG,
        "detection": W_DET,
        "classification": W_CLS,
        "Regression": W_REG,
    }

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

            total_loss_batch = 0.0

            with get_autocast_context(enabled=use_amp, dtype=amp_dtype, device_type=device.type):
                for tid, idxs in idx_by_task.items():
                    cfg = task_configs[tid]
                    task_name = cfg["task_name"]

                    imgs_g = images[idxs]
                    lbls_g = [labels_list[i] for i in idxs]
                    targets = torch.stack(lbls_g, dim=0).to(device, non_blocking=True)

                    _, loss_dict = model(imgs_g, task_id=tid, labels=targets)
                    loss = loss_dict["total_loss"]

                    weight = task_type_weights.get(task_name, 1.0)
                    weighted_loss = loss * float(weight)

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

        print("\n--- Epoch {} Average Train Loss Report ---".format(epoch + 1))
        sorted_task_ids = sorted(epoch_train_losses.keys())
        for task_id in sorted_task_ids:
            avg_loss = np.mean(epoch_train_losses[task_id])
            print(f"  - Task '{task_id:<25}': {avg_loss:.4f}")
        print("-" * 40)

        val_results_df = evaluate(model, val_loader, device)

        score_cols = [col for col in val_results_df.columns if "MAE" not in col and isinstance(val_results_df[col].iloc[0], (int, float, np.floating, np.integer))]
        avg_val_score = 0
        if not val_results_df.empty and score_cols:
            avg_val_score = float(val_results_df[score_cols].mean().mean())

        print("\n--- Epoch {} Validation Report ---".format(epoch + 1))
        if not val_results_df.empty:
            print(val_results_df.to_string(index=False))
        print(f"--- Average Val Score (Higher is better): {avg_val_score:.4f} ---")

        last_path = os.path.join(OUTPUT_DIR, SAVE_NAME_LAST)
        save_checkpoint(last_path, epoch=epoch + 1, model=model, optimizer=optimizer, best_score=best_score)

        if avg_val_score > best_score:
            best_score = float(avg_val_score)
            best_path = os.path.join(OUTPUT_DIR, SAVE_NAME_BEST)
            save_checkpoint(best_path, epoch=epoch + 1, model=model, optimizer=optimizer, best_score=best_score)
            print(f"[Checkpoint] New best saved: {best_path} (best_score={best_score:.4f})")

        if args.dry_run:
            print("[Dry Run] Completed 1 train step + 1 val pass. Exiting.")
            break


if __name__ == "__main__":
    main()
