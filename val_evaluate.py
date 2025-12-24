from collections import defaultdict
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm


def calculate_accuracy(y_true, y_pred_logits):
    y_pred = torch.argmax(y_pred_logits, dim=1).detach().cpu().numpy()
    y_true = y_true.detach().cpu().numpy()
    return float(accuracy_score(y_true, y_pred))


def calculate_f1_score(y_true, y_pred_logits):
    y_pred = torch.argmax(y_pred_logits, dim=1).detach().cpu().numpy()
    y_true = y_true.detach().cpu().numpy()
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def calculate_dice_coefficient(y_true, y_pred_logits):
    y_pred_mask = torch.argmax(y_pred_logits, dim=1)
    num_classes = int(y_pred_logits.shape[1])
    y_true_one_hot = F.one_hot(y_true.long(), num_classes=num_classes).permute(0, 3, 1, 2).float()
    y_pred_one_hot = F.one_hot(y_pred_mask.long(), num_classes=num_classes).permute(0, 3, 1, 2).float()
    inter = torch.sum(y_true_one_hot[:, 1:] * y_pred_one_hot[:, 1:])
    union = torch.sum(y_true_one_hot[:, 1:]) + torch.sum(y_pred_one_hot[:, 1:])
    dice = (2.0 * inter + 1e-6) / (union + 1e-6)
    return float(dice.item())


def calculate_mae(y_true, y_pred, image_size):
    h, w = int(image_size[0]), int(image_size[1])
    y_true_px = y_true.detach().cpu().numpy().copy()
    y_pred_px = y_pred.detach().cpu().numpy().copy()
    y_true_px[:, 0::2] *= w
    y_true_px[:, 1::2] *= h
    y_pred_px[:, 0::2] *= w
    y_pred_px[:, 1::2] *= h
    return float(np.mean(np.abs(y_true_px - y_pred_px)))


def calculate_iou(y_true, y_pred):
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy()
    batch_ious = []
    for i in range(y_true.shape[0]):
        box_true = y_true[i]
        box_pred = y_pred[i]
        if np.any(box_true < 0) or np.any(box_pred < 0):
            continue
        xA = max(box_true[0], box_pred[0])
        yA = max(box_true[1], box_pred[1])
        xB = min(box_true[2], box_pred[2])
        yB = min(box_true[3], box_pred[3])
        inter_area = max(0.0, xB - xA) * max(0.0, yB - yA)
        box_true_area = max(0.0, (box_true[2] - box_true[0])) * max(0.0, (box_true[3] - box_true[1]))
        box_pred_area = max(0.0, (box_pred[2] - box_pred[0])) * max(0.0, (box_pred[3] - box_pred[1]))
        union_area = box_true_area + box_pred_area - inter_area
        iou = inter_area / (union_area + 1e-6)
        batch_ious.append(float(iou))
    if not batch_ious:
        return float("nan")
    return float(np.mean(batch_ious))


@torch.no_grad()
def evaluate(model, val_loader, device):
    model.eval()
    task_metrics = defaultdict(lambda: defaultdict(list))
    task_id_to_cfg = model.task_configs

    loop = tqdm(val_loader, desc="[Validation]", dynamic_ncols=True)
    for batch in loop:
        images = batch["images"].to(device, non_blocking=True)
        labels = batch["labels"]
        task_ids = batch["task_ids"]

        unique_tasks = set(task_ids)

        for task_id in unique_tasks:
            task_indices = [i for i, t_id in enumerate(task_ids) if t_id == task_id]
            task_images = images[task_indices]
            task_labels_list = [labels[i] for i in task_indices]
            task_labels = torch.stack(task_labels_list, 0).to(device, non_blocking=True)

            outputs, _ = model(task_images, task_id=task_id, labels=None)
            task_name = task_id_to_cfg[task_id]["task_name"]

            if task_name == "classification":
                logits = outputs["cls_logits"][:, : int(task_id_to_cfg[task_id]["num_classes"])].float()
                task_metrics[task_id]["Accuracy"].append(calculate_accuracy(task_labels, logits))
                task_metrics[task_id]["F1-Score"].append(calculate_f1_score(task_labels, logits))

            elif task_name == "segmentation":
                logits = outputs["seg_logits"][:, : int(task_id_to_cfg[task_id]["num_classes"])].float()
                task_metrics[task_id]["Dice"].append(calculate_dice_coefficient(task_labels, logits))

            elif task_name == "Regression":
                pred = outputs["reg_pred"][:, : (2 * int(task_id_to_cfg[task_id]["num_classes"]))].float()
                h, w = task_images.shape[-2], task_images.shape[-1]
                task_metrics[task_id]["MAE (pixels)"].append(calculate_mae(task_labels, pred, (h, w)))

            elif task_name == "detection":
                det_map = outputs["det_map"].float()
                batch_size, _, h, w = det_map.shape
                scores = det_map[:, 4, :, :].view(batch_size, -1)
                _, best_indices = torch.max(scores, dim=1)
                best_h = best_indices // w
                best_w = best_indices % w
                final_boxes = torch.zeros((batch_size, 4), device=device, dtype=torch.float32)
                for i in range(batch_size):
                    final_boxes[i] = det_map[i, :4, best_h[i], best_w[i]]
                task_metrics[task_id]["IoU"].append(calculate_iou(task_labels, final_boxes))

    rows = []
    sorted_task_ids = sorted(list(task_id_to_cfg.keys()))
    for task_id in sorted_task_ids:
        task_name = task_id_to_cfg[task_id]["task_name"]
        row = {"Task ID": task_id, "Task Name": task_name, "MAE (pixels)": np.nan, "Accuracy": np.nan, "F1-Score": np.nan, "Dice": np.nan, "IoU": np.nan}
        if task_id in task_metrics:
            for k, v in task_metrics[task_id].items():
                vals = [x for x in v if x is not None and np.isfinite(x)]
                row[k] = float(np.mean(vals)) if vals else np.nan
        rows.append(row)

    return pd.DataFrame(rows, columns=["Task ID", "Task Name", "MAE (pixels)", "Accuracy", "F1-Score", "Dice", "IoU"])
