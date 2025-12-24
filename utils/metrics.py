from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score


def dice_iou_from_masks(pred: torch.Tensor, gt: torch.Tensor, num_classes: int) -> Tuple[float, float]:
    """
    pred: (H,W) int
    gt: (H,W) int
    Compute mean Dice and IoU over non-background classes (1..C-1).
    """
    pred = pred.detach().cpu().numpy().astype(np.int64)
    gt = gt.detach().cpu().numpy().astype(np.int64)

    dices = []
    ious = []
    classes = list(range(1, num_classes)) if num_classes > 1 else [0]

    for c in classes:
        p = (pred == c).astype(np.uint8)
        g = (gt == c).astype(np.uint8)
        inter = (p & g).sum()
        union = p.sum() + g.sum()
        dice = 1.0 if union == 0 else (2.0 * inter) / (union + 1e-7)
        # IoU
        i = inter
        u = (p | g).sum()
        iou = 1.0 if u == 0 else float(i) / float(u + 1e-7)
        dices.append(float(dice))
        ious.append(float(iou))

    return float(np.mean(dices) if dices else 0.0), float(np.mean(ious) if ious else 0.0)


def classification_metrics(y_true: List[int], y_prob: np.ndarray, num_classes: int) -> Dict[str, float]:
    """
    y_prob: (N,C) softmax probs
    """
    y_true = np.array(y_true, dtype=np.int64)
    y_pred = np.argmax(y_prob, axis=1)

    acc = float(accuracy_score(y_true, y_pred))
    f1m = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    auc = 0.0
    try:
        if num_classes == 2:
            auc = float(roc_auc_score(y_true, y_prob[:, 1]))
        elif num_classes > 2:
            auc = float(roc_auc_score(y_true, y_prob, average="macro", multi_class="ovr"))
    except Exception:
        auc = 0.0

    return {"AUC": auc, "Accuracy": acc, "F1_macro": f1m}


def box_iou_xyxy_np(a: np.ndarray, b: np.ndarray) -> float:
    """
    a,b: (4,) xyxy in normalized or pixel coords
    """
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return float(inter / union)


def average_precision_from_scores(tp: np.ndarray, scores: np.ndarray, num_gt: int) -> float:
    """
    Compute AP from boolean tp and confidence scores (one prediction per image assumed).
    """
    if num_gt == 0:
        return 0.0
    # Sort by score desc
    order = np.argsort(-scores)
    tp = tp[order].astype(np.float32)
    fp = (1 - tp).astype(np.float32)

    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    recall = cum_tp / float(num_gt)
    precision = cum_tp / np.maximum(cum_tp + cum_fp, 1e-9)

    # Precision envelope
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    # Area under PR
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))
    return ap


def regression_mae(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - gt)))
