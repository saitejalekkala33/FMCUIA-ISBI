from typing import Dict, Tuple
import torch
import torch.nn.functional as F

def dice_loss_multiclass(logits: torch.Tensor, targets: torch.Tensor, num_classes: int, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1)
    tgt_oh = F.one_hot(targets.long(), num_classes=num_classes).permute(0, 3, 1, 2).float()
    inter = (probs * tgt_oh).sum(dim=(2, 3))
    union = probs.sum(dim=(2, 3)) + tgt_oh.sum(dim=(2, 3))
    dice = (2 * inter + eps) / (union + eps)
    if num_classes > 1:
        dice = dice[:, 1:]
    return 1.0 - dice.mean()


def segmentation_loss(logits: torch.Tensor, targets: torch.Tensor, num_classes: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    parts: Dict[str, torch.Tensor] = {}
    if num_classes <= 1:
        ce = F.cross_entropy(logits.float(), targets.long())
        parts["loss_ce"] = ce
        parts["loss_seg"] = ce
        return ce, parts
    w = torch.ones((num_classes,), device=logits.device, dtype=torch.float32)
    w[0] = 0.20
    ce = F.cross_entropy(logits.float(), targets.long(), weight=w)
    d = dice_loss_multiclass(logits.float(), targets, num_classes=num_classes)
    loss = ce + d
    parts["loss_ce"] = ce
    parts["loss_dice"] = d
    parts["loss_seg"] = loss
    return loss, parts


def classification_loss(logits: torch.Tensor, targets: torch.Tensor, num_classes: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    parts: Dict[str, torch.Tensor] = {}
    ce = F.cross_entropy(logits.float(), targets.long())
    parts["loss_ce"] = ce
    parts["loss_cls"] = ce
    return ce, parts


def regression_loss(pred: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    parts: Dict[str, torch.Tensor] = {}
    l = F.smooth_l1_loss(pred.float(), targets.float(), beta=1.0)
    parts["loss_reg"] = l
    return l, parts


def _iou_xyxy(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    xA = torch.maximum(a[:, 0], b[:, 0])
    yA = torch.maximum(a[:, 1], b[:, 1])
    xB = torch.minimum(a[:, 2], b[:, 2])
    yB = torch.minimum(a[:, 3], b[:, 3])

    inter = torch.clamp(xB - xA, min=0) * torch.clamp(yB - yA, min=0)
    area_a = torch.clamp(a[:, 2] - a[:, 0], min=0) * torch.clamp(a[:, 3] - a[:, 1], min=0)
    area_b = torch.clamp(b[:, 2] - b[:, 0], min=0) * torch.clamp(b[:, 3] - b[:, 1], min=0)
    union = area_a + area_b - inter
    return inter / (union + 1e-6)


def detection_grid_loss(det_map: torch.Tensor, gt_bbox: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    parts: Dict[str, torch.Tensor] = {}
    B, C, H, W = det_map.shape
    boxes = det_map[:, 0:4].float()
    obj_logits = det_map[:, 4:5].float()
    gt = gt_bbox.float()

    valid = (gt[:, 0] >= 0) & (gt[:, 1] >= 0) & (gt[:, 2] >= 0) & (gt[:, 3] >= 0)
    if valid.sum() == 0:
        z = (det_map.sum() * 0.0)
        parts["loss_det_obj"] = z
        parts["loss_det_box"] = z
        parts["loss_det_iou"] = z
        parts["loss_det"] = z
        return z, parts

    yy = (torch.arange(H, device=det_map.device, dtype=torch.float32) + 0.5) / float(H)
    xx = (torch.arange(W, device=det_map.device, dtype=torch.float32) + 0.5) / float(W)
    gy, gx = torch.meshgrid(yy, xx, indexing="ij")
    gx = gx.unsqueeze(0).expand(B, -1, -1)
    gy = gy.unsqueeze(0).expand(B, -1, -1)

    total_obj = torch.tensor(0.0, device=det_map.device)
    total_box = torch.tensor(0.0, device=det_map.device)
    total_iou = torch.tensor(0.0, device=det_map.device)
    n_valid = 0

    for b in range(B):
        if not bool(valid[b].item()):
            continue
        n_valid += 1
        x1, y1, x2, y2 = gt[b]
        x1, x2 = torch.minimum(x1, x2), torch.maximum(x1, x2)
        y1, y2 = torch.minimum(y1, y2), torch.maximum(y1, y2)

        pos = (gx[b] >= x1) & (gx[b] <= x2) & (gy[b] >= y1) & (gy[b] <= y2)
        tgt = pos.float().unsqueeze(0)

        pos_count = tgt.sum().clamp(min=1.0)
        neg_count = (1.0 - tgt).sum().clamp(min=1.0)
        pos_w = (neg_count / pos_count).clamp(min=1.0, max=50.0)
        pos_w = pos_w.view(1)

        obj = F.binary_cross_entropy_with_logits(obj_logits[b], tgt, pos_weight=pos_w)
        total_obj = total_obj + obj

        if pos.any():
            pb = boxes[b].permute(1, 2, 0)[pos].view(-1, 4)
            gb = gt[b].view(1, 4).expand(pb.shape[0], 4)
            box_l1 = F.smooth_l1_loss(pb, gb, beta=0.05)
            iou = _iou_xyxy(pb, gb)
            iou_l = (1.0 - iou).mean()
            total_box = total_box + box_l1
            total_iou = total_iou + iou_l
        else:
            total_box = total_box + (obj * 0.0)
            total_iou = total_iou + (obj * 0.0)

    denom = float(max(n_valid, 1))
    loss_obj = total_obj / denom
    loss_box = total_box / denom
    loss_iou = total_iou / denom

    total = loss_obj + 10.0 * loss_box + 5.0 * loss_iou

    parts["loss_det_obj"] = loss_obj
    parts["loss_det_box"] = loss_box
    parts["loss_det_iou"] = loss_iou
    parts["loss_det"] = total
    return total, parts
