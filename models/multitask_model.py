from typing import Dict, List, Optional
import torch
import torch.nn as nn

from models.backbone_qwen_vl import QwenVLBackbone
from models.heads import SegmentationHead, ClassificationHead, RegressionHead, DetectionGridHead
from losses import segmentation_loss, classification_loss, regression_loss, detection_grid_loss


class MultiTaskModel(nn.Module):
    def __init__(
        self,
        task_configs: Dict[str, Dict],
        model_name: str,
        trust_remote_code: bool = True,
        attn_implementation: Optional[str] = None,
        keep_llm: bool = False,
        fpn_dim: int = 256,
        backbone_dtype: Optional[torch.dtype] = None,
        load_device: str = "cuda:0",
        force_image_size: Optional[int] = None,
    ):
        super().__init__()
        self.task_configs = task_configs

        seg_classes = [cfg["num_classes"] for cfg in task_configs.values() if cfg["task_name"] == "segmentation"]
        cls_classes = [cfg["num_classes"] for cfg in task_configs.values() if cfg["task_name"] == "classification"]
        reg_points = [cfg["num_classes"] for cfg in task_configs.values() if cfg["task_name"] == "Regression"]

        self.max_seg_classes = max(seg_classes) if seg_classes else 2
        self.max_cls_classes = max(cls_classes) if cls_classes else 2
        self.max_reg_dim = max([2 * p for p in reg_points]) if reg_points else 4

        self.backbone = QwenVLBackbone(
            model_name=model_name,
            trust_remote_code=trust_remote_code,
            attn_implementation=attn_implementation,
            torch_dtype=backbone_dtype,
            keep_llm=keep_llm,
            fpn_dim=fpn_dim,
            num_hook_layers=4,
            load_device=load_device,
            force_image_size=force_image_size,
        )

        self.seg_head = SegmentationHead(fpn_dim=fpn_dim, max_classes=self.max_seg_classes)
        self.cls_head = ClassificationHead(max_classes=self.max_cls_classes)
        self.reg_head = RegressionHead(max_dim=self.max_reg_dim)
        self.det_head = DetectionGridHead(fpn_dim=fpn_dim, num_convs=4)

        if torch.cuda.is_available() and load_device.startswith("cuda"):
            if backbone_dtype is None:
                self.seg_head.to(load_device)
                self.cls_head.to(load_device)
                self.reg_head.to(load_device)
                self.det_head.to(load_device)
            else:
                self.seg_head.to(load_device, dtype=backbone_dtype)
                self.cls_head.to(load_device, dtype=backbone_dtype)
                self.reg_head.to(load_device, dtype=backbone_dtype)
                self.det_head.to(load_device, dtype=backbone_dtype)

    def enable_gradient_checkpointing(self):
        self.backbone.enable_gradient_checkpointing()

    def forward(self, images: torch.Tensor, task_id: str, labels: Optional[torch.Tensor] = None):
        cfg = self.task_configs[task_id]
        task_name = cfg["task_name"]
        num_classes = int(cfg["num_classes"])

        feats = self.backbone(images)
        fpn_feats: List[torch.Tensor] = feats["fpn"]
        global_feat: torch.Tensor = feats["global"]

        H_img, W_img = images.shape[-2], images.shape[-1]

        outputs: Dict = {}
        loss_dict: Dict[str, torch.Tensor] = {}

        if task_name == "segmentation":
            logits = self.seg_head(fpn_feats, out_size=(H_img, W_img))
            logits = logits[:, :num_classes, :, :]
            outputs["seg_logits"] = logits
            if labels is not None:
                loss, parts = segmentation_loss(logits, labels, num_classes=num_classes)
                loss_dict.update(parts)
                loss_dict["total_loss"] = loss

        elif task_name == "classification":
            logits = self.cls_head(global_feat)
            logits = logits[:, :num_classes]
            outputs["cls_logits"] = logits
            if labels is not None:
                loss, parts = classification_loss(logits, labels, num_classes=num_classes)
                loss_dict.update(parts)
                loss_dict["total_loss"] = loss

        elif task_name == "Regression":
            pred = self.reg_head(global_feat)
            out_dim = 2 * num_classes
            pred = pred[:, :out_dim].sigmoid()
            outputs["reg_pred"] = pred
            if labels is not None:
                loss, parts = regression_loss(pred, labels)
                loss_dict.update(parts)
                loss_dict["total_loss"] = loss

        elif task_name == "detection":
            det_out = self.det_head(fpn_feats)
            outputs.update(det_out)
            if labels is not None:
                loss, parts = detection_grid_loss(det_out["det_map"], labels)
                loss_dict.update(parts)
                loss_dict["total_loss"] = loss

        else:
            raise ValueError(f"Unknown task_name: {task_name}")

        if labels is None:
            loss_dict["total_loss"] = torch.tensor(0.0, device=images.device)

        return outputs, loss_dict
