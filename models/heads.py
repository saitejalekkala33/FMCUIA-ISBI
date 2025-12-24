from typing import Dict, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvGNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int = 1, groups: int = 32):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False)
        g = min(groups, out_ch)
        self.gn = nn.GroupNorm(g, out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.gn(self.conv(x)))


class UpFuseBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.refine = nn.Sequential(ConvGNAct(ch, ch, 3, 1, 1), ConvGNAct(ch, ch, 3, 1, 1))

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = x + skip
        x = self.refine(x)
        return x


class SegmentationHead(nn.Module):
    def __init__(self, fpn_dim: int, max_classes: int = 5, dropout: float = 0.1):
        super().__init__()
        self.max_classes = max_classes
        self.drop = nn.Dropout2d(dropout)
        self.p5_proj = nn.Sequential(ConvGNAct(fpn_dim, fpn_dim), ConvGNAct(fpn_dim, fpn_dim))
        self.up4 = UpFuseBlock(fpn_dim)
        self.up3 = UpFuseBlock(fpn_dim)
        self.up2 = UpFuseBlock(fpn_dim)
        self.classifier = nn.Conv2d(fpn_dim, max_classes, kernel_size=1)

    def forward(self, fpn_feats: List[torch.Tensor], out_size: Tuple[int, int]) -> torch.Tensor:
        p2, p3, p4, p5 = fpn_feats
        x = self.p5_proj(p5)
        x = self.up4(x, p4)
        x = self.up3(x, p3)
        x = self.up2(x, p2)
        x = self.drop(x)
        logits_low = self.classifier(x)
        logits = F.interpolate(logits_low, size=out_size, mode="bilinear", align_corners=False)
        return logits


class ClassificationHead(nn.Module):
    def __init__(self, max_classes: int, hidden: int = 1024, dropout: float = 0.2):
        super().__init__()
        self.max_classes = max_classes
        self.net = nn.Sequential(
            nn.LazyLinear(hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, max_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RegressionHead(nn.Module):
    def __init__(self, max_dim: int, hidden: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.max_dim = max_dim
        self.net = nn.Sequential(
            nn.LazyLinear(hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, max_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DetectionGridHead(nn.Module):
    def __init__(self, fpn_dim: int, num_convs: int = 4):
        super().__init__()
        tower = []
        for _ in range(num_convs):
            tower.append(ConvGNAct(fpn_dim, fpn_dim))
        self.tower = nn.Sequential(*tower)
        self.box_pred = nn.Conv2d(fpn_dim, 4, kernel_size=1)
        self.obj_pred = nn.Conv2d(fpn_dim, 1, kernel_size=1)

    def forward(self, fpn_feats: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        p2 = fpn_feats[0]
        f = self.tower(p2)

        obj_logits = self.obj_pred(f)
        box_raw = self.box_pred(f)

        box_sig = torch.sigmoid(box_raw).float()
        x1 = box_sig[:, 0:1]
        y1 = box_sig[:, 1:2]
        x2 = box_sig[:, 2:3]
        y2 = box_sig[:, 3:4]

        x1f = torch.minimum(x1, x2).clamp(0.0, 1.0)
        x2f = torch.maximum(x1, x2).clamp(0.0, 1.0)
        y1f = torch.minimum(y1, y2).clamp(0.0, 1.0)
        y2f = torch.maximum(y1, y2).clamp(0.0, 1.0)

        boxes = torch.cat([x1f, y1f, x2f, y2f], dim=1).float()
        det_map = torch.cat([boxes, obj_logits.float()], dim=1)
        return {"det_map": det_map}
