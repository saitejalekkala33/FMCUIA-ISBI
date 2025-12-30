from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class DynConv2d(nn.Module):
    def __init__(self, out_ch: int, k: int = 1, s: int = 1, p: int = 0, bias: bool = False, groups: int = 1, dilation: int = 1):
        super().__init__()
        self.out_ch = int(out_ch)
        self.k = int(k)
        self.s = int(s)
        self.p = int(p)
        self.bias = bool(bias)
        self.groups = int(groups)
        self.dilation = int(dilation)
        self.conv: Optional[nn.Conv2d] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.conv is None:
            in_ch = int(x.shape[1])
            self.conv = nn.Conv2d(
                in_ch,
                self.out_ch,
                kernel_size=self.k,
                stride=self.s,
                padding=self.p,
                bias=self.bias,
                groups=self.groups,
                dilation=self.dilation,
            ).to(device=x.device, dtype=x.dtype)
            nn.init.kaiming_normal_(self.conv.weight, mode="fan_out", nonlinearity="relu")
            if self.conv.bias is not None:
                nn.init.zeros_(self.conv.bias)
        else:
            if self.conv.weight.device != x.device:
                self.conv = self.conv.to(device=x.device)
            if self.conv.weight.dtype != x.dtype:
                self.conv = self.conv.to(dtype=x.dtype)
        return self.conv(x)


class SafeConv2d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int = 1, bias: bool = False, groups: int = 1, dilation: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=bias, groups=groups, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        wd = self.conv.weight.dtype
        if x.dtype != wd:
            x = x.to(dtype=wd)
        if self.conv.weight.device != x.device:
            self.conv = self.conv.to(device=x.device)
        return self.conv(x)


class SafeGroupNorm(nn.Module):
    def __init__(self, num_groups: int, num_channels: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.gn = nn.GroupNorm(num_groups, num_channels, eps=eps, affine=affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        wd = self.gn.weight.dtype if self.gn.weight is not None else x.dtype
        if x.dtype != wd:
            x = x.to(dtype=wd)
        if next(self.gn.parameters()).device != x.device:
            self.gn = self.gn.to(device=x.device)
        return self.gn(x)


class SafeLayerNorm(nn.Module):
    def __init__(self, hidden: int, eps: float = 1e-5):
        super().__init__()
        self.ln = nn.LayerNorm(hidden, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        wd = self.ln.weight.dtype
        if x.dtype != wd:
            x = x.to(dtype=wd)
        if self.ln.weight.device != x.device:
            self.ln = self.ln.to(device=x.device)
        return self.ln(x)


class DynLinear(nn.Module):
    def __init__(self, out_dim: int, bias: bool = True):
        super().__init__()
        self.out_dim = int(out_dim)
        self.bias = bool(bias)
        self.lin: Optional[nn.Linear] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            x = x.view(x.shape[0], -1)
        if self.lin is None:
            in_dim = int(x.shape[1])
            self.lin = nn.Linear(in_dim, self.out_dim, bias=self.bias).to(device=x.device, dtype=x.dtype)
            nn.init.kaiming_uniform_(self.lin.weight, a=5 ** 0.5)
            if self.lin.bias is not None:
                nn.init.zeros_(self.lin.bias)
        else:
            if self.lin.weight.device != x.device:
                self.lin = self.lin.to(device=x.device)
            if self.lin.weight.dtype != x.dtype:
                self.lin = self.lin.to(dtype=x.dtype)
        return self.lin(x)


class SafeLinear(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, bias: bool = True):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            x = x.view(x.shape[0], -1)
        wd = self.lin.weight.dtype
        if x.dtype != wd:
            x = x.to(dtype=wd)
        if self.lin.weight.device != x.device:
            self.lin = self.lin.to(device=x.device)
        return self.lin(x)


class ConvGNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int = 1, groups: int = 32):
        super().__init__()
        self.conv = SafeConv2d(in_ch, out_ch, k=k, s=s, p=p, bias=False)
        g = min(groups, out_ch)
        self.gn = SafeGroupNorm(g, out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.gn(x)
        return self.act(x)


class UpFuseBlockV1(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.refine = nn.Sequential(ConvGNAct(ch, ch, 3, 1, 1), ConvGNAct(ch, ch, 3, 1, 1))

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        if x.dtype != skip.dtype:
            x = x.to(dtype=skip.dtype)
        x = x + skip
        return self.refine(x)


class SegmentationHeadV1(nn.Module):
    def __init__(self, fpn_dim: int, max_classes: int = 5, dropout: float = 0.10):
        super().__init__()
        self.max_classes = int(max_classes)
        self.drop = nn.Dropout2d(float(dropout))
        self.p5_proj = nn.Sequential(ConvGNAct(fpn_dim, fpn_dim), ConvGNAct(fpn_dim, fpn_dim))
        self.up4 = UpFuseBlockV1(fpn_dim)
        self.up3 = UpFuseBlockV1(fpn_dim)
        self.up2 = UpFuseBlockV1(fpn_dim)
        self.classifier = SafeConv2d(fpn_dim, self.max_classes, k=1, s=1, p=0, bias=True)

    def forward(self, fpn_feats: List[torch.Tensor], out_size: Tuple[int, int]) -> torch.Tensor:
        p2, p3, p4, p5 = fpn_feats
        x = self.p5_proj(p5)
        x = self.up4(x, p4)
        x = self.up3(x, p3)
        x = self.up2(x, p2)
        x = self.drop(x)
        logits_low = self.classifier(x)
        return F.interpolate(logits_low, size=out_size, mode="bilinear", align_corners=False)


class ResBlock(nn.Module):
    def __init__(self, ch: int, groups: int = 32):
        super().__init__()
        self.c1 = ConvGNAct(ch, ch, 3, 1, 1, groups=groups)
        self.c2 = SafeConv2d(ch, ch, k=3, s=1, p=1, bias=False)
        g = min(groups, ch)
        self.gn2 = SafeGroupNorm(g, ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.c1(x)
        y = self.gn2(self.c2(y))
        if y.dtype != x.dtype:
            y = y.to(dtype=x.dtype)
        return self.act(x + y)


class SEBlock(nn.Module):
    def __init__(self, ch: int, r: int = 8):
        super().__init__()
        mid = max(ch // r, 8)
        self.fc1 = SafeConv2d(ch, mid, k=1, s=1, p=0, bias=True)
        self.fc2 = SafeConv2d(mid, ch, k=1, s=1, p=0, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = F.adaptive_avg_pool2d(x, 1)
        s = F.silu(self.fc1(s), inplace=True)
        s = torch.sigmoid(self.fc2(s))
        if s.dtype != x.dtype:
            s = s.to(dtype=x.dtype)
        return x * s


class AttnGate(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.g = SafeConv2d(ch, ch, k=1, s=1, p=0, bias=False)
        self.x = SafeConv2d(ch, ch, k=1, s=1, p=0, bias=False)
        self.o = SafeConv2d(ch, 1, k=1, s=1, p=0, bias=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        a = self.g(g) + self.x(x)
        a = F.silu(a, inplace=True)
        a = torch.sigmoid(self.o(a))
        if a.dtype != x.dtype:
            a = a.to(dtype=x.dtype)
        return x * a


class UpBlock(nn.Module):
    def __init__(self, ch: int, groups: int = 32, dropout: float = 0.0):
        super().__init__()
        self.gate = AttnGate(ch)
        self.fuse = nn.Sequential(
            ConvGNAct(ch * 2, ch, 3, 1, 1, groups=groups),
            ResBlock(ch, groups=groups),
            SEBlock(ch),
        )
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        if x.dtype != skip.dtype:
            x = x.to(dtype=skip.dtype)
        skip_g = self.gate(x, skip)
        if skip_g.dtype != x.dtype:
            skip_g = skip_g.to(dtype=x.dtype)
        x = torch.cat([x, skip_g], dim=1)
        x = self.fuse(x)
        x = self.drop(x)
        return x


class ASPP(nn.Module):
    def __init__(self, ch: int, rates: Tuple[int, int, int] = (3, 6, 9), groups: int = 32):
        super().__init__()
        self.b0 = ConvGNAct(ch, ch, 1, 1, 0, groups=groups)
        self.b1c = SafeConv2d(ch, ch, k=3, s=1, p=rates[0], bias=False, dilation=rates[0])
        self.b1n = SafeGroupNorm(min(groups, ch), ch)
        self.b2c = SafeConv2d(ch, ch, k=3, s=1, p=rates[1], bias=False, dilation=rates[1])
        self.b2n = SafeGroupNorm(min(groups, ch), ch)
        self.b3c = SafeConv2d(ch, ch, k=3, s=1, p=rates[2], bias=False, dilation=rates[2])
        self.b3n = SafeGroupNorm(min(groups, ch), ch)
        self.mix = nn.Sequential(
            ConvGNAct(ch * 4, ch, 1, 1, 0, groups=groups),
            ResBlock(ch, groups=groups),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y0 = self.b0(x)
        y1 = F.silu(self.b1n(self.b1c(x)), inplace=True)
        y2 = F.silu(self.b2n(self.b2c(x)), inplace=True)
        y3 = F.silu(self.b3n(self.b3c(x)), inplace=True)
        y = torch.cat([y0, y1, y2, y3], dim=1)
        return self.mix(y)


class SegmentationHead(nn.Module):
    def __init__(self, fpn_dim: int, max_classes: int = 5, dropout: float = 0.08):
        super().__init__()
        self.max_classes = max_classes
        self.p2p = nn.Sequential(ConvGNAct(fpn_dim, fpn_dim), ResBlock(fpn_dim), SEBlock(fpn_dim))
        self.p3p = nn.Sequential(ConvGNAct(fpn_dim, fpn_dim), ResBlock(fpn_dim), SEBlock(fpn_dim))
        self.p4p = nn.Sequential(ConvGNAct(fpn_dim, fpn_dim), ResBlock(fpn_dim), SEBlock(fpn_dim))
        self.p5p = nn.Sequential(ConvGNAct(fpn_dim, fpn_dim), ResBlock(fpn_dim), SEBlock(fpn_dim))
        self.aspp = ASPP(fpn_dim)
        self.u4 = UpBlock(fpn_dim, dropout=dropout)
        self.u3 = UpBlock(fpn_dim, dropout=dropout)
        self.u2 = UpBlock(fpn_dim, dropout=dropout)
        self.ref = nn.Sequential(ResBlock(fpn_dim), ResBlock(fpn_dim))
        self.cls = SafeConv2d(fpn_dim, max_classes, k=1, s=1, p=0, bias=True)

    def forward(self, fpn_feats: List[torch.Tensor], out_size: Tuple[int, int]) -> torch.Tensor:
        p2, p3, p4, p5 = fpn_feats
        p2 = self.p2p(p2)
        p3 = self.p3p(p3)
        p4 = self.p4p(p4)
        p5 = self.p5p(p5)
        x = self.aspp(p5)
        x = self.u4(x, p4)
        x = self.u3(x, p3)
        x = self.u2(x, p2)
        x = self.ref(x)
        logits = self.cls(x)
        return F.interpolate(logits, size=out_size, mode="bilinear", align_corners=False)


class GatedMLP(nn.Module):
    def __init__(self, hidden: int, dropout: float):
        super().__init__()
        self.ln = SafeLayerNorm(hidden)
        self.fc1 = SafeLinear(hidden, hidden * 2)
        self.fc2 = SafeLinear(hidden, hidden)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.ln(x)
        a, b = self.fc1(y).chunk(2, dim=-1)
        y = a * torch.sigmoid(b)
        y = self.drop(y)
        y = self.fc2(y)
        if y.dtype != x.dtype:
            y = y.to(dtype=x.dtype)
        return x + y


class ClassificationHead(nn.Module):
    def __init__(self, max_classes: int, hidden: int = 1024, dropout: float = 0.15):
        super().__init__()
        self.max_classes = int(max_classes)
        self.hidden = int(hidden)
        self.dropout = float(dropout)
        self.pre_fc = DynLinear(self.hidden)
        self.pre_ln = SafeLayerNorm(self.hidden)
        self.block1 = GatedMLP(self.hidden, dropout=self.dropout)
        self.block2 = GatedMLP(self.hidden, dropout=self.dropout)
        self.out_ln = SafeLayerNorm(self.hidden)
        self.out_fc = SafeLinear(self.hidden, self.max_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pre_fc(x)
        x = self.pre_ln(x)
        x = F.gelu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.block1(x)
        x = self.block2(x)
        x = self.out_ln(x)
        return self.out_fc(x)


class RegressionHeadV1(nn.Module):
    def __init__(self, max_dim: int, hidden: int = 1024, dropout: float = 0.10):
        super().__init__()
        self.max_dim = int(max_dim)
        self.hidden = int(hidden)
        self.dropout = float(dropout)
        self.fc1 = DynLinear(self.hidden)
        self.ln1 = SafeLayerNorm(self.hidden)
        self.fc2 = SafeLinear(self.hidden, self.max_dim)
        self.drop = nn.Dropout(self.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.ln1(x)
        x = F.gelu(x)
        x = self.drop(x)
        return self.fc2(x)


class RegressionHead(nn.Module):
    def __init__(self, max_dim: int, hidden: int = 1024, dropout: float = 0.12):
        super().__init__()
        self.max_dim = int(max_dim)
        self.hidden = int(hidden)
        self.dropout = float(dropout)
        self.pre_fc = DynLinear(self.hidden)
        self.pre_ln = SafeLayerNorm(self.hidden)
        self.block1 = GatedMLP(self.hidden, dropout=self.dropout)
        self.block2 = GatedMLP(self.hidden, dropout=self.dropout)
        self.out_ln = SafeLayerNorm(self.hidden)
        self.out_fc = SafeLinear(self.hidden, self.max_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pre_fc(x)
        x = self.pre_ln(x)
        x = F.gelu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.block1(x)
        x = self.block2(x)
        x = self.out_ln(x)
        return self.out_fc(x)


class DetectionGridHead(nn.Module):
    def __init__(self, fpn_dim: int, num_convs: int = 5, dropout: float = 0.05):
        super().__init__()
        self.fuse_conv = DynConv2d(fpn_dim, k=1, s=1, p=0, bias=False)
        self.fuse_gn = SafeGroupNorm(min(32, fpn_dim), fpn_dim)
        self.fuse_act = nn.SiLU(inplace=True)

        tower = []
        for _ in range(max(1, int(num_convs))):
            tower.append(ResBlock(fpn_dim))
        self.tower = nn.Sequential(*tower)

        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.box_pred = SafeConv2d(fpn_dim, 4, k=1, s=1, p=0, bias=True)
        self.obj_pred = SafeConv2d(fpn_dim, 1, k=1, s=1, p=0, bias=True)

    def forward(self, fpn_feats: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        p2, p3, p4 = fpn_feats[0], fpn_feats[1], fpn_feats[2]
        p3u = F.interpolate(p3, size=p2.shape[-2:], mode="bilinear", align_corners=False)
        p4u = F.interpolate(p4, size=p2.shape[-2:], mode="bilinear", align_corners=False)
        if p3u.dtype != p2.dtype:
            p3u = p3u.to(dtype=p2.dtype)
        if p4u.dtype != p2.dtype:
            p4u = p4u.to(dtype=p2.dtype)

        B, _, H, W = p2.shape
        yy = torch.linspace(0.0, 1.0, H, device=p2.device, dtype=p2.dtype)
        xx = torch.linspace(0.0, 1.0, W, device=p2.device, dtype=p2.dtype)
        gy, gx = torch.meshgrid(yy, xx, indexing="ij")
        coords = torch.stack([gx, gy], dim=0).unsqueeze(0).expand(B, -1, -1, -1)

        x = torch.cat([p2, p3u, p4u, coords], dim=1)
        x = self.fuse_conv(x)
        x = self.fuse_gn(x)
        x = self.fuse_act(x)
        x = self.drop(self.tower(x))

        obj_logits = self.obj_pred(x)
        box_raw = self.box_pred(x)

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
