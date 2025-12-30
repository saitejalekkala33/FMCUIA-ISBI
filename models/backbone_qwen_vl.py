import os
import gc
import math
from typing import Dict, List, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


def _tensor_to_pil(img_chw: torch.Tensor) -> Image.Image:
    x = img_chw.detach().cpu()
    if x.dtype != torch.uint8:
        x_max = float(x.max().item()) if x.numel() > 0 else 0.0
        if x_max <= 1.0:
            x = (x * 255.0).clamp(0, 255).to(torch.uint8)
        else:
            x = x.clamp(0, 255).to(torch.uint8)
    x = x.permute(1, 2, 0).contiguous().numpy()
    return Image.fromarray(x)


def _find_transformer_block_list(module: nn.Module) -> Optional[nn.ModuleList]:
    common_names = ["layers", "blocks", "h", "encoder_layers", "transformer_blocks"]
    for name in common_names:
        if hasattr(module, name) and isinstance(getattr(module, name), nn.ModuleList):
            ml = getattr(module, name)
            if len(ml) > 0:
                return ml
    for _, child in module.named_children():
        if isinstance(child, nn.ModuleList) and len(child) > 0:
            elem = child[0]
            if any(hasattr(elem, k) for k in ["attn", "self_attn", "attention"]):
                return child
        else:
            ml = _find_transformer_block_list(child)
            if ml is not None:
                return ml
    return None


def _select_hook_indices(L: int, n: int) -> List[int]:
    if L <= n:
        return list(range(max(0, L - n), L))
    xs = torch.linspace(0, L - 1, steps=n).round().long().tolist()
    xs = sorted(set(int(x) for x in xs))
    if len(xs) < n:
        tail = list(range(L - n, L))
        xs = sorted(set(xs + tail))
    if len(xs) > n:
        xs = xs[-n:]
    return xs


class ConvGNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int = 1, groups: int = 32):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False)
        g = min(groups, out_ch)
        self.gn = nn.GroupNorm(g, out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        wd = self.conv.weight.dtype
        if x.dtype != wd:
            x = x.to(dtype=wd)
        return self.act(self.gn(self.conv(x)))


class DynConv1x1(nn.Module):
    def __init__(self, out_ch: int, bias: bool = False):
        super().__init__()
        self.out_ch = int(out_ch)
        self.bias = bool(bias)
        self.conv: Optional[nn.Conv2d] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.conv is None:
            in_ch = int(x.shape[1])
            self.conv = nn.Conv2d(in_ch, self.out_ch, kernel_size=1, bias=self.bias).to(device=x.device, dtype=x.dtype)
            nn.init.kaiming_normal_(self.conv.weight, mode="fan_out", nonlinearity="relu")
            if self.conv.bias is not None:
                nn.init.zeros_(self.conv.bias)
        else:
            if self.conv.weight.device != x.device:
                self.conv = self.conv.to(device=x.device)
            if self.conv.weight.dtype != x.dtype:
                self.conv = self.conv.to(dtype=x.dtype)
        return self.conv(x)


class TokenPyramid(nn.Module):
    def __init__(self, out_channels: int = 256, groups: int = 32):
        super().__init__()
        self.proj = nn.ModuleList([DynConv1x1(out_channels), DynConv1x1(out_channels), DynConv1x1(out_channels), DynConv1x1(out_channels)])
        self.ref = nn.ModuleList(
            [
                nn.Sequential(ConvGNAct(out_channels, out_channels, 3, 1, 1, groups=groups), ConvGNAct(out_channels, out_channels, 3, 1, 1, groups=groups)),
                nn.Sequential(ConvGNAct(out_channels, out_channels, 3, 1, 1, groups=groups), ConvGNAct(out_channels, out_channels, 3, 1, 1, groups=groups)),
                nn.Sequential(ConvGNAct(out_channels, out_channels, 3, 1, 1, groups=groups), ConvGNAct(out_channels, out_channels, 3, 1, 1, groups=groups)),
                nn.Sequential(ConvGNAct(out_channels, out_channels, 3, 1, 1, groups=groups), ConvGNAct(out_channels, out_channels, 3, 1, 1, groups=groups)),
            ]
        )

    def forward(self, maps: List[torch.Tensor]) -> List[torch.Tensor]:
        f0, f1, f2, f3 = maps
        _, _, H, W = f0.shape
        h3, w3 = max(1, (H + 1) // 2), max(1, (W + 1) // 2)
        h4, w4 = max(1, (H + 3) // 4), max(1, (W + 3) // 4)
        h5, w5 = max(1, (H + 7) // 8), max(1, (W + 7) // 8)

        c2 = self.ref[0](self.proj[0](f0))
        x1 = self.proj[1](f1)
        x2 = self.proj[2](f2)
        x3 = self.proj[3](f3)

        c3 = self.ref[1](F.interpolate(x1, size=(h3, w3), mode="bilinear", align_corners=False))
        c4 = self.ref[2](F.interpolate(x2, size=(h4, w4), mode="bilinear", align_corners=False))
        c5 = self.ref[3](F.interpolate(x3, size=(h5, w5), mode="bilinear", align_corners=False))
        return [c2, c3, c4, c5]


class SimpleFPN(nn.Module):
    def __init__(self, out_channels: int = 256):
        super().__init__()
        self.lateral_convs = nn.ModuleList([DynConv1x1(out_channels), DynConv1x1(out_channels), DynConv1x1(out_channels), DynConv1x1(out_channels)])
        self.out_convs = nn.ModuleList(
            [
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            ]
        )

    @staticmethod
    def _apply_conv(conv: nn.Conv2d, x: torch.Tensor) -> torch.Tensor:
        wd = conv.weight.dtype
        if x.dtype != wd:
            x = x.to(dtype=wd)
        return conv(x)

    def forward(self, feats: List[torch.Tensor]) -> List[torch.Tensor]:
        c2, c3, c4, c5 = feats
        lat2 = self.lateral_convs[0](c2)
        lat3 = self.lateral_convs[1](c3)
        lat4 = self.lateral_convs[2](c4)
        lat5 = self.lateral_convs[3](c5)

        p5 = lat5
        p4 = lat4 + F.interpolate(p5, size=lat4.shape[-2:], mode="nearest")
        p3 = lat3 + F.interpolate(p4, size=lat3.shape[-2:], mode="nearest")
        p2 = lat2 + F.interpolate(p3, size=lat2.shape[-2:], mode="nearest")

        p2 = self._apply_conv(self.out_convs[0], p2)
        p3 = self._apply_conv(self.out_convs[1], p3)
        p4 = self._apply_conv(self.out_convs[2], p4)
        p5 = self._apply_conv(self.out_convs[3], p5)
        return [p2, p3, p4, p5]


class QwenVLBackbone(nn.Module):
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        trust_remote_code: bool = True,
        attn_implementation: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
        keep_llm: bool = False,
        fpn_dim: int = 256,
        num_hook_layers: int = 4,
        load_device: str = "cuda:0",
        offload_folder: str = "offload",
        force_image_size: Optional[int] = None,
    ):
        super().__init__()
        self.model_name = model_name
        self.keep_llm = keep_llm
        self.num_hook_layers = num_hook_layers
        self.force_image_size = force_image_size
        self._processor_kind = "auto"

        if torch_dtype is None:
            torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        try:
            from transformers import AutoProcessor
            self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=trust_remote_code, use_fast=False)
            self._processor_kind = "auto"
        except ImportError as e:
            msg = str(e).lower()
            if "torchvision" in msg or "autovideoprocessor" in msg:
                from transformers import AutoImageProcessor
                self.processor = AutoImageProcessor.from_pretrained(model_name, trust_remote_code=trust_remote_code, use_fast=False)
                self._processor_kind = "image"
            else:
                raise

        model_kwargs = {"trust_remote_code": trust_remote_code, "torch_dtype": torch_dtype, "low_cpu_mem_usage": True, "use_safetensors": True}
        if attn_implementation is not None:
            model_kwargs["attn_implementation"] = attn_implementation

        os.makedirs(offload_folder, exist_ok=True)

        self.vlm = None
        self.visual = None

        try:
            from transformers import Qwen2_5_VLForConditionalGeneration
            self.vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name, **model_kwargs)
        except Exception:
            from transformers import AutoModelForVision2Seq
            self.vlm = AutoModelForVision2Seq.from_pretrained(model_name, **model_kwargs)

        visual = None
        for cand in ["visual", "vision_model", "vision_tower", "vision_encoder"]:
            if hasattr(self.vlm, cand):
                visual = getattr(self.vlm, cand)
                break
        if visual is None:
            raise RuntimeError("Could not locate vision module.")
        self.visual = visual

        self.encoder = getattr(self.visual, "encoder", None)
        if self.encoder is None:
            self.encoder = self.visual

        if not self.keep_llm:
            for cand in ["model", "language_model", "lm", "transformer"]:
                if hasattr(self.vlm, cand):
                    try:
                        setattr(self.vlm, cand, None)
                    except Exception:
                        pass
            self.vlm = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        self._hooked_states: List[Optional[torch.Tensor]] = [None for _ in range(self.num_hook_layers)]
        self._hooks: List[torch.utils.hooks.RemovableHandle] = []

        blocks = _find_transformer_block_list(self.encoder)
        if blocks is None or len(blocks) < self.num_hook_layers:
            raise RuntimeError("Could not find transformer block ModuleList inside vision encoder.")
        self.blocks = blocks

        hook_indices = _select_hook_indices(len(blocks), self.num_hook_layers)

        def _make_hook(slot: int):
            def _hook(_module, _inp, out):
                out0 = out[0] if isinstance(out, (tuple, list)) else out
                if torch.is_tensor(out0):
                    self._hooked_states[slot] = out0
            return _hook

        for slot, idx in enumerate(hook_indices):
            self._hooks.append(self.blocks[idx].register_forward_hook(_make_hook(slot)))

        self.pyramid = TokenPyramid(out_channels=fpn_dim)
        self.fpn = SimpleFPN(out_channels=fpn_dim)
        self.fpn_v1 = SimpleFPN(out_channels=fpn_dim)

        if torch.cuda.is_available() and load_device.startswith("cuda"):
            self.pyramid.to(device=load_device, dtype=torch_dtype)
            self.fpn.to(device=load_device, dtype=torch_dtype)
            self.fpn_v1.to(device=load_device, dtype=torch_dtype)
            self.visual.to(load_device)
        else:
            self.pyramid.to(dtype=torch_dtype)
            self.fpn.to(dtype=torch_dtype)
            self.fpn_v1.to(dtype=torch_dtype)

    def enable_gradient_checkpointing(self):
        for m in [self.visual, self.encoder]:
            if m is None:
                continue
            if hasattr(m, "gradient_checkpointing_enable"):
                try:
                    m.gradient_checkpointing_enable()
                except Exception:
                    pass
            if hasattr(m, "gradient_checkpointing"):
                try:
                    m.gradient_checkpointing = True
                except Exception:
                    pass

    def _call_encoder(self, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor):
        enc = self.encoder
        try:
            return enc(pixel_values, image_grid_thw)
        except TypeError:
            pass
        try:
            return enc(pixel_values, grid_thw=image_grid_thw)
        except TypeError:
            pass
        try:
            return enc(pixel_values=pixel_values, image_grid_thw=image_grid_thw)
        except TypeError:
            pass
        try:
            return enc(pixel_values=pixel_values, grid_thw=image_grid_thw)
        except TypeError:
            pass
        return enc(pixel_values)

    @staticmethod
    def _ensure_flat_tokens(x: torch.Tensor) -> torch.Tensor:
        if x.ndim in (2, 3):
            return x
        raise ValueError(f"Unexpected token tensor shape: {tuple(x.shape)}")

    @staticmethod
    def _split_tokens_to_bnc(tokens: torch.Tensor, grid_thw: torch.Tensor, batch_size: int) -> torch.Tensor:
        grid_thw_cpu = grid_thw.detach().cpu()
        t0, h0, w0 = [int(v) for v in grid_thw_cpu[0].tolist()]
        n0 = t0 * h0 * w0
        for b in range(batch_size):
            t, h, w = [int(v) for v in grid_thw_cpu[b].tolist()]
            if t * h * w != n0:
                raise ValueError("Variable token count per image not supported.")
        if tokens.ndim == 3:
            return tokens
        if tokens.shape[0] != batch_size * n0:
            raise ValueError("Token count mismatch.")
        return tokens.view(batch_size, n0, tokens.shape[1])

    @staticmethod
    def _bnc_to_bchw(tokens_bnc: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        b, n, c = tokens_bnc.shape
        t, h, w = [int(v) for v in grid_thw[0].tolist()]
        if t != 1:
            raise ValueError("Expected t=1.")
        if n != t * h * w:
            raise ValueError("Grid mismatch.")
        x = tokens_bnc.view(b, t, h, w, c)[:, 0]
        return x.permute(0, 3, 1, 2).contiguous()

    def _infer_grid_thw(self, pixel_values: torch.Tensor, bsz: int, device: torch.device) -> torch.Tensor:
        H, W = int(pixel_values.shape[-2]), int(pixel_values.shape[-1])
        cfg = getattr(self.visual, "config", None)
        if cfg is None:
            cfg = getattr(self.encoder, "config", None)

        ps = getattr(cfg, "patch_size", 14) if cfg is not None else 14
        if isinstance(ps, (list, tuple)):
            ps = int(ps[0])
        else:
            ps = int(ps)

        merge = 1
        if cfg is not None:
            m = getattr(cfg, "spatial_merge_size", None)
            if m is None:
                m = getattr(cfg, "merge_size", None)
            if m is not None:
                if isinstance(m, (list, tuple)):
                    merge = int(m[0])
                else:
                    merge = int(m)
                merge = max(1, merge)

        gh = int(math.ceil(H / float(ps)))
        gw = int(math.ceil(W / float(ps)))
        gh = max(1, gh // merge)
        gw = max(1, gw // merge)

        return torch.tensor([[1, gh, gw]] * bsz, device=device, dtype=torch.long)

    @staticmethod
    def _pool_down(x: torch.Tensor, factor: int) -> torch.Tensor:
        f = int(max(1, factor))
        if f == 1:
            return x
        H, W = int(x.shape[-2]), int(x.shape[-1])
        if H < f or W < f:
            oh = max(1, H // f)
            ow = max(1, W // f)
            return F.adaptive_avg_pool2d(x, output_size=(oh, ow))
        return F.avg_pool2d(x, kernel_size=f, stride=f)

    def forward(self, images: torch.Tensor) -> Dict[str, Union[List[torch.Tensor], torch.Tensor, torch.Tensor]]:
        device = images.device
        bsz = images.shape[0]
        pil_images = [_tensor_to_pil(images[i]) for i in range(bsz)]
        if self.force_image_size is not None:
            s = int(self.force_image_size)
            pil_images = [im.resize((s, s), resample=Image.BICUBIC) for im in pil_images]

        if self._processor_kind == "auto" and hasattr(self.processor, "image_processor"):
            vis_inputs = self.processor.image_processor(images=pil_images, return_tensors="pt")
        else:
            vis_inputs = self.processor(images=pil_images, return_tensors="pt")

        if "pixel_values" not in vis_inputs:
            raise RuntimeError("Processor did not return pixel_values.")

        pixel_values = vis_inputs["pixel_values"].to(device=device)
        try:
            enc_dtype = next(self.encoder.parameters()).dtype
        except StopIteration:
            enc_dtype = pixel_values.dtype
        pixel_values = pixel_values.to(dtype=enc_dtype)

        grid = None
        for k in ("image_grid_thw", "vision_grid_thws", "image_grid_thws", "vision_grid_thw", "grid_thw"):
            if k in vis_inputs and vis_inputs[k] is not None:
                grid = vis_inputs[k]
                break
        if grid is None:
            image_grid_thw = self._infer_grid_thw(pixel_values, bsz, device)
        else:
            image_grid_thw = grid.to(device=device, dtype=torch.long)

        self._hooked_states = [None for _ in range(self.num_hook_layers)]
        enc_out = self._call_encoder(pixel_values, image_grid_thw)

        tokens_final = None
        if isinstance(enc_out, (tuple, list)):
            if len(enc_out) >= 1 and torch.is_tensor(enc_out[0]):
                tokens_final = enc_out[0]
        elif torch.is_tensor(enc_out):
            tokens_final = enc_out
        else:
            if hasattr(enc_out, "last_hidden_state"):
                tokens_final = enc_out.last_hidden_state
            else:
                raise RuntimeError("Unexpected encoder output type.")

        _ = self._ensure_flat_tokens(tokens_final)

        if any(s is None for s in self._hooked_states):
            raise RuntimeError("Hooked states missing.")

        feats_maps: List[torch.Tensor] = []
        for hs in self._hooked_states:
            hs = self._ensure_flat_tokens(hs)
            hs_bnc = self._split_tokens_to_bnc(hs, image_grid_thw, bsz)
            fmap = self._bnc_to_bchw(hs_bnc, image_grid_thw)
            feats_maps.append(fmap)

        try:
            target_dtype = next(self.fpn.parameters()).dtype
        except StopIteration:
            target_dtype = feats_maps[0].dtype

        feats_maps = [m.to(dtype=target_dtype) for m in feats_maps]

        c2, c3, c4, c5 = self.pyramid(feats_maps)
        c2 = c2.to(dtype=target_dtype)
        c3 = c3.to(dtype=target_dtype)
        c4 = c4.to(dtype=target_dtype)
        c5 = c5.to(dtype=target_dtype)

        fpn_feats = self.fpn([c2, c3, c4, c5])
        pooled = [F.adaptive_avg_pool2d(p, 1).flatten(1) for p in fpn_feats]
        global_feat = torch.cat(pooled, dim=1)

        fm0, fm1, fm2, fm3 = feats_maps
        c2v1 = fm0
        c3v1 = self._pool_down(fm1, 2)
        c4v1 = self._pool_down(fm2, 4)
        c5v1 = self._pool_down(fm3, 8)
        fpn_feats_v1 = self.fpn_v1([c2v1, c3v1, c4v1, c5v1])
        global_feat_v1 = fm3.mean(dim=(2, 3))

        return {
            "fpn": fpn_feats,
            "global": global_feat,
            "fpn_v1": fpn_feats_v1,
            "global_v1": global_feat_v1,
            "image_grid_thw": image_grid_thw,
        }
