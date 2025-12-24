import os
import gc
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


class SimpleFPN(nn.Module):
    def __init__(self, out_channels: int = 256):
        super().__init__()
        self.lateral_convs = nn.ModuleList(
            [nn.LazyConv2d(out_channels, kernel_size=1), nn.LazyConv2d(out_channels, kernel_size=1), nn.LazyConv2d(out_channels, kernel_size=1), nn.LazyConv2d(out_channels, kernel_size=1)]
        )
        self.out_convs = nn.ModuleList(
            [nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1), nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1), nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1), nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)]
        )

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
        p2 = self.out_convs[0](p2)
        p3 = self.out_convs[1](p3)
        p4 = self.out_convs[2](p4)
        p5 = self.out_convs[3](p5)
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

        from transformers import AutoProcessor

        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=trust_remote_code, use_fast=False)

        if torch_dtype is None:
            torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        model_kwargs = {"trust_remote_code": trust_remote_code, "torch_dtype": torch_dtype, "low_cpu_mem_usage": True, "use_safetensors": True}
        if attn_implementation is not None:
            model_kwargs["attn_implementation"] = attn_implementation

        use_device_map = False
        if torch.cuda.is_available() and load_device.startswith("cuda"):
            try:
                import accelerate
                os.makedirs(offload_folder, exist_ok=True)
                model_kwargs.update({"device_map": "auto", "offload_folder": offload_folder, "offload_state_dict": True})
                use_device_map = True
            except Exception:
                use_device_map = False

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

        self._hooked_states: List[torch.Tensor] = []
        self._hooks: List[torch.utils.hooks.RemovableHandle] = []

        blocks = _find_transformer_block_list(self.encoder)
        if blocks is None or len(blocks) < self.num_hook_layers:
            raise RuntimeError("Could not find transformer block ModuleList inside vision encoder.")
        self.blocks = blocks
        hook_indices = list(range(len(blocks) - self.num_hook_layers, len(blocks)))

        def _make_hook():
            def _hook(_module, _inp, out):
                out0 = out[0] if isinstance(out, (tuple, list)) else out
                if torch.is_tensor(out0):
                    self._hooked_states.append(out0)
            return _hook

        for idx in hook_indices:
            self._hooks.append(self.blocks[idx].register_forward_hook(_make_hook()))

        self.fpn = SimpleFPN(out_channels=fpn_dim)
        self._fpn_dtype = torch_dtype

        if torch.cuda.is_available() and load_device.startswith("cuda"):
            self.fpn.to(device=load_device, dtype=torch_dtype)
        else:
            self.fpn.to(dtype=torch_dtype)

        if torch.cuda.is_available() and (not use_device_map):
            self.visual.to(load_device)

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

    def forward(self, images: torch.Tensor) -> Dict[str, Union[List[torch.Tensor], torch.Tensor, torch.Tensor]]:
        device = images.device
        bsz = images.shape[0]
        pil_images = [_tensor_to_pil(images[i]) for i in range(bsz)]
        if self.force_image_size is not None:
            s = int(self.force_image_size)
            pil_images = [im.resize((s, s), resample=Image.BICUBIC) for im in pil_images]

        if hasattr(self.processor, "image_processor"):
            vis_inputs = self.processor.image_processor(images=pil_images, return_tensors="pt")
        else:
            vis_inputs = self.processor(images=pil_images, return_tensors="pt")

        if "pixel_values" not in vis_inputs:
            raise RuntimeError("Processor did not return pixel_values.")

        pixel_values = vis_inputs["pixel_values"].to(device=device)
        try:
            enc_dtype = next(self.encoder.parameters()).dtype
        except StopIteration:
            enc_dtype = self._fpn_dtype
        pixel_values = pixel_values.to(dtype=enc_dtype)

        grid = None
        for k in ("image_grid_thw", "vision_grid_thws", "image_grid_thws", "vision_grid_thw"):
            if k in vis_inputs and vis_inputs[k] is not None:
                grid = vis_inputs[k]
                break
        if grid is None:
            raise RuntimeError("Processor did not return grid.")
        image_grid_thw = grid.to(device=device)

        self._hooked_states = []
        enc_out = self._call_encoder(pixel_values, image_grid_thw)

        window_index = None
        tokens_final = None
        if isinstance(enc_out, (tuple, list)):
            if len(enc_out) >= 1 and torch.is_tensor(enc_out[0]):
                tokens_final = enc_out[0]
            if len(enc_out) >= 2 and torch.is_tensor(enc_out[1]):
                window_index = enc_out[1]
        elif torch.is_tensor(enc_out):
            tokens_final = enc_out
        else:
            if hasattr(enc_out, "last_hidden_state"):
                tokens_final = enc_out.last_hidden_state
            else:
                raise RuntimeError("Unexpected encoder output type.")

        tokens_final = self._ensure_flat_tokens(tokens_final)

        if window_index is not None and torch.is_tensor(window_index):
            try:
                reverse_indices = torch.argsort(window_index)
                if tokens_final.ndim == 2:
                    tokens_final = tokens_final[reverse_indices, :]
                reordered_states = []
                for hs in self._hooked_states:
                    hs = self._ensure_flat_tokens(hs)
                    if hs.ndim == 2:
                        hs = hs[reverse_indices, :]
                    reordered_states.append(hs)
                self._hooked_states = reordered_states
            except Exception:
                pass

        if len(self._hooked_states) < self.num_hook_layers:
            raise RuntimeError("Hooked states missing.")

        feats_maps: List[torch.Tensor] = []
        for hs in self._hooked_states[-self.num_hook_layers:]:
            hs = self._ensure_flat_tokens(hs)
            hs_bnc = self._split_tokens_to_bnc(hs, image_grid_thw, bsz)
            fmap = self._bnc_to_bchw(hs_bnc, image_grid_thw)
            feats_maps.append(fmap)

        c2 = feats_maps[0]
        c3 = F.avg_pool2d(feats_maps[1], kernel_size=2, stride=2)
        c4 = F.avg_pool2d(feats_maps[2], kernel_size=4, stride=4)
        c5 = F.avg_pool2d(feats_maps[3], kernel_size=8, stride=8)

        fpn_dtype = self._fpn_dtype
        c2 = c2.to(dtype=fpn_dtype)
        c3 = c3.to(dtype=fpn_dtype)
        c4 = c4.to(dtype=fpn_dtype)
        c5 = c5.to(dtype=fpn_dtype)

        fpn_feats = self.fpn([c2, c3, c4, c5])
        global_feat = feats_maps[-1].mean(dim=(2, 3))

        return {"fpn": fpn_feats, "global": global_feat, "image_grid_thw": image_grid_thw}
