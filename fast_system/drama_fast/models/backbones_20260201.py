import torch
import torch.nn as nn
from torchvision.models.video import swin3d_t, Swin3D_T_Weights


def _to_bcthw(x: torch.Tensor) -> torch.Tensor:
    """Convert a Swin3D feature tensor to [B, C, T, H, W] when possible."""
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(x)}")

    # Common cases:
    #  - [B, C, T, H, W]
    #  - [B, T, H, W, C]
    if x.ndim == 5:
        # Heuristic: if last dim is small-ish and matches channel conventions, assume BTHWC.
        # Otherwise assume BCTHW.
        b, d1, d2, d3, d4 = x.shape
        # If x is [B, T, H, W, C], C is last dim.
        if d4 <= 2048 and d1 <= 64 and d2 <= 512 and d3 <= 512:
            # Likely [B, T, H, W, C]
            return x.permute(0, 4, 1, 2, 3).contiguous()
        return x

    # Some internal modules may emit flattened tokens [B, L, C] or [B, C, L].
    # We cannot reliably reshape without extra metadata; return as-is and let caller ignore.
    return x


class VideoSwinTinyBackbone(nn.Module):
    """Phase-1 backbone wrapper: returns (tokens, feat_3d_last)."""

    def __init__(self, pretrained=True):
        super().__init__()
        if pretrained:
            weights = Swin3D_T_Weights.KINETICS400_V1
            self.model = swin3d_t(weights=weights)
        else:
            self.model = swin3d_t(weights=None)

        # Hook to get the final 3D feature map.
        self._hooked_feat = None

        def hook_fn(module, input, output):
            self._hooked_feat = output

        self.model.features.register_forward_hook(hook_fn)

    def forward(self, x):
        tokens = self.model(x)  # [B, 768]
        feat_3d = self._hooked_feat
        if isinstance(feat_3d, (tuple, list)):
            feat_3d = feat_3d[0]
        if isinstance(feat_3d, torch.Tensor):
            feat_3d = _to_bcthw(feat_3d)
        return tokens, feat_3d



class VideoSwinTinyBackboneMultiScale(nn.Module):
    """Video Swin Tiny backbone that returns multi-scale 3D features.

    Motivation:
        Torchvision's `swin3d_t` forward() applies internal shape/layout transforms
        (e.g., patch embedding / channel-last windows) before the transformer blocks.
        Manually iterating `self.model.features` with a raw RGB tensor bypasses those
        steps and leads to LayerNorm shape errors.

    Strategy:
        - Run the official `self.model(x)` forward for correctness.
        - Use forward hooks on `self.model.features` submodules to capture intermediate
          feature maps. We then convert them to [B, C, T, H, W] and pick C3/C4/C5 by
          channel sizes (192/384/768).

    Returns:
        tokens: [B, C5] global pooled last-stage features (useful as a clip-level token)
        (c3, c4, c5): tuple of 3D features at strides 8/16/32, each in [B, C, T, H, W]
            - c3: channels 192
            - c4: channels 384
            - c5: channels 768
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = Swin3D_T_Weights.DEFAULT if pretrained else None
        self.model = swin3d_t(weights=weights)

        self.out_channels = (192, 384, 768)
        self._ch2key = {192: "c3", 384: "c4", 768: "c5"}

        # Buffers filled by hooks during forward()
        self._pyr = {}
        self._feat_last = None

        # Register hooks to capture intermediate outputs during the *official* forward.
        if hasattr(self.model, "features") and isinstance(self.model.features, nn.Sequential):
            for _i, _layer in enumerate(self.model.features):
                _layer.register_forward_hook(self._make_hook())
        else:
            # If this triggers on your env, we'll need a version-specific hook plan.
            raise RuntimeError(
                "[VideoSwinTinyBackboneMultiScale] Expected model.features to be nn.Sequential. "
                "Your torchvision Swin3D implementation differs."
            )

    def _make_hook(self):
        def hook_fn(module, inputs, output):
            y = output[0] if isinstance(output, (tuple, list)) else output
            if not isinstance(y, torch.Tensor):
                return
            t = _to_bcthw(y)
            # We only know how to select by channel dimension when it's [B,C,T,H,W]
            if t.ndim == 5:
                c = int(t.shape[1])
                if c in self._ch2key:
                    self._pyr[self._ch2key[c]] = t
                self._feat_last = t
        return hook_fn

    def forward(self, x: torch.Tensor):
        # Accept both [B,3,T,H,W] and [B,T,3,H,W]
        if x.ndim == 5 and x.shape[1] != 3 and x.shape[2] == 3:
            x = x.permute(0, 2, 1, 3, 4).contiguous()

        # Clear previous forward's cached outputs
        self._pyr = {}
        self._feat_last = None

        # Run official forward (ensures correct internal transforms)
        _ = self.model(x)

        missing = [k for k in ("c3", "c4", "c5") if k not in self._pyr]
        if missing:
            got = {k: tuple(v.shape) for k, v in self._pyr.items()}
            raise RuntimeError(
                f"[VideoSwinTinyBackboneMultiScale] Failed to capture multi-scale features {missing}. "
                f"Captured: {got}. "
                f"Your torchvision Swin3D implementation may not expose per-stage outputs via `model.features`."
            )

        c3, c4, c5 = self._pyr["c3"], self._pyr["c4"], self._pyr["c5"]
        feat_last = self._feat_last if isinstance(self._feat_last, torch.Tensor) else c5

        # Clip-level token: global spatiotemporal average pool
        tokens = feat_last.mean(dim=(2, 3, 4))

        return tokens, (c3, c4, c5)
