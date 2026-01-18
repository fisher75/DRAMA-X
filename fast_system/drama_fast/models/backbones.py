"""Backbones for the fast system.

Phase-1 default: Video Swin-T (Tiny) because:
- retains spatio-temporal grid (T,H,W) -> Phase-2 prior injection (heatmap/tubelet) is natural;
- pyramid features -> stronger localization inductive bias for bbox.

Implementation notes:
- We use torchvision's Swin3D_T as the backbone for minimum dependencies.
- The wrapper returns TWO views:
  1) tokens_flat: [B, L, D]  (for query head)
  2) feat_3d:     [B, D, t, h, w] (for future heatmap / slow prior injection)
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn


class VideoSwinTinyBackbone(nn.Module):
    def __init__(self, pretrained: bool = True, freeze_backbone: Optional[bool] = None, freeze: bool = False):
        super().__init__()
        # ✅ 如果外部传了 freeze_backbone，就覆盖 freeze
        if freeze_backbone is not None:
            freeze = bool(freeze_backbone)
        try:
            from torchvision.models.video import swin3d_t, Swin3D_T_Weights
        except Exception as e:
            raise RuntimeError(
                "torchvision video Swin is required. Install a matching torch/torchvision build. "
                "Example: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121"
            ) from e

        weights = Swin3D_T_Weights.KINETICS400_V1 if pretrained else None
        self.model = swin3d_t(weights=weights)

        # Remove classification head (we need features, not logits).
        # IMPORTANT: For torchvision Swin3D, calling `model.features(x)` directly is WRONG because
        # `features` expects *patch-embedded* tokens, not raw pixels. We will run `model(x)` and
        # grab the 3D feature grid via a forward hook.
        self.model.head = nn.Identity()

        # Cache for feature grid captured from the internal `features` module during forward.
        self._cached_feat = None

        def _hook(_module, _inputs, output):
            self._cached_feat = output

        # `features` exists on torchvision's Swin3D.
        self.model.features.register_forward_hook(_hook)

        # Hidden dim: head input features is usually the last stage channel
        hidden_dim = getattr(getattr(self.model, "head", None), "in_features", None)
        # If head was replaced by Identity, try to look for original attr
        if hidden_dim is None:
            hidden_dim = getattr(self.model, "num_features", None)
        if hidden_dim is None:
            # Conservative fallback; Swin-T typically ends at 768
            hidden_dim = 768
        self.hidden_dim = int(hidden_dim)

        self.ln = nn.LayerNorm(self.hidden_dim)

        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """x: [B, C, T, H, W] -> (tokens_flat [B,L,D], feat_3d [B,D,t,h,w])."""
        self._cached_feat = None

        # Run full forward so patch-embedding happens internally.
        # Output is ignored; we consume the cached `features` output.
        _ = self.model(x)
        if self._cached_feat is None:
            raise RuntimeError("Failed to capture features from Swin3D. Unexpected model API.")

        feat = self._cached_feat

        # `feat` could be either [B, D, t, h, w] or [B, t, h, w, D] depending on torchvision version.
        if feat.ndim == 5:
            # Heuristic: if dim-1 looks like channels, treat as [B, D, t, h, w]
            if feat.shape[1] in {96, 192, 384, 768} or feat.shape[1] == self.hidden_dim:
                feat_3d = feat
                tokens = feat_3d.flatten(2).transpose(1, 2)  # [B, L, D]
            else:
                # [B, t, h, w, D]
                B, t, h, w, D = feat.shape
                tokens = feat.reshape(B, t * h * w, D)
                feat_3d = feat.permute(0, 4, 1, 2, 3).contiguous()
        elif feat.ndim == 3:
            # Already flattened tokens [B, L, D]
            tokens = feat
            # Make a dummy 3D view for Phase-2 interface compatibility.
            feat_3d = tokens.transpose(1, 2).unsqueeze(-1).unsqueeze(-1)  # [B,D,L,1,1]
        else:
            raise RuntimeError(f"Unexpected feature tensor shape from Swin3D: {tuple(feat.shape)}")

        tokens = self.ln(tokens)
        return tokens, feat_3d


def build_backbone(name: str = "swin_t", pretrained: bool = True, freeze: bool = False) -> nn.Module:
    name = name.lower()
    if name in {"swin_t", "swin_tiny", "videoswin_t"}:
        return VideoSwinTinyBackbone(pretrained=pretrained, freeze=freeze)
    raise ValueError(f"Unknown backbone: {name}")
