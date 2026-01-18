import torch
import torch.nn as nn

from .backbones import VideoSwinTinyBackbone
from .head_query import JointQueryHead


class FastSystemPhase1(nn.Module):
    """Fast System Phase-1.

    Inputs:
      pixel_values: [B, C, T, H, W]

    Outputs:
      pred_box:  [B,4]  (xyxy normalized)
      pred_risk: [B]    (0-1)

    `slow_priors` is reserved for Phase-2 (heatmap / tubelet / LLM prior injection).
    """

    def __init__(self, img_size: int = 224, pretrained: bool = True, freeze_backbone: bool = False):
        super().__init__()
        self.backbone = VideoSwinTinyBackbone(pretrained=pretrained, freeze_backbone=freeze_backbone)
        self.head = JointQueryHead(hidden_dim=self.backbone.hidden_dim)

        self.img_size = img_size

    def forward(self, pixel_values: torch.Tensor, slow_priors=None):
        tokens_flat, feat_3d = self.backbone(pixel_values)  # feat_3d reserved for later

        if slow_priors is not None:
            # Phase-2 placeholder:
            # - slow_priors could be (heatmap / box proposals / language priors)
            # - injected either by token biasing or extra cross-attn.
            pass

        pred_box, pred_risk = self.head(tokens_flat)
        return pred_box, pred_risk
