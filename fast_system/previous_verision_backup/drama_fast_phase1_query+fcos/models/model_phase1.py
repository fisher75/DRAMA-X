import torch
import torch.nn as nn

from .backbones import VideoSwinTinyBackbone
from .head_query import JointQueryHead


class FastSystemPhase1(nn.Module):
    """Fast System Phase-1.

    Inputs:
      pixel_values: [B, C, T, H, W]

    Outputs:
      - num_queries == 1:
          pred_box:  [B,4]  (xyxy normalized)
          pred_risk: [B]    (0-1)
      - num_queries  > 1:
          pred_boxes: [B,Q,4] (xyxy normalized)
          pred_risks: [B,Q]   (0-1)

    `slow_priors` is reserved for Phase-2 (heatmap / tubelet / LLM prior injection).
    """

    def __init__(
        self,
        img_size: int = 224,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        num_queries: int = 1,
        num_heads: int = 8,
        mlp_dim: int = 256,
    ):
        super().__init__()
        self.backbone = VideoSwinTinyBackbone(pretrained=pretrained, freeze_backbone=freeze_backbone)
        self.head = JointQueryHead(
            hidden_dim=self.backbone.hidden_dim,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            num_queries=num_queries,
        )

        self.img_size = img_size
        self.num_queries = int(num_queries)

    def forward(self, pixel_values: torch.Tensor, slow_priors=None):
        tokens_flat, feat_3d = self.backbone(pixel_values)  # feat_3d reserved for later

        if slow_priors is not None:
            # Phase-2 placeholder:
            # - slow_priors could be (heatmap / box proposals / language priors)
            # - injected either by token biasing or extra cross-attn.
            pass

        return self.head(tokens_flat)

    @torch.no_grad()
    def predict_primary(self, pixel_values: torch.Tensor):
        """Utility: pick the query with max risk and return a single (box,risk).

        Useful for evaluation / visualization when num_queries > 1.
        """
        out = self.forward(pixel_values)
        if self.num_queries == 1:
            return out  # (box [B,4], risk [B])

        pred_boxes, pred_risks = out
        idx = pred_risks.argmax(dim=1)  # [B]
        b = torch.arange(pred_boxes.shape[0], device=pred_boxes.device)
        primary_box = pred_boxes[b, idx]
        primary_risk = pred_risks[b, idx]
        return primary_box, primary_risk
