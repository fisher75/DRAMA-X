import torch
import torch.nn as nn


class JointQueryHead(nn.Module):
    """Learnable queries + cross-attn -> bbox + risk.

    Supports both single-query (legacy) and multi-query mode.

    - If num_queries == 1: returns (pred_box [B,4], pred_risk [B]) for backward compatibility.
    - If num_queries  > 1: returns (pred_boxes [B,Q,4], pred_risks [B,Q]).

    The bbox head outputs normalized xyxy in [0,1]. We enforce valid boxes by sorting
    x/y coordinates per query: x1=min(x1,x2), x2=max(...), etc.
    """

    def __init__(self, hidden_dim: int, num_heads: int = 8, mlp_dim: int = 256, num_queries: int = 1):
        super().__init__()
        assert num_queries >= 1
        self.num_queries = int(num_queries)

        # Small init helps stabilize early training
        self.query = nn.Parameter(torch.randn(1, self.num_queries, hidden_dim) * 0.02)

        self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)

        self.box_head = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_dim, 4),
            nn.Sigmoid(),  # xyxy in [0,1]
        )

        self.risk_head = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_dim, 1),
            nn.Sigmoid(),  # risk in [0,1]
        )

    def forward(self, tokens_flat: torch.Tensor):
        """tokens_flat: [B, L, D]"""
        b = tokens_flat.shape[0]
        q = self.query.expand(b, -1, -1)  # [B,Q,D]
        out, _ = self.cross_attn(query=q, key=tokens_flat, value=tokens_flat)  # [B,Q,D]

        raw_box = self.box_head(out)  # [B,Q,4]

        # Enforce valid xyxy per query
        x_coords = raw_box[..., [0, 2]]
        y_coords = raw_box[..., [1, 3]]
        x1 = torch.min(x_coords, dim=-1).values
        x2 = torch.max(x_coords, dim=-1).values
        y1 = torch.min(y_coords, dim=-1).values
        y2 = torch.max(y_coords, dim=-1).values
        pred_boxes = torch.stack([x1, y1, x2, y2], dim=-1)  # [B,Q,4]

        pred_risks = self.risk_head(out).squeeze(-1)  # [B,Q]

        if self.num_queries == 1:
            return pred_boxes[:, 0], pred_risks[:, 0]
        return pred_boxes, pred_risks
