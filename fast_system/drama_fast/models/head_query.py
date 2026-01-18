import torch
import torch.nn as nn

class JointQueryHead(nn.Module):
    """Single learnable query -> cross-attn -> bbox+risk.
    
    Modified to ensure valid box coordinates (x1 < x2, y1 < y2).
    """

    def __init__(self, hidden_dim: int, num_heads: int = 8, mlp_dim: int = 256):
        super().__init__()
        # 初始化一个小一点的 query，有助于训练稳定
        self.query = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)

        self.box_head = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim), nn.ReLU(inplace=True),
            nn.Linear(mlp_dim, 4), nn.Sigmoid()  # xyxy in [0,1]
        )
        self.risk_head = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim), nn.ReLU(inplace=True),
            nn.Linear(mlp_dim, 1), nn.Sigmoid()  # risk in [0,1]
        )

    def forward(self, tokens_flat: torch.Tensor):
        """tokens_flat: [B, L, D]"""
        b = tokens_flat.shape[0]
        q = self.query.expand(b, -1, -1)  # [B,1,D]
        out, _ = self.cross_attn(query=q, key=tokens_flat, value=tokens_flat)  # [B,1,D]

        # 原始预测 [B, 4] -> 可能出现 x1 > x2 的情况
        raw_box = self.box_head(out).squeeze(1)
        
        # --- GPT 建议的修复：强制排序 ---
        # 无论模型预测的哪个大哪个小，我们强制把小的当左上(x1, y1)，大的当右下(x2, y2)
        # 这样 IoU 计算就永远合法了
        x_coords = raw_box[:, [0, 2]] # 取出两个 x
        y_coords = raw_box[:, [1, 3]] # 取出两个 y
        
        x1 = torch.min(x_coords, dim=1).values
        y1 = torch.min(y_coords, dim=1).values
        x2 = torch.max(x_coords, dim=1).values
        y2 = torch.max(y_coords, dim=1).values
        
        pred_box = torch.stack([x1, y1, x2, y2], dim=1)
        # ----------------------------

        pred_risk = self.risk_head(out).squeeze(1).squeeze(-1)  # [B]
        
        return pred_box, pred_risk