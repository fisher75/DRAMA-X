"""Stage-2 Risk Selection model (Transformer over proposal tokens).

Design intent
-------------
You are moving to a **two-stage pipeline**:

Stage-1: detector/tracker produces accurate proposals ("框准")
Stage-2: given accurate proposals, select the critical VRU (risk selection / reasoning)

So Stage-2 **does NOT** regress boxes. It only scores proposals.

Implementation (practical)
--------------------------
1) Use Video Swin3D backbone to get a spatiotemporal feature map.
2) Collapse time (avg or last) to get a 2D feature map.
3) ROIAlign each proposal -> per-object feature vector.
4) Build object tokens = [roi_feat, bbox_geom, conf] -> project to d_model.
5) TransformerEncoder over tokens + a learnable query token.
6) Risk score per proposal = dot(query, token) + MLP.

This gives you an explicit "Query" in the model, and you can later replace the
query token with **Driver Tokens** (in-cabin) for true in/out-cabin coupling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torchvision.ops import roi_align

from drama_fast.models.backbones import VideoSwinTinyBackboneMultiScale


@dataclass
class Stage2Config:
    # image size after preprocessing (letterbox or resize)
    img_size: int = 384
    # use which pyramid level for ROI features
    use_level: str = "c4"  # c3/c4/c5
    # roi_align output spatial size
    roi_out: int = 7
    # token dim
    d_model: int = 256
    # transformer
    nhead: int = 8
    num_layers: int = 2
    dim_feedforward: int = 1024
    dropout: float = 0.1
    # token construction
    include_conf: bool = True
    include_geom: bool = True
    # time collapse method: "last" or "mean"
    time_pool: str = "mean"


class RiskSelectionTransformer(nn.Module):
    def __init__(self, cfg: Stage2Config, pretrained_backbone: bool = True):
        super().__init__()
        self.cfg = cfg

        self.backbone = VideoSwinTinyBackboneMultiScale(pretrained=pretrained_backbone)

        # Feature channels per level for Swin-T
        level2c = {"c3": 192, "c4": 384, "c5": 768}
        in_c = level2c.get(cfg.use_level, 384)

        # ROI feature -> token
        roi_feat_dim = in_c * cfg.roi_out * cfg.roi_out
        extra = 0
        if cfg.include_geom:
            extra += 4
        if cfg.include_conf:
            extra += 1
        self.token_proj = nn.Sequential(
            nn.Linear(roi_feat_dim + extra, cfg.d_model),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model, cfg.d_model),
        )

        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=cfg.num_layers, enable_nested_tensor=False)

        # learnable query token (later can be replaced with driver token)
        self.query = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        nn.init.normal_(self.query, std=0.02)

        # scoring head: combine dot(query, token) with MLP
        self.score_mlp = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model, 1),
        )

        # Optional projection for driver token (only used if you later plug in an in-cabin model)
        self.driver_proj: Optional[nn.Linear] = None

    def _pick_level(self, pyr: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        c3, c4, c5 = pyr
        if self.cfg.use_level == "c3":
            return c3
        if self.cfg.use_level == "c5":
            return c5
        return c4

    def _pool_time(self, feat_3d: torch.Tensor) -> torch.Tensor:
        """[B,C,T,H,W] -> [B,C,H,W]"""
        if feat_3d.ndim != 5:
            raise ValueError(f"Expected 5D feat [B,C,T,H,W], got {tuple(feat_3d.shape)}")
        if self.cfg.time_pool == "last":
            return feat_3d[:, :, -1]
        return feat_3d.mean(dim=2)

    def _roi_features(
        self,
        feat2d: torch.Tensor,
        boxes_xyxy_norm: torch.Tensor,
        boxes_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """ROIAlign for normalized boxes.

        Args:
            feat2d: [B,C,Hf,Wf]
            boxes_xyxy_norm: [B,N,4] in [0,1] w.r.t. resized image SxS
            boxes_mask: [B,N] boolean
        Returns:
            roi_flat: [B,N, C*roi_out*roi_out]
            valid_mask: [B,N]
        """
        B, C, Hf, Wf = feat2d.shape
        S = float(self.cfg.img_size)
        spatial_scale = float(Wf) / S  # map image pixels -> feature pixels

        # build rois [K,5] with batch idx, in pixels of resized image
        rois = []
        idx_map = []
        for b in range(B):
            m = boxes_mask[b]
            if m.sum() == 0:
                continue
            boxes = boxes_xyxy_norm[b][m] * S
            # clamp
            boxes = boxes.clamp(0, S)
            bi = torch.full((boxes.shape[0], 1), float(b), device=boxes.device)
            rois.append(torch.cat([bi, boxes], dim=1))
            idx_map.append((b, m.nonzero(as_tuple=False).squeeze(1)))

        if len(rois) == 0:
            # no valid proposals
            empty = feat2d.new_zeros((B, boxes_xyxy_norm.shape[1], C * self.cfg.roi_out * self.cfg.roi_out))
            return empty, boxes_mask

        rois_t = torch.cat(rois, dim=0)
        pooled = roi_align(
            feat2d,
            rois_t,
            output_size=(self.cfg.roi_out, self.cfg.roi_out),
            spatial_scale=spatial_scale,
            sampling_ratio=2,
            aligned=True,
        )  # [K, C, r, r]
        pooled = pooled.flatten(1)  # [K, C*r*r]

        # scatter back to [B,N,...]
        B, N, _ = boxes_xyxy_norm.shape
        out = feat2d.new_zeros((B, N, pooled.shape[1]))
        k0 = 0
        for (b, idxs) in idx_map:
            k1 = k0 + len(idxs)
            out[b, idxs] = pooled[k0:k1]
            k0 = k1
        return out, boxes_mask

    def forward(
        self,
        pixel_values: torch.Tensor,
        proposals: torch.Tensor,
        proposal_mask: torch.Tensor,
        proposal_conf: Optional[torch.Tensor] = None,
        # NOTE: proposal_cls is intentionally NOT used in the minimal Stage-2 model.
        # Keeping the arg here (as **kwargs in Stage2RiskSelector) avoids fragile call-sites.
        driver_tokens: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward.

        Args:
            pixel_values: [B,3,T,S,S]
            proposals: [B,N,4] normalized xyxy
            proposal_mask: [B,N] valid
            proposal_conf: [B,N] optional
            driver_tokens: optional future extension, shape [B,1,D] or [B,D]
        Returns:
            dict with:
              - logits: [B,N]
              - query: [B,D]
              - tokens: [B,N,D]
        """
        _, (c3, c4, c5) = self.backbone(pixel_values)
        feat3d = self._pick_level((c3, c4, c5))  # [B,C,T,H,W]
        feat2d = self._pool_time(feat3d)  # [B,C,H,W]

        roi_flat, valid = self._roi_features(feat2d, proposals, proposal_mask)

        parts = [roi_flat]
        if self.cfg.include_geom:
            parts.append(proposals)
        if self.cfg.include_conf:
            if proposal_conf is None:
                pc = torch.zeros_like(proposals[..., 0])
            else:
                pc = proposal_conf
            parts.append(pc[..., None])
        token_in = torch.cat(parts, dim=-1)  # [B,N, roi+extra]
        tokens = self.token_proj(token_in)  # [B,N,D]

        # prepend query token
        q = self.query.expand(tokens.shape[0], -1, -1)  # [B,1,D]
        if driver_tokens is not None:
            # if provided, replace q with driver token projected to D
            if driver_tokens.ndim == 2:
                driver_tokens = driver_tokens[:, None, :]
            if driver_tokens.shape[-1] != q.shape[-1]:
                if self.driver_proj is None or self.driver_proj.in_features != driver_tokens.shape[-1]:
                    self.driver_proj = nn.Linear(driver_tokens.shape[-1], q.shape[-1]).to(driver_tokens.device)
                driver_tokens = self.driver_proj(driver_tokens)
            q = driver_tokens

        x = torch.cat([q, tokens], dim=1)  # [B, 1+N, D]

        # build padding mask: True means "pad"
        pad_mask = torch.cat(
            [torch.zeros((valid.shape[0], 1), dtype=torch.bool, device=valid.device), ~valid], dim=1
        )

        x = self.encoder(x, src_key_padding_mask=pad_mask)

        q_out = x[:, 0]  # [B,D]
        tok_out = x[:, 1:]  # [B,N,D]

        # score each token. We mix a query-conditioned dot and an MLP score.
        dot = (tok_out * q_out[:, None, :]).sum(dim=-1) / (self.cfg.d_model**0.5)  # [B,N]
        mlp = self.score_mlp(tok_out).squeeze(-1)  # [B,N]
        logits = dot + mlp

        # mask invalid proposals to very negative logits
        logits = logits.masked_fill(~valid, -1e9)

        return {
            "logits": logits,
            "query": q_out,
            "tokens": tok_out,
            "valid_mask": valid,
        }


# -----------------------------------------------------------------------------
# Backward-compatible API
# -----------------------------------------------------------------------------

class Stage2RiskSelector(nn.Module):
    """Stage-2 wrapper.

    This thin module exists so the trainer has a stable import name.
    It wraps :class:`RiskSelectionTransformer` and enforces a maximum number
    of proposal tokens (N) at runtime.

    Inputs:
        pixel_values: [B, T, 3, H, W] float32 in [0,1]
        proposals:    [B, N, 5] with bbox in **normalized xyxy** + conf
        proposal_mask:[B, N] boolean mask (True = valid)

    Outputs:
        logits: [B, N] risk logits for each proposal
    """

    def __init__(
        self,
        pretrained_backbone: bool = True,
        max_tokens: int = 50,
        d_model: int = 256,
        nhead: int = 8,
        depth: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.max_tokens = int(max_tokens)

        # Stage2Config uses `num_layers` (not `depth`). Keep a wrapper arg named
        # `depth` because that's what we expose on CLI.
        cfg = Stage2Config(
            d_model=d_model,
            nhead=nhead,
            num_layers=depth,
            dropout=dropout,
        )
        self.net = RiskSelectionTransformer(cfg=cfg, pretrained_backbone=pretrained_backbone)

    def forward(
        self,
        pixel_values: torch.Tensor,
        proposals: torch.Tensor = None,
        proposal_mask: torch.Tensor = None,
        boxes: torch.Tensor = None,
        box_mask: torch.Tensor = None,
        proposal_conf: torch.Tensor = None,
                **kwargs,
    ):
        """Forward for Stage 2 risk selector.

        This method is intentionally permissive to avoid fragile call-sites.

        Supported input names:
          - proposals / proposal_mask  (preferred)
          - boxes / box_mask          (alias)

        Expected shapes:
          - proposals:     [B, N, 4] normalized xyxy in [0, 1]
          - proposal_conf: [B, N] or [B, N, 1] (optional)
          - proposal_cls:  [B, N] (optional, COCO class id)
        """

        if proposals is None:
            proposals = boxes
        if proposal_mask is None:
            proposal_mask = box_mask

        if proposals is None:
            raise ValueError("Stage2RiskSelector.forward: proposals (or boxes) must be provided.")

        # Be permissive about proposal shapes:
        # - Preferred: [B, N, 4] normalized xyxy
        # - If a caller provides [B, N, 5] (xyxy + conf), we drop the last dim and
        #   use it as proposal_conf when missing.
        if proposals.shape[-1] == 5:
            if proposal_conf is None:
                proposal_conf = proposals[..., 4]
            proposals = proposals[..., :4]

        # Some datasets/callers may pack conf as [B,N,1]. Normalize to [B,N].
        if proposal_conf is not None and proposal_conf.dim() == 3 and proposal_conf.shape[-1] == 1:
            proposal_conf = proposal_conf.squeeze(-1)

        # NOTE: proposal_cls is currently ignored. (Keep it for future token construction.)

        # Enforce a maximum number of proposal tokens to bound compute.
        # If we have confidence scores, keep the top-K highest confidence proposals.
        B, N, _ = proposals.shape
        if N > self.max_tokens:
            if proposal_conf is not None:
                topk = torch.topk(proposal_conf, k=self.max_tokens, dim=1).indices  # [B,K]
                proposals = torch.gather(proposals, 1, topk[..., None].expand(-1, -1, 4))
                proposal_mask = torch.gather(proposal_mask, 1, topk)
                proposal_conf = torch.gather(proposal_conf, 1, topk)
            else:
                proposals = proposals[:, : self.max_tokens]
                proposal_mask = proposal_mask[:, : self.max_tokens]

        out = self.net(
            pixel_values=pixel_values,
            proposals=proposals,
            proposal_mask=proposal_mask,
            proposal_conf=proposal_conf,
            **kwargs,
        )

        # RiskSelectionTransformer returns a dict; trainers/evaluators expect logits tensor.
        if isinstance(out, dict):
            return out["logits"]
        return out

