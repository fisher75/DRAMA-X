"""Phase 1.5 Fast OD model: Video Swin-T + True FPN (P3/P4/P5) + FCOS head.

Key inherited fixes from Phase 1:
- Last-frame selection for temporal alignment (C3/C4/C5 -> take [:,:, -1]).
- Letterbox + Flip handled in dataset; model expects coordinates already in resized square.
- Score fusion = sigmoid(cls) * sigmoid(ctr), NMS in inference.

This file is intentionally self-contained to avoid breaking Phase-1.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones import VideoSwinTinyBackboneMultiScale


# -----------------------------
# Utilities
# -----------------------------

def _nms_xyxy(boxes: torch.Tensor, scores: torch.Tensor, iou_thr: float) -> torch.Tensor:
    """Pure PyTorch NMS for XYXY boxes.

    Args:
        boxes: [N,4] float, XYXY
        scores: [N] float
        iou_thr: float

    Returns:
        keep indices [K]
    """
    if boxes.numel() == 0:
        return boxes.new_zeros((0,), dtype=torch.long)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    order = scores.argsort(descending=True)

    keep = []
    while order.numel() > 0:
        i = order[0]
        keep.append(i)
        if order.numel() == 1:
            break

        rest = order[1:]
        xx1 = torch.maximum(x1[i], x1[rest])
        yy1 = torch.maximum(y1[i], y1[rest])
        xx2 = torch.minimum(x2[i], x2[rest])
        yy2 = torch.minimum(y2[i], y2[rest])

        inter_w = (xx2 - xx1).clamp(min=0)
        inter_h = (yy2 - yy1).clamp(min=0)
        inter = inter_w * inter_h
        union = areas[i] + areas[rest] - inter
        iou = inter / union.clamp(min=1e-6)

        order = rest[iou <= iou_thr]

    return torch.stack(keep) if len(keep) > 0 else boxes.new_zeros((0,), dtype=torch.long)


def _box_area_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    return (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (boxes[:, 3] - boxes[:, 1]).clamp(min=0)


def _giou_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """GIoU loss for XYXY boxes, returns [N]"""
    # intersection
    x1 = torch.maximum(pred[:, 0], target[:, 0])
    y1 = torch.maximum(pred[:, 1], target[:, 1])
    x2 = torch.minimum(pred[:, 2], target[:, 2])
    y2 = torch.minimum(pred[:, 3], target[:, 3])

    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    area_p = _box_area_xyxy(pred)
    area_t = _box_area_xyxy(target)
    union = area_p + area_t - inter
    iou = inter / union.clamp(min=eps)

    # enclosing
    cx1 = torch.minimum(pred[:, 0], target[:, 0])
    cy1 = torch.minimum(pred[:, 1], target[:, 1])
    cx2 = torch.maximum(pred[:, 2], target[:, 2])
    cy2 = torch.maximum(pred[:, 3], target[:, 3])
    c_area = (cx2 - cx1).clamp(min=0) * (cy2 - cy1).clamp(min=0)

    giou = iou - (c_area - union) / c_area.clamp(min=eps)
    return 1.0 - giou


class SigmoidFocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits/targets: [B,N]
        p = torch.sigmoid(logits)
        ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = p * targets + (1.0 - p) * (1.0 - targets)
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        loss = alpha_t * (1.0 - p_t).pow(self.gamma) * ce
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


# -----------------------------
# FPN
# -----------------------------

class FPN(nn.Module):
    """Standard top-down FPN for 3 inputs (C3/C4/C5) -> (P3/P4/P5)."""

    def __init__(self, in_channels: Tuple[int, int, int], out_channels: int = 256):
        super().__init__()
        c3, c4, c5 = in_channels

        self.lateral3 = nn.Conv2d(c3, out_channels, kernel_size=1)
        self.lateral4 = nn.Conv2d(c4, out_channels, kernel_size=1)
        self.lateral5 = nn.Conv2d(c5, out_channels, kernel_size=1)

        self.out3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.out4 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.out5 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, c3: torch.Tensor, c4: torch.Tensor, c5: torch.Tensor) -> List[torch.Tensor]:
        # Laterals
        p5 = self.lateral5(c5)
        p4 = self.lateral4(c4) + F.interpolate(p5, size=c4.shape[-2:], mode="nearest")
        p3 = self.lateral3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode="nearest")

        # Output convs
        p5 = self.out5(p5)
        p4 = self.out4(p4)
        p3 = self.out3(p3)
        return [p3, p4, p5]


# -----------------------------
# FCOS Head (multi-level + per-level Scale)
# -----------------------------

class Scale(nn.Module):
    def __init__(self, init_value: float = 1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(init_value, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale


class FCOSHeadMulti(nn.Module):
    def __init__(self, feat_channels: int = 256, num_levels: int = 3, prior_prob: float = 0.01):
        super().__init__()
        self.num_levels = num_levels

        self.cls_tower = self._make_tower(feat_channels)
        self.bbox_tower = self._make_tower(feat_channels)

        self.cls_logits = nn.Conv2d(feat_channels, 1, kernel_size=3, padding=1)
        self.bbox_pred = nn.Conv2d(feat_channels, 4, kernel_size=3, padding=1)
        self.ctr_logits = nn.Conv2d(feat_channels, 1, kernel_size=3, padding=1)

        # Per-level scale (key Phase 1.5 improvement)
        self.scales = nn.ModuleList([Scale(1.0) for _ in range(num_levels)])

        # Bias init for focal loss stability
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.cls_logits.bias, bias_value)

        for m in [self.cls_logits, self.bbox_pred, self.ctr_logits]:
            nn.init.normal_(m.weight, std=0.01)

    def _make_tower(self, feat_channels: int) -> nn.Sequential:
        layers = []
        for _ in range(4):
            layers.append(nn.Conv2d(feat_channels, feat_channels, kernel_size=3, padding=1))
            layers.append(nn.GroupNorm(32, feat_channels))
            layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def forward(self, feats: List[torch.Tensor]) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        cls_list, reg_list, ctr_list = [], [], []
        assert len(feats) == self.num_levels, f"Expected {self.num_levels} levels, got {len(feats)}"

        for lvl, f in enumerate(feats):
            cls_feat = self.cls_tower(f)
            bbox_feat = self.bbox_tower(f)

            cls = self.cls_logits(cls_feat)
            ctr = self.ctr_logits(bbox_feat)

            reg = self.bbox_pred(bbox_feat)
            reg = self.scales[lvl](reg)
            reg = F.relu(reg)  # l,t,r,b >= 0

            cls_list.append(cls)
            reg_list.append(reg)
            ctr_list.append(ctr)

        return cls_list, reg_list, ctr_list


# -----------------------------
# Model
# -----------------------------

@dataclass
class Phase1_5Config:
    img_size: int = 384
    feat_channels: int = 256
    pretrained_backbone: bool = True
    num_classes: int = 1  # VRU-only; Phase1.5 currently supports only 1 class

    # FCOS
    strides: Tuple[int, int, int] = (8, 16, 32)  # P3/P4/P5
    reg_ranges: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]] = (
        (0.0, 64.0),
        (64.0, 128.0),
        (128.0, 1e8),
    )
    center_sampling_radius: float = 1.5

    # Inference
    score_thr: float = 0.05
    nms_thr: float = 0.6
    topk: int = 200


class Phase1_5VideoSwinFCOS(nn.Module):
    def __init__(self, cfg: Phase1_5Config = Phase1_5Config()):
        super().__init__()
        self.cfg = cfg
        if int(cfg.num_classes) != 1:
            raise ValueError("Phase1_5VideoSwinFCOS currently supports num_classes=1 only.")

        self.backbone = VideoSwinTinyBackboneMultiScale(pretrained=cfg.pretrained_backbone)
        self.fpn = FPN(in_channels=self.backbone.out_channels, out_channels=cfg.feat_channels)
        self.head = FCOSHeadMulti(feat_channels=cfg.feat_channels, num_levels=3)

    @staticmethod
    def _points_for_level(h: int, w: int, stride: int, device: torch.device) -> torch.Tensor:
        ys = (torch.arange(h, device=device, dtype=torch.float32) + 0.5) * stride
        xs = (torch.arange(w, device=device, dtype=torch.float32) + 0.5) * stride
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        return torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)  # [HW,2]

    def forward(self, clip: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Accept both [B,3,T,H,W] and [B,T,3,H,W]
        if clip.ndim == 5 and clip.shape[1] != 3 and clip.shape[2] == 3:
            clip = clip.permute(0, 2, 1, 3, 4).contiguous()
        """Forward.

        Args:
            clip: [B,3,T,H,W]

        Returns:
            dict with flattened predictions across levels.
        """
        tokens, (c3, c4, c5) = self.backbone(clip)

        # Temporal alignment: last-frame selection BEFORE FPN
        # All features are standardized to [B,C,T,H,W]
        c3_2d = c3[:, :, -1]
        c4_2d = c4[:, :, -1]
        c5_2d = c5[:, :, -1]

        feats = self.fpn(c3_2d, c4_2d, c5_2d)  # [P3,P4,P5]
        cls_lvls, reg_lvls, ctr_lvls = self.head(feats)

        # Flatten and concat across levels
        B = clip.shape[0]
        cls_all, reg_all, ctr_all = [], [], []
        points_all, strides_all, ranges_all = [], [], []

        for lvl, (cls, reg, ctr) in enumerate(zip(cls_lvls, reg_lvls, ctr_lvls)):
            stride = self.cfg.strides[lvl]
            reg_lo, reg_hi = self.cfg.reg_ranges[lvl]

            # cls: [B,1,H,W] -> [B,HW]
            cls = cls.flatten(2).squeeze(1)
            ctr = ctr.flatten(2).squeeze(1)
            reg = reg.permute(0, 2, 3, 1).reshape(B, -1, 4)

            h, w = cls_lvls[lvl].shape[-2:]
            pts = self._points_for_level(h, w, stride, device=clip.device)

            cls_all.append(cls)
            ctr_all.append(ctr)
            reg_all.append(reg)
            points_all.append(pts)
            strides_all.append(torch.full((pts.shape[0],), float(stride), device=clip.device))
            ranges_all.append(torch.tensor([reg_lo, reg_hi], device=clip.device).repeat(pts.shape[0], 1))

        out = {
            "tokens": tokens,
            "cls_logits": torch.cat(cls_all, dim=1),
            "ctr_logits": torch.cat(ctr_all, dim=1),
            "reg_preds": torch.cat(reg_all, dim=1),
            "points": torch.cat(points_all, dim=0),
            "strides": torch.cat(strides_all, dim=0),
            "reg_ranges": torch.cat(ranges_all, dim=0),
        }
        return out

    @torch.no_grad()
    def inference(
        self,
        clip: torch.Tensor,
        score_thr: float | None = None,
        nms_thr: float | None = None,
        topk: int | None = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        self.eval()
        score_thr = self.cfg.score_thr if score_thr is None else score_thr
        nms_thr = self.cfg.nms_thr if nms_thr is None else nms_thr
        topk = self.cfg.topk if topk is None else topk

        out = self.forward(clip)
        cls = torch.sigmoid(out["cls_logits"])  # [B,N]
        ctr = torch.sigmoid(out["ctr_logits"])  # [B,N]
        scores = cls * ctr

        reg = out["reg_preds"]  # [B,N,4]
        points = out["points"]  # [N,2]

        all_boxes: List[torch.Tensor] = []
        all_scores: List[torch.Tensor] = []

        for b in range(clip.shape[0]):
            # Accept both [B,3,T,H,W] and [B,T,3,H,W]
            if clip.ndim == 5 and clip.shape[1] != 3 and clip.shape[2] == 3:
                clip = clip.permute(0, 2, 1, 3, 4).contiguous()
            s = scores[b]
            r = reg[b]

            keep = s > score_thr
            if keep.sum() == 0:
                all_boxes.append(torch.zeros((0, 4), device=clip.device))
                all_scores.append(torch.zeros((0,), device=clip.device))
                continue

            s = s[keep]
            r = r[keep]
            p = points[keep]

            # topk pre-NMS
            if s.numel() > topk:
                top_idx = torch.topk(s, k=topk, largest=True).indices
                s = s[top_idx]
                r = r[top_idx]
                p = p[top_idx]

            # decode ltrb -> xyxy
            x1 = p[:, 0] - r[:, 0]
            y1 = p[:, 1] - r[:, 1]
            x2 = p[:, 0] + r[:, 2]
            y2 = p[:, 1] + r[:, 3]
            boxes = torch.stack([x1, y1, x2, y2], dim=1)

            # NMS
            keep_idx = _nms_xyxy(boxes, s, nms_thr)
            boxes = boxes[keep_idx]
            s = s[keep_idx]

            all_boxes.append(boxes)
            all_scores.append(s)

        results = []
        for boxes, scores in zip(all_boxes, all_scores):
            results.append({
                "boxes": boxes,
                "scores": scores,
                "labels": torch.zeros((boxes.shape[0],), device=boxes.device, dtype=torch.long),
            })
        return results



# -----------------------------
# Loss
# -----------------------------

class Phase1_5FCOSLoss(nn.Module):
    def __init__(
        self,
        center_sampling_radius: float = 1.5,
        cls_weight: float = 1.0,
        reg_weight: float = 1.0,
        ctr_weight: float = 1.0,
    ):
        super().__init__()
        self.center_sampling_radius = center_sampling_radius
        self.cls_weight = cls_weight
        self.reg_weight = reg_weight
        self.ctr_weight = ctr_weight

        self.focal = SigmoidFocalLoss(alpha=0.25, gamma=2.0, reduction="sum")

    @staticmethod
    def _centerness_from_ltrb(l: torch.Tensor, t: torch.Tensor, r: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        left_right = torch.stack([l, r], dim=-1)
        top_bottom = torch.stack([t, b], dim=-1)
        centerness = (
            (left_right.min(dim=-1).values / left_right.max(dim=-1).values.clamp(min=1e-6))
            * (top_bottom.min(dim=-1).values / top_bottom.max(dim=-1).values.clamp(min=1e-6))
        )
        return torch.sqrt(centerness.clamp(min=0.0, max=1.0))

    def _assign_single(
        self,
        points: torch.Tensor,          # [N,2]
        strides: torch.Tensor,         # [N]
        reg_ranges: torch.Tensor,      # [N,2]
        gt_boxes: torch.Tensor,        # [M,4]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Assign targets for one image."""
        device = points.device
        N = points.shape[0]

        if gt_boxes.numel() == 0:
            cls_t = torch.zeros((N,), device=device)
            reg_t = torch.zeros((N, 4), device=device)
            ctr_t = torch.zeros((N,), device=device)
            return cls_t, reg_t, ctr_t

        M = gt_boxes.shape[0]
        px = points[:, 0].unsqueeze(1)  # [N,1]
        py = points[:, 1].unsqueeze(1)

        x1 = gt_boxes[:, 0].unsqueeze(0)  # [1,M]
        y1 = gt_boxes[:, 1].unsqueeze(0)
        x2 = gt_boxes[:, 2].unsqueeze(0)
        y2 = gt_boxes[:, 3].unsqueeze(0)

        l = px - x1
        t = py - y1
        r = x2 - px
        b = y2 - py

        reg = torch.stack([l, t, r, b], dim=2)  # [N,M,4]

        # inside box
        inside = (reg.min(dim=2).values > 0)

        # center sampling
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        radius = self.center_sampling_radius
        stride = strides.unsqueeze(1)  # [N,1]
        rdx = radius * stride
        rdy = radius * stride

        c_x1 = cx - rdx
        c_y1 = cy - rdy
        c_x2 = cx + rdx
        c_y2 = cy + rdy

        in_center = (px >= c_x1) & (px <= c_x2) & (py >= c_y1) & (py <= c_y2)

        # size-of-interest range per point
        max_reg = reg.max(dim=2).values  # [N,M]
        lo = reg_ranges[:, 0].unsqueeze(1)
        hi = reg_ranges[:, 1].unsqueeze(1)
        in_range = (max_reg >= lo) & (max_reg <= hi)

        valid = inside & in_center & in_range

        # choose gt with smallest area among valid
        areas = ((x2 - x1) * (y2 - y1)).clamp(min=1e-6)  # [1,M]
        areas = areas.expand(N, M).clone()
        areas[~valid] = float("inf")
        min_area, min_idx = areas.min(dim=1)

        is_pos = torch.isfinite(min_area)

        cls_t = torch.zeros((N,), device=device)
        cls_t[is_pos] = 1.0

        reg_t = torch.zeros((N, 4), device=device)
        if is_pos.any():
            reg_pos = reg[torch.arange(N, device=device), min_idx]  # [N,4]
            reg_t[is_pos] = reg_pos[is_pos]

        ctr_t = torch.zeros((N,), device=device)
        if is_pos.any():
            lpos = reg_t[is_pos, 0]
            tpos = reg_t[is_pos, 1]
            rpos = reg_t[is_pos, 2]
            bpos = reg_t[is_pos, 3]
            ctr_t[is_pos] = self._centerness_from_ltrb(lpos, tpos, rpos, bpos)

        return cls_t, reg_t, ctr_t

    # def forward(self, outputs: Dict[str, torch.Tensor], gt_boxes_list: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
    def forward(self, outputs, gt_boxes_list, gt_labels_list=None, img_hw=None):
        cls_logits = outputs["cls_logits"]  # [B,N]
        ctr_logits = outputs["ctr_logits"]  # [B,N]
        reg_preds = outputs["reg_preds"]    # [B,N,4]
        points = outputs["points"]          # [N,2]
        strides = outputs["strides"]        # [N]
        reg_ranges = outputs["reg_ranges"]  # [N,2]

        B, N = cls_logits.shape

        cls_targets = []
        reg_targets = []
        ctr_targets = []

        for b in range(B):
            gt = gt_boxes_list[b]
            if isinstance(gt, list):
                gt = torch.tensor(gt, device=cls_logits.device, dtype=torch.float32)
            cls_t, reg_t, ctr_t = self._assign_single(points, strides, reg_ranges, gt)
            cls_targets.append(cls_t)
            reg_targets.append(reg_t)
            ctr_targets.append(ctr_t)

        cls_targets = torch.stack(cls_targets, dim=0)  # [B,N]
        reg_targets = torch.stack(reg_targets, dim=0)  # [B,N,4]
        ctr_targets = torch.stack(ctr_targets, dim=0)  # [B,N]

        pos_mask = cls_targets > 0.5
        num_pos = pos_mask.sum().clamp(min=1).float()

        # cls focal
        cls_loss = self.focal(cls_logits, cls_targets) / num_pos

        # centerness BCE (only pos)
        if pos_mask.any():
            ctr_loss = F.binary_cross_entropy_with_logits(ctr_logits[pos_mask], ctr_targets[pos_mask], reduction="sum") / num_pos
        else:
            ctr_loss = ctr_logits.sum() * 0.0

        # regression GIoU (only pos)
        if pos_mask.any():
            # decode predicted boxes
            p = points.unsqueeze(0).expand(B, N, 2)
            # predicted ltrb
            r = reg_preds
            pred_xyxy = torch.stack(
                [p[..., 0] - r[..., 0], p[..., 1] - r[..., 1], p[..., 0] + r[..., 2], p[..., 1] + r[..., 3]],
                dim=-1,
            )
            # target boxes
            t = reg_targets
            tgt_xyxy = torch.stack(
                [p[..., 0] - t[..., 0], p[..., 1] - t[..., 1], p[..., 0] + t[..., 2], p[..., 1] + t[..., 3]],
                dim=-1,
            )

            reg_loss = _giou_loss(pred_xyxy[pos_mask], tgt_xyxy[pos_mask]).sum() / num_pos
        else:
            reg_loss = reg_preds.sum() * 0.0

        total = self.cls_weight * cls_loss + self.reg_weight * reg_loss + self.ctr_weight * ctr_loss

        return {
            "loss": total,
            "loss_cls": cls_loss.detach(),
            "loss_reg": reg_loss.detach(),
            "loss_ctr": ctr_loss.detach(),
            "num_pos": num_pos.detach(),
        }