"""Phase-1 Fast System (Route-2): Video Swin + Pseudo-FPN + FCOS head.

This file implements a lightweight FCOS-style dense detector on top of the
existing Video Swin backbone (timm's video_swin_tiny). It is designed to:

- Produce many candidate boxes + scores efficiently (dense, YOLO-like behaviour).
- Keep 3D (video) backbone to preserve temporal cues.
- Keep code self-contained: no torchvision.ops dependency.

Important limitation
--------------------
Your current supervision jsonl (derived from updated_output.json) provides
bounding boxes only for VRUs (pedestrians/cyclists). It does NOT annotate cars,
trucks, etc. Therefore this detector is trained as a **VRU detector**.

If you later want 'all objects', you must add labels (or use pseudo-labels /
teacher OD model). Otherwise unlabeled cars would become false negatives.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from drama_fast.models.backbones import build_backbone
# -------------------------
# Basic box ops (pure torch)
# -------------------------

def _box_area_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """boxes: [..., 4] in (x1,y1,x2,y2)"""
    return (boxes[..., 2] - boxes[..., 0]).clamp(min=0) * (boxes[..., 3] - boxes[..., 1]).clamp(min=0)


def box_iou_xyxy(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Pairwise IoU for xyxy boxes.

    boxes1: [N,4], boxes2: [M,4]
    returns: [N,M]
    """
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return boxes1.new_zeros((boxes1.shape[0], boxes2.shape[0]))

    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])  # [N,M,2]
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    area1 = _box_area_xyxy(boxes1)[:, None]
    area2 = _box_area_xyxy(boxes2)[None, :]
    union = area1 + area2 - inter
    return inter / union.clamp(min=1e-6)


def generalized_box_iou_xyxy(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Generalized IoU (GIoU), pairwise."""
    iou = box_iou_xyxy(boxes1, boxes2)
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return iou

    lt = torch.min(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    area_c = wh[..., 0] * wh[..., 1]

    area1 = _box_area_xyxy(boxes1)[:, None]
    area2 = _box_area_xyxy(boxes2)[None, :]

    lt_i = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    rb_i = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh_i = (rb_i - lt_i).clamp(min=0)
    inter = wh_i[..., 0] * wh_i[..., 1]
    union = area1 + area2 - inter

    return iou - (area_c - union) / area_c.clamp(min=1e-6)


def _nms_xyxy(boxes: torch.Tensor, scores: torch.Tensor, iou_thr: float) -> torch.Tensor:
    """Pure-torch NMS. Returns kept indices."""
    if boxes.numel() == 0:
        return boxes.new_zeros((0,), dtype=torch.long)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    order = scores.argsort(descending=True)

    keep: List[int] = []
    while order.numel() > 0:
        i = int(order[0])
        keep.append(i)
        if order.numel() == 1:
            break
        rest = order[1:]

        xx1 = torch.max(x1[i], x1[rest])
        yy1 = torch.max(y1[i], y1[rest])
        xx2 = torch.min(x2[i], x2[rest])
        yy2 = torch.min(y2[i], y2[rest])

        w = (xx2 - xx1).clamp(min=0)
        h = (yy2 - yy1).clamp(min=0)
        inter = w * h
        iou = inter / (areas[i] + areas[rest] - inter).clamp(min=1e-6)

        order = rest[iou <= iou_thr]

    return boxes.new_tensor(keep, dtype=torch.long)


# -------------------------
# FCOS targets + losses
# -------------------------

@dataclass
class FCOSConfig:
    num_classes: int = 1  # VRU only # if =2 then 0:ped, 1:cyc
    feat_channels: int = 256
    strides: Tuple[int, int, int] = (8, 16, 32)
    sizes_of_interest: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]] = (
        (0.0, 64.0),
        (64.0, 128.0),
        (128.0, 1e8),
    )
    center_sampling_radius: float = 1.5
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0


def sigmoid_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = "sum",
) -> torch.Tensor:
    """Binary sigmoid focal loss, supports multi-label (targets in {0,1})."""
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce * ((1 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    if reduction == "sum":
        return loss.sum()
    if reduction == "mean":
        return loss.mean()
    return loss


def _make_grid_xy(H: int, W: int, stride: int, device: torch.device) -> torch.Tensor:
    """Return [HW,2] point centers (x,y) in input pixel coords."""
    ys = (torch.arange(H, device=device, dtype=torch.float32) + 0.5) * stride
    xs = (torch.arange(W, device=device, dtype=torch.float32) + 0.5) * stride
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)


def fcos_assign_single_image(
    gt_boxes_xyxy: torch.Tensor,
    gt_labels: torch.Tensor,
    feat_sizes: List[Tuple[int, int]],
    cfg: FCOSConfig,
    device: torch.device,
) -> Dict[str, List[torch.Tensor]]:
    """Assign FCOS targets for one image.

    Returns dict with per-level tensors:
      - cls_target: [HW, C] multi-hot? Actually single-class per location (one-hot).
      - reg_target: [HW, 4] ltrb distances
      - ctr_target: [HW, 1]
      - pos_mask:   [HW] bool
    """
    num_gts = gt_boxes_xyxy.shape[0]

    out = {"cls": [], "reg": [], "ctr": [], "pos": []}

    if num_gts == 0:
        # No GTs => all background
        for (H, W) in feat_sizes:
            HW = H * W
            out["cls"].append(torch.zeros((HW, cfg.num_classes), device=device))
            out["reg"].append(torch.zeros((HW, 4), device=device))
            out["ctr"].append(torch.zeros((HW, 1), device=device))
            out["pos"].append(torch.zeros((HW,), device=device, dtype=torch.bool))
        return out

    gt_areas = _box_area_xyxy(gt_boxes_xyxy)  # [G]

    for level, (H, W) in enumerate(feat_sizes):
        stride = cfg.strides[level]
        lower, upper = cfg.sizes_of_interest[level]

        points = _make_grid_xy(H, W, stride, device=device)  # [P,2]
        P = points.shape[0]

        # [P,G]
        xs = points[:, 0][:, None]
        ys = points[:, 1][:, None]
        x1 = gt_boxes_xyxy[None, :, 0]
        y1 = gt_boxes_xyxy[None, :, 1]
        x2 = gt_boxes_xyxy[None, :, 2]
        y2 = gt_boxes_xyxy[None, :, 3]

        l = xs - x1
        t = ys - y1
        r = x2 - xs
        b = y2 - ys
        reg = torch.stack([l, t, r, b], dim=-1)  # [P,G,4]

        inside_box = reg.min(dim=-1).values > 0

        # size-of-interest
        max_reg = reg.max(dim=-1).values
        in_level = (max_reg >= lower) & (max_reg <= upper)

        # center sampling
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        radius = cfg.center_sampling_radius * stride
        cx1 = cx - radius
        cy1 = cy - radius
        cx2 = cx + radius
        cy2 = cy + radius

        in_center = (xs >= cx1) & (xs <= cx2) & (ys >= cy1) & (ys <= cy2)
        in_center = in_center.squeeze(-1) if in_center.ndim == 3 else in_center

        # valid positions
        is_pos = inside_box & in_level & in_center

        # choose GT with smallest area
        areas = gt_areas[None, :].expand(P, num_gts).clone()
        areas[~is_pos] = float("inf")

        min_area, min_ind = areas.min(dim=1)
        pos_mask = torch.isfinite(min_area)

        # build targets
        cls_t = torch.zeros((P, cfg.num_classes), device=device)
        reg_t = torch.zeros((P, 4), device=device)
        ctr_t = torch.zeros((P, 1), device=device)

        if pos_mask.any():
            chosen = min_ind[pos_mask]  # [Np]
            reg_chosen = reg[pos_mask, chosen, :]  # [Np,4]
            reg_t[pos_mask] = reg_chosen

            # one-hot cls
            lab = gt_labels[chosen].long()  # [Np]
            cls_t[pos_mask, lab] = 1.0

            # centerness
            l_, t_, r_, b_ = reg_chosen.unbind(dim=-1)
            ctr = (
                (torch.min(l_, r_) / torch.max(l_, r_).clamp(min=1e-6))
                * (torch.min(t_, b_) / torch.max(t_, b_).clamp(min=1e-6))
            ).clamp(min=0.0)
            ctr_t[pos_mask, 0] = torch.sqrt(ctr)

        out["cls"].append(cls_t)
        out["reg"].append(reg_t)
        out["ctr"].append(ctr_t)
        out["pos"].append(pos_mask)

    return out


# -------------------------
# Pseudo-FPN + FCOS head
# -------------------------

class ConvGNReLU(nn.Module):
    def __init__(self, c_in: int, c_out: int, k: int = 3, s: int = 1, p: int = 1, groups: int = 32):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, k, s, p, bias=False)
        # 256 channels -> 32 groups works well under small per-GPU batch sizes.
        self.gn = nn.GroupNorm(num_groups=min(groups, c_out), num_channels=c_out)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.gn(self.conv(x)))


class PseudoFPN(nn.Module):
    """A minimal neck that upsamples the last-stage feature to multi-scale maps.

    It is NOT a true FPN (no lateral features from earlier stages). We keep it
    intentionally simple/fast as requested.

    Input:  f2d [B, C_in, H5, W5] (usually stride-32)
    Output: [P3(stride8), P4(stride16), P5(stride32)]
    """

    def __init__(self, c_in: int, c_out: int = 256):
        super().__init__()
        self.p5_lateral = nn.Conv2d(c_in, c_out, kernel_size=1)
        self.p5_out = ConvGNReLU(c_out, c_out, k=3, s=1, p=1)

        self.p4_out = ConvGNReLU(c_out, c_out, k=3, s=1, p=1)
        self.p3_out = ConvGNReLU(c_out, c_out, k=3, s=1, p=1)

    def forward(self, f2d: torch.Tensor) -> List[torch.Tensor]:
        p5 = self.p5_out(self.p5_lateral(f2d))
        p4 = F.interpolate(p5, scale_factor=2.0, mode="bilinear", align_corners=False)
        p4 = self.p4_out(p4)
        p3 = F.interpolate(p4, scale_factor=2.0, mode="bilinear", align_corners=False)
        p3 = self.p3_out(p3)
        return [p3, p4, p5]


class FCOSHead(nn.Module):
    def __init__(self, num_classes: int, feat_channels: int = 256, num_convs: int = 4):
        super().__init__()
        self.cls_tower = nn.Sequential(*[ConvGNReLU(feat_channels, feat_channels) for _ in range(num_convs)])
        self.reg_tower = nn.Sequential(*[ConvGNReLU(feat_channels, feat_channels) for _ in range(num_convs)])

        self.cls_logits = nn.Conv2d(feat_channels, num_classes, kernel_size=3, stride=1, padding=1)
        self.bbox_pred = nn.Conv2d(feat_channels, 4, kernel_size=3, stride=1, padding=1)
        self.ctrness = nn.Conv2d(feat_channels, 1, kernel_size=3, stride=1, padding=1)

        # init: make early training stable
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.cls_logits.bias, bias_value)
        nn.init.normal_(self.cls_logits.weight, std=0.01)
        nn.init.normal_(self.bbox_pred.weight, std=0.01)
        nn.init.constant_(self.bbox_pred.bias, 0.0)
        nn.init.normal_(self.ctrness.weight, std=0.01)
        nn.init.constant_(self.ctrness.bias, 0.0)

    def forward(self, feats: List[torch.Tensor]) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        cls_outs: List[torch.Tensor] = []
        reg_outs: List[torch.Tensor] = []
        ctr_outs: List[torch.Tensor] = []

        for x in feats:
            cls_feat = self.cls_tower(x)
            reg_feat = self.reg_tower(x)

            cls_outs.append(self.cls_logits(cls_feat))
            # FCOS typically uses exp / relu to keep positive distances. ReLU is simpler.
            reg_outs.append(F.relu(self.bbox_pred(reg_feat)))
            ctr_outs.append(self.ctrness(reg_feat))

        return cls_outs, reg_outs, ctr_outs


# -------------------------
# Main model
# -------------------------

class Phase1VideoSwinFCOS(nn.Module):
    def __init__(self, cfg: Optional[FCOSConfig] = None, backbone_name: str = "swin3d_t", pretrained: bool = True):
        super().__init__()
        self.cfg = cfg or FCOSConfig()
        # Use the repo's torchvision Video Swin 3D backbone (already used in Phase-1).
        # NOTE: our dataset provides clips as [B, T, C, H, W], and the backbone will
        # internally permute to [B, C, T, H, W] for torchvision.
        self.backbone = build_backbone(backbone_name, pretrained=pretrained)

        # Use the backbone's last-stage channel dim to build the neck.
        c_in = getattr(self.backbone, "hidden_dim", 768)
        self.neck = PseudoFPN(c_in=c_in, c_out=self.cfg.feat_channels)
        self.head = FCOSHead(num_classes=self.cfg.num_classes, feat_channels=self.cfg.feat_channels)

    def forward(self, clip: torch.Tensor) -> Dict[str, List[torch.Tensor]]:
        # Ensure torchvision Swin3D input: [B,C,T,H,W]
        # clip could be [B,3,T,H,W] or [B,T,3,H,W]
        if clip.ndim != 5:
            raise RuntimeError(f"Expected 5D clip, got {tuple(clip.shape)}")

        if clip.shape[1] == 3:
            pass  # already [B,3,T,H,W]
        elif clip.shape[2] == 3:
            clip = clip.permute(0, 2, 1, 3, 4).contiguous()  # [B,T,3,H,W] -> [B,3,T,H,W]
        else:
            raise RuntimeError(f"Unexpected clip shape: {tuple(clip.shape)} (cannot find channel dim=3)")


        # backbone returns (tokens_flat, feat_3d)
        tokens, feat_3d = self.backbone(clip)  # tokens: [B,L,D], feat_3d: [B,D,t,h,w]

        # Convert 3D feature grid -> 2D map for neck/head
        if feat_3d.ndim != 5:
            raise RuntimeError(f"Expected feat_3d 5D [B,D,t,h,w], got {tuple(feat_3d.shape)}")
        # 只用最后一帧特征，与 keyframe/GT 对齐
        f2d = feat_3d[:, :, -1]   # [B, C, H, W]


        feats = self.neck(f2d)
        cls_outs, reg_outs, ctr_outs = self.head(feats)
        return {"cls": cls_outs, "reg": reg_outs, "ctr": ctr_outs}



    @torch.no_grad()
    def inference(
        self,
        clip: torch.Tensor,
        img_hw: Tuple[int, int],
        score_thr: float = 0.05,
        nms_thr: float = 0.6,
        topk: int = 300,
    ) -> List[Dict[str, torch.Tensor]]:
        """Decode dense predictions into boxes/scores/labels per image."""
        self.eval()
        outs = self.forward(clip)
        cls_outs: List[torch.Tensor] = outs["cls"]
        reg_outs: List[torch.Tensor] = outs["reg"]
        ctr_outs: List[torch.Tensor] = outs["ctr"]

        B = cls_outs[0].shape[0]
        H_img, W_img = img_hw
        device = clip.device

        results: List[Dict[str, torch.Tensor]] = []
        for b in range(B):
            boxes_all: List[torch.Tensor] = []
            scores_all: List[torch.Tensor] = []
            labels_all: List[torch.Tensor] = []

            for lvl, (cls_map, reg_map, ctr_map) in enumerate(zip(cls_outs, reg_outs, ctr_outs)):
                stride = self.cfg.strides[lvl]
                C = cls_map.shape[1]
                H, W = cls_map.shape[2], cls_map.shape[3]

                cls_logits = cls_map[b].permute(1, 2, 0).reshape(-1, C)  # [HW,C]
                reg = reg_map[b].permute(1, 2, 0).reshape(-1, 4)  # [HW,4]
                ctr_logits = ctr_map[b].permute(1, 2, 0).reshape(-1, 1)  # [HW,1]

                cls_prob = torch.sigmoid(cls_logits)
                ctr_prob = torch.sigmoid(ctr_logits)

                # score = cls * ctr
                scores = cls_prob * ctr_prob

                # filter by threshold
                keep = scores > score_thr
                if not keep.any():
                    continue

                idxs = keep.nonzero(as_tuple=False)
                loc_idx = idxs[:, 0]
                cls_idx = idxs[:, 1]
                score_vals = scores[loc_idx, cls_idx]

                # topk
                if score_vals.numel() > topk:
                    score_vals, order = score_vals.topk(topk)
                    loc_idx = loc_idx[order]
                    cls_idx = cls_idx[order]

                points = _make_grid_xy(H, W, stride, device=device)
                xy = points[loc_idx]  # [N,2]
                ltrb = reg[loc_idx]

                x = xy[:, 0]
                y = xy[:, 1]
                l, t, r, b_ = ltrb.unbind(dim=-1)
                x1 = (x - l).clamp(min=0, max=W_img - 1)
                y1 = (y - t).clamp(min=0, max=H_img - 1)
                x2 = (x + r).clamp(min=0, max=W_img - 1)
                y2 = (y + b_).clamp(min=0, max=H_img - 1)

                boxes = torch.stack([x1, y1, x2, y2], dim=-1)

                boxes_all.append(boxes)
                scores_all.append(score_vals)
                labels_all.append(cls_idx)

            if len(boxes_all) == 0:
                results.append(
                    {
                        "boxes": clip.new_zeros((0, 4)),
                        "scores": clip.new_zeros((0,)),
                        "labels": clip.new_zeros((0,), dtype=torch.long),
                    }
                )
                continue

            boxes = torch.cat(boxes_all, dim=0)
            scores = torch.cat(scores_all, dim=0)
            labels = torch.cat(labels_all, dim=0)

            # class-wise NMS
            keep_all: List[torch.Tensor] = []
            for c in range(self.cfg.num_classes):
                m = labels == c
                if m.sum() == 0:
                    continue
                keep_c = _nms_xyxy(boxes[m], scores[m], iou_thr=nms_thr)
                # map back to original indices
                idx_c = m.nonzero(as_tuple=False).squeeze(1)
                keep_all.append(idx_c[keep_c])

            if len(keep_all) == 0:
                keep = boxes.new_zeros((0,), dtype=torch.long)
            else:
                keep = torch.cat(keep_all)
                keep = keep[scores[keep].argsort(descending=True)]

            results.append({"boxes": boxes[keep], "scores": scores[keep], "labels": labels[keep]})

        return results


# -------------------------
# Loss wrapper (for training)
# -------------------------

class FCOSLoss(nn.Module):
    def __init__(self, cfg: Optional[FCOSConfig] = None, lambda_cls: float = 1.0, lambda_box: float = 1.0, lambda_ctr: float = 1.0):
        super().__init__()
        self.cfg = cfg or FCOSConfig()
        self.lambda_cls = lambda_cls
        self.lambda_box = lambda_box
        self.lambda_ctr = lambda_ctr

    def forward(
        self,
        outputs: Dict[str, List[torch.Tensor]],
        gt_boxes_xyxy: List[torch.Tensor],
        gt_labels: List[torch.Tensor],
        img_hw: Tuple[int, int],
    ) -> Dict[str, torch.Tensor]:
        """Compute FCOS losses for a batch.

        outputs: dict of lists, each list has 3 levels.
        gt_boxes_xyxy: list of length B, each [Gi,4] in pixel coords
        gt_labels: list of length B, each [Gi]
        """
        cls_outs = outputs["cls"]
        reg_outs = outputs["reg"]
        ctr_outs = outputs["ctr"]

        device = cls_outs[0].device
        B = cls_outs[0].shape[0]

        feat_sizes = [(m.shape[2], m.shape[3]) for m in cls_outs]

        cls_targets: List[torch.Tensor] = []
        reg_targets: List[torch.Tensor] = []
        ctr_targets: List[torch.Tensor] = []
        pos_masks: List[torch.Tensor] = []

        for i in range(B):
            t = fcos_assign_single_image(
                gt_boxes_xyxy=gt_boxes_xyxy[i],
                gt_labels=gt_labels[i],
                feat_sizes=feat_sizes,
                cfg=self.cfg,
                device=device,
            )
            # per-level to flatten later
            cls_targets.append(torch.cat(t["cls"], dim=0))
            reg_targets.append(torch.cat(t["reg"], dim=0))
            ctr_targets.append(torch.cat(t["ctr"], dim=0))
            pos_masks.append(torch.cat(t["pos"], dim=0))

        # flatten predictions
        cls_preds = []
        reg_preds = []
        ctr_preds = []
        points_all = []
        for lvl, (cls_map, reg_map, ctr_map) in enumerate(zip(cls_outs, reg_outs, ctr_outs)):
            stride = self.cfg.strides[lvl]
            H, W = cls_map.shape[2], cls_map.shape[3]
            points = _make_grid_xy(H, W, stride, device=device)  # [HW,2]
            points_all.append(points)

            cls_preds.append(cls_map.permute(0, 2, 3, 1).reshape(B, -1, self.cfg.num_classes))
            reg_preds.append(reg_map.permute(0, 2, 3, 1).reshape(B, -1, 4))
            ctr_preds.append(ctr_map.permute(0, 2, 3, 1).reshape(B, -1, 1))

        cls_preds = torch.cat(cls_preds, dim=1)  # [B,P,C]
        reg_preds = torch.cat(reg_preds, dim=1)  # [B,P,4]
        ctr_preds = torch.cat(ctr_preds, dim=1)  # [B,P,1]
        points = torch.cat(points_all, dim=0)  # [P,2]

        cls_t = torch.stack(cls_targets, dim=0)  # [B,P,C]
        reg_t = torch.stack(reg_targets, dim=0)  # [B,P,4]
        ctr_t = torch.stack(ctr_targets, dim=0)  # [B,P,1]
        pos_m = torch.stack(pos_masks, dim=0)  # [B,P]

        # classification focal
        cls_loss = sigmoid_focal_loss(
            logits=cls_preds,
            targets=cls_t,
            alpha=self.cfg.focal_alpha,
            gamma=self.cfg.focal_gamma,
            reduction="sum",
        )

        num_pos = pos_m.sum().clamp(min=1).to(cls_loss.dtype)
        cls_loss = cls_loss / num_pos

        # centerness
        if pos_m.any():
            ctr_loss = F.binary_cross_entropy_with_logits(ctr_preds[pos_m], ctr_t[pos_m], reduction="sum") / num_pos
        else:
            ctr_loss = ctr_preds.sum() * 0.0


        # box loss (GIoU) on positives
        if pos_m.any():
            reg_p = reg_preds[pos_m]  # [Np,4]
            reg_t_p = reg_t[pos_m]  # [Np,4]

            # decode to xyxy
            pts = points[None, :, :].expand(B, -1, 2)[pos_m]  # [Np,2]
            x, y = pts[:, 0], pts[:, 1]

            l, t, r, b_ = reg_p.unbind(dim=-1)
            x1p, y1p, x2p, y2p = x - l, y - t, x + r, y + b_
            pred_boxes = torch.stack([x1p, y1p, x2p, y2p], dim=-1)

            l, t, r, b_ = reg_t_p.unbind(dim=-1)
            x1t, y1t, x2t, y2t = x - l, y - t, x + r, y + b_
            tgt_boxes = torch.stack([x1t, y1t, x2t, y2t], dim=-1)

            giou = generalized_box_iou_xyxy(pred_boxes, tgt_boxes).diag()
            box_loss = (1.0 - giou).sum() / num_pos
        else:
            box_loss = reg_preds.sum() * 0.0

        total = self.lambda_cls * cls_loss + self.lambda_box * box_loss + self.lambda_ctr * ctr_loss

        return {
            "loss": total,
            "loss_cls": cls_loss,
            "loss_box": box_loss,
            "loss_ctr": ctr_loss,
            "num_pos": num_pos.detach(),
        }
