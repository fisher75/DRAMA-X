"""Box utilities (xyxy) for DRAMA-X.

All functions assume normalized boxes in [0,1] unless noted otherwise.
"""

from __future__ import annotations

import torch


def box_area_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """Compute area for boxes in xyxy format.

    Args:
        boxes: [...,4]
    Returns:
        area: [...]
    """
    w = (boxes[..., 2] - boxes[..., 0]).clamp(min=0)
    h = (boxes[..., 3] - boxes[..., 1]).clamp(min=0)
    return w * h


def box_iou_xyxy(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Pairwise IoU between two sets of boxes.

    Args:
        boxes1: [N,4]
        boxes2: [M,4]
    Returns:
        iou: [N,M]
    """
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros((boxes1.shape[0], boxes2.shape[0]), dtype=torch.float32, device=boxes1.device)

    # intersection
    lt = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]

    area1 = box_area_xyxy(boxes1)[:, None]
    area2 = box_area_xyxy(boxes2)[None, :]
    union = area1 + area2 - inter
    return inter / (union + 1e-6)
