"""Stage-2 evaluation utilities.

We evaluate a **two-stage** pipeline:

Stage-1: detector/tracker produces proposal boxes (your "框准")
Stage-2: model scores proposals and we check whether the GT VRU box is retrieved in Top-K.

Key requirements for reliability
--------------------------------
1) Metrics must be **monotonic**: recall@10 >= recall@5 >= recall@1.
2) Evaluation must be **mask-aware** (ignore padded proposals).
3) Proposals may contain extra dims (e.g., conf); IoU must use xyxy only.
4) When something looks inconsistent, print the **sample_id** so you can debug
   without "blind running".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from drama_fast.utils.boxes import box_iou_xyxy


@dataclass
class EvalDebug:
    """Optional debug payload returned by evaluate_stage2."""
    bad_sample_ids: List[str]
    bad_reasons: List[str]


def _to_xyxy(proposals: torch.Tensor) -> torch.Tensor:
    """Accept proposals as [...,4] or [...,>=4]; always return [...,4]."""
    if proposals.size(-1) < 4:
        raise ValueError(f"proposals last dim must be >=4, got {proposals.shape}")
    return proposals[..., :4]


def _safe_percentile(x: np.ndarray, q: float) -> float:
    if x.size == 0:
        return 0.0
    return float(np.percentile(x, q))


@torch.no_grad()
def evaluate_stage2(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str,
    iou_thr: float = 0.5,
    topk: Tuple[int, ...] = (1, 5, 10),
    return_debug: bool = False,
    max_debug: int = 50,
) -> Dict[str, float] | Tuple[Dict[str, float], EvalDebug]:
    """Evaluate Stage-2 ranking.

    The loader must yield:
      - pixel_values: [B,3,T,S,S]
      - proposals:    [B,N,4] normalized xyxy in [0,1] (padding allowed)
      - proposal_mask:[B,N] bool (True=valid)
      - gt_box:       [B,4] normalized xyxy
      - (optional) sample_id: list[str] length B
    """
    model.eval()

    ks = tuple(sorted(set(int(k) for k in topk)))
    max_k = max(ks)

    total = 0
    hit_at = {k: 0 for k in ks}
    # "ceiling": does Stage-1 proposals contain any box reaching iou_thr?
    ceiling_hit = 0

    top1_ious: List[float] = []
    max_ious: List[float] = []

    bad_ids: List[str] = []
    bad_reasons: List[str] = []

    for batch in loader:
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        proposals = batch["proposals"].to(device, non_blocking=True)
        proposal_mask = batch["proposal_mask"].to(device, non_blocking=True)
        gt_box = batch["gt_box"].to(device, non_blocking=True)

        sample_ids: Optional[List[str]] = None
        if "sample_id" in batch:
            sample_ids = list(batch["sample_id"])
        elif "meta" in batch and batch["meta"] is not None:
            try:
                sample_ids = [m.get("sample_id", None) for m in batch["meta"]]
                if any(s is None for s in sample_ids):
                    sample_ids = None
            except Exception:
                sample_ids = None

        # forward (model wrapper may return dict or tensor)
        out = model(pixel_values=pixel_values, proposals=proposals, proposal_mask=proposal_mask)
        logits = out["logits"] if isinstance(out, dict) else out  # [B,N]

        B, N = logits.shape
        props_xyxy = _to_xyxy(proposals)  # [B,N,4]
        gt_xyxy = gt_box  # [B,4]

        # IoU between each proposal and its *sample* GT: [B,N]
        # IMPORTANT: torchvision-style box_iou returns a pairwise matrix [N,M].
        # If we pass (B*N) proposals and (B*N) GT boxes, we'd get a gigantic
        # (B*N)x(B*N) matrix, which is NOT what we want and will break reshape.
        #
        # We instead compute IoU per sample (M=1), then scatter back.
        ious = props_xyxy.new_zeros((B, N))
        for b in range(B):
            vb = proposal_mask[b]
            if vb.any():
                ious_b = box_iou_xyxy(props_xyxy[b, vb], gt_xyxy[b].unsqueeze(0)).squeeze(1)  # [Nb]
                ious[b, vb] = ious_b

        # mask invalid
        mask = proposal_mask.bool()
        logits = logits.masked_fill(~mask, -1e9)
        ious = ious.masked_fill(~mask, 0.0)

        # per-sample evaluation
        for b in range(B):
            total += 1
            sid = sample_ids[b] if sample_ids is not None else f"idx={total-1}"

            valid_n = int(mask[b].sum().item())
            if valid_n == 0:
                # No proposals at all => always miss, ceiling miss.
                if return_debug and len(bad_ids) < max_debug:
                    bad_ids.append(sid)
                    bad_reasons.append("no_valid_proposals")
                continue

            # ceiling
            max_iou = float(ious[b].max().item())
            max_ious.append(max_iou)
            if max_iou >= iou_thr:
                ceiling_hit += 1

            # rank proposals by logits among valid ones
            order = torch.argsort(logits[b], descending=True)
            # clamp K to number of valid proposals (important for small N)
            for k in ks:
                kk = min(k, valid_n)
                topk_idx = order[:kk]
                hit = bool((ious[b, topk_idx] >= iou_thr).any().item())
                if hit:
                    hit_at[k] += 1

            # top1 IoU (best-scored)
            top1_idx = int(order[0].item())
            top1_iou = float(ious[b, top1_idx].item())
            top1_ious.append(top1_iou)

            # sanity: monotonic per-sample
            # (aggregate should be monotonic too, but we check here for debug)
            # Here it can only break if NaNs appear.
            if torch.isnan(logits[b]).any() or torch.isnan(ious[b]).any():
                if return_debug and len(bad_ids) < max_debug:
                    bad_ids.append(sid)
                    bad_reasons.append("nan_in_logits_or_ious")

    # aggregate
    metrics: Dict[str, float] = {
        "num_samples": float(total),
        "top1_iou": float(np.mean(top1_ious)) if top1_ious else 0.0,
        "max_iou_mean": float(np.mean(max_ious)) if max_ious else 0.0,
        "max_iou_median": float(np.median(max_ious)) if max_ious else 0.0,
        "max_iou_p90": _safe_percentile(np.asarray(max_ious, dtype=np.float32), 90.0),
        "stage1_ceiling_recall": float(ceiling_hit / max(total, 1)),
    }
    for k in ks:
        metrics[f"recall@{k}"] = float(hit_at[k] / max(total, 1))

    # enforce monotonic at aggregate level; if violated, return debug ids
    # (should not happen with the correct computation above)
    for i in range(1, len(ks)):
        if metrics[f"recall@{ks[i]}"] + 1e-9 < metrics[f"recall@{ks[i-1]}"]:
            if return_debug:
                bad_ids.append("__AGGREGATE__")
                bad_reasons.append(f"non_monotonic_recall: {ks[i]}<{ks[i-1]}")
            break

    if return_debug:
        return metrics, EvalDebug(bad_sample_ids=bad_ids, bad_reasons=bad_reasons)
    return metrics
