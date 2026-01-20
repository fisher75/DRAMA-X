"""Phase-1 evaluation for the fast system.

Reuses:
- drama_fast.dataset_phase1.DramaFastDataset
- drama_fast.models.model_phase1.FastSystemPhase1

Example:
  python -m drama_fast.train.eval_phase1 \
    --jsonl ./splits_v1/val.jsonl \
    --data_root $DRAMA_DATA_ROOT \
    --ckpt ./runs/phase1_swin_t/best.pt \
    --out_json ./runs/phase1_swin_t/eval_val.json \
    --num_frames 8 --img_size 224 --batch_size 8
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from drama_fast.dataset_phase1 import DramaFastDataset
from drama_fast.models.model_phase1 import FastSystemPhase1


def box_iou_xyxy(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """IoU for normalized xyxy boxes.

    a, b: [B,4] in [0,1]
    """
    a = a.clamp(0, 1)
    b = b.clamp(0, 1)

    ax1, ay1, ax2, ay2 = a.unbind(-1)
    bx1, by1, bx2, by2 = b.unbind(-1)

    inter_x1 = torch.maximum(ax1, bx1)
    inter_y1 = torch.maximum(ay1, by1)
    inter_x2 = torch.minimum(ax2, bx2)
    inter_y2 = torch.minimum(ay2, by2)

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter = inter_w * inter_h

    area_a = (ax2 - ax1).clamp(min=0) * (ay2 - ay1).clamp(min=0)
    area_b = (bx2 - bx1).clamp(min=0) * (by2 - by1).clamp(min=0)

    union = area_a + area_b - inter
    return inter / (union + eps)


def load_ckpt(model: torch.nn.Module, ckpt_path: str) -> None:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("model", ckpt)
    # strip DDP prefix
    if any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device, lambda_box: float, lambda_risk: float) -> Dict[str, float]:
    model.eval()

    tot_loss = 0.0
    tot_box = 0.0
    tot_risk = 0.0

    iou_sum = 0.0
    risk_mae_sum = 0.0
    risk_rmse_sum = 0.0

    n = 0

    for batch in tqdm(loader, desc="eval"):
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        gt_box = batch["gt_box"].to(device, non_blocking=True)
        gt_risk = batch["gt_risk"].to(device, non_blocking=True)

        out = model(pixel_values)
        if isinstance(out, tuple) and len(out) == 2 and out[0].dim() == 3:
            pred_boxes, pred_risks = out
            q = pred_risks.argmax(dim=1)
            bidx = torch.arange(pred_boxes.shape[0], device=pred_boxes.device)
            pred_box = pred_boxes[bidx, q]
            pred_risk = pred_risks[bidx, q]
        else:
            pred_box, pred_risk = out
        loss_box = F.l1_loss(pred_box, gt_box)
        loss_risk = F.mse_loss(pred_risk, gt_risk)
        loss = lambda_box * loss_box + lambda_risk * loss_risk

        bs = pixel_values.shape[0]
        tot_loss += float(loss.item()) * bs
        tot_box += float(loss_box.item()) * bs
        tot_risk += float(loss_risk.item()) * bs

        iou = box_iou_xyxy(pred_box, gt_box)
        iou_sum += float(iou.sum().item())

        diff = (pred_risk - gt_risk)
        risk_mae_sum += float(diff.abs().sum().item())
        risk_rmse_sum += float((diff * diff).sum().item())

        n += bs

    n = max(1, n)
    out = {
        "loss": tot_loss / n,
        "loss_box": tot_box / n,
        "loss_risk": tot_risk / n,
        "iou": iou_sum / n,
        "risk_mae": risk_mae_sum / n,
        "risk_rmse": (risk_rmse_sum / n) ** 0.5,
        "num_samples": n,
    }
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument("--jsonl", required=True)
    p.add_argument("--data_root", required=True)
    p.add_argument("--ckpt", required=True)

    p.add_argument("--out_json", default="")

    p.add_argument("--num_frames", type=int, default=8)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--topk_targets", type=int, default=1)

    p.add_argument("--num_queries", type=int, default=1)

    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=8)

    p.add_argument("--lambda_box", type=float, default=1.0)
    p.add_argument("--lambda_risk", type=float, default=2.0)

    p.add_argument("--pretrained", action="store_true")
    p.add_argument("--freeze_backbone", action="store_true")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = DramaFastDataset(
        jsonl_path=args.jsonl,
        data_root=args.data_root,
        num_frames=args.num_frames,
        stride=args.stride,
        img_size=args.img_size,
        topk_targets=args.topk_targets,
    )
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    model = FastSystemPhase1(img_size=args.img_size, pretrained=args.pretrained, freeze_backbone=args.freeze_backbone, num_queries=args.num_queries)
    load_ckpt(model, args.ckpt)
    model.to(device)

    metrics = evaluate(model, dl, device, lambda_box=args.lambda_box, lambda_risk=args.lambda_risk)

    print(json.dumps(metrics, indent=2, ensure_ascii=False))

    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
