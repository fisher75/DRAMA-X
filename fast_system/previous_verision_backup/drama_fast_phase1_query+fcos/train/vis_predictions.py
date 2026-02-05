"""Visualize Phase-1 predictions on keyframes.

Creates images with GT/pred boxes and risk scores.

Reuses:
- drama_fast.dataset_phase1.DramaFastDataset
- drama_fast.models.model_phase1.FastSystemPhase1

Example:
  python -m drama_fast.train.vis_predictions \
    --jsonl ./splits_v1/val.jsonl \
    --data_root $DRAMA_DATA_ROOT \
    --ckpt ./runs/phase1_swin_t/best.pt \
    --out_dir ./runs/phase1_swin_t/vis_val \
    --num_frames 8 --img_size 224 --max_vis 100
    
训练之后：
  python -m drama_fast.train.vis_predictions \
    --jsonl ./splits_v1/val.jsonl \
    --data_root $DRAMA_DATA_ROOT \
    --ckpt /workspace/chz/code/DRAMA-X/fast_system/runs/phase1_swin_t_ddp_full/best.pt \
    --out_dir ./runs/phase1_swin_t_ddp_full/vis_val \
    --num_frames 8 \
    --img_size 224 \
    --max_vis 100

20260120
  python -m drama_fast.train.vis_predictions \
    --jsonl ./splits_v3/val.jsonl \
    --data_root $DRAMA_DATA_ROOT \
    --ckpt /workspace/chz/code/DRAMA-X/fast_system/runs/phase1_swin_t_v3_single_giou_384/best.pt \
    --out_dir ./runs/phase1_swin_t_v3_single_giou_384/vis_val \
    --num_frames 8 \
    --img_size 384 \
    --max_vis 100
"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import torch
from torch.utils.data import DataLoader

from drama_fast.dataset_phase1 import DramaFastDataset
from drama_fast.models.model_phase1 import FastSystemPhase1


# ------------------------------
# helpers
# ------------------------------

def _strip_module_prefix(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if any(k.startswith("module.") for k in state.keys()):
        return {k[len("module."):]: v for k, v in state.items()}
    return state


def load_checkpoint(model: torch.nn.Module, ckpt_path: str, device: torch.device) -> None:
    print(f"Loading checkpoint from {ckpt_path} ...")
    obj = torch.load(ckpt_path, map_location=device)
    # 兼容直接保存 model.state_dict() 或保存了完整 dict 的情况
    if isinstance(obj, dict) and "model" in obj:
        state = obj["model"]
    else:
        state = obj
    
    model.load_state_dict(_strip_module_prefix(state), strict=True)
    print("Checkpoint loaded.")


def box_iou_xyxy(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """IoU for normalized xyxy boxes. a,b: [...,4]"""
    ax1, ay1, ax2, ay2 = a.unbind(-1)
    bx1, by1, bx2, by2 = b.unbind(-1)

    inter_x1 = torch.max(ax1, bx1)
    inter_y1 = torch.max(ay1, by1)
    inter_x2 = torch.min(ax2, bx2)
    inter_y2 = torch.min(ay2, by2)
    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter = inter_w * inter_h

    area_a = (ax2 - ax1).clamp(min=0) * (ay2 - ay1).clamp(min=0)
    area_b = (bx2 - bx1).clamp(min=0) * (by2 - by1).clamp(min=0)
    union = (area_a + area_b - inter).clamp(min=1e-9)
    return inter / union


def denorm_box_to_pixels(box_xyxy: List[float], W: int, H: int) -> Tuple[int, int, int, int]:
    x1 = int(round(max(0.0, min(1.0, float(box_xyxy[0]))) * W))
    y1 = int(round(max(0.0, min(1.0, float(box_xyxy[1]))) * H))
    x2 = int(round(max(0.0, min(1.0, float(box_xyxy[2]))) * W))
    y2 = int(round(max(0.0, min(1.0, float(box_xyxy[3]))) * H))
    # enforce x1<=x2, y1<=y2
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    return x1, y1, x2, y2


def draw_one(image_path: str, out_path: str, gt_box: List[float], pred_box: List[float],
             gt_risk: float, pred_risk: float, extra_text: str = "") -> None:
    from PIL import Image, ImageDraw, ImageFont

    try:
        im = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error opening image {image_path}: {e}")
        return

    W, H = im.size
    draw = ImageDraw.Draw(im)

    # boxes
    gx1, gy1, gx2, gy2 = denorm_box_to_pixels(gt_box, W, H)
    px1, py1, px2, py2 = denorm_box_to_pixels(pred_box, W, H)

    # GT: green, Pred: red
    draw.rectangle([gx1, gy1, gx2, gy2], outline=(0, 255, 0), width=4)
    draw.rectangle([px1, py1, px2, py2], outline=(255, 0, 0), width=4)

    # text
    text = f"GT risk={gt_risk:.3f} | Pred risk={pred_risk:.3f}"
    if extra_text:
        text += f" | {extra_text}"

    try:
        # 尝试加载一个更好看的字体，或者直接用默认
        # Linux常见路径，或者改为你有的 ttf
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    except Exception:
        font = ImageFont.load_default()

    # text background
    try:
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        tw, th = right - left, bottom - top
    except AttributeError:
        # 老版本 Pillow
        tw, th = draw.textsize(text, font=font)

    # 画文字背景框，稍微留点边距
    draw.rectangle([10, 10, 10 + tw + 10, 10 + th + 10], fill=(0, 0, 0))
    draw.text((15, 15), text, fill=(255, 255, 255), font=font)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    im.save(out_path)


# ------------------------------
# main
# ------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True, help="Split jsonl to visualize")
    ap.add_argument("--data_root", required=True, help="DRAMA data root (same as training)")
    ap.add_argument("--ckpt", required=True, help="Checkpoint path (.pt)")
    ap.add_argument("--out_dir", required=True, help="Output directory for images + predictions.jsonl")

    ap.add_argument("--num_frames", type=int, default=8)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--num_queries", type=int, default=1, help="number of learnable queries")
    ap.add_argument("--topk_targets", type=int, default=1)

    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=4)

    ap.add_argument("--max_vis", type=int, default=100)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--pretrained", action="store_true", default=False)
    ap.add_argument("--freeze_backbone", action="store_true", default=False)
    ap.add_argument("--device", default="cuda")

    return ap.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(f"Initializing Dataset from {args.jsonl} ...")
    ds = DramaFastDataset(
        jsonl_path=args.jsonl,
        data_root=args.data_root,
        num_frames=args.num_frames,
        stride=args.stride,
        img_size=args.img_size,
        topk_targets=args.topk_targets,
        # strict=True, # 如果你的 Dataset __init__ 里没这个参数就去掉
    )

    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    print("Initializing Model...")
    model = FastSystemPhase1(img_size=args.img_size, pretrained=args.pretrained, freeze_backbone=args.freeze_backbone, num_queries=args.num_queries).to(device)
    load_checkpoint(model, args.ckpt, device)
    model.eval()

    os.makedirs(args.out_dir, exist_ok=True)
    pred_jsonl_path = os.path.join(args.out_dir, "predictions.jsonl")

    print(f"Starting visualization -> {args.out_dir}")
    n_done = 0
    
    with open(pred_jsonl_path, "w", encoding="utf-8") as f_out:
        with torch.no_grad():
            for batch in dl:
                if n_done >= args.max_vis:
                    break

                pixel_values = batch["pixel_values"].to(device, non_blocking=True)
                gt_box = batch["gt_box"].to(device, non_blocking=True)
                gt_risk = batch["gt_risk"].to(device, non_blocking=True)
                
                # --- 修复部分 Start ---
                # DataLoader 默认把 list[dict] collate 成了 dict[str, list/tuple]
                # 所以 meta 是一个字典，而不是列表
                meta_batch = batch["meta"] 
                # --- 修复部分 End ---

                out = model(pixel_values)
                if isinstance(out, tuple) and len(out) == 2 and isinstance(out[0], torch.Tensor) and out[0].dim() == 3:
                    pred_boxes, pred_risks = out  # [B,Q,4], [B,Q]
                    q = pred_risks.argmax(dim=1)
                    ar = torch.arange(pred_boxes.shape[0], device=pred_boxes.device)
                    pred_box = pred_boxes[ar, q]
                    pred_risk = pred_risks[ar, q]
                else:
                    pred_box, pred_risk = out

                # per-sample loop
                batch_size_curr = pred_box.shape[0]
                for i in range(batch_size_curr):
                    if n_done >= args.max_vis:
                        break

                    # --- 修复部分 Start ---
                    # 从 meta_batch 字典中，通过 key 取出第 i 个元素
                    # 假设 meta_batch = {'sample_id': ('id1', 'id2'), 'keyframe_path': ('p1', 'p2'), ...}
                    sample_id = meta_batch["sample_id"][i]
                    keyframe_path = meta_batch["keyframe_path"][i]
                    # --- 修复部分 End ---

                    if not keyframe_path or not os.path.exists(keyframe_path):
                        # if missing, skip vis but still might want to log? Let's skip vis.
                        pass
                    
                    pb = pred_box[i].detach().cpu().tolist()
                    gb = gt_box[i].detach().cpu().tolist()
                    pr = float(pred_risk[i].detach().cpu())
                    gr = float(gt_risk[i].detach().cpu())

                    iou = float(box_iou_xyxy(torch.tensor(pb), torch.tensor(gb)))

                    rec = {
                        "sample_id": sample_id,
                        "keyframe_path": keyframe_path,
                        "gt_box": gb,
                        "pred_box": pb,
                        "gt_risk": gr,
                        "pred_risk": pr,
                        "iou": iou,
                    }
                    f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")

                    if keyframe_path and os.path.exists(keyframe_path):
                        out_path = os.path.join(args.out_dir, f"{sample_id}.jpg")
                        draw_one(
                            image_path=keyframe_path,
                            out_path=out_path,
                            gt_box=gb,
                            pred_box=pb,
                            gt_risk=gr,
                            pred_risk=pr,
                            extra_text=f"IoU={iou:.3f}",
                        )

                    n_done += 1

    print(f"Wrote: {pred_jsonl_path}")
    print(f"Saved {n_done} visualizations to: {args.out_dir}")


if __name__ == "__main__":
    main()