"""
python -m drama_fast.train.vis_predictions_od \
--ckpt /workspace/chz/code/DRAMA-X/fast_system/runs/phase1_fcos_vru_swin_t_384/ckpts/best.pt \
--jsonl ./splits_v3/val.jsonl \
--data_root $DRAMA_DATA_ROOT \
--out_dir ./runs/phase1_fcos_vru_swin_t_384/vis_val \
--num_frames 8 --img_size 384 --stride 2 \
--max_vis 200 \
--score_thr 0.1 --nms_thr 0.50 --topk 200
"""
"""
Visualize FCOS VRU predictions on validation set.

Key improvements vs old version:
- Draw on ORIGINAL keyframe image (via meta['keyframe_path']) when available -> sharp & clear.
- Properly map:
    * GT boxes: normalized xyxy -> original pixels
    * Pred boxes: pixel xyxy on resized img_size (e.g. 384) -> normalized -> original pixels
- Adaptive line width & font size by resolution
- Text with background for readability
- Fallback to resized tensor image if keyframe_path missing

Example:
python -m drama_fast.train.vis_predictions_od \
  --ckpt /path/to/best.pt \
  --jsonl ./splits_v3/val.jsonl \
  --data_root $DRAMA_DATA_ROOT \
  --out_dir ./runs/phase1_fcos_vru_swin_t_384/vis_val \
  --num_frames 8 --img_size 384 --stride 2 \
  --max_vis 200 \
  --score_thr 0.20 --nms_thr 0.50 --topk 200
"""

import argparse
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from drama_fast.dataset_phase1 import DramaFastDataset as DatasetPhase1
from drama_fast.models.model_phase1_fcos import FCOSConfig, Phase1VideoSwinFCOS


# -----------------------------
# Box conversion helpers
# -----------------------------

def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def gt_norm_to_abs_xyxy(box_norm_xyxy: List[float], W: int, H: int) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = box_norm_xyxy
    x1 = clamp01(x1) * W
    y1 = clamp01(y1) * H
    x2 = clamp01(x2) * W
    y2 = clamp01(y2) * H
    # enforce ordering
    x1, x2 = (x1, x2) if x1 <= x2 else (x2, x1)
    y1, y2 = (y1, y2) if y1 <= y2 else (y2, y1)
    return x1, y1, x2, y2


def pred_px_to_abs_xyxy(pred_px_xyxy: List[float], img_size: int, W: int, H: int) -> Tuple[float, float, float, float]:
    # pred is in pixels under resized square canvas: [0, img_size]
    x1, y1, x2, y2 = [float(v) for v in pred_px_xyxy]
    # convert to normalized in [0,1] then to original pixels
    x1 = clamp01(x1 / float(img_size)) * W
    y1 = clamp01(y1 / float(img_size)) * H
    x2 = clamp01(x2 / float(img_size)) * W
    y2 = clamp01(y2 / float(img_size)) * H
    # enforce ordering
    x1, x2 = (x1, x2) if x1 <= x2 else (x2, x1)
    y1, y2 = (y1, y2) if y1 <= y2 else (y2, y1)
    return x1, y1, x2, y2


# -----------------------------
# Drawing helpers
# -----------------------------

def _adaptive_params(W: int, H: int):
    s = min(W, H)
    lw_gt = max(3, int(round(s * 0.004)))   # ~8px at 1920
    lw_pd = max(2, int(round(s * 0.003)))   # ~6px at 1920
    fs = max(16, int(round(s * 0.020)))     # ~38px at 1920
    return lw_gt, lw_pd, fs


def _load_font(font_size: int) -> ImageFont.ImageFont:
    # Try a decent font on Linux; fallback to default.
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for p in candidates:
        try:
            return ImageFont.truetype(p, font_size)
        except Exception:
            pass
    return ImageFont.load_default()


def _text_bbox(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont):
    # pillow>=8 has textbbox
    try:
        l, t, r, b = draw.textbbox((0, 0), text, font=font)
        return l, t, r, b
    except Exception:
        w, h = draw.textsize(text, font=font)
        return 0, 0, w, h


def draw_boxes_pretty(
    img: Image.Image,
    gt_boxes_abs: List[Tuple[float, float, float, float]],
    pred_boxes_abs: List[Tuple[float, float, float, float]],
    pred_scores: List[float],
    title: str = "",
):
    draw = ImageDraw.Draw(img)
    W, H = img.size
    lw_gt, lw_pd, fs = _adaptive_params(W, H)
    font = _load_font(fs)

    # Title bar
    if title:
        l, t, r, b = _text_bbox(draw, title, font=font)
        tw, th = r - l, b - t
        pad = 10
        draw.rectangle([10, 10, 10 + tw + 2 * pad, 10 + th + 2 * pad], fill=(0, 0, 0))
        draw.text((10 + pad, 10 + pad), title, fill=(255, 255, 255), font=font)

    # GT (green)
    for (x1, y1, x2, y2) in gt_boxes_abs:
        draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=lw_gt)

    # Pred (red)
    for (x1, y1, x2, y2), s in zip(pred_boxes_abs, pred_scores):
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=lw_pd)
        txt = f"{s:.2f}"
        l, t, r, b = _text_bbox(draw, txt, font=font)
        tw, th = r - l, b - t
        pad = 6
        # black bg for text
        draw.rectangle([x1, y1, x1 + tw + 2 * pad, y1 + th + 2 * pad], fill=(0, 0, 0))
        draw.text((x1 + pad, y1 + pad), txt, fill=(255, 80, 80), font=font)


def tensor_last_frame_to_img(clip_t: torch.Tensor) -> Image.Image:
    """
    clip_t expected [T,3,H,W] or [T,C,H,W], values roughly in [0,1].
    """
    # last frame [3,H,W] -> [H,W,3]
    last = clip_t[-1]
    if last.ndim != 3:
        raise RuntimeError(f"Unexpected last frame shape: {tuple(last.shape)}")
    if last.shape[0] != 3:
        # try convert if channel last?
        raise RuntimeError(f"Expected CHW with C=3, got {tuple(last.shape)}")

    arr = last.permute(1, 2, 0).cpu().numpy()
    arr = (np.clip(arr, 0, 1) * 255).astype("uint8")
    return Image.fromarray(arr)


def safe_open_image(path: str) -> Optional[Image.Image]:
    try:
        p = Path(path)
        if p.exists():
            return Image.open(str(p)).convert("RGB")
    except Exception:
        pass
    return None


# -----------------------------
# Main
# -----------------------------

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--num_frames", type=int, default=8)
    ap.add_argument("--img_size", type=int, default=384)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--topk_targets", type=int, default=5)

    ap.add_argument("--score_thr", type=float, default=0.1)
    ap.add_argument("--nms_thr", type=float, default=0.50)
    ap.add_argument("--topk", type=int, default=200)

    ap.add_argument("--max_vis", type=int, default=200)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--backbone", type=str, default="swin_t")

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # dataset (meta required!)
    ds = DatasetPhase1(
        jsonl_path=args.jsonl,
        data_root=args.data_root,
        num_frames=args.num_frames,
        img_size=args.img_size,
        stride=args.stride,
        topk_targets=args.topk_targets,
        return_meta=True,
    )

    ckpt = torch.load(args.ckpt, map_location="cpu")
    cfg = FCOSConfig(**ckpt.get("cfg", {})) if isinstance(ckpt, dict) and "cfg" in ckpt else FCOSConfig()

    # prefer backbone from ckpt args
    if isinstance(ckpt, dict) and "args" in ckpt and isinstance(ckpt["args"], dict) and "backbone" in ckpt["args"]:
        args.backbone = ckpt["args"]["backbone"]

    model = Phase1VideoSwinFCOS(cfg=cfg, backbone_name=args.backbone, pretrained=False)
    sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    if any(k.startswith("module.") for k in sd.keys()):
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)
    model.eval()
    model.to(args.device)

    n_total = min(args.max_vis, len(ds))
    print(f"[Vis] dataset size={len(ds)}, will visualize {n_total} samples -> {out_dir}")

    for i in range(n_total):
        sample = ds[i]

        clip = sample["pixel_values"]  # expected [T,3,H,W]
        if clip.ndim != 4:
            raise RuntimeError(f"Unexpected pixel_values shape: {tuple(clip.shape)}")
        # normalize to [T,3,H,W]
        if clip.shape[1] == 3:
            # [T,3,H,W]
            pass
        elif clip.shape[0] == 3:
            # [3,T,H,W] -> [T,3,H,W]
            clip = clip.permute(1, 0, 2, 3).contiguous()
        else:
            raise RuntimeError(f"Unexpected pixel_values shape: {tuple(clip.shape)}")

        # [Fix] 模型需要 [B, C, T, H, W]，而现在的 clip 是 [T, C, H, W]
        # 所以先 permute 回 [C, T, H, W] 再 unsqueeze
        clip_for_model = clip.permute(1, 0, 2, 3).contiguous()
        clip_b = clip_for_model.unsqueeze(0).to(args.device)

        pred0 = model.inference(
            clip_b,
            img_hw=(args.img_size, args.img_size),
            score_thr=args.score_thr,
            nms_thr=args.nms_thr,
            topk=args.topk,
        )[0]

        p_boxes = pred0["boxes"].detach().cpu()
        p_scores = pred0["scores"].detach().cpu()

        meta = sample.get("meta", {}) or {}
        keyframe_path = meta.get("keyframe_path", None)

        # Use original image if possible
        img = None
        if isinstance(keyframe_path, str) and keyframe_path:
            img = safe_open_image(keyframe_path)
        if img is None:
            # fallback to resized tensor image
            img = tensor_last_frame_to_img(clip)

        W, H = img.size

        # GT abs boxes from normalized
        gt_boxes_abs: List[Tuple[float, float, float, float]] = []
        gt_boxes = sample["gt_boxes_topk"]  # [K,4] normalized
        gt_mask = sample["gt_mask_topk"]    # [K]
        for k in range(int(gt_boxes.shape[0])):
            if float(gt_mask[k].item()) <= 0.5:
                continue
            gt_boxes_abs.append(gt_norm_to_abs_xyxy(gt_boxes[k].tolist(), W, H))

        # Pred abs boxes (pred is in 384-pixel coords)
        pred_boxes_abs: List[Tuple[float, float, float, float]] = []
        pred_scores_list: List[float] = []
        for b, s in zip(p_boxes.tolist(), p_scores.tolist()):
            s = float(s)
            if s < float(args.score_thr):
                continue
            pred_boxes_abs.append(pred_px_to_abs_xyxy(b, args.img_size, W, H))
            pred_scores_list.append(s)

        vid = meta.get("video_id", "unknown")
        fid = meta.get("frame_id", "unknown")
        sid = meta.get("sample_id", i)

        title = f"sample={sid}  vid={vid} frame={fid}  GT={len(gt_boxes_abs)}  Pred>thr={len(pred_boxes_abs)}"
        draw_boxes_pretty(img, gt_boxes_abs, pred_boxes_abs, pred_scores_list, title=title)

        # Save high quality
        save_path = out_dir / f"{i:06d}_{vid}_frame_{fid}.jpg"
        img.save(str(save_path), quality=95, subsampling=0)

    print(f"[Vis] Saved visualizations to: {out_dir}")


if __name__ == "__main__":
    main()
