"""
新 baseline（训练用了 letterbox）：建议直接：
python -m drama_fast.train.vis_predictions_phase1_5_od ... --letterbox

老 baseline（没 letterbox）：用：
python -m drama_fast.train.vis_predictions_phase1_5_od ... --no_letterbox

python -m drama_fast.train.vis_predictions_phase1_5_od \
--ckpt /workspace/chz/code/DRAMA-X/fast_system/runs/phase1_5_fcos_vru_swin_t_384/ckpts/best.pt \
--jsonl ./splits_v3/val.jsonl \
--data_root $DRAMA_DATA_ROOT \
--out_dir ./runs/phase1_5_fcos_vru_swin_t_384/vis_val \
--num_frames 8 --img_size 384 --stride 2 \
--max_vis 200 \
--score_thr 0.1 --nms_thr 0.50 --topk 200
--letterbox
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
python -m drama_fast.train.vis_predictions_phase1_5_od \
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
from drama_fast.models.model_phase1_5_fcos import Phase1_5Config, Phase1_5VideoSwinFCOS
import re

def sanitize(s: str, max_len: int = 80) -> str:
    s = str(s)
    s = re.sub(r"[^a-zA-Z0-9._-]+", "_", s)
    return s[:max_len]

def lb_px_to_abs_xyxy(
        px_xyxy: List[float],
        W: int,
        H: int,
        sx: float,
        sy: float,
        pad_x: int,
        pad_y: int
    ) -> Tuple[float,float,float,float]:
        x1, y1, x2, y2 = [float(v) for v in px_xyxy]

        x1 = (x1 - pad_x) / max(sx, 1e-8)
        x2 = (x2 - pad_x) / max(sx, 1e-8)
        y1 = (y1 - pad_y) / max(sy, 1e-8)
        y2 = (y2 - pad_y) / max(sy, 1e-8)

        x1 = max(0.0, min(x1, float(W)))
        x2 = max(0.0, min(x2, float(W)))
        y1 = max(0.0, min(y1, float(H)))
        y2 = max(0.0, min(y2, float(H)))

        if x2 < x1: x1, x2 = x2, x1
        if y2 < y1: y1, y2 = y2, y1
        return x1, y1, x2, y2


def compute_letterbox_params(W: int, H: int, img_size: int) -> Tuple[float, float, int, int]:
    S = int(img_size)
    scale = float(S) / float(max(W, H))  # ✅ 与dataset一致
    new_w = int(round(W * scale))
    new_h = int(round(H * scale))
    new_w = min(new_w, S)
    new_h = min(new_h, S)
    pad_x = (S - new_w) // 2
    pad_y = (S - new_h) // 2
    sx = new_w / float(W) if W > 0 else 1.0
    sy = new_h / float(H) if H > 0 else 1.0
    return sx, sy, pad_x, pad_y



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
    clip_t expected [T,3,H,W] (normalized by ImageNet mean/std).
    """
    last = clip_t[-1]  # [3,H,W]
    if last.ndim != 3 or last.shape[0] != 3:
        raise RuntimeError(f"Unexpected last frame shape: {tuple(last.shape)}")

    mean = torch.tensor([0.485, 0.456, 0.406], dtype=last.dtype, device=last.device).view(3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225], dtype=last.dtype, device=last.device).view(3,1,1)
    last = last * std + mean
    last = last.clamp(0, 1)

    arr = (last.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
    return Image.fromarray(arr)


def safe_open_image(path: str, data_root: Optional[str] = None) -> Optional[Image.Image]:
    try:
        if not path:
            return None
        p = Path(path)
        if p.exists():
            return Image.open(str(p)).convert("RGB")

        # ✅ fallback: relative to data_root
        if data_root is not None:
            p2 = Path(data_root) / path
            if p2.exists():
                return Image.open(str(p2)).convert("RGB")
    except Exception:
        pass
    return None

def parse_orig_size(orig_size, img_size_hint: int = 384):
    """
    Try to interpret orig_size robustly.
    Returns (W, H) or None.
    """
    if not isinstance(orig_size, (list, tuple)) or len(orig_size) != 2:
        return None
    a, b = int(orig_size[0]), int(orig_size[1])
    if a <= 0 or b <= 0:
        return None

    # Heuristic: typical frames W>=H; if a<b and b is very large while a is around 720/1080, could be (H,W).
    # More robust: if one dimension is close to img_size_hint and the other is not, do nothing.
    # We'll just enforce W>=H when it's clearly swapped.
    if a < b:
        # could be (H,W) or (W,H) portrait; most driving datasets are landscape.
        # if b >= 1200 and a <= 1200, treat as (H,W)
        if b >= 1200 and a <= 1200:
            return (b, a)
    return (a, b)


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
    
    # ---- letterbox vis switch ----
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--letterbox", dest="letterbox", action="store_true")
    g.add_argument("--no_letterbox", dest="letterbox", action="store_false")
    ap.set_defaults(letterbox=None)  # None => try read from ckpt, else fallback


    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) 先 load ckpt
    ckpt = torch.load(args.ckpt, map_location="cpu")
    cfg = Phase1_5Config(**ckpt.get("cfg", {})) if isinstance(ckpt, dict) and "cfg" in ckpt else Phase1_5Config()

    # 2) 决定 use_letterbox（CLI > ckpt > default）
    def _ckpt_arg_get(a, k):
        if a is None:
            return None
        if isinstance(a, dict):
            return a.get(k, None)
        # argparse.Namespace or similar
        return getattr(a, k, None)

    ckpt_letterbox = False
    a = ckpt.get("args", None) if isinstance(ckpt, dict) else None
    for k in ["letterbox", "use_letterbox", "do_letterbox"]:
        v = _ckpt_arg_get(a, k)
        if v is not None:
            ckpt_letterbox = bool(v)
            break


    if args.letterbox is None:
        args.letterbox = ckpt_letterbox
    use_letterbox = bool(args.letterbox)

    # 3) 再创建 dataset，并把 letterbox 开关传进去（⚠️参数名按你 dataset 实现为准）
    ds = DatasetPhase1(
        jsonl_path=args.jsonl,
        data_root=args.data_root,
        num_frames=args.num_frames,
        img_size=args.img_size,
        stride=args.stride,
        topk_targets=args.topk_targets,
        return_meta=True,
        # ✅ 对齐训练/ckpt的 letterbox 开关
        use_letterbox=use_letterbox,
        # ✅ val可视化务必关闭增强
        flip_prob=0.0,
    )


    # prefer backbone from ckpt args
    a = ckpt.get("args", None) if isinstance(ckpt, dict) else None
    v = _ckpt_arg_get(a, "backbone")
    if v is not None:
        args.backbone = str(v)

    model = Phase1_5VideoSwinFCOS(cfg)
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
        
        # ✅ 兼容：如果模型输出是归一化(0~1)，转成像素(0~S)
        if p_boxes.numel() > 0:
            mx = float(p_boxes.max().item())
            mn = float(p_boxes.min().item())
            # 更稳：如果全部落在 [-0.1, 1.1] 视为归一化
            if mx <= 1.1 and mn >= -0.1:
                p_boxes = p_boxes * float(args.img_size)


        meta = sample.get("meta", {}) or {}
        # ---- sanity: img_size must match training/dataset ----
        lb_meta = (meta.get("letterbox", {}) or {})
        if lb_meta.get("img_size", None) is not None:
            S_meta = int(lb_meta["img_size"])
            if S_meta != int(args.img_size):
                raise RuntimeError(
                    f"[Vis] img_size mismatch: meta img_size={S_meta} but args.img_size={args.img_size}. "
                    "This will break GT/Pred mapping under letterbox."
                )
        keyframe_path = meta.get("keyframe_path", None)

        # Use original image if possible
        img = None
        using_orig = False
        if isinstance(keyframe_path, str) and keyframe_path:
            img = safe_open_image(keyframe_path, data_root=args.data_root)
            using_orig = (img is not None)

        if img is None:
            img = tensor_last_frame_to_img(clip)
            using_orig = False
            
        # ---- optional: when keyframe missing but meta has orig_size, reconstruct an "orig-like" canvas for letterbox ----
        if (not using_orig) and use_letterbox:
            if isinstance(meta.get("orig_size", None), (list, tuple)) and len(meta["orig_size"]) == 2:
                wh = parse_orig_size(meta.get("orig_size", None), img_size_hint=args.img_size)
                if wh is not None:
                    W0, H0 = wh
                lbm = (meta.get("letterbox", {}) or {})
                if all(k in lbm for k in ["pad_x", "pad_y", "new_w", "new_h", "img_size"]):
                    pad_x0 = int(lbm["pad_x"]); pad_y0 = int(lbm["pad_y"])
                    new_w0 = int(lbm["new_w"]); new_h0 = int(lbm["new_h"])
                    # img currently is SxS (letterboxed). crop content area and resize back to W0xH0
                    content = img.crop((pad_x0, pad_y0, pad_x0 + new_w0, pad_y0 + new_h0))
                    img = content.resize((W0, H0), resample=Image.BILINEAR)
                    using_orig = True  # now we have an orig-sized canvas for drawing/inverse

        W, H = img.size
        
        # ✅ 健壮性：如果没拿到原图但 meta 里有 orig_size，且当前是 no_letterbox，
        # 就把画布 resize 到原图尺寸，保证 GT norm -> abs 的映射不崩
        if (not using_orig) and (not use_letterbox):
            if isinstance(meta.get("orig_size", None), (list, tuple)) and len(meta["orig_size"]) == 2:
                wh = parse_orig_size(meta.get("orig_size", None), img_size_hint=args.img_size)
                if wh is not None:
                    W0, H0 = wh
                if W0 > 0 and H0 > 0 and (W0, H0) != (W, H):
                    img = img.resize((W0, H0), resample=Image.BILINEAR)
                    W, H = img.size
        
        # ---- compute letterbox params (prefer meta if exists) ----
        sx, sy, pad_x, pad_y = 1.0, 1.0, 0, 0
        if use_letterbox and using_orig:
            lb = (meta.get("letterbox", {}) or {})

            # ✅ 1) 最推荐：用 new_w/new_h + pad 反推出 sx/sy（最贴近真实像素链）
            if all(k in lb for k in ["pad_x", "pad_y", "new_w", "new_h"]):
                pad_x = int(lb["pad_x"]); pad_y = int(lb["pad_y"])
                new_w = int(lb["new_w"]); new_h = int(lb["new_h"])
                sx = new_w / float(W) if W > 0 else 1.0
                sy = new_h / float(H) if H > 0 else 1.0

            # ✅ 2) 其次：如果真的存了 sx/sy，就用 sx/sy
            elif all(k in lb for k in ["sx", "sy", "pad_x", "pad_y"]):
                sx = float(lb["sx"]); sy = float(lb["sy"])
                pad_x = int(lb["pad_x"]); pad_y = int(lb["pad_y"])

            # ✅ 3) 最后：再自己算（可能与dataset存在 1px rounding 差）
            else:
                sx, sy, pad_x, pad_y = compute_letterbox_params(W, H, args.img_size)



        # GT abs boxes from normalized
        gt_boxes_abs: List[Tuple[float, float, float, float]] = []
        gt_boxes = sample["gt_boxes_topk"]  # [K,4] normalized
        gt_mask = sample["gt_mask_topk"]    # [K]
        for k in range(int(gt_boxes.shape[0])):
            if float(gt_mask[k].item()) <= 0.5:
                continue
            if use_letterbox and using_orig:
                px = (gt_boxes[k] * float(args.img_size)).tolist()
                gt_boxes_abs.append(lb_px_to_abs_xyxy(px, W, H, sx, sy, pad_x, pad_y))
            elif use_letterbox and (not using_orig):
                # 直接在 384 上画
                x1,y1,x2,y2 = (gt_boxes[k] * float(args.img_size)).tolist()
                gt_boxes_abs.append((x1,y1,x2,y2))
            else:
                gt_boxes_abs.append(gt_norm_to_abs_xyxy(gt_boxes[k].tolist(), W, H))

        # Pred abs boxes (pred is in 384-pixel coords)
        pred_boxes_abs: List[Tuple[float, float, float, float]] = []
        pred_scores_list: List[float] = []

        S = float(args.img_size)

        for b, s in zip(p_boxes.tolist(), p_scores.tolist()):
            s = float(s)
            if s < float(args.score_thr):
                continue

            if use_letterbox:
                if using_orig:
                    # ✅ 画在原图：需要 inverse letterbox
                    pred_boxes_abs.append(lb_px_to_abs_xyxy(b, W, H, sx, sy, pad_x, pad_y))
                else:
                    # ✅ 画在 S×S tensor 图：不要 inverse，直接 clamp + enforce order
                    x1, y1, x2, y2 = [float(v) for v in b]
                    x1 = max(0.0, min(x1, S))
                    x2 = max(0.0, min(x2, S))
                    y1 = max(0.0, min(y1, S))
                    y2 = max(0.0, min(y2, S))
                    if x2 < x1: x1, x2 = x2, x1
                    if y2 < y1: y1, y2 = y2, y1
                    pred_boxes_abs.append((x1, y1, x2, y2))
            else:
                # no-letterbox：pred 在 S×S 像素系，映射回原图像素
                pred_boxes_abs.append(pred_px_to_abs_xyxy(b, args.img_size, W, H))

            pred_scores_list.append(s)


        sid = meta.get("sample_id", i)
        kf = Path(meta.get("keyframe_path","")).name if meta.get("keyframe_path") else "no_kf"
        title = f"sample={sid}  key={kf}  GT={len(gt_boxes_abs)}  Pred>thr={len(pred_boxes_abs)}"

        # ✅ 画框（建议用 copy，避免原图对象被反复污染）
        vis_img = img.copy()
        draw_boxes_pretty(
            vis_img,
            gt_boxes_abs=gt_boxes_abs,
            pred_boxes_abs=pred_boxes_abs,
            pred_scores=pred_scores_list,
            title=title,
        )

        sid_s = sanitize(sid)
        kf_s  = sanitize(kf)
        save_path = out_dir / f"{i:06d}_{sid_s}_{kf_s}.jpg"
        vis_img.save(str(save_path), quality=95, subsampling=0)

    print(f"[Vis] Saved visualizations to: {out_dir}")


if __name__ == "__main__":
    main()
