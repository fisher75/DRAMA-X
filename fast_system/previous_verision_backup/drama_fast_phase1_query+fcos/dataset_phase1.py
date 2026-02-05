"""Phase-1 Dataset for DRAMA-X fast system.

Reads `annotation_coc/drama_x_fast_sup_v2_rule.jsonl` whose schema is:
{
  "sample_id": str,
  "image": "https://.../data/drama/combined/.../clip_xxx/frame_xxxxx.png",
  "risk_label": int,
  "targets": [
      {
        "frame_idx": int,
        "bbox_xyxy": [x1,y1,x2,y2],
        "risk_score": float,
        ...
      },
      ...
  ]
}

Key design points (why they matter):
- ✅ real T frames, not single-frame copy: otherwise backbone never learns motion/temporal cues.
- ✅ dynamic box normalization: use true image size from the keyframe, never hardcode 1920x1080.
- ✅ optional top-k targets: gives a path to multi-object supervision later.
"""

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

from .utils.path_resolver import (
    resolve_clip_dir,
    resolve_local_image_path,
    extract_frame_index,
    find_keyframe_file,
)


@dataclass
class FrameSampleSpec:
    num_frames: int = 8
    stride: int = 1
    mode: str = "tail"  # 'tail' (end at keyframe) or 'center'


class DramaFastDataset(Dataset):
    def __init__(
        self,
        jsonl_path: str,
        data_root: str = "/data2/automan/data/drama_data",
        img_size: int = 224,
        # === 修改开始：新增这两个参数以兼容 train 脚本 ===
        num_frames: int = 8,   
        stride: int = 1,
        # ===========================================
        frame_spec: Optional[FrameSampleSpec] = None, # 默认改为 None，方便内部判断
        topk: Optional[int] = None,
        topk_targets: int = 1,
        sort_targets_by: str = "risk_score",  # or 'none'
        return_meta: bool = True,
        use_letterbox: bool = True,
        flip_prob: float = 0.5,
        letterbox_fill: int = 114,
    ):
        self.jsonl_path = jsonl_path
        self.data_root = data_root
        self.img_size = img_size
        self.use_letterbox = use_letterbox
        self.flip_prob = flip_prob
        self.letterbox_fill = int(letterbox_fill)

        
        if topk is not None:
            topk_targets = topk
        self.topk_targets = topk_targets
        # === 核心逻辑修改：如果没传 frame_spec，就用 num_frames 造一个 ===
        if frame_spec is None:
            # 这里解决了报错：脚本传进来的 num_frames 被用到了
            self.frame_spec = FrameSampleSpec(num_frames=num_frames, stride=stride)
        else:
            self.frame_spec = frame_spec
        # ==========================================================

        self.topk_targets = topk_targets
        self.sort_targets_by = sort_targets_by
        self.return_meta = return_meta

        # 保留你原本的文件读取逻辑
        self.samples: List[Dict[str, Any]] = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.samples.append(json.loads(line))

        self.mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
        self.std  = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)


    def __len__(self):
        return len(self.samples)

    def _load_image_rgb(self, path: str) -> Image.Image:
        return Image.open(path).convert("RGB")

    def _resize_to_tensor(self, img: Image.Image) -> torch.Tensor:
        # PIL -> float tensor in [0,1], shape [3,H,W]
        img = img.resize((self.img_size, self.img_size))
        arr = np.array(img)
        t = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
        return t

    def _sample_frame_paths(
        self,
        files_sorted: List[Tuple[int, str]],
        keyframe_path: str,
    ) -> List[str]:
        """Return a list of frame paths length == T."""
        T = self.frame_spec.num_frames
        stride = max(1, int(self.frame_spec.stride))
        mode = self.frame_spec.mode

        # locate keyframe position in sorted list
        key_name = os.path.basename(keyframe_path)
        pos = None
        for i, (_, p) in enumerate(files_sorted):
            if os.path.basename(p) == key_name:
                pos = i
                break
        if pos is None:
            # fallback: use last frame
            pos = len(files_sorted) - 1

        if mode == "center":
            half = (T // 2) * stride
            start = max(0, pos - half)
            end = min(len(files_sorted) - 1, pos + half)
            # take centered window, then subsample
            idxs = list(range(start, end + 1, stride))
            # ensure keyframe included by moving window if needed
            if pos not in idxs:
                idxs.append(pos)
                idxs = sorted(idxs)
            # take last T indices
            idxs = idxs[-T:]
        else:
            # 'tail': take frames ending at keyframe
            idxs = list(range(max(0, pos - (T - 1) * stride), pos + 1, stride))
            idxs = idxs[-T:]

        # pad if needed
        if len(idxs) < T:
            pad = [idxs[0]] * (T - len(idxs))
            idxs = pad + idxs

        return [files_sorted[i][1] for i in idxs]

    def _normalize_box_xyxy(self, box_xyxy: List[float], W: int, H: int) -> torch.Tensor:
        x1, y1, x2, y2 = box_xyxy
        # robust clamp
        x1 = max(0.0, min(float(x1), float(W)))
        x2 = max(0.0, min(float(x2), float(W)))
        y1 = max(0.0, min(float(y1), float(H)))
        y2 = max(0.0, min(float(y2), float(H)))
        # avoid negative / inverted boxes
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        return torch.tensor([x1 / W, y1 / H, x2 / W, y2 / H], dtype=torch.float32).clamp(0, 1)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.samples[idx]
        image_url = item["image"]
        sample_id = item.get("sample_id", str(idx))

        # 1) Resolve local paths
        clip_dir = resolve_clip_dir(image_url, data_root=self.data_root)
        key_idx = extract_frame_index(image_url)  # from URL filename
        keyframe_path, files_sorted = find_keyframe_file(clip_dir, keyframe_index=key_idx)

        if keyframe_path is None:
            # last resort: try mapping the URL directly (in case clip dir listing fails)
            keyframe_path = resolve_local_image_path(image_url, data_root=self.data_root)
            if not os.path.exists(keyframe_path):
                raise FileNotFoundError(
                    f"Cannot resolve frames for sample_id={sample_id}.\n"
                    f"clip_dir={clip_dir}\nkeyframe_path={keyframe_path}"
                )
            # build trivial list
            files_sorted = [(extract_frame_index(keyframe_path) or 0, keyframe_path)]

        # 2) Read keyframe size (for dynamic normalization)
        key_img = self._load_image_rgb(keyframe_path)
        W, H = key_img.size

        # 3) Sample T frames
        frame_paths = self._sample_frame_paths(files_sorted, keyframe_path)

        if self.use_letterbox:
            sx, sy, new_w, new_h, pad_x, pad_y = self._letterbox_params(W, H)
            frames = [
                self._letterbox_to_tensor(self._load_image_rgb(p), new_w, new_h, pad_x, pad_y)
                for p in frame_paths
            ]
        else:
            new_w, new_h = self.img_size, self.img_size
            pad_x, pad_y = 0, 0
            sx = self.img_size / float(W) if W > 0 else 1.0
            sy = self.img_size / float(H) if H > 0 else 1.0
            frames = [self._resize_to_tensor(self._load_image_rgb(p)) for p in frame_paths]


        video = torch.stack(frames, dim=0)  # [T,3,S,S]
        video = (video - self.mean) / self.std

        
        # 4) Targets
        targets = item.get("targets", [])

        # Always return fixed-shape topk targets with a valid-mask.
        # This makes default collate safe even when a sample has <K targets.
        K = int(self.topk_targets)
        boxes_topk = torch.zeros((K, 4), dtype=torch.float32)
        risks_topk = torch.zeros((K,), dtype=torch.float32)
        mask_topk = torch.zeros((K,), dtype=torch.bool)

        if targets:
            # optional sorting: make sure top-1 is the 'primary' you want
            if self.sort_targets_by == "risk_score":
                targets = sorted(targets, key=lambda d: float(d.get("risk_score", 0.0)), reverse=True)

            topk = min(K, len(targets))
            top_targets = targets[:topk]

            if self.use_letterbox:
                boxes = [
                    self._normalize_box_xyxy_letterbox(t["bbox_xyxy"], W=W, H=H, sx=sx, sy=sy, pad_x=pad_x, pad_y=pad_y)
                    for t in top_targets
                ]
            else:
                boxes = [self._normalize_box_xyxy(t["bbox_xyxy"], W=W, H=H) for t in top_targets]

            risks = [float(t.get("risk_score", 0.0)) for t in top_targets]

            if boxes:
                boxes_topk[:topk] = torch.stack(boxes, dim=0)
                risks_topk[:topk] = torch.tensor(risks, dtype=torch.float32).clamp(0, 1)
                mask_topk[:topk] = True
        
        do_flip = (self.flip_prob > 0) and (np.random.rand() < self.flip_prob)
        if do_flip:
            # video: [T,3,S,S]
            video = torch.flip(video, dims=[3])

            valid = mask_topk  # [K] bool
            if valid.any().item():
                x1 = boxes_topk[valid, 0].clone()
                x2 = boxes_topk[valid, 2].clone()
                boxes_topk[valid, 0] = 1.0 - x2
                boxes_topk[valid, 2] = 1.0 - x1

                # enforce order
                x1n = torch.minimum(boxes_topk[valid, 0], boxes_topk[valid, 2])
                x2n = torch.maximum(boxes_topk[valid, 0], boxes_topk[valid, 2])
                boxes_topk[valid, 0], boxes_topk[valid, 2] = x1n, x2n

                boxes_topk[valid].clamp_(0.0, 1.0)


        # primary supervision keeps backward compatibility
        if bool(mask_topk[0]):
            gt_box = boxes_topk[0]
            gt_risk = risks_topk[0]
        else:
            gt_box = torch.zeros(4, dtype=torch.float32)
            gt_risk = torch.zeros((), dtype=torch.float32)

        out: Dict[str, Any] = {

            "pixel_values": video,  # [T, 3, H, W]
            "gt_box": gt_box,        # [4] normalized xyxy
            "gt_risk": gt_risk,      # scalar in [0,1]
            "gt_boxes_topk": boxes_topk,
            "gt_risks_topk": risks_topk,
            "gt_mask_topk": mask_topk,
        }

        if self.return_meta:
            out["meta"] = {
                "sample_id": sample_id,
                "image_url": image_url,
                "clip_dir": clip_dir,
                "keyframe_path": keyframe_path,
                "orig_size": (W, H),
                "frame_paths": frame_paths,
                "risk_label": item.get("risk_label", None),
                "letterbox": {
                    "use_letterbox": bool(self.use_letterbox),
                    "sx": float(sx),
                    "sy": float(sy),
                    "pad_x": int(pad_x),
                    "pad_y": int(pad_y),
                    "new_w": int(new_w),
                    "new_h": int(new_h),
                    "img_size": int(self.img_size),
                },

            }

        return out
    
    def _letterbox_params(self, W: int, H: int):
        S = int(self.img_size)
        scale = float(S) / float(max(W, H))  # keep aspect
        new_w = int(round(W * scale))
        new_h = int(round(H * scale))
        new_w = min(new_w, S)
        new_h = min(new_h, S)
        pad_x = (S - new_w) // 2
        pad_y = (S - new_h) // 2

        # IMPORTANT: use realized scales after rounding
        sx = new_w / float(W) if W > 0 else 1.0
        sy = new_h / float(H) if H > 0 else 1.0
        return sx, sy, new_w, new_h, pad_x, pad_y


    def _letterbox_to_tensor(self, img: Image.Image, new_w: int, new_h: int, pad_x: int, pad_y: int) -> torch.Tensor:
        S = int(self.img_size)
        img_resized = img.resize((new_w, new_h), resample=Image.BILINEAR)
        canvas = Image.new("RGB", (S, S), color=(self.letterbox_fill,)*3)
        canvas.paste(img_resized, (pad_x, pad_y))
        arr = np.array(canvas)
        t = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
        return t


    def _normalize_box_xyxy_letterbox(self, box_xyxy, W, H, sx, sy, pad_x, pad_y):
        """
        box in original pixel -> transform to letterboxed pixel -> normalize by img_size
        Return [x1,y1,x2,y2] in [0,1] w.r.t. self.img_size
        """
        x1, y1, x2, y2 = box_xyxy

        # clamp in original image space
        x1 = max(0.0, min(float(x1), float(W)))
        x2 = max(0.0, min(float(x2), float(W)))
        y1 = max(0.0, min(float(y1), float(H)))
        y2 = max(0.0, min(float(y2), float(H)))
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1

        # to letterbox pixel space
        x1 = x1 * sx + pad_x
        x2 = x2 * sx + pad_x
        y1 = y1 * sy + pad_y
        y2 = y2 * sy + pad_y

        S = float(self.img_size)
        return torch.tensor([x1 / S, y1 / S, x2 / S, y2 / S], dtype=torch.float32).clamp(0, 1)



def default_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate that keeps meta as a list.

    Notes:
    - gt_boxes_topk: [B,K,4] padded
    - gt_risks_topk: [B,K] padded
    - gt_mask_topk : [B,K] bool mask
    """
    pixel_values = torch.stack([b["pixel_values"] for b in batch], dim=0)  # [B,3,T,H,W]
    gt_box = torch.stack([b["gt_box"] for b in batch], dim=0)
    gt_risk = torch.stack([b["gt_risk"] for b in batch], dim=0)

    gt_boxes_topk = torch.stack([b["gt_boxes_topk"] for b in batch], dim=0)
    gt_risks_topk = torch.stack([b["gt_risks_topk"] for b in batch], dim=0)
    gt_mask_topk = torch.stack([b["gt_mask_topk"] for b in batch], dim=0)

    out = {
        "pixel_values": pixel_values,
        "gt_box": gt_box,
        "gt_risk": gt_risk,
        "gt_boxes_topk": gt_boxes_topk,
        "gt_risks_topk": gt_risks_topk,
        "gt_mask_topk": gt_mask_topk,
    }
    if "meta" in batch[0]:
        out["meta"] = [b["meta"] for b in batch]
    return out
