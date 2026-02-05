"""Stage-1 (Perception): generate proposal boxes with YOLO (optionally with tracker).

Why this file exists
--------------------
Your new DRAMA-X design is **two-stage**:

1) Stage-1: "框准" (get accurate candidate boxes)
2) Stage-2: from accurate candidates, select the critical VRU (risk selection/reasoning)

This script performs Stage-1 **offline** and writes a proposals jsonl.

Input jsonl requirements
------------------------
Each line is a dict containing at least:
  - sample_id: str
  - image: keyframe url or path (same as your existing jsonl)

Output jsonl
------------
Each line:
{
  "sample_id": ...,
  "image": ...,
  "img_hw": [H, W],
  "proposals": [
      {"bbox_xyxy": [x1,y1,x2,y2], "cls": int, "conf": float}, ...
  ]
}

Notes
-----
* We intentionally generate proposals on the **keyframe** (single frame) first.
  Tracking across clip frames can be added later without changing Stage-2.
* By default we keep COCO classes: person(0), bicycle(1), motorcycle(3).

Usage
-----
python -m drama_fast.pipeline.stage1_generate_yolo_proposals \
  --in_jsonl ./splits_v3/train.jsonl \
  --data_root $DRAMA_DATA_ROOT \
  --out_jsonl ./splits_v3/train_proposals_yolo.jsonl \
  --yolo_model yolov8x.pt \
  --img_size 384 --topk 80
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, Any, List, Optional, Tuple

import numpy as np

try:
    from PIL import Image
except Exception as e:
    raise RuntimeError("PIL is required. Please `pip install pillow`.") from e


def _iter_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _write_jsonl(path: str, items):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for obj in items:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _resolve_local_keyframe(image_url: str, data_root: str) -> str:
    """Resolve DRAMA keyframe url -> local path.

    We re-use the exact parsing logic you already use in `dataset_phase1.py`.
    To avoid circular import from script entry, we import here.
    """
    from drama_fast.utils.path_resolver import resolve_local_image_path

    # In our jsonl, `image` already points to the keyframe (url-like string).
    # `resolve_local_image_path` knows how to map it to a local file.
    return resolve_local_image_path(image_url=image_url, data_root=data_root)


def _load_image_rgb(path: str) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.array(img)


def _load_yolo(model_name: str):
    """Load Ultralytics YOLO.

    We depend on ultralytics for Stage-1 only.
    Install:
      pip install ultralytics
    """
    try:
        from ultralytics import YOLO
    except Exception as e:
        raise RuntimeError(
            "Ultralytics is required for stage-1 proposals. "
            "Please `pip install ultralytics` in your dramax env."
        ) from e
    return YOLO(model_name)


def _filter_and_topk(
    boxes_xyxy: np.ndarray,
    conf: np.ndarray,
    cls: np.ndarray,
    keep_cls: Optional[List[int]],
    topk: int,
    conf_thres: float,
) -> List[Dict[str, Any]]:
    if boxes_xyxy.size == 0:
        return []

    keep = conf >= float(conf_thres)
    if keep_cls is not None and len(keep_cls) > 0:
        keep = keep & np.isin(cls.astype(np.int64), np.array(keep_cls, dtype=np.int64))

    idx = np.where(keep)[0]
    if idx.size == 0:
        return []

    # sort by confidence desc
    idx = idx[np.argsort(-conf[idx])]
    idx = idx[: int(topk)]

    out = []
    for i in idx:
        x1, y1, x2, y2 = boxes_xyxy[i].tolist()
        out.append(
            {
                "bbox_xyxy": [float(x1), float(y1), float(x2), float(y2)],
                "cls": int(cls[i]),
                "conf": float(conf[i]),
            }
        )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_jsonl", type=str, required=True)
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--out_jsonl", type=str, required=True)
    ap.add_argument("--yolo_model", type=str, default="yolov8x.pt")
    ap.add_argument("--img_size", type=int, default=640, help="YOLO inference size")
    ap.add_argument("--topk", type=int, default=80)
    ap.add_argument("--conf_thres", type=float, default=0.10)
    ap.add_argument(
        "--keep_cls",
        type=str,
        default="0,1,3",
        help="COCO class ids to keep. default: person(0), bicycle(1), motorcycle(3)",
    )
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--every", type=int, default=1, help="process every N samples")
    args = ap.parse_args()

    keep_cls = None
    if args.keep_cls and args.keep_cls.strip():
        keep_cls = [int(x) for x in args.keep_cls.split(",") if x.strip()]

    model = _load_yolo(args.yolo_model)

    out_items = []
    for j, item in enumerate(_iter_jsonl(args.in_jsonl)):
        if (j % int(args.every)) != 0:
            continue

        sample_id = item.get("sample_id") or item.get("id") or item.get("uid")
        image_url = item.get("image")
        if not sample_id or not image_url:
            raise ValueError(
                f"Each line must contain sample_id and image. Got keys={list(item.keys())}"
            )

        keyframe_path = _resolve_local_keyframe(image_url=image_url, data_root=args.data_root)
        img = _load_image_rgb(keyframe_path)
        H, W = img.shape[:2]

        # Ultralytics YOLO expects BGR ndarray or path.
        # We pass path to avoid extra copies.
        results = model.predict(
            source=keyframe_path,
            imgsz=int(args.img_size),
            conf=float(args.conf_thres),
            device=args.device,
            verbose=False,
        )
        r0 = results[0]
        if r0.boxes is None or len(r0.boxes) == 0:
            proposals = []
        else:
            b = r0.boxes
            boxes_xyxy = b.xyxy.detach().cpu().numpy()
            conf = b.conf.detach().cpu().numpy()
            cls = b.cls.detach().cpu().numpy()
            proposals = _filter_and_topk(
                boxes_xyxy=boxes_xyxy,
                conf=conf,
                cls=cls,
                keep_cls=keep_cls,
                topk=int(args.topk),
                conf_thres=float(args.conf_thres),
            )

        out_items.append(
            {
                "sample_id": sample_id,
                "image": image_url,
                "img_hw": [int(H), int(W)],
                "proposals": proposals,
            }
        )

        if (len(out_items) % 200) == 0:
            print(f"[stage1] processed {len(out_items)} samples...")

    _write_jsonl(args.out_jsonl, out_items)
    print(f"[stage1] wrote proposals to: {args.out_jsonl} (num_samples={len(out_items)})")


if __name__ == "__main__":
    main()
