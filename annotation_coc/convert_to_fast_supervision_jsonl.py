#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert DRAMA-X CoC jsonl to fast-system supervision jsonl.
Outputs one row per sample with primary bbox + risk_score (joint target).
Only stdlib.

python convert_to_fast_supervision_jsonl.py \
  --input /workspace/chz/code/DRAMA-X/annotation_coc/drama_x_coc_qwen3vl_2b_v2_full_rule_based_20260108.jsonl \
  --output /workspace/chz/code/DRAMA-X/annotation_coc/drama_x_fast_sup_v2_rule.jsonl \
  --topk 1
"""

import argparse
import json
import os
from typing import Any, Dict, List, Optional, Tuple


def read_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield ln, json.loads(line)
            except Exception as e:
                raise RuntimeError(f"[JSON parse fail] {path}:{ln} -> {e}")


def safe_get(d: Dict[str, Any], keys: List[str], default=None):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def get_sample_id(r: Dict[str, Any]) -> str:
    return str(safe_get(r, ["sample_id", "id", "clip_id", "uid"], "UNKNOWN"))


def get_image_path(r: Dict[str, Any]) -> Optional[str]:
    return safe_get(r, ["image", "image_path", "frame_path", "img_path"], None)


def get_risk_label(r: Dict[str, Any]) -> str:
    return str(safe_get(r, ["risk_label", "risk", "risk_level"], "unknown")).lower()


def get_vru_list(r: Dict[str, Any]) -> List[Dict[str, Any]]:
    v = r.get("vru_list") or r.get("vrus") or r.get("objects") or []
    return v if isinstance(v, list) else []


def get_vru_score(vru: Dict[str, Any]) -> Optional[float]:
    s = safe_get(vru, ["risk_score", "score", "riskScore", "risk"], None)
    if s is None:
        return None
    try:
        return float(s)
    except:
        return None


def parse_bbox_xyxy(vru: Dict[str, Any]) -> Optional[List[float]]:
    """
    Try to parse bbox in xyxy format from common keys.
    Accepts:
      - vru["box_xyxy"] / ["bbox_xyxy"] = [x1,y1,x2,y2]
      - vru["bbox"] / ["box"] could be [x1,y1,x2,y2] or [x,y,w,h]
      - vru has x1,y1,x2,y2 fields
    """
    # direct xyxy
    for k in ["box_xyxy", "bbox_xyxy", "xyxy"]:
        b = vru.get(k)
        if isinstance(b, list) and len(b) == 4:
            try:
                return [float(b[0]), float(b[1]), float(b[2]), float(b[3])]
            except:
                pass

    # x1,y1,x2,y2 fields
    if all(k in vru for k in ["x1", "y1", "x2", "y2"]):
        try:
            return [float(vru["x1"]), float(vru["y1"]), float(vru["x2"]), float(vru["y2"])]
        except:
            pass

    # bbox or box: may be xyxy or xywh
    for k in ["bbox", "box"]:
        b = vru.get(k)
        if isinstance(b, list) and len(b) == 4:
            try:
                x, y, a, b2 = float(b[0]), float(b[1]), float(b[2]), float(b[3])
            except:
                continue
            # heuristic: if a,b2 look like width/height (positive and x+a <= 1? unknown),
            # we cannot know image size. assume if a>x and b2>y then xyxy else xywh.
            # Safer: treat as xyxy if (a > x and b2 > y) AND (a - x) > 1 AND (b2 - y) > 1
            if (a > x and b2 > y) and ((a - x) > 1.0) and ((b2 - y) > 1.0):
                return [x, y, a, b2]  # xyxy
            else:
                # interpret as xywh
                return [x, y, x + a, y + b2]

    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--topk", type=int, default=1, help="how many top VRUs to export as targets; default 1 (primary)")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    n = 0
    bad_bbox = 0
    bad_score = 0

    with open(args.output, "w", encoding="utf-8") as out:
        for ln, r in read_jsonl(args.input):
            sid = get_sample_id(r)
            img = get_image_path(r)
            risk_label = get_risk_label(r)

            vrus = get_vru_list(r)
            if not vrus:
                # skip or write empty target
                continue

            targets = []
            for vru in vrus[: max(1, args.topk)]:
                bbox = parse_bbox_xyxy(vru)
                score = get_vru_score(vru)
                vtype = safe_get(vru, ["type", "category", "cls"], None)
                if bbox is None:
                    bad_bbox += 1
                    continue
                if score is None:
                    bad_score += 1
                    continue
                targets.append({
                    "bbox_xyxy": bbox,
                    "risk_score": float(score),
                    "type": vtype,
                    "vru_id": safe_get(vru, ["id", "vru_id", "track_id"], None),
                })

            if not targets:
                continue

            item = {
                "sample_id": sid,
                "image": img,
                "risk_label": risk_label,
                "targets": targets,  # default: len=1 (primary)
            }
            out.write(json.dumps(item, ensure_ascii=False) + "\n")
            n += 1

    print(f"[OK] wrote {n} rows -> {args.output}")
    print(f"[WARN] bad_bbox={bad_bbox}, bad_score={bad_score}")


if __name__ == "__main__":
    main()
