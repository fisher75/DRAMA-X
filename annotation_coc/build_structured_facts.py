#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
把 DRAMA-X 的 drama_x_annotations_populated.jsonl
转换成后续 CoC 生成用的结构化事实文件 drama_x_structured.jsonl

输入字段示例（单条）：
{
  "image_path": "...png",
  "video_path": "...gif",
  "Risk": "Yes",
  "Pedestrians": {
    "1": {
      "Box": [877, 677, 1065, 1199],
      "Intent": ["goes to the left", "moves away from ego vehicle"],
      "Position": "Front of ego vehicle",
      "Description": "..."
    }
  },
  "Cyclists": {},
  "suggested_action": "be aware or cautious ...",
  "id": "clip_1038_000354_frame_000354"
}

输出字段示例（单条）：
{
  "sample_id": "clip_1038_000354_frame_000354",
  "image_path": "...png",
  "video_path": "...gif",
  "risk_label": "high",
  "suggested_action_raw": "be aware or cautious ...",
  "vru_list": [
    {
      "vru_id": "ped_1",
      "type": "pedestrian",
      "box_xyxy": [877, 677, 1065, 1199],
      "box_xywh": [877, 677, 188, 522],
      "intent_list": ["goes to the left", "moves away from ego vehicle"],
      "position_raw": "Front of ego vehicle",
      "position_category": "front",
      "description": "...",
      "distance_level": "near"
    }
  ],
  "primary_vru_id": "ped_1",
  "image_size_est": [1065, 1199]
}
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Tuple


def map_risk_label(risk_value) -> str:
    """
    把原始 Risk 字段映射为 {high, low}
    原始数据一般是 "Yes"/"No"，也兼容 bool / 0/1.
    """
    if isinstance(risk_value, bool):
        return "high" if risk_value else "low"
    if isinstance(risk_value, (int, float)):
        return "high" if risk_value > 0 else "low"
    if isinstance(risk_value, str):
        s = risk_value.strip().lower()
        if s in {"yes", "y", "1", "high", "risky"}:
            return "high"
        if s in {"no", "n", "0", "low", "safe"}:
            return "low"
    # 默认保守一点
    return "low"


def estimate_image_size_from_boxes(
    vru_boxes_xyxy: List[List[float]]
) -> Tuple[int, int]:
    """
    通过所有 VRU 的 box，粗略估计一张图像的尺寸：
    - 宽度 ~ max_x2
    - 高度 ~ max_y2
    这是一个启发式估计，用于计算 bbox 相对大小。
    """
    if not vru_boxes_xyxy:
        return 1, 1  # 防止除 0
    max_x2 = max(box[2] for box in vru_boxes_xyxy)
    max_y2 = max(box[3] for box in vru_boxes_xyxy)
    # 保底，避免 0
    img_w = max(int(max_x2), 1)
    img_h = max(int(max_y2), 1)
    return img_w, img_h


def map_distance_level(
    box_xyxy: List[float], img_w: int, img_h: int
) -> str:
    """
    根据 bbox 相对图像面积，粗糙划分 {near, mid, far}
    只是一个 heuristic，后续如果有更准的 depth / TTC，可以替换。
    """
    x1, y1, x2, y2 = box_xyxy
    w = max(x2 - x1, 1.0)
    h = max(y2 - y1, 1.0)
    area = w * h
    img_area = float(img_w * img_h)
    ratio = area / img_area

    # 阈值可以后续根据统计再微调
    if ratio > 0.05:
        return "near"
    elif ratio > 0.02:
        return "mid"
    else:
        return "far"


def map_position_category(position_raw: str) -> str:
    """
    把原始 Position 文本（如 "Front of ego vehicle"）
    映射成一个粗类别：{front, left, right, behind, other}
    """
    if not position_raw:
        return "other"
    s = position_raw.lower()
    if "front" in s:
        return "front"
    if "left" in s:
        return "left"
    if "right" in s:
        return "right"
    if "behind" in s or "rear" in s or "back" in s:
        return "behind"
    return "other"


def collect_vrus(sample: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    从一条样本中收集所有 VRU（行人+骑行者），统一成 vru_list.
    返回的每个元素包含：
      - vru_id: ped_1 / cyc_2
      - type: "pedestrian" / "cyclist"
      - box_xyxy, box_xywh
      - intent_list
      - position_raw, position_category
      - description
    """
    vru_list: List[Dict[str, Any]] = []

    # Pedestrians
    peds: Dict[str, Any] = sample.get("Pedestrians") or {}
    for pid, pdata in peds.items():
        box = pdata.get("Box", [])
        if not box or len(box) != 4:
            continue
        intent = pdata.get("Intent") or []
        pos = pdata.get("Position", "")
        desc = pdata.get("Description", "")

        x1, y1, x2, y2 = box
        w = x2 - x1
        h = y2 - y1

        vru_list.append(
            {
                "vru_id": f"ped_{pid}",
                "type": "pedestrian",
                "box_xyxy": [x1, y1, x2, y2],
                "box_xywh": [x1, y1, w, h],
                "intent_list": intent,
                "position_raw": pos,
                "position_category": map_position_category(pos),
                "description": desc,
            }
        )

    # Cyclists
    cycs: Dict[str, Any] = sample.get("Cyclists") or {}
    for cid, cdata in cycs.items():
        box = cdata.get("Box", [])
        if not box or len(box) != 4:
            continue
        intent = cdata.get("Intent") or []
        pos = cdata.get("Position", "")
        desc = cdata.get("Description", "")

        x1, y1, x2, y2 = box
        w = x2 - x1
        h = y2 - y1

        vru_list.append(
            {
                "vru_id": f"cyc_{cid}",
                "type": "cyclist",
                "box_xyxy": [x1, y1, x2, y2],
                "box_xywh": [x1, y1, w, h],
                "intent_list": intent,
                "position_raw": pos,
                "position_category": map_position_category(pos),
                "description": desc,
            }
        )

    return vru_list


def choose_primary_vru(vru_list: List[Dict[str, Any]]) -> str:
    """
    选一个 primary_vru_id
    当前策略：bbox 面积最大的那个 VRU.
    """
    if not vru_list:
        return ""

    def area(v):
        x1, y1, x2, y2 = v["box_xyxy"]
        return (x2 - x1) * (y2 - y1)

    primary = max(vru_list, key=area)
    return primary["vru_id"]


def convert_one_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    """
    把原始的一条 DRAMA-X JSON 转成结构化事实条目。
    """
    sample_id = sample.get("sample_id", sample.get("id"))
    if sample_id is None:
        raise ValueError(f"No sample_id/id in sample keys={sample.keys()}")

    image_path = sample.get("image_path", "")
    video_path = sample.get("video_path", "")

    risk_label = map_risk_label(sample.get("Risk"))
    suggested_action_raw = sample.get("suggested_action", "")

    vru_list = collect_vrus(sample)

    # 估计图像尺寸，用于 distance_level
    all_boxes = [v["box_xyxy"] for v in vru_list]
    img_w, img_h = estimate_image_size_from_boxes(all_boxes)

    # 为每个 VRU 添加 distance_level
    for v in vru_list:
        v["distance_level"] = map_distance_level(v["box_xyxy"], img_w, img_h)

    primary_vru_id = choose_primary_vru(vru_list)

    structured = {
        "sample_id": sample_id,
        "image_path": image_path,
        "video_path": video_path,
        "risk_label": risk_label,  # {"high","low"}
        "suggested_action_raw": suggested_action_raw,
        "vru_list": vru_list,
        "primary_vru_id": primary_vru_id,
        "image_size_est": [img_w, img_h],
    }
    return structured


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Build structured facts from DRAMA-X annotations for CoC generation."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="../DRAMA-X_hf/drama_x_annotations_populated.jsonl",
        help="Path to drama_x_annotations_populated.jsonl",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="drama_x_structured.jsonl",
        help="Output JSONL path for structured facts.",
    )

    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    num_in = 0
    num_out = 0
    num_skipped_no_vru = 0

    with in_path.open("r", encoding="utf-8") as fin, out_path.open(
        "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            num_in += 1
            sample = json.loads(line)
            vru_list = (sample.get("Pedestrians") or {}) | (sample.get("Cyclists") or {})
            # 也可以直接看 collect_vrus 的结果来判断是否跳过
            structured = convert_one_sample(sample)
            if not structured["vru_list"]:
                num_skipped_no_vru += 1
                continue
            fout.write(json.dumps(structured) + "\n")
            num_out += 1

    print(f"[build_structured_facts] input samples : {num_in}")
    print(f"[build_structured_facts] output samples: {num_out}")
    print(f"[build_structured_facts] skipped (no VRU): {num_skipped_no_vru}")
    print(f"[build_structured_facts] saved to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
