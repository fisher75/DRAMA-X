#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
build_structured_facts_v2.py (UPDATED)

功能：
- 读取 DRAMA-X 的 drama_x_annotations_populated.jsonl
- 将其转换为 CoC 生成所需的结构化事实 (drama_x_structured_v2.jsonl)
- [关键更新]: 使用基于规则的 Risk Score 策略选择 Primary VRU，而非简单的 bbox 面积最大。

Risk Score 策略 (v1):
- Score = w_p * Position + w_i * Intent + w_d * Distance + w_s * Size
- 权重先验: Pos=0.45, Intent=0.35, Dist=0.15, Size=0.05
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
import math


def map_risk_label(risk_value) -> str:
    """
    把原始 Risk 字段映射为 {high, low}
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
    return "low"


def estimate_image_size_from_boxes(
    vru_boxes_xyxy: List[List[float]]
) -> Tuple[int, int]:
    """
    通过所有 VRU 的 box，粗略估计图像尺寸，用于计算相对大小/距离。
    """
    if not vru_boxes_xyxy:
        return 1920, 1080  # 默认兜底，防止无框时除0
    max_x2 = max(box[2] for box in vru_boxes_xyxy)
    max_y2 = max(box[3] for box in vru_boxes_xyxy)
    img_w = max(int(max_x2), 1)
    img_h = max(int(max_y2), 1)
    # 如果估计值太小，可能是 crop，给个保底值
    return max(img_w, 640), max(img_h, 480)


def map_distance_level(
    box_xyxy: List[float], img_w: int, img_h: int
) -> str:
    """
    根据 bbox 相对图像面积，粗糙划分 {near, mid, far}
    """
    x1, y1, x2, y2 = box_xyxy
    w = max(x2 - x1, 1.0)
    h = max(y2 - y1, 1.0)
    area = w * h
    img_area = float(img_w * img_h)
    ratio = area / img_area

    if ratio > 0.05:
        return "near"
    elif ratio > 0.02:
        return "mid"
    else:
        return "far"


def map_position_category(position_raw: str) -> str:
    """
    映射原始 Position 文本到粗类别
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


# ==========================================
# [新增] 基于规则的 Risk Score 计算函数
# ==========================================
def calculate_vru_risk_score(vru: Dict[str, Any], risk_label: str, img_size: Tuple[int, int]) -> float:
    """
    计算 VRU 的风险分数。分数越高，越可能是 primary_vru。
    Formula: Score = w_p * Pos + w_i * Intent + w_d * Dist + w_s * Size
    """
    score = 0.0
    
    # 辅助：转小写
    def norm(s): return str(s).lower() if s else ""

    # 1. Position Score (0.45) - 最关键：是否在冲突路径
    pos_text = norm(vru.get('position_category', '')) + " " + norm(vru.get('position_raw', ''))
    if any(x in pos_text for x in ['front', 'center', 'ego_lane', 'ahead']):
        s_pos = 1.0
    elif any(x in pos_text for x in ['left', 'right']): # 侧方
        s_pos = 0.6
    else: # rear/other
        s_pos = 0.1

    # 2. Intent Score (0.35) - 次关键：是否有冲突趋势
    intents = [norm(i) for i in vru.get('intent_list', [])]
    # 展平成一个字符串方便匹配
    intent_text = " ".join(intents)
    
    if any(x in intent_text for x in ['cross', 'enter', 'toward', 'cut', 'merg', 'into']):
        s_intent = 1.0 # 高危意图
    elif any(x in intent_text for x in ['wait', 'stand', 'stop']):
        s_intent = 0.6 # 中等意图
    elif any(x in intent_text for x in ['away', 'leav']):
        s_intent = 0.2 # 低危意图
    else:
        s_intent = 0.4 # 未知

    # 3. Distance Score (0.15) - 距离代理
    dist = norm(vru.get('distance_level', ''))
    if 'near' in dist or 'close' in dist:
        s_dist = 1.0
    elif 'mid' in dist or 'medium' in dist:
        s_dist = 0.6
    else:
        s_dist = 0.2

    # 4. Size Score (0.05) - Tie-breaker
    # 计算归一化面积 (0~1)
    bbox = vru.get('box_xyxy', [])
    s_size = 0.0
    if len(bbox) == 4:
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        area = w * h
        img_area = img_size[0] * img_size[1]
        s_size = min(area / img_area, 1.0)

    # 加权求和
    w_p, w_i, w_d, w_s = 0.45, 0.35, 0.15, 0.05
    final_score = (w_p * s_pos) + (w_i * s_intent) + (w_d * s_dist) + (w_s * s_size)

    # 5. Context Bonus (可选)
    # 如果整张图是 High Risk，且该 VRU 在正前方且有意图，额外加分锁定
    if norm(risk_label) == 'high' and ('front' in pos_text) and s_intent >= 0.6:
        final_score += 0.1

    return final_score


def collect_vrus(sample: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    从一条样本中收集所有 VRU（行人+骑行者），统一成 vru_list.
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


def convert_one_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    """
    把原始的一条 DRAMA-X JSON 转成结构化事实条目。
    [Update]: 在这里集成 Risk Score 计算和排序逻辑。
    """
    sample_id = sample.get("sample_id", sample.get("id"))
    if sample_id is None:
        # 容错：有些数据可能没有 id 字段，跳过或生成临时 id
        return {} 

    image_path = sample.get("image_path", "")
    video_path = sample.get("video_path", "")

    risk_label = map_risk_label(sample.get("Risk"))
    suggested_action_raw = sample.get("suggested_action", "")

    # 1. 收集 VRU
    vru_list = collect_vrus(sample)

    # 2. 估计图像尺寸 & 计算 distance_level
    all_boxes = [v["box_xyxy"] for v in vru_list]
    img_w, img_h = estimate_image_size_from_boxes(all_boxes)

    for v in vru_list:
        v["distance_level"] = map_distance_level(v["box_xyxy"], img_w, img_h)

    # 3. [核心修改] 计算 Risk Score 并排序
    if vru_list:
        for v in vru_list:
            v["risk_score"] = calculate_vru_risk_score(v, risk_label, (img_w, img_h))
        
        # 按分数降序排序：此时 vru_list[0] 就是 Primary VRU
        vru_list.sort(key=lambda x: x["risk_score"], reverse=True)
        primary_vru_id = vru_list[0]["vru_id"]
    else:
        primary_vru_id = ""

    structured = {
        "sample_id": sample_id,
        "image_path": image_path,
        "video_path": video_path,
        "risk_label": risk_label,  # {"high","low"}
        "suggested_action_raw": suggested_action_raw,
        "vru_list": vru_list,      # [注意] 这里已经是排好序的列表了
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

    print(f"[INFO] Reading from {in_path} ...")
    with in_path.open("r", encoding="utf-8") as fin, out_path.open(
        "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            num_in += 1
            sample = json.loads(line)
            
            structured = convert_one_sample(sample)
            
            # 过滤掉转换失败（无ID）或无 VRU 的样本
            if not structured or not structured.get("vru_list"):
                num_skipped_no_vru += 1
                continue
            
            fout.write(json.dumps(structured) + "\n")
            num_out += 1

            if num_in % 1000 == 0:
                print(f"Processed {num_in} samples...")

    print(f"[build_structured_facts] input samples : {num_in}")
    print(f"[build_structured_facts] output samples: {num_out}")
    print(f"[build_structured_facts] skipped (no VRU/Inv): {num_skipped_no_vru}")
    print(f"[build_structured_facts] saved to: {out_path.resolve()}")


if __name__ == "__main__":
    main()