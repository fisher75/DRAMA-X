import json
from pathlib import Path

def convert_qwen3_raw(
    raw_path: str,
    out_intent_bbox_path: str,
    out_risk_path: str,
    out_action_path: str,
):
    raw_path = Path(raw_path)
    with raw_path.open("r", encoding="utf-8") as f:
        raw_data = json.load(f)

    pred_intent_bbox = {}  # sample_id -> {obj_name -> {...}}
    pred_risk = {}         # sample_id -> "Yes"/"No"
    pred_action = {}       # sample_id -> suggested action text

    for sample_id, sample_info in raw_data.items():
        sample_objs = {}

        for key, val in sample_info.items():
            if key == "Risk":
                # 直接保留 Yes / No，后面在 risk_eval 里再映射成 0/1 也可以
                pred_risk[sample_id] = val
            elif key == "Suggested_action":
                pred_action[sample_id] = val
            else:
                # 其余都是 object：pedestrian_1 / car_1 / cyclist_1 / motorbike_1 等
                obj_name = key
                obj = {}

                # Intent -> 统一成 list[str]
                intent = val.get("Intent", [])
                if isinstance(intent, str):
                    intent = [intent]
                obj["Intent"] = intent

                # Reason
                obj["Reason"] = val.get("Reason", "")

                # Bounding_box
                bbox = val.get("Bounding_box", [])
                # 有些模型可能返回 None 或空字符串，这里统一清洗成 float
                if isinstance(bbox, dict):
                    bbox = list(bbox.values())
                if isinstance(bbox, list):
                    clean_bbox = []
                    for v in bbox:
                        if v is None or v == "":
                            clean_bbox.append(0.0)
                        else:
                            clean_bbox.append(float(v))
                    bbox = clean_bbox
                obj["Bounding_box"] = bbox

                sample_objs[obj_name] = obj

        pred_intent_bbox[sample_id] = sample_objs

    # 保存三个文件
    with open(out_intent_bbox_path, "w", encoding="utf-8") as f:
        json.dump(pred_intent_bbox, f, indent=2, ensure_ascii=False)

    with open(out_risk_path, "w", encoding="utf-8") as f:
        json.dump(pred_risk, f, indent=2, ensure_ascii=False)

    with open(out_action_path, "w", encoding="utf-8") as f:
        json.dump(pred_action, f, indent=2, ensure_ascii=False)

    print("Done:")
    print(f"  intent + bbox -> {out_intent_bbox_path}")
    print(f"  risk          -> {out_risk_path}")
    print(f"  action        -> {out_action_path}")


if __name__ == "__main__":
    convert_qwen3_raw(
        raw_path="all_raw_Qwen3-VL-2B-Instruct_onepass.json",
        out_intent_bbox_path="qwen3_2b_pred_intent_bbox.json",
        out_risk_path="qwen3_2b_pred_risk.json",
        out_action_path="qwen3_2b_pred_actions.json",
    )
