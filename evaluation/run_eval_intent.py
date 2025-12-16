# /workspace/chz/code/DRAMA-X/evaluation/run_eval_intent.py

import json

# 端到端：用预测 BBox + 预测 Intent
from od_intent_eval import evaluate_bounding_boxes, evaluate_intents as eval_intent_e2e
# 使用 GT bbox：只评估 Intent，本质上是在看“意图预测的上限”
from od_intent_gt_eval import evaluate_intents as eval_intent_gt


def load_drama_gt(jsonl_path: str):
    """读取 DRAMA-X_hf/drama_x_annotations_populated.jsonl"""
    gt = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            sample = json.loads(line)
            # 有的版本用 "id"，有的用 "sample_id"
            sid = sample.get("sample_id", sample.get("id"))
            if sid is None:
                raise ValueError(f"No id/sample_id in sample: {sample.keys()}")
            gt[sid] = sample
    return gt


def main():
    # 1. 载入 GT
    gt_path = "../DRAMA-X_hf/drama_x_annotations_populated.jsonl"
    gt = load_drama_gt(gt_path)

    # 2. 载入预测（你之前转换好的 Qwen3-2B 结果）
    pred_path = "../drama_intent/outputs/qwen3_local/qwen3_2b_pred_intent_bbox.json"
    with open(pred_path, "r", encoding="utf-8") as f:
        preds = json.load(f)

    num_samples = len(preds)

    # ------------------------------------------------
    # (A) End-to-end：预测 BBox + 预测 Intent
    # ------------------------------------------------
    bbox_metrics = evaluate_bounding_boxes(gt, preds, num_samples=num_samples)
    intent_metrics_e2e = eval_intent_e2e(gt, preds, num_samples=num_samples)

    print("===== [A] End-to-end BBox Evaluation (预测框) =====")
    print(bbox_metrics)
    print("\n===== [A] End-to-end Intent Evaluation (预测框 + 预测意图) =====")
    print(intent_metrics_e2e)

    # ------------------------------------------------
    # (B) Intent (GT bbox)：跳过 BBox 匹配，只看意图本身
    # ------------------------------------------------
    # 这里用 od_intent_gt_eval.evaluate_intents，并打开 skip_bbox_matching=True
    intent_metrics_gt = eval_intent_gt(
        gt,
        preds,
        num_samples=num_samples,
        skip_bbox_matching=True,
    )

    print("\n===== [B] Intent Evaluation with GT BBoxes (skip_bbox_matching=True) =====")
    print(intent_metrics_gt)


if __name__ == "__main__":
    main()
